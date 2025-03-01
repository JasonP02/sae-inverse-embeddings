import torch
from sae_lens import SAE, HookedSAETransformer
from transformer_lens import ActivationCache, utils
import plotly.graph_objects as go
import plotly.subplots as sp

def load_models(config):
    """Load the transformer model and SAE based on config."""
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    model = HookedSAETransformer.from_pretrained("EleutherAI/pythia-70m-deduped", device=device)
    sae, _, _ = SAE.from_pretrained(
        release="pythia-70m-deduped-mlp-sm",
        sae_id="blocks.3.hook_mlp_out",
        device=device
    )
    return model, sae

# %% Helper Functions

def get_similar_tokens(embeddings, top_k=5):
    """Get the most similar tokens for each position in the embeddings."""
    with torch.no_grad():
        vocab_embeds = pythia.W_E
        similarity = torch.nn.functional.normalize(embeddings, dim=-1) @ \
                    torch.nn.functional.normalize(vocab_embeds, dim=-1).T
        top_tokens = similarity.topk(top_k, dim=-1)
        
        result = []
        for pos in range(len(embeddings)):
            tokens = [pythia.to_string(idx) for idx in top_tokens.indices[pos]]
            scores = top_tokens.values[pos]
            result.append((tokens, scores))
        return result

def embed_hook(value, hook):
    """Hook to replace embeddings with our parameter P."""
    return P.unsqueeze(0)

def find_high_activation_features(n_samples=100, length=4):
    """Find features with high activation potential."""
    print("\nFinding features with high activation potential...")
    
    with torch.no_grad():
        sample_tokens = torch.randint(0, pythia.cfg.d_vocab, (n_samples, length), device=device)
        
        all_activations = []
        for i in range(0, n_samples, 10):
            batch = sample_tokens[i:i+10]
            _, cache = pythia.run_with_cache_with_saes(
                input=batch,
                return_type="logits",
                saes=[sae]
            )
            batch_acts = cache['blocks.3.hook_mlp_out.hook_sae_acts_post']
            all_activations.append(batch_acts)
        
        combined_acts = torch.cat(all_activations, dim=0)
        max_activations = combined_acts.max(dim=0).values.max(dim=0).values
        top_features = torch.argsort(max_activations, descending=True)[:10]
        
        print("\nTop features by maximum activation:")
        for i, feat_idx in enumerate(top_features):
            pass
            # print(f"Feature {feat_idx}: Max activation = {max_activations[feat_idx]:.4f}")
        
        return top_features[0].item(), max_activations[top_features[0]].item()

def find_contextual_features(prompts=None, top_k=10):
    """Find features with high position variance (contextual features)."""
    if prompts is None:
        prompts = [
            "Jimmy carter was the president of the United States",
            "Luke went to the store to buy",
            "My favorite color is",
            "The sum of 2 and 2 is",
            "You should use an if statement"
        ]
    
    print("\nSearching for contextual features...")
    n_features = sae.cfg.d_sae
    feature_position_variance = torch.zeros(n_features, device=device)
    
    for prompt in prompts:
        print(f"\nAnalyzing prompt: '{prompt}'")
        _, cache = pythia.run_with_cache_with_saes(prompt, saes=[sae])
        acts = cache['blocks.3.hook_mlp_out.hook_sae_acts_post'][0]
        
        for feat_idx in range(n_features):
            feature_position_variance[feat_idx] += acts[:, feat_idx].var()
    
    feature_position_variance /= len(prompts)
    top_contextual = torch.argsort(feature_position_variance, descending=True)[:top_k]
    
    print("\nTop contextual features (highest position variance):")
    for i, feat_idx in enumerate(top_contextual):
        print(f"Feature {feat_idx}: Position variance = {feature_position_variance[feat_idx]:.4f}")
    
    return top_contextual[0].item()

def find_co_activated_features(target_feature, n_samples=50, length=4):
    """Find features that co-activate with the target feature."""
    print(f"\nSearching for features co-activated with feature {target_feature}...")
    
    with torch.no_grad():
        sample_tokens = torch.randint(0, pythia.cfg.d_vocab, (n_samples, length), device=device)
        
        all_acts = []
        for i in range(0, n_samples, 10):
            batch = sample_tokens[i:i+10]
            _, cache = pythia.run_with_cache_with_saes(
                input=batch,
                return_type="logits",
                saes=[sae]
            )
            batch_acts = cache['blocks.3.hook_mlp_out.hook_sae_acts_post']
            all_acts.append(batch_acts)
        
        combined_acts = torch.cat(all_acts, dim=0)
        n_features = sae.cfg.d_sae
        reshaped_acts = combined_acts.reshape(-1, n_features)
        
        if torch.isnan(reshaped_acts).any():
            print("Warning: NaN values detected. Replacing with zeros.")
            reshaped_acts = torch.nan_to_num(reshaped_acts, nan=0.0)
        
        # Add small noise to avoid constant features
        reshaped_acts = reshaped_acts + torch.randn_like(reshaped_acts) * 1e-6
        
        # Use co-activation approach
        feature_co_activation = torch.zeros(n_features, device=device)
        for i in range(n_features):
            if i == target_feature:
                continue
            target_active = reshaped_acts[:, target_feature] > 0.1
            if target_active.sum() > 0:
                feature_co_activation[i] = reshaped_acts[target_active, i].mean()
        
        top_co_activated = torch.argsort(feature_co_activation, descending=True)[:5]
        
        print(f"\nFeatures most co-activated with feature {target_feature}:")
        for feat_idx in top_co_activated:
            co_act = feature_co_activation[feat_idx].item()
            print(f"Feature {feat_idx}: Co-activation = {co_act:.4f}")
        
        return top_co_activated[0].item()

def optimize_for_feature(target_feature, config):
    """Optimize input embeddings to maximize activation of target feature."""
    length = config['length']
    max_steps = config['max_steps']
    lambda_reg = config['lambda_reg']
    noise_scale = config['noise_scale']
    diversity_penalty = config['diversity_penalty']
    verbose = config['verbose']
    
    print(f"\nOptimizing for feature {target_feature}...")
    
    # Initialize with random token embeddings
    P = torch.nn.Parameter(
        pythia.W_E[torch.randint(0, pythia.cfg.d_vocab, (length,))].clone() + 
        torch.randn(length, pythia.cfg.d_model, device=device) * noise_scale,
        requires_grad=True
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW([P], lr=1e-1, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=20, verbose=True
    )
    
    # Create dummy tokens
    dummy_tokens = torch.zeros(1, length, dtype=torch.long, device=device)
    
    # Track statistics
    stats = {
        'loss': [], 'feature_loss': [], 'reg_loss': [],
        'target_activation': [], 'gradient_norm': [],
        'embedding_distance': [], 'similarity_top': [],
        'learning_rates': []
    }
    
    # Initial similarity and activation
    with torch.no_grad():
        vocab_embeds = pythia.W_E
        similarity = torch.nn.functional.normalize(P, dim=-1) @ \
                    torch.nn.functional.normalize(vocab_embeds, dim=-1).T
        initial_top_sim = similarity.max(dim=-1).values.mean().item()
        stats['similarity_top'].append(initial_top_sim)
        
        with pythia.hooks(fwd_hooks=[('hook_embed', embed_hook)]):
            _, cache = pythia.run_with_cache_with_saes(
                input=dummy_tokens,
                return_type="logits",
                saes=[sae]
            )
        initial_activation = cache['blocks.3.hook_mlp_out.hook_sae_acts_post'][0, :, target_feature].max().item()
        print(f"Initial activation: {initial_activation:.4f}, Initial similarity: {initial_top_sim:.4f}")
    
    # Print initial tokens
    initial_tokens = get_similar_tokens(P)
    print("\nInitial tokens:")
    for pos, (tokens, scores) in enumerate(initial_tokens):
        print(f"Position {pos}: {tokens[0]} ({scores[0]:.3f})")
    
    # Training loop
    for step in range(max_steps):
        optimizer.zero_grad()
        
        with pythia.hooks(fwd_hooks=[('hook_embed', embed_hook)]):
            _, cache = pythia.run_with_cache_with_saes(
                input=dummy_tokens,
                return_type="logits",
                saes=[sae]
            )
        
        sae_acts = cache['blocks.3.hook_mlp_out.hook_sae_acts_post'][0]
        target_activation = sae_acts[:, target_feature].max()
        loss_feature = -target_activation
        
        # Regularization
        with torch.no_grad():
            similarity = torch.nn.functional.normalize(P, dim=-1) @ \
                        torch.nn.functional.normalize(vocab_embeds, dim=-1).T
            closest_tokens = similarity.max(dim=1).indices
            closest_embeddings = vocab_embeds[closest_tokens]
        
        embedding_diff = P - closest_embeddings
        embedding_dist = torch.norm(embedding_diff, p='fro')
        loss_reg = lambda_reg * embedding_dist
        
        # Diversity penalty
        token_diversity_penalty = 0.0
        with torch.no_grad():
            position_similarity = torch.zeros(length, length, device=device)
            for i in range(length):
                for j in range(i+1, length):
                    sim = torch.nn.functional.cosine_similarity(P[i:i+1], P[j:j+1], dim=1)
                    position_similarity[i,j] = sim
                    position_similarity[j,i] = sim
            token_diversity_penalty = position_similarity.mean() * diversity_penalty
        
        loss = loss_feature + loss_reg + token_diversity_penalty
        
        # Record stats
        stats['loss'].append(loss.item())
        stats['feature_loss'].append(loss_feature.item())
        stats['reg_loss'].append(loss_reg.item())
        stats['target_activation'].append(target_activation.item())
        
        if step % 20 == 0:
            print(f"\nStep {step}")
            print(f"Feature loss: {loss_feature.item():.4f}, Reg loss: {loss_reg.item():.4f}")
        
        loss.backward()
        
        grad_norm = P.grad.norm().item()
        stats['gradient_norm'].append(grad_norm)
        
        torch.nn.utils.clip_grad_norm_([P], max_norm=10.0)
        optimizer.step()
        
        if step % 10 == 0:
            scheduler.step(loss)
        
        if step % 20 == 0 or step == max_steps - 1:
            with torch.no_grad():
                similarity = torch.nn.functional.normalize(P, dim=-1) @ \
                            torch.nn.functional.normalize(vocab_embeds, dim=-1).T
                top_sim = similarity.max(dim=1).values.mean().item()
                stats['similarity_top'].append(top_sim)
                
                similar_tokens = get_similar_tokens(P, top_k=1)
                tokens_str = " ".join([t[0][0] for t in similar_tokens])
                print(f"Current tokens: {tokens_str}")
    
    # Analyze results
    analyze_results(P, target_feature, dummy_tokens)
    
    return P, stats

def visualize_training(stats):
    """Visualize the training progress."""
    fig = sp.make_subplots(
        rows=2, cols=2,
        subplot_titles=('Loss Components', 'Target Activation', 
                        'Gradient Norm', 'Token Similarity')
    )
    
    fig.add_trace(
        go.Scatter(y=stats['feature_loss'], mode='lines', name='Feature Loss'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(y=stats['reg_loss'], mode='lines', name='Reg Loss'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(y=stats['target_activation'], mode='lines', name='Target Activation'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(y=stats['gradient_norm'], mode='lines', name='Gradient Norm'),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(y=stats['similarity_top'], mode='lines+markers', name='Token Similarity'),
        row=2, col=2
    )
    
    fig.update_layout(height=700, width=900, title_text="Training Progress")
    fig.show()

def analyze_results(P, target_feature, dummy_tokens):
    """Analyze the optimization results."""
    with torch.no_grad():
        # Get final activations
        with pythia.hooks(fwd_hooks=[('hook_embed', embed_hook)]):
            _, final_cache = pythia.run_with_cache_with_saes(
                input=dummy_tokens,
                return_type="logits",
                saes=[sae]
            )
        
        final_acts = final_cache['blocks.3.hook_mlp_out.hook_sae_acts_post'][0]
        
        # Show optimized tokens
        top_tokens = get_similar_tokens(P, top_k=5)
        
        print(f"\n\n=== RESULTS FOR FEATURE {target_feature} ===")
        print("\nOptimized sequence as tokens:")
        for pos, (tokens, scores) in enumerate(top_tokens):
            print(f"Position {pos}:")
            for token, score in zip(tokens, scores):
                print(f"  {token:20} (similarity: {score:.3f})")
        
        # Show activation pattern
        print(f"\nFeature {target_feature} activation pattern:")
        for pos in range(len(P)):
            print(f"Position {pos}: {final_acts[pos, target_feature]:.4f}")
        
        # Plot activation pattern
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(range(len(P))),
            y=final_acts[:, target_feature].cpu().numpy(),
            marker_color='red'
        ))
        
        fig.update_layout(
            title=f'Feature {target_feature} Activation Pattern',
            xaxis_title='Sequence Position',
            yaxis_title='Activation Value',
        )
        
        fig.show()
        
        # Check other activated features
        mean_acts = final_acts.mean(dim=0)
        top_activated = torch.argsort(mean_acts, descending=True)[:10]
        
        print("\nOther strongly activated features:")
        for i, feat_idx in enumerate(top_activated):
            if feat_idx == target_feature:
                print(f"Feature {feat_idx}: {mean_acts[feat_idx]:.4f} (target feature)")
            else:
                print(f"Feature {feat_idx}: {mean_acts[feat_idx]:.4f}")

# %% Experiment Runner

def run_experiment(config):
    """Run a feature optimization experiment."""
    global P
    
    # Use config parameters throughout
    if config['experiment_type'] == 'high_activation':
        target_feature, _ = find_high_activation_features(
            n_samples=config['n_samples'], 
            length=config['length']
        )
    elif config['experiment_type'] == 'contextual':
        target_feature = find_contextual_features()
    elif config['experiment_type'] == 'co_activated':
        # First find a high activation feature
        base_feature, _ = find_high_activation_features(length=config['length'])
        # Then find a co-activated feature
        target_feature = find_co_activated_features(base_feature, length=config['length'])
    else:
        raise ValueError(f"Unknown experiment type: {config['experiment_type']}")
    
    # Pass the entire config to optimize_for_feature
    P, stats = optimize_for_feature(target_feature, config)
    
    # Show plots based on config
    if config['show_plots']:
        visualize_training(stats)
    
    return target_feature, P, stats

# %% Run Experiments

config = {
    # Experiment type
    'experiment_type': 'high_activation',  # Options: 'high_activation', 'contextual', 'co_activated'
    
    # Sequence parameters
    'length': 4,  # Length of token sequence to optimize
    
    # Optimization parameters
    'max_steps': 100,  # Number of optimization steps
    'learning_rate': 1e-1,  # Initial learning rate
    'lambda_reg': 1e-3,  # Regularization strength
    'diversity_penalty': 0.1,  # Penalty for token similarity across positions
    
    # Initialization parameters
    'noise_scale': 0.1,  # Scale of noise added to initial embeddings
    
    # Feature selection parameters
    'n_samples': 100,  # Number of samples for feature selection
    'top_k': 5,  # Number of top features to consider
    
    # Visualization
    'show_plots': True,  # Whether to show plots
    'verbose': True,  # Whether to print detailed progress
}

feature, P, stats = run_experiment(config)

# %%

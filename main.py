#%% imports
import torch
from sae_lens import SAE, HookedSAETransformer
from transformer_lens import ActivationCache, utils
import plotly.graph_objects as go
import plotly.subplots as sp
import gc

# Global model variables
model = None
sae = None

def load_models(config):
    """Load the transformer model and SAE based on config."""
    global model, sae
    
    # Only load if not already loaded
    if model is None or sae is None:
        device = config['device']
        model = HookedSAETransformer.from_pretrained("EleutherAI/pythia-70m-deduped", device=device)
        sae, _, _ = SAE.from_pretrained(
            release="pythia-70m-deduped-mlp-sm",
            sae_id="blocks.3.hook_mlp_out",
    device=device
)

    return model, sae

def clear_cache():
    """Clear CUDA cache to free memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# %% Helper functions
def create_embed_hook(P):
    """Create a hook function that uses the provided embeddings P."""
    def hook(value, hook):
        return P.unsqueeze(0)
    return hook

def get_feature_activations(model, sae, tokens, P=None):
    """Compute feature activations for given tokens."""
    hooks = [('hook_embed', create_embed_hook(P))] if P is not None else []
    with model.hooks(fwd_hooks=hooks):
        _, cache = model.run_with_cache_with_saes(tokens, saes=[sae])
    return cache['blocks.3.hook_mlp_out.hook_sae_acts_post']

def get_similar_tokens(model, embeddings, top_k=5):
    """Get the most similar tokens for each position in the embeddings."""
    with torch.no_grad():
        vocab_embeds = model.W_E
        similarity = torch.nn.functional.normalize(embeddings, dim=-1) @ \
                     torch.nn.functional.normalize(vocab_embeds, dim=-1).T
        top_tokens = similarity.topk(top_k, dim=-1)
        
        # Return tokens grouped by position
        result = []
        for pos in range(len(embeddings)):
            pos_tokens = []
            for i in range(top_k):
                token = model.to_string(top_tokens.indices[pos][i])
                score = top_tokens.values[pos][i].item()
                pos_tokens.append((token, score))
            result.append(pos_tokens)
        return result

def find_target_feature(model, sae, config):
    """Find a target feature based on the experiment type."""
    length = config['length']
    device = model.cfg.device

    if config['experiment_type'] == 'high_activation':
        sample_tokens = torch.randint(0, model.cfg.d_vocab, (config['n_samples'], length), device=device)
        acts = get_feature_activations(model, sae, sample_tokens)
        max_acts = acts.max(dim=0).values.max(dim=0).values
        return torch.argmax(max_acts).item()

    elif config['experiment_type'] == 'contextual':
        prompts = config['prompts']
        variances = torch.zeros(sae.cfg.d_sae, device=device)
        for prompt in prompts:
            acts = get_feature_activations(model, sae, model.to_tokens(prompt))
            variances += acts[0].var(dim=0)
        return torch.argmax(variances).item()

    elif config['experiment_type'] == 'co_activated':
        base_feature = find_target_feature(model, sae, {**config, 'experiment_type': 'high_activation'})
        sample_tokens = torch.randint(0, model.cfg.d_vocab, (config['n_samples'], length), device=device)
        acts = get_feature_activations(model, sae, sample_tokens).reshape(-1, sae.cfg.d_sae)
        target_active = acts[:, base_feature] > 0.1
        co_acts = acts[target_active].mean(dim=0)
        co_acts[base_feature] = -float('inf')  # Exclude the base feature
        return torch.argmax(co_acts).item()

    raise ValueError(f"Unknown experiment type: {config['experiment_type']}")

#%% optimization
def optimize_embeddings(model, sae, target_feature, config):
    """Optimize embeddings to maximize activation of the target feature."""
    length = config['length']
    device = model.cfg.device
    P = torch.nn.Parameter(
        model.W_E[torch.randint(0, model.cfg.d_vocab, (length,))] + 
        torch.randn(length, model.cfg.d_model, device=device) * config['noise_scale']
    )
    optimizer = torch.optim.AdamW([P], lr=config['lr'], weight_decay=0)
    dummy_tokens = torch.zeros(1, length, dtype=torch.long, device=device)
    stats = {'loss': [], 'target_activation': []}

    for step in range(config['max_steps']):
        optimizer.zero_grad()
        acts = get_feature_activations(model, sae, dummy_tokens, P)
        
        # Feature activation loss
        target_activation = acts[0, :, target_feature].max()
        loss_feature = -target_activation
        
        # Regularization loss
        with torch.no_grad():
            similarity = torch.nn.functional.normalize(P, dim=-1) @ \
                        torch.nn.functional.normalize(model.W_E, dim=-1).T
            closest_tokens = similarity.max(dim=1).indices
            closest_embeddings = model.W_E[closest_tokens]
        
        embedding_diff = P - closest_embeddings
        embedding_dist = torch.norm(embedding_diff, p='fro')
        loss_reg = config['lambda_reg'] * embedding_dist
        
        # Diversity penalty
        token_diversity_penalty = 0.0
        if config['diversity_penalty'] > 0:
            with torch.no_grad():
                position_similarity = torch.zeros(config['length'], config['length'], device=model.cfg.device)
                for i in range(config['length']):
                    for j in range(i+1, config['length']):
                        sim = torch.nn.functional.cosine_similarity(P[i:i+1], P[j:j+1], dim=1)
                        position_similarity[i,j] = sim
                        position_similarity[j,i] = sim
                token_diversity_penalty = position_similarity.mean() * config['diversity_penalty']
        
        # Total loss
        loss = loss_feature + loss_reg + token_diversity_penalty
        
        loss.backward()
        optimizer.step()
        stats['loss'].append(loss.item())
        stats['target_activation'].append(acts[0, :, target_feature].max().item())

    return P, stats

# %% analyze, visualize
def analyze_results(model, sae, P, target_feature, config):
    """Analyze and visualize optimization results."""
    dummy_tokens = torch.zeros(1, config['length'], dtype=torch.long, device=model.cfg.device)
    acts = get_feature_activations(model, sae, dummy_tokens, P)
    tokens_by_pos = get_similar_tokens(model, P, top_k=1)
    
    print(f"\n=== RESULTS FOR FEATURE {target_feature} ===")
    print(f"Optimized tokens: {' '.join(tokens_by_pos[pos][0][0] for pos in range(config['length']))}")
    print(f"Target feature activation: {acts[0, :, target_feature].max():.4f}")
    
    # Show activation by position
    print("\nActivation by position:")
    for pos in range(config['length']):
        token, score = tokens_by_pos[pos][0]  # Get first token for this position
        print(f"Position {pos}: {acts[0, pos, target_feature].item():.4f} - '{token}' ({score:.3f})")

def visualize_training(stats):
    """Visualize training progress."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=stats['loss'], mode='lines', name='Loss'))
    fig.add_trace(go.Scatter(y=stats['target_activation'], mode='lines', name='Target Activation'))
    fig.update_layout(title="Training Progress", xaxis_title="Step", yaxis_title="Value")
    fig.show()

# %% Experiment Runner
def run_experiment(config):
    """Run an experiment with the given configuration."""
    model, sae = load_models(config)
    target_feature = find_target_feature(model, sae, config)
    P, stats = optimize_embeddings(model, sae, target_feature, config)
    analyze_results(model, sae, P, target_feature, config)
    if config['visualize']:
        visualize_training(stats)
    
    # Clear cache after run
    clear_cache()
    
    return target_feature, P, stats

config = {
    # Experiment type
    'experiment_type': 'co_activated',  # Options: 'high_activation', 'contextual', 'co_activated'
    
    # Sequence parameters
    'length': 50,  # Length of token sequence to optimize
    
    # Optimization parameters
    'max_steps': 100,  # Number of optimization steps
    'lr': 1e-1,  # Initial learning rate
    'lambda_reg': 1e-3,  # Regularization strength
    'diversity_penalty': 0.1,  # Penalty for token similarity across positions
    
    # Initialization parameters
    'noise_scale': 0.1,  # Scale of noise added to initial embeddings
    
    # Feature selection parameters
    'n_samples': 100,  # Number of samples for feature selection
    'top_k': 5,  # Number of top features to consider
    'prompts': [
        "Jimmy carter was the president of the United States",
        "Luke went to the store to buy",
        "My favorite color is",
        "The sum of 2 and 2 is",
        "You should use an if statement"
    ],  # Prompts for contextual feature finding
    
    # Visualization and output
    'visualize': True,  # Whether to show training plots
    'verbose': True,  # Whether to print detailed progress
    
    # Hardware
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'  # Device to run on
}

# Load models once at the beginning
model, sae = load_models(config)

feature, P, stats = run_experiment(config)
        
# %%

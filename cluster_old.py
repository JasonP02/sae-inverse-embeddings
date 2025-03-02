# %% imports
import torch
from sae_lens import SAE, HookedSAETransformer
from transformer_lens import ActivationCache, utils
import plotly.graph_objects as go
import plotly.subplots as sp
import gc
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
import umap.umap_ as umap
import random
from datasets import load_dataset
import re
import csv
import os
import datetime
import itertools

# Global model variables
model = None
sae = None

def load_models(config):
    """Load the transformer model and SAE based on config."""
    global model, sae
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
    def hook(value, hook):
        return P.unsqueeze(0)
    return hook

def get_feature_activations(model, sae, tokens, P=None):
    hooks = [('hook_embed', create_embed_hook(P))] if P is not None else []
    with model.hooks(fwd_hooks=hooks):
        _, cache = model.run_with_cache_with_saes(tokens, saes=[sae])
    return cache['blocks.3.hook_mlp_out.hook_sae_acts_post']

def get_similar_tokens(model, embeddings, top_k=5):
    with torch.no_grad():
        vocab_embeds = model.W_E
        similarity = torch.nn.functional.normalize(embeddings, dim=-1) @ \
                     torch.nn.functional.normalize(vocab_embeds, dim=-1).T
        top_tokens = similarity.topk(top_k, dim=-1)
        result = []
        for pos in range(len(embeddings)):
            pos_tokens = [(model.to_string(top_tokens.indices[pos][i]), top_tokens.values[pos][i].item()) 
                          for i in range(top_k)]
            result.append(pos_tokens)
        return result

# %% Prompt Loading Functions
def load_diverse_prompts(config):
    n_prompts = config.get('n_prompts', 100)
    prompts = []

    # Use a simple dataset: 'wikitext'
    try:
        print("Loading prompts from WikiText...")
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        wiki_prompts = [item['text'].strip() for item in dataset if len(item['text'].strip()) > 10]
        prompts.extend(wiki_prompts[:n_prompts])
        print(f"Added {len(wiki_prompts[:n_prompts])} prompts from WikiText")
    except Exception as e:
        print(f"Error with WikiText: {e}")

    prompts = list(set(prompts))  # Deduplicate
    random.shuffle(prompts)
    prompts = prompts[:n_prompts]
    print(f"Total unique prompts loaded: {len(prompts)}")
    return prompts

# %% Clustering Functions
def collect_activations(model, sae, prompts):
    all_acts = []
    batch_size = 10
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{len(prompts) // batch_size + 1}...")
        batch_acts = []
        for prompt in batch_prompts:
            try:
                tokens = model.to_tokens(prompt)
                acts = get_feature_activations(model, sae, tokens)
                batch_acts.append(acts.mean(dim=1).squeeze(0))
            except Exception as e:
                print(f"Skipping prompt '{prompt[:30]}...': {e}")
        if batch_acts:
            all_acts.extend(batch_acts)
        if (i + batch_size) % 50 == 0:
            clear_cache()
    acts = torch.stack(all_acts)
    print(f"Collected activations for {acts.shape[0]} prompts, {acts.shape[1]} features")
    return acts

def filter_features(acts, config):
    """Filter features based on entropy and sparsity."""
    n_prompts, n_features = acts.shape
    entropy = torch.zeros(n_features, device=acts.device)
    sparsity = torch.zeros(n_features, device=acts.device)
    
    for i in range(n_features):
        # Use absolute values for entropy calculation
        activations = acts[:, i].abs()  # Changed from clamp(min=0)
        # Normalize to get probability distribution
        probs = activations / (activations.sum() + 1e-10)
        entropy[i] = -torch.sum(probs * torch.log(probs + 1e-10))
        # Consider both positive and negative activations for sparsity
        sparsity[i] = (acts[:, i].abs() > config['activation_threshold']).float().mean()
    
    print(f"Entropy range: {entropy.min():.2f} to {entropy.max():.2f}")
    print(f"Sparsity range: {sparsity.min():.2f} to {sparsity.max():.2f}")
    
    # Add this before applying the mask in filter_features
    print(f"Entropy percentiles: 10%={torch.quantile(entropy, 0.1):.4f}, 50%={torch.quantile(entropy, 0.5):.4f}, 90%={torch.quantile(entropy, 0.9):.4f}")
    print(f"Sparsity percentiles: 10%={torch.quantile(sparsity, 0.1):.4f}, 50%={torch.quantile(sparsity, 0.5):.4f}, 90%={torch.quantile(sparsity, 0.9):.4f}")
    
    mask = (entropy > config['entropy_threshold_low']) & \
           (entropy < config['entropy_threshold_high']) & \
           (sparsity > config['sparsity_min']) & \
           (sparsity < config['sparsity_max'])
    
    print(f"Entropy range: {entropy.min():.2f} to {entropy.max():.2f}")
    print(f"Sparsity range: {sparsity.min():.2f} to {sparsity.max():.2f}")
    print(f"Kept {mask.sum().item()} out of {n_features} features")
    return acts[:, mask], mask.nonzero(as_tuple=True)[0]

def cluster_features(acts, config):
    """Cluster features directly using DBSCAN with cosine distance."""
    if acts.shape[1] <= 1:
        print(f"Warning: Only {acts.shape[1]} features after filtering. Skipping clustering.")
        return np.zeros(acts.shape[1], dtype=int), acts.T.cpu().numpy()
    
    # Transpose to [n_features, n_prompts]
    feature_acts = acts.T.cpu().numpy()  # Shape: [n_features, n_prompts]
    
    # Normalize to unit norm (optional, as cosine distance is scale-invariant)
    norms = np.linalg.norm(feature_acts, axis=1, keepdims=True)
    normalized_acts = feature_acts / (norms + 1e-10)  # Avoid division by zero
    
    # DBSCAN with cosine distance
    eps_values = np.linspace(config['dbscan_eps_min'], 
                            config['dbscan_eps_max'], 
                            config['dbscan_eps_steps'])
    best_eps, best_score = None, -1
    for eps in eps_values:
        clusterer = DBSCAN(eps=eps, min_samples=config['dbscan_min_samples'], metric='cosine')
        labels = clusterer.fit_predict(normalized_acts)
        n_clusters = len(np.unique(labels)) - (1 if -1 in labels else 0)  # Exclude noise
        if n_clusters > 1 and n_clusters < len(normalized_acts):
            score = silhouette_score(normalized_acts, labels, metric='cosine')
            if score > best_score:
                best_score = score
                best_eps = eps
    if best_eps is None:
        print("No valid clustering found, defaulting to eps midpoint")
        best_eps = (config['dbscan_eps_min'] + config['dbscan_eps_max']) / 2
    
    clusterer = DBSCAN(eps=best_eps, min_samples=config['dbscan_min_samples'], metric='cosine')
    labels = clusterer.fit_predict(normalized_acts)
    n_clusters = len(np.unique(labels)) - (1 if -1 in labels else 0)
    print(f"DBSCAN with eps={best_eps:.2f}, found {n_clusters} clusters (excluding noise)")
    
    return labels, normalized_acts

def visualize_clusters(reduced_acts, labels):
    """Visualize clusters using UMAP."""
    try:
        reducer = umap.UMAP(n_components=2, random_state=42)
        umap_acts = reducer.fit_transform(reduced_acts)
        
        fig = go.Figure(data=[go.Scatter(x=umap_acts[:, 0], y=umap_acts[:, 1], mode='markers',
                                         marker=dict(color=labels, colorscale='Viridis'))])
        fig.update_layout(title="Cluster Visualization with UMAP", xaxis_title="UMAP 1", yaxis_title="UMAP 2")
        fig.show()
    except Exception as e:
        print(f"Error visualizing clusters: {e}")
        print("Continuing without visualization...")

def score_features(acts, config):
    """Score features based on the specified method in config."""
    selection_method = config.get('feature_selection_method', 'mean')
    
    if selection_method == 'mean':
        return acts.mean(dim=0)
    elif selection_method == 'max':
        return acts.max(dim=0)[0]  # [0] because max returns values and indices
    elif selection_method == 'percentile':
        percentile = config.get('activation_percentile', 90)
        return torch.quantile(acts, percentile/100.0, dim=0)
    else:
        print(f"Unknown selection method '{selection_method}', defaulting to mean")
        return acts.mean(dim=0)

def select_target_features(acts, labels, original_indices, config):
    """Select top features from interesting clusters."""
    # Get feature selection parameters from config
    top_n = config.get('features_per_cluster', 1)
    
    target_features = []
    for label in np.unique(labels):
        if label == -1:  # Noise in DBSCAN
            continue
        cluster_mask = labels == label
        cluster_acts = acts[:, cluster_mask]
        cluster_indices = original_indices[cluster_mask]
        
        # Score features using the specified method
        feature_scores = score_features(cluster_acts, config)
            
        # Make sure we have valid scores before sorting
        if feature_scores.numel() > 0:
            sorted_indices = torch.argsort(feature_scores, descending=True)[:min(top_n, feature_scores.numel())]
            top_indices = cluster_indices[sorted_indices.cpu()]  # Move indices to CPU
            target_features.extend(top_indices.tolist())
            print(f"Cluster {label}: Selected features {top_indices.tolist()}")
        else:
            print(f"Cluster {label}: No valid features found")
            
    print(f"Total target features: {len(target_features)}")
    return target_features

#%% Optimization (unchanged)
def optimize_embeddings(model, sae, target_feature, config):
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
        target_activation = acts[0, :, target_feature].max()
        loss_feature = -target_activation
        with torch.no_grad():
            similarity = torch.nn.functional.normalize(P, dim=-1) @ \
                        torch.nn.functional.normalize(model.W_E, dim=-1).T
            closest_tokens = similarity.max(dim=1).indices
            closest_embeddings = model.W_E[closest_tokens]
        embedding_diff = P - closest_embeddings
        embedding_dist = torch.norm(embedding_diff, p='fro')
        loss_reg = config['lambda_reg'] * embedding_dist
        
        token_diversity_penalty = 0.0
        if config['diversity_penalty'] > 0:
            window_size = config.get('diversity_window', 5)
            with torch.no_grad():
                sim_matrix = torch.nn.functional.cosine_similarity(
                    P.unsqueeze(1), P.unsqueeze(0), dim=2
                )  # [length, length]
                mask = torch.triu(torch.ones_like(sim_matrix), diagonal=1)  # Upper triangle
                window_mask = torch.abs(torch.arange(config['length'], device=device).unsqueeze(0) - 
                                    torch.arange(config['length'], device=device).unsqueeze(1)) <= window_size
                local_mask = mask * window_mask
                token_diversity_penalty = (sim_matrix * local_mask).sum() / local_mask.sum() * config['diversity_penalty']
        else:
            token_diversity_penalty = torch.tensor(0.0, device=device)

        repetition_penalty = 0.0
        if config['repetition_penalty'] > 0:
            window_size = config.get('repetition_window', 5)
            with torch.no_grad():
                local_repeats = torch.zeros(config['length'], device=device)
                for offset in range(1, window_size + 1):
                    # Compare with tokens shifted forward and backward
                    forward = torch.cat([closest_tokens[offset:], torch.full((offset,), -1, device=device)])
                    backward = torch.cat([torch.full((offset,), -1, device=device), closest_tokens[:-offset]])
                    local_repeats += (closest_tokens == forward).float() + (closest_tokens == backward).float()
                repetition_penalty = config['repetition_penalty'] * local_repeats.sum() / config['length']
        else:
            repetition_penalty = torch.tensor(0.0, device=device)

        loss = loss_feature + loss_reg + token_diversity_penalty + repetition_penalty
        loss.backward()
        optimizer.step()
        stats['loss'].append(loss.item())
        stats['target_activation'].append(acts[0, :, target_feature].max().item())
        
        if step % 20 == 0 and config.get('verbose', True):
            rep_val = repetition_penalty.item() if isinstance(repetition_penalty, torch.Tensor) else repetition_penalty
            div_val = token_diversity_penalty.item() if isinstance(token_diversity_penalty, torch.Tensor) else token_diversity_penalty
            print(f"Step {step}: repetition_penalty = {rep_val:.4f}, token_diversity_penalty = {div_val:.4f}, loss = {loss.item():.4f}, target_activation = {target_activation.item():.4f}")
            
    return P, stats

# %% Analyze, Visualize
def analyze_results(model, sae, P, target_feature, config):
    dummy_tokens = torch.zeros(1, config['length'], dtype=torch.long, device=model.cfg.device)
    acts = get_feature_activations(model, sae, dummy_tokens, P)
    tokens_by_pos = get_similar_tokens(model, P, top_k=1)

    # Calculate statistics
    feature_acts = acts[0, :, target_feature]
    max_act = feature_acts.max().item()
    mean_act = feature_acts.mean().item()
    
    # Calculate skewness manually
    centered = feature_acts - mean_act
    std = feature_acts.std().item()
    if std > 0:
        skew = torch.mean(centered**3).item() / (std**3)
    else:
        skew = 0.0
        
    print(f"Feature {target_feature}: Max Act={max_act:.4f}, Mean Act={mean_act:.4f}, Skew={skew:.4f}")
    
    print(f"\n=== RESULTS FOR FEATURE {target_feature} ===")
    
    # Show top activating positions
    pos_activations = [(pos, acts[0, pos, target_feature].item()) for pos in range(config['length'])]
    top_positions = sorted(pos_activations, key=lambda x: x[1], reverse=True)[:10]
    
    print(f"Target feature activation: {acts[0, :, target_feature].max():.4f}")
    print("\nTop activating positions:")
    for pos, act_val in top_positions:
        token, score = tokens_by_pos[pos][0]
        print(f"Position {pos}: {act_val:.4f} - '{token}' ({score:.3f})")
    
    # Show full sequence for reference
    if config.get('show_full_sequence', False):
        print("\nFull optimized sequence:")
        for pos in range(config['length']):
            token, score = tokens_by_pos[pos][0]
            print(f"Position {pos}: {acts[0, pos, target_feature].item():.4f} - '{token}' ({score:.3f})")

def visualize_training(stats):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=stats['loss'], mode='lines', name='Loss'))
    fig.add_trace(go.Scatter(y=stats['target_activation'], mode='lines', name='Target Activation'))
    fig.update_layout(title="Training Progress", xaxis_title="Step", yaxis_title="Value")
    fig.show()

# %% Data Caching Functions
def save_processed_data(data, filename):
    """Save processed data to disk to avoid reprocessing.
    
    Args:
        data: Dictionary containing processed data
        filename: Path to save the data
    """
    print(f"Saving processed data to {filename}...")
    torch.save(data, filename)
    print("Data saved successfully.")

def load_processed_data(filename):
    """Load processed data from disk.
    
    Args:
        filename: Path to the saved data
        
    Returns:
        Dictionary containing processed data or None if file doesn't exist
    """
    if os.path.exists(filename):
        print(f"Loading processed data from {filename}...")
        try:
            data = torch.load(filename)
            print("Data loaded successfully.")
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    else:
        print(f"No cached data found at {filename}")
        return None

def get_cache_filename(config):
    """Generate a cache filename based on config parameters.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        String with cache filename
    """
    # Create a unique filename based on key parameters
    params = [
        f"prompts_{config['n_prompts']}",
        f"entropy_{config['entropy_threshold_low']:.2f}_{config['entropy_threshold_high']:.2f}",
        f"sparsity_{config['sparsity_min']:.2f}_{config['sparsity_max']:.2f}"
    ]
    
    # Create directory if it doesn't exist
    cache_dir = config.get('cache_dir', 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    return os.path.join(cache_dir, f"processed_data_{'_'.join(params)}.pt")

# %% Experiment Runner
def run_experiment(config, use_cached_data=True):
    model, sae = load_models(config)
    
    # Try to load cached data if requested
    cache_filename = get_cache_filename(config)
    cached_data = None
    
    if use_cached_data:
        cached_data = load_processed_data(cache_filename)
    
    if cached_data is not None:
        # Use cached data
        filtered_acts = cached_data['filtered_acts']
        original_indices = cached_data['original_indices']
        print(f"Using cached data with {filtered_acts.shape[0]} prompts and {filtered_acts.shape[1]} filtered features")
    else:
        # Process data from scratch
        # Step 1: Load diverse prompts
        prompts = load_diverse_prompts(config)
        
        # Step 2: Collect activations from prompts
        acts = collect_activations(model, sae, prompts)
        
        # Step 3: Filter features
        filtered_acts, original_indices = filter_features(acts, config)
        
        # Save processed data for future use if requested
        if config.get('cache_data', True):
            save_processed_data({
                'filtered_acts': filtered_acts,
                'original_indices': original_indices
            }, cache_filename)
    
    # Step 4: Cluster features
    labels, reduced_acts = cluster_features(filtered_acts, config)
    
    # Step 5: Visualize clusters (optional)
    if config.get('visualize_clusters', True):
        visualize_clusters(reduced_acts, labels)
    
    # Step 6: Select target features from clusters
    target_features = select_target_features(filtered_acts, labels, original_indices, config)
    
    # Step 7: Run prompt optimization for each feature
    feature_results = []
    for i, target_feature in enumerate(target_features):
        print(f"\nOptimizing for feature {target_feature} ({i+1}/{len(target_features)})")
        P, stats = optimize_embeddings(model, sae, target_feature, config)
        analyze_results(model, sae, P, target_feature, config)
        
        if config.get('visualize_training', True):
            visualize_training(stats)
        
        # Store comprehensive results for each feature
        dummy_tokens = torch.zeros(1, config['length'], dtype=torch.long, device=config['device'])
        acts = get_feature_activations(model, sae, dummy_tokens, P)
        tokens_by_pos = get_similar_tokens(model, P, top_k=1)
        
        # Extract high-activating tokens with their activations
        pos_activations = [(pos, acts[0, pos, target_feature].item(), tokens_by_pos[pos][0][0], tokens_by_pos[pos][0][1]) 
                           for pos in range(config['length'])]
        sorted_activations = sorted(pos_activations, key=lambda x: x[1], reverse=True)
        
        # Store all the data we need for this feature
        feature_data = {
            'feature_id': target_feature,
            'embeddings': P.detach().clone(),
            'stats': stats,
            'activations': sorted_activations,
            'high_act_tokens': [(token, act) for _, act, token, _ in sorted_activations if act > config['activation_threshold']]
        }
        feature_results.append(feature_data)
        
        # Clear cache between features
        clear_cache()
    
    # Return comprehensive results
    return {
        'target_features': target_features,
        'feature_results': feature_results
    }

# %% Explanation Functions
def generate_explanatory_prompt(model, sae, feature_id, high_act_tokens, config):
    """Generate a coherent explanation using high-activating tokens."""
    # Template options
    templates = [
        "This feature detects {0} related to {1}.",
        "This feature activates for {0} in the context of {1}.",
        "This feature represents {0} within {1} systems.",
        "This feature responds to {0} and {1}."
    ]
    
    results = []
    for template in templates:
        # Try different permutations of tokens
        for tokens in itertools.permutations(high_act_tokens[:5], 2):
            prompt = template.format(tokens[0], tokens[1])
            tokens = model.to_tokens(prompt)
            
            # Measure activation
            acts = get_feature_activations(model, sae, tokens)
            activation = acts[0, :, feature_id].max().item()
            
            results.append((prompt, activation))
    
    # Return top activating coherent explanation
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:5]  # Return top 5 candidates


def evaluate_explanations(model, sae, feature_id, explanations):
    """Evaluate multiple explanations for a feature."""
    results = []
    
    for explanation in explanations:
        tokens = model.to_tokens(explanation)
        acts = get_feature_activations(model, sae, tokens)
        activation = acts[0, :, feature_id].max().item()
        
        # Get activations for other features to check specificity
        all_feature_acts = acts[0].max(dim=0)[0]
        specificity = activation / (all_feature_acts.mean().item())
        
        results.append({
            'explanation': explanation,
            'activation': activation,
            'specificity': specificity,
            'length': len(tokens[0])
        })
    
    return results


def calculate_lm_coherence(P, model, lm=lm):
    tokens_by_pos = get_similar_tokens(model, P, top_k=1)
    text = " ".join([t[0][0] for t in tokens_by_pos])
    inputs = tokenizer(text, return_tensors='pt', truncation=True).to(config['device'])
    outputs = lm(**inputs, labels=inputs['input_ids'])
    return outputs.loss

def run_explanation_experiment(model, sae, experiment_results, config):
    """Run explanation generation for features using direct experiment results.
    
    Args:
        model: The transformer model
        sae: The sparse autoencoder
        experiment_results: Results from run_experiment
        config: Configuration dictionary
        
    Returns:
        Dictionary mapping feature IDs to their explanations and metrics
    """
    results = {}
    feature_results = experiment_results['feature_results']
    
    for feature_data in feature_results:
        feature_id = feature_data['feature_id']
        print(f"\n=== Generating explanations for feature {feature_id} ===")
        
        try:
            # Get high-activating tokens directly from experiment results
            high_act_tokens = [token for token, act in feature_data['high_act_tokens']]
            
            if len(high_act_tokens) < 2:
                print(f"Not enough high-activating tokens for feature {feature_id}, skipping")
                continue
                
            print(f"High-activating tokens: {high_act_tokens[:5]}")
            
            # Generate template-based explanations
            template_explanations = generate_explanatory_prompt(
                model, sae, feature_id, high_act_tokens, config
            )
            
            print("\nTop template-based explanations:")
            for i, (explanation, activation) in enumerate(template_explanations):
                print(f"{i+1}. {explanation} (activation: {activation:.4f})")
            
            # Optimize explanation if requested
            if config.get('optimize_explanations', True):
                print("\nOptimizing explanation...")
                
                # Use the existing function but with our direct data
                candidates = template_explanations
                best_prompt, best_activation = candidates[0]
                
                # Initialize with best template
                tokens = model.to_tokens(best_prompt)
                P = torch.nn.Parameter(model.W_E[tokens[0]])
                
                # Define which positions to freeze (e.g., "This feature detects")
                frozen_mask = torch.zeros_like(P, dtype=torch.bool)
                frozen_mask[:4] = True  # Freeze first 4 tokens
                
                # Optimize
                optimizer = torch.optim.AdamW([P], lr=config.get('lr_explanation', config['lr']))
                
                for step in range(config.get('max_steps_explanation', config['max_steps'])):
                    optimizer.zero_grad()
                    
                    # Get activations
                    acts = get_feature_activations(model, sae, torch.zeros(1, P.shape[0], dtype=torch.long), P)
                    activation = acts[0, :, feature_id].max()
                    
                    # Calculate losses
                    feature_loss = -activation
                    
                    # Coherence loss using GPT-2 (optional)
                    if config.get('use_lm_coherence', False):
                        coherence_loss = calculate_lm_coherence(P, model)
                        total_loss = feature_loss + config['coherence_weight'] * coherence_loss
                    else:
                        total_loss = feature_loss
                        
                    # Add regularization
                    with torch.no_grad():
                        similarity = torch.nn.functional.normalize(P, dim=-1) @ \
                                    torch.nn.functional.normalize(model.W_E, dim=-1).T
                        closest_tokens = similarity.max(dim=1).indices
                        closest_embeddings = model.W_E[closest_tokens]
                    
                    embedding_diff = P - closest_embeddings
                    # Don't regularize frozen positions
                    embedding_diff[frozen_mask] = 0
                    embedding_dist = torch.norm(embedding_diff, p='fro')
                    reg_loss = config.get('lambda_reg_explanation', config['lambda_reg']) * embedding_dist
                    
                    total_loss += reg_loss
                    
                    # Backprop and update
                    total_loss.backward()
                    
                    # Zero gradients for frozen positions
                    if P.grad is not None:
                        P.grad[frozen_mask] = 0
                        
                    optimizer.step()
                    
                    # Logging
                    if step % 100 == 0:
                        tokens_by_pos = get_similar_tokens(model, P, top_k=1)
                        current_text = " ".join([t[0][0] for t in tokens_by_pos])
                        print(f"Step {step}: {current_text} (act={activation.item():.4f})")
                
                final_activation = acts[0, :, feature_id].max().item()
                tokens_by_pos = get_similar_tokens(model, P, top_k=1)
                optimized_explanation = " ".join([t[0][0] for t in tokens_by_pos])
                
                print(f"\nOptimized explanation: {optimized_explanation}")
                print(f"Final activation: {final_activation:.4f}")
                
                # Compare with template explanations
                all_explanations = [explanation for explanation, _ in template_explanations]
                all_explanations.append(optimized_explanation)
                
                evaluation = evaluate_explanations(model, sae, feature_id, all_explanations)
                
                # Store results
                results[feature_id] = {
                    'high_act_tokens': high_act_tokens,
                    'template_explanations': template_explanations,
                    'optimized_explanation': optimized_explanation,
                    'final_activation': final_activation,
                    'evaluation': evaluation
                }
            else:
                # Just store template results
                results[feature_id] = {
                    'high_act_tokens': high_act_tokens,
                    'template_explanations': template_explanations
                }
                
        except Exception as e:
            print(f"Error generating explanations for feature {feature_id}: {e}")
            results[feature_id] = {'error': str(e)}
    
    return results

# %% Config
config = {
    # Sequence parameters
    'length': 10,  # Length of token sequence to optimize
    
    # Optimization parameters
    'max_steps': 250,
    'lr': 1e-3,
    'lambda_reg': 1e-5,
    'diversity_penalty': 0.0, # Penalty for token similarity across positions
    'diversity_window': 5,
    'repetition_penalty': 0.0, # Penalty for repeating tokens 
    'repetition_window': 5,
    'noise_scale': 0.5,
    
    # Data collection parameters
    'n_prompts': 100,
    'min_prompt_length': 10,
    'max_prompt_length': 100,
    'batch_size': 10,  # Batch size for processing prompts
    
    # Feature filtering parameters
    'entropy_threshold_low': 0.25,  # Minimum entropy for feature selection
    'entropy_threshold_high': 5.0,  # Maximum entropy for feature selection
    'sparsity_min': 0.1,  # Minimum activation sparsity
    'sparsity_max': 0.95,  # Maximum activation sparsity
    'activation_threshold': 0.1,  # Threshold for considering a feature activated
    
    # Clustering parameters
    'n_clusters': 10,
    'clustering_method': 'dbscan',  # 'dbscan' or 'kmeans'
    'features_per_cluster': 1, # Number of features to select from each cluster
    'feature_selection_method': 'max',  # Options: 'mean', 'max', 'percentile'
    'pca_multiplier': 3,  # n_components = pca_multiplier * n_clusters
    
    # DBSCAN specific parameters
    'dbscan_eps_min': 0.0,    # Suitable for cosine distance
    'dbscan_eps_max': 5,
    'dbscan_eps_steps': 30,
    'dbscan_min_samples': 1,
    
    # Visualization and output
    'visualize_clusters': True,
    'visualize_training': False,
    'show_full_sequence': True,
    'save_results': False,
    'verbose': True,
    
    # Hardware
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Feature selection parameters
    'activation_percentile': 90,  # Percentile to use if method='percentile'
    
    # Output options
    'save_csv': True,  # Save results to CSV files
    'csv_output_dir': 'feature_results',  # Directory to save CSV files
    'use_lm_coherence': True,
    'coherence_weight': 0.1,
    'max_steps_explanation': 1000,
    'lr_explanation': 1e-3,
    'lambda_reg_explanation': 1e-5,
    'length_explanation': 10,
    
    # Data caching parameters
    'cache_data': True,
    'cache_dir': 'feature_cache',
    'use_cached_data': False,
}

from transformers import GPT2LMHeadModel, GPT2Tokenizer
lm = GPT2LMHeadModel.from_pretrained('distilgpt2').to(config['device'])
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')

# Run the experiment with caching
experiment_results = run_experiment(config, use_cached_data=config.get('use_cached_data', True))

# %%

# Run auto-interpretation directly using the experiment results
results_explanation = run_explanation_experiment(model, sae, experiment_results, config)

# Print summary of explanations
print("\n=== FEATURE EXPLANATION SUMMARY ===")
for feature_id, result in results_explanation.items():
    print(f"\nFeature {feature_id}:")
    if 'error' in result:
        print(f"  Error: {result['error']}")
        continue
        
    if 'template_explanations' in result and result['template_explanations']:
        best_template, template_act = result['template_explanations'][0]
        print(f"  Template: {best_template} (act: {template_act:.4f})")
        
    if 'optimized_explanation' in result:
        print(f"  Optimized: {result['optimized_explanation']} (act: {result['final_activation']:.4f})")

# %%
torch.cuda.empty_cache()
gc.collect()
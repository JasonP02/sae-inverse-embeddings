import torch
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import umap.umap_ as umap
import plotly.graph_objects as go
from einops import rearrange

def filter_features(acts, config):
    """Filter features based on entropy and sparsity.
    
    Args:
        acts: Tensor of shape [n_prompts, n_features] containing feature activations
        config: Configuration dictionary with filtering parameters
        
    Returns:
        Tuple of (filtered_acts, original_indices) where filtered_acts is a tensor
        of shape [n_prompts, n_filtered_features] and original_indices is a tensor
        containing the original indices of the filtered features
    """
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
    
    # Print percentiles for better understanding of the distribution
    print(f"Entropy percentiles: 10%={torch.quantile(entropy, 0.1):.4f}, 50%={torch.quantile(entropy, 0.5):.4f}, 90%={torch.quantile(entropy, 0.9):.4f}")
    print(f"Sparsity percentiles: 10%={torch.quantile(sparsity, 0.1):.4f}, 50%={torch.quantile(sparsity, 0.5):.4f}, 90%={torch.quantile(sparsity, 0.9):.4f}")
    
    # Apply filtering mask based on config thresholds
    mask = (entropy > config['entropy_threshold_low']) & \
           (entropy < config['entropy_threshold_high']) & \
           (sparsity > config['sparsity_min']) & \
           (sparsity < config['sparsity_max'])
    
    print(f"Kept {mask.sum().item()} out of {n_features} features")
    return acts[:, mask], mask.nonzero(as_tuple=True)[0]

def cluster_features(acts, config):
    """Cluster features directly using DBSCAN with cosine distance.
    
    Args:
        acts: Tensor of shape [n_prompts, n_filtered_features] containing filtered feature activations
        config: Configuration dictionary with clustering parameters
        
    Returns:
        Tuple of (labels, normalized_acts) where labels is a numpy array of cluster labels
        and normalized_acts is a numpy array of normalized feature activations
    """
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
    """Visualize clusters using UMAP.
    
    Args:
        reduced_acts: Numpy array of shape [n_features, n_prompts] containing normalized feature activations
        labels: Numpy array of cluster labels
        
    Returns:
        None (displays a plotly figure)
    """
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
    """Score features based on the specified method in config.
    
    Args:
        acts: Tensor of shape [n_prompts, n_features] containing feature activations
        config: Configuration dictionary with scoring parameters
        
    Returns:
        Tensor of shape [n_features] containing feature scores
    """
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
    """Select top features from interesting clusters.
    
    Args:
        acts: Tensor of shape [n_prompts, n_filtered_features] containing filtered feature activations
        labels: Numpy array of cluster labels
        original_indices: Tensor containing the original indices of the filtered features
        config: Configuration dictionary with selection parameters
        
    Returns:
        List of selected feature indices in the original feature space
    """
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
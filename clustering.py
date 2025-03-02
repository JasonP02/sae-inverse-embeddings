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
    
    filtering_config = config.get('filtering', {})
    activation_threshold = filtering_config.get('activation_threshold', 0.1)
    
    for i in range(n_features):
        # Use absolute values for entropy calculation
        activations = acts[:, i].abs()  # Changed from clamp(min=0)
        # Normalize to get probability distribution
        probs = activations / (activations.sum() + 1e-10)
        entropy[i] = -torch.sum(probs * torch.log(probs + 1e-10))
        # Consider both positive and negative activations for sparsity
        sparsity[i] = (acts[:, i].abs() > activation_threshold).float().mean()
    
    print(f"Entropy range: {entropy.min():.2f} to {entropy.max():.2f}")
    print(f"Sparsity range: {sparsity.min():.2f} to {sparsity.max():.2f}")
    
    # Print percentiles for better understanding of the distribution
    print(f"Entropy percentiles: 10%={torch.quantile(entropy, 0.1):.4f}, 50%={torch.quantile(entropy, 0.5):.4f}, 90%={torch.quantile(entropy, 0.9):.4f}")
    print(f"Sparsity percentiles: 10%={torch.quantile(sparsity, 0.1):.4f}, 50%={torch.quantile(sparsity, 0.5):.4f}, 90%={torch.quantile(sparsity, 0.9):.4f}")
    
    # Apply filtering mask based on config thresholds
    mask = (entropy > filtering_config.get('entropy_threshold_low', 0.25)) & \
           (entropy < filtering_config.get('entropy_threshold_high', 5.0)) & \
           (sparsity > filtering_config.get('sparsity_min', 0.1)) & \
           (sparsity < filtering_config.get('sparsity_max', 0.95))
    
    print(f"Kept {mask.sum().item()} out of {n_features} features")
    return acts[:, mask], mask.nonzero(as_tuple=True)[0]

def cluster_features(acts, clustering_config):
    """Cluster features directly using DBSCAN with cosine distance.
    
    Args:
        acts: Tensor of shape [n_prompts, n_filtered_features] containing filtered feature activations
        clustering_config: Configuration dictionary with clustering parameters
        
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
    
    # Get DBSCAN parameters from config
    dbscan_config = clustering_config.get('dbscan', {})
    eps_values = np.linspace(
        dbscan_config.get('eps_min', 0.1),
        dbscan_config.get('eps_max', 5),
        dbscan_config.get('eps_steps', 30)
    )
    min_samples = dbscan_config.get('min_samples', 1)
    
    # DBSCAN with cosine distance
    best_eps, best_score = None, -1
    for eps in eps_values:
        clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = clusterer.fit_predict(normalized_acts)
        n_clusters = len(np.unique(labels)) - (1 if -1 in labels else 0)  # Exclude noise
        if n_clusters > 1 and n_clusters < len(normalized_acts):
            score = silhouette_score(normalized_acts, labels, metric='cosine')
            if score > best_score:
                best_score = score
                best_eps = eps
    if best_eps is None:
        print("No valid clustering found, defaulting to eps midpoint")
        best_eps = (dbscan_config.get('eps_min', 0.1) + dbscan_config.get('eps_max', 5)) / 2
    
    clusterer = DBSCAN(eps=best_eps, min_samples=min_samples, metric='cosine')
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
    cluster_strategy = config.get('strategy', 'all')
    
    # Get unique valid cluster labels (excluding noise points with label -1)
    unique_labels = [label for label in np.unique(labels) if label != -1]
    
    # Determine which clusters to use based on strategy
    if cluster_strategy != 'all':
        # Score all clusters
        cluster_scores = score_clusters(acts, labels, config)
        
        # Sort clusters by score (descending)
        sorted_clusters = sorted(cluster_scores.items(), key=lambda x: x[1], reverse=True)
        
        if cluster_strategy == 'single':
            # Use only the highest scoring cluster
            if sorted_clusters:
                unique_labels = [sorted_clusters[0][0]]
                print(f"Using single highest-scoring cluster: {unique_labels[0]} (score: {sorted_clusters[0][1]:.4f})")
            else:
                unique_labels = []
                print("No valid clusters found for selection")
        
        elif cluster_strategy == 'top_n':
            # Use top N highest scoring clusters
            num_clusters = min(config.get('num_clusters', 3), len(sorted_clusters))
            unique_labels = [cluster[0] for cluster in sorted_clusters[:num_clusters]]
            print(f"Using top {num_clusters} clusters: {unique_labels}")
            
            # Print scores for selected clusters
            for cluster, score in sorted_clusters[:num_clusters]:
                print(f"Cluster {cluster}: score = {score:.4f}")
    
    # Select features from the chosen clusters
    target_features = []
    for label in unique_labels:
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

def score_clusters(acts, labels, config):
    """Score clusters based on their properties relevant to SAE features.
    
    Args:
        acts: Tensor of shape [n_prompts, n_filtered_features] containing feature activations
        labels: Numpy array of cluster labels
        config: Configuration dictionary with scoring parameters
        
    Returns:
        Dictionary mapping cluster labels to scores
    """
    cluster_scores = {}
    scoring_method = config.get('scoring_method', 'composite')
    
    for label in np.unique(labels):
        if label == -1:  # Skip noise
            continue
            
        cluster_mask = labels == label
        cluster_size = np.sum(cluster_mask)
        cluster_acts = acts[:, cluster_mask]
        
        if scoring_method == 'size':
            # Simple size-based scoring
            score = cluster_size
        
        elif scoring_method == 'activation':
            # Average activation magnitude
            score = torch.mean(torch.abs(cluster_acts)).item()
            
        elif scoring_method == 'max_activation':
            # Maximum activation in cluster
            score = torch.max(torch.abs(cluster_acts)).item()
            
        elif scoring_method == 'sparsity':
            # How specific/sparse are the activations
            threshold = config.get('filtering', {}).get('activation_threshold', 0.1)
            sparsity = (torch.abs(cluster_acts) > threshold).float().mean().item()
            score = sparsity * (1 - sparsity)  # Highest for medium sparsity
            
        elif scoring_method == 'composite':
            # Combine multiple metrics
            avg_act = torch.mean(torch.abs(cluster_acts)).item()
            max_act = torch.max(torch.abs(cluster_acts)).item()
            threshold = config.get('filtering', {}).get('activation_threshold', 0.1)
            sparsity = (torch.abs(cluster_acts) > threshold).float().mean().item()
            sparsity_score = sparsity * (1 - sparsity)
            
            # Weighted combination
            score = (0.4 * avg_act + 0.3 * max_act + 0.3 * sparsity_score) * np.log1p(cluster_size)
        
        else:
            print(f"Unknown scoring method '{scoring_method}', defaulting to size")
            score = cluster_size
        
        cluster_scores[label] = score
    
    return cluster_scores

def visualize_cluster_prompt_heatmap(cluster_prompt_matrix, unique_labels, prompts, max_prompts=50):
    """Visualize clusters vs prompts activation as a heatmap.
    
    Args:
        cluster_prompt_matrix: Numpy array of shape [n_clusters, n_prompts] with activation values
        unique_labels: List of cluster labels corresponding to rows in the matrix
        prompts: List of original prompts
        max_prompts: Maximum number of prompts to show (to avoid overcrowding)
        
    Returns:
        None (displays a plotly figure)
    """
    # Limit the number of prompts to avoid overcrowding
    n_prompts = min(max_prompts, cluster_prompt_matrix.shape[1])
    
    # If we have too many prompts, select a subset uniformly
    if len(prompts) > n_prompts:
        indices = np.linspace(0, len(prompts)-1, n_prompts, dtype=int)
        selected_prompts = [prompts[i] for i in indices]
        selected_matrix = cluster_prompt_matrix[:, indices]
    else:
        selected_prompts = prompts
        selected_matrix = cluster_prompt_matrix
    
    # Truncate prompts for better display
    truncated_prompts = [p[:30] + "..." if len(p) > 30 else p for p in selected_prompts]
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=selected_matrix,
        x=truncated_prompts,
        y=[f"Cluster {label}" for label in unique_labels],
        colorscale='Viridis',
        colorbar=dict(title="Activation")
    ))
    
    fig.update_layout(
        title="Cluster vs Prompt Activation Heatmap",
        xaxis=dict(
            title="Prompts",
            tickangle=45,
        ),
        yaxis=dict(
            title="Clusters",
        ),
        height=max(400, 100 + 20 * len(unique_labels)),  # Dynamic height based on cluster count
        width=max(800, 100 + 10 * n_prompts),  # Dynamic width based on prompt count
    )
    
    fig.show()

def explore_cluster_activations(acts, labels, original_indices, prompts, config):
    """Explore which prompts most strongly activate each cluster.
    
    Args:
        acts: Tensor of shape [n_prompts, n_filtered_features] containing filtered feature activations
        labels: Numpy array of cluster labels
        original_indices: Tensor containing the original indices of the filtered features
        prompts: List of text prompts used to generate the activations
        config: Configuration dictionary with parameters
        
    Returns:
        Dictionary with cluster analysis results
    """
    if len(prompts) != acts.shape[0]:
        print(f"Warning: Number of prompts ({len(prompts)}) doesn't match activation batch size ({acts.shape[0]})")
        print("Cannot perform prompt-based cluster analysis")
        return {}
    
    # Track results for each cluster
    cluster_analysis = {}
    
    # Get unique valid cluster labels (excluding noise points with label -1)
    unique_labels = [label for label in np.unique(labels) if label != -1]
    print(f"\n=== Cluster Activation Analysis ===")
    print(f"Found {len(unique_labels)} clusters to analyze")
    
    # For each cluster
    for label in unique_labels:
        # Get features in this cluster
        cluster_mask = labels == label
        cluster_size = np.sum(cluster_mask)
        cluster_indices = original_indices[cluster_mask]
        
        # Get activations for all prompts on this cluster's features
        cluster_acts = acts[:, cluster_mask]  # [n_prompts, n_cluster_features]
        
        # Compute average activation of each prompt on this cluster's features
        prompt_activations = torch.mean(torch.abs(cluster_acts), dim=1)  # [n_prompts]
        
        # Find top activating prompts
        top_k = min(5, len(prompts))  # Show top 5 prompts or all if fewer
        top_prompt_indices = torch.argsort(prompt_activations, descending=True)[:top_k]
        
        # Format output with truncated prompts (first 50 chars)
        top_prompts = [(prompts[i], prompt_activations[i].item()) for i in top_prompt_indices]
        
        # Store results
        cluster_analysis[label] = {
            'size': cluster_size,
            'indices': cluster_indices.tolist(),
            'top_prompts': top_prompts,
            'mean_activation': torch.mean(torch.abs(cluster_acts)).item(),
            'prompt_activations': prompt_activations.cpu().numpy()  # Store all prompt activations
        }
        
        # Print results
        print(f"\nCluster {label} ({cluster_size} features):")
        print(f"  Feature indices: {cluster_indices.tolist()[:5]}{'...' if cluster_size > 5 else ''}")
        print(f"  Mean activation: {cluster_analysis[label]['mean_activation']:.4f}")
        print(f"  Top activating prompts:")
        for i, (prompt, act) in enumerate(top_prompts):
            truncated = prompt[:50] + ("..." if len(prompt) > 50 else "")
            print(f"    {i+1}. \"{truncated}\" (activation: {act:.4f})")
    
    # Find clusters with similar prompt activation patterns
    print("\n=== Clusters with Similar Activation Patterns ===")
    if len(unique_labels) > 1:  # Only if we have multiple clusters
        # Create matrix of [cluster, prompt_activation]
        cluster_prompt_matrix = np.zeros((len(unique_labels), len(prompts)))
        for i, label in enumerate(unique_labels):
            cluster_mask = labels == label
            cluster_acts = acts[:, cluster_mask]
            cluster_prompt_matrix[i] = torch.mean(torch.abs(cluster_acts), dim=1).cpu().numpy()
        
        # Create the heatmap visualization
        if config.get('visualize_cluster_heatmap', True):
            print("\nGenerating cluster vs prompt activation heatmap...")
            visualize_cluster_prompt_heatmap(cluster_prompt_matrix, unique_labels, prompts)
        
        # Normalize each cluster's activations
        normalized_matrix = cluster_prompt_matrix / (np.linalg.norm(cluster_prompt_matrix, axis=1, keepdims=True) + 1e-10)
        
        # Compute correlation matrix between clusters based on prompt activations
        correlation_matrix = np.corrcoef(normalized_matrix)
        
        # Find most similar pairs
        np.fill_diagonal(correlation_matrix, 0)  # Zero out self-correlations
        for i, label_i in enumerate(unique_labels):
            # Get top 2 most similar clusters
            similar_indices = np.argsort(correlation_matrix[i])[-2:][::-1]
            if len(similar_indices) > 0 and correlation_matrix[i, similar_indices[0]] > 0.5:  # Only show if correlation > 0.5
                similar_labels = [unique_labels[j] for j in similar_indices]
                similarities = [correlation_matrix[i, j] for j in similar_indices]
                print(f"Cluster {label_i} is similar to: " + ", ".join([f"Cluster {l} (corr: {s:.2f})" for l, s in zip(similar_labels, similarities)]))
    
    return cluster_analysis 
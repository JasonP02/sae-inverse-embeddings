import torch
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import umap.umap_ as umap
import plotly.graph_objects as go
from einops import rearrange
import os
import datetime
import wandb
import hdbscan

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
    
    # Calculate entropy and sparsity
    activations = acts.abs()
    probs = activations / (activations.sum(dim=0) + 1e-10)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=0)
    
    filtering_config = config.get('filtering', {})
    activation_threshold = filtering_config.get('activation_threshold', 0.1)
    sparsity = (acts.abs() > activation_threshold).float().mean(dim=0)
    
    # Apply filtering mask
    mask = (entropy > filtering_config.get('entropy_threshold_low', 0.25)) & \
           (entropy < filtering_config.get('entropy_threshold_high', 5.0)) & \
           (sparsity > filtering_config.get('sparsity_min', 0.1)) & \
           (sparsity < filtering_config.get('sparsity_max', 0.95))
    
    print(f"Kept {mask.sum().item()} out of {n_features} features")
    return acts[:, mask], mask.nonzero(as_tuple=True)[0]

def apply_umap_preprocessing(acts, preprocessing_config):
    """Apply UMAP dimensionality reduction if beneficial."""
    feature_acts = acts.T.cpu().numpy()
    normalized_acts = feature_acts / (np.linalg.norm(feature_acts, axis=1, keepdims=True) + 1e-10)
    
    n_samples, n_dims = normalized_acts.shape
    min_dims_for_reduction = 50  # Only reduce if we have more than 50 dimensions
    
    if n_dims <= min_dims_for_reduction:
        print(f"Skipping UMAP reduction - input dimensionality ({n_dims}) is already manageable")
        return normalized_acts, normalized_acts
        
    # For high-dimensional data, apply UMAP reduction
    umap_config = preprocessing_config.get('umap', {})
    target_dims = min(umap_config.get('n_components', 50), n_samples - 1)
    
    reducer = umap.UMAP(
        n_components=target_dims,
        n_neighbors=min(umap_config.get('n_neighbors', 15), n_samples - 1),
        min_dist=umap_config.get('min_dist', 0.1),
        metric=umap_config.get('metric', 'cosine'),
        random_state=42
    )
    
    try:
        reduced_acts = reducer.fit_transform(normalized_acts)
        print(f"Applied UMAP reduction: {n_dims}d → {target_dims}d")
        return reduced_acts, normalized_acts
    except Exception as e:
        print(f"Error during UMAP reduction: {e}")
        return normalized_acts, normalized_acts

def cluster_features(acts, clustering_config):
    """Cluster features using UMAP preprocessing (if needed) and HDBSCAN."""
    if acts.shape[1] <= 1:
        return np.zeros(acts.shape[1], dtype=int), acts.T.cpu().numpy()
    
    # Preprocess with UMAP if enabled and beneficial
    if clustering_config.get('use_umap_preprocessing', True):
        reduced_acts, normalized_acts = apply_umap_preprocessing(acts, clustering_config)
    else:
        feature_acts = acts.T.cpu().numpy()
        normalized_acts = feature_acts / (np.linalg.norm(feature_acts, axis=1, keepdims=True) + 1e-10)
        reduced_acts = normalized_acts
    
    # Run HDBSCAN clustering
    labels = run_hdbscan(reduced_acts, clustering_config)
    
    n_clusters = len(np.unique(labels)) - (1 if -1 in labels else 0)
    noise_ratio = np.mean(labels == -1) if -1 in labels else 0
    print(f"Clustering complete: found {n_clusters} clusters with {noise_ratio:.2%} noise points")
    
    return labels, normalized_acts

def run_hdbscan(reduced_acts, clustering_config):
    """Run HDBSCAN clustering."""
    hdbscan_config = clustering_config.get('hdbscan', {})
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=hdbscan_config.get('min_cluster_size', 5),
        min_samples=hdbscan_config.get('min_samples', 1),
        metric=hdbscan_config.get('metric', 'euclidean'),
        cluster_selection_epsilon=hdbscan_config.get('cluster_selection_epsilon', 0.0)
    )
    return clusterer.fit_predict(reduced_acts)

def visualize_clusters(reduced_acts, labels, config=None):
    """Visualize clusters using UMAP."""
    try:
        # If reduced_acts is already 2D, use it directly, otherwise reduce to 2D
        if reduced_acts.shape[1] > 2:
            n_samples = reduced_acts.shape[0]
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=min(15, n_samples - 1),
                random_state=42
            )
            umap_acts = reducer.fit_transform(reduced_acts)
        else:
            umap_acts = reduced_acts
        
        # Create color mapping with noise points in gray
        colors = labels.copy()
        noise_mask = labels == -1
        
        # Create a colorscale where noise points are gray
        colorscale = [[0, 'rgba(128, 128, 128, 0.5)']]  # Start with gray for noise (-1)
        
        # Get unique non-noise labels
        unique_labels = sorted(list(set(labels) - {-1}))
        if unique_labels:
            # Add colors for actual clusters
            for i, label in enumerate(unique_labels):
                pos = (i + 1) / (len(unique_labels) + 1)  # Distribute colors evenly
                colorscale.append([pos, f'hsl({360 * pos}, 100%, 50%)'])
            colorscale.append([1.0, 'hsl(360, 100%, 50%)'])
            
            # Remap labels for colorscale (noise: 0, clusters: 1 to n)
            colors = np.zeros_like(labels, dtype=float)
            colors[noise_mask] = 0  # Noise maps to 0
            for i, label in enumerate(unique_labels):
                colors[labels == label] = i + 1
            colors = colors / (len(unique_labels) + 1)
        
        # Create scatter plot
        fig = go.Figure(data=[go.Scatter(
            x=umap_acts[:, 0], 
            y=umap_acts[:, 1], 
            mode='markers',
            marker=dict(
                color=colors,
                colorscale=colorscale,
                size=8,
                opacity=0.7
            ),
            text=[f"Cluster: {l}" for l in labels]
        )])
        
        # Count points per cluster
        cluster_counts = {label: np.sum(labels == label) for label in np.unique(labels)}
        title = f"Cluster Visualization | {len(cluster_counts) - (1 if -1 in cluster_counts else 0)} clusters"
        if -1 in cluster_counts:
            title += f" | {cluster_counts[-1]} noise points ({100 * cluster_counts[-1] / len(labels):.1f}%)"
            
        fig.update_layout(
            title=title,
            xaxis_title="UMAP 1", 
            yaxis_title="UMAP 2",
            width=800,
            height=600
        )
        
        # Try to log to wandb if available and initialized
        try:
            if wandb.run is not None:
                wandb.log({"cluster_visualization": fig})
        except:
            pass
        
        fig.show()
        
    except Exception as e:
        print(f"Error visualizing clusters: {e}")

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
    """Select top features from clusters."""
    top_n = config.get('features_per_cluster', 1)
    cluster_strategy = config.get('strategy', 'all')
    unique_labels = [label for label in np.unique(labels) if label != -1]
    
    # Score and select clusters if needed
    if cluster_strategy != 'all':
        cluster_scores = score_clusters(acts, labels, config)
        sorted_clusters = sorted(cluster_scores.items(), key=lambda x: x[1], reverse=True)
        
        if cluster_strategy == 'single':
            unique_labels = [sorted_clusters[0][0]] if sorted_clusters else []
        elif cluster_strategy == 'top_n':
            num_clusters = min(config.get('num_clusters', 3), len(sorted_clusters))
            unique_labels = [cluster[0] for cluster in sorted_clusters[:num_clusters]]
    
    # Select features from chosen clusters
    target_features = []
    for label in unique_labels:
        cluster_mask = labels == label
        cluster_acts = acts[:, cluster_mask]
        cluster_indices = original_indices[cluster_mask]
        
        feature_scores = score_features(cluster_acts, config)
        if feature_scores.numel() > 0:
            sorted_indices = torch.argsort(feature_scores, descending=True)[:min(top_n, feature_scores.numel())]
            target_features.extend(cluster_indices[sorted_indices.cpu()].tolist())
    
    print(f"Total target features: {len(target_features)}")
    return target_features

def score_clusters(acts, labels, config):
    """
    Score clusters based on their properties relevant to SAE features.
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
        None
    """
    try:
        import wandb
        
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
        
        # Log to wandb
        wandb.log({"cluster_prompt_heatmap": fig})
        
    except Exception as e:
        print(f"Error creating heatmap visualization: {e}")
        print("Continuing without heatmap visualization...")

def explore_cluster_activations(acts, labels, original_indices, prompts, config):
    """Explore which prompts most strongly activate each cluster.
    Provides detailed statistics about each cluster and its features.
    """
    if len(prompts) != acts.shape[0]:
        print(f"Warning: Number of prompts ({len(prompts)}) doesn't match activation batch size ({acts.shape[0]})")
        print("Cannot perform prompt-based cluster analysis")
        return {}
    
    # Track results for each cluster
    cluster_analysis = {}
    
    # Get unique valid cluster labels (including noise points)
    all_labels = np.unique(labels)
    unique_labels = [label for label in all_labels if label != -1]
    
    # Calculate global statistics
    total_features = len(labels)
    noise_points = np.sum(labels == -1)
    noise_ratio = noise_points / total_features if total_features > 0 else 0
    
    print(f"\n=== Cluster Analysis Summary ===")
    print(f"Total Features: {total_features}")
    print(f"Number of Clusters: {len(unique_labels)}")
    print(f"Noise Points: {noise_points} ({noise_ratio:.2%})")
    
    # For each cluster (including noise)
    for label in all_labels:
        cluster_type = "Noise Cluster" if label == -1 else f"Cluster {label}"
        cluster_mask = labels == label
        cluster_size = np.sum(cluster_mask)
        cluster_indices = original_indices[cluster_mask]
        
        print(f"\n=== {cluster_type} ===")
        print(f"Size: {cluster_size} features ({(cluster_size/total_features):.2%} of total)")
        print(f"Feature Indices: {cluster_indices.tolist()}")
        
        if label == -1:  # Skip activation analysis for noise cluster
            cluster_analysis[label] = {
                'size': cluster_size,
                'indices': cluster_indices.tolist(),
                'ratio': cluster_size/total_features,
                'is_noise': True
            }
            continue
            
        # Get activations for all prompts on this cluster's features
        cluster_acts = acts[:, cluster_mask]  # [n_prompts, n_cluster_features]
        
        # Compute activation statistics
        mean_activation = torch.mean(torch.abs(cluster_acts)).item()
        max_activation = torch.max(torch.abs(cluster_acts)).item()
        std_activation = torch.std(torch.abs(cluster_acts)).item()
        sparsity = (torch.abs(cluster_acts) > 0.1).float().mean().item()
        
        print(f"\nActivation Statistics:")
        print(f"  Mean Activation: {mean_activation:.4f}")
        print(f"  Max Activation: {max_activation:.4f}")
        print(f"  Std Deviation: {std_activation:.4f}")
        print(f"  Sparsity: {sparsity:.4f}")
        
        # Compute average activation of each prompt on this cluster's features
        prompt_activations = torch.mean(torch.abs(cluster_acts), dim=1)  # [n_prompts]
        
        # Find top activating prompts
        top_k = min(5, len(prompts))
        top_prompt_indices = torch.argsort(prompt_activations, descending=True)[:top_k]
        top_prompts = [(prompts[i], prompt_activations[i].item()) for i in top_prompt_indices]
        
        print(f"\nTop {top_k} Activating Prompts:")
        for i, (prompt, act) in enumerate(top_prompts, 1):
            # Show first 100 chars, with special formatting for section headers
            truncated = prompt[:100]
            if len(prompt) > 100:
                truncated += "..."
            
            # Format section headers more clearly
            if "=" in truncated:
                sections = [s.strip() for s in truncated.split("=") if s.strip()]
                if sections:
                    truncated = f"[SECTION] {' > '.join(sections)}"
            
            # Split into tokens if the prompt contains spaces
            tokens = truncated.split()
            if len(tokens) > 15:
                token_display = " ".join(tokens[:15]) + " ..."
            else:
                token_display = truncated
                
            print(f"  {i}. \"{token_display}\"")
            print(f"     Activation: {act:.4f}")
            print(f"     Total Length: {len(prompt)} chars, {len(prompt.split())} tokens")
            print()
        
        # Store detailed results
        cluster_analysis[label] = {
            'size': cluster_size,
            'indices': cluster_indices.tolist(),
            'ratio': cluster_size/total_features,
            'is_noise': False,
            'mean_activation': mean_activation,
            'max_activation': max_activation,
            'std_activation': std_activation,
            'sparsity': sparsity,
            'top_prompts': top_prompts,
            'prompt_activations': prompt_activations.cpu().numpy()
        }
    
    # Print cluster similarity analysis
    if len(unique_labels) > 1:
        print("\n=== Cluster Similarity Analysis ===")
        for i, label1 in enumerate(unique_labels):
            for label2 in unique_labels[i+1:]:
                acts1 = cluster_analysis[label1]['prompt_activations']
                acts2 = cluster_analysis[label2]['prompt_activations']
                correlation = np.corrcoef(acts1, acts2)[0, 1]
                if abs(correlation) > 0.5:  # Only show significant correlations
                    print(f"Clusters {label1} and {label2}: correlation = {correlation:.4f}")
    
    return cluster_analysis 
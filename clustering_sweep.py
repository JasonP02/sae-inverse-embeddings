import wandb
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import torch
from collections import Counter
from scipy.stats import entropy
import numpy as np
from einops import rearrange

# Import necessary functions from pipeline module
from pipeline import (
    load_experiment_models,
    collect_and_filter_data,
    run_clustering
)
from clustering import cluster_features
from sweep_analysis import run_clustering_sweep

def calculate_cluster_metrics(acts, labels, normalized_acts, prompts=None):
    """Calculate comprehensive clustering metrics.
    
    Args:
        acts: Original activations [n_prompts, n_features]
        labels: Cluster labels
        normalized_acts: Normalized activations used for clustering
        prompts: Optional list of text prompts for prompt-coherence scoring
        
    Returns:
        Dictionary of metrics
    """
    # Skip metrics if only noise points
    if len(np.unique(labels)) <= 1:
        return {
            'n_clusters': 0,
            'noise_ratio': 1.0,
            'silhouette': -1,
            'davies_bouldin': -1,
            'calinski_harabasz': -1,
            'mean_cluster_size': 0,
            'cluster_size_std': 0,
            'cluster_activation_entropy': 0,
            'prompt_coherence': -1,
            'sparsity_preservation': -1
        }
    
    # Basic clustering metrics
    valid_mask = labels != -1
    valid_acts = normalized_acts[valid_mask]
    valid_labels = labels[valid_mask]
    
    # Only calculate if we have valid clusters
    if len(np.unique(valid_labels)) > 1 and len(valid_labels) > 1:
        try:
            silhouette = silhouette_score(valid_acts, valid_labels, metric='cosine')
            davies_bouldin = davies_bouldin_score(valid_acts, valid_labels)
            calinski_harabasz = calinski_harabasz_score(valid_acts, valid_labels)
        except:
            silhouette = -1
            davies_bouldin = -1
            calinski_harabasz = -1
    else:
        silhouette = -1
        davies_bouldin = -1
        calinski_harabasz = -1
    
    # Cluster size statistics
    cluster_sizes = Counter(labels[labels != -1])
    mean_size = np.mean(list(cluster_sizes.values())) if cluster_sizes else 0
    size_std = np.std(list(cluster_sizes.values())) if cluster_sizes else 0
    
    # Calculate activation patterns per cluster
    cluster_activations = []
    for label in np.unique(labels):
        if label != -1:
            cluster_mask = labels == label
            cluster_acts = acts[:, cluster_mask]
            mean_activation = torch.mean(torch.abs(cluster_acts)).item()
            cluster_activations.append(mean_activation)
    
    # Calculate entropy of cluster activations
    if cluster_activations:
        cluster_acts_norm = np.array(cluster_activations) / np.sum(cluster_activations)
        activation_entropy = entropy(cluster_acts_norm)
    else:
        activation_entropy = 0
    
    # Calculate prompt coherence if prompts are provided
    prompt_coherence = -1
    if prompts and len(prompts) == acts.shape[0]:
        prompt_coherence = calculate_prompt_coherence(acts, labels, prompts)
    
    # Calculate sparsity preservation
    sparsity_preservation = calculate_sparsity_preservation(acts, labels)
        
    return {
        'n_clusters': len(np.unique(labels)) - (1 if -1 in labels else 0),
        'noise_ratio': np.mean(labels == -1),
        'silhouette': silhouette,
        'davies_bouldin': davies_bouldin,
        'calinski_harabasz': calinski_harabasz,
        'mean_cluster_size': mean_size,
        'cluster_size_std': size_std,
        'cluster_activation_entropy': activation_entropy,
        'prompt_coherence': prompt_coherence,
        'sparsity_preservation': sparsity_preservation
    }

def calculate_prompt_coherence(acts, labels, prompts):
    """Calculate how well clusters correspond to semantically similar prompts.
    
    This is a heuristic metric for estimating if clusters capture semantically meaningful features
    by measuring the average activation similarity within semantic prompt groups.
    
    Args:
        acts: Activations [n_prompts, n_features]
        labels: Cluster labels
        prompts: List of text prompts corresponding to activations
        
    Returns:
        Coherence score between 0 and 1 (higher is better)
    """
    try:
        # Simple heuristic: estimate semantic similarity by common prefix words
        prompt_groups = {}
        
        # Group prompts by first 3 words
        for i, prompt in enumerate(prompts):
            # Get the first few words as a key
            words = prompt.strip().split()
            prefix = " ".join(words[:min(3, len(words))])
            
            if prefix not in prompt_groups:
                prompt_groups[prefix] = []
            prompt_groups[prefix].append(i)
        
        # Only consider groups with multiple prompts
        valid_groups = {k: v for k, v in prompt_groups.items() if len(v) > 1}
        
        if not valid_groups:
            return 0  # No valid groups found
        
        # Calculate cluster coherence across prompt groups
        coherence_scores = []
        
        for group_indices in valid_groups.values():
            # Get clusters for this group
            group_labels = [labels[i] for i in group_indices]
            
            # Count cluster occurrences
            cluster_counts = Counter(l for l in group_labels if l != -1)
            
            # Calculate coherence as ratio of most common cluster to total valid points
            total_valid = sum(1 for l in group_labels if l != -1)
            most_common_count = cluster_counts.most_common(1)[0][1] if cluster_counts else 0
            
            # Coherence for this group
            group_coherence = most_common_count / total_valid if total_valid > 0 else 0
            coherence_scores.append(group_coherence)
        
        # Average coherence across all groups
        return np.mean(coherence_scores)
    except Exception as e:
        print(f"Error calculating prompt coherence: {e}")
        return 0

def calculate_sparsity_preservation(acts, labels):
    """Calculate how well the clustering preserves sparsity patterns.
    
    Args:
        acts: Activations [n_prompts, n_features]
        labels: Cluster labels
        
    Returns:
        Sparsity preservation score between 0 and 1 (higher is better)
    """
    try:
        # Skip if no valid clusters
        if -1 in labels and np.all(labels == -1):
            return 0
        
        # Calculate original feature sparsity patterns
        activation_threshold = 0.1  # Same as default in filter_features
        original_sparsity = torch.mean((torch.abs(acts) > activation_threshold).float(), dim=0)
        
        # Calculate sparsity patterns per cluster
        cluster_sparsity_match = []
        
        for label in np.unique(labels):
            if label == -1:  # Skip noise
                continue
                
            cluster_mask = labels == label
            
            # Skip if cluster is too small
            if np.sum(cluster_mask) < 2:
                continue
                
            # Calculate mean sparsity for features in this cluster
            cluster_features = acts[:, cluster_mask]
            cluster_sparsity = torch.mean((torch.abs(cluster_features) > activation_threshold).float(), dim=0)
            
            # Calculate std of sparsity within cluster
            # Lower std means more similar sparsity patterns within cluster
            sparsity_std = torch.std(cluster_sparsity).item()
            
            # Convert to a score where 0 is bad (high std) and 1 is good (low std)
            # Using a negative exponential transform
            sparsity_score = np.exp(-5 * sparsity_std)
            cluster_sparsity_match.append(sparsity_score)
            
        # Overall score is average across clusters
        return np.mean(cluster_sparsity_match) if cluster_sparsity_match else 0
    except Exception as e:
        print(f"Error calculating sparsity preservation: {e}")
        return 0

def run_clustering_sweep(config=None):
    """Run a single clustering experiment with given config.
    
    Args:
        config: W&B config object with parameters to try
    """
    # Initialize W&B
    with wandb.init(config=config):
        config = wandb.config
        
        try:
            # Load models
            print(f"Loading models with config: {config.clustering_method}")
            model, sae, _, _ = load_experiment_models({
                'model': {
                    'transformer': "EleutherAI/pythia-70m-deduped",
                    'sae_release': "pythia-70m-deduped-mlp-sm",
                    'sae_hook': "blocks.3.hook_mlp_out"
                },
                'hardware': {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
            })
            
            # Collect data with current config
            print(f"Collecting data with n_prompts={config.n_prompts}")
            data_config = {
                'n_prompts': config.n_prompts,
                'min_prompt_length': config.min_prompt_length,
                'max_prompt_length': config.max_prompt_length,
                'filtering': {
                    'entropy_threshold_low': config.entropy_threshold_low,
                    'entropy_threshold_high': config.entropy_threshold_high,
                    'sparsity_min': config.sparsity_min,
                    'sparsity_max': config.sparsity_max,
                    'activation_threshold': config.activation_threshold
                }
            }
            
            filtered_acts, original_indices, prompts = collect_and_filter_data(
                model, sae, data_config, use_cached=False
            )
            
            # Run clustering with current config
            print(f"Running clustering with method={config.clustering_method}, UMAP={config.use_umap_preprocessing}")
            clustering_config = {
                'method': config.clustering_method,
                'use_umap_preprocessing': config.use_umap_preprocessing,
                'umap': {
                    'n_components': config.umap_n_components,
                    'n_neighbors': config.umap_n_neighbors,
                    'min_dist': config.umap_min_dist,
                    'metric': config.umap_metric
                },
                'hdbscan': {
                    'min_cluster_size': config.hdbscan_min_cluster_size,
                    'min_samples': config.hdbscan_min_samples,
                    'metric': config.hdbscan_metric,
                    'cluster_selection_epsilon': config.hdbscan_cluster_selection_epsilon
                },
                'dbscan': {
                    'eps_min': config.eps_min,
                    'eps_max': config.eps_max,
                    'eps_steps': config.eps_steps,
                    'min_samples': config.dbscan_min_samples
                }
            }
            
            # Either use pipeline function or direct function for flexibility
            if config.get('use_pipeline', True):
                labels, normalized_acts = run_clustering(
                    filtered_acts, clustering_config
                )
            else:
                labels, normalized_acts = cluster_features(filtered_acts, clustering_config)
            
            # Calculate and log metrics
            print("Calculating metrics")
            metrics = calculate_cluster_metrics(filtered_acts, labels, normalized_acts, prompts)
            wandb.log(metrics)
            
            print(f"Run complete: found {metrics['n_clusters']} clusters, silhouette={metrics['silhouette']:.4f}")
            
        except Exception as e:
            import traceback
            print(f"Error in sweep run: {e}")
            print(traceback.format_exc())
            # Log error to wandb
            wandb.log({
                'error': str(e),
                'n_clusters': 0,
                'silhouette': -1
            })

def main():
    # Define sweep configuration
    sweep_config = {
        'method': 'bayes',  # Can also use 'random' or 'grid'
        'metric': {'name': 'silhouette', 'goal': 'maximize'},
        'parameters': {
            # Run control
            'use_pipeline': {'values': [True]},  # Whether to use pipeline function
            
            # Data collection parameters (logarithmic scale)
            'n_prompts': {'values': [10, 30, 100, 300, 1000]},
            'min_prompt_length': {'values': [10, 20, 30]},
            'max_prompt_length': {'values': [50, 100, 150]},
            
            # Feature filtering parameters
            'entropy_threshold_low': {'min': 0.1, 'max': 0.5},
            'entropy_threshold_high': {'min': 3.0, 'max': 7.0},
            'sparsity_min': {'min': 0.05, 'max': 0.2},
            'sparsity_max': {'min': 0.8, 'max': 0.99},
            'activation_threshold': {'min': 0.05, 'max': 0.2},
            
            # Clustering method selection
            'clustering_method': {'values': ['hdbscan']},
            
            # UMAP parameters
            'use_umap_preprocessing': {'values': [True, False]},
            'umap_n_components': {'values': [25, 50, 100]},
            'umap_n_neighbors': {'values': [5, 15, 30]},
            'umap_min_dist': {'values': [0.0, 0.1, 0.5]},
            'umap_metric': {'values': ['cosine', 'euclidean']},
            
            # HDBSCAN parameters
            'hdbscan_min_cluster_size': {'values': [2, 5, 10, 20]},
            'hdbscan_min_samples': {'values': [1, 2, 5]},
            'hdbscan_metric': {'values': ['euclidean', 'manhattan']},
            'hdbscan_cluster_selection_epsilon': {'min': 0.0, 'max': 0.5},
        
        }
    }
    
    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project="sae-clustering-sweep")
    
    # Run the sweep
    wandb.agent(sweep_id, function=run_clustering_sweep, count=50)  # Adjust count as needed


if __name__ == "__main__":
    main() 
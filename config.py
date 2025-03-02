import torch

def get_default_config():
    """Return default configuration settings."""
    return {
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

def update_config(base_config, updates):
    """Update configuration with new values."""
    config = base_config.copy()
    config.update(updates)
    return config 
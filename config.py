import torch

def get_default_config():
    """Return default configuration settings in a hierarchical structure."""
    return {
        # Pipeline control - enable/disable entire sections
        'pipeline': {
            'run_data_collection': True,      # Collect and filter data
            'run_clustering': True,           # Run clustering analysis
            'run_feature_selection': True,    # Select features from clusters
            'run_prompt_optimization': True,  # Optimize prompts for selected features
            'run_explanations': True,         # Generate explanations for features
        },
        
        # Hardware settings
        'hardware': {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        },
        
        # Model settings
        'model': {
            'transformer': "EleutherAI/pythia-70m-deduped",
            'sae_release': "pythia-70m-deduped-mlp-sm",
            'sae_hook': "blocks.3.hook_mlp_out",
            'lm': "distilgpt2"  # For coherence scoring
        },
        
        # Data collection & processing settings
        'data': {
            # Collection parameters
            'n_prompts': 100,
            'min_prompt_length': 10,
            'max_prompt_length': 100,
            'batch_size': 10,  # Batch size for processing prompts
            
            # Caching parameters
            'cache_data': True,
            'cache_dir': 'feature_cache',
            'use_cached_data': False,
            
            # Feature filtering parameters
            'filtering': {
                'entropy_threshold_low': 0.25,   # Minimum entropy for feature selection
                'entropy_threshold_high': 5.0,   # Maximum entropy for feature selection
                'sparsity_min': 0.1,             # Minimum activation sparsity
                'sparsity_max': 0.95,            # Maximum activation sparsity
                'activation_threshold': 0.1,     # Threshold for considering a feature activated
            },
        },
        
        # Clustering parameters
        'clustering': {
            # General clustering settings
            'method': 'dbscan',              # 'dbscan' or 'kmeans'
            'visualize_clusters': True,      # Visualize with UMAP
            'n_clusters': 10,                # For kmeans or as target for DBSCAN
            'pca_multiplier': 3,             # n_components = pca_multiplier * n_clusters
            
            # Cluster analysis
            'explore_clusters': True,                # Enable cluster activation analysis
            'visualize_cluster_heatmap': True,       # Show heatmap of cluster vs prompt activations
            'max_prompts_heatmap': 50,              # Maximum prompts to show in heatmap
            
            # DBSCAN specific parameters
            'dbscan': {
                'eps_min': 0.1,              # Minimum epsilon for search
                'eps_max': 5,                # Maximum epsilon for search
                'eps_steps': 30,             # Number of steps in epsilon search
                'min_samples': 1,            # Minimum samples for core point
            },
            
            # Cluster selection for feature extraction
            'selection': {
                'strategy': 'all',           # Options: 'all', 'single', 'top_n'
                'num_clusters': 3,           # If strategy is 'top_n', how many to select
                'scoring_method': 'composite',  # How to score clusters: 'size', 'activation', 'max_activation', 'sparsity', 'composite'
                'features_per_cluster': 1,   # How many features to take from each cluster
                'feature_selection_method': 'max',  # How to select features: 'mean', 'max', 'percentile'
                'activation_percentile': 90,  # Percentile to use if method='percentile'
            },
        },
        
        # Prompt optimization parameters
        'optimization': {
            # Sequence parameters
            'length': 10,                    # Length of token sequence to optimize
            
            # Training parameters
            'max_steps': 250,
            'lr': 1e-3,
            'lambda_reg': 1e-5,
            
            # Regularization options
            'diversity_penalty': 0.0,        # Penalty for token similarity across positions
            'diversity_window': 5,
            'repetition_penalty': 0.0,       # Penalty for repeating tokens
            'repetition_window': 5,
            'noise_scale': 0.5,
            
            # Visualization
            'visualize_training': False,
            'show_full_sequence': True,
            'verbose': True,                 # Print optimization progress
        },
        
        # Explanation generation parameters
        'explanation': {
            'use_lm_coherence': True,        # Use language model for coherence
            'coherence_weight': 0.1,         # Weight of coherence loss
            'max_steps': 10,
            'lr': 1e-3,
            'lambda_reg': 1e-5,
            'length': 10,
            'optimize_explanations': True,    # Whether to optimize explanations
            'frozen_prefix_length': 4,        # Number of tokens to freeze at start
        },
        
        # Output settings
        'output': {
            'verbose': True,
            'save_results': False,
            'save_csv': True,
            'csv_output_dir': 'feature_results',
        },
    }

def update_config(base_config, updates):
    """Update configuration with new values, handling nested dictionaries properly."""
    if updates is None:
        return base_config
        
    result = base_config.copy()
    
    def _update_dict_recursive(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = _update_dict_recursive(d[k].copy(), v)
            else:
                d[k] = v
        return d
    
    return _update_dict_recursive(result, updates) 
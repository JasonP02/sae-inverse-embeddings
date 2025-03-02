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
            
            # Cluster analysis
            'explore_clusters': True,                # Enable cluster activation analysis
            'visualize_cluster_heatmap': True,       # Show heatmap of cluster vs prompt activations
            
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
                'scoring_method': 'composite',  # How to score clusters
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
        },
        
        # Explanation generation parameters
        'explanation': {
            'use_lm_coherence': True,
            'coherence_weight': 0.1,
            'max_steps': 1000,
            'lr': 1e-3,
            'lambda_reg': 1e-5,
            'length': 10,
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

def get_legacy_config(hierarchical_config):
    """Convert hierarchical config to flat legacy format for backward compatibility."""
    flat_config = {}
    
    # Extract hardware settings
    if 'hardware' in hierarchical_config:
        flat_config.update({
            'device': hierarchical_config['hardware'].get('device', 'cpu')
        })
    
    # Extract data settings
    if 'data' in hierarchical_config:
        data_config = hierarchical_config['data']
        flat_config.update({
            'n_prompts': data_config.get('n_prompts', 100),
            'min_prompt_length': data_config.get('min_prompt_length', 10),
            'max_prompt_length': data_config.get('max_prompt_length', 100),
            'batch_size': data_config.get('batch_size', 10),
            'cache_data': data_config.get('cache_data', True),
            'cache_dir': data_config.get('cache_dir', 'feature_cache'),
            'use_cached_data': data_config.get('use_cached_data', False)
        })
        
        # Extract filtering settings
        if 'filtering' in data_config:
            filtering = data_config['filtering']
            flat_config.update({
                'entropy_threshold_low': filtering.get('entropy_threshold_low', 0.25),
                'entropy_threshold_high': filtering.get('entropy_threshold_high', 5.0),
                'sparsity_min': filtering.get('sparsity_min', 0.1),
                'sparsity_max': filtering.get('sparsity_max', 0.95),
                'activation_threshold': filtering.get('activation_threshold', 0.1)
            })
    
    # Extract clustering settings
    if 'clustering' in hierarchical_config:
        clustering = hierarchical_config['clustering']
        flat_config.update({
            'clustering_method': clustering.get('method', 'dbscan'),
            'visualize_clusters': clustering.get('visualize_clusters', True),
            'explore_clusters': clustering.get('explore_clusters', True),
            'visualize_cluster_heatmap': clustering.get('visualize_cluster_heatmap', True)
        })
        
        # DBSCAN settings
        if 'dbscan' in clustering:
            dbscan = clustering['dbscan']
            flat_config.update({
                'dbscan_eps_min': dbscan.get('eps_min', 0.1),
                'dbscan_eps_max': dbscan.get('eps_max', 5),
                'dbscan_eps_steps': dbscan.get('eps_steps', 30),
                'dbscan_min_samples': dbscan.get('min_samples', 1)
            })
        
        # Selection settings
        if 'selection' in clustering:
            selection = clustering['selection']
            flat_config.update({
                'cluster_selection_strategy': selection.get('strategy', 'all'),
                'num_clusters_to_select': selection.get('num_clusters', 3),
                'cluster_scoring_method': selection.get('scoring_method', 'composite'),
                'features_per_cluster': selection.get('features_per_cluster', 1),
                'feature_selection_method': selection.get('feature_selection_method', 'max'),
                'activation_percentile': selection.get('activation_percentile', 90)
            })
    
    # Extract optimization settings
    if 'optimization' in hierarchical_config:
        optimization = hierarchical_config['optimization']
        flat_config.update({
            'length': optimization.get('length', 10),
            'max_steps': optimization.get('max_steps', 250),
            'lr': optimization.get('lr', 1e-3),
            'lambda_reg': optimization.get('lambda_reg', 1e-5),
            'diversity_penalty': optimization.get('diversity_penalty', 0.0),
            'diversity_window': optimization.get('diversity_window', 5),
            'repetition_penalty': optimization.get('repetition_penalty', 0.0),
            'repetition_window': optimization.get('repetition_window', 5),
            'noise_scale': optimization.get('noise_scale', 0.5),
            'visualize_training': optimization.get('visualize_training', False),
            'show_full_sequence': optimization.get('show_full_sequence', True)
        })
    
    # Extract explanation settings
    if 'explanation' in hierarchical_config:
        explanation = hierarchical_config['explanation']
        flat_config.update({
            'use_lm_coherence': explanation.get('use_lm_coherence', True),
            'coherence_weight': explanation.get('coherence_weight', 0.1),
            'max_steps_explanation': explanation.get('max_steps', 1000),
            'lr_explanation': explanation.get('lr', 1e-3),
            'lambda_reg_explanation': explanation.get('lambda_reg', 1e-5),
            'length_explanation': explanation.get('length', 10)
        })
    
    # Extract output settings
    if 'output' in hierarchical_config:
        output = hierarchical_config['output']
        flat_config.update({
            'verbose': output.get('verbose', True),
            'save_results': output.get('save_results', False),
            'save_csv': output.get('save_csv', True),
            'csv_output_dir': output.get('csv_output_dir', 'feature_results')
        })
    
    # Add pipeline control flags
    if 'pipeline' in hierarchical_config:
        pipeline = hierarchical_config['pipeline']
        flat_config.update({
            'run_data_collection': pipeline.get('run_data_collection', True),
            'run_clustering': pipeline.get('run_clustering', True),
            'run_feature_selection': pipeline.get('run_feature_selection', True),
            'run_prompt_optimization': pipeline.get('run_prompt_optimization', True),
            'run_explanations': pipeline.get('run_explanations', True)
        })
    
    return flat_config 
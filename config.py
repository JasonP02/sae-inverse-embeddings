import torch

class Config:
    """Config class that automatically updates when config.py changes."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._config = None
        return cls._instance
    
    @property
    def config(self):
        """Get the current config, refreshing from get_default_config."""
        self._config = get_default_config()
        return self._config
    
    def __getitem__(self, key):
        """Allow dictionary-style access to config."""
        return self.config[key]
    
    def get(self, key, default=None):
        """Mimic dict.get() functionality."""
        return self.config.get(key, default)
    
    def update(self, updates):
        """Update configuration with new values."""
        if updates is None:
            return
            
        def _update_dict_recursive(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    d[k] = _update_dict_recursive(d[k].copy(), v)
                else:
                    d[k] = v
            return d
        
        self._config = _update_dict_recursive(self.config.copy(), updates)
        return self._config

def get_default_config():
    """Return flattened configuration settings."""
    return {
        # Hardware
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # Model settings
        'transformer': "EleutherAI/pythia-70m-deduped",
        'sae_release': "pythia-70m-deduped-mlp-sm",
        'sae_hook': "blocks.3.hook_mlp_out",
        'lm': "distilgpt2",
        
        # Data collection
        'n_prompts': 1000,
        'min_prompt_length': 25,
        'max_prompt_length': 250,
        'batch_size': 1,
        
        # Caching
        'cache_data': True,
        'cache_dir': 'feature_cache',
        'use_cached_data': True,
        
        # Feature filtering
        'entropy_threshold_low': 0.0,
        'entropy_threshold_high': 15.0,
        'sparsity_min': 0.005,
        'sparsity_max': 1.0,
        'activation_threshold': 0.005,
        
        # Clustering
        'clustering_method': 'hdbscan',
        'visualize_features': True,
        'n_clusters': 10,
        
        # UMAP settings
        'use_umap': True,
        'umap_components': 50,
        'umap_neighbors': 15,
        'umap_min_dist': 0.1,
        'umap_metric': 'cosine',
        
        # HDBSCAN settings
        'hdbscan_min_cluster_size': 10,
        'hdbscan_min_samples': 10,
        'hdbscan_metric': 'precomputed',
        'hdbscan_cluster_selection_epsilon': 0.01,
        
        # Feature selection
        'selection_strategy': 'all',
        'num_clusters': 1,
        'scoring_method': 'composite',
        'features_per_cluster': 7,
        'feature_selection_method': 'max',
        'activation_percentile': 50,
        
        # Optimization
        'sequence_length': 10,
        'max_steps': 250,
        'learning_rate': 1e-3,
        'lambda_reg': 1e-5,
        'diversity_penalty': 0.0,
        'diversity_window': 5,
        'repetition_penalty': 0.0,
        'repetition_window': 5,
        'noise_scale': 0.5,
        
        # Explanation
        'use_lm_coherence': True,
        'coherence_weight': 0.1,
        'frozen_prefix_length': 4,
        
        # Output
        'verbose': False,
        'save_results': False,
        'save_csv': True,
        'csv_output_dir': 'feature_results',
    }

# Create a global config instance
config = Config()
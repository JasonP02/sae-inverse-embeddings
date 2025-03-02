"""SAE Inverse Embeddings.

This package explores the inverse problem of finding input embeddings that maximize 
the activation of specific sparse autoencoder (SAE) features in a transformer model.
"""

__version__ = "0.1.0"

from .models import load_models, get_feature_activations, get_similar_tokens
from .data import load_diverse_prompts, collect_activations
from .clustering import filter_features, cluster_features, select_target_features, visualize_clusters
from .optimization import optimize_embeddings, analyze_results
from .visualization import visualize_training, visualize_feature_activations
from .explanation import generate_explanatory_prompt, run_explanation_experiment
from .config import get_default_config, update_config
from .utils import save_results_to_csv, print_tensor_stats

__all__ = [
    # Models
    'load_models',
    'get_feature_activations',
    'get_similar_tokens',
    
    # Data
    'load_diverse_prompts',
    'collect_activations',
    
    # Clustering
    'filter_features',
    'cluster_features',
    'select_target_features',
    'visualize_clusters',
    
    # Optimization
    'optimize_embeddings',
    'analyze_results',
    
    # Visualization
    'visualize_training',
    'visualize_feature_activations',
    
    # Explanation
    'generate_explanatory_prompt',
    'run_explanation_experiment',
    
    # Config
    'get_default_config',
    'update_config',
    
    # Utils
    'save_results_to_csv',
    'print_tensor_stats',
] 
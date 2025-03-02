"""Example script demonstrating custom experiment configuration."""

import os
import sys
import torch

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_default_config, update_config
from main import run_experiment
from visualization import visualize_multiple_features, visualize_token_heatmap
from utils import save_results_to_csv

def main():
    """Run a custom experiment with modified configuration."""
    # Get default configuration
    config = get_default_config()
    
    # Update with custom values
    custom_config = {
        # Data collection
        'n_prompts': 50,  # Use fewer prompts for faster execution
        
        # Optimization
        'max_steps': 150,  # Fewer optimization steps
        'lr': 2e-3,  # Higher learning rate
        
        # Feature filtering
        'entropy_threshold_low': 0.3,
        'entropy_threshold_high': 4.0,
        'sparsity_min': 0.15,
        'sparsity_max': 0.9,
        
        # Clustering
        'features_per_cluster': 2,  # Select more features per cluster
        
        # Visualization
        'visualize_clusters': True,
        'visualize_training': True,
        
        # Caching
        'use_cached_data': True,  # Use cached data if available
        
        # Output
        'save_csv': True,
        'csv_output_dir': 'custom_results',
    }
    
    # Update config
    config = update_config(config, custom_config)
    
    print("Running custom experiment with the following configuration:")
    for key, value in custom_config.items():
        print(f"  {key}: {value}")
    
    # Run the experiment
    results = run_experiment(config)
    
    # Save results
    save_results_to_csv(results, config)
    
    # Visualize results
    if results['feature_results']:
        # Create a dictionary mapping feature_id to result
        feature_dict = {data['feature_id']: data for data in results['feature_results']}
        
        # Visualize multiple features
        visualize_multiple_features(feature_dict)
        
        # Visualize token heatmap for the first feature
        first_feature = results['feature_results'][0]['feature_id']
        visualize_token_heatmap(feature_dict, first_feature)
    
    return results

if __name__ == "__main__":
    main() 
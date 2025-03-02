# %% Imports and setup
%load_ext autoreload
%autoreload 2

import os
import sys
import torch
import numpy as np

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_default_config, update_config
from pipeline import (
    load_experiment_models,
    collect_and_filter_data,
    run_clustering,
    analyze_clusters,
    select_features,
    optimize_feature,
    generate_explanations,
    visualize_results
)
from utils import save_results_to_csv

"""Run a custom experiment with modified configuration."""

# %% Configuration
# Get default configuration
config = get_default_config()

# Update with custom values
custom_config = {
    # Data collection & processing settings
    'data': {
        # Collection parameters
        'n_prompts': 100,
        'min_prompt_length': 10,
        'max_prompt_length': 100,
        'batch_size': 10,  # Batch size for processing prompts
        
        # Caching parameters
        'cache_data': True,
        'use_cached_data': False,
        
        # Feature filtering parameters
        'filtering': {
            'entropy_threshold_low': 0.25,  # Minimum entropy for feature selection
            'entropy_threshold_high': 5.0,  # Maximum entropy for feature selection
            'sparsity_min': 0.1,            # Minimum activation sparsity
            'sparsity_max': 0.95,           # Maximum activation sparsity
            'activation_threshold': 0.1,    # Threshold for considering a feature activated
        },
    },
    
    # Clustering parameters
    'clustering': {
        # General clustering settings
        'method': 'dbscan',                  # 'dbscan' or 'kmeans'
        'visualize_clusters': True,          # Show UMAP visualization of clusters
        'explore_clusters': True,            # Enable cluster activation analysis 
        'visualize_cluster_heatmap': True,   # Enable heatmap visualization
        
        # DBSCAN specific parameters
        'dbscan': {
            'eps_min': 0.1,                  # Minimum epsilon for search
            'eps_max': 5,                    # Maximum epsilon for search
            'eps_steps': 30,                 # Number of steps for epsilon search
            'min_samples': 1,                # Minimum samples for core point
        },
        
        # Cluster selection for feature extraction
        'selection': {
            'strategy': 'single',            # Options: 'all', 'single', 'top_n'
            'num_clusters': 3,               # If strategy is 'top_n', how many to select
            'scoring_method': 'composite',   # How to score clusters
            'features_per_cluster': 3,       # Features to select from each cluster
            'feature_selection_method': 'percentile',  # Options: 'mean', 'max', 'percentile'
            'activation_percentile': 25,     # Percentile to use if method='percentile'
        },
    },
    
    # Prompt optimization parameters
    'optimization': {
        # Sequence parameters
        'length': 10,                       # Length of token sequence to optimize
        
        # Training parameters
        'max_steps': 250,
        'lr': 1e-3,
        'lambda_reg': 1e-5,
        
        # Regularization options
        'diversity_penalty': 0.0,           # Penalty for token similarity
        'diversity_window': 5,
        'repetition_penalty': 0.0,          # Penalty for repeating tokens
        'repetition_window': 5,
        'noise_scale': 0.5,
        
        # Visualization
        'visualize_training': True,
        'show_full_sequence': True,
    },
    
    # Explanation generation parameters
    'explanation': {
        'use_lm_coherence': True,
        'coherence_weight': 0.1,
        'max_steps': 25,
        'lr': 1e-3,
        'lambda_reg': 1e-5,
        'length': 10,
    },
    
    # Output settings
    'output': {
        'verbose': True,
        'save_results': False,
        'save_csv': True,
    },
    
    # Hardware settings
    'hardware': {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    },
}

# Update config
config = update_config(config, custom_config)

print("Running custom experiment with the following configuration:")
# Print only the top-level keys for readability
for key, value in custom_config.items():
    if isinstance(value, dict):
        print(f"  {key}: {{...}} ({len(value)} settings)")
    else:
        print(f"  {key}: {value}")

# %% Step 1: Load models
# This only needs to be done once per notebook session
model, sae, lm, tokenizer = load_experiment_models(config['hardware'])
print(f"Models loaded successfully. Using device: {config['hardware']['device']}")

# %% Step 2: Collect and filter data
# This can be skipped if using cached data
filtered_acts, original_indices, prompts = collect_and_filter_data(
    model, sae, 
    config['data'], 
    use_cached=config['data'].get('use_cached_data', False)
)

print(f"Data collection complete: {filtered_acts.shape[0]} prompts, {filtered_acts.shape[1]} filtered features")

# %% Step 3: Cluster features
# Run clustering analysis on the filtered features
labels, reduced_acts = run_clustering(
    filtered_acts, 
    config['clustering'],
    visualize=config['clustering'].get('visualize_clusters', True)
)

print(f"Clustering complete: {len(np.unique(labels))} clusters identified")

# %% Step 4: Analyze clusters (optional)
# Explore which prompts activate each cluster
if config['clustering'].get('explore_clusters', True) and prompts:
    cluster_analysis = analyze_clusters(
        filtered_acts, 
        labels, 
        original_indices, 
        prompts, 
        config['clustering']
    )
    
    print(f"Cluster analysis complete for {len(cluster_analysis)} clusters")

# %% Step 5: Select target features
# Select features based on clustering results
target_features = select_features(
    filtered_acts, 
    labels, 
    original_indices, 
    config['clustering']['selection']
)

print(f"Feature selection complete: {len(target_features)} target features identified")

# %% Step 6: Optimize prompts (can be run per feature)
# Initialize results list
feature_results = []

# Optimize for each target feature
for feature_id in target_features:
    result = optimize_feature(
        model, 
        sae, 
        feature_id, 
        config['optimization'],
        visualize=config['optimization'].get('visualize_training', True)
    )
    feature_results.append(result)
    
    # Optional: save results after each feature
    if config['output'].get('save_csv', True):
        interim_results = {
            'target_features': target_features,
            'feature_results': feature_results
        }
        save_results_to_csv(interim_results, config)

print(f"Optimization complete for {len(feature_results)} features")

# %% Step 7: Generate explanations (optional)
if config['explanation'].get('use_lm_coherence', True) and lm is not None:
    explanation_results = generate_explanations(
        model, 
        sae, 
        feature_results, 
        lm, 
        tokenizer, 
        config['explanation']
    )
    
    # Print explanations
    for feature_id, result in explanation_results.items():
        print(f"\nFeature {feature_id}:")
        if 'error' in result:
            print(f"  Error: {result['error']}")
            continue
            
        if 'template_explanations' in result and result['template_explanations']:
            best_template, template_act = result['template_explanations'][0]
            print(f"  Template: {best_template} (act: {template_act:.4f})")
            
        if 'optimized_explanation' in result:
            print(f"  Optimized: {result['optimized_explanation']} (act: {result['final_activation']:.4f})")

# %% Step 8: Visualize results
if feature_results:
    visualize_results(feature_results)
    
    # Save final results
    if config['output'].get('save_csv', True):
        final_results = {
            'target_features': target_features,
            'feature_results': feature_results
        }
        save_results_to_csv(final_results, config)
        print("Results saved to CSV")
else:
    print("No results to visualize")

# %%

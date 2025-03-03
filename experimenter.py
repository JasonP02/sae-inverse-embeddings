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
)
from utils import save_results_to_csv

"""Run a custom experiment with modified configuration."""

# %% Configuration
# Get default configuration
config = get_default_config()


# %% Step 1: Load models
# This only needs to be done once per notebook session
model, sae, lm, tokenizer = load_experiment_models(config)
print(f"Models loaded successfully. Using device: {config['hardware']['device']}")

# %% Step 2: Collect and filter data
# This can be skipped if using cached data
filtered_acts, original_indices, prompts = collect_and_filter_data(
    model, sae, 
    config['data'], 
    use_cached=config['data'].get('use_cached_data', True)
)

print(f"Data collection complete: {filtered_acts.shape[0]} prompts, {filtered_acts.shape[1]} filtered features")

# %% Step 3: Cluster features
# Run clustering analysis on the filtered features
labels, reduced_acts = run_clustering(
    filtered_acts,
    config['clustering']
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
    
    print(f"Cluster analysis complete")

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
        config['optimization']
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
    optimization_results = {
        'target_features': target_features,
        'feature_results': feature_results
    }
    explanation_results = generate_explanations(
        model, 
        sae,
        optimization_results['feature_results'],
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

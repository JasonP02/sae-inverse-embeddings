import torch
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np

# Import modules
from models import load_models, clear_cache
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
from config import get_default_config, update_config, get_legacy_config
from utils import save_results_to_csv

def run_experiment(model=None, sae=None, config=None, use_cached_data=None):
    """Run the full experiment pipeline with the given configuration.
    
    This is a simplified wrapper around the modular pipeline functions,
    allowing for batch execution of the full pipeline.
    
    Args:
        model: The transformer model (optional, will be loaded if None)
        sae: The sparse autoencoder model (optional, will be loaded if None)
        config: Configuration dictionary (optional, will use default if None)
        use_cached_data: Whether to use cached data (overrides config setting)
        
    Returns:
        Dictionary with experiment results
    """
    # Use default config if none provided
    if config is None:
        config = get_default_config()
    
    # Convert hierarchical config to flat config if needed
    if 'pipeline' in config:
        legacy_config = get_legacy_config(config)
    else:
        legacy_config = config
    
    # Step 1: Load models
    if model is None or sae is None:
        model, sae, lm, tokenizer = load_experiment_models(config)
    else:
        # Load LM if needed for explanations
        lm, tokenizer = None, None
        if config.get('explanation', {}).get('use_lm_coherence', True):
            lm = GPT2LMHeadModel.from_pretrained('distilgpt2').to(config.get('hardware', {}).get('device', 'cpu'))
            tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    
    results = {
        'target_features': [],
        'feature_results': [],
        'explanation_results': None,
        'cluster_analysis': {}
    }
    
    # Step 2: Data Collection and Filtering
    if not config.get('pipeline', {}).get('run_data_collection', True):
        print("Skipping data collection as specified in config")
        return results
    
    filtered_acts, original_indices, prompts = collect_and_filter_data(
        model, sae, 
        config.get('data', {}), 
        use_cached=use_cached_data
    )
    
    # Step 3: Clustering
    if not config.get('pipeline', {}).get('run_clustering', True):
        print("Skipping clustering as specified in config")
        labels = np.zeros(filtered_acts.shape[1], dtype=int)  # Default all to same cluster
        reduced_acts = filtered_acts.T.cpu().numpy()
        cluster_analysis = {}
    else:
        labels, reduced_acts = run_clustering(
            filtered_acts, 
            config.get('clustering', {}),
            visualize=config.get('clustering', {}).get('visualize_clusters', True)
        )
        
        # Analyze clusters if enabled
        if config.get('clustering', {}).get('explore_clusters', True) and prompts:
            results['cluster_analysis'] = analyze_clusters(
                filtered_acts, 
                labels, 
                original_indices, 
                prompts, 
                config.get('clustering', {})
            )
    
    # Step 4: Feature Selection
    if not config.get('pipeline', {}).get('run_feature_selection', True):
        print("Skipping feature selection as specified in config")
        results['target_features'] = []
    else:
        results['target_features'] = select_features(
            filtered_acts, 
            labels, 
            original_indices, 
            config.get('clustering', {}).get('selection', {})
        )
    
    # Step 5: Prompt Optimization
    if not results['target_features'] or not config.get('pipeline', {}).get('run_prompt_optimization', True):
        if not results['target_features']:
            print("No target features selected, skipping prompt optimization")
        else:
            print("Skipping prompt optimization as specified in config")
        return results
    
    for i, target_feature in enumerate(results['target_features']):
        print(f"\nOptimizing for feature {target_feature} ({i+1}/{len(results['target_features'])})")
        feature_data = optimize_feature(
            model, 
            sae, 
            target_feature, 
            config.get('optimization', {})
        )
        results['feature_results'].append(feature_data)
    
    # Step 6: Explanation Generation
    if not results['feature_results'] or not config.get('pipeline', {}).get('run_explanations', True):
        if not results['feature_results']:
            print("No feature results available, skipping explanations")
        else:
            print("Skipping explanation generation as specified in config")
        return results
    
    results['explanation_results'] = generate_explanations(
        model, 
        sae, 
        results['feature_results'], 
        lm, 
        tokenizer, 
        config.get('explanation', {})
    )
    
    return results

def main():
    """Main entry point."""
    # Get default configuration
    config = get_default_config()
    
    # Example of updating config with custom values
    # custom_config = {
    #     'data': {
    #         'n_prompts': 50,
    #     },
    #     'optimization': {
    #         'max_steps': 100,
    #     },
    #     'clustering': {
    #         'visualize_clusters': True
    #     }
    # }
    # config = update_config(config, custom_config)
    
    # Run the experiment
    results = run_experiment(config=config)
    
    # Print summary
    print("\n=== EXPERIMENT SUMMARY ===")
    print(f"Analyzed {len(results['target_features'])} features")
    
    # Example of accessing results
    if results['explanation_results']:
        print("\n=== FEATURE EXPLANATION SUMMARY ===")
        for feature_id, result in results['explanation_results'].items():
            print(f"\nFeature {feature_id}:")
            if 'error' in result:
                print(f"  Error: {result['error']}")
                continue
                
            if 'template_explanations' in result and result['template_explanations']:
                best_template, template_act = result['template_explanations'][0]
                print(f"  Template: {best_template} (act: {template_act:.4f})")
                
            if 'optimized_explanation' in result:
                print(f"  Optimized: {result['optimized_explanation']} (act: {result['final_activation']:.4f})")
    
    # Visualize results
    if results['feature_results']:
        visualize_results(results['feature_results'])
        
        # Save results to CSV
        if config.get('output', {}).get('save_csv', True):
            save_results_to_csv(results, config)
    
    # Clean up
    clear_cache()
    
    return results

if __name__ == "__main__":
    main() 
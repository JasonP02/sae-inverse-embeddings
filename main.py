import torch
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Import modules
from models import load_models, clear_cache
from data import load_diverse_prompts, collect_activations, get_cache_filename, load_processed_data, save_processed_data
from clustering import filter_features, cluster_features, visualize_clusters, select_target_features
from optimization import optimize_embeddings, analyze_results
from visualization import visualize_training, visualize_feature_activations, visualize_multiple_features
from explanation import run_explanation_experiment
from config import get_default_config, update_config

def run_experiment(config, use_cached_data=True):
    """Run the full experiment pipeline."""
    # Step 0: Load models
    model, sae = load_models(config)
    
    # Initialize language model for explanations if needed
    if config.get('use_lm_coherence', True):
        lm = GPT2LMHeadModel.from_pretrained('distilgpt2').to(config['device'])
        tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    else:
        lm, tokenizer = None, None
    
    # Try to load cached data if requested
    cache_filename = get_cache_filename(config)
    cached_data = None
    
    if use_cached_data:
        cached_data = load_processed_data(cache_filename)
    
    if cached_data is not None:
        # Use cached data
        filtered_acts = cached_data['filtered_acts']
        original_indices = cached_data['original_indices']
        print(f"Using cached data with {filtered_acts.shape[0]} prompts and {filtered_acts.shape[1]} filtered features")
    else:
        # Process data from scratch
        # Step 1: Load diverse prompts
        prompts = load_diverse_prompts(config)
        
        # Step 2: Collect activations from prompts
        acts = collect_activations(model, sae, prompts, config)
        
        # Step 3: Filter features
        filtered_acts, original_indices = filter_features(acts, config)
        
        # Save processed data for future use if requested
        if config.get('cache_data', True):
            save_processed_data({
                'filtered_acts': filtered_acts,
                'original_indices': original_indices
            }, cache_filename)
    
    # Step 4: Cluster features
    labels, reduced_acts = cluster_features(filtered_acts, config)
    
    # Step 5: Visualize clusters (optional)
    if config.get('visualize_clusters', True):
        visualize_clusters(reduced_acts, labels)
    
    # Step 6: Select target features from clusters
    target_features = select_target_features(filtered_acts, labels, original_indices, config)
    
    # Step 7: Run prompt optimization for each feature
    feature_results = []
    for i, target_feature in enumerate(target_features):
        print(f"\nOptimizing for feature {target_feature} ({i+1}/{len(target_features)})")
        P, stats = optimize_embeddings(model, sae, target_feature, config)
        result = analyze_results(model, sae, P, target_feature, config)
        
        if config.get('visualize_training', True):
            visualize_training(stats)
        
        # Store comprehensive results for each feature
        dummy_tokens = torch.zeros(1, config['length'], dtype=torch.long, device=config['device'])
        acts = get_feature_activations(model, sae, dummy_tokens, P)
        tokens_by_pos = get_similar_tokens(model, P, top_k=1)
        
        # Extract high-activating tokens with their activations
        pos_activations = [(pos, acts[0, pos, target_feature].item(), tokens_by_pos[pos][0][0], tokens_by_pos[pos][0][1]) 
                           for pos in range(config['length'])]
        sorted_activations = sorted(pos_activations, key=lambda x: x[1], reverse=True)
        
        # Store all the data we need for this feature
        feature_data = {
            'feature_id': target_feature,
            'embeddings': P.detach().clone(),
            'stats': stats,
            'activations': [(pos, act) for pos, act, _, _ in sorted_activations],
            'high_act_tokens': [(token, act) for _, act, token, _ in sorted_activations if act > config['activation_threshold']]
        }
        feature_results.append(feature_data)
        
        # Clear cache between features
        clear_cache()
    
    # Step 8: Generate explanations if requested
    explanation_results = None
    if config.get('generate_explanations', True) and lm is not None:
        print("\n=== Generating Feature Explanations ===")
        explanation_results = run_explanation_experiment(
            model, sae, 
            {'target_features': target_features, 'feature_results': feature_results},
            tokenizer, lm, config
        )
    
    # Return comprehensive results
    return {
        'target_features': target_features,
        'feature_results': feature_results,
        'explanation_results': explanation_results
    }

def main():
    """Main entry point."""
    # Get default configuration
    config = get_default_config()
    
    # Example of updating config with custom values
    # custom_config = {
    #     'n_prompts': 50,
    #     'max_steps': 100,
    #     'visualize_clusters': True
    # }
    # config = update_config(config, custom_config)
    
    # Run the experiment
    results = run_experiment(config, use_cached_data=config.get('use_cached_data', True))
    
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
    
    # Clean up
    clear_cache()
    
    return results

if __name__ == "__main__":
    main() 
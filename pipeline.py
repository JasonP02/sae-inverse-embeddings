"""Pipeline functions for modular experiment execution.

This module provides standalone functions for each stage of the SAE feature analysis pipeline,
designed for use in Jupyter notebooks and experimental research.
"""

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from models import load_models, clear_cache, get_feature_activations, get_similar_tokens
from data import load_diverse_prompts, collect_activations, get_cache_filename, load_processed_data, save_processed_data
from clustering import filter_features, cluster_features, visualize_clusters, select_target_features, explore_cluster_activations
from optimization import optimize_embeddings, analyze_results
from visualization import visualize_training, visualize_feature_activations, visualize_multiple_features
from explanation import run_explanation_experiment


def load_experiment_models(config):
    """Load models needed for the experiment.
    
    Args:
        config: Configuration dictionary with model parameters
        
    Returns:
        tuple of (model, sae, lm, tokenizer) where lm and tokenizer may be None
    """
    # Load main models
    model, sae = load_models(config)
    
    # Optionally load language model for explanations
    lm, tokenizer = None, None
    if config.get('use_lm_coherence', True):
        lm = GPT2LMHeadModel.from_pretrained('distilgpt2').to(config.get('device', 'cpu'))
        tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    
    return model, sae, lm, tokenizer


def collect_and_filter_data(model, sae, data_config, use_cached=None):
    """Collect and filter feature activations.
    
    Args:
        model: The transformer model
        sae: The sparse autoencoder model
        data_config: Configuration dictionary with data collection parameters
        use_cached: Whether to use cached data (overrides config setting)
        
    Returns:
        Tuple of (filtered_acts, original_indices, prompts) containing the filtered
        activations, their original indices, and the prompts used to generate them
    """
    # Determine whether to use cached data
    use_cached_data = use_cached if use_cached is not None else data_config.get('use_cached_data', False)
    cache_filename = get_cache_filename(data_config)
    
    # Try to load cached data if requested
    if use_cached_data:
        cached_data = load_processed_data(cache_filename)
        if cached_data is not None:
            filtered_acts = cached_data['filtered_acts']
            original_indices = cached_data['original_indices']
            prompts = cached_data.get('prompts', [])
            print(f"Using cached data with {filtered_acts.shape[0]} prompts and {filtered_acts.shape[1]} filtered features")
            return filtered_acts, original_indices, prompts
    
    # Otherwise process data from scratch
    # Step 1: Load diverse prompts
    prompts = load_diverse_prompts(data_config)
    
    # Step 2: Collect activations from prompts
    acts = collect_activations(model, sae, prompts, data_config)
    
    # Step 3: Filter features
    filtered_acts, original_indices = filter_features(acts, data_config)
    
    # Save processed data for future use if requested
    if data_config.get('cache_data', True):
        save_processed_data({
            'filtered_acts': filtered_acts,
            'original_indices': original_indices,
            'prompts': prompts
        }, cache_filename)
    
    return filtered_acts, original_indices, prompts


def run_clustering(filtered_acts, clustering_config, visualize=None):
    """Run clustering on filtered activations.
    
    Args:
        filtered_acts: Tensor of filtered activations [n_prompts, n_filtered_features]
        clustering_config: Configuration dictionary with clustering parameters
        visualize: Whether to visualize clusters (overrides config setting)
        
    Returns:
        Tuple of (labels, reduced_acts) containing the cluster labels and reduced activations
    """
    # Step 1: Cluster features
    labels, reduced_acts = cluster_features(filtered_acts, clustering_config)
    
    # Step 2: Optionally visualize clusters
    should_visualize = visualize if visualize is not None else clustering_config.get('visualize_clusters', True)
    if should_visualize:
        visualize_clusters(reduced_acts, labels)
    
    return labels, reduced_acts


def analyze_clusters(filtered_acts, labels, original_indices, prompts, clustering_config):
    """Analyze cluster activations in relation to prompts.
    
    Args:
        filtered_acts: Tensor of filtered activations [n_prompts, n_filtered_features]
        labels: Array of cluster labels
        original_indices: Original indices of filtered features
        prompts: List of text prompts used to generate activations
        clustering_config: Configuration dictionary with clustering parameters
        
    Returns:
        Dictionary with cluster analysis results
    """
    if not prompts or len(prompts) != filtered_acts.shape[0]:
        print(f"Warning: Number of prompts ({len(prompts) if prompts else 0}) doesn't match activation batch size ({filtered_acts.shape[0]})")
        print("Cannot perform prompt-based cluster analysis")
        return {}
    
    return explore_cluster_activations(filtered_acts, labels, original_indices, prompts, clustering_config)


def select_features(filtered_acts, labels, original_indices, selection_config):
    """Select target features from clusters.
    
    Args:
        filtered_acts: Tensor of filtered activations [n_prompts, n_filtered_features]
        labels: Array of cluster labels
        original_indices: Original indices of filtered features
        selection_config: Configuration dictionary with feature selection parameters
        
    Returns:
        List of selected feature indices in the original feature space
    """
    return select_target_features(filtered_acts, labels, original_indices, selection_config)


def optimize_feature(model, sae, feature_id, optimization_config, visualize=None):
    """Optimize embeddings to activate a specific feature.
    
    Args:
        model: The transformer model
        sae: The sparse autoencoder model
        feature_id: The ID of the target feature to optimize for
        optimization_config: Configuration dictionary with optimization parameters
        visualize: Whether to visualize training progress (overrides config setting)
        
    Returns:
        Dictionary with optimization results for the feature
    """
    print(f"\nOptimizing for feature {feature_id}")
    
    # Run optimization
    P, stats = optimize_embeddings(model, sae, feature_id, optimization_config)
    result = analyze_results(model, sae, P, feature_id, optimization_config)
    
    # Optionally visualize training progress
    should_visualize = visualize if visualize is not None else optimization_config.get('visualize_training', False)
    if should_visualize:
        visualize_training(stats)
    
    # Store comprehensive results for the feature
    dummy_tokens = torch.zeros(1, optimization_config.get('length', 10), dtype=torch.long, device=optimization_config.get('device', 'cpu'))
    acts = get_feature_activations(model, sae, dummy_tokens, P)
    tokens_by_pos = get_similar_tokens(model, P, top_k=1)
    
    # Extract high-activating tokens with their activations
    pos_activations = [(pos, acts[0, pos, feature_id].item(), tokens_by_pos[pos][0][0], tokens_by_pos[pos][0][1]) 
                       for pos in range(optimization_config.get('length', 10))]
    sorted_activations = sorted(pos_activations, key=lambda x: x[1], reverse=True)
    
    # Store all the data we need for this feature
    feature_data = {
        'feature_id': feature_id,
        'embeddings': P.detach().clone(),
        'stats': stats,
        'activations': [(pos, act) for pos, act, _, _ in sorted_activations],
        'high_act_tokens': [(token, act) for _, act, token, _ in sorted_activations if act > optimization_config.get('activation_threshold', 0.1)]
    }
    
    # Clear cache
    clear_cache()
    
    return feature_data


def generate_explanations(model, sae, feature_results, lm, tokenizer, explanation_config):
    """Generate explanations for features.
    
    Args:
        model: The transformer model
        sae: The sparse autoencoder model
        feature_results: List of feature result dictionaries from optimize_feature
        lm: Language model for coherence scoring
        tokenizer: Tokenizer for the language model
        explanation_config: Configuration dictionary with explanation parameters
        
    Returns:
        Dictionary mapping feature IDs to explanation results
    """
    if not feature_results:
        print("No feature results available, skipping explanations")
        return {}
    
    if lm is None or tokenizer is None:
        print("Language model not available, skipping explanations")
        return {}
    
    print("\n=== Generating Feature Explanations ===")
    target_features = [result['feature_id'] for result in feature_results]
    
    return run_explanation_experiment(
        model, sae, 
        {'target_features': target_features, 'feature_results': feature_results},
        tokenizer, lm, explanation_config
    )


def visualize_results(feature_results, which_features=None):
    """Visualize results for multiple features.
    
    Args:
        feature_results: List of feature result dictionaries from optimize_feature
        which_features: List of feature IDs to visualize (None for all)
        
    Returns:
        None (displays visualizations)
    """
    if not feature_results:
        print("No feature results to visualize")
        return
    
    # Create a dictionary mapping feature_id to result
    feature_dict = {data['feature_id']: data for data in feature_results}
    
    # Filter to specific features if requested
    if which_features:
        feature_dict = {fid: data for fid, data in feature_dict.items() if fid in which_features}
    
    # Visualize multiple features
    visualize_multiple_features(feature_dict)
    
    # Visualize token heatmap for the first feature
    first_feature = list(feature_dict.keys())[0]
    visualize_token_heatmap(feature_dict, first_feature) 
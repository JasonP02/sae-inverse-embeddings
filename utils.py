import os
import csv
import datetime
import torch
import numpy as np
from einops import rearrange

def ensure_dir(directory):
    """Ensure a directory exists, creating it if necessary."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def save_results_to_csv(results, config):
    """Save experiment results to CSV files."""
    if not config.get('save_csv', True):
        return
    
    # Create output directory
    output_dir = config.get('csv_output_dir', 'feature_results')
    ensure_dir(output_dir)
    
    # Generate timestamp for filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save feature summary
    summary_file = os.path.join(output_dir, f"feature_summary_{timestamp}.csv")
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Feature ID', 'Max Activation', 'Mean Activation', 'Top Tokens'])
        
        for feature_data in results['feature_results']:
            feature_id = feature_data['feature_id']
            high_act_tokens = [token for token, _ in feature_data['high_act_tokens'][:5]]
            
            # Get activation statistics
            activations = [act for _, act in feature_data['activations']]
            max_act = max(activations)
            mean_act = sum(activations) / len(activations) if activations else 0
            
            writer.writerow([
                feature_id,
                f"{max_act:.4f}",
                f"{mean_act:.4f}",
                ", ".join(high_act_tokens)
            ])
    
    print(f"Saved feature summary to {summary_file}")
    
    # Save detailed results for each feature
    for feature_data in results['feature_results']:
        feature_id = feature_data['feature_id']
        feature_file = os.path.join(output_dir, f"feature_{feature_id}_{timestamp}.csv")
        
        with open(feature_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Position', 'Activation', 'Token', 'Token Score'])
            
            # Get token data
            tokens_by_pos = feature_data.get('tokens', [])
            activations = feature_data['activations']
            
            # Sort by activation (descending)
            sorted_acts = sorted(activations, key=lambda x: x[1], reverse=True)
            
            for pos, act_val in sorted_acts:
                if tokens_by_pos:
                    token, score = tokens_by_pos[pos][0]
                    writer.writerow([pos, f"{act_val:.4f}", token, f"{score:.4f}"])
                else:
                    writer.writerow([pos, f"{act_val:.4f}", "", ""])
        
        print(f"Saved detailed results for feature {feature_id} to {feature_file}")
    
    # Save explanation results if available
    if results.get('explanation_results'):
        explanation_file = os.path.join(output_dir, f"explanations_{timestamp}.csv")
        with open(explanation_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Feature ID', 'Template Explanation', 'Template Activation', 
                            'Optimized Explanation', 'Optimized Activation'])
            
            for feature_id, result in results['explanation_results'].items():
                if 'error' in result:
                    writer.writerow([feature_id, f"Error: {result['error']}", "", "", ""])
                    continue
                
                template_explanation = ""
                template_activation = ""
                if 'template_explanations' in result and result['template_explanations']:
                    template_explanation, template_activation = result['template_explanations'][0]
                    template_activation = f"{template_activation:.4f}"
                
                optimized_explanation = ""
                optimized_activation = ""
                if 'optimized_explanation' in result:
                    optimized_explanation = result['optimized_explanation']
                    optimized_activation = f"{result['final_activation']:.4f}"
                
                writer.writerow([
                    feature_id,
                    template_explanation,
                    template_activation,
                    optimized_explanation,
                    optimized_activation
                ])
        
        print(f"Saved explanation results to {explanation_file}")

def save_tensor_data(data, filename):
    """Save tensor data to disk."""
    torch.save(data, filename)
    print(f"Saved tensor data to {filename}")

def load_tensor_data(filename):
    """Load tensor data from disk."""
    if os.path.exists(filename):
        data = torch.load(filename)
        print(f"Loaded tensor data from {filename}")
        return data
    else:
        print(f"File not found: {filename}")
        return None

def print_tensor_stats(tensor, name="Tensor"):
    """Print statistics about a tensor."""
    if not isinstance(tensor, torch.Tensor):
        print(f"{name} is not a tensor")
        return
    
    print(f"{name} stats:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Device: {tensor.device}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Min: {tensor.min().item():.4f}")
    print(f"  Max: {tensor.max().item():.4f}")
    print(f"  Mean: {tensor.mean().item():.4f}")
    print(f"  Std: {tensor.std().item():.4f}")
    
    # Count non-zero elements
    non_zero = (tensor != 0).sum().item()
    sparsity = non_zero / tensor.numel()
    print(f"  Non-zero elements: {non_zero} ({sparsity:.2%})")

def batch_process(items, batch_size, process_fn, *args, **kwargs):
    """Process items in batches to manage memory."""
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = process_fn(batch, *args, **kwargs)
        results.extend(batch_results)
    return results 
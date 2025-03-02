import torch
import itertools
from models import get_feature_activations, get_similar_tokens

def generate_explanatory_prompt(model, sae, feature_id, high_act_tokens, config):
    """Generate a coherent explanation using high-activating tokens."""
    # Template options
    templates = [
        "This feature detects {0} related to {1}.",
        "This feature activates for {0} in the context of {1}.",
        "This feature represents {0} within {1} systems.",
        "This feature responds to {0} and {1}."
    ]
    
    results = []
    for template in templates:
        # Try different permutations of tokens
        for tokens in itertools.permutations(high_act_tokens[:5], 2):
            prompt = template.format(tokens[0], tokens[1])
            tokens = model.to_tokens(prompt)
            
            # Measure activation
            acts = get_feature_activations(model, sae, tokens)
            activation = acts[0, :, feature_id].max().item()
            
            results.append((prompt, activation))
    
    # Return top activating coherent explanation
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:5]  # Return top 5 candidates

def evaluate_explanations(model, sae, feature_id, explanations):
    """Evaluate multiple explanations for a feature."""
    results = []
    
    for explanation in explanations:
        tokens = model.to_tokens(explanation)
        acts = get_feature_activations(model, sae, tokens)
        activation = acts[0, :, feature_id].max().item()
        
        # Get activations for other features to check specificity
        all_feature_acts = acts[0].max(dim=0)[0]
        specificity = activation / (all_feature_acts.mean().item())
        
        results.append({
            'explanation': explanation,
            'activation': activation,
            'specificity': specificity,
            'length': len(tokens[0])
        })
    
    return results

def calculate_lm_coherence(P, model, tokenizer, lm, config):
    """Calculate language model coherence score for a sequence."""
    tokens_by_pos = get_similar_tokens(model, P, top_k=1)
    text = " ".join([t[0][0] for t in tokens_by_pos])
    inputs = tokenizer(text, return_tensors='pt', truncation=True).to(config['device'])
    outputs = lm(**inputs, labels=inputs['input_ids'])
    return outputs.loss

def optimize_explanation(model, sae, feature_id, initial_prompt, tokenizer, lm, config):
    """Optimize an explanation to maximize feature activation."""
    tokens = model.to_tokens(initial_prompt)
    P = torch.nn.Parameter(model.W_E[tokens[0]])
    
    # Define which positions to freeze (e.g., "This feature detects")
    frozen_mask = torch.zeros_like(P, dtype=torch.bool)
    frozen_mask[:4] = True  # Freeze first 4 tokens
    
    # Optimize
    optimizer = torch.optim.AdamW([P], lr=config.get('lr_explanation', config['lr']))
    
    for step in range(config.get('max_steps_explanation', config['max_steps'])):
        optimizer.zero_grad()
        
        # Get activations
        acts = get_feature_activations(model, sae, torch.zeros(1, P.shape[0], dtype=torch.long, device=config['device']), P)
        activation = acts[0, :, feature_id].max()
        
        # Calculate losses
        feature_loss = -activation
        
        # Coherence loss using language model (optional)
        if config.get('use_lm_coherence', False):
            coherence_loss = calculate_lm_coherence(P, model, tokenizer, lm, config)
            total_loss = feature_loss + config['coherence_weight'] * coherence_loss
        else:
            total_loss = feature_loss
            
        # Add regularization
        with torch.no_grad():
            similarity = torch.nn.functional.normalize(P, dim=-1) @ \
                        torch.nn.functional.normalize(model.W_E, dim=-1).T
            closest_tokens = similarity.max(dim=1).indices
            closest_embeddings = model.W_E[closest_tokens]
        
        embedding_diff = P - closest_embeddings
        # Don't regularize frozen positions
        embedding_diff[frozen_mask] = 0
        embedding_dist = torch.norm(embedding_diff, p='fro')
        reg_loss = config.get('lambda_reg_explanation', config['lambda_reg']) * embedding_dist
        
        total_loss += reg_loss
        
        # Backprop and update
        total_loss.backward()
        
        # Zero gradients for frozen positions
        if P.grad is not None:
            P.grad[frozen_mask] = 0
            
        optimizer.step()
        
        # Logging
        if step % 100 == 0:
            tokens_by_pos = get_similar_tokens(model, P, top_k=1)
            current_text = " ".join([t[0][0] for t in tokens_by_pos])
            print(f"Step {step}: {current_text} (act={activation.item():.4f})")
    
    # Get final results
    tokens_by_pos = get_similar_tokens(model, P, top_k=1)
    optimized_explanation = " ".join([t[0][0] for t in tokens_by_pos])
    
    return optimized_explanation, activation.item()

def run_explanation_experiment(model, sae, experiment_results, tokenizer, lm, config):
    """Run explanation generation for features using direct experiment results."""
    results = {}
    feature_results = experiment_results['feature_results']
    
    for feature_data in feature_results:
        feature_id = feature_data['feature_id']
        print(f"\n=== Generating explanations for feature {feature_id} ===")
        
        try:
            # Get high-activating tokens directly from experiment results
            high_act_tokens = [token for token, act in feature_data['high_act_tokens']]
            
            if len(high_act_tokens) < 2:
                print(f"Not enough high-activating tokens for feature {feature_id}, skipping")
                continue
                
            print(f"High-activating tokens: {high_act_tokens[:5]}")
            
            # Generate template-based explanations
            template_explanations = generate_explanatory_prompt(
                model, sae, feature_id, high_act_tokens, config
            )
            
            print("\nTop template-based explanations:")
            for i, (explanation, activation) in enumerate(template_explanations):
                print(f"{i+1}. {explanation} (activation: {activation:.4f})")
            
            # Optimize explanation if requested
            if config.get('optimize_explanations', True):
                print("\nOptimizing explanation...")
                
                # Use the best template as starting point
                best_prompt, _ = template_explanations[0]
                
                optimized_explanation, final_activation = optimize_explanation(
                    model, sae, feature_id, best_prompt, tokenizer, lm, config
                )
                
                print(f"\nOptimized explanation: {optimized_explanation}")
                print(f"Final activation: {final_activation:.4f}")
                
                # Compare with template explanations
                all_explanations = [explanation for explanation, _ in template_explanations]
                all_explanations.append(optimized_explanation)
                
                evaluation = evaluate_explanations(model, sae, feature_id, all_explanations)
                
                # Store results
                results[feature_id] = {
                    'high_act_tokens': high_act_tokens,
                    'template_explanations': template_explanations,
                    'optimized_explanation': optimized_explanation,
                    'final_activation': final_activation,
                    'evaluation': evaluation
                }
            else:
                # Just store template results
                results[feature_id] = {
                    'high_act_tokens': high_act_tokens,
                    'template_explanations': template_explanations
                }
                
        except Exception as e:
            print(f"Error generating explanations for feature {feature_id}: {e}")
            results[feature_id] = {'error': str(e)}
    
    return results 
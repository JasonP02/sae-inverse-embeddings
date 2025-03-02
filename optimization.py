import torch
from models import get_feature_activations, get_similar_tokens

def optimize_embeddings(model, sae, target_feature, config):
    """Optimize embeddings to maximize activation of a target feature."""
    length = config['length']
    device = model.cfg.device
    P = torch.nn.Parameter(
        model.W_E[torch.randint(0, model.cfg.d_vocab, (length,))] + 
        torch.randn(length, model.cfg.d_model, device=device) * config['noise_scale']
    )
    optimizer = torch.optim.AdamW([P], lr=config['lr'], weight_decay=0)
    dummy_tokens = torch.zeros(1, length, dtype=torch.long, device=device)
    stats = {'loss': [], 'target_activation': []}

    for step in range(config['max_steps']):
        optimizer.zero_grad()
        acts = get_feature_activations(model, sae, dummy_tokens, P)
        target_activation = acts[0, :, target_feature].max()
        loss_feature = -target_activation
        with torch.no_grad():
            similarity = torch.nn.functional.normalize(P, dim=-1) @ \
                        torch.nn.functional.normalize(model.W_E, dim=-1).T
            closest_tokens = similarity.max(dim=1).indices
            closest_embeddings = model.W_E[closest_tokens]
        embedding_diff = P - closest_embeddings
        embedding_dist = torch.norm(embedding_diff, p='fro')
        loss_reg = config['lambda_reg'] * embedding_dist
        
        token_diversity_penalty = 0.0
        if config['diversity_penalty'] > 0:
            window_size = config.get('diversity_window', 5)
            with torch.no_grad():
                sim_matrix = torch.nn.functional.cosine_similarity(
                    P.unsqueeze(1), P.unsqueeze(0), dim=2
                )  # [length, length]
                mask = torch.triu(torch.ones_like(sim_matrix), diagonal=1)  # Upper triangle
                window_mask = torch.abs(torch.arange(config['length'], device=device).unsqueeze(0) - 
                                    torch.arange(config['length'], device=device).unsqueeze(1)) <= window_size
                local_mask = mask * window_mask
                token_diversity_penalty = (sim_matrix * local_mask).sum() / local_mask.sum() * config['diversity_penalty']
        else:
            token_diversity_penalty = torch.tensor(0.0, device=device)

        repetition_penalty = 0.0
        if config['repetition_penalty'] > 0:
            window_size = config.get('repetition_window', 5)
            with torch.no_grad():
                local_repeats = torch.zeros(config['length'], device=device)
                for offset in range(1, window_size + 1):
                    # Compare with tokens shifted forward and backward
                    forward = torch.cat([closest_tokens[offset:], torch.full((offset,), -1, device=device)])
                    backward = torch.cat([torch.full((offset,), -1, device=device), closest_tokens[:-offset]])
                    local_repeats += (closest_tokens == forward).float() + (closest_tokens == backward).float()
                repetition_penalty = config['repetition_penalty'] * local_repeats.sum() / config['length']
        else:
            repetition_penalty = torch.tensor(0.0, device=device)

        loss = loss_feature + loss_reg + token_diversity_penalty + repetition_penalty
        loss.backward()
        optimizer.step()
        stats['loss'].append(loss.item())
        stats['target_activation'].append(acts[0, :, target_feature].max().item())
        
        if step % 20 == 0 and config.get('verbose', True):
            rep_val = repetition_penalty.item() if isinstance(repetition_penalty, torch.Tensor) else repetition_penalty
            div_val = token_diversity_penalty.item() if isinstance(token_diversity_penalty, torch.Tensor) else token_diversity_penalty
            print(f"Step {step}: repetition_penalty = {rep_val:.4f}, token_diversity_penalty = {div_val:.4f}, loss = {loss.item():.4f}, target_activation = {target_activation.item():.4f}")
            
    return P, stats

def analyze_results(model, sae, P, target_feature, config):
    """Analyze the results of optimization."""
    dummy_tokens = torch.zeros(1, config['length'], dtype=torch.long, device=model.cfg.device)
    acts = get_feature_activations(model, sae, dummy_tokens, P)
    tokens_by_pos = get_similar_tokens(model, P, top_k=1)

    # Calculate statistics
    feature_acts = acts[0, :, target_feature]
    max_act = feature_acts.max().item()
    mean_act = feature_acts.mean().item()
    
    # Calculate skewness manually
    centered = feature_acts - mean_act
    std = feature_acts.std().item()
    if std > 0:
        skew = torch.mean(centered**3).item() / (std**3)
    else:
        skew = 0.0
        
    print(f"Feature {target_feature}: Max Act={max_act:.4f}, Mean Act={mean_act:.4f}, Skew={skew:.4f}")
    
    print(f"\n=== RESULTS FOR FEATURE {target_feature} ===")
    
    # Show top activating positions
    pos_activations = [(pos, acts[0, pos, target_feature].item()) for pos in range(config['length'])]
    top_positions = sorted(pos_activations, key=lambda x: x[1], reverse=True)[:10]
    
    print(f"Target feature activation: {acts[0, :, target_feature].max():.4f}")
    print("\nTop activating positions:")
    for pos, act_val in top_positions:
        token, score = tokens_by_pos[pos][0]
        print(f"Position {pos}: {act_val:.4f} - '{token}' ({score:.3f})")
    
    # Show full sequence for reference
    if config.get('show_full_sequence', False):
        print("\nFull optimized sequence:")
        for pos in range(config['length']):
            token, score = tokens_by_pos[pos][0]
            print(f"Position {pos}: {acts[0, pos, target_feature].item():.4f} - '{token}' ({score:.3f})")
            
    return {
        'feature_id': target_feature,
        'embeddings': P.detach().clone(),
        'activations': pos_activations,
        'tokens': tokens_by_pos,
        'max_activation': max_act,
        'mean_activation': mean_act,
        'skewness': skew
    } 
#%% imports
import torch
from sae_lens import SAE, HookedSAETransformer
from transformer_lens import ActivationCache, utils
import plotly.graph_objects as go
import plotly.subplots as sp

def load_models(config):
    """Load the transformer model and SAE based on config."""
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    model = HookedSAETransformer.from_pretrained("EleutherAI/pythia-70m-deduped", device=device)
    sae, _, _ = SAE.from_pretrained(
        release="pythia-70m-deduped-mlp-sm",
        sae_id="blocks.3.hook_mlp_out",
        device=device
    )
    return model, sae


# %% Helper functions
def create_embed_hook(P):
    """Create a hook function that uses the provided embeddings P."""
    def hook(value, hook):
        return P.unsqueeze(0)
    return hook

def get_feature_activations(model, sae, tokens, P=None):
    """Compute feature activations for given tokens."""
    hooks = [('hook_embed', create_embed_hook(P))] if P is not None else []
    with model.hooks(fwd_hooks=hooks):
        _, cache = model.run_with_cache_with_saes(tokens, saes=[sae])
    return cache['blocks.3.hook_mlp_out.hook_sae_acts_post']

def get_similar_tokens(model, embeddings, top_k=5):
    """Get the most similar tokens for each position in the embeddings."""
    with torch.no_grad():
        vocab_embeds = model.W_E
        similarity = torch.nn.functional.normalize(embeddings, dim=-1) @ \
                     torch.nn.functional.normalize(vocab_embeds, dim=-1).T
        top_tokens = similarity.topk(top_k, dim=-1)
        return [(model.to_string(idx), score) for pos in range(len(embeddings))
                for idx, score in zip(top_tokens.indices[pos], top_tokens.values[pos])]

def find_target_feature(model, sae, config):
    """Find a target feature based on the experiment type."""
    length = config['length']
    device = model.cfg.device

    if config['experiment_type'] == 'high_activation':
        sample_tokens = torch.randint(0, model.cfg.d_vocab, (config['n_samples'], length), device=device)
        acts = get_feature_activations(model, sae, sample_tokens)
        max_acts = acts.max(dim=0).values.max(dim=0).values
        return torch.argmax(max_acts).item()

    elif config['experiment_type'] == 'contextual':
        prompts = config.get('prompts', ["Jimmy carter was the president", "Luke went to the store"])
        variances = torch.zeros(sae.cfg.d_sae, device=device)
        for prompt in prompts:
            acts = get_feature_activations(model, sae, model.to_tokens(prompt))
            variances += acts[0].var(dim=0)
        return torch.argmax(variances).item()

    elif config['experiment_type'] == 'co_activated':
        base_feature = find_target_feature(model, sae, {**config, 'experiment_type': 'high_activation'})
        sample_tokens = torch.randint(0, model.cfg.d_vocab, (config['n_samples'], length), device=device)
        acts = get_feature_activations(model, sae, sample_tokens).reshape(-1, sae.cfg.d_sae)
        target_active = acts[:, base_feature] > 0.1
        co_acts = acts[target_active].mean(dim=0)
        co_acts[base_feature] = -float('inf')  # Exclude the base feature
        return torch.argmax(co_acts).item()

    raise ValueError(f"Unknown experiment type: {config['experiment_type']}")

#%% optimization
def optimize_embeddings(model, sae, target_feature, config):
    """Optimize embeddings to maximize activation of the target feature."""
    length = config['length']
    device = model.cfg.device
    P = torch.nn.Parameter(
        model.W_E[torch.randint(0, model.cfg.d_vocab, (length,))] + 
        torch.randn(length, model.cfg.d_model, device=device) * config.get('noise_scale', 0.1)
    )
    optimizer = torch.optim.AdamW([P], lr=config.get('lr', 1e-1), weight_decay=0)
    dummy_tokens = torch.zeros(1, length, dtype=torch.long, device=device)
    stats = {'loss': [], 'target_activation': []}

    for step in range(config['max_steps']):
        optimizer.zero_grad()
        acts = get_feature_activations(model, sae, dummy_tokens, P)
        loss = -acts[0, :, target_feature].max()  # Simple loss for example
        loss.backward()
        optimizer.step()
        stats['loss'].append(loss.item())
        stats['target_activation'].append(acts[0, :, target_feature].max().item())

    return P, stats

# %% analyze, visualize
def analyze_results(model, sae, P, target_feature, config):
    """Analyze and visualize optimization results."""
    dummy_tokens = torch.zeros(1, config['length'], dtype=torch.long, device=model.cfg.device)
    acts = get_feature_activations(model, sae, dummy_tokens, P)
    tokens = get_similar_tokens(model, P, top_k=1)
    print(f"Optimized tokens: {' '.join(t[0] for t in tokens)}")
    print(f"Target feature {target_feature} activation: {acts[0, :, target_feature].max():.4f}")

def visualize_training(stats):
    """Visualize training progress."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=stats['loss'], mode='lines', name='Loss'))
    fig.add_trace(go.Scatter(y=stats['target_activation'], mode='lines', name='Target Activation'))
    fig.update_layout(title="Training Progress", xaxis_title="Step", yaxis_title="Value")
    fig.show()

# %% Experiment Runner
def run_experiment(config):
    """Run an experiment with the given configuration."""
    model, sae = load_models(config)
    target_feature = find_target_feature(model, sae, config)
    P, stats = optimize_embeddings(model, sae, target_feature, config)
    analyze_results(model, sae, P, target_feature, config)
    if config.get('visualize', True):
        visualize_training(stats)
    return target_feature, P, stats

# Example usage
config = {
    'experiment_type': 'high_activation',
    'length': 4,
    'max_steps': 100,
    'n_samples': 100,
    'lr': 1e-1,
    'noise_scale': 0.1,
    'visualize': True
}
feature, P, stats = run_experiment(config)

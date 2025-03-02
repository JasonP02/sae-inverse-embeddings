import torch
import gc
from sae_lens import SAE, HookedSAETransformer

# Global model variables
model = None
sae = None

def load_models(config):
    """Load the transformer model and SAE based on config."""
    global model, sae
    if model is None or sae is None:
        device = config.get('hardware', {}).get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        model = HookedSAETransformer.from_pretrained("EleutherAI/pythia-70m-deduped", device=device)
        sae, _, _ = SAE.from_pretrained(
            release="pythia-70m-deduped-mlp-sm",
            sae_id="blocks.3.hook_mlp_out",
            device=device
        )
    return model, sae

def clear_cache():
    """Clear CUDA cache to free memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def create_embed_hook(P):
    def hook(value, hook):
        return P.unsqueeze(0)
    return hook

def get_feature_activations(model, sae, tokens, P=None):
    hooks = [('hook_embed', create_embed_hook(P))] if P is not None else []
    with model.hooks(fwd_hooks=hooks):
        _, cache = model.run_with_cache_with_saes(tokens, saes=[sae])
    return cache['blocks.3.hook_mlp_out.hook_sae_acts_post']

def get_similar_tokens(model, embeddings, top_k=5):
    with torch.no_grad():
        vocab_embeds = model.W_E
        similarity = torch.nn.functional.normalize(embeddings, dim=-1) @ \
                     torch.nn.functional.normalize(vocab_embeds, dim=-1).T
        top_tokens = similarity.topk(top_k, dim=-1)
        result = []
        for pos in range(len(embeddings)):
            pos_tokens = [(model.to_string(top_tokens.indices[pos][i]), top_tokens.values[pos][i].item()) 
                          for i in range(top_k)]
            result.append(pos_tokens)
        return result 
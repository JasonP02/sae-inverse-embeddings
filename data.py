import torch
import os
import random
from datasets import load_dataset
from models import get_feature_activations, clear_cache

def load_diverse_prompts(config):
    """Load diverse prompts from datasets."""
    n_prompts = config.get('n_prompts', 100)
    prompts = []

    # Use a simple dataset: 'wikitext'
    try:
        print("Loading prompts from WikiText...")
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        wiki_prompts = [item['text'].strip() for item in dataset if len(item['text'].strip()) > 10]
        prompts.extend(wiki_prompts[:n_prompts])
        print(f"Added {len(wiki_prompts[:n_prompts])} prompts from WikiText")
    except Exception as e:
        print(f"Error with WikiText: {e}")

    prompts = list(set(prompts))  # Deduplicate
    random.shuffle(prompts)
    prompts = prompts[:n_prompts]
    print(f"Total unique prompts loaded: {len(prompts)}")
    return prompts

def collect_activations(model, sae, prompts, config):
    """Collect feature activations from prompts."""
    all_acts = []
    batch_size = config.get('batch_size', 10)
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{len(prompts) // batch_size + 1}...")
        batch_acts = []
        for prompt in batch_prompts:
            try:
                tokens = model.to_tokens(prompt)
                acts = get_feature_activations(model, sae, tokens)
                batch_acts.append(acts.mean(dim=1).squeeze(0))
            except Exception as e:
                print(f"Skipping prompt '{prompt[:30]}...': {e}")
        if batch_acts:
            all_acts.extend(batch_acts)
        if (i + batch_size) % 50 == 0:
            clear_cache()
    acts = torch.stack(all_acts)
    print(f"Collected activations for {acts.shape[0]} prompts, {acts.shape[1]} features")
    return acts

def save_processed_data(data, filename):
    """Save processed data to disk to avoid reprocessing."""
    print(f"Saving processed data to {filename}...")
    torch.save(data, filename)
    print("Data saved successfully.")

def load_processed_data(filename):
    """Load processed data from disk."""
    if os.path.exists(filename):
        print(f"Loading processed data from {filename}...")
        try:
            data = torch.load(filename)
            print("Data loaded successfully.")
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    else:
        print(f"No cached data found at {filename}")
        return None

def get_cache_filename(config):
    """Generate a cache filename based on config parameters."""
    # Create a unique filename based on key parameters
    params = [
        f"prompts_{config['n_prompts']}",
        f"entropy_{config['entropy_threshold_low']:.2f}_{config['entropy_threshold_high']:.2f}",
        f"sparsity_{config['sparsity_min']:.2f}_{config['sparsity_max']:.2f}"
    ]
    
    # Create directory if it doesn't exist
    cache_dir = config.get('cache_dir', 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    return os.path.join(cache_dir, f"processed_data_{'_'.join(params)}.pt") 
# SAE Inverse Embeddings

This package explores the inverse problem of finding input embeddings that maximize 
the activation of specific sparse autoencoder (SAE) features in a transformer model.

## Overview

The SAE Inverse Embeddings project provides tools to:

1. Collect SAE feature activations from diverse prompts
2. Filter features based on entropy and sparsity
3. Cluster features to identify interesting patterns
4. Select target features for further analysis
5. Optimize prompts to maximize activation of selected features
6. Generate explanations for what features might represent

## Modular Approach

The codebase is designed with a modular approach for experimental research. Each stage of the pipeline can be run independently:

- Data collection and filtering
- Clustering and analysis
- Feature selection
- Prompt optimization
- Explanation generation

## Usage

### Jupyter Notebook Style (Recommended for Experimentation)

```python
# Import pipeline functions
from config import get_default_config, update_config
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

# Configure your experiment
config = get_default_config()
custom_config = {
    'data': {
        'n_prompts': 100,
        # other data settings...
    },
    'clustering': {
        # clustering settings...
    }
    # etc.
}
config = update_config(config, custom_config)

# Load models (once per notebook session)
model, sae, lm, tokenizer = load_experiment_models(config['hardware'])

# Run only the parts you need
filtered_acts, original_indices, prompts = collect_and_filter_data(model, sae, config['data'])
labels, reduced_acts = run_clustering(filtered_acts, config['clustering'])
cluster_analysis = analyze_clusters(filtered_acts, labels, original_indices, prompts, config['clustering'])
target_features = select_features(filtered_acts, labels, original_indices, config['clustering']['selection'])

# Optimize for each feature separately (or just the ones you're interested in)
feature_results = []
for feature_id in target_features[:3]:  # Only the first 3 for example
    result = optimize_feature(model, sae, feature_id, config['optimization'])
    feature_results.append(result)

# Visualize results
visualize_results(feature_results)
```

### Batch Execution

For running the entire pipeline at once:

```python
from main import run_experiment
from config import get_default_config, update_config

# Configure your experiment
config = get_default_config()
config = update_config(config, custom_config)

# Run the entire pipeline
results = run_experiment(config=config)
```

## Configuration

The configuration system uses a hierarchical structure for better organization:

```python
{
    'hardware': {
        'device': 'cuda',  # or 'cpu'
    },
    'data': {
        'n_prompts': 100,
        # Data collection parameters
        'filtering': {
            # Feature filtering parameters
        }
    },
    'clustering': {
        # Clustering parameters
        'dbscan': {
            # DBSCAN specific parameters
        },
        'selection': {
            # Feature selection parameters
        }
    },
    'optimization': {
        # Optimization parameters
    },
    'explanation': {
        # Explanation parameters
    },
    'output': {
        # Output parameters
    }
}
```

## Pipeline Control

To control which stages of the pipeline run in batch mode, use the 'pipeline' section:

```python
'pipeline': {
    'run_data_collection': True,      # Collect and filter data
    'run_clustering': True,           # Run clustering analysis
    'run_feature_selection': True,    # Select features from clusters
    'run_prompt_optimization': False, # Skip prompt optimization
    'run_explanations': False,        # Skip explanation generation
}
```

This allows for flexible experimentation by enabling/disabling specific stages.

## Examples

See `examples/custom_experiment.py` for a complete Jupyter-style example.

## Project Structure

The codebase is organized into the following modules:

- `models.py`: Model loading and management
- `data.py`: Data loading and processing
- `clustering.py`: Feature clustering functionality
- `optimization.py`: Embedding optimization
- `visualization.py`: Visualization utilities
- `explanation.py`: Feature explanation generation
- `utils.py`: Utility functions
- `config.py`: Configuration settings
- `main.py`: Main script to run experiments

## Requirements

- Python 3.8+
- PyTorch
- transformer_lens
- sae_lens
- scikit-learn
- UMAP
- Plotly
- Datasets (HuggingFace)
- Transformers (HuggingFace)

## Usage

### Basic Usage

To run the default experiment:

```python
python main.py
```

### Custom Configuration

You can modify the configuration in `main.py` by updating the default config:

```python
from config import get_default_config, update_config

# Get default configuration
config = get_default_config()

# Update with custom values
custom_config = {
    'n_prompts': 50,
    'max_steps': 100,
    'visualize_clusters': True
}
config = update_config(config, custom_config)

# Run with custom config
results = run_experiment(config)
```

## Workflow

1. **Data Collection**: Load diverse prompts and collect feature activations
2. **Feature Filtering**: Filter features based on entropy and sparsity
3. **Clustering**: Cluster features to find related groups
4. **Feature Selection**: Select representative features from each cluster
5. **Optimization**: Optimize input embeddings to maximize feature activation
6. **Analysis**: Analyze and visualize the results
7. **Explanation**: Generate explanations for what features detect

## Configuration Options

Key configuration parameters:

- `n_prompts`: Number of prompts to use for feature analysis
- `length`: Length of token sequence to optimize
- `max_steps`: Maximum optimization steps
- `entropy_threshold_low/high`: Entropy thresholds for feature filtering
- `sparsity_min/max`: Sparsity thresholds for feature filtering
- `features_per_cluster`: Number of features to select from each cluster
- `visualize_clusters`: Whether to visualize feature clusters
- `use_cached_data`: Whether to use cached data if available

See `config.py` for the full list of configuration options.

## Visualization

The project includes several visualization utilities:

- Cluster visualization with UMAP
- Training progress visualization
- Feature activation visualization
- Token heatmaps

## Extending the Project

To add new functionality:

1. Add new functions to the appropriate module
2. Update the configuration in `config.py` if needed
3. Integrate with the main workflow in `main.py`

## License

[MIT License](LICENSE) 
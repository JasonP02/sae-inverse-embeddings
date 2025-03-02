# SAE Inverse Embeddings

This project explores the inverse problem of finding input embeddings that maximize the activation of specific sparse autoencoder (SAE) features in a transformer model.

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
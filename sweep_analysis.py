"""Tools for analyzing and interpreting clustering sweep results."""

import wandb
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from typing import Dict, List, Optional

def load_sweep_results(sweep_id: str, entity: Optional[str] = None, project: Optional[str] = None) -> pd.DataFrame:
    """Load results from a W&B sweep into a pandas DataFrame.
    
    Args:
        sweep_id: The ID of the sweep to analyze
        entity: Optional W&B entity name
        project: Optional W&B project name
        
    Returns:
        DataFrame containing sweep results
    """
    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}" if entity and project else sweep_id)
    
    # Get all runs in the sweep
    runs = sweep.runs
    
    # Extract run data
    data = []
    for run in runs:
        run_data = {
            'run_id': run.id,
            'state': run.state,
            **run.config,  # Spread run configuration
            **{k: v for k, v in run.summary.items() if not k.startswith('_')}  # Spread metrics
        }
        data.append(run_data)
    
    return pd.DataFrame(data)

def analyze_parameter_importance(df: pd.DataFrame, target_metric: str = 'silhouette') -> Dict:
    """Analyze the importance of different parameters on the target metric.
    
    Args:
        df: DataFrame containing sweep results
        target_metric: Metric to analyze parameter importance for
        
    Returns:
        Dictionary containing parameter importance scores
    """
    importance_scores = {}
    
    # For each parameter
    for col in df.columns:
        if col in ['run_id', 'state', target_metric]:
            continue
            
        # Skip if parameter has only one value
        if len(df[col].unique()) <= 1:
            continue
            
        # Calculate correlation for numeric parameters
        if pd.api.types.is_numeric_dtype(df[col]):
            correlation = df[col].corr(df[target_metric])
            importance_scores[col] = abs(correlation)
        
        # For categorical parameters, use ANOVA F-statistic
        else:
            try:
                groups = [group[target_metric].values for name, group in df.groupby(col)]
                if len(groups) > 1:  # Need at least 2 groups
                    from scipy import stats
                    f_stat, _ = stats.f_oneway(*groups)
                    importance_scores[col] = f_stat
            except:
                continue
    
    # Normalize scores
    max_score = max(importance_scores.values())
    importance_scores = {k: v/max_score for k, v in importance_scores.items()}
    
    return dict(sorted(importance_scores.items(), key=lambda x: x[1], reverse=True))

def plot_parameter_effects(df: pd.DataFrame, 
                         target_metric: str = 'silhouette',
                         output_dir: str = 'sweep_analysis',
                         top_n: int = 5) -> str:
    """Create visualizations of parameter effects on the target metric.
    
    Args:
        df: DataFrame containing sweep results
        target_metric: Metric to analyze
        output_dir: Directory to save plots
        top_n: Number of top parameters to analyze
        
    Returns:
        Path to saved visualization file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get top N most important parameters
    importance_scores = analyze_parameter_importance(df, target_metric)
    top_params = list(importance_scores.items())[:top_n]
    
    # Create subplots
    fig = make_subplots(
        rows=len(top_params), 
        cols=1,
        subplot_titles=[f"{param} (importance: {score:.3f})" for param, score in top_params],
        vertical_spacing=0.1
    )
    
    # Plot each parameter's effect
    for i, (param, _) in enumerate(top_params, 1):
        if pd.api.types.is_numeric_dtype(df[param]):
            # For numeric parameters, use scatter plot
            fig.add_trace(
                go.Scatter(
                    x=df[param],
                    y=df[target_metric],
                    mode='markers',
                    name=param,
                    marker=dict(
                        size=8,
                        opacity=0.6,
                        color=df[target_metric],
                        colorscale='Viridis',
                        showscale=False
                    )
                ),
                row=i, col=1
            )
        else:
            # For categorical parameters, use box plot
            fig.add_trace(
                go.Box(
                    x=df[param],
                    y=df[target_metric],
                    name=param,
                    boxpoints='all',
                    jitter=0.3,
                    pointpos=-1.8
                ),
                row=i, col=1
            )
    
    # Update layout
    fig.update_layout(
        height=300 * len(top_params),
        width=1000,
        showlegend=False,
        title=f"Parameter Effects on {target_metric}"
    )
    
    # Save figure
    filepath = os.path.join(output_dir, f'parameter_effects_{target_metric}.html')
    fig.write_html(filepath)
    print(f"Saved parameter effects visualization to {filepath}")
    
    return filepath

def find_best_configs(df: pd.DataFrame, 
                     target_metric: str = 'silhouette',
                     n_configs: int = 5) -> pd.DataFrame:
    """Find the best performing configurations from the sweep.
    
    Args:
        df: DataFrame containing sweep results
        target_metric: Metric to optimize for
        n_configs: Number of top configurations to return
        
    Returns:
        DataFrame containing the top N configurations
    """
    # Sort by target metric
    best_runs = df.nlargest(n_configs, target_metric)
    
    # Select relevant columns
    param_cols = [col for col in df.columns if col not in ['run_id', 'state']]
    best_configs = best_runs[param_cols].copy()
    
    return best_configs

def analyze_sweep(sweep_id: str, 
                 target_metric: str = 'silhouette',
                 output_dir: str = 'sweep_analysis',
                 entity: Optional[str] = None,
                 project: Optional[str] = None) -> Dict:
    """Perform comprehensive analysis of a sweep.
    
    Args:
        sweep_id: The ID of the sweep to analyze
        target_metric: Metric to analyze
        output_dir: Directory to save analysis artifacts
        entity: Optional W&B entity name
        project: Optional W&B project name
        
    Returns:
        Dictionary containing analysis results
    """
    # Load sweep results
    df = load_sweep_results(sweep_id, entity, project)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze parameter importance
    importance_scores = analyze_parameter_importance(df, target_metric)
    
    # Create visualizations
    viz_path = plot_parameter_effects(df, target_metric, output_dir)
    
    # Find best configurations
    best_configs = find_best_configs(df, target_metric)
    
    # Save results
    results = {
        'parameter_importance': importance_scores,
        'best_configs': best_configs.to_dict('records'),
        'visualization_path': viz_path,
        'summary_stats': {
            'n_runs': len(df),
            f'best_{target_metric}': df[target_metric].max(),
            f'mean_{target_metric}': df[target_metric].mean(),
            f'std_{target_metric}': df[target_metric].std()
        }
    }
    
    # Save analysis results
    import json
    results_path = os.path.join(output_dir, 'sweep_analysis.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def print_sweep_summary(results: Dict):
    """Print a human-readable summary of sweep analysis results.
    
    Args:
        results: Dictionary containing analysis results from analyze_sweep
    """
    print("\n=== SWEEP ANALYSIS SUMMARY ===")
    
    # Print basic stats
    stats = results['summary_stats']
    print(f"\nRan {stats['n_runs']} experiments")
    print(f"Best score: {stats['best_silhouette']:.4f}")
    print(f"Mean score: {stats['mean_silhouette']:.4f} (±{stats['std_silhouette']:.4f})")
    
    # Print top parameters by importance
    print("\nParameter Importance:")
    for param, score in results['parameter_importance'].items():
        print(f"  {param}: {score:.3f}")
    
    # Print best configurations
    print("\nTop Configurations:")
    for i, config in enumerate(results['best_configs'], 1):
        print(f"\n{i}. Score: {config.get('silhouette', 'N/A'):.4f}")
        for param, value in config.items():
            if param != 'silhouette':
                print(f"   {param}: {value}")
    
    print(f"\nVisualization saved to: {results['visualization_path']}") 
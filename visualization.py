import plotly.graph_objects as go
from einops import rearrange

def create_figure(data, layout_params):
    """Create and return a plotly figure with common parameters."""
    fig = go.Figure(data=data)
    fig.update_layout(**layout_params)
    return fig

def visualize_training(stats):
    """Visualize training progress."""
    data = [
        go.Scatter(y=stats['loss'], mode='lines', name='Loss'),
        go.Scatter(y=stats['target_activation'], mode='lines', name='Target Activation')
    ]
    layout = dict(title="Training Progress", xaxis_title="Step", yaxis_title="Value")
    create_figure(data, layout).show()

def visualize_feature_activations(activations, feature_id):
    """Visualize activations for a specific feature."""
    sorted_acts = sorted(activations, key=lambda x: x[0])
    data = [go.Bar(x=[pos for pos, _ in sorted_acts], 
                   y=[val for _, val in sorted_acts], 
                   name=f'Feature {feature_id}')]
    layout = dict(title=f"Activations for Feature {feature_id}",
                 xaxis_title="Position", yaxis_title="Activation Value")
    create_figure(data, layout).show()

def visualize_multiple_features(results_dict, top_n=5):
    """Visualize activations for multiple features."""
    features = list(results_dict.items())[:top_n]
    first_feature_id = features[0][0]
    
    # Create traces for each feature
    data = []
    buttons = []
    for i, (feature_id, result) in enumerate(features):
        sorted_acts = sorted(result['activations'], key=lambda x: x[0])
        data.append(go.Bar(
            x=[pos for pos, _ in sorted_acts],
            y=[val for _, val in sorted_acts],
            name=f'Feature {feature_id}',
            visible=(i == 0)
        ))
        
        # Create visibility settings for dropdown
        visibility = [i == j for j in range(len(features))]
        buttons.append(dict(
            method="update",
            args=[{"visible": visibility}, {"title": f"Activations for Feature {feature_id}"}],
            label=f"Feature {feature_id}"
        ))
    
    layout = dict(
        title=f"Activations for Feature {first_feature_id}",
        xaxis_title="Position",
        yaxis_title="Activation Value",
        updatemenus=[dict(
            active=0, buttons=buttons,
            direction="down", pad={"r": 10, "t": 10},
            showactive=True, x=0.1, xanchor="left",
            y=1.15, yanchor="top"
        )]
    )
    create_figure(data, layout).show()

def visualize_token_heatmap(results, feature_id):
    """Visualize token activations as a heatmap."""
    result = results[feature_id]
    activations = [act for _, act in result['activations']]
    
    data = [go.Heatmap(
        z=[activations],
        x=list(range(len(activations))),
        y=['Activation'],
        colorscale='Viridis',
        showscale=True
    )]
    
    # Add token annotations
    token_positions = {pos: token for pos, act in result['activations'] 
                      for token, _ in result['high_act_tokens'] if act > 0}
    
    layout = dict(
        title=f"Token Activations for Feature {feature_id}",
        xaxis_title="Position",
        xaxis_showticklabels=False,
        yaxis_showticklabels=False,
        height=300,
        annotations=[
            dict(x=i, y=0, text=token_positions.get(pos, ""),
                 showarrow=False, textangle=-90, yshift=10)
            for i, (pos, _) in enumerate(result['activations'])
            if pos in token_positions
        ]
    )
    create_figure(data, layout).show()

def visualize_feature_comparison(results, feature_ids):
    """Compare activations across multiple features."""
    if len(feature_ids) < 2:
        return
    
    data = []
    for feature_id in feature_ids:
        if feature_id not in results:
            continue
        sorted_acts = sorted(results[feature_id]['activations'], key=lambda x: x[0])
        data.append(go.Scatter(
            x=[pos for pos, _ in sorted_acts],
            y=[val for _, val in sorted_acts],
            mode='lines+markers',
            name=f'Feature {feature_id}'
        ))
    
    layout = dict(
        title="Feature Activation Comparison",
        xaxis_title="Position",
        yaxis_title="Activation Value"
    )
    create_figure(data, layout).show() 
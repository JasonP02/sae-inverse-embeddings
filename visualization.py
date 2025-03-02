import plotly.graph_objects as go
from einops import rearrange

def visualize_training(stats):
    """Visualize training progress."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=stats['loss'], mode='lines', name='Loss'))
    fig.add_trace(go.Scatter(y=stats['target_activation'], mode='lines', name='Target Activation'))
    fig.update_layout(title="Training Progress", xaxis_title="Step", yaxis_title="Value")
    fig.show()

def visualize_feature_activations(activations, feature_id):
    """Visualize activations for a specific feature."""
    # Sort activations by position
    sorted_acts = sorted(activations, key=lambda x: x[0])
    positions = [pos for pos, _ in sorted_acts]
    values = [val for _, val in sorted_acts]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=positions, y=values, name=f'Feature {feature_id}'))
    fig.update_layout(
        title=f"Activations for Feature {feature_id}",
        xaxis_title="Position",
        yaxis_title="Activation Value"
    )
    fig.show()

def visualize_multiple_features(results_dict, top_n=5):
    """Visualize activations for multiple features."""
    fig = go.Figure()
    
    for feature_id, result in list(results_dict.items())[:top_n]:
        # Get activations sorted by position
        activations = result['activations']
        sorted_acts = sorted(activations, key=lambda x: x[0])
        positions = [pos for pos, _ in sorted_acts]
        values = [val for _, val in sorted_acts]
        
        fig.add_trace(go.Bar(
            x=positions, 
            y=values,
            name=f'Feature {feature_id}',
            visible=(feature_id == list(results_dict.keys())[0])  # Only first feature visible initially
        ))
    
    # Create dropdown menu for feature selection
    buttons = []
    for i, feature_id in enumerate(list(results_dict.keys())[:top_n]):
        visibility = [i == j for j in range(min(top_n, len(results_dict)))]
        buttons.append(
            dict(
                method="update",
                args=[{"visible": visibility},
                      {"title": f"Activations for Feature {feature_id}"}],
                label=f"Feature {feature_id}"
            )
        )
    
    fig.update_layout(
        updatemenus=[dict(
            active=0,
            buttons=buttons,
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.1,
            xanchor="left",
            y=1.15,
            yanchor="top"
        )],
        title=f"Activations for Feature {list(results_dict.keys())[0]}",
        xaxis_title="Position",
        yaxis_title="Activation Value"
    )
    
    fig.show()

def visualize_token_heatmap(results, feature_id):
    """Visualize token activations as a heatmap."""
    result = results[feature_id]
    # Use high_act_tokens instead of tokens
    tokens = [t for t, _ in result['high_act_tokens']]
    activations = [act for _, act in result['activations']]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=[activations],
        x=list(range(len(activations))),
        y=['Activation'],
        colorscale='Viridis',
        showscale=True
    ))
    
    # Add token labels for positions with tokens
    token_positions = {pos: token for pos, act in result['activations'] 
                      for token, _ in result['high_act_tokens'] if act > 0}
    
    for i, (pos, _) in enumerate(result['activations']):
        token = token_positions.get(pos, "")
        if token:
            fig.add_annotation(
                x=i,
                y=0,
                text=token,
                showarrow=False,
                textangle=-90,
                yshift=10
            )
    
    fig.update_layout(
        title=f"Token Activations for Feature {feature_id}",
        xaxis_title="Position",
        xaxis_showticklabels=False,
        yaxis_showticklabels=False,
        height=300
    )
    
    fig.show()

def visualize_feature_comparison(results, feature_ids):
    """Compare activations across multiple features."""
    if not feature_ids or len(feature_ids) < 2:
        print("Need at least 2 feature IDs to compare")
        return
    
    fig = go.Figure()
    
    for feature_id in feature_ids:
        if feature_id not in results:
            print(f"Feature {feature_id} not found in results")
            continue
            
        result = results[feature_id]
        activations = result['activations']
        sorted_acts = sorted(activations, key=lambda x: x[0])
        positions = [pos for pos, _ in sorted_acts]
        values = [val for _, val in sorted_acts]
        
        fig.add_trace(go.Scatter(
            x=positions,
            y=values,
            mode='lines+markers',
            name=f'Feature {feature_id}'
        ))
    
    fig.update_layout(
        title="Feature Activation Comparison",
        xaxis_title="Position",
        yaxis_title="Activation Value"
    )
    
    fig.show() 
import plotly.graph_objects as go
import numpy as np

from .data.schemas import BatchSample
from .model import TrackPrediction


def visualize_track_predictions(
    batch: BatchSample,
    output: TrackPrediction,
    track_idx: int = 0
) -> go.Figure:
    """
    Create an interactive 3D visualization of track predictions using Plotly.

    Parameters:
    - batch (BatchSample): The batch of input data containing hits and targets.
    - output (TrackPrediction): The model's predicted coordinates and radii.
    - track_idx (int): The index of the track to visualize within the batch.

    Returns:
    - go.Figure: A Plotly figure object containing the 3D visualization.
    """
    input_length = batch['input_lengths'][track_idx]

    # Create figure
    fig = go.Figure()

    # Add input hits
    fig.add_trace(go.Scatter3d(
        x=batch['inputs'][track_idx, :input_length, 0],
        y=batch['inputs'][track_idx, :input_length, 1],
        z=batch['inputs'][track_idx, :input_length, 2],
        mode='markers',
        name='Input Hits',
        marker=dict(
            size=8,
            color='#8884d8',
        )
    ))

    # Add target hits
    fig.add_trace(go.Scatter3d(
        x=batch['targets'][track_idx, :input_length, 0],
        y=batch['targets'][track_idx, :input_length, 1],
        z=batch['targets'][track_idx, :input_length, 2],
        mode='markers',
        name='Target Hits',
        marker=dict(
            size=6,
            color='#82ca9d',
        )
    ))

    # Add t1 predictions with spheres
    for i in range(input_length):
        # Prediction point
        fig.add_trace(go.Scatter3d(
            x=[output['coords_t1'][track_idx, i, 0]],
            y=[output['coords_t1'][track_idx, i, 1]],
            z=[output['coords_t1'][track_idx, i, 2]],
            mode='markers',
            name=f't1 Prediction {i+1}' if i == 0 else None,
            showlegend=i == 0,
            marker=dict(
                size=6,
                color='#ff7300',
            )
        ))

        # Prediction sphere
        radius = output['radius_t1'][track_idx, i, 0]
        u = np.linspace(0, 2*np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = radius * np.outer(np.cos(u), np.sin(v)) + \
            output['coords_t1'][track_idx, i, 0]
        y = radius * np.outer(np.sin(u), np.sin(v)) + \
            output['coords_t1'][track_idx, i, 1]
        z = radius * np.outer(np.ones(20), np.cos(v)) + \
            output['coords_t1'][track_idx, i, 2]

        fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            name=f't1 Sphere {i+1}',
            showscale=False,
            opacity=0.2,
            colorscale=[[0, '#ff7300'], [1, '#ff7300']],
            showlegend=False
        ))

    # Add t2 predictions with spheres
    for i in range(input_length-1):
        # Prediction point
        fig.add_trace(go.Scatter3d(
            x=[output['coords_t2'][track_idx, i, 0]],
            y=[output['coords_t2'][track_idx, i, 1]],
            z=[output['coords_t2'][track_idx, i, 2]],
            mode='markers',
            name=f't2 Prediction {i+1}' if i == 0 else None,
            showlegend=i == 0,
            marker=dict(
                size=6,
                color='#ff0000',
            )
        ))

        # Prediction sphere
        radius = output['radius_t2'][track_idx, i, 0]
        x = radius * np.outer(np.cos(u), np.sin(v)) + \
            output['coords_t2'][track_idx, i, 0]
        y = radius * np.outer(np.sin(u), np.sin(v)) + \
            output['coords_t2'][track_idx, i, 1]
        z = radius * np.outer(np.ones(20), np.cos(v)) + \
            output['coords_t2'][track_idx, i, 2]

        fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            name=f't2 Sphere {i+1}',
            showscale=False,
            opacity=0.2,
            colorscale=[[0, '#ff0000'], [1, '#ff0000']],
            showlegend=False
        ))

    # Update layout
    fig.update_layout(
        title='Track Visualization',
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data',  # Changed from 'cube' to 'data'
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    return fig

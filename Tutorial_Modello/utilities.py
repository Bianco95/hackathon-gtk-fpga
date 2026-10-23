"""Helper functions for Tutorial_Transformer.ipynb.

Long, non-didactic code (plots, metric tables) lives here so that the notebook
stays focused on the pipeline. The tutorial does not import anything from src/.
"""
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.colors


def plot_3d_interactive_clean(pred_tracks, gt_tracks, false_positive_tracks, data, save_path='gtk_3d.html'):
    """
    Fully interactive 3D view of one event (zoom, rotation, pan) with Plotly.
    Same function as in src/Notebooks/Paper_Figures.ipynb.

    pred_tracks:           predicted tracks (lists of hit indices)
    gt_tracks:             true tracks (not drawn: the truth is shown by the colour of the hits)
    false_positive_tracks: predicted tracks that are not true tracks, drawn in red
    data:                 hit table of the event (cl_x, cl_y, GTKstation, cl_kinepart)
    """
    # 1. One colour per true particle (kinepart)
    unique_kineparts = sorted([k for k in data['cl_kinepart'].unique() if k != -1])
    palette = plotly.colors.qualitative.Dark24
    kinepart_colors = {kp: palette[i % len(palette)] for i, kp in enumerate(unique_kineparts)}
    kinepart_colors[-1] = 'rgb(128, 128, 128)'  # grey for noise hits

    # Coordinates from the DataFrame
    x = data['cl_x'].to_numpy()
    y = data['cl_y'].to_numpy()
    z_station = data['GTKstation'].to_numpy().astype(float)
    kineparts = data['cl_kinepart'].to_numpy()

    fig = go.Figure()

    # 2. The GTK stations (background planes)
    x_lim, y_lim = (-30.4, 30.4), (-13.5, 13.5)
    X_plane = [x_lim[0], x_lim[1], x_lim[1], x_lim[0]]
    Z_plane = [y_lim[0], y_lim[0], y_lim[1], y_lim[1]]

    for s in sorted(np.unique(z_station)):
        # Surface of the sensor
        fig.add_trace(go.Mesh3d(
            x=X_plane, y=[s] * 4, z=Z_plane,
            i=[0, 0], j=[1, 2], k=[2, 3],
            opacity=0.2, color='gray', showlegend=False, hoverinfo='skip'
        ))
        # Border of the station
        fig.add_trace(go.Scatter3d(
            x=[X_plane[0], X_plane[1], X_plane[2], X_plane[3], X_plane[0]],
            y=[s, s, s, s, s],
            z=[Z_plane[0], Z_plane[1], Z_plane[2], Z_plane[3], Z_plane[0]],
            mode='lines', line=dict(color='black', width=2), showlegend=False, hoverinfo='skip'
        ))

    # 3. The hits, coloured by true particle
    for kp in unique_kineparts:
        mask = kineparts == kp
        color = kinepart_colors.get(kp, 'black')
        fig.add_trace(go.Scatter3d(
            x=x[mask], y=z_station[mask], z=y[mask],
            mode='markers',
            marker=dict(size=6, color=color, line=dict(color='black', width=1)),
            name=f"Hits Kinepart {int(kp)}",
            legendgroup=f"kp_{kp}"
        ))

    # 4. The tracks (lines connecting the hits)
    # Correct predicted tracks -> black solid lines
    # (the false positives are skipped here: they are drawn in red below, not twice)
    false_positives = {tuple(track) for track in false_positive_tracks}
    correct_tracks = [track for track in pred_tracks if tuple(track) not in false_positives]
    for i, track in enumerate(correct_tracks):
        idx = np.asarray(track, dtype=int)
        if idx.size == 0: continue
        fig.add_trace(go.Scatter3d(
            x=x[idx], y=z_station[idx], z=y[idx],
            mode='lines', line=dict(color='black', width=3),
            name="Predicted Track" if i == 0 else "",
            showlegend=True if i == 0 else False,
            legendgroup="pred_tracks"
        ))

    # False positive tracks -> red solid lines, thicker
    for i, track in enumerate(false_positive_tracks):
        idx = np.asarray(track, dtype=int)
        if idx.size == 0: continue
        fig.add_trace(go.Scatter3d(
            x=x[idx], y=z_station[idx], z=y[idx],
            mode='lines', line=dict(color='red', width=6),
            name="False Positive" if i == 0 else "",
            showlegend=True if i == 0 else False,
            legendgroup="fp_tracks"
        ))

    # Layout of the interactive scene
    fig.update_layout(
        title="GTK Reconstruction",
        legend=dict(title_text='Legend', x=0.85, y=0.9), font=dict(size=15),
        scene=dict(
            xaxis=dict(title='X [mm]', range=[-45, 45]),
            yaxis=dict(
                title='',
                tickmode='array', tickvals=[0, 1, 2, 3],
                ticktext=['GTK 0', 'GTK 1', 'GTK 2', 'GTK 3'],
                range=[-0.5, 3.5]
            ),
            zaxis=dict(title='Y [mm]', range=[-35, 35]),

            # Camera position: move `eye` to change the point of view
            camera=dict(
                eye=dict(x=1.0, y=1.5, z=0.5),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)          # the Z axis of the plot points up
            ),

            aspectmode='manual',
            aspectratio=dict(x=1, y=1.5, z=1)
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )

    # Also saved as HTML, to open the figure outside the notebook
    pio.write_html(fig, file=save_path, auto_open=False, include_plotlyjs='cdn')

    # Only the figure is returned: Jupyter shows it once
    return fig


def plot_attention_maps(attention_map, df, title=""):
    """
    Attention weights of one event, one panel per layer and head.
    Same figure as in src/Notebooks/Paper_Figures.ipynb.

    attention_map: {layer: tensor (heads, N, N)}; row = hit that looks (query), column = hit looked at (key)
    df:            hit table of the event (GTKstation)
    """
    import matplotlib.pyplot as plt

    # Hits sorted by station, so that the GTK0/1/2/3 blocks are readable
    stations = df['GTKstation'].values.astype(int)
    order = np.argsort(stations, kind='stable')
    boundaries = np.flatnonzero(np.diff(stations[order])) + 0.5
    centers = [np.mean(np.flatnonzero(stations[order] == s)) for s in np.unique(stations)]

    num_layers, num_heads = len(attention_map), attention_map[0].shape[0]
    fig, axes = plt.subplots(num_layers, num_heads, figsize=(5 * num_heads, 4.5 * num_layers), squeeze=False)
    for layer in range(num_layers):
        for head in range(num_heads):
            w = attention_map[layer][head].numpy()[np.ix_(order, order)]
            ax = axes[layer, head]
            im = ax.imshow(w, cmap='viridis')
            for x in boundaries:
                ax.axvline(x, color='white', lw=0.8)
                ax.axhline(x, color='white', lw=0.8)
            ax.set_xticks(centers, [f'GTK{s}' for s in np.unique(stations)])
            ax.set_yticks(centers, [f'GTK{s}' for s in np.unique(stations)])
            ax.set_xlabel('Hit (key)')
            ax.set_ylabel('Hit (query)')
            ax.set_title(f'Layer {layer} - Head {head}')
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

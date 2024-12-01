import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import plotly.graph_objects as go

# Create a custom colormap
cmap_name = "reddish_brown_to_yellow"
rbrown_yellow_cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, [
    "#5E2C1E",  # Deep reddish-brown
    "#B17359",  # Medium reddish-brown
    "#EAE5BF"   # Pale yellow (center color)
])

rbrown_yellow_cmap_inv = mcolors.LinearSegmentedColormap.from_list(cmap_name, [
    "#EAE5BF",  # Pale yellow (center color)
    "#B17359",  # Medium reddish-brown
    "#5E2C1E"  # Deep reddish-brown
])

# Create a custom colormap with more brown tones
# brownish_cmap = mcolors.LinearSegmentedColormap.from_list("brownish",colors=[
#                                                             (0.4, 0.2, 0.1),  # Brown
#                                                             (0.8, 0.6, 0.4),  # Light brown
#                                                             (0.9, 0.8, 0.6),  # Beige
#                                                             (1, 1, 0.8),  # Yellowish
#                                                             (1, 1, 1),  # White
#                                                         ])


def simple_2d_plot(data, num_points):
    # Plotting the points
    plt.figure(figsize=(4, 4))
    plt.scatter(data[:, 0], data[:, 1], s=5, alpha=0.7)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f'{num_points} Points with in 2D Annular Region')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    # plt.grid(True)
    # set xlim and ylim
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.show()

def interactive_3d_plot(data_vae_3d, num_points):
    # Create a 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=data_vae_3d[:, 0],
        y=data_vae_3d[:, 1],
        z=data_vae_3d[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            opacity=0.7
        )
    )])
    # set xlim and ylim
    fig.update_layout(scene=dict(
        xaxis=dict(range=[-3, 3]),
        yaxis=dict(range=[-3, 3]),
        zaxis=dict(range=[-3, 3])
    ))
    # Set plot title and axis labels
    fig.update_layout(
        title=f'{num_points} Points in 3D Annular Region',
        scene=dict(
            xaxis_title='X-axis',
            yaxis_title='Y-axis',
            zaxis_title='Z-axis'
        )
    )
    # Show the plot
    fig.show()

def simple_3d_plot(data_vae_3d, num_points):
    # Plotting the points
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(data_vae_3d[:, 0], data_vae_3d[:, 1], data_vae_3d[:, 2], s=2, alpha=0.7)
    ax.set_title(f'{num_points} Points with in 3D Annular Region')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.grid(True)
    plt.show()
import numpy as np
import torch
from torch import nn


# Function to generate circular distribution with a hole
def generate_annular_points(num_points, r_inner, r_outer):
    points = []
    for _ in range(num_points):
        # Generate random radius within the annular region
        r = np.sqrt(np.random.uniform(r_inner**2, r_outer**2))
        # Generate random angle
        theta = np.random.uniform(0, 2 * np.pi)
        # Convert polar to Cartesian coordinates
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append((x, y))
    return torch.tensor(points, dtype=torch.float32)

# Function to generate spherical distribution with a hole in 3D
def generate_annular_points_3d(num_points, r_inner, r_outer):
    points = []
    for _ in range(num_points):
        # Generate random radius within the spherical shell
        r = np.cbrt(np.random.uniform(r_inner**3, r_outer**3))
        # Generate random angles for spherical coordinates
        theta = np.random.uniform(0, 2 * np.pi)  # Azimuthal angle
        phi = np.arccos(np.random.uniform(-1, 1))  # Polar angle
        # Convert spherical to Cartesian coordinates
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        points.append((x, y, z))
    return torch.tensor(points, dtype=torch.float16)

# Function to embed 2D points into a 100D space with nonlinearity
def embed_points_2d_to_100d(points_2d):
    # Example non-linear transformations
    def nonlinear_map(x, y):
        features = []
        # Polynomial terms
        features.append(x)
        features.append(y)
        features.append(x**2)
        features.append(y**2)
        #features.append(torch.sqrt_(torch.abs(x)))
        #features.append(torch.sqrt_(torch.abs(y)))
#        features.append(x * y)
        # Trigonometric terms
#        features.append(np.sin(x))
#        features.append(np.cos(y))
#        features.append(np.sin(x * y))
        # More complex interactions
#        features.append(np.sqrt(np.abs(x * y)))
#        features.append(np.exp(-x**2 - y**2))
        # Expand to 100 dimensions by repeating patterns
        while len(features) < 100:
            features.append(features[len(features) % len(features)])  # Repeat patterns
        return np.array(features[:100])  # Truncate or pad to exactly 100 features

    # Apply the transformation to each 2D point
    points_100d = np.array([nonlinear_map(x, y) for x, y in points_2d])
    return points_100d

# create a dataset and a dataloader in which x is the data and y is the data_vae
class DumbDataset(torch.utils.data.Dataset):
    def __init__(self, z, y):
        self.z = z
        self.y = y
    def __len__(self):
        return len(self.z)
    def __getitem__(self, idx):
        return self.z[idx], self.y[idx]

def get_decoder(z_dim=2):
    decoder_net = nn.Sequential(
        nn.Linear(z_dim, 10),
        nn.Tanh(),
        nn.Linear(10, 20),
        nn.Tanh(),
        nn.Linear(20, 50),
        nn.Tanh(),
        nn.Linear(50, 100),
        nn.Tanh(),
    )
    return decoder_net


def generate_grid(v=2, grid_size=100, device='cpu'):
    x_lin = torch.linspace(-v, v, grid_size).to(device)
    y_lin = torch.linspace(-v, v, grid_size).to(device)
    x_grid, y_grid = torch.meshgrid(x_lin, y_lin, indexing='ij')
    grid = torch.stack((x_grid, y_grid), dim=-1)
    return grid, x_lin.cpu(), y_lin.cpu()
import numpy as np
import torch
from stochman.manifold import EmbeddedManifold
from torch import nn
from torch.func import jacrev, jacfwd, vmap
from torch.distributions import MultivariateNormal


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
def embed_points_2d_to_higher_dim(points_2d, n_dim=100):
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
        while len(features) < n_dim:
            features.append(features[len(features) % len(features)])  # Repeat patterns
        return np.array(features[:n_dim])  # Truncate or pad to exactly 100 features

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

def get_decoder(z_dim=2, out_dim=100):
    decoder_net = nn.Sequential(
        nn.Linear(z_dim, 10),
        nn.Tanh(),
        nn.Linear(10, 20),
        nn.Tanh(),
        nn.Linear(20, 50),
        nn.Tanh(),
        nn.Linear(50, out_dim),
        nn.Tanh(),
    )
    return decoder_net


def generate_grid(bound=2, grid_size=100, device='cpu'):
    x_lin = torch.linspace(-bound, bound, grid_size).to(device)
    y_lin = torch.linspace(-bound, bound, grid_size).to(device)
    x_grid, y_grid = torch.meshgrid(x_lin, y_lin, indexing='ij')
    grid = torch.stack((x_grid, y_grid), dim=-1)
    return grid, x_lin.cpu(), y_lin.cpu()


class DensityMetric(nn.Module):
    def __init__(self, data, sigma):
        super(DensityMetric, self).__init__()
        self.data = data
        self.sigma = sigma

    def density(self, x):
        """
        Evaluate a kernel density estimate at x.

        Parameters:
        x: points where the density is evaluated. Dimensions (num_points)x(data_dim)
        """
        N, D = x.shape
        M, _ = self.data.shape
        sigma2 = self.sigma**2
        normalization = (2 * 3.14159)**(D/2) * self.sigma**D  # scalar
        distances = torch.cdist(x, self.data)  # NxM
        K = torch.exp(-0.5 * distances**2 / sigma2) / normalization  # NxM
        p = torch.sum(K, dim=1)  # N
        return p

    def curve_energy(self, C):
        """
        Compute the length of a curve

        Parameters:
        C: points along a curve. Dimensions (batch)x(num_points_along_curve)x(data_dim)
        """
        if C.dim() == 2:
            C = C.unsqueeze(0)
        B, lenT, D = C.shape
        CC = C.reshape(-1, D)
        p = self.density(CC)  # (B*lenT)
        metric = 1 / (p + 1e-4).reshape(B, lenT)  # (B)x(lenT)
        avg_metric = 0.5 * metric[:, 1:] + 0.5 * metric[:, :-1]  # (B)x(lenT-1)
        delta = C[:, 1:] - C[:, :-1]  # (B)x(lenT-1)x(D)
        energy = torch.sum(torch.sum(delta**2, dim=2) * avg_metric, dim=1)  # (B)
        return energy

class DoubleDecoder(nn.Module):
    def __init__(self, z_dim=2, out_dim=100, mean_value=0.0, std_value=1.0, device='cpu'):
        super(DoubleDecoder, self).__init__()
        self.z_dim = z_dim
        self.out_dim = out_dim
        self.mu = get_decoder()
        self.sigma = get_decoder()
        _mean = torch.full((out_dim,), mean_value, device=device)
        _std = torch.full((out_dim,), std_value, device=device)
        covariance_matrix = torch.diag(_std ** 2)  # Diagonal covariance matrix
        self.noise = MultivariateNormal(_mean, covariance_matrix)
        self.diagonalizer = torch.vmap(torch.diag)
        self.noise_scaler = 0.1

    def forward(self, x, randomness=False, return_sigma=False):
        if randomness:
            noise = self.noise.sample((x.shape[0],)).to(x.device) * self.noise_scaler
            diag_noise = self.diagonalizer(noise)
        else:
            diag_noise = torch.eye(self.out_dim).repeat(x.shape[0], 1, 1).to(x.device)

        I = torch.eye(self.out_dim).repeat(x.shape[0], 1, 1).to(x.device)
        A = torch.concat((I, diag_noise), dim=2)
        _sigma = self.sigma(x)
        _mu = self.mu(x)
        v = torch.concat((_mu, _sigma), dim=1)
        _y_hat = torch.matmul(A, v.unsqueeze(-1)).squeeze(-1)
        if return_sigma:
            return _y_hat, _sigma
        else:
            return _y_hat


class DoubleDecoderInducedManifold(EmbeddedManifold):
    def __init__(self, decoder, input_dim=2):
        self.decoder = decoder
        self.input_dim = input_dim

    def metric(self, points: torch.Tensor):
        _, J_mean, J_sigma = self.embed(points, jacobian=True)  # NxDx(d)
        M_m = torch.einsum("bji,bjk->bik", J_mean, J_mean)
        M_s = torch.einsum("bji,bjk->bik", J_sigma, J_sigma)
        return M_m + M_s

    def embed(self, c, jacobian=False):
        reshaped = False
        if c.dim() == 3:
            B, N, D = c.shape
            c = c.reshape(-1, self.input_dim)
            reshaped = True
        out = self.decoder(c)
        out = out.reshape(B, N, -1) if reshaped else out
        if jacobian:
            J_mean = vmap(jacfwd(self.decoder.mu))(c)
            J_sigma = vmap(jacfwd(self.decoder.sigma))(c)
            return out, J_mean, J_sigma
        else:
            return out
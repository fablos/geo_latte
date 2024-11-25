from torchvision import datasets, transforms
import torch.utils.data
import matplotlib.pyplot as plt

from stochman.geodesic import geodesic_minimizing_energy
from stochman.curves import CubicSpline

from utils import *

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


if __name__ == '__main__':
    toydata = np.load('data/toybanana.npy')
    data = torch.from_numpy(toydata).to(dtype=torch.float32)  # 992x2
    num_curves = 30
    N, D = data.shape
    curve_indices = torch.randint(data.shape[0], (num_curves, 2))
    dm = DensityMetric(data, 0.1)
    dm_energy = lambda model, curve: dm.curve_energy(curve.points())
    plt.scatter(data[:, 0], data[:, 1], s=1, alpha=0.5)
    for k in tqdm(range(num_curves), leave=False):
        c = Spline(begin=data[curve_indices[k, 0]], end=data[curve_indices[k, 1]], num_nodes=20)
        connecting_geodesic(dm, c, energy_fun=dm_energy, max_iter=200)
        c.plot()
    plt.show()
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.neighbors import NearestNeighbors
import tqdm
import networkx as nx
import numpy as np
import torch
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import tqdm

class RiemannianMetric:
    def __init__(self, x, z):
        """
        Initialize the RiemannianMetric object.
        x: Decoder output as a torch tensor.
        z: Latent input as a torch tensor.
        """
        self.x = x
        self.z = z

    def create_torch_graph(self):
        """
        Creates the metric tensor (J^T J, where J is the Jacobian of the decoder),
        which can be evaluated at any point in Z,
        and the magnification factor.
        """

        # Compute the Jacobian of x with respect to z
        output_dim = self.x.shape[1]
        batch_size = self.z.shape[0]
        jacobians = []

        for i in range(output_dim):
            grad_outputs = torch.zeros_like(self.x)
            grad_outputs[:, i] = 1.0
            J_i = torch.autograd.grad(self.x, self.z, grad_outputs=grad_outputs, retain_graph=True, create_graph=True)[0]
            jacobians.append(J_i)

        # Stack the Jacobian along the output dimension
        J = torch.stack(jacobians, dim=1)  # batch_size x output_dim x latent_dim
        self.J = J

        # Compute the metric tensor G = J^T @ J
        G = torch.einsum('bik,bjk->bij', J, J)
        self.G = G

        # Magnification factor
        MF = torch.sqrt(torch.linalg.det(G + 1e-6 * torch.eye(G.shape[-1], device=G.device)))
        self.MF = MF

    def riemannian_distance_along_line(self, z1, z2, n_steps):
        """
        Calculates the Riemannian distance between two near points in latent space on a straight line.
        """
        # Discretize the line integral
        t = torch.linspace(0, 1, n_steps, device=z1.device)
        dt = t[1] - t[0]
        the_line = torch.cat([(t_ * z1 + (1 - t_) * z2).unsqueeze(0) for t_ in t], dim=0)

        # Evaluate G along the line
        G_eval = []
        for z_point in the_line:
            z_point = z_point.unsqueeze(0)  # Add batch dimension
            x_eval = self.x  # Decoder output should be recomputed if z depends on x
            self.z = z_point.requires_grad_(True)
            self.create_torch_graph()
            G_eval.append(self.G.squeeze(0))  # Remove batch dimension

        G_eval = torch.stack(G_eval, dim=0)  # n_steps x latent_dim x latent_dim

        # Compute the integrand
        dz = (z1 - z2).unsqueeze(0)  # 1 x latent_dim
        integrands = [torch.sqrt((dz @ G @ dz.T).squeeze()).item() for G in G_eval]

        # Compute the integral
        L = dt * sum(integrands)

        return L


class RiemannianTree:
    """A class to construct a graph based on Riemannian distances."""

    def __init__(self, riemann_metric):
        """
        Initialize the RiemannianTree object.
        riemann_metric: An instance of the RiemannianMetric class.
        """
        self.riemann_metric = riemann_metric

    def create_riemannian_graph(self, z, n_steps, n_neighbors):
        """
        Creates a graph with nodes representing points in z and edges weighted by
        Riemannian distances.

        z: A numpy array of latent space points.
        n_steps: Number of steps for discretization of the integral.
        n_neighbors: Number of nearest neighbors for graph construction.
        """
        n_data = len(z)
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        knn.fit(z)

        G = nx.Graph()

        # Add nodes with attributes
        for i in range(n_data):
            n_attr = {f'z{k}': float(z[i, k]) for k in range(z.shape[1])}
            G.add_node(i, **n_attr)

        # Add edges based on Riemannian and Euclidean distances
        for i in tqdm.trange(n_data):
            distances, indices = knn.kneighbors(z[i:i + 1])
            distances = distances[0]
            indices = indices[0]

            for ix, dist in zip(indices, distances):
                if (i, ix) in G.edges or (ix, i) in G.edges or i == ix:
                    continue

                # Convert numpy arrays to PyTorch tensors
                z1 = torch.tensor(z[i:i + 1], dtype=torch.float32, requires_grad=True)
                z2 = torch.tensor(z[ix:ix + 1], dtype=torch.float32, requires_grad=True)

                # Compute Riemannian distance
                L_riemann = self.riemann_metric.riemannian_distance_along_line(z1, z2, n_steps=n_steps)

                # Record edge attributes
                edge_attr = {
                    'weight': float(1 / L_riemann),
                    'weight_euclidean': float(1 / dist),
                    'distance_riemann': float(L_riemann),
                    'distance_euclidean': float(dist)
                }
                G.add_edge(i, ix, **edge_attr)

        return G


# Define the neural network model
class SimpleModel(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_dim, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.layers(x)


# Main function
def main():
    latent_dim = 2
    output_dim = 1

    # Initialize the model
    model = SimpleModel(latent_dim, output_dim)
    model.eval()

    # Generate input data
    z = torch.FloatTensor(np.random.uniform(-50, 50, size=(1000, latent_dim)))
    with torch.no_grad():
        x_hat = model(z)

    # Plot the input and output
    plt.figure()
    plt.scatter(z[:, 0].numpy(), z[:, 1].numpy())
    plt.title("Input Data")
    plt.figure()
    plt.scatter(x_hat[:, 0].numpy(), x_hat[:, 0].numpy())
    plt.title("Output Data")

    # Initialize Riemannian Metric
    rmetric = RiemannianMetric(x=x_hat, z=z)
    rmetric.create_torch_graph()

    # Compute magnification factor
    mf = []
    with torch.no_grad():
        for _z in z:
            mf.append(rmetric.MF.item())
    mf = np.array(mf)

    plt.figure()
    plt.scatter(z[:, 0].numpy(), z[:, 1].numpy(), c=mf)
    plt.title("Magnification Factor")

    # Example points for Riemannian distance
    z1 = torch.FloatTensor([[1, 10]])
    z2 = torch.FloatTensor([[10, 2]])

    # Uncomment to test Riemannian distance with different steps
    # for steps in [100, 1000, 10000, 100000]:
    #     q = rmetric.riemannian_distance_along_line(z1, z2, n_steps=steps)
    #     print(q)

    # Generate Swiss Roll data
    z, _ = make_swiss_roll(n_samples=1000, noise=0.5)
    z = z[:, [0, 2]]
    #z = torch.FloatTensor(np.random.uniform(-50, 50, size=(1000, latent_dim)))

    with torch.no_grad():
        x_hat = model(z)

    plt.figure()
    plt.scatter(x_hat[:, 0].numpy(), x_hat[:, 0].numpy())
    plt.title("Model Output from Swiss Roll Data")

    # Construct Riemannian Tree
    rTree = RiemannianTree(rmetric)
    G = rTree.create_riemannian_graph(z.numpy(), n_steps=1000, n_neighbors=10)
    # Use G for shortest path or other graph operations


if __name__ == "__main__":
    main()
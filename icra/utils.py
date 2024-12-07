import numpy as np
import torch
import torch.nn as nn
import torch.distributions as td
from stochman import CubicSpline
from torch.distributions.kl import kl_divergence as KL
from tqdm import tqdm
import matplotlib.pyplot as plt

# I can use spline from the StochMan library
class Poly2(nn.Module):
    def __init__(self, x0, x1, N, device='cpu'):
        """
        Represent the second-order polynomial
          t --> a*t**2 + b**t + c

        a, b, c: [torch.Tensor]
           Polynomial coefficients. These must have identical dimensionality
        """
        super(Poly2, self).__init__()
        self.x0 = x0.reshape(1, -1)  # 1xD
        self.x1 = x1.reshape(1, -1)  # 1xD
        self.N = N
        self.device = device

        self.w = nn.Parameter(torch.zeros_like(self.x0, device=device))  # 1xD

    def points(self):
        return self.forward(torch.linspace(0, 1, self.N, device=self.device, dtype=torch.float32))

    def forward(self, t):
        _t = t.reshape(-1, 1)  # Tx1
        line = (1 - _t) @ self.x0 + _t @ self.x1  # TxD
        a = -self.w  # 1xD
        b = self.w  # 1xD
        poly = ( _t**2 ) @ a + _t @ b  # TxD
        return line + poly  # TxD

    def plot(self, *args, **kwargs):
        c = self.points().detach().cpu().numpy()
        plt.plot(c[:, 0], c[:, 1], *args, **kwargs)


class Spline(CubicSpline):
    # write the init function passing all the arguments to the parent class
    def __init__(self, *args, **kwargs):
        super(Spline, self).__init__(*args, **kwargs)

    def points(self):
        return self.forward(torch.linspace(0, 1, self._num_nodes, device=self.device))


def connecting_geodesic(model, curve, energy_fun, max_iter=200, lr=0.1):
    opt = torch.optim.RMSprop(curve.parameters(), lr=lr)
    def closure():
        opt.zero_grad()
        _energy = energy_fun(model, curve)
        _energy.backward()
        return _energy

    for _ in range(max_iter):
        opt.zero_grad()
        __energy = opt.step(closure)
    return __energy


## VAE Part

def mnist_subsample(data, targets, num_data, num_classes):
    idx = targets < num_classes
    new_data = data[idx][:num_data].unsqueeze(1).to(torch.float32) / 255
    new_targets = targets[idx][:num_data]

    return torch.utils.data.TensorDataset(new_data, new_targets)


def get_encoder(z_dim):
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, stride=2, padding=1),
        nn.Softplus(),
        nn.Conv2d(16, 32, 3, stride=2, padding=1),
        nn.Flatten(),
        nn.Linear(1568, 2*z_dim),
    )

def get_decoder(z_dim):
    decoder_net = nn.Sequential(
        nn.Linear(z_dim, 512),
        nn.Unflatten(-1, (32, 4, 4)),
        nn.Softplus(),
        nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=0),
        nn.Softplus(),
        nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
        nn.Softplus(),
        nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
    )
    return decoder_net


class GaussianPrior(nn.Module):
    def __init__(self, z_dim):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        z_dim: [int]
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.z_dim = z_dim
        self.mean = nn.Parameter(torch.zeros(self.z_dim), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.z_dim), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net
        self.std = nn.Parameter(torch.ones(28, 28)*0.5, requires_grad=True)

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor]
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        return td.Independent(td.ContinuousBernoulli(logits=logits), 3)


def train(models, optimizers, data_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """
    num_ensembles = len(models)
    for m in range(num_ensembles):
        models[m].train()
        models[m].to(device)
    num_steps = num_ensembles*len(data_loader)*epochs
    epoch = 0

    def noise(x, std=0.1):  # std=0.05
        eps = std * torch.randn_like(x)
        return torch.clamp(x + eps, min=0.0, max=1.0)


    with tqdm(range(num_steps)) as pbar:
        for step in pbar:
            x = next(iter(data_loader))[0]
            x = noise(x.to(device))
            ensemble_idx = torch.randint(num_ensembles, (1,))
            model = models[ensemble_idx]
            optimizer = optimizers[ensemble_idx]
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()

            # Report
            if step % 5 ==0 :
                loss = loss.detach().cpu()
                pbar.set_description(f"epoch={epoch}, step={step}, loss={loss:.1f}")

            if (step+1) % len(data_loader) == 0:
                epoch += 1


class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """
    def __init__(self, prior, decoder, encoder):
        """
        Parameters:
        prior: [torch.nn.Module]
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """

        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(x)
        z = q.rsample()
        elbo = torch.mean(self.decoder(z).log_prob(x) - q.log_prob(z) + self.prior().log_prob(z), dim=0)
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.

        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()

    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)

    def curve_energy(self, C):
        """
        Compute the length of a curve when passed through the decoder mean.

        Parameters:
        C: points along a curve. Dimensions (num_points_along_curve)x(latent_dim)
        """
        lenT, latent_dim = C.shape
        ambient_C = self.decoder(C).mean  # |T| x (data_shape)

        delta = (ambient_C[1:] - ambient_C[:-1])  # (|T|-1) x (data_shape)
        retval = torch.sum(delta.reshape(lenT-1, -1)**2, dim=1).sum(dim=0)  # scalar
        return retval


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
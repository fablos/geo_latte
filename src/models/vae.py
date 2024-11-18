import torch
import torch.nn as nn

class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """
    def __init__(self, prior, decoders, encoder, num_ensembles):
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
        self.num_ensembles = num_ensembles
        self.decoders = decoders
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
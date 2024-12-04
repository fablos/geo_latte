import math
import torch
import torch.nn.functional as F

"""
Train loop for any model, given two view x_a and x_b
i) compute embeddings z_a and z_b
ii) compute the ssl-loss between z_a and z_b, it is basically the recontruction loss
iii) compute the max_ent_criterion for z_a and z_b and sum up the losses
"""

def off_diagonal(cov_x):
    assert cov_x.ndim == 2
    return cov_x - torch.diag(cov_x.diagonal())

def batched_off_diagonal(cov_x):
    assert cov_x.ndim == 3
    # Subtract the diagonal for each covariance matrix in the batch
    return cov_x - torch.diagonal(cov_x, dim1=-2, dim2=-1).diag_embed()

def max_ent_criterion(x, type):
    if type == 'hypercube':
        # apply the sigmoid transformation
        x_hyper = torch.sigmoid(x)
    elif type == 'hypersphere':
        # apply the CDF of 0-mean, 1 variance gaussian
        x_hyper = 0.5 * (1 + torch.erf(x / math.sqrt(2)))
    ent_loss = m_spacings_estimator(x_hyper)
    cov_loss = sample_cov_estimator(x_hyper)
    return (ent_loss, cov_loss)

def m_spacings_estimator(x):
    n = x.shape[0] # batch size
    m = round(math.sqrt(n)) # window size
    eps = 1e-7 # small constant to avoid underflow
    x, _ = torch.sort(x, dim=0) # order statistics
    x = x[m:] - x[:n - m] # m-spaced differences
    x = x * (n + 1) / m
    marginal_ents = torch.log(x + eps).sum(dim=0) / (n - m)
    return marginal_ents.mean()

def sample_cov_estimator(x):
    n, d = x.shape
    x = x - x.mean(dim=0) # mean subtraction
    cov_x = (x.T @ x) / (n - 1) # sample covariance matrix
    cov_loss = off_diagonal(cov_x).pow(2).sum().div(d)
    return cov_loss

 def _get_vicreg_loss(self, z_a, z_b, batch_idx):
        """
            invariance_loss_weight: float = 25.0
            variance_loss_weight: float = 25.0
            covariance_loss_weight: float = 1.0
            variance_loss_epsilon: float = 1e-04
        :param self:
        :param z_a:
        :param z_b:
        :param batch_idx:
        :return:
        """
        assert z_a.shape == z_b.shape and len(z_a.shape) == 2

        # invariance loss
        loss_inv = F.mse_loss(z_a, z_b)

        # variance loss
        std_z_a = torch.sqrt(z_a.var(dim=0) + self.hparams.variance_loss_epsilon)
        std_z_b = torch.sqrt(z_b.var(dim=0) + self.hparams.variance_loss_epsilon)
        loss_v_a = torch.mean(F.relu(1 - std_z_a))
        loss_v_b = torch.mean(F.relu(1 - std_z_b))
        loss_var = loss_v_a + loss_v_b

        # covariance loss
        N, D = z_a.shape
        z_a = z_a - z_a.mean(dim=0)
        z_b = z_b - z_b.mean(dim=0)
        cov_z_a = ((z_a.T @ z_a) / (N - 1)).square()  # DxD
        cov_z_b = ((z_b.T @ z_b) / (N - 1)).square()  # DxD
        loss_c_a = (cov_z_a.sum() - cov_z_a.diagonal().sum()) / D
        loss_c_b = (cov_z_b.sum() - cov_z_b.diagonal().sum()) / D
        loss_cov = loss_c_a + loss_c_b

        weighted_inv = loss_inv * self.hparams.invariance_loss_weight
        weighted_var = loss_var * self.hparams.variance_loss_weight
        weighted_cov = loss_cov * self.hparams.covariance_loss_weight

        loss = weighted_inv + weighted_var + weighted_cov

        return {
            "loss": loss,
            "loss_invariance": weighted_inv,
            "loss_variance": weighted_var,
            "loss_covariance": weighted_cov,
        }

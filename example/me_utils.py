import math
import torch

"""
Train loop for any model, given two view x_a and x_b
i) compute embeddings z_a and z_b
ii) compute the ssl-loss between z_a and z_b, it is basically the recontruction loss
iii) compute the max_ent_criterion for z_a and z_b and sum up the losses
"""

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
import torch
from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.transforms import normalize, unnormalize
from botorch.models.transforms import Standardize, Normalize

from botorch import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.acquisition.monte_carlo import qExpectedImprovement

def gen_initial_data(nn_scorer, n=50, z_dim=2, bounds=None, device='cpu', dtype=torch.float32):
    # generate training data
    train_x = unnormalize(torch.rand(n, z_dim, device=device, dtype=dtype), bounds=bounds)
    train_obj = nn_scorer(train_x).unsqueeze(-1)
    best_observed_value = train_obj.max().item()
    return train_x, train_obj, best_observed_value

def get_fitted_model(train_x, train_obj, bounds, state_dict=None):
    # initialize and fit model
    model = SingleTaskGP(train_X=normalize(train_x, bounds), train_Y=train_obj, outcome_transform=Standardize(m=1))
    if state_dict is not None:
        model.load_state_dict(state_dict)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    mll.to(train_x)
    fit_gpytorch_mll(mll)
    return model

def optimize_acqf_and_get_observation(acq_func, nn_scorer, train_x, bounds, z_dim=2, bs=8,
                                      n_restart=10, raw_samples=256, device='cpu', dtype=torch.float32):
    """Optimizes the acquisition function, and returns a
    new candidate and a noisy observation"""

    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=torch.stack(
            [torch.zeros(z_dim, dtype=dtype, device=device),
             torch.ones(z_dim, dtype=dtype, device=device),]
        ),
        q=bs,
        num_restarts=n_restart,
        raw_samples=raw_samples,
    )

    # observe new values
    new_x = unnormalize(candidates.detach(), bounds=bounds)
    new_obj = nn_scorer(train_x).unsqueeze(-1)
    return new_x, new_obj
import math

import torch
import torch.nn.functional as F
from torch import nn

PRIORS = ("gaussian", "logistic")


class Flow(nn.Module):
    def __init__(self, n, prior="gaussian"):
        super().__init__()
        if prior not in PRIORS:
            raise ValueError(f"prior должен быть одним из {PRIORS}, получено {prior!r}")
        self.n = n
        self.prior = prior
        self.log_scale = nn.Parameter(torch.zeros(n))

    @property
    def prior_std(self):
        return 1.0 if self.prior == "gaussian" else math.pi / math.sqrt(3)

    def prior_log_prob(self, z):
        if self.prior == "gaussian":
            return -0.5 * (z.square() + math.log(2 * math.pi)).sum(dim=1)
        return (-z - 2 * F.softplus(-z)).sum(dim=1)

    def log_prob(self, x):
        z, log_det = self(x)
        return self.prior_log_prob(z) + log_det

    @torch.no_grad()
    def sample(self, batch_size, temperature=1.0):
        device = self.log_scale.device
        if self.prior == "gaussian":
            z = torch.randn(batch_size, self.n, device=device)
        else:
            z = torch.logit(torch.rand(batch_size, self.n, device=device), eps=1e-6)
        return self.inverse(temperature * z)

    @torch.no_grad()
    def init_log_scale(self, x):
        self.log_scale.zero_()
        z, _ = self(x)
        self.log_scale.copy_(math.log(self.prior_std) - z.std(0).clamp_min(1e-3).log())

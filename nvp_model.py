import nice_model
import torch
import torch.nn as nn
from typing import Callable, List, Iterable, Tuple
import math

class Normalizer(nn.Module):
    def __init__(self, shape: Tuple[int], moment: float = 0.1, eps: float = 1e-3):
        super().__init__()
        self.eps = eps
        # self.register_buffer("eps", eps)
        self.m = moment
        # self.register_buffer("m", moment)

        # running statistics
        self.register_buffer("running_mean", torch.zeros(shape))
        self.register_buffer("running_var", torch.ones(shape))
        self.shape = shape
        self.size = shape[0] * shape[1] * shape[2]
        self.det = - self.eps * self.size / 2

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.training:
            mean = X.mean(dim=0)
            var = X.var(dim=0)

            self.running_mean.mul_(1 - self.m).add_(self.m * mean)
            self.running_var.mul_(1 - self.m).add_(self.m * var)

            self.det = - self.eps * self.size / 2 - self.running_var.sum() / 2

        return (X - self.running_mean) / (self.running_var + self.eps).sqrt()

    def inverse(self, Y: torch.Tensor) -> torch.Tensor:
        Y *= (self.running_var + self.eps).sqrt()
        Y += self.running_mean
        return Y

    def jacobian_det(self) -> float:
        return self.det

class AffineCoupling(nn.Module):
    def __init__(self,
                 masking: torch.Tensor,
                 s: nn.Module,
                 t: nn.Module):
        super().__init__()
        self.register_buffer("masking", masking)
        self.s = s
        self.t = t
        self.det = 0
        self.batch_norm = Normalizer(masking.shape)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        b = self.masking
        X_1 = X * b
        scale = (1 - b) * self.s(X_1)
        Y = X * torch.exp(scale) + (1 - b) * self.t(X_1)
        Y = self.batch_norm(Y)
        if self.training:
            self.det = scale.sum() + self.batch_norm.jacobian_det()
        return Y

    def inverse(self, Y: torch.Tensor) -> torch.Tensor:
        b = self.masking
        Y = self.batch_norm.inverse(Y)
        Y_1 = Y * b
        scale = (1 - b) * self.s(Y_1)
        X = (Y - (1 - b) * self.t(Y_1)) * torch.exp(-scale)
        return X

    def jacobian_det(self) -> float:
        return self.det

class ScaleLayer(nn.Module):
    def __init__(coupling_layers: nn.ModuleList, in_shape: Tuple[int], mod_shape: Tuple[int]):
        super().__init__()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        pass
    def inverse(self, Y: torch.Tensor) -> torch.Tensor:
        pass
    def jacobian_det(self) -> float:
        return 0

class NVP(nn.Module):
    def __init__(self, scale_layers: nn.ModuleList):
        super().__init__()
    def forward(self, X: torch.Tensor):
        pass
    def inverse(self, Y: torch.Tensor) -> torch.Tensor:
        pass
    def jacobian_det(self) -> float:
        return 0

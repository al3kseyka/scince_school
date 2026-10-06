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

def squeeze(x: torch.Tensor):
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // 2, 2, W // 2, 2)
    x = x.permute(0, 3, 5, 1, 2, 4)
    return x.reshape(B, 4 * C, H // 2, W // 2)

def unsqueeze(x: torch.Tensor):
    B, C4, H2, W2 = x.shape
    x = x.reshape(B, 2, 2, C4 // 4, H2, W2)
    x = x.permute(0, 3, 4, 1, 5, 2)
    return x.reshape(B, C4 // 4, H2 * 2, W2 * 2)

class Squizer(nn.Module):
    def __init__(self, reverse: bool):
        super().__init__()
        self.reverse = reverse

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if (self.reverse):
            return unsqueeze(X)
        return squeeze(X)

    def inverse(self, Y: torch.Tensor) -> torch.Tensor:
        if (self.reverse):
            return squeeze(Y)
        return unsqueeze(Y)

    def jacobian_det(self) -> float:
        return 0

def checkerboard(C: int, H: int, W: int, parity: bool):
    i = torch.arange(H).view(-1, 1)
    j = torch.arange(W).view(1, -1)
    return ((i + j) % 2 == parity).int().expand(C, H, W)

def chanelwise(C: int, H: int, W: int, parity: bool):
    a = torch.ones((C // 2, H, W), dtype=torch.int32)
    staking = [a, a * 0] if parity else [a * 0, a]
    return torch.stack(staking)

def squized_size(C: int, H: int, W: int):
    return (C * 4, H // 2, W // 2)

class ScaleLayer(nn.Module):
    def __init__(self, coupling_layers: nn.ModuleList, H: int, W: int):
        super().__init__()
        self.coupling_layers = coupling_layers
        self.H, self.W = H, W
        self.det = 0

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        patch = X[..., :self.H, :self.W]
        for model in self.coupling_layers:
            patch = model(patch)
        if self.training:
            self.det = 0
            for model in self.coupling_layers:
                self.det += model.jacobian_det()
        X[..., :self.H, :self.W] = patch
        return patch

    def inverse(self, Y: torch.Tensor) -> torch.Tensor:
        patch = Y[..., :self.H, :self.W]
        for model in reversed(self.coupling_layers):
            patch = model.inverse(patch)
        Y[..., :self.H, :self.W] = patch
        return Y

    def jacobian_det(self) -> float:
        return self.det

class NVP(nn.Module):
    def __init__(self, scale_layers: nn.ModuleList):
        super().__init__()
        self.det = 0
        self.scale_layers = scale_layers
    def forward(self, X: torch.Tensor):
        for scale in self.scale_layers:
            X = scale(X)
        if self.training:
            self.det = 0
            for model in self.scale_layers:
                self.det += model.jacobian_det()
        return X
    def inverse(self, Y: torch.Tensor) -> torch.Tensor:
        for scale in self.scale_layers:
            Y = scale(Y)    
        return Y

    def jacobian_det(self) -> float:
        return self.det

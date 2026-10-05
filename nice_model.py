import torch
import torch.nn as nn
from typing import Callable
import math

class Spliter:
    def __init__(self, even: bool, n: int):
        self.less = n // 2
        self.more = (n + 1) // 2
        self.even = even
        if not even:
            self.left = self.less
            self.right = self.more
        else:
            self.left = self.more
            self.right = self.less

    def split(self, batch: torch.Tensor):
        less_out = batch[:, :self.less]
        more_out = batch[:, self.less:]
        if not self.even:
            left_out = less_out
            right_out = more_out
        else:
            left_out = more_out
            right_out = less_out
        return left_out, right_out

    def merge(self, left_batch: torch.Tensor, right_batch: torch.Tensor):
        if not self.even:
            res = torch.cat([left_batch, right_batch], dim=1)
        else:
            res = torch.cat([right_batch, left_batch], dim=1)
        return res

class AdditiveCoupling(nn.Module):
    def __init__(self, spliter: Spliter, mid_model: nn.Module):
        super().__init__()
        self.mid_model = mid_model
        self.spliter = spliter

    def forward(self, x, inverse=False):
        a, b = self.spliter.split(x)
        b = b + (-1 if inverse else 1) * self.mid_model(a)
        return self.spliter.merge(a, b)


class NICE(nn.Module):
    def __init__(self, mid_model: Callable[[int, int], nn.Module],
                 width=2,
                 spliter_list=[Spliter(i%2, 2) for i in range(8)]):
        super().__init__()
        self.layers = nn.ModuleList([
            AdditiveCoupling(spliter, mid_model(spliter.left, spliter.right)) for spliter in spliter_list
        ])
        self.log_scale = nn.Parameter(torch.zeros(width))

    def forward(self, x):  # Данные -> латентное пространство.
        for layer in self.layers:
            x = layer(x)
        z = x * self.log_scale.exp()
        return z

    def inverse(self, z):  # Латентное пространство -> данные.
        x = z * (-self.log_scale).exp()
        for layer in reversed(self.layers):
            x = layer(x, inverse=True)
        return x

    def jacobian_det(self):
        return self.log_scale.sum()

def logprop(out: torch.Tensor):
    return -0.5 * (out.square() + math.log(2 * torch.pi)).sum(dim=1)

def loss_function(out: torch.Tensor, jacobian_add):
    out.flatten()
    sum_of_log = logprop(out) + jacobian_add
    return -sum_of_log.mean()

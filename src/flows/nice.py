import torch
from torch import nn

from .base import Flow


def mlp(n_in, n_out, hidden_features, hidden_layers):
    layers = [nn.Linear(n_in, hidden_features), nn.Tanh()]
    for _ in range(hidden_layers - 1):
        layers += [nn.Linear(hidden_features, hidden_features), nn.Tanh()]
    layers.append(nn.Linear(hidden_features, n_out))
    nn.init.zeros_(layers[-1].weight)
    nn.init.zeros_(layers[-1].bias)
    return nn.Sequential(*layers)


class AdditiveCoupling(nn.Module):
    def __init__(self, keep, n, hidden_features=100, hidden_layers=4):
        super().__init__()
        self.keep = keep
        n_from = (n + 1 - keep) // 2
        self.net = mlp(n_from, n - n_from, hidden_features, hidden_layers)

    def forward(self, x, inverse=False):
        y = x.clone()
        y[:, 1 - self.keep::2] += (-1 if inverse else 1) * self.net(x[:, self.keep::2])
        return y, x.new_zeros(len(x))


class AffineCoupling(nn.Module):
    def __init__(self, keep, n, hidden_features=256, hidden_layers=2):
        super().__init__()
        self.keep = keep
        n_from = (n + 1 - keep) // 2
        self.n_to = n - n_from
        self.net = mlp(n_from, 2 * self.n_to, hidden_features, hidden_layers)

    def forward(self, x, inverse=False):
        x_changed = x[:, 1 - self.keep::2]
        net_output = self.net(x[:, self.keep::2])
        log_scale = torch.tanh(net_output[:, :self.n_to])
        shift = net_output[:, self.n_to:]

        y = x.clone()
        if inverse:
            y[:, 1 - self.keep::2] = (x_changed - shift) * torch.exp(-log_scale)
            return y, -log_scale.sum(dim=1)

        y[:, 1 - self.keep::2] = x_changed * torch.exp(log_scale) + shift
        return y, log_scale.sum(dim=1)


COUPLINGS = {"additive": AdditiveCoupling, "affine": AffineCoupling}


class NICE(Flow):
    def __init__(self, n, layers=8, hidden_features=100, hidden_layers=4,
                 coupling="additive", prior="gaussian"):
        super().__init__(n, prior)
        coupling_cls = COUPLINGS[coupling]
        self.layers = nn.ModuleList([
            coupling_cls(i % 2, n, hidden_features, hidden_layers) for i in range(layers)
        ])

    def forward(self, x):
        log_det = 0
        for layer in self.layers:
            x, layer_log_det = layer(x)
            log_det = log_det + layer_log_det
        return x * self.log_scale.exp(), log_det + self.log_scale.sum()

    def inverse(self, z):
        x = z * (-self.log_scale).exp()
        for layer in reversed(self.layers):
            x, _ = layer(x, inverse=True)
        return x

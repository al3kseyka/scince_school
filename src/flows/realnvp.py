import math

import torch
from torch import nn

from .base import Flow


def squeeze(x):
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // 2, 2, W // 2, 2)
    x = x.permute(0, 3, 5, 1, 2, 4)
    return x.reshape(B, 4 * C, H // 2, W // 2)


def unsqueeze(x):
    B, C4, H2, W2 = x.shape
    x = x.reshape(B, 2, 2, C4 // 4, H2, W2)
    x = x.permute(0, 3, 4, 1, 5, 2)
    return x.reshape(B, C4 // 4, H2 * 2, W2 * 2)


class Squeeze(nn.Module):
    def forward(self, x, inverse=False):
        return (unsqueeze(x) if inverse else squeeze(x)), 0


class Split(nn.Module):
    pass


class ResBlock(nn.Module):
    def __init__(self, channels, batch_norm=True):
        super().__init__()
        if batch_norm:
            self.net = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(channels, channels, 3, padding=1),
            )

    def forward(self, x):
        return x + self.net(x)


def conv_net(c_in, c_out, hidden, n_blocks, batch_norm=True):
    layers = [nn.Conv2d(c_in, hidden, 3, padding=1), nn.ReLU()]
    for _ in range(n_blocks):
        layers.append(ResBlock(hidden, batch_norm))
        layers.append(nn.ReLU())
    layers.append(nn.Conv2d(hidden, c_out, 3, padding=1))
    bn = nn.BatchNorm2d(c_out)
    nn.init.zeros_(bn.weight)
    layers.append(bn)
    return nn.Sequential(*layers)


class CouplingNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))

    def forward(self, x, mask=1, inverse=False):
        if self.training:
            variance, mean = torch.var_mean(x * mask, dim=(0, 2, 3), unbiased=False)
            with torch.no_grad():
                self.running_mean.lerp_(mean.detach(), 0.1)
                self.running_var.lerp_(variance.detach(), 0.1)
        else:
            mean, variance = self.running_mean, self.running_var
        shift = mean[None, :, None, None] * mask
        log_scale = -0.5 * (variance + 1e-4).log()[None, :, None, None] * mask
        y = x * (-log_scale).exp() + shift if inverse else (x - shift) * log_scale.exp()
        logdet = log_scale.expand_as(x).flatten(1).sum(dim=1)
        return y, -logdet if inverse else logdet


def scaled_tanh(s, scale):
    return torch.tanh(s) if scale is None else scale * torch.tanh(s)


class CheckerCoupling(nn.Module):
    def __init__(self, c, h, w, parity, hidden, n_blocks, learn_scale=True, batch_norm=True):
        super().__init__()
        self.parity = parity
        self.net = conv_net(c, 2 * c, hidden, n_blocks, batch_norm)
        self.bn_out = CouplingNorm(c)
        self.scale = nn.Parameter(torch.ones(c, h, w)) if learn_scale else None

    def forward(self, x, inverse=False):
        H, W = x.shape[2], x.shape[3]
        i = torch.arange(H, device=x.device).view(-1, 1)
        j = torch.arange(W, device=x.device).view(1, -1)
        b = ((i + j) % 2 == self.parity).to(x.dtype)

        s, t = self.net(x * b).chunk(2, dim=1)
        s = scaled_tanh(s, self.scale) * (1 - b)
        t = t * (1 - b)
        logdet = s.flatten(1).sum(dim=1)

        if inverse:
            x, norm_ld = self.bn_out(x, mask=1 - b, inverse=True)
            return (x - t) * torch.exp(-s), norm_ld - logdet

        y, norm_ld = self.bn_out(x * torch.exp(s) + t, mask=1 - b)
        return y, logdet + norm_ld


class ChannelCoupling(nn.Module):
    def __init__(self, c, h, w, swap, hidden, n_blocks, learn_scale=True, batch_norm=True):
        super().__init__()
        self.swap = swap
        self.net = conv_net(c // 2, 2 * (c // 2), hidden, n_blocks, batch_norm)
        self.bn_out = CouplingNorm(c // 2)
        self.scale = nn.Parameter(torch.ones(c // 2, h, w)) if learn_scale else None

    def forward(self, x, inverse=False):
        a, b = x.chunk(2, dim=1)
        if self.swap:
            a, b = b, a

        s, t = self.net(a).chunk(2, dim=1)
        s = scaled_tanh(s, self.scale)
        logdet = s.flatten(1).sum(dim=1)

        if inverse:
            b, norm_ld = self.bn_out(b, inverse=True)
            b = (b - t) * torch.exp(-s)
            logdet = norm_ld - logdet
        else:
            b, norm_ld = self.bn_out(b * torch.exp(s) + t)
            logdet = logdet + norm_ld
        if self.swap:
            a, b = b, a
        return torch.cat([a, b], dim=1), logdet


class RealNVP(Flow):
    def __init__(self, c, h, w, hidden_channels=64, n_blocks=2, n_scales=4, max_hidden=256,
                 logit_alpha=0.05, learn_scale=True, batch_norm=True):
        super().__init__(c * h * w)
        self.logit_alpha = logit_alpha

        layers, z_shapes = [], []
        hid = hidden_channels
        options = {"learn_scale": learn_scale, "batch_norm": batch_norm}
        for _ in range(n_scales):
            layers += [CheckerCoupling(c, h, w, i % 2, hid, n_blocks, **options) for i in range(3)]
            layers.append(Squeeze())
            c, h, w = 4 * c, h // 2, w // 2
            layers += [ChannelCoupling(c, h, w, i % 2, hid, n_blocks, **options) for i in range(3)]
            layers.append(Split())
            c = c // 2
            z_shapes.append((c, h, w))
            hid = min(2 * hid, max_hidden)

        layers += [CheckerCoupling(c, h, w, i % 2, hid, n_blocks, **options) for i in range(4)]
        z_shapes.append((c, h, w))

        self.layers = nn.ModuleList(layers)
        self.z_shapes = z_shapes

    def to_logit(self, x):
        alpha = self.logit_alpha
        p = (alpha + (1 - alpha) * x).clamp(max=1 - 1e-6)
        log_p, log_1mp = p.log(), torch.log(1 - p)
        log_det = (math.log(1 - alpha) - log_p - log_1mp).flatten(1).sum(dim=1)
        return log_p - log_1mp, log_det

    def from_logit(self, y):
        return (torch.sigmoid(y) - self.logit_alpha) / (1 - self.logit_alpha)

    def forward(self, x):
        log_det = 0
        if self.logit_alpha is not None:
            x, log_det = self.to_logit(x)
        zs = []
        for layer in self.layers:
            if isinstance(layer, Split):
                z_out, x = x.chunk(2, dim=1)
                zs.append(z_out.flatten(1))
            else:
                x, layer_log_det = layer(x)
                log_det = log_det + layer_log_det
        zs.append(x.flatten(1))
        z = torch.cat(zs, dim=1) * self.log_scale.exp()
        return z, log_det + self.log_scale.sum()

    def inverse(self, z):
        z = z * (-self.log_scale).exp()
        parts = list(torch.split(z, [math.prod(s) for s in self.z_shapes], dim=1))
        x = parts.pop().reshape(-1, *self.z_shapes[-1])
        for layer in reversed(self.layers):
            if isinstance(layer, Split):
                shape = self.z_shapes[len(parts) - 1]
                x = torch.cat([parts.pop().reshape(-1, *shape), x], dim=1)
            else:
                x, _ = layer(x, inverse=True)
        return x if self.logit_alpha is None else self.from_logit(x)

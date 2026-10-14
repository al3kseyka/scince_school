# %pip install torch numpy matplotlib
import matplotlib.pyplot as plt
import math
import time
import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms

SEED, STEPS, BATCH_SIZE = 42, 4000, 256
torch.manual_seed(SEED)
torch.set_num_threads(2)  # Для маленьких MLP на CPU.

# размер картинок 32 на 32 на 3, вектор будет размера 3072
# предварительная нормализация данных
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])

dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

def data_loader():
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True)

loader = data_loader()


class AdditiveCoupling(nn.Module):
    def __init__(self, keep, width=64):
        super().__init__()
        self.keep = keep
        self.net = nn.Sequential(nn.Linear(1536, width), nn.Tanh(),
                                 nn.Linear(width, width), nn.Tanh(),
                                 nn.Linear(width, 1536))
        nn.init.zeros_(self.net[-1].weight)  # Начинаем с identity.
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, inverse=False):
        a = x[:, self.keep % 2::2]
        b = x[:, (self.keep + 1) % 2::2]

        b = b + (-1 if inverse else 1) * self.net(a)

        return torch.cat(
            [a, b] if self.keep == 0 else [b, a],
            dim=1
        )


class BatchNormFlow(nn.Module):
    def __init__(self, dim=3072, eps=1e-5, moment=0.1):
        super().__init__()
        self.eps = eps
        self.m = m

        # running statistics
        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_var", torch.ones(dim))

    def forward(self, x, inverse=False):

        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)

            # running average
            self.running_mean.mul_(1 - self.m).add_(self.m * mean)
            self.running_var.mul_(1 - self.m).add_(self.m * var)
        else:
            mean = self.running_mean
            var = self.running_var

        if not inverse:
            x = (x - mean) / torch.sqrt(var + self.eps)
            log_det = (-0.5 * torch.log(var + self.eps)).sum()

            return x, log_det.expand(x.shape[0])

        else:
            x = x * torch.sqrt(var + self.eps) + mean
            return x


class NICE(nn.Module):
    def __init__(self, n_layers=8, width=64):
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(n_layers):
            self.layers.append(AdditiveCoupling(i % 2, width))
            self.layers.append(BatchNormFlow(3072))

        self.log_scale = nn.Parameter(torch.zeros(3072))

    def forward(self, x):  # Данные -> латентное пространство.
        log_det_total = torch.zeros(
            x.shape[0],
            device=x.device
        )

        for layer in self.layers:
            if isinstance(layer, BatchNormFlow):
                x, log_det = layer(x)
                log_det_total += log_det
            else:
                x = layer(x)

        z = x * self.log_scale.exp()
        log_det_total += self.log_scale.sum()

        return z, log_det_total

    def inverse(self, z):
        x = z * (-self.log_scale).exp()

        for layer in reversed(self.layers):
            x = layer(x, inverse=True)

        return x

    def log_prob(self, x):
        z, log_det = self(x)

        log_base = -0.5 * (z.square() + math.log(2 * math.pi)).sum(dim=1)

        return log_base + log_det

    @torch.no_grad()
    def sample(self, n):
        return self.inverse(
            torch.randn(
                n,
                3072,
                device=self.log_scale.device
            )
        )


model = NICE()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
history = []


@torch.no_grad()
def record(step, batch):
    nll = -model.log_prob(batch).mean().item()
    history.append((step, nll))


train_iter = iter(loader)

started = time.perf_counter()
model.train()

for step in range(1, STEPS + 1):

    try:
        images, labels = next(train_iter)
    except StopIteration:
        train_iter = iter(loader)
        images, labels = next(train_iter)

    batch = images.flatten(start_dim=1)

    loss = -model.log_prob(batch).mean()

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=5.0
    )

    optimizer.step()

    if step % 100 == 0:
        record(step, batch)

model.eval()

print(
    f"Обучение: "
    f"{time.perf_counter() - started:.1f} с; "
    f"{STEPS} шагов"
)

print(
    f"NLL: "
    f"{history[0][1]:.3f} -> "
    f"{history[-1][1]:.3f} нат/точку"
)

h = np.array(history)

fig, ax = plt.subplots( figsize=(6.5, 2.6), layout="constrained")

ax.plot(h[:, 0], h[:, 1], label="Обучающая")

ax.set(xlabel="Шаг", ylabel="NLL, нат/точку", title="Обучение NICE")

ax.legend(frameon=False, fontsize=8)

plt.show()

# from torchdiffeq import odeint
# from importlib import reload
import src.science_school.continousFlows.CNF as CNF
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
from tqdm import tqdm
# from IPython.display import display
# import os
from pathlib import Path

DEVICE = "cpu"
torch.set_num_threads(16)
torch.set_num_interop_threads(16)

FIELDS = 64
HIDDEN_LAYERS = 5
HIDDEN_FEATURES = 64
STEPS = 1000
SAVE_AFTER = 50
DATASET_SIZE = 100000
BATCH_SIZE = 64
RTOL = 1e-2
ATOL = 1e-3
METHOD = "adaptive_heun"
EPOCHES = 50
LEARNING_RATE = 0.01

CHECKPOINT_NAME = "cnf_checkerboard.pt"
CHECKPOINTS_ROOT = Path("checkpoints")


def get_batch(n: int):
    # X, _, _ = make_checkerboard((n,2), n_clusters=8, noise=0.06, minval=-2, maxval=2)
    x_min, x_max = -2, 2
    y_min, y_max = -2, 2

    n *= 3
    X = np.random.uniform(
        [x_min, y_min],
        [x_max, y_max],
        size=(n, 2),
    )

    # Размер клетки
    cell_size = 0.5

    # Метка зависит от клетки шахматной доски
    y = (
        (np.floor(X[:, 0] / cell_size)
        + np.floor(X[:, 1] / cell_size)) % 2
    )
    X = X[y == 0]
    print(y.shape, X.shape, y.sum())
    assert X.shape[0] >= n // 3
    return torch.tensor(X[:n // 3], dtype=torch.float32, device=DEVICE)

def density(X, wraped):
    distribution = torch.distributions.MultivariateNormal(torch.zeros(2, device=DEVICE), torch.eye(2, device=DEVICE))
    Y, logprop = wraped.forward(X, log_jac=True)
    return distribution.log_prob(Y) - logprop

def show_density(fig, ax, wraped):
    dist = 2.5
    # Координаты сетки
    x = torch.linspace(-dist, dist, 200)
    y = torch.linspace(-dist, dist, 200)

    X, Y = torch.meshgrid(x, y, indexing="xy")

    # Форма: (200 * 200, 2)
    points = torch.stack([X.flatten(), Y.flatten()], dim=1)
    # print(points, points.shape)
    points = points.detach().to(DEVICE)

    # Твоя функция принимает (batch_size, 2)
    Z = density(points, wraped)

    # Возвращаем результат к форме сетки
    Z = Z.reshape(200, 200)
    Z = torch.exp(Z)

#     plt.clf()
    im = ax.imshow(
    Z.cpu().numpy(),
            extent=[-dist, dist, -dist, dist],
            origin="lower",
            aspect="equal",
    )
    if hasattr(fig, "_cbar"):
        fig._cbar.update_normal(im)
    else:
        fig._cbar = fig.colorbar(im, ax=ax)

def save_checkpoint(model, optim, step):
    data = {"model": model, "optimizer": optim, "step": step}
    torch.save(data, CHECKPOINTS_ROOT/CHECKPOINT_NAME)

def main():
    dataset = get_batch(DATASET_SIZE)
    dataset = torch.utils.data.TensorDataset(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # reload(CNF)
    # cpnf: CNF.Cpnf = CNF.Cpnf(2)
    # cpnf.to(DEVICE)
    hidden_features = HIDDEN_FEATURES
    hidden_layers = HIDDEN_LAYERS
    filds_num = FIELDS
    cpnflist = torch.nn.ModuleList([CNF.Cpnf(2).to(DEVICE) for i in range(filds_num)])
    hidden_l = []
    for i in range(hidden_layers):
        hidden_l.append(torch.nn.Linear(hidden_features, hidden_features))
        hidden_l.append(torch.nn.ReLU())
    time_gates = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(1, hidden_features),
                                    *hidden_l,
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(hidden_features, 1),
                                    torch.nn.Sigmoid()) for i in range(filds_num)])
    cnf = CNF.CNF(fields=cpnflist, activations=time_gates).to(DEVICE)

    # cnf.load_state_dict(torch.load("models/cnf.pt", weights_only=True))
    # cnf.compile()
    wraped = CNF.Wraper(cnf, rtol=RTOL, atol=ATOL, method=METHOD)
    # from torch.profiler import profile, ProfilerActivity

    # activities = [ProfilerActivity.CUDA]
    optimizer = torch.optim.Adam(wraped.field.parameters(), lr=LEARNING_RATE)
    step = 0

    if (CHECKPOINTS_ROOT/CHECKPOINT_NAME).exists():
        data = torch.load(CHECKPOINTS_ROOT/CHECKPOINT_NAME)
        cnf.load_state_dict(data["model"])
        optimizer.load_state_dict(data["optimizer"])
        step = data["step"] + 1

    # history = []
    fig, ax = plt.subplots()
    # out = display(fig, display_id=True)
    loss = 0
    # out2 = display(loss, display_id=True)

    # with profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
    pbar = tqdm()
    for _ in range(EPOCHES):
        for X, in dataloader:
            # print(X)
            Y, log_jac = wraped.forward(X, True)
            Y.requires_grad_()
            loss = CNF.loss_function(2, Y, log_jac)
            optimizer.zero_grad()
            wraped.backward(Y, loss)
            optimizer.step()
            if step % SAVE_AFTER == 0:
                # out2.update(loss)
            
                # fig.clear()
                show_density(fig, ax, wraped)
                # out.update(fig)
                torch.save(cnf.state_dict(), "models/cnf.pt")
                fig.savefig("gallery/fig" + str(step) + ".png")
                save_checkpoint(wraped.field.state_dict(), optimizer.state_dict(), step)
            # if step == 10:
            #     break
            step+=1
            pbar.update(1)
    show_density(fig, ax, wraped)
    torch.save(cnf.state_dict(), "models/cnf.pt")
    fig.savefig("gallery/fig" + str(step) + ".png")
    save_checkpoint(wraped.field.state_dict(), optimizer.state_dict(), step)
    pbar.close()

if __name__ == "__main__":
    main()

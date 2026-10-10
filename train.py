# from torchdiffeq import odeint
# from importlib import reload
import CNF
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
from tqdm import tqdm
# from IPython.display import display
# import os

DEVICE = "cpu"

FIELDS = 16
HIDDEN_LAYERS = 8
STEPS = 1000
SAVE_AFTER = 50
BATCH_SIZE = 64
EPOCHES = 50

def get_batch(n: int):
    X, _ = make_circles(n, noise=0.06, factor=0.5)
    return torch.tensor(X, dtype=torch.float32, device=DEVICE)

def density(X, wraped):
    distribution = torch.distributions.MultivariateNormal(torch.zeros(2, device=DEVICE), torch.eye(2, device=DEVICE))
    Y, logprop = wraped.forward(X, log_jac=True)
    return distribution.log_prob(Y) - logprop

def show_density(fig, ax, wraped):
    dist = 5
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

def main():
    dataset = get_batch(10000)
    dataset = torch.utils.data.TensorDataset(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # reload(CNF)
    # cpnf: CNF.Cpnf = CNF.Cpnf(2)
    # cpnf.to(DEVICE)
    hidden_features = HIDDEN_LAYERS
    filds_num = FIELDS
    cpnflist = torch.nn.ModuleList([CNF.Cpnf(2).to(DEVICE) for i in range(filds_num)])
    time_gates = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(1, hidden_features),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(hidden_features, hidden_features),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(hidden_features, 1),
                                    torch.nn.Sigmoid()) for i in range(filds_num)])
    cnf = CNF.CNF(fields=cpnflist, activations=time_gates).to(DEVICE)

    cnf.load_state_dict(torch.load("models/cnf.pt", weights_only=True))
    # cnf.compile()
    wraped = CNF.Wraper(cnf)
    # from torch.profiler import profile, ProfilerActivity

    # activities = [ProfilerActivity.CUDA]
    optimizer = torch.optim.Adam(wraped.field.parameters(), lr=0.001)
    step = 0
    # history = []
    fig, ax = plt.subplots()
    # out = display(fig, display_id=True)
    loss = 0
    # out2 = display(loss, display_id=True)

    # with profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
    for _ in range(EPOCHES):
        for X, in tqdm(dataloader):
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
            # if step == 10:
            #     break
            step+=1
    show_density(fig, ax, wraped)
    torch.save(cnf.state_dict(), "models/cnf.pt")
    fig.savefig("gallery/fig" + str(step) + ".png")

if __name__ == "__main__":
    main()

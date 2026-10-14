import copy
import math
import time

import torch


def bits_per_dim(nll, n_dims):
    return (nll + n_dims * math.log(256)) / (n_dims * math.log(2))


@torch.no_grad()
def mean_nll(model, x, transform=None, seed=None, chunk_size=250):
    generator = None if seed is None else torch.Generator(device=x.device).manual_seed(seed)
    was_training = model.training
    model.eval()
    try:
        total = 0.0
        for start in range(0, len(x), chunk_size):
            chunk = x[start:start + chunk_size]
            if transform is not None:
                chunk = transform(chunk, generator)
            total += -model.log_prob(chunk).sum().item()
        return total / len(x)
    finally:
        model.train(was_training)


def make_optimizer(model, lr, scale_weight_decay=5e-5):
    scales = [p for n, p in model.named_parameters() if n.endswith(".scale")]
    others = [p for n, p in model.named_parameters() if not n.endswith(".scale")]
    return torch.optim.Adam([
        {"params": others},
        {"params": scales, "weight_decay": scale_weight_decay},
    ], lr=lr)


class Trainer:
    def __init__(self, model, optimizer, get_batch, evaluate, clip_grad_norm=None):
        self.model = model
        self.optimizer = optimizer
        self.get_batch = get_batch
        self.evaluate = evaluate
        self.clip_grad_norm = clip_grad_norm

        self.history = []
        self.steps_done = 0
        self.last_step = 0
        self.best_step = 0
        self.best_val_nll = math.inf
        self.best_state = None
        self._started = time.perf_counter()

    @property
    def elapsed(self):
        return time.perf_counter() - self._started

    def record(self):
        train_nll, val_nll = self.evaluate()
        self.history.append((self.steps_done, train_nll, val_nll))
        if val_nll < self.best_val_nll:
            self.best_val_nll = val_nll
            self.best_step = self.steps_done
            self.best_state = copy.deepcopy(self.model.state_dict())

    def fit(self, steps, record_every, on_record=None):
        self._started = time.perf_counter()
        self.last_step = self.steps_done + steps
        self.model.train()
        for step in range(self.steps_done + 1, self.last_step + 1):
            loss = -self.model.log_prob(self.get_batch()).mean()
            self.optimizer.zero_grad()
            loss.backward()
            if self.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm)
            self.optimizer.step()
            self.steps_done = step

            if step % record_every == 0:
                self.record()
                if on_record is not None:
                    on_record(self)
                    self.model.train()
        self.model.eval()
        return self.elapsed

    def load_best(self):
        self.model.load_state_dict(self.best_state)
        self.model.eval()

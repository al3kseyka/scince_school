import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .sampling import best_samples

TRAIN_COLOR = "#2878B5"
VAL_COLOR = "#E07A3F"

NOT_HUMAN_CMAP = LinearSegmentedColormap.from_list(
    "not_human",
    [
        (0.000, "#010502"),
        (0.107, "#022112"),
        (0.230, "#04402a"),
        (0.357, "#046141"),
        (0.463, "#047a6e"),
        (0.532, "#048a90"),
        (0.618, "#039faf"),
        (0.751, "#0ac0c3"),
        (0.868, "#06dfdd"),
        (1.000, "#58eaeb"),
    ],
)


def use_style():
    plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 10})


def _grid(n, ncols, size):
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * size, nrows * size))
    axes = np.ravel(axes)
    for ax in axes:
        ax.axis("off")
    return axes


def show_images(images, ncols=5, size=1.6, titles=None):
    images = images.detach().cpu().clamp(0, 1).permute(0, 2, 3, 1).numpy()
    titles = titles or [None] * len(images)
    for ax, image, title in zip(_grid(len(images), ncols, size), images, titles):
        ax.imshow(image)
        if title is not None:
            ax.set_title(title, fontsize=7)
    plt.tight_layout()
    plt.show()


def show_best_n_samples(model, n, inception, real_stats, temperatures=(0.5, 0.7, 1.0), n_candidates=1000,
                        batch_size=100, ncols=5, size=1.6):
    images, scores, temps = best_samples(model, n, inception, real_stats, temperatures, n_candidates,
                                         batch_size)
    titles = [f"T={t:g} · FID {s:.0f}" for t, s in zip(temps.tolist(), scores.tolist())]
    show_images(images, ncols, size, titles)


def show_gray_images(images, shape, ncols=5, size=1.6, cmap=NOT_HUMAN_CMAP, vmin=None, vmax=None):
    for ax, image in zip(_grid(len(images), ncols, size), images):
        ax.imshow(image.detach().cpu().reshape(shape).numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
    plt.tight_layout()
    plt.show()


def show_gray_image(image, shape, cmap=NOT_HUMAN_CMAP):
    plt.figure()
    plt.imshow(image.detach().cpu().reshape(shape).numpy(), cmap=cmap)
    plt.axis("off")
    plt.colorbar()
    plt.show()


def plot_history(history, title="", best_step=None, metric=None, ylabel="NLL, нат/изображение"):
    points = np.array(history)
    train, val = points[:, 1], points[:, 2]
    if metric is not None:
        train, val = metric(train), metric(val)

    fig, ax = plt.subplots(figsize=(6.5, 2.6), layout="constrained")
    ax.plot(points[:, 0], train, label="Обучающая", color=TRAIN_COLOR, marker="o", markersize=3)
    ax.plot(points[:, 0], val, label="Проверочная", color=VAL_COLOR, marker="o", markersize=3)
    if best_step is not None:
        ax.axvline(best_step, color="gray", linestyle="--", linewidth=1, label="Лучшая модель")
    ax.set(xlabel="Шаг", ylabel=ylabel, title=title)
    ax.legend(frameon=False, fontsize=8)
    plt.show()

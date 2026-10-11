import os

import torch
from sklearn.datasets import fetch_lfw_people, fetch_olivetti_faces
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2


def to_float(images):
    return images.float() / 255


def dequantize(x, generator=None):
    noise = torch.rand(x.shape, generator=generator, dtype=x.dtype, device=x.device)
    return (255 * x + noise) / 256


def dequantize_uint8(images, generator=None):
    return dequantize(to_float(images), generator)


def add_uniform_noise(x, generator=None):
    return x + torch.rand(x.shape, generator=generator, dtype=x.dtype, device=x.device) / 256


def random_hflip(images):
    flip = torch.rand(len(images), device=images.device) < 0.5
    images = images.clone()
    images[flip] = images[flip].flip(-1)
    return images


class FaceAugment:
    def __init__(self, size):
        self.transform = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomResizedCrop(size, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
            v2.ColorJitter(brightness=0.1, contrast=0.1),
        ])

    def __call__(self, images):
        return torch.stack([self.transform(image) for image in images]).clamp(0, 1)


def load_olivetti(size=64, shuffle_seed=None):
    data = fetch_olivetti_faces(shuffle=shuffle_seed is not None, random_state=shuffle_seed)
    images = torch.tensor(data.data.reshape(-1, 1, 64, 64))
    if size != 64:
        images = transforms.Resize((size, size))(images)
    return images


def load_lfw():
    lfw = fetch_lfw_people(slice_=(slice(69, 197), slice(61, 189)), resize=0.5, color=True)
    faces = torch.tensor(lfw.images).permute(0, 3, 1, 2)
    return faces / 255 if faces.max() > 1 else faces


def load_celeba(root, split, size=64, crop=148, num_workers=8):
    cache = os.path.join(root, f"celeba{size}_{split}.pt")
    if os.path.exists(cache):
        return torch.load(cache)

    transform = transforms.Compose([
        transforms.CenterCrop(crop),
        transforms.Resize(size),
        transforms.PILToTensor(),
    ])
    dataset = datasets.CelebA(root, split=split, target_type="identity",
                              transform=transform, download=False)
    loader = DataLoader(dataset, batch_size=512, num_workers=num_workers)
    images = torch.empty(len(dataset), 3, size, size, dtype=torch.uint8)
    filled = 0
    for batch, _ in loader:
        images[filled:filled + len(batch)] = batch
        filled += len(batch)
    torch.save(images, cache)
    return images

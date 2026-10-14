import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import Inception_V3_Weights, inception_v3

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class InceptionFeatures:
    def __init__(self, device="cpu"):
        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        model.fc = nn.Identity()
        self.model = model.eval().to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, images, batch_size=100):
        features = []
        for start in range(0, len(images), batch_size):
            x = images[start:start + batch_size].to(self.device).float().clamp(0, 1)
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
            x = (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)
            features.append(self.model(x).cpu().double())
        return torch.cat(features)

    def stats(self, images, batch_size=100):
        features = self(images, batch_size)
        return features.mean(0), torch.cov(features.T)


def sample_fid(features, mu, sigma):
    # FID одной точки против гауссианы реальных признаков: ковариация точки нулевая
    return (features - mu).square().sum(1) + sigma.trace()

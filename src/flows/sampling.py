import torch

from .fid import sample_fid


@torch.no_grad()
def best_samples(model, n, inception, real_stats, temperatures=(0.5, 0.7, 1.0), n_candidates=1000,
                 batch_size=100):
    was_training = model.training
    model.eval()
    try:
        images, scores, temps = [], [], []
        for temperature in temperatures:
            for start in range(0, n_candidates, batch_size):
                x = model.sample(min(batch_size, n_candidates - start), temperature=temperature).cpu()
                images.append(x)
                scores.append(sample_fid(inception(x), *real_stats))
                temps.append(torch.full((len(x),), temperature))
    finally:
        model.train(was_training)

    images, scores, temps = torch.cat(images), torch.cat(scores), torch.cat(temps)
    scores = torch.where(torch.isfinite(scores), scores, torch.inf)
    best = scores.argsort()[:n]
    return images[best], scores[best], temps[best]

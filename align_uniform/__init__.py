import torch


def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


def dot_loss(x, y, t=2):
    return (x - y).norm(p=2, dim=1).pow(2).mean() - 0.5 * t * (
        torch.pdist(x, p=2).pow(2).mean() + torch.pdist(y, p=2).pow(2).mean()
    )


__all__ = ["align_loss", "uniform_loss", "dot_loss"]

import torch


def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


def dot_loss(x, y, t=2):
    return (
        -torch.tensordot(x, y, dims=([1], [1])).mean()
        + t * 0.5 * torch.tensordot(x[::2, :], x[1::2, :], dims=([1], [1])).mean()
        + t * 0.5 * torch.tensordot(y[::2, :], y[1::2, :], dims=([1], [1])).mean()
    )


__all__ = ["align_loss", "uniform_loss", "dot_loss"]

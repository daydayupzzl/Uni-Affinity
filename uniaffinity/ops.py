# scatter reductions, safe_norm, RBF. Written out here so the package does not need torch-scatter.
from __future__ import annotations

import math
import torch
from torch import Tensor


def torch_load(path, *, map_location="cpu", weights_only=False):
    """Load across PyTorch versions (weights_only was added after 1.11)."""
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


def safe_norm(x: Tensor, dim: int = -1, keepdim: bool = False, eps: float = 1e-8) -> Tensor:
    """Euclidean norm with a finite gradient at the zero vector."""
    return torch.sqrt(x.square().sum(dim=dim, keepdim=keepdim) + eps)


def scatter_sum(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    out = src.new_zeros((dim_size,) + src.shape[1:])
    if index.numel():
        out.index_add_(0, index, src)
    return out


def scatter_mean(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    out = scatter_sum(src, index, dim_size)
    count = scatter_sum(src.new_ones((src.shape[0], 1)), index, dim_size).clamp_min_(1)
    return out / count.view((dim_size,) + (1,) * (src.ndim - 1))


def scatter_max(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    flat = src.reshape(src.shape[0], -1)
    out = src.new_full((dim_size, flat.shape[1]), -torch.inf)
    if hasattr(out, "scatter_reduce_"):
        out.scatter_reduce_(0, index[:, None].expand_as(flat), flat, reduce="amax", include_self=True)
    else:  # pragma: no cover - old PyTorch fallback
        rows = []
        for group in range(dim_size):
            selected = flat[index == group]
            rows.append(selected.max(dim=0).values if len(selected) else flat.new_zeros(flat.shape[1]))
        out = torch.stack(rows)
    # empty segments come back as -inf
    out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
    return out.reshape((dim_size,) + src.shape[1:])


def scatter_softmax(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """Small dependency-free segment softmax (src is E or E x 1)."""
    shape = src.shape
    flat = src.reshape(src.shape[0], -1)
    # detached: the maxima are only a stability offset, and autograd trips over them in the fallback below
    flat_detached = flat.detach()
    maxes = src.new_full((dim_size, flat.shape[1]), -torch.inf)
    if hasattr(maxes, "scatter_reduce_"):
        maxes.scatter_reduce_(0, index[:, None].expand_as(flat), flat_detached, reduce="amax", include_self=True)
    else:  # pragma: no cover - old PyTorch fallback
        for i in range(flat.shape[0]):
            maxes[index[i]] = torch.maximum(maxes[index[i]], flat_detached[i])
    exp = torch.exp(flat - maxes[index])
    denom = scatter_sum(exp, index, dim_size).clamp_min_(1e-12)
    return (exp / denom[index]).reshape(shape)


class GaussianRBF(torch.nn.Module):
    def __init__(self, n: int, cutoff: float):
        super().__init__()
        centers = torch.linspace(0.0, cutoff, n)
        self.register_buffer("centers", centers)
        self.gamma = float(n / cutoff) ** 2
        self.cutoff = cutoff

    def forward(self, distance: Tensor) -> Tensor:
        d = distance.unsqueeze(-1)
        rbf = torch.exp(-self.gamma * (d - self.centers) ** 2)
        envelope = 0.5 * (torch.cos(math.pi * d / self.cutoff) + 1.0)
        return rbf * envelope * (d <= self.cutoff)


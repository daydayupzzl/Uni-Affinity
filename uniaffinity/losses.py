# Huber + pairwise-difference Huber + (1 - Pearson), weighted 1.0 / 0.2 / 0.1.
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


# the relative and Pearson terms are what force ligand discrimination
def affinity_loss(pred: Tensor, target: Tensor, cfg: dict) -> tuple[Tensor, dict[str, Tensor]]:
    pred, target = pred.float(), target.float()   # fp32: the Pearson term is touchy otherwise
    zero = pred.sum() * 0

    huber = F.huber_loss(pred, target, delta=cfg["huber_delta"])

    relative = zero
    if pred.numel() > 1:
        i, j = torch.triu_indices(pred.numel(), pred.numel(), offset=1, device=pred.device)
        relative = F.huber_loss(pred[i] - pred[j], target[i] - target[j], delta=cfg["huber_delta"])

    pearson = zero
    if pred.numel() > 2:
        pc, tc = pred - pred.mean(), target - target.mean()
        # eps inside the sqrt: all-identical predictions in a batch otherwise give an infinite gradient
        corr = (pc * tc).sum() / ((pc.square().sum() + 1e-8).sqrt() * (tc.square().sum() + 1e-8).sqrt())
        pearson = 1.0 - corr

    total = cfg["huber_weight"] * huber + cfg["relative_weight"] * relative + cfg["pearson_weight"] * pearson
    return total, {"loss": total.detach(), "huber": huber.detach(),
                   "relative": relative.detach(), "pearson_loss": pearson.detach()}

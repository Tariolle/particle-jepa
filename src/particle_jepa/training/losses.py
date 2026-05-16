from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def acceleration_loss(predicted: Tensor, target: Tensor) -> Tensor:
    return F.mse_loss(predicted, target)


def jepa_loss(prediction: Tensor, target: Tensor, normalize: bool = True) -> Tensor:
    if normalize:
        prediction = F.normalize(prediction, dim=-1)
        target = F.normalize(target, dim=-1)
    return F.smooth_l1_loss(prediction, target)


class HybridLoss(nn.Module):
    def __init__(self, dynamics_weight: float = 1.0, jepa_weight: float = 0.2) -> None:
        super().__init__()
        self.dynamics_weight = dynamics_weight
        self.jepa_weight = jepa_weight

    def forward(self, outputs: dict[str, Tensor], graph) -> dict[str, Tensor]:
        dyn = acceleration_loss(outputs["acceleration"], graph.y_acceleration)
        rep = jepa_loss(outputs["prediction"], outputs["target"])
        total = self.dynamics_weight * dyn + self.jepa_weight * rep
        return {"loss": total, "dynamics_loss": dyn.detach(), "jepa_loss": rep.detach()}


def covariance_regularizer(latents: Tensor, eps: float = 1e-4) -> Tensor:
    latents = latents - latents.mean(dim=0, keepdim=True)
    cov = latents.T @ latents / max(latents.size(0) - 1, 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return off_diag.pow(2).mean() + eps * torch.diag(cov).add(-1).pow(2).mean()

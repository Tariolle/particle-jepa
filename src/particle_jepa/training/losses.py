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


def sigreg_loss(latents: Tensor, sketch_dim: int = 64, eps: float = 1e-4) -> Tensor:
    """Sketched isotropic Gaussian regularization for anti-collapse."""
    latents = latents.float()
    latents = latents.flatten(0, -2) if latents.ndim > 2 else latents
    if latents.size(0) < 2:
        return latents.new_tensor(0.0)
    if sketch_dim > 0 and sketch_dim < latents.size(-1):
        generator = torch.Generator(device=latents.device).manual_seed(0)
        sketch = torch.randn(
            latents.size(-1),
            sketch_dim,
            generator=generator,
            device=latents.device,
            dtype=latents.dtype,
        )
        sketch = sketch / latents.size(-1) ** 0.5
        latents = latents @ sketch
    latents = latents - latents.mean(dim=0, keepdim=True)
    std = latents.std(dim=0)
    variance_loss = F.relu(1.0 - std).pow(2).mean()
    cov = latents.T @ latents / max(latents.size(0) - 1, 1)
    identity = torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)
    covariance_loss = (cov - identity).pow(2).mean()
    mean_loss = latents.mean(dim=0).pow(2).mean()
    return variance_loss + covariance_loss + eps * mean_loss


def temporal_graph_jepa_loss(outputs: dict[str, Tensor], config: dict) -> dict[str, Tensor]:
    prediction = jepa_loss(outputs["prediction"], outputs["target"].detach())
    node_prediction = jepa_loss(outputs["node_prediction"], outputs["node_target"].detach())
    sketch_dim = config.get("train", {}).get("sigreg_sketch_dim", 64)
    sigreg = (
        sigreg_loss(outputs["context"], sketch_dim=sketch_dim)
        + sigreg_loss(outputs["target"], sketch_dim=sketch_dim)
        + sigreg_loss(outputs["prediction"], sketch_dim=sketch_dim)
        + sigreg_loss(outputs["node_prediction"], sketch_dim=sketch_dim)
    ) / 4.0
    train_cfg = config.get("train", {})
    total = (
        train_cfg.get("prediction_weight", 1.0) * prediction
        + train_cfg.get("node_prediction_weight", 1.0) * node_prediction
        + train_cfg.get("sigreg_weight", 0.05) * sigreg
    )
    return {
        "loss": total,
        "prediction_loss": prediction.detach(),
        "node_prediction_loss": node_prediction.detach(),
        "sigreg_loss": sigreg.detach(),
    }


class HybridLoss(nn.Module):
    def __init__(
        self,
        dynamics_weight: float = 1.0,
        jepa_weight: float = 0.2,
        node_weight: float = 1.0,
        sigreg_weight: float = 0.05,
        sigreg_sketch_dim: int = 64,
    ) -> None:
        super().__init__()
        self.dynamics_weight = dynamics_weight
        self.jepa_weight = jepa_weight
        self.node_weight = node_weight
        self.sigreg_weight = sigreg_weight
        self.sigreg_sketch_dim = sigreg_sketch_dim

    def forward(self, outputs: dict[str, Tensor], graph) -> dict[str, Tensor]:
        dyn = acceleration_loss(outputs["acceleration"], graph.y_acceleration)
        rep = jepa_loss(outputs["prediction"], outputs["target"].detach())
        node = jepa_loss(outputs["node_prediction"], outputs["node_target"].detach())
        sigreg = (
            sigreg_loss(outputs["context"], sketch_dim=self.sigreg_sketch_dim)
            + sigreg_loss(outputs["target"], sketch_dim=self.sigreg_sketch_dim)
            + sigreg_loss(outputs["prediction"], sketch_dim=self.sigreg_sketch_dim)
            + sigreg_loss(outputs["node_prediction"], sketch_dim=self.sigreg_sketch_dim)
        ) / 4.0
        total = self.dynamics_weight * dyn + self.jepa_weight * (
            rep + self.node_weight * node + self.sigreg_weight * sigreg
        )
        return {
            "loss": total,
            "dynamics_loss": dyn.detach(),
            "jepa_loss": rep.detach(),
            "node_prediction_loss": node.detach(),
            "sigreg_loss": sigreg.detach(),
        }


def covariance_regularizer(latents: Tensor, eps: float = 1e-4) -> Tensor:
    latents = latents - latents.mean(dim=0, keepdim=True)
    cov = latents.T @ latents / max(latents.size(0) - 1, 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return off_diag.pow(2).mean() + eps * torch.diag(cov).add(-1).pow(2).mean()

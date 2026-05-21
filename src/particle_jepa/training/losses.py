from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

_SIGREG_SKETCH_CACHE: dict[tuple[int, int, str, int | None, torch.dtype], Tensor] = {}


def acceleration_loss(predicted: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    error = (predicted - target).pow(2)
    if mask is None:
        return error.mean()
    if mask.ndim == 1:
        mask = mask[:, None]
    mask = mask.to(device=error.device, dtype=error.dtype)
    return (error * mask).sum() / (mask.sum() * error.size(-1)).clamp_min(1.0)


def latent_prediction_loss(
    prediction: Tensor, target: Tensor, normalize: bool = True, eps: float = 1e-6
) -> Tensor:
    prediction = prediction.float()
    target = target.float()
    if normalize:
        prediction = F.normalize(prediction, dim=-1, eps=eps)
        target = F.normalize(target, dim=-1, eps=eps)
    return F.smooth_l1_loss(prediction, target)


def masked_latent_prediction_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
    normalize: bool = True,
    eps: float = 1e-6,
) -> Tensor:
    prediction = prediction.float()
    target = target.float()
    if normalize:
        prediction = F.normalize(prediction, dim=-1, eps=eps)
        target = F.normalize(target, dim=-1, eps=eps)
    error = F.smooth_l1_loss(prediction, target, reduction="none")
    if mask is None:
        return error.mean()
    while mask.ndim < error.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.to(device=error.device, dtype=error.dtype)
    return (error * mask).sum() / (mask.sum() * error.size(-1)).clamp_min(1.0)


def sigreg_loss(latents: Tensor, sketch_dim: int = 64, eps: float = 1e-4) -> Tensor:
    """Sketched isotropic Gaussian regularization for anti-collapse."""
    latents = latents.float()
    latents = latents.flatten(0, -2) if latents.ndim > 2 else latents
    latents = latents[torch.isfinite(latents).all(dim=-1)]
    if latents.size(0) < 2:
        return latents.new_tensor(0.0)
    if sketch_dim > 0 and sketch_dim < latents.size(-1):
        sketch = _sigreg_sketch(latents.size(-1), sketch_dim, latents.device, latents.dtype)
        latents = latents @ sketch
    latents = latents - latents.mean(dim=0, keepdim=True)
    std = latents.std(dim=0, unbiased=False)
    variance_loss = F.relu(1.0 - std).pow(2).mean()
    cov = latents.T @ latents / max(latents.size(0) - 1, 1)
    identity = torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)
    covariance_loss = (cov - identity).pow(2).mean()
    mean_loss = latents.mean(dim=0).pow(2).mean()
    return variance_loss + covariance_loss + eps * mean_loss


def temporal_graph_jepa_loss(outputs: dict[str, Tensor], config: dict) -> dict[str, Tensor]:
    prediction = latent_prediction_loss(outputs["prediction"], outputs["target"])
    node_prediction = latent_prediction_loss(outputs["node_prediction"], outputs["node_target"])
    masked_node_prediction = masked_latent_prediction_loss(
        outputs["node_prediction"], outputs["node_target"], outputs.get("node_mask")
    )
    region_prediction = latent_prediction_loss(
        outputs["region_prediction"], outputs["region_target"]
    )
    prediction_delta = latent_prediction_loss(
        outputs["prediction"] - outputs["context"],
        (outputs["target"] - outputs["context"]).detach(),
    )
    node_delta_target = outputs["node_target"] - outputs["node_context"]
    node_delta_prediction = latent_prediction_loss(
        outputs["node_prediction"] - outputs["node_context"],
        node_delta_target.detach(),
    )
    masked_node_delta_prediction = masked_latent_prediction_loss(
        outputs["node_prediction"] - outputs["node_context"],
        node_delta_target.detach(),
        outputs.get("node_mask"),
    )
    region_delta_target = outputs["region_target"] - _region_context(outputs)
    region_delta_prediction = latent_prediction_loss(
        outputs["region_prediction"] - _region_context(outputs),
        region_delta_target.detach(),
    )
    sketch_dim = config.get("train", {}).get("sigreg_sketch_dim", 64)
    sigreg = (
        sigreg_loss(outputs["context"], sketch_dim=sketch_dim)
        + sigreg_loss(outputs["target"], sketch_dim=sketch_dim)
        + sigreg_loss(outputs["prediction"], sketch_dim=sketch_dim)
        + sigreg_loss(outputs["node_prediction"], sketch_dim=sketch_dim)
        + sigreg_loss(outputs["region_prediction"], sketch_dim=sketch_dim)
    ) / 5.0
    train_cfg = config.get("train", {})
    total = (
        train_cfg.get("prediction_weight", 1.0) * prediction
        + train_cfg.get("node_prediction_weight", 1.0) * masked_node_prediction
        + train_cfg.get("region_prediction_weight", 1.0) * region_prediction
        + train_cfg.get("delta_prediction_weight", 0.0) * prediction_delta
        + train_cfg.get("node_delta_prediction_weight", 0.0) * masked_node_delta_prediction
        + train_cfg.get("region_delta_prediction_weight", 0.0) * region_delta_prediction
        + train_cfg.get("sigreg_weight", 0.05) * sigreg
    )
    return {
        "loss": total,
        "prediction_loss": prediction.detach(),
        "node_prediction_loss": node_prediction.detach(),
        "masked_node_prediction_loss": masked_node_prediction.detach(),
        "region_prediction_loss": region_prediction.detach(),
        "prediction_delta_loss": prediction_delta.detach(),
        "node_delta_prediction_loss": node_delta_prediction.detach(),
        "masked_node_delta_prediction_loss": masked_node_delta_prediction.detach(),
        "region_delta_prediction_loss": region_delta_prediction.detach(),
        "sigreg_loss": sigreg.detach(),
    }


def covariance_regularizer(latents: Tensor, eps: float = 1e-4) -> Tensor:
    latents = latents - latents.mean(dim=0, keepdim=True)
    cov = latents.T @ latents / max(latents.size(0) - 1, 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return off_diag.pow(2).mean() + eps * torch.diag(cov).add(-1).pow(2).mean()


def _region_context(outputs: dict[str, Tensor]) -> Tensor:
    if "region_context" in outputs:
        return outputs["region_context"]
    context = outputs["context"]
    return context[:, None, :].expand_as(outputs["region_prediction"])


def _sigreg_sketch(
    input_dim: int, sketch_dim: int, device: torch.device, dtype: torch.dtype
) -> Tensor:
    key = (input_dim, sketch_dim, device.type, device.index, dtype)
    sketch = _SIGREG_SKETCH_CACHE.get(key)
    if sketch is not None:
        return sketch
    generator = torch.Generator(device=device).manual_seed(0)
    sketch = torch.randn(
        input_dim,
        sketch_dim,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    sketch = sketch / input_dim**0.5
    _SIGREG_SKETCH_CACHE[key] = sketch
    return sketch

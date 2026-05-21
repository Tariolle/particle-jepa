from __future__ import annotations

import torch
from torch import nn

from particle_jepa.training.losses import (
    acceleration_loss,
    latent_prediction_loss,
    sigreg_loss,
    temporal_graph_jepa_loss,
)


def test_acceleration_loss_is_zero_for_equal_tensors() -> None:
    target = torch.ones(4, 2)
    assert acceleration_loss(target, target).item() == 0.0


def test_acceleration_loss_mask_matches_selected_unmasked_mean() -> None:
    predicted = torch.tensor([[1.0, 2.0], [10.0, 10.0], [4.0, 6.0]])
    target = torch.tensor([[0.0, 0.0], [0.0, 0.0], [2.0, 2.0]])
    mask = torch.tensor([1.0, 0.0, 1.0])

    expected = (predicted[[0, 2]] - target[[0, 2]]).pow(2).mean()

    assert torch.allclose(acceleration_loss(predicted, target, mask), expected)


def test_jepa_loss_is_finite() -> None:
    prediction = torch.randn(3, 8)
    target = torch.randn(3, 8)
    loss = latent_prediction_loss(prediction, target)
    assert torch.isfinite(loss)


def test_jepa_loss_decreases_on_tiny_overfit_batch() -> None:
    torch.manual_seed(0)
    predictor = nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=0.05)
    context = torch.randn(8, 4)
    target = torch.randn(8, 4)

    first_loss = None
    last_loss = None
    for step in range(40):
        optimizer.zero_grad(set_to_none=True)
        loss = latent_prediction_loss(predictor(context), target)
        if step == 0:
            first_loss = loss.item()
        loss.backward()
        optimizer.step()
        last_loss = loss.item()

    assert first_loss is not None
    assert last_loss is not None
    assert last_loss < first_loss


def test_sigreg_loss_penalizes_collapsed_latents() -> None:
    collapsed = torch.zeros(16, 8)
    varied = torch.randn(16, 8)

    assert sigreg_loss(collapsed) > sigreg_loss(varied)


def test_temporal_jepa_loss_is_finite_for_tiny_fp16_latents() -> None:
    tiny = torch.full((4, 8), 1e-8, dtype=torch.float16)
    node_tiny = torch.full((16, 8), 1e-8, dtype=torch.float16)
    region_tiny = torch.full((4, 16, 8), 1e-8, dtype=torch.float16)
    outputs = {
        "prediction": tiny,
        "target": tiny.clone(),
        "context": tiny.clone(),
        "node_prediction": node_tiny,
        "node_target": node_tiny.clone(),
        "node_context": node_tiny.clone(),
        "region_prediction": region_tiny,
        "region_target": region_tiny.clone(),
        "node_mask": torch.ones(16),
    }

    losses = temporal_graph_jepa_loss(outputs, {"train": {"sigreg_sketch_dim": 4}})

    assert torch.isfinite(losses["loss"])


def test_temporal_jepa_loss_includes_weighted_delta_terms() -> None:
    outputs = {
        "prediction": torch.tensor([[1.0, 1.0]]),
        "target": torch.tensor([[3.0, 0.0]]),
        "context": torch.tensor([[1.0, 0.0]]),
        "node_prediction": torch.tensor([[1.0, 1.0], [1.0, 2.0]]),
        "node_target": torch.tensor([[3.0, 0.0], [5.0, 0.0]]),
        "node_context": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        "region_prediction": torch.tensor([[[1.0, 1.0]]]),
        "region_context": torch.tensor([[[1.0, 0.0]]]),
        "region_target": torch.tensor([[[3.0, 0.0]]]),
        "node_mask": torch.ones(2),
    }

    no_delta = temporal_graph_jepa_loss(
        outputs,
        {
            "train": {
                "prediction_weight": 0.0,
                "node_prediction_weight": 0.0,
                "region_prediction_weight": 0.0,
                "sigreg_weight": 0.0,
            }
        },
    )["loss"]
    with_delta = temporal_graph_jepa_loss(
        outputs,
        {
            "train": {
                "prediction_weight": 0.0,
                "node_prediction_weight": 0.0,
                "region_prediction_weight": 0.0,
                "delta_prediction_weight": 1.0,
                "node_delta_prediction_weight": 1.0,
                "region_delta_prediction_weight": 1.0,
                "sigreg_weight": 0.0,
            }
        },
    )

    assert no_delta.item() == 0.0
    assert with_delta["loss"] > 0.0
    assert with_delta["prediction_delta_loss"] > 0.0
    assert with_delta["masked_node_delta_prediction_loss"] > 0.0
    assert with_delta["region_delta_prediction_loss"] > 0.0

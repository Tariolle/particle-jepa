from __future__ import annotations

import torch
from torch_geometric.loader import DataLoader

from particle_jepa.data.toy_dataset import ToyParticleConfig, ToyParticleDataset
from particle_jepa.models import GraphNetworkSimulator, HybridGNSJEPA, ParticleJEPA


def _batch():
    dataset = ToyParticleDataset(
        ToyParticleConfig(num_trajectories=2, num_particles=10, sequence_length=6, future_offset=2)
    )
    return next(iter(DataLoader(dataset, batch_size=2)))


def test_gns_forward_shape() -> None:
    context, _future = _batch()
    model = GraphNetworkSimulator(hidden_dim=32, message_passing_steps=2)
    acceleration = model(context)
    assert acceleration.shape == context.y_acceleration.shape


def test_jepa_forward_shape() -> None:
    context, future = _batch()
    model = ParticleJEPA(hidden_dim=32, latent_dim=24, message_passing_steps=2)
    outputs = model(context, future)
    assert outputs["prediction"].shape == (2, 24)
    assert outputs["target"].shape == (2, 24)


def test_jepa_target_encoder_ema_update() -> None:
    model = ParticleJEPA(hidden_dim=32, latent_dim=24, message_passing_steps=2)
    before = [parameter.detach().clone() for parameter in model.target_encoder.parameters()]
    with torch.no_grad():
        for parameter in model.context_encoder.parameters():
            parameter.add_(0.1)
    model.update_target_encoder(decay=0.5)
    after = list(model.target_encoder.parameters())

    assert any(not torch.equal(old, new) for old, new in zip(before, after, strict=True))


def test_hybrid_forward_shape() -> None:
    context, future = _batch()
    model = HybridGNSJEPA(hidden_dim=32, latent_dim=24, message_passing_steps=2)
    outputs = model(context, future)
    assert outputs["acceleration"].shape == context.y_acceleration.shape
    assert outputs["prediction"].shape == outputs["target"].shape

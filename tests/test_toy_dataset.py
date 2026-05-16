from __future__ import annotations

from particle_jepa.data.toy_dataset import ToyParticleConfig, ToyParticleDataset


def test_toy_dataset_returns_context_future_pair() -> None:
    dataset = ToyParticleDataset(
        ToyParticleConfig(num_trajectories=2, num_particles=8, sequence_length=6, future_offset=2)
    )
    context, future = dataset[0]

    assert len(dataset) == 8
    assert context.x.shape == (8, 7)
    assert future.x.shape == (8, 7)
    assert context.y_acceleration.shape == (8, 2)

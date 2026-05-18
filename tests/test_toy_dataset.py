from __future__ import annotations

from particle_jepa.data.toy_dataset import (
    ToyParticleConfig,
    ToyParticleDataset,
    generate_toy_rollouts,
)


def test_toy_rollout_dict_has_expected_shapes() -> None:
    rollout = generate_toy_rollouts(
        ToyParticleConfig(num_trajectories=2, num_particles=8, sequence_length=6, future_offset=2)
    )

    assert rollout["positions"].shape == (2, 6, 8, 2)
    assert rollout["velocities"].shape == (2, 6, 8, 2)
    assert rollout["particle_types"].shape == (2, 8)
    assert rollout["metadata"]["dimension"] == 2


def test_toy_dataset_returns_context_future_pair() -> None:
    dataset = ToyParticleDataset(
        ToyParticleConfig(num_trajectories=2, num_particles=8, sequence_length=6, future_offset=2)
    )
    context, future = dataset[0]

    assert len(dataset) == 8
    assert context.x.shape == (8, 7)
    assert future.x.shape == (8, 7)
    assert context.y_acceleration.shape == (8, 2)
    assert context.horizon.item() == 2


def test_toy_dataset_cycles_future_offsets_without_growing_budget() -> None:
    dataset = ToyParticleDataset(
        ToyParticleConfig(
            num_trajectories=1,
            num_particles=8,
            sequence_length=8,
            future_offset=1,
            future_offsets=(1, 2, 4),
        )
    )

    assert len(dataset) == 4
    assert [dataset[index][0].horizon.item() for index in range(len(dataset))] == [1, 2, 4, 1]

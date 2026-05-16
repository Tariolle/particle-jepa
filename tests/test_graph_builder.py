from __future__ import annotations

import torch

from particle_jepa.data.graph_builder import ParticleGraphBuilder


def test_radius_graph_has_expected_edge_features() -> None:
    positions = torch.tensor([[0.0, 0.0], [0.1, 0.0], [0.9, 0.9]])
    velocities = torch.zeros_like(positions)
    graph = ParticleGraphBuilder(radius=0.2).build(positions, velocities)

    assert graph.x.shape == (3, 7)
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_attr.shape[1] == 5
    assert graph.edge_index.shape[1] == 2

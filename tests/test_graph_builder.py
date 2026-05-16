from __future__ import annotations

import torch

from particle_jepa.data.graph_builder import ParticleGraphBuilder, build_radius_graph


def test_radius_graph_has_expected_edge_features() -> None:
    positions = torch.tensor([[0.0, 0.0], [0.1, 0.0], [0.9, 0.9]])
    velocities = torch.zeros_like(positions)
    graph = ParticleGraphBuilder(radius=0.2).build(positions, velocities)

    assert graph.x.shape == (3, 7)
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_attr.shape[1] == 6
    assert graph.edge_index.shape[1] == 2


def test_build_radius_graph_function_returns_edges_and_features() -> None:
    positions = torch.tensor([[0.0, 0.0], [0.1, 0.0]])
    velocities = torch.zeros_like(positions)
    edge_index, edge_attr = build_radius_graph(positions, velocities, radius=0.2)

    assert edge_index.shape == (2, 2)
    assert edge_attr.shape == (2, 6)

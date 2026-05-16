# Particle-JEPA

**Particle-JEPA: Self-Supervised Graph World Models for Particle Physics**

Particle-JEPA is a research-oriented project for learning graph world models on particle simulations. A physical state is represented as a dynamic particle graph, and the model learns to predict the next graph state in latent space.

The project is not an LLM project and not a generic graph benchmark. It is about modeling the evolution of a physical environment represented as graphs.

## Core Idea

Given the current particle graph `G_t`, predict the latent representation of the next particle graph `G_t+1`.

The current state already contains positions, velocities, particle type, material flags, and boundary information, so it is rich enough to serve as the context state.

```text
current particle state
        |
dynamic radius graph G_t
        |
GNN context encoder
        |
node + graph latent predictor
        |
predicted next-state latent

next particle state G_t+1
        |
same GNN encoder
        |
target next-state latent
```

The Particle-JEPA loss is intentionally simple:

```text
prediction loss + SIGReg anti-collapse loss
```

SIGReg is used as the clean anti-collapse regularizer. There is no EMA target encoder in the default design.

## Model Comparisons

The intended comparison set is:

1. **Particle-JEPA**: graph-native JEPA adapted to particle dynamics, with SIGReg.
2. **GNS baseline**: Graph Network Simulator-style learned physical simulator.
3. **Hybrid GNS + JEPA**: dynamics prediction plus the Particle-JEPA auxiliary objective.

## Graph Representation

- Nodes are particles.
- Edges are nearby interactions from a dynamic radius graph.
- Node features include position, velocity, particle type, material type, and boundary flags.
- Edge features include relative position, relative velocity, distance, and normalized distance.

## Current Capabilities

- Toy 2D particle rollout generator.
- Dynamic particle graph construction with PyTorch Geometric.
- Particle-JEPA with node-level and graph-level latent next-state prediction.
- SIGReg anti-collapse regularization.
- GNS-style baseline.
- Hybrid GNS + JEPA model.
- Hydra configuration.
- Weights & Biases tracking support.
- FP16 autocast on CUDA.
- `torch.compile(..., mode="reduce-overhead")` support.
- Rollout strip visualization.
- Latent future retrieval visualization with chance and random baselines.

## Installation

Python 3.11+ is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pre-commit install
```

PyTorch Geometric installation can vary by CUDA and PyTorch version. If the default install fails, follow the official PyG installation selector:

[https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

## Quick Start

Run tests:

```bash
pytest
```

Train Particle-JEPA:

```bash
python scripts/train.py
```

Train the GNS-style baseline:

```bash
python scripts/train.py --config-name gns
```

Train the hybrid model:

```bash
python scripts/train.py --config-name hybrid
```

Use Hydra overrides:

```bash
python scripts/train.py data.num_particles=128 data.horizon=1 training.epochs=20
```

Enable Weights & Biases:

```bash
wandb login
python scripts/train.py tracking.enabled=true
```

Export a rollout strip from a GNS or hybrid checkpoint:

```bash
python scripts/rollout.py --checkpoint runs/<run>_gns/checkpoints/last.pt --steps 32
```

Export a Particle-JEPA retrieval panel:

```bash
python scripts/retrieve.py --checkpoint runs/<run>_jepa/checkpoints/last.pt --top-k 5
```

Training writes run artifacts to:

```text
runs/
└── YYYYMMDD_HHMMSS_model_name/
    ├── config.yaml
    ├── checkpoints/
    ├── logs.jsonl
    ├── metrics.json
    └── visualizations/
```

## Evaluation

Particle-JEPA is evaluated with:

- latent next-state retrieval top-k accuracy,
- chance retrieval baseline,
- random latent retrieval baseline,
- prediction-target cosine,
- latent standard deviation diagnostics,
- latent trajectory visualizations.

GNS and hybrid models are evaluated with:

- one-step prediction error,
- rollout position error,
- rollout visual comparison,
- rollout Chamfer distance.

## Dataset Roadmap

The repository currently uses a toy particle simulator. DeepMind Learning-to-Simulate integration is intentionally left as the next dataset milestone.

Planned dataset work:

- TFRecord or exported-array conversion.
- Dataset-specific normalization.
- Material and boundary metadata parsing.
- Multi-material rollout evaluation.

## References

- Sanchez-Gonzalez et al., *Learning to Simulate Complex Physics with Graph Networks*, ICML 2020.
- LeCun, *A Path Towards Autonomous Machine Intelligence*, 2022.
- Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*, CVPR 2023.

## License

MIT License. See [LICENSE](LICENSE).

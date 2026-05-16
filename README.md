# Particle-JEPA

**Particle-JEPA: Self-Supervised Graph World Models for Particle Physics**

Particle-JEPA is a compact research project for learning graph-native world models on particle simulations. It represents physical systems as dynamic graphs and trains Graph Neural Networks with a JEPA-style latent future prediction objective.

Particle-JEPA explores whether JEPA-style latent future prediction can improve graph-based learned physical simulators. Instead of only predicting the next particle positions, the model learns to predict the representation of a future particle graph, enabling future retrieval, latent rollout analysis, and potentially more robust long-horizon world modeling.

The central question:

> Can a JEPA-style latent future prediction objective improve learned graph-based physical world models, especially in representation quality, future retrieval, and long-horizon rollout behavior?

This repository is initialized as a research-grade starter: runnable toy data, graph construction, baseline GNS-style dynamics, Particle-JEPA latent prediction, hybrid training hooks, evaluation utilities, and visualization entry points.

## Why This Project

Particle systems are naturally graph structured:

- **Nodes** are particles.
- **Edges** are nearby physical interactions.
- **Node features** include position, velocity, particle type, material type, and boundary flags.
- **Edge features** include relative displacement, relative velocity, and distance.

Instead of predicting only the next particle position, Particle-JEPA learns to predict a future graph representation in latent space:

```text
Particle rollout
      ↓
Dynamic radius graph
      ↓
GNN context encoder ───────┐
                           ↓
                     JEPA predictor ──→ predicted future embedding
                           ↑
Future graph ─→ GNN target encoder ───→ target future embedding
```

## Current Capabilities

- Toy particle dataset with simple spring-like dynamics for smoke tests and demos.
- Dynamic radius graph construction with PyTorch Geometric `Data` objects.
- Hydra experiment configuration.
- Weights & Biases experiment tracking.
- GNS-style message-passing simulator baseline.
- Particle-JEPA encoder and latent predictor.
- Hybrid model that combines acceleration prediction with latent future prediction.
- Losses for dynamics and JEPA-style representation learning.
- Rollout, retrieval, and latent trajectory metrics.
- Matplotlib visualization utilities for particles and latent paths.
- CLI scripts for training, evaluation, rollout, preprocessing, and visualization.
- Pytest coverage for graph construction, toy data, models, and losses.

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

Train the JEPA toy model:

```bash
python scripts/train.py
```

Train the baseline GNS-style model:

```bash
python scripts/train.py --config-name gns
```

Train the hybrid model:

```bash
python scripts/train.py --config-name hybrid
```

Use Hydra overrides:

```bash
python scripts/train.py data.num_particles=128 data.horizon=8 training.epochs=20
```

Enable Weights & Biases tracking:

```bash
wandb login
python scripts/train.py tracking.enabled=true
```

Create a toy rollout visualization:

```bash
python scripts/visualize.py --run latest
```

Export a latent future retrieval panel from a Particle-JEPA checkpoint:

```bash
python scripts/retrieve.py --checkpoint runs/<run>_jepa/checkpoints/last.pt --top-k 3
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

## Repository Layout

```text
configs/                  YAML experiment configuration
data/                     raw and processed dataset locations
notebooks/                exploration notebooks
scripts/                  command-line entry points
src/particle_jepa/        package source code
tests/                    smoke and unit tests
```

## Model Families

### GNS Baseline

A compact Graph Network Simulator-style model predicts particle acceleration from the current graph. It supports next-step prediction and autoregressive rollouts.

### Particle-JEPA

A graph encoder embeds the current particle graph. A target encoder embeds a future particle graph. A predictor maps the context embedding toward the future embedding using a latent regression objective.

### Hybrid GNS + JEPA

The hybrid model shares graph representations across physical prediction and latent future prediction, enabling combined supervised dynamics and self-supervised representation learning.

The planned paper angle:

> Can JEPA-style latent future prediction provide a useful auxiliary objective for graph neural particle simulators?

Model variants:

```text
1. GNS baseline
2. Particle-JEPA
3. Hybrid GNS + JEPA
```

## Evaluation Ideas

- Next-step position and velocity prediction error.
- Long-horizon rollout RMSE.
- Rollout Chamfer distance.
- Latent future retrieval accuracy.
- Latent prediction MSE and cosine similarity.
- Latent trajectory alignment.
- Qualitative side-by-side rollout videos.
- Latent future retrieval panels.

## Dataset Roadmap

The initial implementation uses a toy particle simulator. Extension points are included for DeepMind Learning-to-Simulate datasets in `src/particle_jepa/data/lts_dataset.py`.

Expected future integrations:

- TFRecord conversion or exported NumPy trajectories.
- Material and boundary metadata parsing.
- Dataset-specific normalization.
- Multi-material rollout evaluation.

## Roadmap

- Add full DeepMind Learning-to-Simulate conversion and normalization.
- Add EMA target encoder updates for Particle-JEPA.
- Expand rollout evaluation with dataset-specific boundary handling.
- Add retrieval visualizations from trained checkpoints.
- Add optional experiment tracking with wandb or TensorBoard.
- Produce polished GIF/MP4 rollout comparisons for portfolio use.

## References

- Sanchez-Gonzalez et al., *Learning to Simulate Complex Physics with Graph Networks*, ICML 2020.
- LeCun, *A Path Towards Autonomous Machine Intelligence*, 2022.
- Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*, CVPR 2023.

## License

MIT License. See [LICENSE](LICENSE).

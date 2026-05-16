# Particle-JEPA

**Particle-JEPA: Self-Supervised Graph World Models for Particle Physics**

Particle-JEPA is a compact research project for learning graph-native world models on particle simulations. It represents physical systems as dynamic graphs and trains Graph Neural Networks with a JEPA-style latent future prediction objective.

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
graph at t -> context encoder -> latent context
graph at t+k -> target encoder  -> latent target
latent context -> predictor      -> future latent target
```

## Current Capabilities

- Toy particle dataset with simple spring-like dynamics for smoke tests and demos.
- Dynamic radius graph construction with PyTorch Geometric `Data` objects.
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
python scripts/train.py --config configs/default.yaml --experiment jepa
```

Train the baseline GNS-style model:

```bash
python scripts/train.py --config configs/default.yaml --experiment gns
```

Create a toy rollout visualization:

```bash
python scripts/visualize.py --output outputs/toy_particles.png
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

## Evaluation Ideas

- Next-step position and velocity prediction error.
- Long-horizon rollout RMSE.
- Latent future retrieval accuracy.
- Latent trajectory alignment.
- Qualitative side-by-side rollout videos.

## Dataset Roadmap

The initial implementation uses a toy particle simulator. Extension points are included for DeepMind Learning-to-Simulate datasets in `src/particle_jepa/data/lts_dataset.py`.

Expected future integrations:

- TFRecord conversion or exported NumPy trajectories.
- Material and boundary metadata parsing.
- Dataset-specific normalization.
- Multi-material rollout evaluation.

## License

MIT License. See [LICENSE](LICENSE).

# Particle-JEPA

Particle-JEPA was a research prototype testing whether a graph-native JEPA
objective could learn useful particle-simulation dynamics from DeepMind
Learning-to-Simulate data.

The final conclusion is negative for this prototype: the supervised GNS baseline
learned plausible WaterRamps dynamics, but the frozen Particle-JEPA latents were
not dynamically sufficient. A lightweight probe trained on JEPA context latents
produced physically wrong acceleration directions, and a probe trained on JEPA
predicted latents produced incoherent particle rollouts.

This repository is therefore best read as an archived experiment, not a finished
simulator.

## Question Tested

Can a particle graph model learn a useful latent world representation by
predicting the next graph latent, regularized with SIGReg, without supervising
decoded acceleration or rollout?

```text
current particle graph G_t
  -> shared graph encoder
  -> context node/region/graph latents
  -> latent predictor
  -> predicted future latents

future particle graph G_t+1
  -> same graph encoder
  -> target future latents
```

The intended JEPA objective was:

```text
latent prediction + SIGReg anti-collapse
```

No EMA target encoder was intended. SIGReg was intentional.

## Final Result

The prototype did not validate the JEPA approach for simulator-grade particle
rollout.

Final diagnostic probes on WaterRamps:

```text
raw_features probe:
  learned gravity, but not ramp contact
  rollout position error: 0.012546

node_context probe:
  physically wrong direction; gravity pulled left instead of down
  rollout position error: 0.217165

node_prediction probe:
  predicted latents were incoherent; particle block degraded before contact
  rollout position error: 0.038809
```

Because the supervised GNS baseline worked decently on the same data and rollout
pipeline, the failure is attributed to the JEPA representation/prediction
objective rather than to dataset loading, graph construction, acceleration
normalization, or rollout integration.

## Interpretation

The core issue is identifiability. A non-collapsed latent that predicts another
latent is not necessarily a latent that preserves the physical variables needed
for rollout: velocity, contact geometry, boundary relation, particle type
effects, and force-relevant local neighborhoods.

For particle simulation, small acceleration errors compound immediately. The
current JEPA objective can satisfy latent agreement while discarding information
that a simulator needs.

## What Remains Useful

- Learning-to-Simulate TFRecord loading without TensorFlow.
- Dynamic PyG graph construction for particle trajectories.
- A supervised GNS-style baseline.
- Rollout visualization and evaluation scripts.
- Frozen-latent probe tooling for representation diagnostics.
- Training utilities for mixed precision, checkpointing, and resume.

## Setup

Python 3.11+ is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pre-commit install
```

PyTorch Geometric installation can vary by CUDA and PyTorch version. If needed,
use the official selector:

<https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html>

## Reproduce

Run tests:

```bash
pytest
```

Download an LTS dataset:

```bash
python scripts/download_data.py --dataset WaterRamps --splits metadata train valid
```

Train Particle-JEPA:

```bash
python scripts/train.py --config-name lts_particle_jepa
```

Resume Particle-JEPA:

```bash
python scripts/train.py --config-name lts_particle_jepa training.resume_from=runs/<run>_jepa/checkpoints/last.pt
```

Train diagnostic probes:

```bash
python scripts/train_probe.py --checkpoint runs/<run>_jepa/checkpoints/last.pt --batch-size 5 --epochs 10 --rollout-steps 48 --latent-source raw_features
python scripts/train_probe.py --checkpoint runs/<run>_jepa/checkpoints/last.pt --batch-size 5 --epochs 10 --rollout-steps 48 --latent-source node_context
python scripts/train_probe.py --checkpoint runs/<run>_jepa/checkpoints/last.pt --batch-size 5 --epochs 10 --rollout-steps 48 --latent-source node_prediction
```

Train the supervised GNS baseline:

```bash
python scripts/train.py --config-name lts_gns
```

Render a GNS rollout:

```bash
python scripts/rollout.py --checkpoint runs/<run>_gns/checkpoints/last.pt --steps 64
```

## References

- Sanchez-Gonzalez et al., *Learning to Simulate Complex Physics with Graph
  Networks*, ICML 2020.
- Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding
  Predictive Architecture*, CVPR 2023.

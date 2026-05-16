# Data

This directory is intentionally lightweight.

- `raw/` stores downloaded or externally exported datasets.
- `processed/` stores graph-ready tensors, NumPy arrays, or cached PyG objects.

The initial repository ships with a generated toy particle dataset in code. DeepMind Learning-to-Simulate integration should be added through `src/particle_jepa/data/lts_dataset.py` and `scripts/download_data.py`.

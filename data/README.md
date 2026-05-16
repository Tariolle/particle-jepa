# Data

This directory is intentionally lightweight.

- `raw/` stores downloaded or externally exported datasets.
- `processed/` stores graph-ready tensors, NumPy arrays, or cached PyG objects.

The repository ships with a generated toy particle dataset in code and supports
DeepMind Learning-to-Simulate TFRecords under `raw/<DATASET_NAME>/`.

Example:

```bash
python scripts/download_data.py --dataset WaterDropSample --splits metadata train valid
```

# Learning-to-Simulate Integration

The repository supports DeepMind Learning-to-Simulate TFRecords without a
TensorFlow runtime dependency. Records are decoded with protobuf, converted to
trajectory tensors, and exposed as PyTorch Geometric particle graphs.

## Supported Use

The loader is useful and remains one of the reusable parts of the project.

It supports:

- official metadata radius and bounds,
- finite-difference velocities from positions,
- official-style acceleration normalization,
- kinematic-particle masking,
- multi-frame position history,
- dynamic radius graphs,
- WaterRamps-style 2D rollout visualization.

## Recommended Dataset

Use `WaterRamps` for the final prototype diagnostics:

```bash
python scripts/download_data.py --dataset WaterRamps --splits metadata train valid
```

`WaterDropSample` is still useful for quick loader tests:

```bash
python scripts/download_data.py --dataset WaterDropSample --splits metadata train valid
```

## Graph State

For the official-style 2D configs, node features include position history,
velocity history, particle type, and boundary information. Edge features use
relative displacement and normalized distance. The graph radius defaults to
`metadata.json/default_connectivity_radius`.

Rollout integration uses the LTS convention of implicit `dt=1`.

## Final Note

Because the supervised GNS baseline learned plausible WaterRamps behavior, the
LTS loader and rollout path are considered adequate for this prototype. The
negative result is attributed to the JEPA objective, not the LTS integration.

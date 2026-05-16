# DeepMind Learning-to-Simulate Integration

Particle-JEPA now reads the official DeepMind Learning-to-Simulate TFRecord
release without requiring TensorFlow as a runtime dependency. The loader decodes
`tf.SequenceExample` records with protobuf, converts them to trajectory tensors,
builds PyG radius graphs, and returns the same `(G_t, G_t+k)` interface as the
toy dataset.

## Official Downloadable Datasets

- `WaterDrop`
- `Water`
- `Sand`
- `Goop`
- `MultiMaterial`
- `RandomFloor`
- `WaterRamps`
- `SandRamps`
- `FluidShake`
- `FluidShakeBox`
- `Continuous`
- `WaterDrop-XL`
- `Water-3D`
- `Sand-3D`
- `Goop-3D`
- `WaterDropSample`

## Recommended First Path

Start with `WaterDropSample` because it is small enough for loader and visual
smoke tests:

```bash
python scripts/download_data.py --dataset WaterDropSample --splits metadata train valid
python scripts/train.py --config-name lts_particle_jepa
python scripts/train.py --config-name lts_gns
python scripts/rollout.py --checkpoint runs/<gns-run>/checkpoints/last.pt --steps 64 --output runs/<gns-run>/visualizations/lts_rollout.gif
```

Then scale to `WaterRamps` or `SandRamps` for visually meaningful obstacle
generalization. Keep 3D datasets for a later pass, because the current rollout
visualizers are 2D-first.

## Current LTS Graph State

For 2D datasets each node has 7 features:

```text
x, y, vx, vy, particle_type, material_type, boundary_flag
```

Each edge has 6 features:

```text
dx, dy, dvx, dvy, distance, distance / radius
```

Velocities are frame-to-frame finite differences from official positions,
matching the official GNS convention where rollout integration uses implicit
`dt=1`. The graph radius defaults to `metadata.json/default_connectivity_radius`.

from __future__ import annotations

from particle_jepa.data.toy_dataset import ToyParticleConfig, ToyParticleDataset


def build_dataset(config: dict) -> ToyParticleDataset:
    name = config.get("dataset", config.get("name", "toy_particles"))
    if name != "toy_particles":
        msg = f"Unsupported dataset '{name}'. Add it in particle_jepa.data.dataset."
        raise ValueError(msg)
    aliases = {
        "trajectory_length": "sequence_length",
        "graph_radius": "radius",
        "horizon": "future_offset",
    }
    normalized = {aliases.get(key, key): value for key, value in config.items()}
    if "num_train_trajectories" in normalized or "num_val_trajectories" in normalized:
        normalized["num_trajectories"] = normalized.get(
            "num_train_trajectories", 0
        ) + normalized.get("num_val_trajectories", 0)
    fields = ToyParticleConfig.__dataclass_fields__
    kwargs = {key: value for key, value in normalized.items() if key in fields}
    return ToyParticleDataset(ToyParticleConfig(**kwargs))

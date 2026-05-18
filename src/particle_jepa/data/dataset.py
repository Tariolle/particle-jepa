from __future__ import annotations

from particle_jepa.data.lts_dataset import (
    LearningToSimulateConfig,
    LearningToSimulateDataset,
)
from particle_jepa.data.toy_dataset import ToyParticleConfig, ToyParticleDataset


def build_dataset(config: dict) -> ToyParticleDataset | LearningToSimulateDataset:
    name = config.get("dataset", config.get("name", "toy_particles"))
    if name in {"learning_to_simulate", "lts"}:
        aliases = {
            "data_root": "root",
            "graph_radius": "radius",
        }
        normalized = {aliases.get(key, key): value for key, value in config.items()}
        if isinstance(config.get("horizon"), list):
            normalized["future_offsets"] = config["horizon"]
        elif "horizon" in config:
            normalized["future_offset"] = config["horizon"]
        if "horizons" in config:
            normalized["future_offsets"] = config["horizons"]
        fields = LearningToSimulateConfig.__dataclass_fields__
        kwargs = {key: value for key, value in normalized.items() if key in fields}
        return LearningToSimulateDataset(LearningToSimulateConfig(**kwargs))
    if name != "toy_particles":
        msg = f"Unsupported dataset '{name}'. Add it in particle_jepa.data.dataset."
        raise ValueError(msg)
    aliases = {
        "trajectory_length": "sequence_length",
        "graph_radius": "radius",
    }
    normalized = {aliases.get(key, key): value for key, value in config.items()}
    if isinstance(config.get("horizon"), list):
        normalized["future_offsets"] = config["horizon"]
    elif "horizon" in config:
        normalized["future_offset"] = config["horizon"]
    if "horizons" in config:
        normalized["future_offsets"] = config["horizons"]
    if "num_train_trajectories" in normalized or "num_val_trajectories" in normalized:
        normalized["num_trajectories"] = normalized.get(
            "num_train_trajectories", 0
        ) + normalized.get("num_val_trajectories", 0)
    fields = ToyParticleConfig.__dataclass_fields__
    kwargs = {key: value for key, value in normalized.items() if key in fields}
    return ToyParticleDataset(ToyParticleConfig(**kwargs))

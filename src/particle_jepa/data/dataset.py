from __future__ import annotations

from particle_jepa.data.toy_dataset import ToyParticleConfig, ToyParticleDataset


def build_dataset(config: dict) -> ToyParticleDataset:
    name = config.get("name", "toy_particles")
    if name != "toy_particles":
        msg = f"Unsupported dataset '{name}'. Add it in particle_jepa.data.dataset."
        raise ValueError(msg)
    fields = ToyParticleConfig.__dataclass_fields__
    kwargs = {key: value for key, value in config.items() if key in fields}
    return ToyParticleDataset(ToyParticleConfig(**kwargs))

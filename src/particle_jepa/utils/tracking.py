from __future__ import annotations

from pathlib import Path
from typing import Any


class NullTracker:
    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        return None

    def finish(self) -> None:
        return None


class WandbTracker:
    def __init__(self, run) -> None:
        self.run = run

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self.run.log(metrics, step=step)

    def finish(self) -> None:
        self.run.finish()


def init_tracker(config: dict[str, Any], run_dir: str | Path):
    tracking = config.get("tracking", {})
    if not tracking.get("enabled", False):
        return NullTracker()
    if tracking.get("provider", "wandb") != "wandb":
        msg = f"Unsupported tracking provider: {tracking.get('provider')}"
        raise ValueError(msg)

    import wandb

    run = wandb.init(
        project=tracking.get("project", "particle-jepa"),
        entity=tracking.get("entity"),
        tags=tracking.get("tags"),
        mode=tracking.get("mode", "online"),
        dir=str(run_dir),
        config=config.get("raw_config", config),
        name=config.get("project", {}).get("run_name"),
    )
    return WandbTracker(run)

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


def create_run_dir(model_name: str, run_root: str | Path = "runs") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(run_root) / f"{timestamp}_{model_name}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "visualizations").mkdir(parents=True, exist_ok=True)
    return run_dir


def copy_config(config_path: str | Path, run_dir: Path) -> None:
    shutil.copy2(config_path, run_dir / "config.yaml")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")

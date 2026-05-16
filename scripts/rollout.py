# ruff: noqa: E402, I001
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from particle_jepa.data.toy_dataset import ToyParticleConfig, generate_toy_trajectories
from particle_jepa.evaluation.rollout_eval import rollout_error


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a toy rollout sanity check.")
    parser.add_argument("--steps", type=int, default=16)
    args = parser.parse_args()

    config = ToyParticleConfig(sequence_length=max(args.steps, 2), num_trajectories=1)
    positions, _velocities = generate_toy_trajectories(config)
    error = rollout_error(positions[0, 1:], positions[0, :-1])
    print(f"toy adjacent-frame displacement: {error.item():.4f}")


if __name__ == "__main__":
    main()

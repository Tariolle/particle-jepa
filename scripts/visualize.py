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
from particle_jepa.visualization.particles import plot_particles


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize generated toy particles.")
    parser.add_argument("--output", default="outputs/toy_particles.png")
    parser.add_argument("--time", type=int, default=0)
    parser.add_argument(
        "--run", default=None, help="Use 'latest' to save into the latest run directory."
    )
    args = parser.parse_args()

    output = _resolve_output(args.output, args.run)
    config = ToyParticleConfig(num_trajectories=1)
    positions, _ = generate_toy_trajectories(config)
    plot_particles(positions[0, args.time], output=output, title=f"Toy particles t={args.time}")
    print(f"saved visualization: {output}")


def _resolve_output(default_output: str, run: str | None) -> Path:
    if run != "latest":
        return Path(default_output)
    runs = sorted(Path("runs").glob("*"))
    if not runs:
        return Path(default_output)
    return runs[-1] / "visualizations" / "toy_particles.png"


if __name__ == "__main__":
    main()

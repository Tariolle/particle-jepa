from __future__ import annotations

import argparse

from particle_jepa.data.toy_dataset import ToyParticleConfig, generate_toy_trajectories
from particle_jepa.visualization.particles import plot_particles


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize generated toy particles.")
    parser.add_argument("--output", default="outputs/toy_particles.png")
    parser.add_argument("--time", type=int, default=0)
    args = parser.parse_args()

    config = ToyParticleConfig(num_trajectories=1)
    positions, _ = generate_toy_trajectories(config)
    plot_particles(
        positions[0, args.time], output=args.output, title=f"Toy particles t={args.time}"
    )
    print(f"saved visualization: {args.output}")


if __name__ == "__main__":
    main()

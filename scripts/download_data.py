from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset download scaffold.")
    parser.add_argument("--output-dir", default="data/raw")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    print(
        "DeepMind Learning-to-Simulate download integration is intentionally left "
        f"as an extension point. Place raw exports under {output}."
    )


if __name__ == "__main__":
    main()

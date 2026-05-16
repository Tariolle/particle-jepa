from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess particle datasets into graph-ready tensors."
    )
    parser.add_argument("--input-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"preprocess scaffold: {args.input_dir} -> {args.output_dir}")


if __name__ == "__main__":
    main()

# ruff: noqa: E402, I001
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from particle_jepa.data.lts_dataset import BASE_URL, LTS_DATASETS


def main() -> None:
    parser = argparse.ArgumentParser(description="Download DeepMind Learning-to-Simulate data.")
    parser.add_argument("--dataset", default="WaterDropSample", choices=LTS_DATASETS)
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["metadata", "train", "valid", "test"],
        choices=["metadata", "train", "valid", "test"],
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_dir) / args.dataset
    output.mkdir(parents=True, exist_ok=True)
    filenames = [
        "metadata.json" if split == "metadata" else f"{split}.tfrecord" for split in args.splits
    ]
    for filename in filenames:
        url = f"{BASE_URL}/{args.dataset}/{filename}"
        destination = output / filename
        if destination.exists() and not args.force:
            print(f"exists: {destination}")
            continue
        _download(url, destination)
        print(f"downloaded: {destination}")


def _download(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("Content-Length", 0))
        progress = tqdm(total=total, unit="B", unit_scale=True, desc=destination.name)
        with destination.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
                progress.update(len(chunk))
        progress.close()


if __name__ == "__main__":
    main()

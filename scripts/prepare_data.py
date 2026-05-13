#!/usr/bin/env python3
"""Prepare public datasets used by the CornerDrive reproduction scripts."""

from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CornerDrive datasets.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "real")
    parser.add_argument(
        "--download-torchvision",
        action="store_true",
        help="Download MNIST and FashionMNIST into data/real.",
    )
    parser.add_argument(
        "--check-femnist",
        action="store_true",
        help="Check whether LEAF/FEMNIST JSON files are present.",
    )
    return parser.parse_args()


def download_torchvision(data_dir: Path) -> None:
    from torchvision.datasets import FashionMNIST, MNIST

    data_dir.mkdir(parents=True, exist_ok=True)
    for dataset in (MNIST, FashionMNIST):
        for train in (True, False):
            dataset(root=str(data_dir), train=train, download=True)
        print(f"prepared {dataset.__name__} under {data_dir}")


def check_femnist(data_dir: Path) -> None:
    femnist_root = data_dir / "femnist"
    train = femnist_root / "train"
    test = femnist_root / "test"
    train_files = sorted(train.glob("*.json")) if train.exists() else []
    test_files = sorted(test.glob("*.json")) if test.exists() else []
    if train_files and test_files:
        print(
            f"found FEMNIST LEAF files: {len(train_files)} train shards, "
            f"{len(test_files)} test shards"
        )
        return
    print(
        "FEMNIST LEAF files not found. Place LEAF-format JSON shards under "
        f"{train} and {test} before reproducing the FEMNIST real-gradient rows."
    )


def main() -> int:
    args = parse_args()
    if args.download_torchvision:
        download_torchvision(args.data_dir)
    if args.check_femnist:
        check_femnist(args.data_dir)
    if not args.download_torchvision and not args.check_femnist:
        print("Nothing requested. Use --download-torchvision and/or --check-femnist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

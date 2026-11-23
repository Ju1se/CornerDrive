#!/usr/bin/env python3
"""Annotate one quarter of CIFAR-10 as the golden dataset (normal vs rare).

Writes, under data/validation/golden/cifar10/:

    manifest.csv         one row per image: partition, golden label, test role
    summary.json         rule, seed, counts, source md5
    golden_preview.png   a contact sheet of the two golden sets

The other three quarters stay unannotated and are used for testing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from dual_dataset_benchmark.cifar_golden import (  # noqa: E402
    ANNOTATION_SEED,
    CLASS_NAMES,
    GOLDEN_FRACTION,
    NORMAL_LABELS,
    RARE_LABELS,
    annotate_golden,
    load_cifar10,
    write_manifest,
)


def write_preview(data, ann, path: Path, per_class: int = 12) -> None:
    from PIL import Image, ImageDraw

    x, y = data["x"], data["y"]
    cell, pad, label_w = 32, 2, 150
    rows = [("normal", k) for k in NORMAL_LABELS] + [("rare", k) for k in RARE_LABELS]
    width = label_w + per_class * (cell + pad)
    height = len(rows) * (cell + pad) + pad
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    for r, (tag, k) in enumerate(rows):
        source = ann.golden_rare if tag == "rare" else ann.golden_normal
        picks = source[y[source] == k][:per_class]
        top = pad + r * (cell + pad)
        draw.text((4, top + 10), f"{tag}: {CLASS_NAMES[k]}",
                  fill=(170, 30, 30) if tag == "rare" else (40, 40, 40))
        for c, idx in enumerate(picks):
            tile = Image.fromarray(np.transpose(x[idx], (1, 2, 0)))
            sheet.paste(tile, (label_w + c * (cell + pad), top))
    sheet.resize((width * 2, height * 2), Image.NEAREST).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cifar-dir", default="data/real/cifar-10-batches-py")
    parser.add_argument("--out-dir", default="data/validation/golden/cifar10")
    parser.add_argument("--seed", type=int, default=ANNOTATION_SEED)
    parser.add_argument("--golden-fraction", type=float, default=GOLDEN_FRACTION)
    args = parser.parse_args()

    data = load_cifar10(PROJECT_ROOT / args.cifar_dir)
    ann = annotate_golden(data["y"], seed=args.seed, golden_fraction=args.golden_fraction)
    out_dir = PROJECT_ROOT / args.out_dir
    summary = write_manifest(data, ann, out_dir)
    write_preview(data, ann, out_dir / "golden_preview.png")
    print(json.dumps({k: summary[k] for k in ("counts", "per_class", "rare_labels")}, indent=2))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()

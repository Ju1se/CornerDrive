#!/usr/bin/env python3
"""Embed all 60,000 CIFAR-10 images with a frozen ImageNet ResNet-50.

The embedding model takes part in the federated loop as a fixed public
encoder: clients and the server feed its 2048-d pooled features to the
federated head. This script downloads the official torchvision weights
(``resnet50-11ad3fa6.pth``, IMAGENET1K_V2, about 98 MB, hash-checked by
torchvision) on first use and writes

    data/real/cifar10_resnet50_embeddings.npy    float16 [60000, 2048]
    data/real/cifar10_resnet50_embeddings.json   provenance

Row i is pool image i (official train split first, then the test split), the
same indexing as the golden manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from dual_dataset_benchmark.cifar_golden import load_cifar10  # noqa: E402

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cifar-dir", default="data/real/cifar-10-batches-py")
    parser.add_argument("--out", default="data/real/cifar10_resnet50_embeddings.npy")
    parser.add_argument("--batch", type=int, default=200)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from torchvision.models import ResNet50_Weights, resnet50

    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    device = torch.device(args.device)
    model = model.to(device).eval().to(memory_format=torch.channels_last)

    data = load_cifar10(PROJECT_ROOT / args.cifar_dir)
    images = torch.from_numpy(data["x"])
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    out = np.zeros((len(images), 2048), dtype=np.float16)
    started = time.time()
    with torch.no_grad():
        for start in range(0, len(images), args.batch):
            batch = images[start:start + args.batch].to(device).float() / 255.0
            batch = F.interpolate(batch, size=args.size, mode="bilinear", align_corners=False)
            batch = ((batch - mean) / std).contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                features = model(batch)
            out[start:start + args.batch] = features.float().cpu().numpy().astype(np.float16)
            if (start // args.batch) % 50 == 0:
                print(f"{start + len(batch)}/{len(images)}  {time.time() - started:.0f}s", flush=True)

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, out)
    out_path.with_suffix(".json").write_text(json.dumps({
        "encoder": "torchvision resnet50, ResNet50_Weights.IMAGENET1K_V2, fc removed",
        "weights_url": weights.url,
        "input": f"32x32 bilinear-upsampled to {args.size}x{args.size}, ImageNet mean/std",
        "shape": list(out.shape),
        "dtype": "float16",
        "row_order": "pool index: official train split 0..49999, test split 50000..59999",
        "source_md5": data["md5"],
    }, indent=2))
    print(f"wrote {out_path}  ({out.nbytes / 1e6:.0f} MB, {time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()

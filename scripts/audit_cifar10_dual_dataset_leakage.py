#!/usr/bin/env python3
"""Leakage audit of the CIFAR-10 dual-dataset benchmark.

Checks that can be run on the data itself:

1. the golden, client-pool and evaluation surfaces are disjoint and complete;
2. no image appears byte-identically on two surfaces;
3. how many images have a near-duplicate on another surface (cosine similarity
   of the frozen ResNet-50 embeddings), with a contact sheet of the closest
   pairs so the threshold can be checked by eye. CIFAR-10 contains repeated
   photographs; they are reported, not removed, because a server's golden data
   and its clients' data overlap in the same way in practice;
4. every client shard of every development and confirmatory seed is drawn
   from the client pool only and shares no image with another shard;
5. the development and confirmatory seed sets are disjoint.

    python scripts/audit_cifar10_dual_dataset_leakage.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from dual_dataset_benchmark.cifar_golden import (  # noqa: E402
    CLASS_NAMES,
    FederationConfig,
    annotate_golden,
    build_federation,
    load_cifar10,
)

DEV_SEEDS = (20261101, 20261102)
CONFIRM_SEEDS = (20261110, 20261111, 20261112, 20261113, 20261114)
THRESHOLDS = (0.90, 0.95, 0.98)
NEAR_DUPLICATE = 0.90      # pairs at or above this are the same photograph (see the contact sheet)


def nearest(query: torch.Tensor, bank: torch.Tensor, chunk: int = 2000):
    """Best cosine similarity of every query row inside the bank (+ its index)."""
    best, where = [], []
    for start in range(0, len(query), chunk):
        sims = query[start:start + chunk] @ bank.t()
        value, index = sims.max(dim=1)
        best.append(value)
        where.append(index)
    return torch.cat(best).cpu().numpy(), torch.cat(where).cpu().numpy()


def contact_sheet(x: np.ndarray, pairs: list[tuple[int, int, float]], path: Path) -> None:
    from PIL import Image, ImageDraw

    cell, pad, cols = 64, 6, 6
    rows = (len(pairs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (2 * cell + 3 * pad), rows * (cell + 18 + pad) + pad), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (a, b, sim) in enumerate(pairs):
        left = (i % cols) * (2 * cell + 3 * pad) + pad
        top = (i // cols) * (cell + 18 + pad) + pad
        for j, idx in enumerate((a, b)):
            tile = Image.fromarray(np.transpose(x[idx], (1, 2, 0))).resize((cell, cell), Image.NEAREST)
            sheet.paste(tile, (left + j * (cell + 2), top))
        draw.text((left, top + cell + 2), f"cos {sim:.3f}", fill=(40, 40, 40))
    sheet.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cifar-dir", default="data/real/cifar-10-batches-py")
    parser.add_argument("--embeddings", default="data/real/cifar10_resnet50_embeddings.npy")
    parser.add_argument("--out-dir", default="results/cifar10_dual_dataset/leakage_audit")
    parser.add_argument("--sheet", default="artifacts/figures/cifar10_near_duplicates_eval_vs_client_pool.png")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    data = load_cifar10(PROJECT_ROOT / args.cifar_dir)
    x, y = data["x"], data["y"]
    ann = annotate_golden(y)
    surfaces = {"golden": ann.golden, "client_pool": ann.client_pool, "eval": ann.eval}

    # 1. disjoint and complete
    everything = np.concatenate(list(surfaces.values()))
    report["surfaces_disjoint_and_complete"] = bool(len(np.unique(everything)) == len(y) == len(everything))
    surface_of = np.empty(len(y), dtype=object)
    for name, idx in surfaces.items():
        surface_of[idx] = name

    # 2. byte-identical images on two surfaces
    groups: dict[str, list[int]] = defaultdict(list)
    for i in range(len(x)):
        groups[hashlib.md5(x[i].tobytes()).hexdigest()].append(i)
    crossing = [g for g in groups.values() if len({surface_of[i] for i in g}) > 1]
    report["byte_identical_images"] = {
        "groups_within_dataset": int(sum(len(g) > 1 for g in groups.values())),
        "groups_crossing_surfaces": len(crossing),
    }

    # 3. near-duplicates by embedding similarity
    emb = torch.from_numpy(np.load(PROJECT_ROOT / args.embeddings)).float().to(args.device)
    emb = torch.nn.functional.normalize(emb, dim=1)
    near = {}
    examples = []
    for query_name, bank_name in (("eval", "client_pool"), ("golden", "client_pool"), ("eval", "golden")):
        q, b = surfaces[query_name], surfaces[bank_name]
        sims, where = nearest(emb[torch.from_numpy(q).to(args.device)],
                              emb[torch.from_numpy(b).to(args.device)])
        same_label = y[q] == y[b[where]]
        near[f"{query_name}_vs_{bank_name}"] = {
            "images": int(len(q)),
            "median_best_similarity": float(np.median(sims)),
            **{f"best_similarity_at_least_{t:g}": int((sims >= t).sum()) for t in THRESHOLDS},
            **{f"share_at_least_{t:g}": float((sims >= t).mean()) for t in THRESHOLDS},
            "same_label_among_at_least_0.95": float(same_label[sims >= 0.95].mean())
            if (sims >= 0.95).any() else None,
        }
        hit = sims >= NEAR_DUPLICATE
        near[f"{query_name}_vs_{bank_name}"]["share_by_class_at_least_0.9"] = {
            CLASS_NAMES[k]: float(hit[y[q] == k].mean()) for k in range(len(CLASS_NAMES))
        }
        order = np.argsort(-sims)
        if query_name == "eval" and bank_name == "client_pool":
            picks = list(order[:12]) + [order[np.searchsorted(-sims[order], -t)] for t in
                                         (0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90,
                                          0.88, 0.86, 0.84)]
            examples = [(int(q[i]), int(b[where[i]]), float(sims[i])) for i in picks]
    report["near_duplicates"] = near
    sheet_path = PROJECT_ROOT / args.sheet
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    contact_sheet(x, examples, sheet_path)

    # 4. client shards of every seed
    shards = {}
    golden_or_eval = np.concatenate([ann.golden, ann.eval])
    for seed in (*DEV_SEEDS, *CONFIRM_SEEDS):
        for scenario in ("mixed", "no_attack"):
            fed = build_federation(y, ann.client_pool, seed=seed,
                                   config=FederationConfig(scenario=scenario))
            flat = fed.client_index.ravel()
            ok = (len(np.unique(flat)) == flat.size
                  and bool(np.isin(flat, ann.client_pool).all())
                  and not bool(np.isin(flat, golden_or_eval).any()))
            shards[f"{seed}:{scenario}"] = bool(ok)
    report["client_shards_clean"] = bool(all(shards.values()))

    # 5. seed sets
    report["dev_and_confirm_seeds_disjoint"] = not (set(DEV_SEEDS) & set(CONFIRM_SEEDS))
    report["note"] = (
        "The golden quarter, the client-pool images and the evaluation images are the same "
        "for every seed; a seed changes the client shards, the roles, the initial weights and "
        "the sampling."
    )

    (out_dir / "leakage_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"contact sheet: {sheet_path}")


if __name__ == "__main__":
    main()

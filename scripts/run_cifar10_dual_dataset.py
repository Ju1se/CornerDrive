#!/usr/bin/env python3
"""Run units of the CIFAR-10 dual-dataset rarity benchmark (resumable).

Every (seed, scenario, method) trajectory is written to

    <out-dir>/unit_<scenario>_<method>_s<seed>.npz

as soon as it finishes and is never recomputed: stop the run at any time and
start the same command again to resume.

Examples
--------
    # development seeds: reference trajectories with every score logged
    python scripts/run_cifar10_dual_dataset.py --stage dev --seeds 20261101,20261102 \
        --methods oracle,fedavg --extra-audit-steps 0.5,0.25

    # confirmatory seeds with the frozen policy
    python scripts/run_cifar10_dual_dataset.py --stage confirm \
        --seeds 20261110,20261111,20261112,20261113,20261114 \
        --policy configs/cifar10_dual_dataset_policy.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from dual_dataset_benchmark.backend import CNN, head_backend  # noqa: E402
from dual_dataset_benchmark.cifar_golden import (  # noqa: E402
    FederationConfig,
    annotate_golden,
    load_cifar10,
)
from dual_dataset_benchmark.engine import (  # noqa: E402
    CORNERDRIVE,
    METHODS,
    CornerDrivePolicy,
    EngineConfig,
    SeedContext,
    run_unit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", default="dev", help="sub-folder of results/cifar10_dual_dataset")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--cifar-dir", default="data/real/cifar-10-batches-py")
    parser.add_argument("--model", default="cnn", choices=("cnn", "r50head"),
                        help="cnn: CNN from scratch; r50head: MLP head on frozen ResNet-50 embeddings")
    parser.add_argument("--embeddings", default="data/real/cifar10_resnet50_embeddings.npy")
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--scenarios", default="mixed")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--policy", default="", help="CornerDrive policy JSON")
    parser.add_argument("--rounds", type=int, default=EngineConfig.rounds)
    parser.add_argument("--warmup-rounds", type=int, default=EngineConfig.warmup_rounds)
    parser.add_argument("--local-lr", type=float, default=EngineConfig.local_lr)
    parser.add_argument("--local-epochs", type=int, default=EngineConfig.local_epochs)
    parser.add_argument("--audit-per-class", type=int, default=EngineConfig.audit_per_class)
    parser.add_argument("--extra-audit-steps", default="")
    parser.add_argument("--rare-rich-share", type=float, default=FederationConfig.rare_rich_share)
    parser.add_argument("--corner-proxy", default="rare", choices=("rare", "partial", "random_main"))
    parser.add_argument("--log-all-scores", action="store_true")
    parser.add_argument("--tag", default="", help="suffix added to unit file names")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else (
        PROJECT_ROOT / "results" / "cifar10_dual_dataset" / args.stage)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for method in methods:
        if method not in METHODS:
            raise SystemExit(f"unknown method {method!r}; choose from {', '.join(METHODS)}")

    policies = None
    if args.policy:
        policy_path = Path(args.policy)
        if not policy_path.is_absolute():
            policy_path = PROJECT_ROOT / policy_path
        payload = json.loads(policy_path.read_text())
        if payload.get("model") not in (None, args.model):
            raise SystemExit(f"policy was calibrated for {payload['model']}, not {args.model}")
        policies = {name: CornerDrivePolicy.from_json(entry)
                    for name, entry in payload["policies"].items()}
    if policies is None and any(m in CORNERDRIVE for m in methods):
        raise SystemExit("CornerDrive methods need --policy")

    cfg = EngineConfig(
        rounds=args.rounds,
        warmup_rounds=args.warmup_rounds,
        local_lr=args.local_lr,
        local_epochs=args.local_epochs,
        audit_per_class=args.audit_per_class,
        extra_audit_steps=tuple(float(v) for v in args.extra_audit_steps.split(",") if v.strip()),
        corner_proxy=args.corner_proxy,
    )
    fed_cfg = FederationConfig(rare_rich_share=args.rare_rich_share)
    (out_dir / "run_args.json").write_text(json.dumps({
        "args": vars(args), "engine": asdict(cfg), "federation": asdict(fed_cfg),
        "policies": {k: v.to_json() for k, v in policies.items()} if policies else None,
    }, indent=2))

    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    data = load_cifar10(PROJECT_ROOT / args.cifar_dir)
    ann = annotate_golden(data["y"])
    y_dev = torch.from_numpy(data["y"]).to(device)
    if args.model == "cnn":
        backend = CNN
        x_dev = torch.from_numpy(data["x"]).to(device)
    else:
        # Standardise every embedding dimension with statistics of the golden
        # quarter only (the data the server is allowed to look at).
        emb = torch.from_numpy(np.load(PROJECT_ROOT / args.embeddings)).float().to(device)
        golden = torch.from_numpy(ann.golden).to(device)
        mean, std = emb[golden].mean(dim=0), emb[golden].std(dim=0).clamp_min(1e-3)
        x_dev = (emb - mean) / std
        backend = head_backend(x_dev.shape[1])

    todo = [(seed, scenario, method) for seed in seeds for scenario in scenarios
            for method in methods]
    suffix = f"_{args.tag}" if args.tag else ""

    def unit_path(seed: int, scenario: str, method: str) -> Path:
        return out_dir / f"unit_{scenario}_{method}_s{seed}{suffix}.npz"

    pending = [u for u in todo if not unit_path(*u).exists()]
    print(f"{len(todo)} units, {len(todo) - len(pending)} already done, {len(pending)} to run",
          flush=True)
    started = time.time()
    done = 0
    contexts: dict[int, SeedContext] = {}
    for seed, scenario, method in pending:
        if seed not in contexts:
            contexts.clear()
            contexts[seed] = SeedContext(data, ann, seed, cfg, fed_cfg, device,
                                         x_device=x_dev, y_device=y_dev, backend=backend)
            contexts[seed].warmup(cache_dir=out_dir)
        ctx = contexts[seed]
        result = run_unit(ctx, scenario, method, policies,
                          log_all_scores=True if args.log_all_scores else None)
        tmp = unit_path(seed, scenario, method).with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **result)
        tmp.replace(unit_path(seed, scenario, method))
        done += 1
        elapsed = time.time() - started
        eta = elapsed / done * (len(pending) - done)
        tail = slice(-10, None)
        print(
            f"[{done}/{len(pending)}] s{seed} {scenario:9s} {method:15s} "
            f"normal {np.mean(result['r_acc_normal'][tail]):.3f} "
            f"rare {np.mean(result['r_acc_rare'][tail]):.3f} "
            f"({json.loads(result['meta'])['seconds']:.0f}s, eta {eta / 60:.1f} min)",
            flush=True,
        )


if __name__ == "__main__":
    main()

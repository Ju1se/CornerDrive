#!/usr/bin/env python3
"""Export a CornerDrive benchmark built from real client data gradients."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"

for candidate in (PROJECT_ROOT, BACKEND_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from policy_agent.analysis.real_gradient_benchmark import (  # noqa: E402
    RealGradientBenchmarkConfig,
    run_real_gradient_benchmark,
    write_real_gradient_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a real-data gradient benchmark. Prefer LEAF/FEMNIST JSON via "
            "--source leaf_femnist; otherwise use torchvision MNIST/FashionMNIST."
        )
    )
    parser.add_argument(
        "--source",
        default="auto",
        choices=["auto", "leaf_femnist", "femnist", "mnist", "fashionmnist", "torchvision_mnist", "torchvision_fashionmnist"],
    )
    parser.add_argument("--leaf-data-dir", default="data/real/femnist")
    parser.add_argument("--data-dir", default="data/real")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-clients", type=int, default=80)
    parser.add_argument("--min-samples-per-client", type=int, default=8)
    parser.add_argument("--max-samples-per-client", type=int, default=32)
    parser.add_argument("--clients-per-round", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--pretrain-steps", type=int, default=40)
    parser.add_argument("--local-batch-size", type=int, default=16)
    parser.add_argument("--attack-fraction", type=float, default=0.20)
    parser.add_argument("--corner-harm-fraction", type=float, default=0.05)
    parser.add_argument("--noise-fraction", type=float, default=0.05)
    parser.add_argument("--sign-flip-scale", type=float, default=3.0)
    parser.add_argument("--corner-harm-scale", type=float, default=2.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "real_gradient_benchmark",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RealGradientBenchmarkConfig(
        source=args.source,
        leaf_data_dir=args.leaf_data_dir,
        data_dir=args.data_dir,
        download=args.download,
        max_clients=args.max_clients,
        min_samples_per_client=args.min_samples_per_client,
        max_samples_per_client=args.max_samples_per_client,
        clients_per_round=args.clients_per_round,
        rounds=args.rounds,
        seed=args.seed,
        pretrain_steps=args.pretrain_steps,
        local_batch_size=args.local_batch_size,
        attack_fraction=args.attack_fraction,
        corner_harm_fraction=args.corner_harm_fraction,
        noise_fraction=args.noise_fraction,
        sign_flip_scale=args.sign_flip_scale,
        corner_harm_scale=args.corner_harm_scale,
    )
    result = run_real_gradient_benchmark(config)
    write_real_gradient_outputs(result, args.output_dir)
    summary = result["methods"]["cornerdrive"]["summary"]
    print(f"Wrote real-gradient benchmark artifacts to {args.output_dir}")
    print(
        "CornerDrive: "
        f"main_acc={summary['main_accuracy_avg']:.4f}, "
        f"corner_acc={summary['corner_accuracy_avg']:.4f}, "
        f"fraud_survival={summary['fraud_survival_rate_avg']:.4f}, "
        f"rarity_retention={summary['rarity_retention_rate_avg']:.4f}"
    )


if __name__ == "__main__":
    main()

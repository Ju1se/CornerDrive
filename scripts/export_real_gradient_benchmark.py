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

from common.schemas import DEFAULT_POLICY  # noqa: E402
from policy_agent.analysis.real_gradient_benchmark import (  # noqa: E402
    RealGradientBenchmarkConfig,
    make_real_data_adaptive_v41_policy,
    run_real_gradient_benchmark,
    write_real_gradient_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a real-data gradient benchmark. Prefer LEAF/FEMNIST JSON via "
            "--source leaf_femnist; use --source bdd100k for IoV image "
            "attribute pseudo-clients; otherwise use torchvision MNIST/FashionMNIST."
        )
    )
    parser.add_argument(
        "--source",
        default="auto",
        choices=[
            "auto",
            "leaf_femnist",
            "femnist",
            "bdd",
            "bdd100k",
            "mnist",
            "fashionmnist",
            "torchvision_mnist",
            "torchvision_fashionmnist",
        ],
    )
    parser.add_argument("--leaf-data-dir", default="data/real/femnist")
    parser.add_argument("--bdd-data-dir", default="data/real/bdd100k")
    parser.add_argument("--bdd-label-file", default="")
    parser.add_argument("--bdd-image-dir", default="")
    parser.add_argument("--bdd-image-size", type=int, default=32)
    parser.add_argument(
        "--bdd-target-attribute",
        choices=["weather", "timeofday", "scene"],
        default="weather",
    )
    parser.add_argument("--bdd-client-group", default="weather_timeofday")
    parser.add_argument("--bdd-corner-values", default="rainy,snowy,foggy")
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
    parser.add_argument("--reference-split-fraction", type=float, default=0.50)
    parser.add_argument("--max-reference-samples", type=int, default=4096)
    parser.add_argument("--max-evaluation-samples", type=int, default=4096)
    parser.add_argument("--attack-fraction", type=float, default=0.20)
    parser.add_argument("--corner-harm-fraction", type=float, default=0.05)
    parser.add_argument("--noise-fraction", type=float, default=0.05)
    parser.add_argument("--rarity-label-fraction-threshold", type=float, default=0.30)
    parser.add_argument("--sign-flip-scale", type=float, default=3.0)
    parser.add_argument("--corner-harm-scale", type=float, default=2.0)
    parser.add_argument("--zeno-score-penalty", type=float, default=1e-4)
    parser.add_argument("--zenopp-score-temperature", type=float, default=0.05)
    parser.add_argument(
        "--policy-profile",
        choices=["default", "real_data_adaptive", "real_data_adaptive_v41"],
        default="real_data_adaptive_v41",
        help=(
            "CornerDrive policy profile. real_data_adaptive and "
            "real_data_adaptive_v41 both use the calibrated V4.1 real-gradient "
            "profile; default preserves the original baseline policy."
        ),
    )
    parser.add_argument("--theta-tol", type=float, default=None)
    parser.add_argument("--theta-rare", type=float, default=None)
    parser.add_argument("--theta-rarity-main-tol", type=float, default=None)
    parser.add_argument("--cosine-filter-threshold", type=float, default=None)
    parser.add_argument("--recheck-probability", type=float, default=None)
    parser.add_argument("--cornerdrive-l1-mode", default=None)
    parser.add_argument("--cornerdrive-l1-cos-weight", type=float, default=None)
    parser.add_argument("--cornerdrive-l1-norm-weight", type=float, default=None)
    parser.add_argument("--cornerdrive-l1-sign-weight", type=float, default=None)
    parser.add_argument("--cornerdrive-l1-norm-mad-threshold", type=float, default=None)
    parser.add_argument("--cornerdrive-l1-sign-threshold", type=float, default=None)
    parser.add_argument("--cornerdrive-l1-sign-topk-ratio", type=float, default=None)
    parser.add_argument("--cornerdrive-l1-queue-budget-ratio", type=float, default=None)
    parser.add_argument("--cornerdrive-l1-random-recheck-ratio", type=float, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "real_gradient_benchmark",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.policy_profile in {"real_data_adaptive", "real_data_adaptive_v41"}:
        l1_defaults = {
            "cornerdrive_l1_mode": "v4_m4_dual_proxy_budgeted",
            "cornerdrive_l1_cos_weight": 0.35,
            "cornerdrive_l1_norm_weight": 0.20,
            "cornerdrive_l1_sign_weight": 0.15,
            "cornerdrive_l1_norm_mad_threshold": 1.5,
            "cornerdrive_l1_sign_threshold": 0.40,
            "cornerdrive_l1_sign_topk_ratio": 0.10,
            "cornerdrive_l1_queue_budget_ratio": 0.80,
            "cornerdrive_l1_random_recheck_ratio": 0.05,
        }
    else:
        l1_defaults = {
            "cornerdrive_l1_mode": "v25_cosine_fixed",
            "cornerdrive_l1_cos_weight": 0.35,
            "cornerdrive_l1_norm_weight": 0.20,
            "cornerdrive_l1_sign_weight": 0.15,
            "cornerdrive_l1_norm_mad_threshold": 3.0,
            "cornerdrive_l1_sign_threshold": 0.65,
            "cornerdrive_l1_sign_topk_ratio": 0.10,
            "cornerdrive_l1_queue_budget_ratio": 0.35,
            "cornerdrive_l1_random_recheck_ratio": 0.05,
        }

    config = RealGradientBenchmarkConfig(
        source=args.source,
        leaf_data_dir=args.leaf_data_dir,
        bdd_data_dir=args.bdd_data_dir,
        bdd_label_file=args.bdd_label_file,
        bdd_image_dir=args.bdd_image_dir,
        bdd_image_size=args.bdd_image_size,
        bdd_target_attribute=args.bdd_target_attribute,
        bdd_client_group=args.bdd_client_group,
        bdd_corner_values=args.bdd_corner_values,
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
        reference_split_fraction=args.reference_split_fraction,
        max_reference_samples=args.max_reference_samples,
        max_evaluation_samples=args.max_evaluation_samples,
        attack_fraction=args.attack_fraction,
        corner_harm_fraction=args.corner_harm_fraction,
        noise_fraction=args.noise_fraction,
        rarity_label_fraction_threshold=args.rarity_label_fraction_threshold,
        sign_flip_scale=args.sign_flip_scale,
        corner_harm_scale=args.corner_harm_scale,
        zeno_score_penalty=args.zeno_score_penalty,
        zenopp_score_temperature=args.zenopp_score_temperature,
        cornerdrive_l1_mode=args.cornerdrive_l1_mode or l1_defaults["cornerdrive_l1_mode"],
        cornerdrive_l1_cos_weight=(
            args.cornerdrive_l1_cos_weight
            if args.cornerdrive_l1_cos_weight is not None
            else l1_defaults["cornerdrive_l1_cos_weight"]
        ),
        cornerdrive_l1_norm_weight=(
            args.cornerdrive_l1_norm_weight
            if args.cornerdrive_l1_norm_weight is not None
            else l1_defaults["cornerdrive_l1_norm_weight"]
        ),
        cornerdrive_l1_sign_weight=(
            args.cornerdrive_l1_sign_weight
            if args.cornerdrive_l1_sign_weight is not None
            else l1_defaults["cornerdrive_l1_sign_weight"]
        ),
        cornerdrive_l1_norm_mad_threshold=(
            args.cornerdrive_l1_norm_mad_threshold
            if args.cornerdrive_l1_norm_mad_threshold is not None
            else l1_defaults["cornerdrive_l1_norm_mad_threshold"]
        ),
        cornerdrive_l1_sign_threshold=(
            args.cornerdrive_l1_sign_threshold
            if args.cornerdrive_l1_sign_threshold is not None
            else l1_defaults["cornerdrive_l1_sign_threshold"]
        ),
        cornerdrive_l1_sign_topk_ratio=(
            args.cornerdrive_l1_sign_topk_ratio
            if args.cornerdrive_l1_sign_topk_ratio is not None
            else l1_defaults["cornerdrive_l1_sign_topk_ratio"]
        ),
        cornerdrive_l1_queue_budget_ratio=(
            args.cornerdrive_l1_queue_budget_ratio
            if args.cornerdrive_l1_queue_budget_ratio is not None
            else l1_defaults["cornerdrive_l1_queue_budget_ratio"]
        ),
        cornerdrive_l1_random_recheck_ratio=(
            args.cornerdrive_l1_random_recheck_ratio
            if args.cornerdrive_l1_random_recheck_ratio is not None
            else l1_defaults["cornerdrive_l1_random_recheck_ratio"]
        ),
    )
    if args.policy_profile in {"real_data_adaptive", "real_data_adaptive_v41"}:
        policy = make_real_data_adaptive_v41_policy()
    else:
        policy = DEFAULT_POLICY
    policy_updates = {
        key: value
        for key, value in {
            "theta_tol": args.theta_tol,
            "theta_rare": args.theta_rare,
            "theta_rarity_main_tol": args.theta_rarity_main_tol,
            "cosine_filter_threshold": args.cosine_filter_threshold,
            "recheck_probability": args.recheck_probability,
        }.items()
        if value is not None
    }
    policy = policy.model_copy(update=policy_updates) if policy_updates else policy
    result = run_real_gradient_benchmark(config, policy=policy)
    write_real_gradient_outputs(result, args.output_dir)
    summary = result["methods"]["cornerdrive"]["summary"]
    print(f"Wrote real-gradient benchmark artifacts to {args.output_dir}")
    print(
        "CornerDrive: "
        f"main_acc={summary['main_accuracy_avg']:.4f}, "
        f"corner_acc={summary['corner_accuracy_avg']:.4f}, "
        f"fraud_survival={summary['fraud_survival_rate_avg']:.4f}, "
        f"rarity_retention={summary['rarity_retention_rate_avg']:.4f}, "
        f"l1_review_rate={summary.get('l1_review_rate_avg', 0.0):.4f}"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run larger multi-seed real-gradient benchmarks with confidence intervals."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"

for candidate in (PROJECT_ROOT, BACKEND_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from common.schemas import DEFAULT_POLICY  # noqa: E402
from policy_agent.analysis.real_gradient_benchmark import (  # noqa: E402
    RealGradientBenchmarkConfig,
    make_real_data_adaptive_policy,
    make_real_data_adaptive_v41_policy,
    run_real_gradient_benchmark,
    write_real_gradient_outputs,
)


DEFAULT_SOURCES = "mnist,fashionmnist,femnist"
DEFAULT_SEEDS = "20260507,20260508,20260509,20260510,20260511,20260512,20260513,20260514,20260515,20260516"
METHOD_ORDER = ("krum", "fltrust", "zeno", "zenopp", "cornerdrive")
METRIC_KEYS = (
    "main_accuracy_avg",
    "corner_accuracy_avg",
    "fraud_survival_rate_avg",
    "rarity_retention_rate_avg",
    "effective_fraud_mass_survival_avg",
    "effective_rarity_mass_retention_avg",
    "selected_total_avg",
    "l1_review_rate_avg",
    "l1_fraud_recall_avg",
    "l2_fraud_reject_rate_given_routed_avg",
    "fraud_survival_unrouted_avg",
    "fraud_survival_l2_accepted_avg",
    "l2_fraud_as_rarity_accept_rate_avg",
    "l2_conflict_update_reject_rate_avg",
    "l2_accepted_positive_main_drift_rate_avg",
    "fraud_quarantine_rate_avg",
)


def parse_csv_values(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_seed_values(raw: str) -> list[int]:
    return [int(item) for item in parse_csv_values(raw)]


def source_slug(source: str) -> str:
    return source.lower().replace("/", "_").replace("leaf_", "")


def profile_l1_defaults(policy_profile: str) -> dict[str, float | str]:
    if policy_profile in {"real_data_adaptive", "real_data_adaptive_v4", "real_data_adaptive_v41"}:
        mode = (
            "v4_m4_dual_proxy_budgeted"
            if policy_profile in {"real_data_adaptive_v4", "real_data_adaptive_v41"}
            else "v3_m3_budgeted"
        )
        return {
            "cornerdrive_l1_mode": mode,
            "cornerdrive_l1_cos_weight": 0.35,
            "cornerdrive_l1_norm_weight": 0.20,
            "cornerdrive_l1_sign_weight": 0.15,
            "cornerdrive_l1_norm_mad_threshold": 1.5,
            "cornerdrive_l1_sign_threshold": 0.40,
            "cornerdrive_l1_sign_topk_ratio": 0.10,
            "cornerdrive_l1_queue_budget_ratio": 0.80,
            "cornerdrive_l1_random_recheck_ratio": 0.05,
        }
    return {
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


def build_policy(args: argparse.Namespace):
    if args.policy_profile == "real_data_adaptive_v41":
        policy = make_real_data_adaptive_v41_policy()
    elif args.policy_profile != "default":
        policy = make_real_data_adaptive_policy()
    else:
        policy = DEFAULT_POLICY
    updates = {
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
    return policy.model_copy(update=updates) if updates else policy


def build_config(args: argparse.Namespace, source: str, seed: int) -> RealGradientBenchmarkConfig:
    l1_defaults = profile_l1_defaults(args.policy_profile)
    return RealGradientBenchmarkConfig(
        source=source,
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
        seed=seed,
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
        cornerdrive_l1_mode=str(args.cornerdrive_l1_mode or l1_defaults["cornerdrive_l1_mode"]),
        cornerdrive_l1_cos_weight=float(
            args.cornerdrive_l1_cos_weight
            if args.cornerdrive_l1_cos_weight is not None
            else l1_defaults["cornerdrive_l1_cos_weight"]
        ),
        cornerdrive_l1_norm_weight=float(
            args.cornerdrive_l1_norm_weight
            if args.cornerdrive_l1_norm_weight is not None
            else l1_defaults["cornerdrive_l1_norm_weight"]
        ),
        cornerdrive_l1_sign_weight=float(
            args.cornerdrive_l1_sign_weight
            if args.cornerdrive_l1_sign_weight is not None
            else l1_defaults["cornerdrive_l1_sign_weight"]
        ),
        cornerdrive_l1_norm_mad_threshold=float(
            args.cornerdrive_l1_norm_mad_threshold
            if args.cornerdrive_l1_norm_mad_threshold is not None
            else l1_defaults["cornerdrive_l1_norm_mad_threshold"]
        ),
        cornerdrive_l1_sign_threshold=float(
            args.cornerdrive_l1_sign_threshold
            if args.cornerdrive_l1_sign_threshold is not None
            else l1_defaults["cornerdrive_l1_sign_threshold"]
        ),
        cornerdrive_l1_sign_topk_ratio=float(
            args.cornerdrive_l1_sign_topk_ratio
            if args.cornerdrive_l1_sign_topk_ratio is not None
            else l1_defaults["cornerdrive_l1_sign_topk_ratio"]
        ),
        cornerdrive_l1_queue_budget_ratio=float(
            args.cornerdrive_l1_queue_budget_ratio
            if args.cornerdrive_l1_queue_budget_ratio is not None
            else l1_defaults["cornerdrive_l1_queue_budget_ratio"]
        ),
        cornerdrive_l1_random_recheck_ratio=float(
            args.cornerdrive_l1_random_recheck_ratio
            if args.cornerdrive_l1_random_recheck_ratio is not None
            else l1_defaults["cornerdrive_l1_random_recheck_ratio"]
        ),
    )


def truth_observation_counts(result: dict[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    cornerdrive = result["methods"]["cornerdrive"]
    for row in cornerdrive["round_records"]:
        counts.update(row.get("truth_counts", {}))
    return counts


def run_row(
    *,
    result: dict[str, Any],
    source: str,
    seed: int,
    method_id: str,
) -> dict[str, Any]:
    payload = result["methods"][method_id]
    summary = payload["summary"]
    dataset = result["dataset"]
    config = result["config"]
    truth_counts = truth_observation_counts(result)
    row = {
        "source": source,
        "seed": seed,
        "method": payload["label"],
        "method_id": method_id,
        "dataset_source": dataset["source"],
        "client_count": dataset["client_count"],
        "client_sample_count": dataset["client_sample_count"],
        "client_sample_min": dataset["client_sample_min"],
        "client_sample_max": dataset["client_sample_max"],
        "client_sample_avg": dataset["client_sample_avg"],
        "rounds": config["rounds"],
        "clients_per_round": config["clients_per_round"],
        "client_observations": config["rounds"] * config["clients_per_round"],
        "fraud_observations": truth_counts.get("FRAUD", 0),
        "rarity_observations": truth_counts.get("RARITY", 0),
        "noise_observations": truth_counts.get("NOISE", 0),
        "honest_observations": truth_counts.get("HONEST", 0),
        "policy_theta_tol": result["policy"]["theta_tol"],
        "policy_theta_rare": result["policy"]["theta_rare"],
        "policy_theta_rarity_main_tol": result["policy"].get("theta_rarity_main_tol", ""),
        "policy_cosine_filter_threshold": result["policy"]["cosine_filter_threshold"],
        "policy_recheck_probability": result["policy"]["recheck_probability"],
        "cornerdrive_l1_mode": config["cornerdrive_l1_mode"],
        "cornerdrive_l1_cos_weight": config["cornerdrive_l1_cos_weight"],
        "cornerdrive_l1_norm_weight": config["cornerdrive_l1_norm_weight"],
        "cornerdrive_l1_sign_weight": config["cornerdrive_l1_sign_weight"],
        "cornerdrive_l1_norm_mad_threshold": config["cornerdrive_l1_norm_mad_threshold"],
        "cornerdrive_l1_sign_threshold": config["cornerdrive_l1_sign_threshold"],
        "cornerdrive_l1_sign_topk_ratio": config["cornerdrive_l1_sign_topk_ratio"],
        "cornerdrive_l1_queue_budget_ratio": config["cornerdrive_l1_queue_budget_ratio"],
        "cornerdrive_l1_random_recheck_ratio": config["cornerdrive_l1_random_recheck_ratio"],
        "cornerdrive_l1_theta_corner_harm_proxy": config.get("cornerdrive_l1_theta_corner_harm_proxy", ""),
        "cornerdrive_l1_quarantine_risk_threshold": config.get("cornerdrive_l1_quarantine_risk_threshold", ""),
        "cornerdrive_l1_low_weight_risk_threshold": config.get("cornerdrive_l1_low_weight_risk_threshold", ""),
        "cornerdrive_l1_safe_weight": config.get("cornerdrive_l1_safe_weight", ""),
        "cornerdrive_l1_low_weight": config.get("cornerdrive_l1_low_weight", ""),
    }
    for metric in METRIC_KEYS:
        row[metric] = summary.get(metric, "")
    if method_id == "cornerdrive":
        row["fraud_survival_by_attack_family"] = json.dumps(
            summary.get("fraud_survival_by_attack_family", {}),
            sort_keys=True,
        )
    else:
        row["fraud_survival_by_attack_family"] = "{}"
    return row


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["source"]), str(row["method"]))].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (source, method), group in sorted(grouped.items()):
        first = group[0]
        out: dict[str, Any] = {
            "source": source,
            "method": method,
            "runs": len(group),
            "seeds": ",".join(str(row["seed"]) for row in group),
            "client_count_mean": mean(float(row["client_count"]) for row in group),
            "client_sample_count_mean": mean(float(row["client_sample_count"]) for row in group),
            "client_observations_total": sum(int(row["client_observations"]) for row in group),
            "fraud_observations_total": sum(int(row["fraud_observations"]) for row in group),
            "rarity_observations_total": sum(int(row["rarity_observations"]) for row in group),
            "policy_theta_tol": first.get("policy_theta_tol", ""),
            "policy_theta_rare": first.get("policy_theta_rare", ""),
            "policy_theta_rarity_main_tol": first.get("policy_theta_rarity_main_tol", ""),
            "cornerdrive_l1_mode": first["cornerdrive_l1_mode"],
            "cornerdrive_l1_queue_budget_ratio": first["cornerdrive_l1_queue_budget_ratio"],
            "cornerdrive_l1_random_recheck_ratio": first["cornerdrive_l1_random_recheck_ratio"],
            "cornerdrive_l1_cos_weight": first["cornerdrive_l1_cos_weight"],
            "cornerdrive_l1_norm_weight": first["cornerdrive_l1_norm_weight"],
            "cornerdrive_l1_sign_weight": first["cornerdrive_l1_sign_weight"],
            "cornerdrive_l1_theta_corner_harm_proxy": first.get("cornerdrive_l1_theta_corner_harm_proxy", ""),
            "cornerdrive_l1_quarantine_risk_threshold": first.get("cornerdrive_l1_quarantine_risk_threshold", ""),
            "cornerdrive_l1_low_weight_risk_threshold": first.get("cornerdrive_l1_low_weight_risk_threshold", ""),
            "cornerdrive_l1_safe_weight": first.get("cornerdrive_l1_safe_weight", ""),
            "cornerdrive_l1_low_weight": first.get("cornerdrive_l1_low_weight", ""),
        }
        for metric in METRIC_KEYS:
            values = [
                float(row[metric])
                for row in group
                if row.get(metric) not in {"", None}
            ]
            if not values:
                continue
            metric_mean = mean(values)
            metric_stdev = stdev(values) if len(values) > 1 else 0.0
            metric_ci95 = 1.96 * metric_stdev / math.sqrt(len(values)) if len(values) > 1 else 0.0
            out[f"{metric}_mean"] = metric_mean
            out[f"{metric}_stdev"] = metric_stdev
            out[f"{metric}_ci95"] = metric_ci95
        summary_rows.append(out)
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run larger real-gradient benchmarks across datasets and seeds, "
            "then export mean/stdev/95% CI evidence tables."
        )
    )
    parser.add_argument("--sources", default=DEFAULT_SOURCES)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--leaf-data-dir", default="data/real/femnist")
    parser.add_argument("--bdd-data-dir", default="data/real/bdd100k")
    parser.add_argument("--bdd-label-file", default="")
    parser.add_argument("--bdd-image-dir", default="")
    parser.add_argument("--bdd-image-size", type=int, default=32)
    parser.add_argument("--bdd-target-attribute", choices=["weather", "timeofday", "scene"], default="weather")
    parser.add_argument("--bdd-client-group", default="weather_timeofday")
    parser.add_argument("--bdd-corner-values", default="rainy,snowy,foggy")
    parser.add_argument("--data-dir", default="data/real")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-clients", type=int, default=120)
    parser.add_argument("--min-samples-per-client", type=int, default=8)
    parser.add_argument("--max-samples-per-client", type=int, default=48)
    parser.add_argument("--clients-per-round", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--pretrain-steps", type=int, default=50)
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
            choices=["default", "real_data_adaptive", "real_data_adaptive_v4", "real_data_adaptive_v41"],
            default="real_data_adaptive",
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
        "--methods",
        default=",".join(METHOD_ORDER),
        help="Comma-separated method ids to include in exported tables.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "real_gradient_reliability_medium",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.verbose:
        logging.disable(logging.CRITICAL)

    output_dir = args.output_dir
    run_root = output_dir / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = parse_csv_values(args.sources)
    seeds = parse_seed_values(args.seeds)
    method_ids = parse_csv_values(args.methods)
    policy = build_policy(args)

    run_rows: list[dict[str, Any]] = []
    run_manifest: list[dict[str, Any]] = []
    for source in sources:
        for seed in seeds:
            config = build_config(args, source, seed)
            result = run_real_gradient_benchmark(config, policy=policy)
            run_dir = run_root / source_slug(source) / f"seed_{seed}"
            write_real_gradient_outputs(result, run_dir)
            run_manifest.append({
                "source": source,
                "seed": seed,
                "run_dir": str(run_dir.relative_to(output_dir)),
                "dataset": result["dataset"],
                "config": result["config"],
            })
            for method_id in method_ids:
                run_rows.append(
                    run_row(
                        result=result,
                        source=source,
                        seed=seed,
                        method_id=method_id,
                    )
                )
            cornerdrive = result["methods"]["cornerdrive"]["summary"]
            print(
                f"{source} seed={seed}: "
                f"CornerDrive fraud={cornerdrive['fraud_survival_rate_avg']:.4f}, "
                f"rarity={cornerdrive['rarity_retention_rate_avg']:.4f}, "
                f"l1_review={cornerdrive.get('l1_review_rate_avg', 0.0):.4f}",
                flush=True,
            )

    summary_rows = summarize(run_rows)
    write_csv(output_dir / "real_gradient_reliability_runs.csv", run_rows)
    write_csv(output_dir / "real_gradient_reliability_summary.csv", summary_rows)
    (output_dir / "real_gradient_reliability_summary.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "sources": sources,
                "seeds": seeds,
                "methods": method_ids,
                "runs": run_manifest,
                "summary": summary_rows,
            },
            indent=2,
        )
    )
    print(f"Wrote reliability benchmark artifacts to {output_dir}", flush=True)


if __name__ == "__main__":
    main()

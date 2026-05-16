#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

for candidate in (PROJECT_ROOT, BACKEND_DIR, SCRIPTS_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from common.schemas import DEFAULT_POLICY  # noqa: E402
from export_thesis_artifacts import (  # noqa: E402
    build_eval_bundle,
    build_round_bundles,
    clone_policy,
    policy_with_recheck_probability,
    pretrain_initial_checkpoint,
    run_flpg_with_artifacts,
    write_csv,
    write_json,
)
from export_synthetic_alg_benchmark import parse_seed_values  # noqa: E402
from export_synthetic_stress_tests import run_level_metrics  # noqa: E402
from generate_demo_data import cosine_similarity, normalize  # noqa: E402
from l1_linear_defense.config import make_l1_router_config  # noqa: E402
from policy_agent.analysis.unified_benchmark import _make_generator  # noqa: E402


DEFAULT_RHO_VALUES = "0.0,0.3,0.5,0.7,1.0"
DEFAULT_SEEDS = "20260318,20260319,20260320"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export corner-family divergence rho-sweep for synthetic ALG anti-circularity checks."
    )
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--cycle-rounds", type=int, default=12)
    parser.add_argument("--pretrain-epochs", type=int, default=5)
    parser.add_argument("--pretrain-batch-size", type=int, default=64)
    parser.add_argument("--pretrain-learning-rate", type=float, default=1e-3)
    parser.add_argument("--p-recheck", type=float, default=0.10)
    parser.add_argument("--rho-values", type=str, default=DEFAULT_RHO_VALUES)
    parser.add_argument("--seeds", type=str, default=DEFAULT_SEEDS)
    parser.add_argument("--corner-family-split-size", type=int, default=50)
    parser.add_argument(
        "--l1-router-mode",
        type=str,
        default="cosine_recheck",
        help="L1 router mode to use during the sweep.",
    )
    parser.add_argument(
        "--l1-queue-budget-ratio",
        "--queue-budget-ratio",
        dest="l1_queue_budget_ratio",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--l1-random-recheck-ratio",
        "--random-recheck-ratio",
        dest="l1_random_recheck_ratio",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--skip-oracle-drift",
        action="store_true",
        help="Skip oracle drift columns for faster smoke runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "corner_family_divergence",
    )
    return parser.parse_args()


def parse_float_values(raw: str) -> list[float]:
    values: list[float] = []
    for part in raw.split(","):
        stripped = part.strip()
        if stripped:
            values.append(float(stripped))
    return values or [1.0]


def format_mean_std(values: list[float]) -> str:
    if not values:
        return ""
    sigma = stdev(values) if len(values) > 1 else 0.0
    return f"{mean(values):.6f} +/- {sigma:.6f}"


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    group_keys: list[str],
    metric_fields: list[str],
    count_fields: list[str] | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in group_keys)].append(row)

    summaries: list[dict[str, Any]] = []
    for key_values, items in sorted(grouped.items()):
        out = {key: value for key, value in zip(group_keys, key_values)}
        out["runs"] = len(items)
        for field in metric_fields:
            values = [
                float(item[field])
                for item in items
                if item.get(field) not in {"", None}
            ]
            out[field] = format_mean_std(values)
            out[f"{field}_mean"] = mean(values) if values else ""
            out[f"{field}_std"] = stdev(values) if len(values) > 1 else 0.0 if values else ""
        for field in count_fields or []:
            out[field] = sum(int(float(item.get(field, 0))) for item in items)
        summaries.append(out)
    return summaries


def add_context(rows: list[dict[str, Any]], **context: Any) -> list[dict[str, Any]]:
    return [{**context, **row} for row in rows]


def trace_summary(trace_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not trace_rows:
        return {
            "rarity_trace_candidates": 0,
            "rarity_trace_accept_rate": 0.0,
            "trace_family_a_score_mean": "",
            "trace_family_b_score_mean": "",
        }
    accepted = [row for row in trace_rows if bool(row.get("accepted_as_rarity"))]
    a_scores = [float(row["corner_family_a_score"]) for row in accepted if row.get("corner_family_a_score") not in {"", None}]
    b_scores = [float(row["corner_family_b_score"]) for row in accepted if row.get("corner_family_b_score") not in {"", None}]
    return {
        "rarity_trace_candidates": len(trace_rows),
        "rarity_trace_accept_rate": len(accepted) / len(trace_rows),
        "trace_family_a_score_mean": mean(a_scores) if a_scores else "",
        "trace_family_b_score_mean": mean(b_scores) if b_scores else "",
    }


def main() -> None:
    args = parse_args()
    rho_values = parse_float_values(args.rho_values)
    seeds = parse_seed_values(args.seeds)

    started_at = datetime.now(timezone.utc).isoformat()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    policy = policy_with_recheck_probability(clone_policy(DEFAULT_POLICY), args.p_recheck)
    initial_model, pretrain_info = pretrain_initial_checkpoint(
        epochs=args.pretrain_epochs,
        batch_size=args.pretrain_batch_size,
        learning_rate=args.pretrain_learning_rate,
    )
    eval_bundle = build_eval_bundle()
    l1_router_config = make_l1_router_config(
        args.l1_router_mode,
        cos_deviation_threshold=policy.cosine_filter_threshold,
        queue_budget_ratio=args.l1_queue_budget_ratio,
        random_recheck_ratio=args.l1_random_recheck_ratio,
    )

    run_rows: list[dict[str, Any]] = []
    raw_l1_rows: list[dict[str, Any]] = []
    raw_l2_rows: list[dict[str, Any]] = []
    raw_round_rows: list[dict[str, Any]] = []
    raw_trace_rows: list[dict[str, Any]] = []

    for rho in rho_values:
        for seed in seeds:
            generator = _make_generator(
                initial_model=initial_model,
                eval_bundle=eval_bundle,
                generator_seed=seed,
                corner_family_divergence=rho,
                corner_family_split_size=args.corner_family_split_size,
            )
            generator.ground_truth_mode = "archetype"
            rounds, _overall_counts = build_round_bundles(
                policy=policy,
                total_rounds=args.rounds,
                cycle_rounds=args.cycle_rounds,
                generator=generator,
            )
            (
                round_summary_rows,
                l1_rows,
                l2_rows,
                _policy_rows,
                _flpg_baseline_rows,
                _summary,
            ) = run_flpg_with_artifacts(
                rounds=rounds,
                initial_model=initial_model,
                eval_bundle=eval_bundle,
                reference_policy=policy,
                fixed_recheck_probability=args.p_recheck,
                adapt_policy=False,
                audit_mode="dual",
                recheck_seed=20260428 + seed,
                compute_oracle_drift=not args.skip_oracle_drift,
                l1_router_config=l1_router_config,
            )
            run = {
                "round_summary_rows": round_summary_rows,
                "l1_rows": l1_rows,
                "l2_rows": l2_rows,
            }
            trace_rows = [
                {**trace, "trace_round_id": round_bundle.round_id}
                for round_bundle in rounds
                for trace in round_bundle.rarity_generation_trace
            ]
            context = {
                "rho": rho,
                "seed": seed,
                "l1_router_mode": args.l1_router_mode,
                "p_recheck": args.p_recheck,
            }
            metrics = run_level_metrics(run)
            audit_corner_gradient = normalize(
                generator._compute_dataset_gradient(eval_bundle.audit_corner)
            )
            run_rows.append({
                **context,
                "corner_family_actual_cosine": generator.corner_family_actual_cosine,
                "corner_family_seed_cosine": generator.corner_family_seed_cosine,
                "corner_family_a_audit_cosine": cosine_similarity(
                    generator.corner_family_a_gradient,
                    audit_corner_gradient,
                ),
                "corner_family_b_audit_cosine": cosine_similarity(
                    generator.corner_family_b_gradient,
                    audit_corner_gradient,
                ),
                **metrics,
                **trace_summary(trace_rows),
            })
            raw_l1_rows.extend(add_context(l1_rows, **context))
            raw_l2_rows.extend(add_context(l2_rows, **context))
            raw_round_rows.extend(add_context(round_summary_rows, **context))
            raw_trace_rows.extend(add_context(trace_rows, **context))

    metric_fields = [
        "corner_family_actual_cosine",
        "corner_family_seed_cosine",
        "corner_family_a_audit_cosine",
        "corner_family_b_audit_cosine",
        "main_acc",
        "corner_acc",
        "audit_ratio",
        "rarity_recog",
        "rarity_l2_recog",
        "rarity_retention",
        "false_rarity",
        "corner_harm_survival",
        "sign_flip_survival",
        "audit_oracle_corner_sign_agree",
        "audit_oracle_corner_spearman",
        "audit_oracle_main_sign_agree",
        "audit_oracle_main_spearman",
        "rarity_trace_accept_rate",
        "trace_family_a_score_mean",
        "trace_family_b_score_mean",
    ]
    summary_rows = summarize_rows(
        run_rows,
        group_keys=["rho"],
        metric_fields=metric_fields,
        count_fields=["false_rarity_count", "audited_with_oracle", "rarity_trace_candidates"],
    )

    write_json(
        output_dir / "corner_family_divergence_config.json",
        {
            "started_at": started_at,
            "rho_values": rho_values,
            "seeds": seeds,
            "rounds": args.rounds,
            "cycle_rounds": args.cycle_rounds,
            "p_recheck": args.p_recheck,
            "l1_router_mode": args.l1_router_mode,
            "l1_queue_budget_ratio": args.l1_queue_budget_ratio,
            "l1_random_recheck_ratio": args.l1_random_recheck_ratio,
            "corner_family_split_size": args.corner_family_split_size,
            "pretrain": pretrain_info,
            "compute_oracle_drift": not args.skip_oracle_drift,
        },
    )
    write_csv(output_dir / "corner_family_divergence_raw.csv", [], run_rows)
    write_csv(output_dir / "corner_family_divergence_summary.csv", [], summary_rows)
    write_csv(output_dir / "corner_family_divergence_l1_raw.csv", [], raw_l1_rows)
    write_csv(output_dir / "corner_family_divergence_l2_raw.csv", [], raw_l2_rows)
    write_csv(output_dir / "corner_family_divergence_rounds_raw.csv", [], raw_round_rows)
    write_csv(output_dir / "corner_family_divergence_trace_raw.csv", [], raw_trace_rows)

    print(f"Wrote corner-family divergence artifacts to {output_dir}")


if __name__ == "__main__":
    main()

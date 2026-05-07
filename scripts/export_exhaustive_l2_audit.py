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
from export_v25_artifacts import (  # noqa: E402
    bool_value,
    family_stats_from_l2,
    parse_seed_values,
    safe_ratio,
)
from l1_linear_defense.config import make_l1_router_config  # noqa: E402
from policy_agent.analysis.unified_benchmark import _make_generator  # noqa: E402


DEFAULT_SEEDS = "20260318,20260319,20260320,20260321,20260322"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Exhaustive L2 Audit upper-bound visibility ablation."
    )
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--cycle-rounds", type=int, default=12)
    parser.add_argument("--pretrain-epochs", type=int, default=5)
    parser.add_argument("--pretrain-batch-size", type=int, default=64)
    parser.add_argument("--pretrain-learning-rate", type=float, default=1e-3)
    parser.add_argument("--seeds", type=str, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--l1-router-mode",
        type=str,
        default="v25_cosine_fixed",
        help="Router mode retained for telemetry; p_recheck=1.0 makes visibility exhaustive.",
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
        help="Skip oracle drift columns for faster upper-bound exports.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "exhaustive_l2_audit",
    )
    return parser.parse_args()


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


def run_metrics(run: dict[str, Any]) -> dict[str, Any]:
    round_rows = run["round_summary_rows"]
    l1_rows = run["l1_rows"]
    l2_rows = run["l2_rows"]
    family_stats = family_stats_from_l2(l2_rows)
    rarity_rows = [row for row in l2_rows if row.get("planned_role") == "RARITY"]
    rarity_recognized = [row for row in rarity_rows if row.get("verdict") == "RARITY"]
    rarity_retained = [
        row for row in rarity_rows
        if float(row.get("aggregation_weight", 0.0)) > 0.0
    ]
    non_rarity_rows = [row for row in l2_rows if row.get("planned_role") != "RARITY"]
    false_rarity = [row for row in non_rarity_rows if row.get("verdict") == "RARITY"]
    routed_rows = [row for row in l1_rows if bool_value(row.get("routed_to_l2"))]
    corner_harm_rows = [
        row for row in l2_rows
        if row.get("attack_family") == "corner_harm"
    ]
    corner_harm_caught = [
        row for row in corner_harm_rows
        if row.get("verdict") == "FRAUD"
    ]
    return {
        "main_acc": mean(float(row["main_task_accuracy"]) for row in round_rows),
        "corner_acc": mean(float(row["corner_case_accuracy"]) for row in round_rows),
        "audit_queue_ratio": mean(float(row["queue_ratio_qt"]) for row in round_rows),
        "rarity_recog": safe_ratio(len(rarity_recognized), len(rarity_rows)),
        "rarity_retention": safe_ratio(len(rarity_retained), len(rarity_rows)),
        "false_rarity": safe_ratio(len(false_rarity), len(non_rarity_rows)),
        "false_rarity_count": len(false_rarity),
        "sign_flip_survival": family_stats["survival_rate"]["sign_flip_proxy"],
        "corner_harm_survival": family_stats["survival_rate"]["corner_harm"],
        "l2_evals": len(routed_rows),
        "updates_total": len(l1_rows),
        "corner_harm_total": len(corner_harm_rows),
        "corner_harm_caught": len(corner_harm_caught),
        "corner_harm_l2_catch_rate": safe_ratio(len(corner_harm_caught), len(corner_harm_rows)),
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seed_values(args.seeds)

    reference_policy = policy_with_recheck_probability(clone_policy(DEFAULT_POLICY), 0.5)
    initial_model, pretrain_info = pretrain_initial_checkpoint(
        epochs=args.pretrain_epochs,
        batch_size=args.pretrain_batch_size,
        learning_rate=args.pretrain_learning_rate,
    )
    eval_bundle = build_eval_bundle()
    l1_router_config = make_l1_router_config(
        args.l1_router_mode,
        cos_deviation_threshold=reference_policy.cosine_filter_threshold,
        queue_budget_ratio=args.l1_queue_budget_ratio,
        random_recheck_ratio=args.l1_random_recheck_ratio,
    )

    raw_rows: list[dict[str, Any]] = []
    raw_l1_rows: list[dict[str, Any]] = []
    raw_l2_rows: list[dict[str, Any]] = []
    raw_round_rows: list[dict[str, Any]] = []

    for seed in seeds:
        generator = _make_generator(
            initial_model=initial_model,
            eval_bundle=eval_bundle,
            generator_seed=seed,
        )
        generator.ground_truth_mode = "archetype"
        rounds, _overall_counts = build_round_bundles(
            policy=reference_policy,
            total_rounds=args.rounds,
            cycle_rounds=args.cycle_rounds,
            generator=generator,
        )
        (
            round_summary_rows,
            l1_rows,
            l2_rows,
            policy_rows,
            flpg_baseline_rows,
            summary,
        ) = run_flpg_with_artifacts(
            rounds=rounds,
            initial_model=initial_model,
            eval_bundle=eval_bundle,
            reference_policy=reference_policy,
            fixed_recheck_probability=0.5,
            adapt_policy=False,
            audit_mode="dual",
            recheck_seed=20260428 + seed,
            compute_oracle_drift=not args.skip_oracle_drift,
            l1_router_config=l1_router_config,
            force_exhaustive_l2=True,
        )
        run = {
            "round_summary_rows": round_summary_rows,
            "l1_rows": l1_rows,
            "l2_rows": l2_rows,
            "policy_rows": policy_rows,
            "flpg_baseline_rows": flpg_baseline_rows,
            "summary": summary,
        }
        context = {
            "method": "Exhaustive L2 Audit",
            "seed": seed,
            "p_recheck": 1.0,
            "l1_router_mode": args.l1_router_mode,
        }
        raw_rows.append({**context, **run_metrics(run)})
        raw_l1_rows.extend(add_context(l1_rows, **context))
        raw_l2_rows.extend(add_context(l2_rows, **context))
        raw_round_rows.extend(add_context(round_summary_rows, **context))

    metric_fields = [
        "main_acc",
        "corner_acc",
        "audit_queue_ratio",
        "rarity_recog",
        "rarity_retention",
        "false_rarity",
        "sign_flip_survival",
        "corner_harm_survival",
        "corner_harm_l2_catch_rate",
    ]
    summary_rows = summarize_rows(
        raw_rows,
        group_keys=["method", "p_recheck", "l1_router_mode"],
        metric_fields=metric_fields,
        count_fields=[
            "false_rarity_count",
            "l2_evals",
            "updates_total",
            "corner_harm_total",
            "corner_harm_caught",
        ],
    )

    write_json(
        output_dir / "exhaustive_l2_audit_config.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "rounds": args.rounds,
            "cycle_rounds": args.cycle_rounds,
            "pretrain_epochs": args.pretrain_epochs,
            "seeds": seeds,
            "p_recheck_reported": 1.0,
            "policy_recheck_probability": 0.5,
            "force_exhaustive_l2": True,
            "scope": "upper-bound visibility/cost ablation; all updates are routed to L2",
            "pretrain": pretrain_info,
        },
    )
    write_csv(output_dir / "exhaustive_l2_audit_raw.csv", [], raw_rows)
    write_csv(output_dir / "exhaustive_l2_audit_summary.csv", [], summary_rows)
    write_csv(output_dir / "exhaustive_l2_audit_l1_raw.csv", [], raw_l1_rows)
    write_csv(output_dir / "exhaustive_l2_audit_l2_raw.csv", [], raw_l2_rows)
    write_csv(output_dir / "exhaustive_l2_audit_rounds_raw.csv", [], raw_round_rows)
    print(f"Wrote Exhaustive L2 Audit artifacts to {output_dir}")


if __name__ == "__main__":
    main()

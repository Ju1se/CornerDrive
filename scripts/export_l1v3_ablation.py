#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
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
    safe_mean,
    write_csv,
    write_json,
)
from export_v25_artifacts import (  # noqa: E402
    bool_value,
    family_stats_from_l2,
    parse_seed_values,
    rarity_metrics_for_run,
    safe_ratio,
)
from l1_linear_defense.config import make_l1_router_config, normalize_l1_mode  # noqa: E402
from policy_agent.analysis.unified_benchmark import _make_generator  # noqa: E402


DEFAULT_MODES = "m0,m1,m2,m3,m4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export M0-M4 CornerDrive-L1V3 visibility-router ablation."
    )
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--cycle-rounds", type=int, default=12)
    parser.add_argument("--pretrain-epochs", type=int, default=5)
    parser.add_argument("--pretrain-batch-size", type=int, default=64)
    parser.add_argument("--pretrain-learning-rate", type=float, default=1e-3)
    parser.add_argument("--p-recheck", type=float, default=0.10)
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
        "--seeds",
        type=str,
        default="20260318,20260319,20260320,20260321,20260322",
    )
    parser.add_argument("--modes", type=str, default=DEFAULT_MODES)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "l1v3_ablation",
    )
    return parser.parse_args()


def parse_modes(raw: str) -> list[str]:
    modes: list[str] = []
    for part in raw.split(","):
        stripped = part.strip()
        if stripped:
            modes.append(normalize_l1_mode(stripped))
    return modes or [normalize_l1_mode(mode) for mode in DEFAULT_MODES.split(",")]


def run_metrics(run: dict[str, Any]) -> dict[str, Any]:
    round_rows = run["round_summary_rows"]
    l1_rows = run["l1_rows"]
    l2_rows = run["l2_rows"]
    family_stats = family_stats_from_l2(l2_rows)
    rarity = rarity_metrics_for_run(run)
    honest_rows = [row for row in l1_rows if row.get("planned_role") == "HONEST"]
    honest_routed = [
        row for row in honest_rows
        if bool_value(row.get("routed_to_l2"))
    ]
    routed_rows = [row for row in l1_rows if bool_value(row.get("routed_to_l2"))]
    reason_counts = Counter(str(row.get("routing_reason", "unknown")) for row in routed_rows)

    return {
        "main_accuracy": safe_mean([float(row["main_task_accuracy"]) for row in round_rows]),
        "corner_accuracy": safe_mean([float(row["corner_case_accuracy"]) for row in round_rows]),
        "rarity_recognition": rarity["l2_rarity_recall"],
        "rarity_retention": rarity["end_to_end_rarity_retention"],
        "false_rarity": rarity["false_rarity_preservation_rate"],
        "sign_flip_survival": family_stats["survival_rate"]["sign_flip_proxy"],
        "corner_harm_survival": family_stats["survival_rate"]["corner_harm"],
        "audit_queue_ratio": safe_mean([float(row["queue_ratio_qt"]) for row in round_rows]),
        "honest_routed_rate": safe_ratio(len(honest_routed), len(honest_rows)),
        "routed_total": len(routed_rows),
        "routing_reason_counts": dict(reason_counts),
    }


def summarize_by_mode(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["l1_mode"])].append(row)

    metric_keys = [
        "main_accuracy",
        "corner_accuracy",
        "rarity_recognition",
        "rarity_retention",
        "false_rarity",
        "sign_flip_survival",
        "corner_harm_survival",
        "audit_queue_ratio",
        "honest_routed_rate",
        "routed_total",
    ]
    summary: list[dict[str, Any]] = []
    for mode, items in sorted(grouped.items()):
        record: dict[str, Any] = {"l1_mode": mode, "seeds": len(items)}
        for key in metric_keys:
            values = [float(item[key]) for item in items]
            record[f"{key}_mean"] = mean(values) if values else 0.0
            record[f"{key}_std"] = stdev(values) if len(values) > 1 else 0.0
        summary.append(record)
    return summary


def routing_reason_rows(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in seed_rows:
        counts = row["routing_reason_counts"]
        total = sum(counts.values())
        for reason, count in sorted(counts.items()):
            rows.append({
                "seed": row["seed"],
                "l1_mode": row["l1_mode"],
                "routing_reason": reason,
                "count": count,
                "rate": safe_ratio(count, total),
            })
    return rows


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    seeds = parse_seed_values(args.seeds)
    modes = parse_modes(args.modes)
    policy = policy_with_recheck_probability(clone_policy(DEFAULT_POLICY), args.p_recheck)

    initial_model, checkpoint_info = pretrain_initial_checkpoint(
        epochs=args.pretrain_epochs,
        batch_size=args.pretrain_batch_size,
        learning_rate=args.pretrain_learning_rate,
    )
    eval_bundle = build_eval_bundle()

    seed_rows: list[dict[str, Any]] = []
    raw_l1_rows: list[dict[str, Any]] = []
    raw_l2_rows: list[dict[str, Any]] = []
    raw_round_rows: list[dict[str, Any]] = []

    for seed in seeds:
        generator = _make_generator(
            initial_model=initial_model,
            eval_bundle=eval_bundle,
            generator_seed=seed,
        )
        rounds, _overall_counts = build_round_bundles(
            policy=policy,
            total_rounds=args.rounds,
            cycle_rounds=args.cycle_rounds,
            generator=generator,
        )
        for mode in modes:
            l1_config = make_l1_router_config(
                mode,
                cos_deviation_threshold=policy.cosine_filter_threshold,
                queue_budget_ratio=args.l1_queue_budget_ratio,
                random_recheck_ratio=args.l1_random_recheck_ratio,
            )
            (
                round_summary_rows,
                l1_rows,
                l2_rows,
                _policy_rows,
                flpg_baseline_rows,
                summary,
            ) = run_flpg_with_artifacts(
                rounds=rounds,
                initial_model=initial_model,
                eval_bundle=eval_bundle,
                reference_policy=policy,
                fixed_recheck_probability=args.p_recheck,
                adapt_policy=False,
                audit_mode="dual",
                recheck_seed=20260428 + seed,
                l1_router_config=l1_config,
            )
            run = {
                "round_summary_rows": round_summary_rows,
                "l1_rows": l1_rows,
                "l2_rows": l2_rows,
                "flpg_baseline_rows": flpg_baseline_rows,
                "summary": summary,
            }
            metrics = run_metrics(run)
            seed_rows.append({"seed": seed, "l1_mode": mode, **metrics})
            raw_l1_rows.extend({**row, "seed": seed, "l1_mode": mode} for row in l1_rows)
            raw_l2_rows.extend({**row, "seed": seed, "l1_mode": mode} for row in l2_rows)
            raw_round_rows.extend(
                {**row, "seed": seed, "l1_mode": mode}
                for row in round_summary_rows
            )

    write_json(
        output_dir / "l1v3_ablation_run_config.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "rounds": args.rounds,
            "cycle_rounds": args.cycle_rounds,
            "p_recheck": args.p_recheck,
            "seeds": seeds,
            "modes": modes,
            "l1_queue_budget_ratio": args.l1_queue_budget_ratio,
            "l1_random_recheck_ratio": args.l1_random_recheck_ratio,
            "initial_checkpoint": checkpoint_info,
        },
    )
    write_csv(output_dir / "l1v3_ablation_by_seed.csv", [], seed_rows)
    write_csv(output_dir / "l1v3_ablation_summary.csv", [], summarize_by_mode(seed_rows))
    write_csv(output_dir / "l1v3_routing_by_reason.csv", [], routing_reason_rows(seed_rows))
    write_csv(output_dir / "l1v3_l1_routing_raw.csv", [], raw_l1_rows)
    write_csv(output_dir / "l1v3_l2_audit_raw.csv", [], raw_l2_rows)
    write_csv(output_dir / "l1v3_round_summary_raw.csv", [], raw_round_rows)
    print(f"Exported L1V3 ablation artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

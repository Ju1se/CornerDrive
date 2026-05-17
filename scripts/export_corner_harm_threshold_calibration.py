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
    validate_synthetic_router_mode,
    write_csv,
    write_json,
)
from export_synthetic_alg_benchmark import (  # noqa: E402
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
        description="Export theta_corner_harm calibration robustness check."
    )
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--cycle-rounds", type=int, default=12)
    parser.add_argument("--pretrain-epochs", type=int, default=5)
    parser.add_argument("--pretrain-batch-size", type=int, default=64)
    parser.add_argument("--pretrain-learning-rate", type=float, default=1e-3)
    parser.add_argument("--p-recheck", type=float, default=0.10)
    parser.add_argument("--seeds", type=str, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--fixed-thresholds",
        type=str,
        default="-0.005,0.0,0.005",
        help="Comma-separated fixed theta_corner_harm values.",
    )
    parser.add_argument(
        "--l1-router-mode",
        type=str,
        default="cosine_recheck",
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
        help="Skip oracle drift columns for faster calibration exports.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "corner_harm_threshold_calibration",
    )
    return parser.parse_args()


def parse_float_values(raw: str) -> list[float]:
    values: list[float] = []
    for part in raw.split(","):
        stripped = part.strip()
        if stripped:
            values.append(float(stripped))
    return values or [0.0]


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


def run_cornerdrive(
    *,
    seed: int,
    rounds: list[Any],
    initial_model: Any,
    eval_bundle: Any,
    policy: Any,
    p_recheck: float,
    theta_corner_harm: float,
    l1_router_mode: str,
    l1_queue_budget_ratio: float,
    l1_random_recheck_ratio: float,
    compute_oracle_drift: bool,
) -> dict[str, Any]:
    policy = policy_with_recheck_probability(policy, p_recheck)
    l1_router_config = make_l1_router_config(
        l1_router_mode,
        cos_deviation_threshold=policy.cosine_filter_threshold,
        queue_budget_ratio=l1_queue_budget_ratio,
        random_recheck_ratio=l1_random_recheck_ratio,
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
        reference_policy=policy,
        fixed_recheck_probability=p_recheck,
        adapt_policy=False,
        audit_mode="dual",
        recheck_seed=20260428 + seed,
        compute_oracle_drift=compute_oracle_drift,
        l1_router_config=l1_router_config,
        theta_corner_harm_proxy=theta_corner_harm,
    )
    return {
        "round_summary_rows": round_summary_rows,
        "l1_rows": l1_rows,
        "l2_rows": l2_rows,
        "policy_rows": policy_rows,
        "flpg_baseline_rows": flpg_baseline_rows,
        "summary": summary,
    }


def calibration_from_benign(l2_rows: list[dict[str, Any]]) -> dict[str, float]:
    benign = [
        float(row["delta_l_corner"])
        for row in l2_rows
        if row.get("planned_role") == "HONEST"
        and float(row.get("delta_l_main", 0.0)) <= 0.0
    ]
    if not benign:
        return {
            "benign_corner_delta_mean": 0.0,
            "benign_corner_delta_std": 0.0,
            "calibrated_theta_corner_harm": 0.0,
            "benign_calibration_count": 0,
        }
    mu = mean(benign)
    sigma = stdev(benign) if len(benign) > 1 else 0.0
    return {
        "benign_corner_delta_mean": mu,
        "benign_corner_delta_std": sigma,
        "calibrated_theta_corner_harm": mu + sigma,
        "benign_calibration_count": len(benign),
    }


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
    routed_honest = [
        row for row in l2_rows
        if row.get("planned_role") == "HONEST" and bool_value(row.get("audited_in_l2"))
    ]
    honest_reject_corner_harm = [
        row for row in routed_honest if row.get("action") == "reject_corner_harm"
    ]
    routed_corner_harm = [
        row for row in l2_rows
        if row.get("attack_family") == "corner_harm" and bool_value(row.get("audited_in_l2"))
    ]
    caught_corner_harm = [
        row for row in routed_corner_harm if row.get("verdict") == "FRAUD"
    ]
    action_counts = Counter(str(row.get("action", "")) for row in l2_rows)
    return {
        "main_acc": mean(float(row["main_task_accuracy"]) for row in round_rows),
        "corner_acc": mean(float(row["corner_case_accuracy"]) for row in round_rows),
        "audit_ratio": mean(float(row["queue_ratio_qt"]) for row in round_rows),
        "rarity_recog": safe_ratio(len(rarity_recognized), len(rarity_rows)),
        "rarity_retention": safe_ratio(len(rarity_retained), len(rarity_rows)),
        "false_rarity": safe_ratio(len(false_rarity), len(non_rarity_rows)),
        "false_rarity_count": len(false_rarity),
        "corner_harm_survival": family_stats["survival_rate"]["corner_harm"],
        "sign_flip_survival": family_stats["survival_rate"]["sign_flip_proxy"],
        "corner_harm_l2_catch_rate": safe_ratio(len(caught_corner_harm), len(routed_corner_harm)),
        "corner_harm_audited": len(routed_corner_harm),
        "corner_harm_caught": len(caught_corner_harm),
        "honest_corner_harm_false_reject_rate": safe_ratio(
            len(honest_reject_corner_harm),
            len(routed_honest),
        ),
        "honest_routed": len(routed_honest),
        "honest_corner_harm_false_reject": len(honest_reject_corner_harm),
        "reject_corner_harm_total": action_counts["reject_corner_harm"],
    }


def threshold_label(value: float) -> str:
    if value < 0.0:
        return f"strict_{value:.3f}"
    if value == 0.0:
        return "default_0.000"
    return f"relaxed_{value:.3f}"


def main() -> None:
    args = parse_args()
    validate_synthetic_router_mode(args.l1_router_mode)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seed_values(args.seeds)
    fixed_thresholds = parse_float_values(args.fixed_thresholds)

    reference_policy = clone_policy(DEFAULT_POLICY)
    initial_model, pretrain_info = pretrain_initial_checkpoint(
        epochs=args.pretrain_epochs,
        batch_size=args.pretrain_batch_size,
        learning_rate=args.pretrain_learning_rate,
    )
    eval_bundle = build_eval_bundle()

    raw_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
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
        calibration_run = run_cornerdrive(
            seed=seed,
            rounds=rounds,
            initial_model=initial_model,
            eval_bundle=eval_bundle,
            policy=reference_policy,
            p_recheck=args.p_recheck,
            theta_corner_harm=0.0,
            l1_router_mode=args.l1_router_mode,
            l1_queue_budget_ratio=args.l1_queue_budget_ratio,
            l1_random_recheck_ratio=args.l1_random_recheck_ratio,
            compute_oracle_drift=False,
        )
        calibration = calibration_from_benign(calibration_run["l2_rows"])
        calibration_rows.append({"seed": seed, **calibration})

        threshold_settings = [
            (threshold_label(value), value, "fixed")
            for value in fixed_thresholds
        ]
        threshold_settings.append((
            "calibrated_mu_plus_1sigma",
            calibration["calibrated_theta_corner_harm"],
            "benign_mu_plus_1sigma",
        ))

        for label, theta_corner_harm, source in threshold_settings:
            run = run_cornerdrive(
                seed=seed,
                rounds=rounds,
                initial_model=initial_model,
                eval_bundle=eval_bundle,
                policy=reference_policy,
                p_recheck=args.p_recheck,
                theta_corner_harm=theta_corner_harm,
                l1_router_mode=args.l1_router_mode,
                l1_queue_budget_ratio=args.l1_queue_budget_ratio,
                l1_random_recheck_ratio=args.l1_random_recheck_ratio,
                compute_oracle_drift=not args.skip_oracle_drift,
            )
            context = {
                "seed": seed,
                "setting": label,
                "theta_corner_harm": theta_corner_harm,
                "theta_source": source,
                "p_recheck": args.p_recheck,
                "l1_router_mode": args.l1_router_mode,
                **calibration,
            }
            raw_rows.append({**context, **run_metrics(run)})
            raw_l2_rows.extend(add_context(run["l2_rows"], **context))
            raw_round_rows.extend(add_context(run["round_summary_rows"], **context))

    metric_fields = [
        "theta_corner_harm",
        "benign_corner_delta_mean",
        "benign_corner_delta_std",
        "calibrated_theta_corner_harm",
        "main_acc",
        "corner_acc",
        "audit_ratio",
        "rarity_recog",
        "rarity_retention",
        "false_rarity",
        "corner_harm_survival",
        "sign_flip_survival",
        "corner_harm_l2_catch_rate",
        "honest_corner_harm_false_reject_rate",
    ]
    summary_rows = summarize_rows(
        raw_rows,
        group_keys=["setting", "theta_source", "p_recheck", "l1_router_mode"],
        metric_fields=metric_fields,
        count_fields=[
            "false_rarity_count",
            "corner_harm_audited",
            "corner_harm_caught",
            "honest_routed",
            "honest_corner_harm_false_reject",
            "reject_corner_harm_total",
            "benign_calibration_count",
        ],
    )

    write_json(
        output_dir / "corner_harm_threshold_calibration_config.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "rounds": args.rounds,
            "cycle_rounds": args.cycle_rounds,
            "pretrain_epochs": args.pretrain_epochs,
            "p_recheck": args.p_recheck,
            "seeds": seeds,
            "fixed_thresholds": fixed_thresholds,
            "calibrated_threshold": "mean(delta_l_corner for HONEST with delta_l_main <= 0) + std",
            "scope": "theta_corner_harm robustness/calibration check, not full L4 evaluation",
            "pretrain": pretrain_info,
        },
    )
    write_csv(output_dir / "corner_harm_threshold_calibration_by_seed.csv", [], calibration_rows)
    write_csv(output_dir / "corner_harm_threshold_calibration_raw.csv", [], raw_rows)
    write_csv(output_dir / "corner_harm_threshold_calibration_summary.csv", [], summary_rows)
    write_csv(output_dir / "corner_harm_threshold_calibration_l2_raw.csv", [], raw_l2_rows)
    write_csv(output_dir / "corner_harm_threshold_calibration_rounds_raw.csv", [], raw_round_rows)
    print(f"Wrote corner-harm threshold calibration artifacts to {output_dir}")


if __name__ == "__main__":
    main()

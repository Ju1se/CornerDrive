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

from common.schemas import DEFAULT_POLICY, Policy  # noqa: E402
from export_thesis_artifacts import (  # noqa: E402
    build_eval_bundle,
    build_round_bundles,
    clone_policy,
    pretrain_initial_checkpoint,
    run_flpg_with_artifacts,
    safe_mean,
    write_csv,
    write_json,
)
from export_synthetic_alg_benchmark import (  # noqa: E402
    bool_value,
    family_stats_from_l2,
    parse_seed_values,
    safe_ratio,
)
from l1_linear_defense.config import make_l1_router_config, normalize_l1_mode  # noqa: E402
from policy_agent.analysis.unified_benchmark import _make_generator  # noqa: E402


DEFAULT_SEEDS = "20260318,20260319,20260320"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export L1/L2 operating curves: L1 visibility budget frontier and "
            "L2 dual-channel threshold grid on fixed generated rounds."
        )
    )
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--cycle-rounds", type=int, default=12)
    parser.add_argument("--pretrain-epochs", type=int, default=5)
    parser.add_argument("--pretrain-batch-size", type=int, default=64)
    parser.add_argument("--pretrain-learning-rate", type=float, default=1e-3)
    parser.add_argument("--seeds", type=str, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--sweep",
        choices=("frontier", "threshold", "both"),
        default="both",
        help="frontier sweeps L1 budget/recheck; threshold sweeps L2 theta grid.",
    )
    parser.add_argument(
        "--l1-modes",
        type=str,
        default="cosine_recheck,dual_proxy_budgeted",
        help="Comma-separated L1 modes for the frontier sweep.",
    )
    parser.add_argument(
        "--p-recheck-values",
        type=str,
        default="0.0,0.05,0.10",
        help="Comma-separated L1 probabilistic recheck values for frontier sweep.",
    )
    parser.add_argument(
        "--budget-values",
        type=str,
        default="0.20,0.35,0.50",
        help="Comma-separated L1 queue budget ratios for budgeted router modes.",
    )
    parser.add_argument(
        "--random-recheck-ratio",
        type=float,
        default=0.05,
        help="Stratified random recheck ratio used by budgeted L1 modes.",
    )
    parser.add_argument(
        "--threshold-l1-mode",
        type=str,
        default="dual_proxy_budgeted",
        help="L1 mode held fixed for L2 threshold-grid sweep.",
    )
    parser.add_argument("--threshold-p-recheck", type=float, default=0.10)
    parser.add_argument("--threshold-budget", type=float, default=0.35)
    parser.add_argument(
        "--theta-tol-values",
        type=str,
        default="0.025,0.05,0.075",
        help="Comma-separated L2 main-task fraud thresholds for threshold sweep.",
    )
    parser.add_argument(
        "--theta-rare-values",
        type=str,
        default="-0.01,-0.03,-0.05",
        help="Comma-separated L2 corner-case rarity thresholds for threshold sweep.",
    )
    parser.add_argument(
        "--skip-oracle-drift",
        action="store_true",
        help="Skip oracle drift columns for faster exports.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "l1_l2_operating_curve",
    )
    return parser.parse_args()


def parse_float_values(raw: str) -> list[float]:
    values: list[float] = []
    for part in raw.split(","):
        stripped = part.strip()
        if stripped:
            values.append(float(stripped))
    return values


def parse_modes(raw: str) -> list[str]:
    modes: list[str] = []
    for part in raw.split(","):
        stripped = part.strip()
        if stripped:
            modes.append(normalize_l1_mode(stripped))
    return modes or [normalize_l1_mode("dual_proxy_budgeted")]


def policy_with_overrides(policy: Policy, **overrides: Any) -> Policy:
    payload = policy.model_dump()
    payload.update(overrides)
    return Policy.model_validate(payload)


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
    count_fields: list[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in group_keys)].append(row)

    summary: list[dict[str, Any]] = []
    for key_values, items in sorted(grouped.items()):
        record = {key: value for key, value in zip(group_keys, key_values)}
        record["runs"] = len(items)
        for field in metric_fields:
            values = [float(item[field]) for item in items if item.get(field) not in {"", None}]
            record[field] = format_mean_std(values)
            record[f"{field}_mean"] = mean(values) if values else ""
            record[f"{field}_std"] = stdev(values) if len(values) > 1 else 0.0 if values else ""
        for field in count_fields:
            record[field] = sum(int(float(item.get(field, 0))) for item in items)
        summary.append(record)
    return summary


def metric_rows_for_run(
    *,
    seed: int,
    sweep: str,
    l1_mode: str,
    p_recheck: float,
    budget_ratio: float,
    random_recheck_ratio: float,
    theta_tol: float,
    theta_rare: float,
    run: dict[str, Any],
) -> dict[str, Any]:
    round_rows = run["round_summary_rows"]
    l1_rows = run["l1_rows"]
    l2_rows = run["l2_rows"]
    family_stats = family_stats_from_l2(l2_rows)

    total_by_label = Counter(str(row.get("true_label", "")) for row in l1_rows)
    routed_by_label = Counter(
        str(row.get("true_label", ""))
        for row in l1_rows
        if bool_value(row.get("routed_to_l2"))
    )
    routed_rows = [row for row in l1_rows if bool_value(row.get("routed_to_l2"))]
    honest_rows = [row for row in l1_rows if row.get("true_label") == "HONEST"]
    high_cosine_dropped = [
        row for row in l1_rows
        if (
            float(row.get("deviation_score_geomed", 0.0))
            > float(row.get("tau_screen", theta_tol))
            and not bool_value(row.get("routed_to_l2"))
        )
    ]

    audited_rows = [row for row in l2_rows if bool_value(row.get("audited_in_l2"))]
    audited_fraud = [row for row in audited_rows if row.get("true_label") == "FRAUD"]
    audited_rarity = [row for row in audited_rows if row.get("true_label") == "RARITY"]
    predicted_fraud = [row for row in audited_rows if row.get("verdict") == "FRAUD"]
    predicted_rarity = [row for row in audited_rows if row.get("verdict") == "RARITY"]
    true_fraud = [row for row in l2_rows if row.get("true_label") == "FRAUD"]
    true_rarity = [row for row in l2_rows if row.get("true_label") == "RARITY"]
    true_non_rarity = [row for row in l2_rows if row.get("true_label") != "RARITY"]
    false_rarity = [row for row in true_non_rarity if row.get("verdict") == "RARITY"]
    retained_rarity = [
        row for row in true_rarity
        if float(row.get("aggregation_weight", 0.0)) > 0.0
    ]
    fraud_survived = [
        row for row in true_fraud
        if float(row.get("aggregation_weight", 0.0)) > 0.0
    ]

    corner_harm_rows = [
        row for row in l2_rows
        if row.get("true_label") == "FRAUD" and row.get("attack_family") == "corner_harm"
    ]
    sign_flip_rows = [
        row for row in l2_rows
        if row.get("true_label") == "FRAUD" and row.get("attack_family") == "sign_flip_proxy"
    ]
    corner_harm_caught = [row for row in corner_harm_rows if row.get("verdict") == "FRAUD"]
    sign_flip_caught = [row for row in sign_flip_rows if row.get("verdict") == "FRAUD"]

    return {
        "seed": seed,
        "sweep": sweep,
        "l1_mode": l1_mode,
        "p_recheck": p_recheck,
        "l1_queue_budget_ratio": budget_ratio,
        "l1_random_recheck_ratio": random_recheck_ratio,
        "theta_tol": theta_tol,
        "theta_rare": theta_rare,
        "updates_total": len(l1_rows),
        "l2_evals": len(audited_rows),
        "audit_queue_ratio": safe_mean([float(row["queue_ratio_qt"]) for row in round_rows]),
        "main_accuracy": safe_mean([float(row["main_task_accuracy"]) for row in round_rows]),
        "corner_accuracy": safe_mean([float(row["corner_case_accuracy"]) for row in round_rows]),
        "l1_precision_non_honest": safe_ratio(
            sum(1 for row in routed_rows if row.get("true_label") != "HONEST"),
            len(routed_rows),
        ),
        "l1_honest_routed_rate": safe_ratio(
            sum(1 for row in honest_rows if bool_value(row.get("routed_to_l2"))),
            len(honest_rows),
        ),
        "l1_fraud_recall": safe_ratio(routed_by_label["FRAUD"], total_by_label["FRAUD"]),
        "l1_rarity_recall": safe_ratio(routed_by_label["RARITY"], total_by_label["RARITY"]),
        "l1_noise_recall": safe_ratio(routed_by_label["NOISE"], total_by_label["NOISE"]),
        "l1_high_cosine_dropped": len(high_cosine_dropped),
        "l2_fraud_precision_cond": safe_ratio(
            sum(1 for row in predicted_fraud if row.get("true_label") == "FRAUD"),
            len(predicted_fraud),
        ),
        "l2_fraud_recall_cond": safe_ratio(
            sum(1 for row in audited_fraud if row.get("verdict") == "FRAUD"),
            len(audited_fraud),
        ),
        "e2e_fraud_catch_rate": safe_ratio(
            sum(1 for row in true_fraud if row.get("verdict") == "FRAUD"),
            len(true_fraud),
        ),
        "fraud_survival_rate": safe_ratio(len(fraud_survived), len(true_fraud)),
        "l2_rarity_precision_cond": safe_ratio(
            sum(1 for row in predicted_rarity if row.get("true_label") == "RARITY"),
            len(predicted_rarity),
        ),
        "l2_rarity_recall_cond": safe_ratio(
            sum(1 for row in audited_rarity if row.get("verdict") == "RARITY"),
            len(audited_rarity),
        ),
        "e2e_rarity_recognition": safe_ratio(
            sum(1 for row in true_rarity if row.get("verdict") == "RARITY"),
            len(true_rarity),
        ),
        "rarity_retention": safe_ratio(len(retained_rarity), len(true_rarity)),
        "false_rarity_rate": safe_ratio(len(false_rarity), len(true_non_rarity)),
        "sign_flip_catch_rate": safe_ratio(len(sign_flip_caught), len(sign_flip_rows)),
        "corner_harm_catch_rate": safe_ratio(len(corner_harm_caught), len(corner_harm_rows)),
        "sign_flip_survival": family_stats["survival_rate"]["sign_flip_proxy"],
        "corner_harm_survival": family_stats["survival_rate"]["corner_harm"],
        "false_rarity_count": len(false_rarity),
    }


def frontier_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    if args.sweep not in {"frontier", "both"}:
        return configs
    for mode in parse_modes(args.l1_modes):
        mode_uses_budget = make_l1_router_config(mode).uses_budget
        budget_values = parse_float_values(args.budget_values) if mode_uses_budget else [0.0]
        for p_recheck in parse_float_values(args.p_recheck_values):
            for budget in budget_values:
                configs.append({
                    "sweep": "l1_frontier",
                    "l1_mode": mode,
                    "p_recheck": p_recheck,
                    "budget_ratio": budget,
                    "random_recheck_ratio": args.random_recheck_ratio if mode_uses_budget else 0.0,
                    "theta_tol": DEFAULT_POLICY.theta_tol,
                    "theta_rare": DEFAULT_POLICY.theta_rare,
                })
    return configs


def threshold_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    if args.sweep not in {"threshold", "both"}:
        return configs
    mode = normalize_l1_mode(args.threshold_l1_mode)
    for theta_tol in parse_float_values(args.theta_tol_values):
        for theta_rare in parse_float_values(args.theta_rare_values):
            configs.append({
                "sweep": "l2_threshold_grid",
                "l1_mode": mode,
                "p_recheck": args.threshold_p_recheck,
                "budget_ratio": args.threshold_budget,
                "random_recheck_ratio": args.random_recheck_ratio,
                "theta_tol": theta_tol,
                "theta_rare": theta_rare,
            })
    return configs


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seed_values(args.seeds)
    configs = frontier_configs(args) + threshold_configs(args)
    if not configs:
        raise ValueError("No operating-curve configurations selected")

    initial_model, checkpoint_info = pretrain_initial_checkpoint(
        epochs=args.pretrain_epochs,
        batch_size=args.pretrain_batch_size,
        learning_rate=args.pretrain_learning_rate,
    )
    eval_bundle = build_eval_bundle()

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        generator = _make_generator(
            initial_model=initial_model,
            eval_bundle=eval_bundle,
            generator_seed=seed,
        )
        generation_policy = clone_policy(DEFAULT_POLICY)
        rounds, _overall_counts = build_round_bundles(
            policy=generation_policy,
            total_rounds=args.rounds,
            cycle_rounds=args.cycle_rounds,
            generator=generator,
        )
        for config in configs:
            eval_policy = policy_with_overrides(
                clone_policy(DEFAULT_POLICY),
                theta_tol=config["theta_tol"],
                theta_rare=config["theta_rare"],
                recheck_probability=config["p_recheck"],
            )
            l1_config = make_l1_router_config(
                config["l1_mode"],
                cos_deviation_threshold=eval_policy.cosine_filter_threshold,
                queue_budget_ratio=config["budget_ratio"],
                random_recheck_ratio=config["random_recheck_ratio"],
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
                reference_policy=eval_policy,
                fixed_recheck_probability=config["p_recheck"],
                adapt_policy=False,
                audit_mode="dual",
                recheck_seed=20260428 + seed,
                compute_oracle_drift=not args.skip_oracle_drift,
                l1_router_config=l1_config,
            )
            run = {
                "round_summary_rows": round_summary_rows,
                "l1_rows": l1_rows,
                "l2_rows": l2_rows,
                "flpg_baseline_rows": flpg_baseline_rows,
                "summary": summary,
            }
            rows.append(metric_rows_for_run(seed=seed, run=run, **config))

    group_keys = [
        "sweep",
        "l1_mode",
        "p_recheck",
        "l1_queue_budget_ratio",
        "l1_random_recheck_ratio",
        "theta_tol",
        "theta_rare",
    ]
    metric_fields = [
        "audit_queue_ratio",
        "main_accuracy",
        "corner_accuracy",
        "l1_precision_non_honest",
        "l1_honest_routed_rate",
        "l1_fraud_recall",
        "l1_rarity_recall",
        "l1_noise_recall",
        "l2_fraud_precision_cond",
        "l2_fraud_recall_cond",
        "e2e_fraud_catch_rate",
        "fraud_survival_rate",
        "l2_rarity_precision_cond",
        "l2_rarity_recall_cond",
        "e2e_rarity_recognition",
        "rarity_retention",
        "false_rarity_rate",
        "sign_flip_catch_rate",
        "corner_harm_catch_rate",
        "sign_flip_survival",
        "corner_harm_survival",
    ]
    count_fields = [
        "updates_total",
        "l2_evals",
        "l1_high_cosine_dropped",
        "false_rarity_count",
    ]

    write_json(
        output_dir / "l1_l2_operating_curve_config.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "scope": (
                "L1 visibility-cost frontier plus L2 threshold grid. Generated rounds "
                "are fixed per seed using DEFAULT_POLICY so threshold comparisons do "
                "not change the underlying sample distribution."
            ),
            "rounds": args.rounds,
            "cycle_rounds": args.cycle_rounds,
            "seeds": seeds,
            "sweep": args.sweep,
            "configs": configs,
            "pretrain": checkpoint_info,
            "oracle_drift": not args.skip_oracle_drift,
        },
    )
    write_csv(output_dir / "l1_l2_operating_curve_by_seed.csv", [], rows)
    write_csv(
        output_dir / "l1_l2_operating_curve_summary.csv",
        [],
        summarize_rows(
            rows,
            group_keys=group_keys,
            metric_fields=metric_fields,
            count_fields=count_fields,
        ),
    )
    print(f"Wrote L1/L2 operating-curve artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

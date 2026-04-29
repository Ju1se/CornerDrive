#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

for candidate in (PROJECT_ROOT, BACKEND_DIR, SCRIPTS_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from common.schemas import DEFAULT_POLICY, Policy  # noqa: E402
from export_thesis_artifacts import (  # noqa: E402
    RoundBundle,
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
    spearman_corr,
)
from generate_demo_data import cosine_similarity  # noqa: E402
from l1_linear_defense.config import make_l1_router_config  # noqa: E402
from l2_dual_audit.classifier import Classification  # noqa: E402
from policy_agent.analysis.unified_benchmark import _make_generator  # noqa: E402


DEFAULT_STRESS_SEEDS = "20260318,20260319,20260320,20260321,20260322"
DEFAULT_THRESHOLD_SEEDS = "20260318,20260319,20260320"
RARITY_STRESS_CONFIGS = ("baseline_outlier", "mixed_overlap", "hard_mixed")
PROXY_CONFIGS = ("default", "small_50", "biased_1_7", "random_main")
THRESHOLD_CONFIGS = (
    ("theta_rare", -0.01),
    ("theta_rare", -0.03),
    ("theta_rare", -0.05),
    ("theta_tol", 0.025),
    ("theta_tol", 0.050),
    ("theta_tol", 0.075),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export V2.5 mechanism stress tests without changing the main benchmark."
    )
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--cycle-rounds", type=int, default=12)
    parser.add_argument("--pretrain-epochs", type=int, default=5)
    parser.add_argument("--pretrain-batch-size", type=int, default=64)
    parser.add_argument("--pretrain-learning-rate", type=float, default=1e-3)
    parser.add_argument("--p-recheck", type=float, default=0.10)
    parser.add_argument(
        "--l1-router-mode",
        type=str,
        default="v25_cosine_fixed",
        help="L1 router mode: v25_cosine_fixed, m1, m2, m3, or m4/l1v3.",
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
    parser.add_argument("--seeds", type=str, default=DEFAULT_STRESS_SEEDS)
    parser.add_argument("--threshold-seeds", type=str, default=DEFAULT_THRESHOLD_SEEDS)
    parser.add_argument(
        "--experiments",
        type=str,
        default="rarity,proxy,threshold",
        help="Comma-separated subset: rarity,proxy,threshold.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "v25_stress_tests",
    )
    return parser.parse_args()


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


def make_generator_for_seed(seed: int, initial_model: Any, eval_bundle: Any):
    generator = _make_generator(
        initial_model=initial_model,
        eval_bundle=eval_bundle,
        generator_seed=seed,
    )
    generator.ground_truth_mode = "archetype"
    return generator


def build_seed_rounds(
    *,
    seed: int,
    initial_model: Any,
    eval_bundle: Any,
    policy: Policy,
    rounds: int,
    cycle_rounds: int,
) -> tuple[Any, list[RoundBundle]]:
    generator = make_generator_for_seed(seed, initial_model, eval_bundle)
    round_bundles, _counts = build_round_bundles(
        policy=policy,
        total_rounds=rounds,
        cycle_rounds=cycle_rounds,
        generator=generator,
    )
    return generator, round_bundles


def honest_reference_for_round(round_bundle: RoundBundle) -> np.ndarray:
    honest = [
        gradient
        for role, gradient in zip(round_bundle.planned_role_by_index, round_bundle.gradients)
        if role == "HONEST"
    ]
    if honest:
        return np.mean(np.stack(honest), axis=0)
    return np.mean(np.stack(round_bundle.gradients), axis=0)


def candidate_metrics(generator: Any, vector: np.ndarray, honest_reference: np.ndarray):
    return generator._evaluate_candidate(vector, honest_reference)


def search_overlap_rarity(
    *,
    generator: Any,
    policy: Policy,
    round_index: int,
    vehicle_id: str,
    honest_reference: np.ndarray,
) -> tuple[np.ndarray, Any]:
    round_drift = generator._round_drift_vector(round_index)
    style = generator._vehicle_style_vector(vehicle_id)
    bases = [
        0.82 * generator.main_gradient + 1.05 * generator.corner_gradient + 0.06 * round_drift,
        0.95 * generator.main_gradient + 1.18 * generator.corner_gradient + 0.04 * style,
        0.70 * generator.main_gradient
        + 1.00 * generator.corner_gradient
        + 0.08 * generator._basis_from_key(f"rarity-overlap:{vehicle_id}"),
    ]
    scales = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]

    best: tuple[float, np.ndarray, Any] | None = None
    for base in bases:
        for scale in scales:
            vector = generator._make_candidate(base, scale)
            metrics = candidate_metrics(generator, vector, honest_reference)
            main_safe = metrics.delta_main <= policy.theta_tol
            corner_help = metrics.delta_corner <= policy.theta_rare
            if main_safe and corner_help:
                score = (
                    -abs(metrics.deviation - 0.35)
                    + max(0.0, policy.theta_rare - metrics.delta_corner) * 4.0
                    - max(0.0, metrics.delta_main) * 2.0
                )
            else:
                score = (
                    -abs(metrics.delta_corner - policy.theta_rare)
                    - max(0.0, metrics.delta_main - policy.theta_tol) * 5.0
                    - abs(metrics.deviation - 0.35) * 0.5
                )
            if best is None or score > best[0]:
                best = (score, vector, metrics)

    assert best is not None
    return best[1], best[2]


def search_weak_rarity(
    *,
    generator: Any,
    policy: Policy,
    round_index: int,
    vehicle_id: str,
    honest_reference: np.ndarray,
) -> tuple[np.ndarray, Any]:
    round_drift = generator._round_drift_vector(round_index)
    style = generator._vehicle_style_vector(vehicle_id)
    bases = [
        0.20 * generator.main_gradient + 1.05 * generator.corner_gradient + 0.04 * round_drift,
        0.35 * generator.main_gradient + 0.90 * generator.corner_gradient + 0.04 * style,
        -0.05 * generator.main_gradient
        + 0.90 * generator.corner_gradient
        + 0.05 * generator._basis_from_key(f"rarity-weak:{vehicle_id}"),
    ]
    scales = [0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]

    best: tuple[float, np.ndarray, Any] | None = None
    for base in bases:
        for scale in scales:
            vector = generator._make_candidate(base, scale)
            metrics = candidate_metrics(generator, vector, honest_reference)
            score = (
                -abs(metrics.delta_corner - policy.theta_rare)
                - max(0.0, metrics.delta_main - policy.theta_tol) * 6.0
                - abs(metrics.deviation - 0.30) * 0.25
            )
            if best is None or score > best[0]:
                best = (score, vector, metrics)

    assert best is not None
    return best[1], best[2]


def annotate_rarity(
    *,
    round_bundle: RoundBundle,
    idx: int,
    subtype: str,
    vector: np.ndarray,
    metrics: Any,
    policy: Policy,
    stress_config: str,
) -> None:
    round_bundle.gradients[idx] = vector.astype(float)
    round_bundle.updates[idx]["gradient_data"] = vector.astype(float).tolist()
    metadata = round_bundle.updates[idx]["metadata"]
    metadata.update({
        "stress_experiment": "rarity_overlap",
        "stress_config": stress_config,
        "rarity_subtype": subtype,
        "preflight_role": Classification.RARITY.value,
        "preflight_delta_main": metrics.delta_main,
        "preflight_delta_corner": metrics.delta_corner,
        "preflight_deviation": metrics.deviation,
        "preflight_near_threshold": abs(metrics.delta_corner - policy.theta_rare),
    })


def apply_rarity_stress(
    *,
    rounds: list[RoundBundle],
    generator: Any,
    policy: Policy,
    stress_config: str,
) -> list[RoundBundle]:
    stressed = copy.deepcopy(rounds)
    rarity_counter = 0
    for round_bundle in stressed:
        honest_reference = honest_reference_for_round(round_bundle)
        for idx, role in enumerate(round_bundle.planned_role_by_index):
            if role != "RARITY":
                continue
            if stress_config == "baseline_outlier":
                subtype = "outlier"
            elif stress_config == "mixed_overlap":
                subtype = "outlier" if rarity_counter % 2 == 0 else "overlap"
            elif stress_config == "hard_mixed":
                subtype = ("outlier", "overlap", "weak")[rarity_counter % 3]
            else:
                raise ValueError(f"Unsupported rarity stress config: {stress_config}")

            if subtype == "outlier":
                vector = np.array(round_bundle.gradients[idx], dtype=float)
                metrics = candidate_metrics(generator, vector, honest_reference)
            elif subtype == "overlap":
                vector, metrics = search_overlap_rarity(
                    generator=generator,
                    policy=policy,
                    round_index=round_bundle.cycle_round_index,
                    vehicle_id=round_bundle.vehicle_ids[idx],
                    honest_reference=honest_reference,
                )
            else:
                vector, metrics = search_weak_rarity(
                    generator=generator,
                    policy=policy,
                    round_index=round_bundle.cycle_round_index,
                    vehicle_id=round_bundle.vehicle_ids[idx],
                    honest_reference=honest_reference,
                )

            annotate_rarity(
                round_bundle=round_bundle,
                idx=idx,
                subtype=subtype,
                vector=vector,
                metrics=metrics,
                policy=policy,
                stress_config=stress_config,
            )
            rarity_counter += 1
    return stressed


def tensor_dataset_subset(dataset: Any, indices: list[int]) -> torch.utils.data.TensorDataset:
    idx = torch.tensor(indices, dtype=torch.long)
    return torch.utils.data.TensorDataset(
        dataset.data[idx].clone(),
        dataset.targets[idx].clone(),
    )


def proxy_eval_bundle(eval_bundle: Any, proxy_type: str) -> Any:
    audit_corner = eval_bundle.audit_corner
    if proxy_type == "default":
        proxy = audit_corner
    elif proxy_type == "small_50":
        proxy = tensor_dataset_subset(audit_corner, list(range(min(50, len(audit_corner)))))
    elif proxy_type == "biased_1_7":
        targets = audit_corner.targets
        indices = [
            idx
            for idx, target in enumerate(targets.tolist())
            if int(target) in {1, 7}
        ]
        proxy = tensor_dataset_subset(audit_corner, indices)
    elif proxy_type == "random_main":
        count = min(len(audit_corner), len(eval_bundle.audit_main))
        proxy = tensor_dataset_subset(eval_bundle.audit_main, list(range(count)))
    else:
        raise ValueError(f"Unsupported proxy type: {proxy_type}")
    return replace(eval_bundle, audit_corner=proxy)


def run_cornerdrive(
    *,
    seed: int,
    rounds: list[RoundBundle],
    initial_model: Any,
    eval_bundle: Any,
    policy: Policy,
    p_recheck: float,
    l1_router_mode: str,
    l1_queue_budget_ratio: float,
    l1_random_recheck_ratio: float,
    compute_oracle_drift: bool = True,
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
    )
    return {
        "round_summary_rows": round_summary_rows,
        "l1_rows": l1_rows,
        "l2_rows": l2_rows,
        "policy_rows": policy_rows,
        "flpg_baseline_rows": flpg_baseline_rows,
        "summary": summary,
    }


def audit_oracle_summary(l2_rows: list[dict[str, Any]]) -> dict[str, float]:
    audited = [
        row
        for row in l2_rows
        if bool_value(row.get("audited_in_l2"))
        and row.get("oracle_delta_l_corner") not in {"", None}
    ]
    if not audited:
        return {
            "audit_oracle_corner_sign_agree": 0.0,
            "audit_oracle_corner_spearman": 0.0,
            "audit_oracle_main_sign_agree": 0.0,
            "audit_oracle_main_spearman": 0.0,
            "audited_with_oracle": 0,
        }

    def sign(value: float) -> int:
        return 1 if value > 0 else -1 if value < 0 else 0

    corner_audit = [float(row["delta_l_corner"]) for row in audited]
    corner_oracle = [float(row["oracle_delta_l_corner"]) for row in audited]
    main_audit = [float(row["delta_l_main"]) for row in audited]
    main_oracle = [float(row["oracle_delta_l_main"]) for row in audited]
    return {
        "audit_oracle_corner_sign_agree": safe_ratio(
            sum(1 for left, right in zip(corner_audit, corner_oracle) if sign(left) == sign(right)),
            len(audited),
        ),
        "audit_oracle_corner_spearman": spearman_corr(corner_audit, corner_oracle),
        "audit_oracle_main_sign_agree": safe_ratio(
            sum(1 for left, right in zip(main_audit, main_oracle) if sign(left) == sign(right)),
            len(audited),
        ),
        "audit_oracle_main_spearman": spearman_corr(main_audit, main_oracle),
        "audited_with_oracle": len(audited),
    }


def run_level_metrics(run: dict[str, Any]) -> dict[str, float]:
    round_rows = run["round_summary_rows"]
    l1_rows = run["l1_rows"]
    l2_rows = run["l2_rows"]
    rarity_total = sum(1 for row in l2_rows if row.get("planned_role") == "RARITY")
    rarity_routed = sum(
        1
        for row in l1_rows
        if row.get("planned_role") == "RARITY" and bool_value(row.get("routed_to_l2"))
    )
    rarity_recognized = sum(
        1 for row in l2_rows if row.get("planned_role") == "RARITY" and row.get("verdict") == "RARITY"
    )
    rarity_retained = sum(
        1
        for row in l2_rows
        if row.get("planned_role") == "RARITY" and float(row.get("aggregation_weight", 0.0)) > 0.0
    )
    non_rarity = [row for row in l2_rows if row.get("planned_role") != "RARITY"]
    false_rarity = sum(1 for row in non_rarity if row.get("verdict") == "RARITY")
    family_stats = family_stats_from_l2(l2_rows)
    oracle = audit_oracle_summary(l2_rows)
    return {
        "main_acc": mean(float(row["main_task_accuracy"]) for row in round_rows),
        "corner_acc": mean(float(row["corner_case_accuracy"]) for row in round_rows),
        "audit_ratio": mean(float(row["queue_ratio_qt"]) for row in round_rows),
        "rarity_recog": safe_ratio(rarity_recognized, rarity_total),
        "rarity_l2_recog": safe_ratio(rarity_recognized, rarity_routed),
        "rarity_retention": safe_ratio(rarity_retained, rarity_total),
        "false_rarity": safe_ratio(false_rarity, len(non_rarity)),
        "false_rarity_count": false_rarity,
        "corner_harm_survival": family_stats["survival_rate"]["corner_harm"],
        "sign_flip_survival": family_stats["survival_rate"]["sign_flip_proxy"],
        **oracle,
    }


def rarity_type_metrics(run: dict[str, Any]) -> list[dict[str, Any]]:
    l1_by_key = {
        (row["round_id"], row["client_id"]): row
        for row in run["l1_rows"]
        if row.get("planned_role") == "RARITY"
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run["l2_rows"]:
        if row.get("planned_role") != "RARITY":
            continue
        subtype = str(row.get("rarity_subtype") or "outlier")
        grouped[subtype].append(row)

    rows: list[dict[str, Any]] = []
    for subtype, items in sorted(grouped.items()):
        total = len(items)
        routed = sum(
            1
            for row in items
            if bool_value(l1_by_key[(row["round_id"], row["client_id"])].get("routed_to_l2"))
        )
        recognized = sum(1 for row in items if row.get("verdict") == "RARITY")
        retained = sum(1 for row in items if float(row.get("aggregation_weight", 0.0)) > 0.0)
        rows.append({
            "rarity_subtype": subtype,
            "rarity_total": total,
            "l1_route": safe_ratio(routed, total),
            "l2_recog": safe_ratio(recognized, routed),
            "e2e_recog": safe_ratio(recognized, total),
            "retention": safe_ratio(retained, total),
            "mean_delta_l_corner": mean(float(row["delta_l_corner"]) for row in items),
            "mean_delta_l_main": mean(float(row["delta_l_main"]) for row in items),
        })
    return rows


def run_rarity_stress(
    *,
    seeds: list[int],
    args: argparse.Namespace,
    initial_model: Any,
    eval_bundle: Any,
    reference_policy: Policy,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_rows: list[dict[str, Any]] = []
    type_rows: list[dict[str, Any]] = []
    for seed in seeds:
        generator, base_rounds = build_seed_rounds(
            seed=seed,
            initial_model=initial_model,
            eval_bundle=eval_bundle,
            policy=reference_policy,
            rounds=args.rounds,
            cycle_rounds=args.cycle_rounds,
        )
        for config in RARITY_STRESS_CONFIGS:
            rounds = apply_rarity_stress(
                rounds=base_rounds,
                generator=generator,
                policy=reference_policy,
                stress_config=config,
            )
            run = run_cornerdrive(
                seed=seed,
                rounds=rounds,
                initial_model=initial_model,
                eval_bundle=eval_bundle,
                policy=reference_policy,
                p_recheck=args.p_recheck,
                l1_router_mode=args.l1_router_mode,
                l1_queue_budget_ratio=args.l1_queue_budget_ratio,
                l1_random_recheck_ratio=args.l1_random_recheck_ratio,
            )
            raw_rows.append({
                "experiment": "rarity_overlap",
                "config": config,
                "seed": seed,
                **run_level_metrics(run),
            })
            for row in rarity_type_metrics(run):
                type_rows.append({
                    "experiment": "rarity_overlap",
                    "config": config,
                    "seed": seed,
                    **row,
                })
    return raw_rows, type_rows


def run_proxy_stress(
    *,
    seeds: list[int],
    args: argparse.Namespace,
    initial_model: Any,
    eval_bundle: Any,
    reference_policy: Policy,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        _generator, rounds = build_seed_rounds(
            seed=seed,
            initial_model=initial_model,
            eval_bundle=eval_bundle,
            policy=reference_policy,
            rounds=args.rounds,
            cycle_rounds=args.cycle_rounds,
        )
        for proxy_type in PROXY_CONFIGS:
            proxy_bundle = proxy_eval_bundle(eval_bundle, proxy_type)
            run = run_cornerdrive(
                seed=seed,
                rounds=rounds,
                initial_model=initial_model,
                eval_bundle=proxy_bundle,
                policy=reference_policy,
                p_recheck=args.p_recheck,
                l1_router_mode=args.l1_router_mode,
                l1_queue_budget_ratio=args.l1_queue_budget_ratio,
                l1_random_recheck_ratio=args.l1_random_recheck_ratio,
            )
            rows.append({
                "experiment": "corner_proxy",
                "proxy_type": proxy_type,
                "proxy_corner_size": len(proxy_bundle.audit_corner),
                "seed": seed,
                **run_level_metrics(run),
            })
    return rows


def run_threshold_stress(
    *,
    seeds: list[int],
    args: argparse.Namespace,
    initial_model: Any,
    eval_bundle: Any,
    reference_policy: Policy,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        generator, base_rounds = build_seed_rounds(
            seed=seed,
            initial_model=initial_model,
            eval_bundle=eval_bundle,
            policy=reference_policy,
            rounds=args.rounds,
            cycle_rounds=args.cycle_rounds,
        )
        rounds = apply_rarity_stress(
            rounds=base_rounds,
            generator=generator,
            policy=reference_policy,
            stress_config="hard_mixed",
        )
        for parameter, value in THRESHOLD_CONFIGS:
            policy = policy_with_overrides(reference_policy, **{parameter: value})
            run = run_cornerdrive(
                seed=seed,
                rounds=rounds,
                initial_model=initial_model,
                eval_bundle=eval_bundle,
                policy=policy,
                p_recheck=0.5,
                l1_router_mode=args.l1_router_mode,
                l1_queue_budget_ratio=args.l1_queue_budget_ratio,
                l1_random_recheck_ratio=args.l1_random_recheck_ratio,
            )
            rows.append({
                "experiment": "threshold",
                "stress_config": "hard_mixed",
                "visibility_mode": "max_policy_recheck",
                "p_recheck": 0.5,
                "parameter": parameter,
                "value": value,
                "seed": seed,
                **run_level_metrics(run),
            })
    return rows


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    experiments = {part.strip() for part in args.experiments.split(",") if part.strip()}
    seed_values = parse_seed_values(args.seeds)
    threshold_seeds = parse_seed_values(args.threshold_seeds)

    reference_policy = clone_policy(DEFAULT_POLICY)
    initial_model, checkpoint_info = pretrain_initial_checkpoint(
        epochs=args.pretrain_epochs,
        batch_size=args.pretrain_batch_size,
        learning_rate=args.pretrain_learning_rate,
    )
    eval_bundle = build_eval_bundle()

    rarity_raw: list[dict[str, Any]] = []
    rarity_type_raw: list[dict[str, Any]] = []
    proxy_raw: list[dict[str, Any]] = []
    threshold_raw: list[dict[str, Any]] = []

    if "rarity" in experiments:
        rarity_raw, rarity_type_raw = run_rarity_stress(
            seeds=seed_values,
            args=args,
            initial_model=initial_model,
            eval_bundle=eval_bundle,
            reference_policy=reference_policy,
        )

    if "proxy" in experiments:
        proxy_raw = run_proxy_stress(
            seeds=seed_values,
            args=args,
            initial_model=initial_model,
            eval_bundle=eval_bundle,
            reference_policy=reference_policy,
        )

    if "threshold" in experiments:
        threshold_raw = run_threshold_stress(
            seeds=threshold_seeds,
            args=args,
            initial_model=initial_model,
            eval_bundle=eval_bundle,
            reference_policy=reference_policy,
        )

    metric_fields = [
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
    ]
    type_metric_fields = [
        "l1_route",
        "l2_recog",
        "e2e_recog",
        "retention",
        "mean_delta_l_corner",
        "mean_delta_l_main",
    ]

    write_json(
        output_dir / "v25_stress_run_config.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "rounds": args.rounds,
            "cycle_rounds": args.cycle_rounds,
            "p_recheck": args.p_recheck,
            "l1_router_mode": args.l1_router_mode,
            "l1_queue_budget_ratio": args.l1_queue_budget_ratio,
            "l1_random_recheck_ratio": args.l1_random_recheck_ratio,
            "seeds": seed_values,
            "threshold_seeds": threshold_seeds,
            "experiments": sorted(experiments),
            "rarity_configs": list(RARITY_STRESS_CONFIGS),
            "proxy_configs": list(PROXY_CONFIGS),
            "threshold_configs": [
                {"parameter": parameter, "value": value}
                for parameter, value in THRESHOLD_CONFIGS
            ],
            "initial_checkpoint": checkpoint_info,
        },
    )

    write_csv(output_dir / "stress_rarity_overlap_raw.csv", [], rarity_raw)
    write_csv(
        output_dir / "stress_rarity_overlap_summary.csv",
        [],
        summarize_rows(
            rarity_raw,
            group_keys=["experiment", "config"],
            metric_fields=metric_fields,
            count_fields=["false_rarity_count", "audited_with_oracle"],
        ),
    )
    write_csv(output_dir / "stress_rarity_by_type_raw.csv", [], rarity_type_raw)
    write_csv(
        output_dir / "stress_rarity_by_type_summary.csv",
        [],
        summarize_rows(
            rarity_type_raw,
            group_keys=["experiment", "config", "rarity_subtype"],
            metric_fields=type_metric_fields,
            count_fields=["rarity_total"],
        ),
    )
    write_csv(output_dir / "stress_proxy_sensitivity_raw.csv", [], proxy_raw)
    write_csv(
        output_dir / "stress_proxy_sensitivity_summary.csv",
        [],
        summarize_rows(
            proxy_raw,
            group_keys=["experiment", "proxy_type"],
            metric_fields=metric_fields,
            count_fields=["false_rarity_count", "audited_with_oracle"],
        ),
    )
    write_csv(output_dir / "stress_threshold_sensitivity_raw.csv", [], threshold_raw)
    write_csv(
        output_dir / "stress_threshold_sensitivity_summary.csv",
        [],
        summarize_rows(
            threshold_raw,
            group_keys=["experiment", "stress_config", "visibility_mode", "p_recheck", "parameter", "value"],
            metric_fields=metric_fields,
            count_fields=["false_rarity_count", "audited_with_oracle"],
        ),
    )

    print(f"Exported V2.5 stress-test artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

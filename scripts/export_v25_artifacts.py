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

from common.config import L2_LEARNING_RATE  # noqa: E402
from common.schemas import DEFAULT_POLICY, Policy  # noqa: E402
from export_thesis_artifacts import (  # noqa: E402
    RoundBundle,
    build_round_bundles,
    build_run_config,
    clone_policy,
    policy_with_recheck_probability,
    pretrain_initial_checkpoint,
    build_eval_bundle,
    classify_shadow_audit,
    run_baseline_method,
    run_flpg_with_artifacts,
    safe_mean,
    write_csv,
    write_json,
)
from generate_demo_data import (  # noqa: E402
    ARCHETYPE_TO_ATTACK_FAMILY,
    ARCHETYPE_TO_GROUND_TRUTH_LABEL,
    GRADIENT_REFRESH_INTERVAL,
    ROLES_ORDER,
)
from l1_linear_defense.config import make_l1_router_config  # noqa: E402
from policy_agent.analysis.unified_benchmark import _make_auditor, _make_generator  # noqa: E402


ARCHETYPE_TABLE_ORDER = ("HONEST", "RARITY", "NOISE", "FRAUD", "FRAUD_CORNER_HARM")
BASELINE_METHODS = ("fedavg", "geomed", "krum")
METHOD_LABELS = {
    "fedavg": "FedAvg",
    "geomed": "GeoMed",
    "krum": "Multi-Krum",
}
FAMILY_ORDER = ("sign_flip_proxy", "corner_harm")
SHADOW_VERDICT_COLUMNS = ("FRAUD", "RARITY", "HONEST", "NOISE")
UPDATE_CONFUSION_COLUMNS = ("Fraud", "Rarity", "HonestSafe", "Noise", "Not audited")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export V2.5 anti-circularity benchmark tables."
    )
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--cycle-rounds", type=int, default=12)
    parser.add_argument("--pretrain-epochs", type=int, default=5)
    parser.add_argument("--pretrain-batch-size", type=int, default=64)
    parser.add_argument("--pretrain-learning-rate", type=float, default=1e-3)
    parser.add_argument("--krum-byzantine-budget", type=int, default=10)
    parser.add_argument(
        "--recheck-values",
        type=str,
        default="0.0,0.10",
        help="Comma-separated CornerDrive recheck probabilities.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="20260318",
        help="Comma-separated generator seeds. Use multiple seeds for stress-test aggregation.",
    )
    parser.add_argument(
        "--ablation-recheck-probability",
        type=float,
        default=0.10,
        help="Recheck probability used by main-only and corner-only audit ablations.",
    )
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "v25_artifacts",
    )
    return parser.parse_args()


def parse_recheck_values(raw: str) -> list[float]:
    values: list[float] = []
    for part in raw.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        values.append(float(stripped))
    return values or [0.0, 0.10]


def parse_seed_values(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        values.append(int(stripped))
    return values or [20260318]


def p_label(probability: float) -> str:
    return f"p={probability:.2f}"


def p_column(probability: float) -> str:
    return f"p{probability:.2f}"


def archetype_label(archetype: str) -> str:
    if archetype == "FRAUD":
        return "FRAUD sign-flip"
    if archetype == "FRAUD_CORNER_HARM":
        return "FRAUD corner-harm"
    return archetype


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)


def true_type_label(row: dict[str, Any]) -> str:
    planned_role = str(row.get("planned_role", ""))
    if planned_role == "FRAUD":
        return "Sign-flip"
    if planned_role == "FRAUD_CORNER_HARM":
        return "Corner-harm"
    if planned_role == "HONEST":
        return "Honest"
    if planned_role == "RARITY":
        return "Rarity"
    if planned_role == "NOISE":
        return "Noise"
    return planned_role or "Unknown"


def verdict_label(row: dict[str, Any], *, end_to_end: bool) -> str:
    if end_to_end and not bool_value(row.get("audited_in_l2")):
        return "Not audited"
    verdict = str(row.get("verdict", row.get("shadow_verdict", "NOISE")))
    if verdict == "FRAUD":
        return "Fraud"
    if verdict == "RARITY":
        return "Rarity"
    if verdict in {"HONEST", "HONEST_SAFE"}:
        return "HonestSafe"
    return "Noise"


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def family_stats_from_l2(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals: Counter = Counter()
    survived: Counter = Counter()
    caught_by_reason: Counter = Counter()
    caught_by_family_reason: Counter = Counter()

    for row in rows:
        if str(row.get("true_label")) != "FRAUD":
            continue
        family = str(row.get("attack_family", "none"))
        if family == "none":
            continue
        totals[family] += 1
        if float(row.get("aggregation_weight", 0.0)) > 0.0:
            survived[family] += 1
        if (
            bool_value(row.get("audited_in_l2"))
            and str(row.get("shadow_verdict")) == "FRAUD"
        ):
            reason = str(row.get("routing_reason", "unknown"))
            caught_by_reason[reason] += 1
            caught_by_family_reason[(family, reason)] += 1

    return {
        "totals": totals,
        "survived": survived,
        "survival_rate": {
            family: survived[family] / totals[family] if totals[family] else 0.0
            for family in FAMILY_ORDER
        },
        "caught_by_reason": caught_by_reason,
        "caught_by_family_reason": caught_by_family_reason,
    }


def baseline_family_stats(rows: list[dict[str, Any]], method_id: str) -> dict[str, float]:
    method_rows = [row for row in rows if row["method"] == method_id]
    stats: dict[str, float] = {}
    for family, prefix in (
        ("sign_flip_proxy", "sign_flip_proxy"),
        ("corner_harm", "corner_harm"),
    ):
        total = sum(int(row.get(f"{prefix}_total", 0)) for row in method_rows)
        survived = sum(int(row.get(f"{prefix}_survival_count", 0)) for row in method_rows)
        stats[family] = survived / total if total else 0.0
    return stats


def honest_recheck_rate_for_run(run: dict[str, Any]) -> float:
    honest_rows = [row for row in run["l1_rows"] if row.get("planned_role") == "HONEST"]
    honest_rechecked = [
        row
        for row in honest_rows
        if bool_value(row.get("routed_to_l2"))
        and str(row.get("routing_reason")) == "probabilistic_recheck"
    ]
    return safe_ratio(len(honest_rechecked), len(honest_rows))


def build_main_result_table(
    *,
    baseline_rows: list[dict[str, Any]],
    flpg_runs: dict[float, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for method_id in BASELINE_METHODS:
        method_rows = [row for row in baseline_rows if row["method"] == method_id]
        family_stats = baseline_family_stats(baseline_rows, method_id)
        rows.append({
            "method": METHOD_LABELS[method_id],
            "main_accuracy": safe_mean([float(row["main_task_accuracy"]) for row in method_rows]),
            "corner_accuracy": safe_mean([float(row["corner_case_accuracy"]) for row in method_rows]),
            "rarity_recall": "",
            "false_rarity": "",
            "sign_flip_survival": family_stats["sign_flip_proxy"],
            "corner_harm_survival": family_stats["corner_harm"],
            "audit_queue_ratio": "",
            "honest_recheck_rate": "",
            "rounds": len(method_rows),
        })

    for probability, run in flpg_runs.items():
        round_rows = run["round_summary_rows"]
        family_stats = family_stats_from_l2(run["l2_rows"])
        rarity = rarity_metrics_for_run(run)
        rows.append({
            "method": f"CornerDrive {p_label(probability)}",
            "main_accuracy": safe_mean(
                [float(row["main_task_accuracy"]) for row in round_rows]
            ),
            "corner_accuracy": safe_mean(
                [float(row["corner_case_accuracy"]) for row in round_rows]
            ),
            "rarity_recall": rarity["l2_rarity_recall"],
            "false_rarity": rarity["false_rarity_preservation_rate"],
            "sign_flip_survival": family_stats["survival_rate"]["sign_flip_proxy"],
            "corner_harm_survival": family_stats["survival_rate"]["corner_harm"],
            "audit_queue_ratio": safe_mean(
                [float(row["queue_ratio_qt"]) for row in round_rows]
            ),
            "honest_recheck_rate": honest_recheck_rate_for_run(run),
            "rounds": len(round_rows),
        })

    return rows


def rarity_metrics_for_run(run: dict[str, Any]) -> dict[str, float]:
    l1_rows = run["l1_rows"]
    l2_rows = run["l2_rows"]
    rarity_total = sum(1 for row in l1_rows if row["planned_role"] == "RARITY")
    rarity_routed = sum(
        1
        for row in l1_rows
        if row["planned_role"] == "RARITY" and bool_value(row["routed_to_l2"])
    )
    l2_rarity_hits = sum(
        1
        for row in l2_rows
        if row["planned_role"] == "RARITY"
        and bool_value(row["audited_in_l2"])
        and row["verdict"] == "RARITY"
    )
    retained_rarity = sum(
        1
        for row in l2_rows
        if row["planned_role"] == "RARITY"
        and float(row.get("aggregation_weight", 0.0)) > 0.0
    )
    non_rarity_total = sum(1 for row in l2_rows if row["planned_role"] != "RARITY")
    false_rarity = sum(
        1
        for row in l2_rows
        if row["planned_role"] != "RARITY" and row["verdict"] == "RARITY"
    )
    return {
        "rarity_total": float(rarity_total),
        "l1_rarity_routing_rate": rarity_routed / rarity_total if rarity_total else 0.0,
        "l2_rarity_recall": l2_rarity_hits / rarity_routed if rarity_routed else 0.0,
        "end_to_end_rarity_retention": retained_rarity / rarity_total if rarity_total else 0.0,
        "false_rarity_preservation_count": float(false_rarity),
        "false_rarity_preservation_rate": false_rarity / non_rarity_total if non_rarity_total else 0.0,
    }


def build_archetype_generation_counts(rounds: list[RoundBundle]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    totals: Counter = Counter()

    for round_bundle in rounds:
        counts = Counter(round_bundle.planned_role_by_index)
        totals.update(counts)
        for archetype in ARCHETYPE_TABLE_ORDER:
            rows.append({
                "round_id": round_bundle.round_id,
                "phase_name": round_bundle.phase,
                "archetype": archetype,
                "ground_truth": ARCHETYPE_TO_GROUND_TRUTH_LABEL[archetype],
                "attack_family": ARCHETYPE_TO_ATTACK_FAMILY[archetype],
                "count": int(counts.get(archetype, 0)),
            })

    for archetype in ARCHETYPE_TABLE_ORDER:
        rows.append({
            "round_id": "TOTAL",
            "phase_name": "ALL",
            "archetype": archetype,
            "ground_truth": ARCHETYPE_TO_GROUND_TRUTH_LABEL[archetype],
            "attack_family": ARCHETYPE_TO_ATTACK_FAMILY[archetype],
            "count": int(totals.get(archetype, 0)),
        })

    return rows


def build_l1_routing_by_archetype(
    flpg_runs: dict[float, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for probability, run in flpg_runs.items():
        grouped: dict[str, Counter] = defaultdict(Counter)
        for row in run["l1_rows"]:
            archetype = str(row["planned_role"])
            grouped[archetype]["total"] += 1
            routed = bool_value(row["routed_to_l2"])
            reason = str(row.get("routing_reason", "bypass"))
            if routed and reason == "cosine_screening":
                grouped[archetype]["routed_by_cosine"] += 1
            elif routed and reason == "probabilistic_recheck":
                grouped[archetype]["routed_by_recheck"] += 1
            else:
                grouped[archetype]["bypassed"] += 1

        for archetype in ARCHETYPE_TABLE_ORDER:
            counts = grouped[archetype]
            total = int(counts.get("total", 0))
            routed_total = int(counts.get("routed_by_cosine", 0)) + int(
                counts.get("routed_by_recheck", 0)
            )
            rows.append({
                "setting": p_label(probability),
                "archetype": archetype,
                "ground_truth": ARCHETYPE_TO_GROUND_TRUTH_LABEL[archetype],
                "attack_family": ARCHETYPE_TO_ATTACK_FAMILY[archetype],
                "total": total,
                "routed_by_cosine": int(counts.get("routed_by_cosine", 0)),
                "routed_by_recheck": int(counts.get("routed_by_recheck", 0)),
                "bypassed": int(counts.get("bypassed", 0)),
                "routing_rate": routed_total / total if total else 0.0,
            })

    return rows


def normalize_shadow_verdict(verdict: str) -> str:
    if verdict == "HONEST_SAFE":
        return "HONEST"
    return verdict if verdict in SHADOW_VERDICT_COLUMNS else "NOISE"


def build_l2_confusion_matrix(flpg_runs: dict[float, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for probability, run in flpg_runs.items():
        matrix: dict[str, Counter] = defaultdict(Counter)
        audited_counts: Counter = Counter()
        for row in run["l2_rows"]:
            archetype = str(row["planned_role"])
            verdict = normalize_shadow_verdict(str(row["shadow_verdict"]))
            matrix[archetype][verdict] += 1
            if bool_value(row.get("audited_in_l2")):
                audited_counts[archetype] += 1

        for archetype in ARCHETYPE_TABLE_ORDER:
            counts = matrix[archetype]
            out = {
                "setting": p_label(probability),
                "ground_truth_archetype": archetype_label(archetype),
                "ground_truth": ARCHETYPE_TO_GROUND_TRUTH_LABEL[archetype],
                "attack_family": ARCHETYPE_TO_ATTACK_FAMILY[archetype],
                "audited_in_l2_count": int(audited_counts.get(archetype, 0)),
            }
            for verdict in SHADOW_VERDICT_COLUMNS:
                out[verdict] = int(counts.get(verdict, 0))
            rows.append(out)

    return rows


def build_rarity_discovery_metrics(flpg_runs: dict[float, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for probability, run in flpg_runs.items():
        metrics = rarity_metrics_for_run(run)
        rows.append({
            "setting": p_label(probability),
            **metrics,
        })

    return rows


def std_value(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def summarize_multiseed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)

    metric_names = [
        "main_accuracy",
        "corner_accuracy",
        "rarity_recall",
        "false_rarity",
        "sign_flip_survival",
        "corner_harm_survival",
        "audit_queue_ratio",
        "honest_recheck_rate",
    ]
    out: list[dict[str, Any]] = []
    for method, items in grouped.items():
        record: dict[str, Any] = {"method": method, "seeds": len(items)}
        for metric in metric_names:
            values = [
                float(item[metric])
                for item in items
                if item.get(metric) not in {"", None}
            ]
            if not values:
                record[f"{metric}_mean"] = ""
                record[f"{metric}_std"] = ""
                record[metric] = ""
                continue
            metric_mean = mean(values)
            metric_std = std_value(values)
            record[f"{metric}_mean"] = metric_mean
            record[f"{metric}_std"] = metric_std
            record[metric] = f"{metric_mean:.6f} +/- {metric_std:.6f}"
        out.append(record)
    return out


def build_ablation_result_table(
    *,
    ablation_runs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode, run in ablation_runs.items():
        round_rows = run["round_summary_rows"]
        family_stats = family_stats_from_l2(run["l2_rows"])
        rarity = rarity_metrics_for_run(run)
        rows.append({
            "audit_mode": mode,
            "main_accuracy": safe_mean([float(row["main_task_accuracy"]) for row in round_rows]),
            "corner_accuracy": safe_mean([float(row["corner_case_accuracy"]) for row in round_rows]),
            "rarity_recall": rarity["l2_rarity_recall"],
            "false_rarity": rarity["false_rarity_preservation_rate"],
            "sign_flip_survival": family_stats["survival_rate"]["sign_flip_proxy"],
            "corner_harm_survival": family_stats["survival_rate"]["corner_harm"],
            "audit_queue_ratio": safe_mean([float(row["queue_ratio_qt"]) for row in round_rows]),
            "honest_recheck_rate": honest_recheck_rate_for_run(run),
        })
    return rows


def build_recheck_sweep_table(seed_main_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        row
        for row in seed_main_rows
        if str(row.get("method", "")).startswith("CornerDrive ")
    ]
    summarized = summarize_multiseed_rows(rows)
    for row in summarized:
        row["p_recheck"] = str(row["method"]).replace("CornerDrive p=", "")
    return sorted(summarized, key=lambda row: float(row["p_recheck"]))


def build_update_confusion_matrix(
    l2_rows: list[dict[str, Any]],
    *,
    end_to_end: bool,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], Counter] = defaultdict(Counter)
    totals: Counter = Counter()
    for row in l2_rows:
        if not end_to_end and not bool_value(row.get("audited_in_l2")):
            continue
        setting = str(row.get("setting", ""))
        true_type = true_type_label(row)
        label = verdict_label(row, end_to_end=end_to_end)
        grouped[(setting, true_type)][label] += 1
        totals[(setting, true_type)] += 1

    rows: list[dict[str, Any]] = []
    type_order = ("Honest", "Rarity", "Noise", "Sign-flip", "Corner-harm")
    for setting in sorted({key[0] for key in grouped.keys()}):
        for true_type in type_order:
            counts = grouped.get((setting, true_type), Counter())
            total = int(totals.get((setting, true_type), 0))
            if total == 0:
                continue
            out: dict[str, Any] = {
                "setting": setting,
                "true_type": true_type,
                "total": total,
            }
            for column in UPDATE_CONFUSION_COLUMNS:
                if column == "Not audited" and not end_to_end:
                    continue
                out[column] = int(counts.get(column, 0))
                out[f"{column}_rate"] = safe_ratio(float(counts.get(column, 0)), total)
            rows.append(out)
    return rows


def build_rarity_recognition_retention_rows(
    l2_rows: list[dict[str, Any]],
    *,
    group_key: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in l2_rows:
        grouped[str(row.get(group_key, ""))].append(row)

    rows: list[dict[str, Any]] = []
    for group_value, items in sorted(grouped.items()):
        rarity_rows = [row for row in items if row.get("planned_role") == "RARITY"]
        non_rarity_rows = [row for row in items if row.get("planned_role") != "RARITY"]
        rarity_total = len(rarity_rows)
        routed = sum(1 for row in rarity_rows if bool_value(row.get("audited_in_l2")))
        recognized = sum(1 for row in rarity_rows if row.get("verdict") == "RARITY")
        retained = sum(
            1 for row in rarity_rows if float(row.get("aggregation_weight", 0.0)) > 0.0
        )
        rejected = rarity_total - retained
        false_rarity = sum(1 for row in non_rarity_rows if row.get("verdict") == "RARITY")
        record = {
            group_key: group_value,
            "rarity_total": rarity_total,
            "rarity_routed_to_l2": routed,
            "rarity_recognized_as_rarity": recognized,
            "rarity_retained": retained,
            "rarity_rejected": rejected,
            "recognition_rate": safe_ratio(recognized, rarity_total),
            "conditional_recognition_rate": safe_ratio(recognized, routed),
            "retention_rate": safe_ratio(retained, rarity_total),
            "false_rarity_count": false_rarity,
            "false_rarity_rate": safe_ratio(false_rarity, len(non_rarity_rows)),
            "false_rejection_rate": safe_ratio(rejected, rarity_total),
        }
        rows.append(record)
    return rows


def build_ablation_main_harm_rows(l2_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in l2_rows:
        grouped[str(row.get("audit_mode", ""))].append(row)

    rows: list[dict[str, Any]] = []
    for audit_mode, items in sorted(grouped.items()):
        accepted = [
            row for row in items if float(row.get("aggregation_weight", 0.0)) > 0.0
        ]
        main_harm = [
            row
            for row in accepted
            if float(row.get("delta_l_main", 0.0)) > float(row.get("theta_tol", 0.0))
        ]
        conflict = [
            row
            for row in main_harm
            if float(row.get("delta_l_corner", 0.0)) < 0.0
        ]
        rarity_rows = [row for row in items if row.get("planned_role") == "RARITY"]
        retained_rarity = [
            row for row in rarity_rows if float(row.get("aggregation_weight", 0.0)) > 0.0
        ]
        rows.append({
            "audit_mode": audit_mode,
            "accepted_total": len(accepted),
            "main_harm_accepted": len(main_harm),
            "main_harm_accepted_rate": safe_ratio(len(main_harm), len(accepted)),
            "conflict_accepted": len(conflict),
            "conflict_accepted_rate": safe_ratio(len(conflict), len(accepted)),
            "rarity_retention": safe_ratio(len(retained_rarity), len(rarity_rows)),
        })
    return rows


def build_conflict_probe_rows(
    *,
    seed: int,
    generator,
    initial_model,
    eval_bundle,
    policy: Policy,
) -> list[dict[str, Any]]:
    auditor = _make_auditor(
        initial_model,
        main_dataset=eval_bundle.audit_main,
        corner_dataset=eval_bundle.audit_corner,
    )
    auditor.apply_policy(policy)
    base_main_loss = auditor.compute_loss(auditor.model, auditor.main_loader)
    base_corner_loss = auditor.compute_loss(auditor.model, auditor.corner_loader)
    bases = [
        -1.20 * generator.main_gradient
        + 1.70 * generator.corner_gradient
        + 0.20 * generator._basis_from_key("conflict-probe-a"),
        -0.90 * generator.main_gradient
        + 1.45 * generator.corner_gradient
        + 0.25 * generator._basis_from_key("conflict-probe-b"),
        -1.55 * generator.main_gradient
        + 2.05 * generator.corner_gradient
        + 0.20 * generator._basis_from_key("conflict-probe-c"),
    ]
    scales = [1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0]

    selected = None
    selected_metrics = None
    best = None
    best_score = float("-inf")
    best_metrics = None
    for base_index, base in enumerate(bases):
        for scale in scales:
            candidate = generator._make_candidate(base, scale)
            metrics = classify_shadow_audit(
                auditor=auditor,
                vehicle_id=f"conflict_probe:{seed}:{base_index}:{scale}",
                gradient=candidate,
                base_main_loss=base_main_loss,
                base_corner_loss=base_corner_loss,
                audit_mode="dual",
            )
            conflict = (
                float(metrics["delta_l_main"]) > auditor.fraud_threshold
                and float(metrics["delta_l_corner"]) <= auditor.rarity_threshold
            )
            if conflict:
                selected = candidate
                selected_metrics = metrics
                break
            score = float(metrics["delta_l_main"]) - float(metrics["delta_l_corner"])
            if score > best_score:
                best_score = score
                best = candidate
                best_metrics = metrics
        if selected is not None:
            break

    probe = selected if selected is not None else best
    metrics = selected_metrics if selected_metrics is not None else best_metrics
    if probe is None or metrics is None:
        return []

    rows: list[dict[str, Any]] = []
    for audit_mode in ("main_only", "corner_only", "dual"):
        verdict = classify_shadow_audit(
            auditor=auditor,
            vehicle_id=f"conflict_probe:{seed}:{audit_mode}",
            gradient=probe,
            base_main_loss=base_main_loss,
            base_corner_loss=base_corner_loss,
            audit_mode=audit_mode,
        )
        accepted = bool_value(verdict["include_in_aggregation"])
        main_harm = float(verdict["delta_l_main"]) > auditor.fraud_threshold
        conflict = main_harm and float(verdict["delta_l_corner"]) <= auditor.rarity_threshold
        rows.append({
            "seed": seed,
            "audit_mode": audit_mode,
            "delta_l_main": verdict["delta_l_main"],
            "delta_l_corner": verdict["delta_l_corner"],
            "theta_tol": auditor.fraud_threshold,
            "theta_rare": auditor.rarity_threshold,
            "verdict": verdict["verdict"],
            "action": verdict["action"],
            "accepted": accepted,
            "main_harm": main_harm,
            "corner_improves": float(verdict["delta_l_corner"]) < 0.0,
            "conflict": conflict,
            "target_found": selected is not None,
        })
    return rows


def rank_values(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    idx = 0
    while idx < len(indexed):
        end = idx + 1
        while end < len(indexed) and indexed[end][1] == indexed[idx][1]:
            end += 1
        average_rank = (idx + 1 + end) / 2.0
        for original_index, _value in indexed[idx:end]:
            ranks[original_index] = average_rank
        idx = end
    return ranks


def pearson_corr(left: list[float], right: list[float]) -> float:
    if len(left) < 2 or len(left) != len(right):
        return 0.0
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_den = sum((x - left_mean) ** 2 for x in left) ** 0.5
    right_den = sum((y - right_mean) ** 2 for y in right) ** 0.5
    if left_den == 0.0 or right_den == 0.0:
        return 0.0
    return numerator / (left_den * right_den)


def spearman_corr(left: list[float], right: list[float]) -> float:
    return pearson_corr(rank_values(left), rank_values(right))


def sign_bucket(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def build_audit_oracle_consistency_rows(l2_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in l2_rows:
        if not bool_value(row.get("audited_in_l2")):
            continue
        if row.get("oracle_delta_l_main") in {"", None}:
            continue
        grouped[str(row.get("setting", ""))].append(row)

    rows: list[dict[str, Any]] = []
    for setting, items in sorted(grouped.items()):
        for signal, audit_key, oracle_key in (
            ("main_drift", "delta_l_main", "oracle_delta_l_main"),
            ("corner_drift", "delta_l_corner", "oracle_delta_l_corner"),
        ):
            audit_values = [float(row[audit_key]) for row in items]
            oracle_values = [float(row[oracle_key]) for row in items]
            agree = sum(
                1
                for audit_value, oracle_value in zip(audit_values, oracle_values)
                if sign_bucket(audit_value) == sign_bucket(oracle_value)
            )
            rows.append({
                "setting": setting,
                "signal": signal,
                "audited_updates": len(items),
                "sign_agree": safe_ratio(agree, len(items)),
                "spearman": spearman_corr(audit_values, oracle_values),
            })
    return rows


def build_fraud_survival_by_family(
    flpg_runs: dict[float, dict[str, Any]],
    recheck_values: list[float],
) -> list[dict[str, Any]]:
    stats_by_p = {
        probability: family_stats_from_l2(run["l2_rows"])
        for probability, run in flpg_runs.items()
    }
    visibility_probability = max(recheck_values)
    visibility_stats = stats_by_p[visibility_probability]
    rows: list[dict[str, Any]] = []

    for family in FAMILY_ORDER:
        row: dict[str, Any] = {"family": family}
        for probability in recheck_values:
            row[f"survival_{p_column(probability)}"] = stats_by_p[probability][
                "survival_rate"
            ][family]
        total = int(visibility_stats["totals"].get(family, 0))
        cosine_count = int(
            visibility_stats["caught_by_family_reason"].get((family, "cosine_screening"), 0)
        )
        recheck_count = int(
            visibility_stats["caught_by_family_reason"].get(
                (family, "probabilistic_recheck"), 0
            )
        )
        row.update({
            "total_in_visibility_setting": total,
            "caught_by_cosine": cosine_count,
            "caught_by_recheck": recheck_count,
            "caught_by_cosine_rate": cosine_count / total if total else 0.0,
            "caught_by_recheck_rate": recheck_count / total if total else 0.0,
        })
        rows.append(row)

    return rows


def build_energy_attack_validation(rounds: list[RoundBundle]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for round_bundle in rounds:
        for update in round_bundle.updates:
            metadata = update["metadata"]
            family = str(metadata.get("attack_family", "none"))
            if family in FAMILY_ORDER:
                grouped[family].append(metadata)

    rows: list[dict[str, Any]] = []
    for family in FAMILY_ORDER:
        records = grouped[family]
        if family == "sign_flip_proxy":
            delta_key = "anchor_delta_main"
        else:
            delta_key = "anchor_delta_corner"

        deltas = [float(record.get(delta_key, 0.0)) for record in records]
        delta_main = [float(record.get("anchor_delta_main", 0.0)) for record in records]
        delta_corner = [float(record.get("anchor_delta_corner", 0.0)) for record in records]
        search_scales = [
            float(record["anchor_search_scale_used"])
            for record in records
            if record.get("anchor_search_scale_used") is not None
        ]
        anchor_norms = [
            float(record["anchor_scale_used"])
            for record in records
            if record.get("anchor_scale_used") is not None
        ]
        passes = [bool_value(record.get("target_passed")) for record in records]
        target_condition = str(records[0].get("target_condition", "")) if records else ""

        rows.append({
            "family": family,
            "target_condition": target_condition,
            "sample_count": len(records),
            "pass_rate": sum(1 for passed in passes if passed) / len(passes) if passes else 0.0,
            "mean_delta": safe_mean(deltas),
            "min_delta": min(deltas) if deltas else 0.0,
            "mean_delta_main": safe_mean(delta_main),
            "mean_delta_corner": safe_mean(delta_corner),
            "search_scale_min": min(search_scales) if search_scales else 0.0,
            "search_scale_max": max(search_scales) if search_scales else 0.0,
            "scale_range_used": (
                f"{min(search_scales):.2f}-{max(search_scales):.2f}"
                if search_scales
                else "n/a"
            ),
            "anchor_norm_min": min(anchor_norms) if anchor_norms else 0.0,
            "anchor_norm_max": max(anchor_norms) if anchor_norms else 0.0,
        })

    return rows


def build_dataset_isolation_config(eval_bundle) -> dict[str, Any]:
    return {
        "ground_truth_mode": "archetype",
        "split_isolation": "generator_proto_vs_l2_audit_vs_final_oracle",
        "disjoint_by_construction": True,
        "disjointness_basis": (
            "PlaceholderDataset instances use separate seeds for proto, audit, and oracle "
            "paths; generator directions use proto splits, L2 uses audit splits, and "
            "reported accuracies use oracle splits."
        ),
        "sets": {
            "D_proto_main": {"size": len(eval_bundle.proto_main), "owner": "generator"},
            "D_proto_corner": {"size": len(eval_bundle.proto_corner), "owner": "generator"},
            "D_audit_main": {"size": len(eval_bundle.audit_main), "owner": "l2_auditor"},
            "D_audit_corner": {"size": len(eval_bundle.audit_corner), "owner": "l2_auditor"},
            "D_oracle_main": {"size": len(eval_bundle.oracle_main), "owner": "evaluator"},
            "D_oracle_corner": {"size": len(eval_bundle.oracle_corner), "owner": "evaluator"},
        },
    }


def build_v25_run_config(
    *,
    args: argparse.Namespace,
    base_config: dict[str, Any],
    recheck_values: list[float],
    checkpoint_info: dict[str, Any],
    overall_counts: Counter,
) -> dict[str, Any]:
    payload = dict(base_config)
    payload.update({
        "experiment_id": (
            f"cornerdrive_v25_r{args.rounds}_c{args.cycle_rounds}_"
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        ),
        "benchmark_version": "V2.5",
        "ground_truth_mode": "archetype",
        "policy_adaptation": "disabled",
        "policy_adaptation_reason": (
            "V2.5 isolates server-side audit behavior; oracle splits are report-only "
            "and do not feed threshold updates."
        ),
        "recheck_values": recheck_values,
        "gradient_refresh_interval": GRADIENT_REFRESH_INTERVAL,
        "attack_scale_search_ranges": {
            "sign_flip_proxy": [10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0],
            "corner_harm": [2.0, 5.0, 10.0, 20.0, 40.0, 80.0],
        },
        "initial_checkpoint": checkpoint_info,
        "overall_ground_truth_counts": dict(overall_counts),
        "server_lr_eta": L2_LEARNING_RATE,
        "result_tables": [
            "v25_main_result_table.csv",
            "v25_main_result_table_by_seed.csv",
            "v25_recheck_sweep_table.csv",
            "v25_audit_ablation_table.csv",
            "v25_audit_ablation_by_seed.csv",
            "v25_audit_ablation_l2_raw.csv",
            "v25_update_confusion_l2_conditional.csv",
            "v25_update_confusion_end_to_end.csv",
            "v25_rarity_recognition_retention.csv",
            "v25_audit_ablation_rarity_recognition_retention.csv",
            "v25_audit_ablation_main_harm.csv",
            "v25_audit_ablation_conflict_probe.csv",
            "v25_audit_oracle_consistency.csv",
            "v25_archetype_generation_counts.csv",
            "v25_l1_routing_by_archetype_reason.csv",
            "v25_l2_confusion_matrix.csv",
            "v25_rarity_discovery_metrics.csv",
            "v25_fraud_survival_by_family.csv",
            "v25_energy_attack_validation.csv",
        ],
    })
    return payload


def add_setting(rows: list[dict[str, Any]], probability: float) -> list[dict[str, Any]]:
    return [{**row, "setting": p_label(probability)} for row in rows]


def add_seed(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    return [{**row, "seed": seed} for row in rows]


def summarize_multiseed_ablation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = [{**row, "method": row["audit_mode"]} for row in rows]
    summarized = summarize_multiseed_rows(normalized)
    for row in summarized:
        row["audit_mode"] = row.pop("method")
    return summarized


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    recheck_values = parse_recheck_values(args.recheck_values)
    seed_values = parse_seed_values(args.seeds)

    reference_policy = clone_policy(DEFAULT_POLICY)
    base_l1_router_config = make_l1_router_config(
        args.l1_router_mode,
        cos_deviation_threshold=reference_policy.cosine_filter_threshold,
        queue_budget_ratio=args.l1_queue_budget_ratio,
        random_recheck_ratio=args.l1_random_recheck_ratio,
    )
    initial_model, checkpoint_info = pretrain_initial_checkpoint(
        epochs=args.pretrain_epochs,
        batch_size=args.pretrain_batch_size,
        learning_rate=args.pretrain_learning_rate,
    )
    eval_bundle = build_eval_bundle()

    seed_main_rows: list[dict[str, Any]] = []
    seed_ablation_rows: list[dict[str, Any]] = []
    all_baseline_rows: list[dict[str, Any]] = []
    all_archetype_counts: list[dict[str, Any]] = []
    all_l1_routing: list[dict[str, Any]] = []
    all_l2_confusion: list[dict[str, Any]] = []
    all_rarity_metrics: list[dict[str, Any]] = []
    all_fraud_family: list[dict[str, Any]] = []
    all_energy_validation: list[dict[str, Any]] = []
    raw_l1: list[dict[str, Any]] = []
    raw_l2: list[dict[str, Any]] = []
    raw_round_summary: list[dict[str, Any]] = []
    raw_policy: list[dict[str, Any]] = []
    raw_flpg_baseline: list[dict[str, Any]] = []
    raw_ablation_l2: list[dict[str, Any]] = []
    conflict_probe_rows: list[dict[str, Any]] = []
    recheck_summaries: dict[str, Any] = {}
    ablation_summaries: dict[str, Any] = {}
    overall_counts_all: Counter = Counter()
    first_generator = None
    first_rounds: list[RoundBundle] = []

    for seed_index, generator_seed in enumerate(seed_values):
        generator = _make_generator(
            initial_model=initial_model,
            eval_bundle=eval_bundle,
            generator_seed=generator_seed,
        )
        generator.ground_truth_mode = "archetype"
        rounds, overall_counts = build_round_bundles(
            policy=reference_policy,
            total_rounds=args.rounds,
            cycle_rounds=args.cycle_rounds,
            generator=generator,
        )
        if first_generator is None:
            first_generator = generator
            first_rounds = rounds
        overall_counts_all.update(overall_counts)
        conflict_probe_rows.extend(
            build_conflict_probe_rows(
                seed=generator_seed,
                generator=generator,
                initial_model=initial_model,
                eval_bundle=eval_bundle,
                policy=reference_policy,
            )
        )

        baseline_rows: list[dict[str, Any]] = []
        for method_id in BASELINE_METHODS:
            baseline_rows.extend(
                run_baseline_method(
                    method_id=method_id,
                    rounds=rounds,
                    initial_model=initial_model,
                    eval_bundle=eval_bundle,
                    krum_byzantine_budget=args.krum_byzantine_budget,
                )
            )
        for row in baseline_rows:
            row["seed"] = generator_seed
        all_baseline_rows.extend(baseline_rows)

        flpg_runs: dict[float, dict[str, Any]] = {}
        for probability in recheck_values:
            policy = policy_with_recheck_probability(reference_policy, probability)
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
                fixed_recheck_probability=probability,
                adapt_policy=False,
                audit_mode="dual",
                recheck_seed=20260428 + generator_seed,
                l1_router_config=base_l1_router_config,
            )
            flpg_runs[probability] = {
                "round_summary_rows": add_seed(add_setting(round_summary_rows, probability), generator_seed),
                "l1_rows": add_seed(add_setting(l1_rows, probability), generator_seed),
                "l2_rows": add_seed(add_setting(l2_rows, probability), generator_seed),
                "policy_rows": add_seed(add_setting(policy_rows, probability), generator_seed),
                "flpg_baseline_rows": add_seed(add_setting(flpg_baseline_rows, probability), generator_seed),
                "summary": summary,
            }
            recheck_summaries[f"seed={generator_seed}:{p_label(probability)}"] = summary

        ablation_runs: dict[str, dict[str, Any]] = {}
        for audit_mode in ("main_only", "corner_only", "dual"):
            policy = policy_with_recheck_probability(
                reference_policy,
                args.ablation_recheck_probability,
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
                fixed_recheck_probability=args.ablation_recheck_probability,
                adapt_policy=False,
                audit_mode=audit_mode,
                recheck_seed=20260428 + generator_seed,
                l1_router_config=base_l1_router_config,
            )
            ablation_runs[audit_mode] = {
                "round_summary_rows": add_seed(round_summary_rows, generator_seed),
                "l1_rows": add_seed(l1_rows, generator_seed),
                "l2_rows": add_seed(l2_rows, generator_seed),
                "policy_rows": add_seed(policy_rows, generator_seed),
                "flpg_baseline_rows": add_seed(flpg_baseline_rows, generator_seed),
                "summary": summary,
            }
            ablation_summaries[f"seed={generator_seed}:{audit_mode}"] = summary
            raw_ablation_l2.extend([
                {**row, "seed": generator_seed, "audit_mode": audit_mode}
                for row in l2_rows
            ])

        main_seed_table = build_main_result_table(
            baseline_rows=baseline_rows,
            flpg_runs=flpg_runs,
        )
        for row in main_seed_table:
            row["seed"] = generator_seed
        seed_main_rows.extend(main_seed_table)

        ablation_seed_table = build_ablation_result_table(ablation_runs=ablation_runs)
        for row in ablation_seed_table:
            row["seed"] = generator_seed
        seed_ablation_rows.extend(ablation_seed_table)

        all_archetype_counts.extend(add_seed(build_archetype_generation_counts(rounds), generator_seed))
        all_l1_routing.extend(add_seed(build_l1_routing_by_archetype(flpg_runs), generator_seed))
        all_l2_confusion.extend(add_seed(build_l2_confusion_matrix(flpg_runs), generator_seed))
        all_rarity_metrics.extend(add_seed(build_rarity_discovery_metrics(flpg_runs), generator_seed))
        all_fraud_family.extend(add_seed(build_fraud_survival_by_family(flpg_runs, recheck_values), generator_seed))
        all_energy_validation.extend(add_seed(build_energy_attack_validation(rounds), generator_seed))

        raw_l1.extend([row for run in flpg_runs.values() for row in run["l1_rows"]])
        raw_l2.extend([row for run in flpg_runs.values() for row in run["l2_rows"]])
        raw_round_summary.extend([
            row for run in flpg_runs.values() for row in run["round_summary_rows"]
        ])
        raw_policy.extend([row for run in flpg_runs.values() for row in run["policy_rows"]])
        raw_flpg_baseline.extend([
            row for run in flpg_runs.values() for row in run["flpg_baseline_rows"]
        ])

    if first_generator is None:
        raise RuntimeError("No generator seeds were provided")

    base_config = build_run_config(
        args=args,
        initial_policy=reference_policy,
        eval_bundle=eval_bundle,
        clients_total=len(first_generator.vehicle_pool),
        clients_per_round=len(first_rounds[0].gradients) if first_rounds else 0,
    )
    run_config = build_v25_run_config(
        args=args,
        base_config=base_config,
        recheck_values=recheck_values,
        checkpoint_info=checkpoint_info,
        overall_counts=overall_counts_all,
    )
    run_config["seeds"] = seed_values
    run_config["seed_count"] = len(seed_values)
    run_config["ablation_recheck_probability"] = args.ablation_recheck_probability
    run_config["l1_router_mode"] = args.l1_router_mode
    run_config["l1_queue_budget_ratio"] = args.l1_queue_budget_ratio
    run_config["l1_random_recheck_ratio"] = args.l1_random_recheck_ratio

    main_table = summarize_multiseed_rows(seed_main_rows)
    ablation_table = summarize_multiseed_ablation_rows(seed_ablation_rows)
    recheck_sweep_table = build_recheck_sweep_table(seed_main_rows)
    l2_conditional_confusion = build_update_confusion_matrix(raw_l2, end_to_end=False)
    end_to_end_confusion = build_update_confusion_matrix(raw_l2, end_to_end=True)
    rarity_recognition = build_rarity_recognition_retention_rows(
        raw_l2,
        group_key="setting",
    )
    ablation_rarity_recognition = build_rarity_recognition_retention_rows(
        raw_ablation_l2,
        group_key="audit_mode",
    )
    ablation_main_harm = build_ablation_main_harm_rows(raw_ablation_l2)
    audit_oracle_consistency = build_audit_oracle_consistency_rows(raw_l2)
    archetype_counts = all_archetype_counts
    l1_routing = all_l1_routing
    l2_confusion = all_l2_confusion
    rarity_metrics = all_rarity_metrics
    fraud_family = all_fraud_family
    energy_validation = all_energy_validation
    dataset_isolation = build_dataset_isolation_config(eval_bundle)

    write_json(output_dir / "v25_run_config.json", run_config)
    write_json(output_dir / "v25_dataset_isolation_config.json", dataset_isolation)
    write_json(
        output_dir / "v25_recheck_summaries.json",
        recheck_summaries,
    )
    write_json(
        output_dir / "v25_audit_ablation_summaries.json",
        ablation_summaries,
    )
    write_csv(
        output_dir / "v25_main_result_table.csv",
        [
            "method",
            "seeds",
            "main_accuracy",
            "corner_accuracy",
            "rarity_recall",
            "false_rarity",
            "sign_flip_survival",
            "corner_harm_survival",
            "audit_queue_ratio",
            "honest_recheck_rate",
        ],
        main_table,
    )
    write_csv(
        output_dir / "v25_main_result_table_by_seed.csv",
        [
            "seed",
            "method",
            "main_accuracy",
            "corner_accuracy",
            "rarity_recall",
            "false_rarity",
            "sign_flip_survival",
            "corner_harm_survival",
            "audit_queue_ratio",
            "honest_recheck_rate",
        ],
        seed_main_rows,
    )
    write_csv(
        output_dir / "v25_recheck_sweep_table.csv",
        [
            "p_recheck",
            "seeds",
            "main_accuracy",
            "corner_accuracy",
            "corner_harm_survival",
            "audit_queue_ratio",
            "honest_recheck_rate",
            "sign_flip_survival",
            "rarity_recall",
            "false_rarity",
        ],
        recheck_sweep_table,
    )
    write_csv(
        output_dir / "v25_audit_ablation_table.csv",
        [
            "audit_mode",
            "seeds",
            "main_accuracy",
            "corner_accuracy",
            "rarity_recall",
            "false_rarity",
            "sign_flip_survival",
            "corner_harm_survival",
            "audit_queue_ratio",
            "honest_recheck_rate",
        ],
        ablation_table,
    )
    write_csv(
        output_dir / "v25_audit_ablation_by_seed.csv",
        [
            "seed",
            "audit_mode",
            "main_accuracy",
            "corner_accuracy",
            "rarity_recall",
            "false_rarity",
            "sign_flip_survival",
            "corner_harm_survival",
            "audit_queue_ratio",
            "honest_recheck_rate",
        ],
        seed_ablation_rows,
    )
    write_csv(
        output_dir / "v25_update_confusion_l2_conditional.csv",
        [
            "setting",
            "true_type",
            "total",
            "Fraud",
            "Rarity",
            "HonestSafe",
            "Noise",
        ],
        l2_conditional_confusion,
    )
    write_csv(
        output_dir / "v25_update_confusion_end_to_end.csv",
        [
            "setting",
            "true_type",
            "total",
            "Fraud",
            "Rarity",
            "HonestSafe",
            "Noise",
            "Not audited",
        ],
        end_to_end_confusion,
    )
    write_csv(
        output_dir / "v25_rarity_recognition_retention.csv",
        [
            "setting",
            "rarity_total",
            "rarity_routed_to_l2",
            "rarity_recognized_as_rarity",
            "rarity_retained",
            "recognition_rate",
            "conditional_recognition_rate",
            "retention_rate",
            "false_rarity_count",
            "false_rarity_rate",
            "false_rejection_rate",
        ],
        rarity_recognition,
    )
    write_csv(
        output_dir / "v25_audit_ablation_rarity_recognition_retention.csv",
        [
            "audit_mode",
            "rarity_total",
            "rarity_routed_to_l2",
            "rarity_recognized_as_rarity",
            "rarity_retained",
            "recognition_rate",
            "conditional_recognition_rate",
            "retention_rate",
            "false_rarity_count",
            "false_rarity_rate",
            "false_rejection_rate",
        ],
        ablation_rarity_recognition,
    )
    write_csv(
        output_dir / "v25_audit_ablation_main_harm.csv",
        [
            "audit_mode",
            "accepted_total",
            "main_harm_accepted",
            "main_harm_accepted_rate",
            "conflict_accepted",
            "conflict_accepted_rate",
            "rarity_retention",
        ],
        ablation_main_harm,
    )
    write_csv(
        output_dir / "v25_audit_ablation_l2_raw.csv",
        [
            "seed",
            "audit_mode",
            "round_id",
            "client_id",
            "true_label",
            "planned_role",
            "attack_family",
            "delta_l_main",
            "delta_l_corner",
            "oracle_delta_l_main",
            "oracle_delta_l_corner",
            "shadow_verdict",
            "verdict",
            "action",
            "aggregation_weight",
            "audited_in_l2",
            "routing_reason",
            "recheck_probability",
            "phase_name",
        ],
        raw_ablation_l2,
    )
    write_csv(
        output_dir / "v25_audit_ablation_conflict_probe.csv",
        [
            "seed",
            "audit_mode",
            "delta_l_main",
            "delta_l_corner",
            "theta_tol",
            "theta_rare",
            "verdict",
            "action",
            "accepted",
            "main_harm",
            "corner_improves",
            "conflict",
            "target_found",
        ],
        conflict_probe_rows,
    )
    write_csv(
        output_dir / "v25_audit_oracle_consistency.csv",
        [
            "setting",
            "signal",
            "audited_updates",
            "sign_agree",
            "spearman",
        ],
        audit_oracle_consistency,
    )
    write_csv(
        output_dir / "v25_archetype_generation_counts.csv",
        [
            "round_id",
            "phase_name",
            "archetype",
            "ground_truth",
            "attack_family",
            "count",
        ],
        archetype_counts,
    )
    write_csv(
        output_dir / "v25_l1_routing_by_archetype_reason.csv",
        [
            "setting",
            "archetype",
            "ground_truth",
            "attack_family",
            "total",
            "routed_by_cosine",
            "routed_by_recheck",
            "bypassed",
            "routing_rate",
        ],
        l1_routing,
    )
    write_csv(
        output_dir / "v25_l2_confusion_matrix.csv",
        [
            "setting",
            "ground_truth_archetype",
            "ground_truth",
            "attack_family",
            "audited_in_l2_count",
            "FRAUD",
            "RARITY",
            "HONEST",
            "NOISE",
        ],
        l2_confusion,
    )
    write_csv(
        output_dir / "v25_rarity_discovery_metrics.csv",
        [
            "setting",
            "rarity_total",
            "l1_rarity_routing_rate",
            "l2_rarity_recall",
            "end_to_end_rarity_retention",
            "false_rarity_preservation_count",
            "false_rarity_preservation_rate",
        ],
        rarity_metrics,
    )
    write_csv(
        output_dir / "v25_fraud_survival_by_family.csv",
        [
            "family",
            *[f"survival_{p_column(probability)}" for probability in recheck_values],
            "total_in_visibility_setting",
            "caught_by_cosine",
            "caught_by_recheck",
            "caught_by_cosine_rate",
            "caught_by_recheck_rate",
        ],
        fraud_family,
    )
    write_csv(
        output_dir / "v25_energy_attack_validation.csv",
        [
            "family",
            "target_condition",
            "sample_count",
            "pass_rate",
            "mean_delta",
            "min_delta",
            "scale_range_used",
        ],
        energy_validation,
    )
    write_csv(
        output_dir / "v25_baseline_diagnostics_raw.csv",
        [
            "round_id",
            "method",
            "fraud_survival_count",
            "fraud_survival_rate",
            "sign_flip_proxy_total",
            "sign_flip_proxy_survival_count",
            "sign_flip_proxy_survival_rate",
            "corner_harm_total",
            "corner_harm_survival_count",
            "corner_harm_survival_rate",
            "corner_case_accuracy",
            "main_task_accuracy",
            "phase_name",
        ],
        all_baseline_rows,
    )
    write_csv(
        output_dir / "v25_l1_routing_raw.csv",
        [
            "setting",
            "round_id",
            "client_id",
            "true_label",
            "planned_role",
            "routed_to_l2",
            "routing_reason",
            "recheck_probability",
            "phase_name",
        ],
        raw_l1,
    )
    write_csv(
        output_dir / "v25_l2_audit_raw.csv",
        [
            "setting",
            "round_id",
            "client_id",
            "true_label",
            "planned_role",
            "attack_family",
            "delta_l_main",
            "delta_l_corner",
            "oracle_delta_l_main",
            "oracle_delta_l_corner",
            "shadow_verdict",
            "verdict",
            "action",
            "aggregation_weight",
            "audited_in_l2",
            "routing_reason",
            "recheck_probability",
            "phase_name",
        ],
        raw_l2,
    )
    write_csv(
        output_dir / "v25_round_summary_raw.csv",
        [
            "setting",
            "round_id",
            "phase_name",
            "participants",
            "queue_size_l1",
            "queue_ratio_qt",
            "accepted_rarity",
            "rejected_fraud",
            "rejected_corner_harm",
            "main_task_accuracy",
            "corner_case_accuracy",
            "recheck_probability",
        ],
        raw_round_summary,
    )
    write_csv(
        output_dir / "v25_policy_trace_raw.csv",
        [
            "setting",
            "round_id",
            "tau_screen",
            "theta_tol",
            "theta_rare",
            "theta_corner_harm",
            "recheck_probability",
            "update_reason",
            "phase_name",
        ],
        raw_policy,
    )
    write_csv(
        output_dir / "v25_cornerdrive_diagnostics_raw.csv",
        [
            "setting",
            "round_id",
            "method",
            "fraud_survival_count",
            "corner_case_accuracy",
            "main_task_accuracy",
            "phase_name",
        ],
        raw_flpg_baseline,
    )

    print(f"Exported V2.5 benchmark artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

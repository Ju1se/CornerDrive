#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROUTING_COLUMNS = (
    "cosine_screening",
    "norm_mad_screening",
    "sign_screening",
    "risk_topB",
    "stratified_random",
    "probabilistic_recheck",
    "not_routed",
)
CONFUSION_COLUMNS = ("Fraud", "Rarity", "HonestSafe", "Noise", "Not audited")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build thesis-facing tables from L1V3 ablation artifacts."
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        required=True,
        help="Directory produced by scripts/export_l1v3_ablation.py.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def truth_label(row: dict[str, Any]) -> str:
    planned = str(row.get("planned_role", ""))
    if planned == "FRAUD":
        return "Sign-flip"
    if planned == "FRAUD_CORNER_HARM":
        return "Corner-harm"
    if planned == "HONEST":
        return "Honest"
    if planned == "RARITY":
        return "Rarity"
    if planned == "NOISE":
        return "Noise"
    return planned or "Unknown"


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)


def verdict_label(row: dict[str, Any]) -> str:
    if not bool_value(row.get("audited_in_l2")):
        return "Not audited"
    verdict = str(row.get("verdict", row.get("shadow_verdict", "")))
    if verdict == "FRAUD":
        return "Fraud"
    if verdict == "RARITY":
        return "Rarity"
    if verdict in {"HONEST", "HONEST_SAFE"}:
        return "HonestSafe"
    return "Noise"


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _e2e_rarity_by_mode(l2_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in l2_rows:
        if row.get("planned_role") == "RARITY":
            grouped[str(row["l1_mode"])].append(row)

    metrics: dict[str, dict[str, float]] = {}
    for mode, items in grouped.items():
        recognized = sum(1 for row in items if row.get("verdict") == "RARITY")
        retained = sum(
            1 for row in items if float(row.get("aggregation_weight", 0.0)) > 0.0
        )
        metrics[mode] = {
            "rarity_total": float(len(items)),
            "rarity_e2e_recognized": float(recognized),
            "rarity_e2e_recognition": safe_ratio(recognized, len(items)),
            "rarity_retained": float(retained),
        }
    return metrics


def table1_main_by_mode(
    summary_rows: list[dict[str, Any]],
    l2_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    e2e_rarity = _e2e_rarity_by_mode(l2_rows)
    rows: list[dict[str, Any]] = []
    for row in summary_rows:
        rarity = e2e_rarity.get(str(row["l1_mode"]), {})
        conditional_key = (
            "rarity_l2_conditional_recognition_mean"
            if "rarity_l2_conditional_recognition_mean" in row
            else "rarity_recognition_mean"
        )
        rows.append({
            "mode": row["l1_mode"],
            "main_acc_mean": row["main_accuracy_mean"],
            "main_acc_std": row["main_accuracy_std"],
            "corner_acc_mean": row["corner_accuracy_mean"],
            "corner_acc_std": row["corner_accuracy_std"],
            "rarity_l2_conditional_recogn_mean": row[conditional_key],
            "rarity_e2e_recognized": int(rarity.get("rarity_e2e_recognized", 0)),
            "rarity_total": int(rarity.get("rarity_total", 0)),
            "rarity_e2e_recogn_mean": rarity.get("rarity_e2e_recognition", 0.0),
            "rarity_retention_mean": row["rarity_retention_mean"],
            "false_rarity_mean": row["false_rarity_mean"],
            "sign_surv_mean": row["sign_flip_survival_mean"],
            "corner_harm_surv_mean": row["corner_harm_survival_mean"],
            "corner_harm_surv_std": row["corner_harm_survival_std"],
            "queue_ratio_mean": row["audit_queue_ratio_mean"],
            "queue_ratio_std": row["audit_queue_ratio_std"],
            "honest_routed_mean": row["honest_routed_rate_mean"],
        })
    return rows


def table2_routing_by_archetype(l1_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], Counter] = defaultdict(Counter)
    totals: Counter = Counter()
    for row in l1_rows:
        mode = str(row["l1_mode"])
        archetype = truth_label(row)
        key = (mode, archetype)
        totals[key] += 1
        reason = str(row.get("routing_reason", "bypass"))
        if reason == "bypass" or not bool_value(row.get("routed_to_l2")):
            grouped[key]["not_routed"] += 1
        else:
            grouped[key][reason] += 1

    rows: list[dict[str, Any]] = []
    for key in sorted(totals):
        mode, archetype = key
        total = totals[key]
        record: dict[str, Any] = {
            "mode": mode,
            "archetype": archetype,
            "total": total,
        }
        routed_total = 0
        for column in ROUTING_COLUMNS:
            count = grouped[key][column]
            record[column] = count
            record[f"{column}_rate"] = safe_ratio(count, total)
            if column != "not_routed":
                routed_total += count
        record["routed_total"] = routed_total
        record["routed_rate"] = safe_ratio(routed_total, total)
        rows.append(record)
    return rows


def table3_confusion_by_mode(l2_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: Counter = Counter()
    totals: Counter = Counter()
    for row in l2_rows:
        key = (str(row["l1_mode"]), truth_label(row))
        prediction = verdict_label(row)
        grouped[(key[0], key[1], prediction)] += 1
        totals[key] += 1

    rows: list[dict[str, Any]] = []
    for mode, true_type in sorted(totals):
        total = totals[(mode, true_type)]
        record: dict[str, Any] = {
            "mode": mode,
            "true_type": true_type,
            "total": total,
        }
        for column in CONFUSION_COLUMNS:
            count = grouped[(mode, true_type, column)]
            record[column] = count
            record[f"{column}_rate"] = safe_ratio(count, total)
        rows.append(record)
    return rows


def table4_cost_effectiveness(l2_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in l2_rows:
        grouped[str(row["l1_mode"])].append(row)

    rows: list[dict[str, Any]] = []
    for mode, items in sorted(grouped.items()):
        l2_evals = sum(1 for row in items if bool_value(row.get("audited_in_l2")))
        corner_harm_total = sum(1 for row in items if truth_label(row) == "Corner-harm")
        detected_corner_harm = sum(
            1
            for row in items
            if truth_label(row) == "Corner-harm"
            and bool_value(row.get("audited_in_l2"))
            and str(row.get("verdict")) == "FRAUD"
        )
        sign_total = sum(1 for row in items if truth_label(row) == "Sign-flip")
        detected_sign = sum(
            1
            for row in items
            if truth_label(row) == "Sign-flip"
            and bool_value(row.get("audited_in_l2"))
            and str(row.get("verdict")) == "FRAUD"
        )
        total_updates = len(items)
        rows.append({
            "mode": mode,
            "total_updates": total_updates,
            "l2_evals": l2_evals,
            "queue_ratio": safe_ratio(l2_evals, total_updates),
            "detected_corner_harm": detected_corner_harm,
            "corner_harm_total": corner_harm_total,
            "corner_harm_detect_rate": safe_ratio(detected_corner_harm, corner_harm_total),
            "cost_per_detected_corner_harm": safe_ratio(l2_evals, detected_corner_harm),
            "detected_sign_flip": detected_sign,
            "sign_flip_total": sign_total,
            "cost_per_detected_fraud_all": safe_ratio(l2_evals, detected_corner_harm + detected_sign),
        })
    return rows


def main() -> int:
    args = parse_args()
    artifact_dir = args.artifact_dir
    summary_rows = read_csv(artifact_dir / "l1v3_ablation_summary.csv")
    l1_rows = read_csv(artifact_dir / "l1v3_l1_routing_raw.csv")
    l2_rows = read_csv(artifact_dir / "l1v3_l2_audit_raw.csv")

    write_csv(
        artifact_dir / "l1v3_table1_main_by_mode.csv",
        table1_main_by_mode(summary_rows, l2_rows),
    )
    write_csv(
        artifact_dir / "l1v3_table2_routing_by_archetype_reason.csv",
        table2_routing_by_archetype(l1_rows),
    )
    write_csv(
        artifact_dir / "l1v3_table3_confusion_by_mode.csv",
        table3_confusion_by_mode(l2_rows),
    )
    write_csv(
        artifact_dir / "l1v3_table4_cost_effectiveness.csv",
        table4_cost_effectiveness(l2_rows),
    )
    print(f"Wrote L1V3 comparison tables to {artifact_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

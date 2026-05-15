#!/usr/bin/env python3
"""Build paper-facing CSV tables from CornerDrive reproduction outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHOD_ORDER = ["Multi-Krum", "FLTrust", "Zeno", "Zeno++", "CornerDrive"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate thesis CSV tables.")
    parser.add_argument(
        "--real-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "real_gradient_reliability_medium",
    )
    parser.add_argument(
        "--v25-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "audit_reproduction" / "v25_artifacts_b24",
    )
    parser.add_argument(
        "--stress-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "audit_reproduction" / "v25_stress_tests_b24",
    )
    parser.add_argument(
        "--divergence-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "audit_reproduction" / "corner_family_divergence_b24",
    )
    parser.add_argument(
        "--corner-harm-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "audit_reproduction" / "corner_harm_threshold_calibration_b24",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "tables",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        print(f"skip missing input: {path.relative_to(PROJECT_ROOT)}")
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path.relative_to(PROJECT_ROOT)}")


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def real_gradient_provenance(real_dir: Path) -> dict[str, Any]:
    payload = read_json(real_dir / "real_gradient_reliability_summary.json")
    runs = payload.get("runs", [])
    first_config = runs[0].get("config", {}) if runs else {}
    return {
        "summary_json": str(
            (real_dir / "real_gradient_reliability_summary.json").relative_to(PROJECT_ROOT)
        ),
        "sources": payload.get("sources", []),
        "seeds": payload.get("seeds", []),
        "methods": payload.get("methods", []),
        "rounds": first_config.get("rounds"),
        "clients_per_round": first_config.get("clients_per_round"),
        "max_clients": first_config.get("max_clients"),
        "max_samples_per_client": first_config.get("max_samples_per_client"),
        "reference_split_fraction": first_config.get("reference_split_fraction"),
        "max_reference_samples": first_config.get("max_reference_samples"),
        "max_evaluation_samples": first_config.get("max_evaluation_samples"),
        "attack_fraction": first_config.get("attack_fraction"),
        "corner_harm_fraction": first_config.get("corner_harm_fraction"),
        "noise_fraction": first_config.get("noise_fraction"),
        "cornerdrive_l1_mode": first_config.get("cornerdrive_l1_mode"),
        "cornerdrive_l1_cos_weight": first_config.get("cornerdrive_l1_cos_weight"),
        "cornerdrive_l1_norm_weight": first_config.get("cornerdrive_l1_norm_weight"),
        "cornerdrive_l1_sign_weight": first_config.get("cornerdrive_l1_sign_weight"),
        "cornerdrive_l1_norm_mad_threshold": first_config.get("cornerdrive_l1_norm_mad_threshold"),
        "cornerdrive_l1_sign_threshold": first_config.get("cornerdrive_l1_sign_threshold"),
        "cornerdrive_l1_sign_topk_ratio": first_config.get("cornerdrive_l1_sign_topk_ratio"),
        "cornerdrive_l1_queue_budget_ratio": first_config.get("cornerdrive_l1_queue_budget_ratio"),
        "cornerdrive_l1_random_recheck_ratio": first_config.get("cornerdrive_l1_random_recheck_ratio"),
    }


def write_provenance(args: argparse.Namespace) -> None:
    inputs = {
        "real_gradient_summary": args.real_dir / "real_gradient_reliability_summary.csv",
        "v25_main_result_table": args.v25_dir / "v25_main_result_table.csv",
        "stress_rarity_overlap": args.stress_dir / "stress_rarity_overlap_summary.csv",
        "stress_proxy_sensitivity": args.stress_dir / "stress_proxy_sensitivity_summary.csv",
        "corner_family_divergence": args.divergence_dir / "corner_family_divergence_summary.csv",
        "corner_harm_threshold_calibration": (
            args.corner_harm_dir / "corner_harm_threshold_calibration_summary.csv"
        ),
    }
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/make_paper_tables.py",
        "real_gradient_config": real_gradient_provenance(args.real_dir),
        "inputs": {
            key: {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(path),
                "exists": path.exists(),
            }
            for key, path in inputs.items()
        },
        "outputs": [
            "artifacts/tables/table_5_1_real_gradient_macro.csv",
            "artifacts/tables/table_5_2_cornerdrive_real_gradient_by_dataset.csv",
            "artifacts/tables/alg_main_result_table.csv",
            "artifacts/tables/appendix_rarity_overlap.csv",
            "artifacts/tables/appendix_proxy_sensitivity.csv",
            "artifacts/tables/appendix_corner_family_divergence.csv",
            "artifacts/tables/appendix_corner_harm_threshold_calibration.csv",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "table_provenance.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {path.relative_to(PROJECT_ROOT)}")


def f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    raw = row.get(key, "")
    return float(raw) if raw not in {"", None} else default


def pct(value: float) -> float:
    return round(100.0 * value, 4)


def build_real_macro(real_dir: Path, output_dir: Path) -> None:
    rows = read_csv(real_dir / "real_gradient_reliability_summary.csv")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)

    out: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        items = grouped.get(method, [])
        if not items:
            continue
        out.append(
            {
                "method": method,
                "main_accuracy": round(mean(f(r, "main_accuracy_avg_mean") for r in items), 4),
                "corner_accuracy": round(mean(f(r, "corner_accuracy_avg_mean") for r in items), 4),
                "fraud_survival": round(mean(f(r, "fraud_survival_rate_avg_mean") for r in items), 4),
                "rarity_retention": round(mean(f(r, "rarity_retention_rate_avg_mean") for r in items), 4),
            }
        )
    write_csv(output_dir / "table_5_1_real_gradient_macro.csv", out)


def build_real_cornerdrive_by_dataset(real_dir: Path, output_dir: Path) -> None:
    rows = [
        row
        for row in read_csv(real_dir / "real_gradient_reliability_summary.csv")
        if row.get("method") == "CornerDrive"
    ]
    out = [
        {
            "dataset": row.get("source", ""),
            "main_accuracy_mean": round(f(row, "main_accuracy_avg_mean"), 4),
            "main_accuracy_ci95": round(f(row, "main_accuracy_avg_ci95"), 4),
            "corner_accuracy_mean": round(f(row, "corner_accuracy_avg_mean"), 4),
            "corner_accuracy_ci95": round(f(row, "corner_accuracy_avg_ci95"), 4),
            "fraud_survival_mean": round(f(row, "fraud_survival_rate_avg_mean"), 4),
            "fraud_survival_ci95": round(f(row, "fraud_survival_rate_avg_ci95"), 4),
            "rarity_retention_mean": round(f(row, "rarity_retention_rate_avg_mean"), 4),
            "rarity_retention_ci95": round(f(row, "rarity_retention_rate_avg_ci95"), 4),
            "l1_review_mean": round(f(row, "l1_review_rate_avg_mean"), 4),
            "l1_review_ci95": round(f(row, "l1_review_rate_avg_ci95"), 4),
        }
        for row in rows
    ]
    write_csv(output_dir / "table_5_2_cornerdrive_real_gradient_by_dataset.csv", out)


def build_alg_main(v25_dir: Path, output_dir: Path) -> None:
    rows = read_csv(v25_dir / "v25_main_result_table.csv")
    out = [
        {
            "method": row.get("method", ""),
            "main_accuracy_percent": pct(f(row, "main_accuracy_mean")),
            "main_accuracy_std_percent": pct(f(row, "main_accuracy_std")),
            "corner_accuracy_percent": pct(f(row, "corner_accuracy_mean")),
            "corner_accuracy_std_percent": pct(f(row, "corner_accuracy_std")),
            "rarity_recall_percent": pct(f(row, "rarity_recall_mean")) if row.get("rarity_recall_mean") else "",
            "sign_flip_survival_percent": pct(f(row, "sign_flip_survival_mean")) if row.get("sign_flip_survival_mean") else "",
            "corner_harm_survival_percent": pct(f(row, "corner_harm_survival_mean")) if row.get("corner_harm_survival_mean") else "",
        }
        for row in rows
    ]
    write_csv(output_dir / "alg_main_result_table.csv", out)


def build_stress_tables(stress_dir: Path, output_dir: Path) -> None:
    rarity_rows = read_csv(stress_dir / "stress_rarity_overlap_summary.csv")
    write_csv(
        output_dir / "appendix_rarity_overlap.csv",
        [
            {
                "setting": row.get("config", ""),
                "rarity_recognition_percent": pct(f(row, "rarity_recog_mean")),
                "retention_percent": pct(f(row, "rarity_retention_mean")),
                "false_rarity_count": row.get("false_rarity_count", ""),
                "corner_accuracy_percent": pct(f(row, "corner_acc_mean")),
            }
            for row in rarity_rows
        ],
    )

    proxy_rows = read_csv(stress_dir / "stress_proxy_sensitivity_summary.csv")
    write_csv(
        output_dir / "appendix_proxy_sensitivity.csv",
        [
            {
                "proxy": row.get("proxy_type", ""),
                "rarity_recognition_percent": pct(f(row, "rarity_recog_mean")),
                "retention_percent": pct(f(row, "rarity_retention_mean")),
                "corner_accuracy_percent": pct(f(row, "corner_acc_mean")),
                "spearman": round(f(row, "audit_oracle_corner_spearman_mean"), 4),
                "false_rarity_count": row.get("false_rarity_count", ""),
            }
            for row in proxy_rows
        ],
    )


def build_divergence_table(divergence_dir: Path, output_dir: Path) -> None:
    rows = read_csv(divergence_dir / "corner_family_divergence_summary.csv")
    write_csv(
        output_dir / "appendix_corner_family_divergence.csv",
        [
            {
                "rho": row.get("rho", ""),
                "runs": row.get("runs", ""),
                "rarity_recognition_percent": pct(f(row, "rarity_recog_mean")),
                "rarity_recognition_std_percent": pct(f(row, "rarity_recog_std")),
                "retention_percent": pct(f(row, "rarity_retention_mean")),
                "retention_std_percent": pct(f(row, "rarity_retention_std")),
                "false_rarity_count": row.get("false_rarity_count", ""),
                "main_accuracy_percent": pct(f(row, "main_acc_mean")),
                "main_accuracy_std_percent": pct(f(row, "main_acc_std")),
                "corner_accuracy_percent": pct(f(row, "corner_acc_mean")),
                "corner_accuracy_std_percent": pct(f(row, "corner_acc_std")),
            }
            for row in rows
        ],
    )


def build_corner_harm_table(corner_harm_dir: Path, output_dir: Path) -> None:
    rows = read_csv(corner_harm_dir / "corner_harm_threshold_calibration_summary.csv")
    write_csv(
        output_dir / "appendix_corner_harm_threshold_calibration.csv",
        [
            {
                "setting": row.get("setting", ""),
                "theta_corner_harm": round(f(row, "theta_corner_harm_mean"), 6),
                "main_accuracy_percent": pct(f(row, "main_acc_mean")),
                "corner_accuracy_percent": pct(f(row, "corner_acc_mean")),
                "corner_harm_survival_percent": pct(f(row, "corner_harm_survival_mean")),
                "honest_false_corner_harm_reject_percent": pct(
                    f(row, "honest_corner_harm_false_reject_rate_mean")
                ),
            }
            for row in rows
        ],
    )


def main() -> int:
    args = parse_args()
    build_real_macro(args.real_dir, args.output_dir)
    build_real_cornerdrive_by_dataset(args.real_dir, args.output_dir)
    build_alg_main(args.v25_dir, args.output_dir)
    build_stress_tables(args.stress_dir, args.output_dir)
    build_divergence_table(args.divergence_dir, args.output_dir)
    build_corner_harm_table(args.corner_harm_dir, args.output_dir)
    write_provenance(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

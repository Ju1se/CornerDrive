#!/usr/bin/env python3
"""Export dev/test calibration split tables for real-gradient threshold tuning.

The threshold sweep artifacts separate calibration seeds from a holdout seed.
This exporter normalizes those layouts into reviewer-facing tables that make
the tuning surface explicit. Deprecated router profiles are intentionally
excluded from this reproduction helper.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEV_SEEDS = {20260507, 20260508}
TEST_SEEDS = {20260509}

PROFILE_SPECS = [
    {
        "profile": "current",
        "label": "Initial profile",
        "dev_dir": "calibration_current",
        "test_dir": "holdout_current",
    },
    {
        "profile": "l2_strict",
        "label": "L2 strict only",
        "dev_dir": "calibration_l2_strict",
        "test_dir": "holdout_l2_strict",
    },
    {
        "profile": "recheck50",
        "label": "Recheck 0.50",
        "dev_dir": "calibration_recheck50",
        "test_dir": "holdout_recheck50",
    },
    {
        "profile": "l1_aggressive",
        "label": "L1 aggressive",
        "dev_dir": "calibration_l1_aggressive",
        "test_dir": "holdout_l1_aggressive",
    },
    {
        "profile": "balanced_strict",
        "label": "Balanced strict",
        "dev_dir": "calibration_balanced_strict",
        "test_dir": "holdout_balanced_strict",
    },
]

METRICS = {
    "main_accuracy": "main_accuracy_avg",
    "corner_accuracy": "corner_accuracy_avg",
    "fraud_survival": "fraud_survival_rate_avg",
    "rarity_retention": "rarity_retention_rate_avg",
    "review_rate": "l1_review_rate_avg",
    "selected_total": "selected_total_avg",
}

COUNTS = {
    "client_observations": "client_observations",
    "fraud_observations": "fraud_observations",
    "rarity_observations": "rarity_observations",
}

PARAMS = {
    "theta_tol": "policy_theta_tol",
    "theta_rare": "policy_theta_rare",
    "theta_rarity_main_tol": "policy_theta_rarity_main_tol",
    "cosine_filter_threshold": "policy_cosine_filter_threshold",
    "recheck_probability": "policy_recheck_probability",
    "l1_mode": "cornerdrive_l1_mode",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
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


def as_float(row: dict[str, str], key: str) -> float:
    raw = row.get(key, "")
    return float(raw) if raw not in {"", None} else 0.0


def as_int(row: dict[str, str], key: str) -> int:
    raw = row.get(key, "")
    return int(float(raw)) if raw not in {"", None} else 0


def filter_seeds(rows: list[dict[str, str]], seeds: set[int]) -> list[dict[str, str]]:
    return [row for row in rows if as_int(row, "seed") in seeds]


def metric_mean(rows: list[dict[str, str]], key: str) -> float | str:
    values = [as_float(row, key) for row in rows if row.get(key) not in {"", None}]
    return round(mean(values), 6) if values else ""


def metric_sum(rows: list[dict[str, str]], key: str) -> int:
    return sum(as_int(row, key) for row in rows)


def seed_list(rows: list[dict[str, str]]) -> str:
    seeds = sorted({as_int(row, "seed") for row in rows if row.get("seed")})
    return ";".join(str(seed) for seed in seeds)


def summarize_split(
    profile: str,
    label: str,
    split: str,
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "profile": profile,
        "label": label,
        "split": split,
        "seeds": seed_list(rows),
        "runs": len(rows),
    }
    for out_key, source_key in COUNTS.items():
        record[out_key] = metric_sum(rows, source_key)
    for out_key, source_key in METRICS.items():
        record[out_key] = metric_mean(rows, source_key)
    if rows:
        first = rows[0]
        for out_key, source_key in PARAMS.items():
            record[out_key] = first.get(source_key, "")
    return record


def rows_for_spec(root: Path, spec: dict[str, str], split: str) -> list[dict[str, str]]:
    combined_dir = spec.get("combined_dir")
    if combined_dir:
        rows = read_csv(root / combined_dir / "real_gradient_reliability_runs.csv")
        return filter_seeds(rows, DEV_SEEDS if split == "dev" else TEST_SEEDS)
    split_dir = spec["dev_dir"] if split == "dev" else spec["test_dir"]
    return read_csv(root / split_dir / "real_gradient_reliability_runs.csv")


def build_long_rows(root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for spec in PROFILE_SPECS:
        for split in ("dev", "test"):
            rows = rows_for_spec(root, spec, split)
            out.append(summarize_split(spec["profile"], spec["label"], split, rows))
    return out


def build_wide_rows(long_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in long_rows:
        grouped.setdefault(str(row["profile"]), {})[str(row["split"])] = row

    out: list[dict[str, Any]] = []
    for profile, splits in grouped.items():
        dev = splits.get("dev", {})
        test = splits.get("test", {})
        record: dict[str, Any] = {
            "profile": profile,
            "label": dev.get("label") or test.get("label"),
            "dev_seeds": dev.get("seeds", ""),
            "test_seeds": test.get("seeds", ""),
            "dev_runs": dev.get("runs", 0),
            "test_runs": test.get("runs", 0),
            "dev_fraud_survival": dev.get("fraud_survival", ""),
            "test_fraud_survival": test.get("fraud_survival", ""),
            "test_rarity_retention": test.get("rarity_retention", ""),
            "test_corner_accuracy": test.get("corner_accuracy", ""),
            "test_main_accuracy": test.get("main_accuracy", ""),
            "test_review_rate": test.get("review_rate", ""),
            "theta_tol": test.get("theta_tol") or dev.get("theta_tol", ""),
            "theta_rare": test.get("theta_rare") or dev.get("theta_rare", ""),
            "cosine_filter_threshold": test.get("cosine_filter_threshold") or dev.get("cosine_filter_threshold", ""),
            "recheck_probability": test.get("recheck_probability") or dev.get("recheck_probability", ""),
            "l1_mode": test.get("l1_mode") or dev.get("l1_mode", ""),
        }
        out.append(record)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export reviewer-facing dev/test calibration split tables."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "real_gradient_threshold_sweep",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "real_gradient_calibration_split",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    long_rows = build_long_rows(args.input_root)
    wide_rows = build_wide_rows(long_rows)
    write_csv(args.output_dir / "real_gradient_calibration_split_long.csv", long_rows)
    write_csv(args.output_dir / "real_gradient_calibration_split_summary.csv", wide_rows)
    print(f"Wrote calibration split artifacts to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

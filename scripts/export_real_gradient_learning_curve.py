#!/usr/bin/env python3
"""Export end-to-end real-gradient FL learning curves.

The existing real-gradient benchmark reports per-round records, but the thesis
table uses aggregate metrics. This exporter keeps the same leakage-safe data
surfaces and writes a curve-oriented artifact: per-round metrics, per-round
means across runs, a final-round summary table, and a lightweight SVG figure.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"

for candidate in (PROJECT_ROOT, BACKEND_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from common.schemas import DEFAULT_POLICY  # noqa: E402
from policy_agent.analysis.real_gradient_benchmark import (  # noqa: E402
    RealGradientBenchmarkConfig,
    make_real_gradient_calibrated_policy,
    run_real_gradient_benchmark,
)


CALIBRATED_POLICY_PROFILES = {
    "real_gradient_calibrated",
}


DEFAULT_SOURCES = "mnist,fashionmnist,femnist"
DEFAULT_SEEDS = "20260507,20260508,20260509,20260510,20260511"
DEFAULT_METHODS = "fedavg,krum,fltrust,zenopp,cornerdrive"
METHOD_LABELS = {
    "fedavg": "FedAvg",
    "geomed": "GeoMed",
    "krum": "Multi-Krum",
    "fltrust": "FLTrust",
    "zeno": "Zeno",
    "zenopp": "Zeno++",
    "cornerdrive": "CornerDrive",
}
METHOD_COLORS = {
    "FedAvg": "#6b7280",
    "GeoMed": "#8b5cf6",
    "Multi-Krum": "#2563eb",
    "FLTrust": "#059669",
    "Zeno": "#d97706",
    "Zeno++": "#dc2626",
    "CornerDrive": "#111827",
}
METRICS = (
    "main_accuracy",
    "corner_accuracy",
    "fraud_survival_rate",
    "rarity_retention_rate",
    "effective_fraud_mass_survival",
    "l2_fraud_as_rarity_accept_rate",
    "l2_conflict_update_reject_rate",
    "selected_total",
    "l1_review_rate",
)


def parse_csv_values(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_seed_values(raw: str) -> list[int]:
    return [int(part) for part in parse_csv_values(raw)]


def source_slug(source: str) -> str:
    return source.lower().replace("/", "_").replace("leaf_", "")


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


def mean_or_blank(values: list[float]) -> float | str:
    return mean(values) if values else ""


def stdev_or_zero(values: list[float]) -> float | str:
    return stdev(values) if len(values) > 1 else 0.0 if values else ""


def round_float(value: Any, digits: int = 6) -> Any:
    return round(float(value), digits) if value not in {"", None} else ""


def profile_l1_defaults(policy_profile: str) -> dict[str, float | str]:
    if policy_profile in CALIBRATED_POLICY_PROFILES:
        return {
            "cornerdrive_l1_mode": "dual_proxy_budgeted",
            "cornerdrive_l1_cos_weight": 0.35,
            "cornerdrive_l1_norm_weight": 0.20,
            "cornerdrive_l1_sign_weight": 0.15,
            "cornerdrive_l1_norm_mad_threshold": 1.5,
            "cornerdrive_l1_sign_threshold": 0.40,
            "cornerdrive_l1_sign_topk_ratio": 0.10,
            "cornerdrive_l1_queue_budget_ratio": 0.80,
            "cornerdrive_l1_random_recheck_ratio": 0.05,
        }
    return {
        "cornerdrive_l1_mode": "cosine_recheck",
        "cornerdrive_l1_cos_weight": 0.35,
        "cornerdrive_l1_norm_weight": 0.20,
        "cornerdrive_l1_sign_weight": 0.15,
        "cornerdrive_l1_norm_mad_threshold": 3.0,
        "cornerdrive_l1_sign_threshold": 0.65,
        "cornerdrive_l1_sign_topk_ratio": 0.10,
        "cornerdrive_l1_queue_budget_ratio": 0.35,
        "cornerdrive_l1_random_recheck_ratio": 0.05,
    }


def build_policy(args: argparse.Namespace):
    if args.policy_profile in CALIBRATED_POLICY_PROFILES:
        policy = make_real_gradient_calibrated_policy()
    else:
        policy = DEFAULT_POLICY
    updates = {
        key: value
        for key, value in {
            "theta_tol": args.theta_tol,
            "theta_rare": args.theta_rare,
            "theta_rarity_main_tol": args.theta_rarity_main_tol,
            "cosine_filter_threshold": args.cosine_filter_threshold,
            "recheck_probability": args.recheck_probability,
        }.items()
        if value is not None
    }
    return policy.model_copy(update=updates) if updates else policy


def build_config(args: argparse.Namespace, source: str, seed: int) -> RealGradientBenchmarkConfig:
    l1_defaults = profile_l1_defaults(args.policy_profile)
    return RealGradientBenchmarkConfig(
        source=source,
        leaf_data_dir=args.leaf_data_dir,
        data_dir=args.data_dir,
        download=args.download,
        max_clients=args.max_clients,
        min_samples_per_client=args.min_samples_per_client,
        max_samples_per_client=args.max_samples_per_client,
        clients_per_round=args.clients_per_round,
        rounds=args.rounds,
        seed=seed,
        pretrain_steps=args.pretrain_steps,
        local_batch_size=args.local_batch_size,
        reference_split_fraction=args.reference_split_fraction,
        max_reference_samples=args.max_reference_samples,
        max_evaluation_samples=args.max_evaluation_samples,
        attack_fraction=args.attack_fraction,
        corner_harm_fraction=args.corner_harm_fraction,
        noise_fraction=args.noise_fraction,
        rarity_label_fraction_threshold=args.rarity_label_fraction_threshold,
        sign_flip_scale=args.sign_flip_scale,
        corner_harm_scale=args.corner_harm_scale,
        zeno_score_penalty=args.zeno_score_penalty,
        zenopp_score_temperature=args.zenopp_score_temperature,
        cornerdrive_l1_mode=str(args.cornerdrive_l1_mode or l1_defaults["cornerdrive_l1_mode"]),
        cornerdrive_l1_cos_weight=float(args.cornerdrive_l1_cos_weight if args.cornerdrive_l1_cos_weight is not None else l1_defaults["cornerdrive_l1_cos_weight"]),
        cornerdrive_l1_norm_weight=float(args.cornerdrive_l1_norm_weight if args.cornerdrive_l1_norm_weight is not None else l1_defaults["cornerdrive_l1_norm_weight"]),
        cornerdrive_l1_sign_weight=float(args.cornerdrive_l1_sign_weight if args.cornerdrive_l1_sign_weight is not None else l1_defaults["cornerdrive_l1_sign_weight"]),
        cornerdrive_l1_norm_mad_threshold=float(args.cornerdrive_l1_norm_mad_threshold if args.cornerdrive_l1_norm_mad_threshold is not None else l1_defaults["cornerdrive_l1_norm_mad_threshold"]),
        cornerdrive_l1_sign_threshold=float(args.cornerdrive_l1_sign_threshold if args.cornerdrive_l1_sign_threshold is not None else l1_defaults["cornerdrive_l1_sign_threshold"]),
        cornerdrive_l1_sign_topk_ratio=float(args.cornerdrive_l1_sign_topk_ratio if args.cornerdrive_l1_sign_topk_ratio is not None else l1_defaults["cornerdrive_l1_sign_topk_ratio"]),
        cornerdrive_l1_queue_budget_ratio=float(args.cornerdrive_l1_queue_budget_ratio if args.cornerdrive_l1_queue_budget_ratio is not None else l1_defaults["cornerdrive_l1_queue_budget_ratio"]),
        cornerdrive_l1_random_recheck_ratio=float(args.cornerdrive_l1_random_recheck_ratio if args.cornerdrive_l1_random_recheck_ratio is not None else l1_defaults["cornerdrive_l1_random_recheck_ratio"]),
    )


def summarize_by_round(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), int(row["round"]))].append(row)

    out: list[dict[str, Any]] = []
    for (method, round_index), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        record: dict[str, Any] = {"method": method, "round": round_index, "runs": len(group)}
        for metric in METRICS:
            values = [
                float(row[metric])
                for row in group
                if row.get(metric) not in {"", None}
            ]
            record[f"{metric}_mean"] = round_float(mean_or_blank(values))
            record[f"{metric}_std"] = round_float(stdev_or_zero(values))
        out.append(record)
    return out


def final_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    per_run_final: list[dict[str, Any]] = []
    grouped_run: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_run[(str(row["method"]), str(row["source"]), int(row["seed"]))].append(row)
    for (_method, _source, _seed), group in grouped_run.items():
        per_run_final.append(max(group, key=lambda row: int(row["round"])))

    grouped_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_run_final:
        grouped_method[str(row["method"])].append(row)

    out: list[dict[str, Any]] = []
    for method, group in sorted(grouped_method.items()):
        record: dict[str, Any] = {"method": method, "runs": len(group)}
        for metric in METRICS:
            values = [
                float(row[metric])
                for row in group
                if row.get(metric) not in {"", None}
            ]
            record[f"final_{metric}_mean"] = round_float(mean_or_blank(values))
            record[f"final_{metric}_std"] = round_float(stdev_or_zero(values))
        out.append(record)
    return out


def svg_line_chart(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    methods = sorted({str(row["method"]) for row in rows})
    max_round = max(int(row["round"]) for row in rows)
    width = 980
    height = 460
    margin_left = 70
    margin_right = 180
    panel_gap = 48
    panel_width = (width - margin_left - margin_right - panel_gap) / 2
    panel_height = 310
    top = 72
    metrics = [
        ("main_accuracy_mean", "Main accuracy"),
        ("corner_accuracy_mean", "Corner accuracy"),
    ]

    def point(row: dict[str, Any], metric: str, panel_idx: int) -> tuple[float, float]:
        x0 = margin_left + panel_idx * (panel_width + panel_gap)
        x = x0 + (int(row["round"]) / max(max_round, 1)) * panel_width
        y = top + panel_height - float(row[metric]) * panel_height
        return x, y

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="70" y="36" font-family="Arial" font-size="20" font-weight="700">End-to-end real-gradient FL learning curves</text>',
        '<text x="70" y="58" font-family="Arial" font-size="12" fill="#4b5563">Mean over selected sources/seeds; higher is better.</text>',
    ]
    for panel_idx, (metric, title) in enumerate(metrics):
        x0 = margin_left + panel_idx * (panel_width + panel_gap)
        y0 = top
        lines.append(f'<text x="{x0}" y="{y0 - 14}" font-family="Arial" font-size="14" font-weight="700">{title}</text>')
        lines.append(f'<line x1="{x0}" y1="{y0 + panel_height}" x2="{x0 + panel_width}" y2="{y0 + panel_height}" stroke="#111827" stroke-width="1"/>')
        lines.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + panel_height}" stroke="#111827" stroke-width="1"/>')
        for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
            y = y0 + panel_height - tick * panel_height
            lines.append(f'<line x1="{x0 - 4}" y1="{y}" x2="{x0 + panel_width}" y2="{y}" stroke="#e5e7eb" stroke-width="1"/>')
            lines.append(f'<text x="{x0 - 10}" y="{y + 4}" text-anchor="end" font-family="Arial" font-size="10" fill="#4b5563">{tick:.2f}</text>')
        for tick_round in [0, max_round // 2, max_round]:
            x = x0 + (tick_round / max(max_round, 1)) * panel_width
            lines.append(f'<text x="{x}" y="{y0 + panel_height + 20}" text-anchor="middle" font-family="Arial" font-size="10" fill="#4b5563">{tick_round}</text>')

        for method in methods:
            series = [
                row for row in rows
                if row["method"] == method and row.get(metric) not in {"", None}
            ]
            series.sort(key=lambda row: int(row["round"]))
            if not series:
                continue
            points = " ".join(f"{x:.1f},{y:.1f}" for x, y in (point(row, metric, panel_idx) for row in series))
            color = METHOD_COLORS.get(method, "#374151")
            lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')

    legend_x = width - margin_right + 35
    legend_y = top
    for idx, method in enumerate(methods):
        y = legend_y + idx * 22
        color = METHOD_COLORS.get(method, "#374151")
        lines.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 22}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{legend_x + 30}" y="{y + 4}" font-family="Arial" font-size="12" fill="#111827">{html.escape(method)}</text>')
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export real-gradient FL learning curves.")
    parser.add_argument("--sources", default=DEFAULT_SOURCES)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--leaf-data-dir", default="data/real/femnist")
    parser.add_argument("--data-dir", default="data/real")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-clients", type=int, default=120)
    parser.add_argument("--min-samples-per-client", type=int, default=8)
    parser.add_argument("--max-samples-per-client", type=int, default=48)
    parser.add_argument("--clients-per-round", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--pretrain-steps", type=int, default=50)
    parser.add_argument("--local-batch-size", type=int, default=16)
    parser.add_argument("--reference-split-fraction", type=float, default=0.50)
    parser.add_argument("--max-reference-samples", type=int, default=4096)
    parser.add_argument("--max-evaluation-samples", type=int, default=4096)
    parser.add_argument("--attack-fraction", type=float, default=0.20)
    parser.add_argument("--corner-harm-fraction", type=float, default=0.05)
    parser.add_argument("--noise-fraction", type=float, default=0.05)
    parser.add_argument("--rarity-label-fraction-threshold", type=float, default=0.30)
    parser.add_argument("--sign-flip-scale", type=float, default=3.0)
    parser.add_argument("--corner-harm-scale", type=float, default=2.0)
    parser.add_argument("--zeno-score-penalty", type=float, default=1e-4)
    parser.add_argument("--zenopp-score-temperature", type=float, default=0.05)
    parser.add_argument(
        "--policy-profile",
        choices=["default", "real_gradient_calibrated"],
        default="real_gradient_calibrated",
    )
    parser.add_argument("--theta-tol", type=float, default=None)
    parser.add_argument("--theta-rare", type=float, default=None)
    parser.add_argument("--theta-rarity-main-tol", type=float, default=None)
    parser.add_argument("--cosine-filter-threshold", type=float, default=None)
    parser.add_argument("--recheck-probability", type=float, default=None)
    parser.add_argument("--cornerdrive-l1-mode", default=None)
    parser.add_argument("--cornerdrive-l1-cos-weight", type=float, default=None)
    parser.add_argument("--cornerdrive-l1-norm-weight", type=float, default=None)
    parser.add_argument("--cornerdrive-l1-sign-weight", type=float, default=None)
    parser.add_argument("--cornerdrive-l1-norm-mad-threshold", type=float, default=None)
    parser.add_argument("--cornerdrive-l1-sign-threshold", type=float, default=None)
    parser.add_argument("--cornerdrive-l1-sign-topk-ratio", type=float, default=None)
    parser.add_argument("--cornerdrive-l1-queue-budget-ratio", type=float, default=None)
    parser.add_argument("--cornerdrive-l1-random-recheck-ratio", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "real_gradient_learning_curve")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Keep per-round benchmark logs enabled. By default only exporter progress is printed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.verbose:
        logging.disable(logging.CRITICAL)
    sources = parse_csv_values(args.sources)
    seeds = parse_seed_values(args.seeds)
    method_ids = parse_csv_values(args.methods)
    method_labels = {METHOD_LABELS.get(method_id, method_id) for method_id in method_ids}
    policy = build_policy(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for source in sources:
        for seed in seeds:
            config = build_config(args, source, seed)
            result = run_real_gradient_benchmark(config, policy=policy)
            manifest.append({
                "source": source,
                "seed": seed,
                "dataset": result["dataset"],
                "config": result["config"],
            })
            for method_id, payload in result["methods"].items():
                label = payload["label"]
                if method_id not in method_ids and label not in method_labels:
                    continue
                for record in payload["round_records"]:
                    raw_rows.append({
                        "source": source_slug(source),
                        "seed": seed,
                        "method_id": method_id,
                        "method": label,
                        "round": int(record["round"]) + 1,
                        "main_accuracy": record["main_accuracy"],
                        "corner_accuracy": record["corner_accuracy"],
                        "fraud_survival_rate": record["fraud_survival_rate"],
                        "rarity_retention_rate": record["rarity_retention_rate"],
                        "effective_fraud_mass_survival": record["effective_fraud_mass_survival"],
                        "l2_fraud_as_rarity_accept_rate": record.get("l2_fraud_as_rarity_accept_rate", ""),
                        "l2_conflict_update_reject_rate": record.get("l2_conflict_update_reject_rate", ""),
                        "selected_total": record["selected_total"],
                        "l1_review_rate": record.get("l1_review_rate", ""),
                    })
            cornerdrive = result["methods"]["cornerdrive"]["round_records"][-1]
            print(
                f"{source} seed={seed}: final CornerDrive "
                f"main={cornerdrive['main_accuracy']:.4f}, "
                f"corner={cornerdrive['corner_accuracy']:.4f}, "
                f"fraud={cornerdrive['fraud_survival_rate']:.4f}",
                flush=True,
            )

    by_round = summarize_by_round(raw_rows)
    summary = final_summary(raw_rows)
    write_csv(args.output_dir / "real_gradient_learning_curve_rounds.csv", raw_rows)
    write_csv(args.output_dir / "real_gradient_learning_curve_by_round.csv", by_round)
    write_csv(args.output_dir / "real_gradient_learning_curve_final_summary.csv", summary)
    svg_line_chart(args.output_dir / "real_gradient_learning_curve.svg", by_round)
    (args.output_dir / "real_gradient_learning_curve_config.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "sources": sources,
                "seeds": seeds,
                "methods": method_ids,
                "runs": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote learning-curve artifacts to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export audit cost vs. safety/utility frontier artifacts.

This script turns the existing V2.5 recheck sweep table into a compact
paper-facing frontier table and a dependency-free SVG figure.
"""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
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


def build_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    frontier: list[dict[str, Any]] = []
    for row in rows:
        audit_queue = as_float(row, "audit_queue_ratio_mean")
        corner_harm = as_float(row, "corner_harm_survival_mean")
        corner_acc = as_float(row, "corner_accuracy_mean")
        main_acc = as_float(row, "main_accuracy_mean")
        rarity_recall = as_float(row, "rarity_recall_mean")
        frontier.append({
            "p_recheck": as_float(row, "p_recheck"),
            "l1_review_rate": round(audit_queue, 6),
            "l1_review_percent": round(100.0 * audit_queue, 2),
            "corner_harm_survival": round(corner_harm, 6),
            "corner_harm_survival_percent": round(100.0 * corner_harm, 2),
            "corner_accuracy": round(corner_acc, 6),
            "corner_accuracy_percent": round(100.0 * corner_acc, 2),
            "main_accuracy": round(main_acc, 6),
            "main_accuracy_percent": round(100.0 * main_acc, 2),
            "rarity_recall": round(rarity_recall, 6),
            "rarity_recall_percent": round(100.0 * rarity_recall, 2),
        })
    return sorted(frontier, key=lambda item: item["p_recheck"])


def svg_frontier(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    width = 900
    height = 420
    left = 76
    right = 34
    top = 72
    bottom = 62
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_values = [float(row["l1_review_percent"]) for row in rows]
    x_min = min(x_values)
    x_max = max(x_values)
    x_span = max(x_max - x_min, 1e-9)

    def x_pos(x_value: float) -> float:
        return left + ((x_value - x_min) / x_span) * plot_width

    def y_pos(percent_value: float) -> float:
        return top + plot_height - (percent_value / 100.0) * plot_height

    def points(metric: str) -> str:
        return " ".join(
            f"{x_pos(float(row['l1_review_percent'])):.1f},{y_pos(float(row[metric])):.1f}"
            for row in rows
        )

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="72" y="34" font-family="Arial" font-size="20" font-weight="700">Audit cost vs. safety/utility frontier</text>',
        '<text x="72" y="56" font-family="Arial" font-size="12" fill="#4b5563">Existing recheck sweep; lower corner-harm survival and higher corner accuracy are better.</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111827" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111827" stroke-width="1"/>',
    ]
    for tick in [0, 25, 50, 75, 100]:
        y = y_pos(tick)
        lines.append(f'<line x1="{left - 4}" y1="{y}" x2="{left + plot_width}" y2="{y}" stroke="#e5e7eb" stroke-width="1"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4}" text-anchor="end" font-family="Arial" font-size="10" fill="#4b5563">{tick}</text>')
    for row in rows:
        x = x_pos(float(row["l1_review_percent"]))
        lines.append(f'<line x1="{x}" y1="{top + plot_height}" x2="{x}" y2="{top + plot_height + 5}" stroke="#111827" stroke-width="1"/>')
        lines.append(f'<text x="{x}" y="{top + plot_height + 20}" text-anchor="middle" font-family="Arial" font-size="10" fill="#4b5563">{float(row["l1_review_percent"]):.1f}</text>')
    lines.extend([
        f'<polyline points="{points("corner_harm_survival_percent")}" fill="none" stroke="#dc2626" stroke-width="3"/>',
        f'<polyline points="{points("corner_accuracy_percent")}" fill="none" stroke="#2563eb" stroke-width="3"/>',
    ])
    for row in rows:
        x = x_pos(float(row["l1_review_percent"]))
        y1 = y_pos(float(row["corner_harm_survival_percent"]))
        y2 = y_pos(float(row["corner_accuracy_percent"]))
        label = html.escape(f"p={float(row['p_recheck']):.2f}")
        lines.append(f'<circle cx="{x:.1f}" cy="{y1:.1f}" r="4" fill="#dc2626"/>')
        lines.append(f'<circle cx="{x:.1f}" cy="{y2:.1f}" r="4" fill="#2563eb"/>')
        lines.append(f'<text x="{x + 6:.1f}" y="{min(y1, y2) - 6:.1f}" font-family="Arial" font-size="10" fill="#374151">{label}</text>')
    lines.extend([
        f'<text x="{left + plot_width / 2}" y="{height - 18}" text-anchor="middle" font-family="Arial" font-size="12" fill="#111827">L1 review / audit queue rate (%)</text>',
        f'<text x="18" y="{top + plot_height / 2}" text-anchor="middle" transform="rotate(-90 18 {top + plot_height / 2})" font-family="Arial" font-size="12" fill="#111827">Metric (%)</text>',
        f'<line x1="{width - 250}" y1="{top + 8}" x2="{width - 228}" y2="{top + 8}" stroke="#dc2626" stroke-width="3"/>',
        f'<text x="{width - 220}" y="{top + 12}" font-family="Arial" font-size="12">Corner-harm survival</text>',
        f'<line x1="{width - 250}" y1="{top + 30}" x2="{width - 228}" y2="{top + 30}" stroke="#2563eb" stroke-width="3"/>',
        f'<text x="{width - 220}" y="{top + 34}" font-family="Arial" font-size="12">Corner accuracy</text>',
        "</svg>",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export audit cost frontier from the recheck sweep.")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "results" / "audit_reproduction" / "v25_artifacts_b24" / "v25_recheck_sweep_table.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "cost_performance_frontier")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows(read_csv(args.input))
    write_csv(args.output_dir / "cost_performance_frontier.csv", rows)
    svg_frontier(args.output_dir / "audit_cost_frontier.svg", rows)
    print(f"Wrote cost frontier artifacts to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

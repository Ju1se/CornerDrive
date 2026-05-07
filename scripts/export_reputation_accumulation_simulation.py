#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

for candidate in (PROJECT_ROOT, SCRIPTS_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from export_thesis_artifacts import write_csv, write_json  # noqa: E402


DEFAULT_SCENARIOS = "0.10:5,0.10:10,0.10:20,0.20:10,0.30:10"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export L4 reputation accumulation design-validation table."
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=DEFAULT_SCENARIOS,
        help="Comma-separated p:T entries, e.g. 0.10:5,0.20:10.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "reputation_accumulation",
    )
    return parser.parse_args()


def parse_scenarios(raw: str) -> list[tuple[float, int]]:
    scenarios: list[tuple[float, int]] = []
    for part in raw.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        p_raw, rounds_raw = stripped.split(":", maxsplit=1)
        scenarios.append((float(p_raw), int(rounds_raw)))
    return scenarios or [(0.10, 10)]


def accumulated_probability(p: float, rounds: int) -> float:
    return 1.0 - (1.0 - p) ** rounds


def build_rows(scenarios: list[tuple[float, int]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p, rounds in scenarios:
        probability = accumulated_probability(p, rounds)
        rows.append({
            "per_round_catch_probability": p,
            "rounds_t": rounds,
            "accumulated_catch_probability": probability,
            "accumulated_catch_probability_percent": probability * 100.0,
            "miss_probability": 1.0 - probability,
            "formula": "1 - (1 - p)^T",
            "interpretation": "design_validation_not_full_l4_evaluation",
        })
    return rows


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = parse_scenarios(args.scenarios)
    rows = build_rows(scenarios)
    write_csv(
        output_dir / "reputation_accumulation_simulation.csv",
        [
            "per_round_catch_probability",
            "rounds_t",
            "accumulated_catch_probability",
            "accumulated_catch_probability_percent",
            "miss_probability",
            "formula",
            "interpretation",
        ],
        rows,
    )
    write_json(
        output_dir / "reputation_accumulation_config.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "scenarios": [{"p": p, "rounds_t": rounds} for p, rounds in scenarios],
            "formula": "1 - (1 - p)^T",
            "scope": "simulation/design validation only; not a full smart-contract or L4 evaluation",
        },
    )
    print(f"Wrote reputation accumulation simulation to {output_dir}")


if __name__ == "__main__":
    main()

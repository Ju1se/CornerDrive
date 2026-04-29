#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"

for candidate in (PROJECT_ROOT, BACKEND_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from common.schemas import Policy  # noqa: E402
from policy_agent.analysis.baselines import build_baseline_analysis  # noqa: E402


def fetch_live_policy(timeout: float = 2.0) -> Policy | None:
    url = "http://127.0.0.1:8083/api/v1/policy/current"
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        return Policy.model_validate(response.json())
    except Exception:
        return None


def choose_policy_source(source: str) -> tuple[Policy | None, str]:
    if source == "default":
        return None, "default_policy"
    if source == "live":
        live_policy = fetch_live_policy()
        if live_policy is None:
            raise RuntimeError("Could not fetch live policy from http://127.0.0.1:8083")
        return live_policy, "live_policy"

    live_policy = fetch_live_policy()
    if live_policy is not None:
        return live_policy, "live_policy"
    return None, "default_policy"


def format_pct(value: float) -> str:
    return f"{value * 100:6.2f}%"


def print_summary(payload: dict) -> None:
    baselines = payload["baselines"]
    header = (
        f"{'Strategy':<24}"
        f"{'Main':>10}"
        f"{'Corner':>10}"
        f"{'FalseSlash':>12}"
        f"{'RarityKeep':>12}"
        f"{'FraudRec':>10}"
        f"{'RarityRec':>11}"
    )
    print(header)
    print("-" * len(header))

    for baseline in baselines:
        summary = baseline["summary"]
        print(
            f"{baseline['label']:<24}"
            f"{format_pct(summary['main_accuracy_avg']):>10}"
            f"{format_pct(summary['corner_accuracy_avg']):>10}"
            f"{format_pct(summary['false_slash_estimate_avg']):>12}"
            f"{format_pct(summary['rarity_retention_rate_avg']):>12}"
            f"{format_pct(summary['fraud_recall']):>10}"
            f"{format_pct(summary['rarity_recall']):>11}"
        )


def print_comparison(payload: dict) -> None:
    indexed = {entry["id"]: entry for entry in payload["baselines"]}
    fedavg = indexed.get("fedavg")
    adaptive = indexed.get("adaptive")
    if fedavg is None or adaptive is None:
        return

    fedavg_summary = fedavg["summary"]
    adaptive_summary = adaptive["summary"]
    print()
    print("FedAvg vs Full FLPG")
    print(
        "  Main accuracy delta: "
        f"{(adaptive_summary['main_accuracy_avg'] - fedavg_summary['main_accuracy_avg']) * 100:+.2f} pp"
    )
    print(
        "  Corner accuracy delta: "
        f"{(adaptive_summary['corner_accuracy_avg'] - fedavg_summary['corner_accuracy_avg']) * 100:+.2f} pp"
    )
    print(
        "  False-slash delta: "
        f"{(adaptive_summary['false_slash_estimate_avg'] - fedavg_summary['false_slash_estimate_avg']) * 100:+.2f} pp"
    )


def print_rounds(payload: dict) -> None:
    print()
    for baseline in payload["baselines"]:
        print(f"{baseline['label']}")
        for entry in baseline["rounds"]:
            print(
                f"  R{entry['round_id']:>2} {entry['phase']:<16}"
                f" main={entry['main_accuracy'] * 100:6.2f}%"
                f" corner={entry['corner_accuracy'] * 100:6.2f}%"
                f" retained={entry['rarity_retention_rate'] * 100:6.2f}%"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FedAvg baseline evaluation on the same simulated gradient rounds as FLPG."
    )
    parser.add_argument("--rounds", type=int, default=12, help="Number of simulated rounds.")
    parser.add_argument(
        "--policy-source",
        choices=("auto", "default", "live"),
        default="auto",
        help="Use the live policy if available, otherwise fall back to the default policy.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write the raw JSON payload.",
    )
    parser.add_argument(
        "--show-rounds",
        action="store_true",
        help="Print round-by-round accuracy traces for each baseline.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    current_policy, resolved_source = choose_policy_source(args.policy_source)
    payload = asyncio.run(build_baseline_analysis(current_policy, rounds=args.rounds))

    print(f"Policy source: {resolved_source}")
    print(f"Scenario round: {payload['scenario_policy_round']}")
    print(f"Rounds: {payload['classification_rounds']}")
    print()
    print_summary(payload)
    print_comparison(payload)

    if args.show_rounds:
        print_rounds(payload)

    if args.json_out is not None:
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print()
        print(f"Saved raw payload to {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

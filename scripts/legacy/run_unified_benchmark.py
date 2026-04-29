#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"

for candidate in (PROJECT_ROOT, BACKEND_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from policy_agent.analysis.unified_benchmark import run_unified_benchmark  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the unified synthetic benchmark with a shared evaluation pipeline."
    )
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--cycle-rounds", type=int, default=12)
    parser.add_argument("--pretrain-epochs", type=int, default=7)
    parser.add_argument("--pretrain-batch-size", type=int, default=64)
    parser.add_argument("--pretrain-learning-rate", type=float, default=1e-3)
    parser.add_argument("--krum-byzantine-budget", type=int, default=10)
    parser.add_argument("--full-out", type=Path, default=None)
    parser.add_argument("--summary-out", type=Path, default=None)
    return parser.parse_args()


def build_summary(payload: dict) -> dict:
    return {
        "generated_at": payload["generated_at"],
        "config": payload["config"],
        "comparison_vs_fedavg": payload["comparison_vs_fedavg"],
        "methods": {
            method_id: {
                "label": method_payload["label"],
                "summary": method_payload["summary"],
                "phase_breakdown": method_payload["phase_breakdown"],
                "classification_summary": method_payload["classification_summary"],
            }
            for method_id, method_payload in payload["methods"].items()
        },
    }


def print_summary(summary: dict) -> None:
    config = summary["config"]
    print(f"Rounds: {config['rounds']}")
    print(f"Cycle rounds: {config['cycle_rounds']}")
    print(f"Clients/round: {config['clients_per_round']}")
    print(f"Vehicle pool: {config['vehicle_pool_size']}")
    checkpoint = config["initial_checkpoint"]
    print(
        "Initial checkpoint: "
        f"main={checkpoint['initial_main_accuracy'] * 100:.2f}% "
        f"corner={checkpoint['initial_corner_accuracy'] * 100:.2f}%"
    )
    first_cycle_counts = config.get("first_cycle_ground_truth_counts", config.get("first_cycle_preflight_counts", {}))
    print(f"First-cycle ground-truth counts: {first_cycle_counts}")
    print()

    header = (
        f"{'Method':<22}"
        f"{'Main':>10}"
        f"{'Corner':>10}"
        f"{'FraudSurv':>12}"
        f"{'FraudRec':>10}"
        f"{'RarityRec':>11}"
    )
    print(header)
    print("-" * len(header))

    for method_id, method_payload in summary["methods"].items():
        stats = method_payload["summary"]
        print(
            f"{method_payload['label']:<22}"
            f"{stats['main_accuracy_avg'] * 100:10.2f}%"
            f"{stats['corner_accuracy_avg'] * 100:10.2f}%"
            f"{stats['fraud_survival_rate_overall'] * 100:12.2f}%"
            f"{stats.get('fraud_recall', 0.0) * 100:10.2f}%"
            f"{stats.get('rarity_recall', 0.0) * 100:11.2f}%"
        )

    print()
    for method_id, delta in summary["comparison_vs_fedavg"].items():
        print(
            f"{summary['methods'][method_id]['label']}: "
            f"main {delta['main_accuracy_delta_pp']:+.2f}pp, "
            f"corner {delta['corner_accuracy_delta_pp']:+.2f}pp, "
            f"fraud survival {delta['fraud_survival_delta_pp']:+.2f}pp"
        )


def main() -> int:
    args = parse_args()
    payload = run_unified_benchmark(
        rounds=args.rounds,
        cycle_rounds=args.cycle_rounds,
        pretrain_epochs=args.pretrain_epochs,
        pretrain_batch_size=args.pretrain_batch_size,
        pretrain_learning_rate=args.pretrain_learning_rate,
        krum_byzantine_budget=args.krum_byzantine_budget,
    )
    summary = build_summary(payload)
    print_summary(summary)

    if args.full_out is not None:
        args.full_out.parent.mkdir(parents=True, exist_ok=True)
        args.full_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved full payload to {args.full_out}")

    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved summary payload to {args.summary_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Compare ALG synthetic gradients with real client-SGD gradients.

This exporter operationalizes the dissertation's first future-work item: before
claiming deployment realism, check whether real client gradients show comparable
norm, within-round cosine, and dual loss-drift distributions to the controlled
ALG archetypes.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

for candidate in (PROJECT_ROOT, BACKEND_DIR, SCRIPTS_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from common.schemas import DEFAULT_POLICY  # noqa: E402
from export_thesis_artifacts import (  # noqa: E402
    build_eval_bundle,
    build_round_bundles,
    classify_shadow_audit,
    clone_policy,
    pretrain_initial_checkpoint,
)
from l1_linear_defense.aggregation import cosine_similarity, geometric_median  # noqa: E402
from l2_dual_audit.classifier import DualChannelAuditor  # noqa: E402
from policy_agent.analysis.real_gradient_benchmark import (  # noqa: E402
    DEFAULT_CORNER_LABELS,
    RealGradientBenchmarkConfig,
    _client_is_corner_heavy,
    _pretrain_model,
    _split_reference_clients,
    compute_client_gradient,
    load_real_clients,
)
from policy_agent.analysis.unified_benchmark import _make_auditor, _make_generator  # noqa: E402


SCALAR_METRICS = (
    "gradient_norm",
    "log_norm",
    "cosine_to_geomed",
    "deviation_from_geomed",
    "delta_l_main",
    "delta_l_corner",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a synthetic-vs-real gradient calibration table for the "
            "CornerDrive dissertation future-work sanity check."
        )
    )
    parser.add_argument("--synthetic-rounds", type=int, default=6)
    parser.add_argument("--cycle-rounds", type=int, default=12)
    parser.add_argument("--synthetic-seed", type=int, default=20260318)
    parser.add_argument("--pretrain-epochs", type=int, default=5)
    parser.add_argument("--pretrain-batch-size", type=int, default=64)
    parser.add_argument("--pretrain-learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--source",
        default="auto",
        choices=[
            "auto",
            "leaf_femnist",
            "femnist",
            "mnist",
            "fashionmnist",
            "torchvision_mnist",
            "torchvision_fashionmnist",
        ],
    )
    parser.add_argument("--leaf-data-dir", default="data/real/femnist")
    parser.add_argument("--data-dir", default="data/real")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-real-clients", type=int, default=80)
    parser.add_argument("--min-samples-per-client", type=int, default=8)
    parser.add_argument("--max-samples-per-client", type=int, default=32)
    parser.add_argument("--real-rounds", type=int, default=6)
    parser.add_argument("--real-clients-per-round", type=int, default=16)
    parser.add_argument("--real-seed", type=int, default=20260507)
    parser.add_argument("--real-pretrain-steps", type=int, default=40)
    parser.add_argument("--local-batch-size", type=int, default=16)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "synthetic_real_gradient_calibration",
    )
    return parser.parse_args()


def safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def safe_std(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def safe_float(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else 0.0


def gradient_feature_row(
    *,
    source: str,
    dataset: str,
    round_id: int,
    client_id: str,
    label: str,
    family: str,
    gradient: np.ndarray,
    reference: np.ndarray,
    mean_reference: np.ndarray,
    delta_l_main: float,
    delta_l_corner: float,
    verdict: str,
    sample_count: int | str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    norm = float(np.linalg.norm(gradient))
    row = {
        "source": source,
        "dataset": dataset,
        "round_id": round_id,
        "client_id": client_id,
        "label": label,
        "family": family,
        "sample_count": sample_count,
        "gradient_dim": int(np.ravel(gradient).size),
        "gradient_norm": norm,
        "log_norm": math.log(norm + 1e-12),
        "cosine_to_geomed": safe_float(cosine_similarity(gradient, reference)),
        "cosine_to_mean": safe_float(cosine_similarity(gradient, mean_reference)),
        "deviation_from_geomed": 1.0 - safe_float(cosine_similarity(gradient, reference)),
        "delta_l_main": safe_float(delta_l_main),
        "delta_l_corner": safe_float(delta_l_corner),
        "audit_verdict": verdict,
    }
    if extra:
        row.update(extra)
    return row


def synthetic_gradient_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    initial_model, checkpoint_info = pretrain_initial_checkpoint(
        epochs=args.pretrain_epochs,
        batch_size=args.pretrain_batch_size,
        learning_rate=args.pretrain_learning_rate,
    )
    eval_bundle = build_eval_bundle()
    generator = _make_generator(
        initial_model=initial_model,
        eval_bundle=eval_bundle,
        generator_seed=args.synthetic_seed,
    )
    rounds, counts = build_round_bundles(
        policy=clone_policy(DEFAULT_POLICY),
        total_rounds=args.synthetic_rounds,
        cycle_rounds=args.cycle_rounds,
        generator=generator,
    )
    auditor = _make_auditor(
        initial_model,
        main_dataset=eval_bundle.audit_main,
        corner_dataset=eval_bundle.audit_corner,
    )
    auditor.apply_policy(DEFAULT_POLICY)
    base_main_loss = auditor.compute_loss(auditor.model, auditor.main_loader)
    base_corner_loss = auditor.compute_loss(auditor.model, auditor.corner_loader)

    rows: list[dict[str, Any]] = []
    for round_bundle in rounds:
        geomed, _iterations = geometric_median(round_bundle.gradients)
        mean_reference = np.mean(np.stack(round_bundle.gradients), axis=0)
        for idx, gradient in enumerate(round_bundle.gradients):
            metadata = round_bundle.updates[idx]["metadata"]
            shadow = classify_shadow_audit(
                auditor=auditor,
                vehicle_id=round_bundle.vehicle_ids[idx],
                gradient=gradient,
                base_main_loss=base_main_loss,
                base_corner_loss=base_corner_loss,
                audit_mode="dual",
            )
            rows.append(
                gradient_feature_row(
                    source="synthetic_alg",
                    dataset="ALG",
                    round_id=round_bundle.round_id,
                    client_id=round_bundle.vehicle_ids[idx],
                    label=round_bundle.role_by_index[idx],
                    family=str(metadata.get("attack_family", "none")),
                    gradient=gradient,
                    reference=geomed,
                    mean_reference=mean_reference,
                    delta_l_main=shadow["delta_l_main"],
                    delta_l_corner=shadow["delta_l_corner"],
                    verdict=shadow["verdict"],
                    sample_count=int(round_bundle.sample_counts[idx]),
                    extra={
                        "planned_role": round_bundle.planned_role_by_index[idx],
                        "phase_name": round_bundle.phase,
                    },
                )
            )

    return rows, {
        "synthetic_seed": args.synthetic_seed,
        "synthetic_rounds": args.synthetic_rounds,
        "synthetic_counts": dict(counts),
        "checkpoint": checkpoint_info,
        "audit_main_size": len(eval_bundle.audit_main),
        "audit_corner_size": len(eval_bundle.audit_corner),
    }


def real_gradient_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = RealGradientBenchmarkConfig(
        source=args.source,
        leaf_data_dir=args.leaf_data_dir,
        data_dir=args.data_dir,
        download=args.download,
        max_clients=args.max_real_clients,
        min_samples_per_client=args.min_samples_per_client,
        max_samples_per_client=args.max_samples_per_client,
        clients_per_round=args.real_clients_per_round,
        rounds=args.real_rounds,
        seed=args.real_seed,
        pretrain_steps=args.real_pretrain_steps,
        local_batch_size=args.local_batch_size,
        attack_fraction=0.0,
        corner_harm_fraction=0.0,
        noise_fraction=0.0,
    )
    clients, dataset_info = load_real_clients(config)
    input_dim = int(clients[0].inputs.view(clients[0].inputs.size(0), -1).size(1))
    output_dim = int(max(int(client.targets.max().item()) for client in clients) + 1)
    model = _pretrain_model(
        clients[: max(args.real_clients_per_round, 8)],
        input_dim=input_dim,
        output_dim=output_dim,
        steps=args.real_pretrain_steps,
        batch_size=args.local_batch_size,
        seed=args.real_seed,
    )
    main_dataset, corner_dataset = _split_reference_clients(clients)
    auditor = DualChannelAuditor(
        model=copy.deepcopy(model),
        main_dataset=main_dataset,
        corner_dataset=corner_dataset,
    )
    auditor.apply_policy(DEFAULT_POLICY)
    base_main_loss = auditor.compute_loss(auditor.model, auditor.main_loader)
    base_corner_loss = auditor.compute_loss(auditor.model, auditor.corner_loader)

    rng = random.Random(args.real_seed)
    rows: list[dict[str, Any]] = []
    for round_id in range(args.real_rounds):
        k = min(args.real_clients_per_round, len(clients))
        round_clients = rng.sample(clients, k=k)
        gradients = [
            compute_client_gradient(model, client, batch_size=args.local_batch_size)
            for client in round_clients
        ]
        geomed, _iterations = geometric_median(gradients)
        mean_reference = np.mean(np.stack(gradients), axis=0)
        for client, gradient in zip(round_clients, gradients):
            shadow = classify_shadow_audit(
                auditor=auditor,
                vehicle_id=client.client_id,
                gradient=gradient,
                base_main_loss=base_main_loss,
                base_corner_loss=base_corner_loss,
                audit_mode="dual",
            )
            label = (
                "REAL_CORNER_HEAVY"
                if _client_is_corner_heavy(client, DEFAULT_CORNER_LABELS)
                else "REAL_CLIENT"
            )
            rows.append(
                gradient_feature_row(
                    source="real_client_sgd",
                    dataset=str(dataset_info.get("source", args.source)),
                    round_id=round_id,
                    client_id=client.client_id,
                    label=label,
                    family="none",
                    gradient=gradient,
                    reference=geomed,
                    mean_reference=mean_reference,
                    delta_l_main=shadow["delta_l_main"],
                    delta_l_corner=shadow["delta_l_corner"],
                    verdict=shadow["verdict"],
                    sample_count=client.size,
                    extra={
                        "label_histogram": json.dumps(client.label_histogram, sort_keys=True),
                    },
                )
            )

    return rows, {
        "real_seed": args.real_seed,
        "real_rounds": args.real_rounds,
        "real_clients_per_round": args.real_clients_per_round,
        "dataset": {
            **dataset_info,
            "client_count": len(clients),
            "input_dim": input_dim,
            "output_dim": output_dim,
            "corner_labels": list(DEFAULT_CORNER_LABELS),
        },
    }


def grouped_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["source"]), str(row["label"]))].append(row)

    summary: list[dict[str, Any]] = []
    for (source, label), items in sorted(grouped.items()):
        record: dict[str, Any] = {
            "source": source,
            "label": label,
            "count": len(items),
            "fractions_of_source": len(items)
            / max(sum(1 for row in rows if row["source"] == source), 1),
        }
        for metric in SCALAR_METRICS:
            values = [float(item[metric]) for item in items]
            record[f"{metric}_mean"] = safe_mean(values)
            record[f"{metric}_std"] = safe_std(values)
        summary.append(record)
    return summary


def distance_rows(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    synthetic = [row for row in summary if row["source"] == "synthetic_alg"]
    real = [row for row in summary if row["source"] == "real_client_sgd"]
    distances: list[dict[str, Any]] = []
    for real_row in real:
        for synthetic_row in synthetic:
            smds: list[float] = []
            record: dict[str, Any] = {
                "real_label": real_row["label"],
                "synthetic_label": synthetic_row["label"],
            }
            for metric in SCALAR_METRICS:
                real_mean = float(real_row[f"{metric}_mean"])
                synth_mean = float(synthetic_row[f"{metric}_mean"])
                real_std = float(real_row[f"{metric}_std"])
                synth_std = float(synthetic_row[f"{metric}_std"])
                pooled = math.sqrt((real_std**2 + synth_std**2) / 2.0)
                smd = abs(real_mean - synth_mean) / pooled if pooled > 1e-12 else 0.0
                record[f"{metric}_smd"] = smd
                smds.append(smd)
            record["mean_smd"] = safe_mean(smds)
            distances.append(record)
    return sorted(distances, key=lambda row: (row["real_label"], row["mean_smd"]))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    synthetic_rows, synthetic_info = synthetic_gradient_rows(args)
    real_rows, real_info = real_gradient_rows(args)
    rows = synthetic_rows + real_rows
    summary = grouped_summary(rows)
    distances = distance_rows(summary)

    config = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Sanity check for dissertation future work: compare controlled ALG "
            "synthetic gradients with real client-SGD gradients using scalar "
            "norm, within-round cosine, and dual loss-drift distributions. "
            "Gradient vectors are not compared directly across models."
        ),
        "synthetic": synthetic_info,
        "real": real_info,
        "metrics": list(SCALAR_METRICS),
    }
    (output_dir / "synthetic_real_gradient_calibration_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
    write_csv(output_dir / "synthetic_real_gradient_features.csv", rows)
    write_csv(output_dir / "synthetic_real_gradient_summary.csv", summary)
    write_csv(output_dir / "synthetic_real_gradient_distance.csv", distances)

    print(f"Wrote synthetic-real gradient calibration artifacts to {output_dir}")
    for real_label in sorted({row["real_label"] for row in distances}):
        nearest = next(row for row in distances if row["real_label"] == real_label)
        print(
            f"{real_label}: nearest synthetic={nearest['synthetic_label']} "
            f"mean_smd={nearest['mean_smd']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

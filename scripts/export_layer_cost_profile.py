#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
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

from common.schemas import DEFAULT_POLICY  # noqa: E402
from export_thesis_artifacts import (  # noqa: E402
    build_eval_bundle,
    build_round_bundles,
    classify_shadow_audit,
    clone_policy,
    policy_with_recheck_probability,
    pretrain_initial_checkpoint,
    run_flpg_with_artifacts,
    write_csv,
    write_json,
)
from export_synthetic_alg_benchmark import parse_seed_values  # noqa: E402
from l1_linear_defense.config import make_l1_router_config  # noqa: E402
from policy_agent.analysis.unified_benchmark import _make_auditor, _make_generator  # noqa: E402


DEFAULT_SEEDS = "20260318,20260319,20260320,20260321,20260322"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile L1+L2 and Exhaustive L2 audit cost on this host."
    )
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--cycle-rounds", type=int, default=12)
    parser.add_argument("--pretrain-epochs", type=int, default=5)
    parser.add_argument("--pretrain-batch-size", type=int, default=64)
    parser.add_argument("--pretrain-learning-rate", type=float, default=1e-3)
    parser.add_argument("--seeds", type=str, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "layer_cost_profile",
    )
    parser.add_argument(
        "--worker-setting",
        choices=("l1_p010", "exhaustive_l2"),
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def safe_std(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    try:
        import torch

        info.update({
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count()
            if torch.cuda.is_available()
            else 0,
            "mps_available": bool(
                getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
            ),
        })
    except Exception as exc:  # pragma: no cover - environment diagnostics only
        info["torch_error"] = f"{type(exc).__name__}: {exc}"
    return info


def measure_l2_unit_cost(
    *,
    initial_model: Any,
    eval_bundle: Any,
    reference_policy: Any,
    seed: int,
    rounds: int,
    cycle_rounds: int,
    repeats: int = 3,
) -> dict[str, float]:
    generator = _make_generator(
        initial_model=initial_model,
        eval_bundle=eval_bundle,
        generator_seed=seed,
    )
    generator.ground_truth_mode = "archetype"
    round_bundles, _overall_counts = build_round_bundles(
        policy=reference_policy,
        total_rounds=max(1, rounds),
        cycle_rounds=cycle_rounds,
        generator=generator,
    )
    gradients = round_bundles[0].gradients
    auditor = _make_auditor(
        initial_model,
        main_dataset=eval_bundle.audit_main,
        corner_dataset=eval_bundle.audit_corner,
    )
    auditor.apply_policy(reference_policy)

    base_start = time.perf_counter()
    for _ in range(repeats):
        base_main_loss = auditor.compute_loss(auditor.model, auditor.main_loader)
        base_corner_loss = auditor.compute_loss(auditor.model, auditor.corner_loader)
    baseline_seconds_per_round = (time.perf_counter() - base_start) / repeats

    base_main_loss = auditor.compute_loss(auditor.model, auditor.main_loader)
    base_corner_loss = auditor.compute_loss(auditor.model, auditor.corner_loader)
    candidate_count = len(gradients) * repeats
    candidate_start = time.perf_counter()
    for _ in range(repeats):
        for idx, gradient in enumerate(gradients):
            classify_shadow_audit(
                auditor=auditor,
                vehicle_id=f"micro_{idx}",
                gradient=gradient,
                base_main_loss=base_main_loss,
                base_corner_loss=base_corner_loss,
                audit_mode="dual",
            )
    candidate_seconds_per_eval = (
        time.perf_counter() - candidate_start
    ) / candidate_count

    return {
        "micro_baseline_seconds_per_round": baseline_seconds_per_round,
        "micro_candidate_seconds_per_l2_eval": candidate_seconds_per_eval,
        "micro_candidate_eval_count": float(candidate_count),
    }


def run_worker(args: argparse.Namespace) -> int:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seed_values(args.seeds)
    setting = str(args.worker_setting)

    setup_start = time.perf_counter()
    initial_model, pretrain_info = pretrain_initial_checkpoint(
        epochs=args.pretrain_epochs,
        batch_size=args.pretrain_batch_size,
        learning_rate=args.pretrain_learning_rate,
    )
    eval_bundle = build_eval_bundle()
    setup_wall_clock = time.perf_counter() - setup_start

    if setting == "l1_p010":
        method = "L1+L2 p=0.10"
        reported_p = 0.10
        policy_p = 0.10
        force_exhaustive_l2 = False
        fixed_recheck_probability = 0.10
    else:
        method = "Exhaustive L2"
        reported_p = 1.0
        policy_p = 0.50
        force_exhaustive_l2 = True
        fixed_recheck_probability = 0.50

    reference_policy = policy_with_recheck_probability(
        clone_policy(DEFAULT_POLICY),
        policy_p,
    )
    l1_router_config = make_l1_router_config(
        "cosine_recheck",
        cos_deviation_threshold=reference_policy.cosine_filter_threshold,
    )
    micro_cost = measure_l2_unit_cost(
        initial_model=initial_model,
        eval_bundle=eval_bundle,
        reference_policy=reference_policy,
        seed=seeds[0],
        rounds=args.rounds,
        cycle_rounds=args.cycle_rounds,
    )

    seed_rows: list[dict[str, Any]] = []
    phase_start = time.perf_counter()
    cpu_start = time.process_time()
    for seed in seeds:
        generator = _make_generator(
            initial_model=initial_model,
            eval_bundle=eval_bundle,
            generator_seed=seed,
        )
        generator.ground_truth_mode = "archetype"
        rounds, _overall_counts = build_round_bundles(
            policy=reference_policy,
            total_rounds=args.rounds,
            cycle_rounds=args.cycle_rounds,
            generator=generator,
        )
        (
            round_summary_rows,
            l1_rows,
            _l2_rows,
            _policy_rows,
            _flpg_baseline_rows,
            _summary,
        ) = run_flpg_with_artifacts(
            rounds=rounds,
            initial_model=initial_model,
            eval_bundle=eval_bundle,
            reference_policy=reference_policy,
            fixed_recheck_probability=fixed_recheck_probability,
            adapt_policy=False,
            audit_mode="dual",
            recheck_seed=20260428 + seed,
            compute_oracle_drift=False,
            l1_router_config=l1_router_config,
            force_exhaustive_l2=force_exhaustive_l2,
        )
        l2_evals = sum(
            1
            for row in l1_rows
            if str(row.get("routed_to_l2", "")).lower() in {"true", "1", "yes"}
        )
        updates_total = len(l1_rows)
        seed_rows.append({
            "method": method,
            "seed": seed,
            "p_recheck_reported": reported_p,
            "updates_total": updates_total,
            "l2_evals": l2_evals,
            "audit_queue_ratio": l2_evals / updates_total if updates_total else 0.0,
            "main_accuracy": mean(
                float(row["main_task_accuracy"]) for row in round_summary_rows
            ),
            "corner_accuracy": mean(
                float(row["corner_case_accuracy"]) for row in round_summary_rows
            ),
        })

    phase_wall_clock = time.perf_counter() - phase_start
    phase_cpu_seconds = time.process_time() - cpu_start
    total_updates = sum(int(row["updates_total"]) for row in seed_rows)
    total_l2_evals = sum(int(row["l2_evals"]) for row in seed_rows)
    total_rounds = args.rounds * len(seeds)
    audit_samples_per_eval = len(eval_bundle.audit_main) + len(eval_bundle.audit_corner)
    baseline_audit_sample_evals = total_rounds * audit_samples_per_eval
    candidate_audit_sample_evals = total_l2_evals * audit_samples_per_eval
    estimated_deployed_audit_wall_clock = (
        total_rounds * micro_cost["micro_baseline_seconds_per_round"]
        + total_l2_evals * micro_cost["micro_candidate_seconds_per_l2_eval"]
    )

    metrics = {
        "method": method,
        "setting": setting,
        "seeds": len(seeds),
        "rounds_per_seed": args.rounds,
        "setup_wall_clock_seconds": setup_wall_clock,
        "phase_wall_clock_seconds": phase_wall_clock,
        "phase_cpu_seconds": phase_cpu_seconds,
        "updates_total": total_updates,
        "l2_evals": total_l2_evals,
        "audit_queue_ratio": total_l2_evals / total_updates if total_updates else 0.0,
        "l2_evals_per_round": total_l2_evals / total_rounds if total_rounds else 0.0,
        "audit_samples_per_l2_eval": audit_samples_per_eval,
        "baseline_audit_sample_evals": baseline_audit_sample_evals,
        "candidate_audit_sample_evals": candidate_audit_sample_evals,
        "total_audit_sample_evals": (
            baseline_audit_sample_evals + candidate_audit_sample_evals
        ),
        **micro_cost,
        "estimated_deployed_audit_wall_clock_seconds": (
            estimated_deployed_audit_wall_clock
        ),
        "main_accuracy_mean": mean(float(row["main_accuracy"]) for row in seed_rows),
        "main_accuracy_std": safe_std(
            [float(row["main_accuracy"]) for row in seed_rows]
        ),
        "corner_accuracy_mean": mean(float(row["corner_accuracy"]) for row in seed_rows),
        "corner_accuracy_std": safe_std(
            [float(row["corner_accuracy"]) for row in seed_rows]
        ),
        "pretrain": pretrain_info,
    }
    write_csv(output_dir / "cost_profile_by_seed.csv", [], seed_rows)
    write_json(output_dir / "cost_profile_worker_metrics.json", metrics)
    return 0


def maxrss_mb(raw_maxrss: int) -> float:
    if platform.system() == "Darwin":
        return raw_maxrss / (1024 * 1024)
    return raw_maxrss / 1024


def run_measured_child(
    setting: str,
    args: argparse.Namespace,
    parent_output_dir: Path,
) -> dict[str, Any]:
    child_output_dir = parent_output_dir / setting
    child_output_dir.mkdir(parents=True, exist_ok=True)
    log_path = child_output_dir / "run.log"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-setting",
        setting,
        "--rounds",
        str(args.rounds),
        "--cycle-rounds",
        str(args.cycle_rounds),
        "--pretrain-epochs",
        str(args.pretrain_epochs),
        "--pretrain-batch-size",
        str(args.pretrain_batch_size),
        "--pretrain-learning-rate",
        str(args.pretrain_learning_rate),
        "--seeds",
        args.seeds,
        "--output-dir",
        str(child_output_dir),
    ]
    start = time.perf_counter()
    with log_path.open("w") as log_file:
        proc = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        _pid, status, usage = os.wait4(proc.pid, 0)
    wall_clock = time.perf_counter() - start
    metrics_path = child_output_dir / "cost_profile_worker_metrics.json"
    with metrics_path.open() as f:
        metrics = json.load(f)
    metrics.update({
        "process_wall_clock_seconds": wall_clock,
        "process_user_cpu_seconds": usage.ru_utime,
        "process_system_cpu_seconds": usage.ru_stime,
        "process_total_cpu_seconds": usage.ru_utime + usage.ru_stime,
        "peak_rss_mb": maxrss_mb(usage.ru_maxrss),
        "exit_status": status,
        "log_path": str(log_path),
    })
    return metrics


def add_relative_columns(rows: list[dict[str, Any]]) -> None:
    baseline = next(row for row in rows if row["setting"] == "l1_p010")
    total_rounds = float(baseline["rounds_per_seed"]) * float(baseline["seeds"])
    normalized_base_seconds = float(baseline["micro_baseline_seconds_per_round"])
    normalized_candidate_seconds = float(
        baseline["micro_candidate_seconds_per_l2_eval"]
    )
    for row in rows:
        row["normalized_estimated_deployed_audit_wall_clock_seconds"] = (
            total_rounds * normalized_base_seconds
            + float(row["l2_evals"]) * normalized_candidate_seconds
        )
        for key in (
            "process_wall_clock_seconds",
            "phase_wall_clock_seconds",
            "process_total_cpu_seconds",
            "l2_evals",
            "candidate_audit_sample_evals",
            "total_audit_sample_evals",
            "estimated_deployed_audit_wall_clock_seconds",
            "normalized_estimated_deployed_audit_wall_clock_seconds",
        ):
            base_value = float(baseline[key])
            row[f"relative_{key}_vs_l1"] = (
                float(row[key]) / base_value if base_value else ""
            )
        row["gpu_time_seconds"] = ""
        row["energy_joules"] = ""
        row["gpu_energy_note"] = (
            "not measured: current PyTorch runtime reports CUDA=False and MPS=False; "
            "no GPU or power telemetry was available on this host"
        )


def main() -> int:
    args = parse_args()
    if args.worker_setting:
        return run_worker(args)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        run_measured_child("l1_p010", args, output_dir),
        run_measured_child("exhaustive_l2", args, output_dir),
    ]
    add_relative_columns(rows)

    write_json(
        output_dir / "layer_cost_profile_config.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "rounds": args.rounds,
            "cycle_rounds": args.cycle_rounds,
            "pretrain_epochs": args.pretrain_epochs,
            "seeds": parse_seed_values(args.seeds),
            "hardware": hardware_info(),
            "scope": (
                "Host-local cost profile. Wall-clock and RSS are machine-dependent; "
                "audit sample evaluations are the portable compute proxy."
            ),
        },
    )
    write_csv(output_dir / "layer_cost_profile_summary.csv", [], rows)
    print(f"Wrote layer cost profile artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

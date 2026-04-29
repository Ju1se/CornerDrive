"""Unified benchmark runner with a shared pipeline across all baselines."""

from __future__ import annotations

import asyncio
import copy
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = PROJECT_ROOT / "backend"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

for candidate in (PROJECT_ROOT, BACKEND_DIR, SCRIPTS_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from common.config import L2_LEARNING_RATE  # noqa: E402
from common.demo_audit_assets import (  # noqa: E402
    DEMO_CLASS_CENTER_SEED,
    DEMO_CLASS_STYLE_SEED,
    DEMO_CORNER_DATASET_SEED,
    DEMO_INPUT_DIM,
    DEMO_MAIN_DATASET_SEED,
    DEMO_OUTPUT_DIM,
    PlaceholderDataset,
    SimpleMLP,
)
from common.schemas import DEFAULT_POLICY, Policy, RoundTelemetry  # noqa: E402
from generate_demo_data import BATCH_SIZE, VEHICLE_POOL_SIZE, DemoDataGenerator  # noqa: E402
from l1_linear_defense.aggregation import filter_suspects, geometric_median  # noqa: E402
from l2_dual_audit.classifier import DualChannelAuditor  # noqa: E402
from policy_agent.constraints.safety_guard import SafetyGuard  # noqa: E402
from policy_agent.engine.rule_engine import RuleEngine  # noqa: E402


BENCHMARK_PHASE_CYCLE = [
    "steady",
    "fraud_wave",
    "steady",
    "corner_gap",
    "steady",
    "false_slash_risk",
    "steady",
    "drift_burst",
    "steady",
    "fraud_wave",
    "steady",
    "corner_gap",
]
TARGET_LABELS = ("RARITY", "FRAUD")
PRETRAIN_SEED = 20260404


@dataclass
class EnvironmentRound:
    round_id: int
    cycle_round_index: int
    phase: str
    gradients: list[np.ndarray]
    sample_counts: list[int]
    role_by_index: list[str]
    role_counts: Counter
    vehicle_ids: list[str]


@dataclass
class ModelMetrics:
    loss: float
    accuracy: float


@dataclass
class EvalBundle:
    proto_main: torch.utils.data.Dataset
    proto_corner: torch.utils.data.Dataset
    audit_main: torch.utils.data.Dataset
    audit_corner: torch.utils.data.Dataset
    oracle_main: torch.utils.data.Dataset
    oracle_corner: torch.utils.data.Dataset

    @property
    def main_train(self) -> torch.utils.data.Dataset:
        return self.proto_main

    @property
    def corner_train(self) -> torch.utils.data.Dataset:
        return self.proto_corner

    @property
    def main_eval(self) -> torch.utils.data.Dataset:
        return self.oracle_main

    @property
    def corner_eval(self) -> torch.utils.data.Dataset:
        return self.oracle_corner


@dataclass
class StrategyRuntime:
    model: nn.Module
    main_loader: torch.utils.data.DataLoader
    corner_loader: torch.utils.data.DataLoader
    criterion: nn.Module


def clone_policy(policy: Policy, round_id: int | None = None) -> Policy:
    payload = policy.model_dump()
    if round_id is not None:
        payload["round_id"] = round_id
        payload["effective_from_round"] = round_id
    return Policy.model_validate(payload)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return vector.copy()
    return vector / norm


def _build_dataset_components() -> tuple[torch.Tensor, torch.Tensor]:
    centers_generator = torch.Generator().manual_seed(DEMO_CLASS_CENTER_SEED)
    class_centers = F.normalize(
        torch.randn(DEMO_OUTPUT_DIM, DEMO_INPUT_DIM, generator=centers_generator),
        dim=1,
    ) * 2.5

    styles_generator = torch.Generator().manual_seed(DEMO_CLASS_STYLE_SEED)
    class_styles = F.normalize(
        torch.randn(DEMO_OUTPUT_DIM, DEMO_INPUT_DIM, generator=styles_generator),
        dim=1,
    )

    return class_centers, class_styles


def build_eval_bundle() -> EvalBundle:
    class_centers, class_styles = _build_dataset_components()
    proto_main = PlaceholderDataset(
        size=300,
        seed=DEMO_MAIN_DATASET_SEED,
        variant="main",
        class_centers=class_centers,
        class_styles=class_styles,
    )
    proto_corner = PlaceholderDataset(
        size=100,
        seed=DEMO_CORNER_DATASET_SEED,
        variant="corner",
        class_centers=class_centers,
        class_styles=class_styles,
    )
    audit_main = PlaceholderDataset(
        size=500,
        seed=DEMO_MAIN_DATASET_SEED + 100,
        variant="main",
        class_centers=class_centers,
        class_styles=class_styles,
    )
    audit_corner = PlaceholderDataset(
        size=100,
        seed=DEMO_CORNER_DATASET_SEED + 100,
        variant="corner",
        class_centers=class_centers,
        class_styles=class_styles,
    )
    oracle_main = PlaceholderDataset(
        size=1000,
        seed=DEMO_MAIN_DATASET_SEED + 200,
        variant="main",
        class_centers=class_centers,
        class_styles=class_styles,
    )
    oracle_corner = PlaceholderDataset(
        size=200,
        seed=DEMO_CORNER_DATASET_SEED + 200,
        variant="corner",
        class_centers=class_centers,
        class_styles=class_styles,
    )
    return EvalBundle(
        proto_main=proto_main,
        proto_corner=proto_corner,
        audit_main=audit_main,
        audit_corner=audit_corner,
        oracle_main=oracle_main,
        oracle_corner=oracle_corner,
    )


def pretrain_initial_checkpoint(
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[nn.Module, dict[str, Any]]:
    eval_bundle = build_eval_bundle()
    torch.manual_seed(PRETRAIN_SEED)
    model = SimpleMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    train_loader = torch.utils.data.DataLoader(
        eval_bundle.proto_main,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(PRETRAIN_SEED),
    )

    for _ in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    runtime = StrategyRuntime(
        model=model,
        main_loader=torch.utils.data.DataLoader(eval_bundle.oracle_main, batch_size=64, shuffle=False),
        corner_loader=torch.utils.data.DataLoader(eval_bundle.oracle_corner, batch_size=64, shuffle=False),
        criterion=criterion,
    )
    main_metrics, corner_metrics = _evaluate_runtime(runtime)

    checkpoint_info = {
        "pretrain_seed": PRETRAIN_SEED,
        "pretrain_epochs": epochs,
        "pretrain_batch_size": batch_size,
        "pretrain_learning_rate": learning_rate,
        "initial_main_accuracy": main_metrics.accuracy,
        "initial_corner_accuracy": corner_metrics.accuracy,
        "initial_main_loss": main_metrics.loss,
        "initial_corner_loss": corner_metrics.loss,
    }
    return model, checkpoint_info


def _evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
) -> ModelMetrics:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            predictions = outputs.argmax(dim=1)

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_correct += int((predictions == targets).sum().item())
            total_samples += batch_size

    if total_samples == 0:
        return ModelMetrics(loss=0.0, accuracy=0.0)

    return ModelMetrics(
        loss=total_loss / total_samples,
        accuracy=total_correct / total_samples,
    )


def _evaluate_runtime(runtime: StrategyRuntime) -> tuple[ModelMetrics, ModelMetrics]:
    return (
        _evaluate_model(runtime.model, runtime.main_loader, runtime.criterion),
        _evaluate_model(runtime.model, runtime.corner_loader, runtime.criterion),
    )


def _make_runtime(initial_model: nn.Module, eval_bundle: EvalBundle) -> StrategyRuntime:
    return StrategyRuntime(
        model=copy.deepcopy(initial_model),
        main_loader=torch.utils.data.DataLoader(eval_bundle.oracle_main, batch_size=64, shuffle=False),
        corner_loader=torch.utils.data.DataLoader(eval_bundle.oracle_corner, batch_size=64, shuffle=False),
        criterion=nn.CrossEntropyLoss(),
    )


def _apply_aggregated_gradient(
    model: nn.Module,
    aggregated_gradient: np.ndarray,
    learning_rate: float = L2_LEARNING_RATE,
) -> nn.Module:
    model_copy = copy.deepcopy(model)
    flat_params = torch.cat([
        parameter.data.view(-1)
        for parameter in model_copy.parameters()
    ])
    gradient_tensor = torch.tensor(aggregated_gradient, dtype=torch.float32)

    if flat_params.numel() != gradient_tensor.numel():
        raise ValueError("Aggregated gradient dimension does not match model parameter dimension")

    flat_params -= learning_rate * gradient_tensor

    offset = 0
    for parameter in model_copy.parameters():
        param_size = parameter.numel()
        parameter.data = flat_params[offset:offset + param_size].view(parameter.shape)
        offset += param_size

    return model_copy


def _mean_gradient(gradients: list[np.ndarray]) -> np.ndarray:
    if not gradients:
        raise ValueError("Cannot average an empty gradient list")
    return np.mean(np.stack(gradients), axis=0)


def _make_generator(
    *,
    initial_model: nn.Module,
    eval_bundle: EvalBundle,
    generator_seed: int = 20260318,
) -> DemoDataGenerator:
    generator = DemoDataGenerator(seed=generator_seed)
    generator.ground_truth_mode = "archetype"
    generator.prototype_model = copy.deepcopy(initial_model)
    generator.main_dataset = eval_bundle.proto_main
    generator.corner_dataset = eval_bundle.proto_corner
    generator.auditor = DualChannelAuditor(
        model=copy.deepcopy(initial_model),
        main_dataset=eval_bundle.proto_main,
        corner_dataset=eval_bundle.proto_corner,
    )
    generator.main_gradient = _normalize_vector(
        generator._compute_dataset_gradient(eval_bundle.proto_main)
    )
    generator.corner_gradient = _normalize_vector(
        generator._compute_dataset_gradient(eval_bundle.proto_corner)
    )
    generator.random_gradient = generator._build_random_basis()
    return generator


def _build_base_cycle(
    policy: Policy,
    cycle_rounds: int,
    generator: DemoDataGenerator,
) -> tuple[list[EnvironmentRound], Counter]:
    generator._phase_name = lambda round_index: BENCHMARK_PHASE_CYCLE[round_index % len(BENCHMARK_PHASE_CYCLE)]  # type: ignore[method-assign]
    environment: list[EnvironmentRound] = []
    total_counts: Counter = Counter()

    for round_index in range(cycle_rounds):
        round_policy = clone_policy(policy, round_id=policy.round_id + round_index)
        (
            updates,
            _planned_roles,
            vehicle_addresses,
            _new_vehicle_count,
            preflight_counts,
            _l1_projection,
        ) = generator._build_batch(
            round_id=round_policy.round_id,
            policy=round_policy,
            round_index=round_index,
        )
        gradients = [np.array(payload["gradient_data"], dtype=float) for payload in updates]
        sample_counts = [int(payload["data_sample_count"]) for payload in updates]
        role_by_index = [
            str(
                payload["metadata"].get(
                    "ground_truth_label",
                    payload["metadata"].get("ground_truth_role", payload["metadata"]["planned_role"]),
                )
            )
            for payload in updates
        ]
        total_counts.update(role_by_index)
        environment.append(
            EnvironmentRound(
                round_id=round_policy.round_id,
                cycle_round_index=round_index % len(BENCHMARK_PHASE_CYCLE),
                phase=BENCHMARK_PHASE_CYCLE[round_index % len(BENCHMARK_PHASE_CYCLE)],
                gradients=gradients,
                sample_counts=sample_counts,
                role_by_index=role_by_index,
                role_counts=Counter(role_by_index),
                vehicle_ids=vehicle_addresses,
            )
        )

    return environment, total_counts


def _expand_environment(
    policy: Policy,
    *,
    total_rounds: int,
    cycle_rounds: int,
    generator: DemoDataGenerator,
) -> tuple[list[EnvironmentRound], Counter]:
    _ = cycle_rounds
    return _build_base_cycle(policy, total_rounds, generator)


def _make_auditor(
    model: nn.Module,
    *,
    main_dataset: torch.utils.data.Dataset,
    corner_dataset: torch.utils.data.Dataset,
) -> DualChannelAuditor:
    return DualChannelAuditor(
        model=copy.deepcopy(model),
        main_dataset=main_dataset,
        corner_dataset=corner_dataset,
    )


def _role_counts_for_indices(env_round: EnvironmentRound, indices: set[int]) -> Counter:
    return Counter(env_round.role_by_index[idx] for idx in indices)


def _pairwise_squared_distances(gradients: list[np.ndarray]) -> np.ndarray:
    matrix = np.stack(gradients)
    gram = matrix @ matrix.T
    squared_norms = np.sum(matrix * matrix, axis=1, keepdims=True)
    distances = squared_norms + squared_norms.T - 2.0 * gram
    return np.maximum(distances, 0.0)


def _multi_krum_indices(
    gradients: list[np.ndarray],
    *,
    byzantine_budget: int,
) -> list[int]:
    n = len(gradients)
    if n == 0:
        raise ValueError("Cannot run Multi-Krum on empty gradients")
    if n <= 2:
        return list(range(n))

    f = max(1, min(byzantine_budget, (n - 3) // 2))
    neighbor_count = max(1, n - f - 2)
    selection_count = max(1, n - f - 2)

    distances = _pairwise_squared_distances(gradients)
    np.fill_diagonal(distances, np.inf)

    scores: list[tuple[float, int]] = []
    for idx in range(n):
        nearest = np.partition(distances[idx], neighbor_count - 1)[:neighbor_count]
        scores.append((float(np.sum(nearest)), idx))

    scores.sort(key=lambda item: (item[0], item[1]))
    return [idx for _, idx in scores[:selection_count]]


def _phase_breakdown(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["phase"])].append(record)

    breakdown: dict[str, dict[str, float]] = {}
    for phase, items in grouped.items():
        breakdown[phase] = {
            "rounds": len(items),
            "main_accuracy_avg": mean(float(item["main_accuracy"]) for item in items),
            "corner_accuracy_avg": mean(float(item["corner_accuracy"]) for item in items),
            "fraud_survival_rate_avg": mean(float(item["fraud_survival_rate"]) for item in items),
        }
    return breakdown


def _classification_summary(
    *,
    predicted_by_round: list[list[str]] | None,
    rounds: list[EnvironmentRound],
) -> dict[str, Any] | None:
    if predicted_by_round is None:
        return None

    total_by_role = Counter()
    predicted_by_role = Counter()
    true_positive_by_role = Counter()

    for env_round, predicted in zip(rounds, predicted_by_round):
        for ground_truth, pred in zip(env_round.role_by_index, predicted):
            if ground_truth in TARGET_LABELS:
                total_by_role[ground_truth] += 1
                if pred == ground_truth:
                    true_positive_by_role[ground_truth] += 1
            if pred in TARGET_LABELS:
                predicted_by_role[pred] += 1

    per_role: dict[str, dict[str, float]] = {}
    for role in TARGET_LABELS:
        total = total_by_role[role]
        predicted_total = predicted_by_role[role]
        per_role[role] = {
            "precision": true_positive_by_role[role] / predicted_total if predicted_total else 0.0,
            "recall": true_positive_by_role[role] / total if total else 0.0,
            "support": total,
        }
    return per_role


def _round_record(
    *,
    round_index: int,
    env_round: EnvironmentRound,
    method_id: str,
    method_label: str,
    main_metrics: ModelMetrics,
    corner_metrics: ModelMetrics,
    selected_role_counts: Counter,
    predicted_role_counts: Counter | None,
    fraud_total: int,
) -> dict[str, Any]:
    fraud_selected = int(selected_role_counts.get("FRAUD", 0))
    return {
        "round": round_index,
        "phase": env_round.phase,
        "method": method_id,
        "method_label": method_label,
        "main_accuracy": main_metrics.accuracy,
        "corner_accuracy": corner_metrics.accuracy,
        "main_loss": main_metrics.loss,
        "corner_loss": corner_metrics.loss,
        "fraud_count": int((predicted_role_counts or selected_role_counts).get("FRAUD", 0)),
        "rarity_count": int((predicted_role_counts or selected_role_counts).get("RARITY", 0)),
        "honest_count": int((predicted_role_counts or selected_role_counts).get("HONEST", 0)),
        "noise_count": int((predicted_role_counts or selected_role_counts).get("NOISE", 0)),
        "round_fraud_total": int(env_round.role_counts.get("FRAUD", 0)),
        "round_rarity_total": int(env_round.role_counts.get("RARITY", 0)),
        "round_honest_total": int(env_round.role_counts.get("HONEST", 0)),
        "round_noise_total": int(env_round.role_counts.get("NOISE", 0)),
        "selected_fraud_count": fraud_selected,
        "selected_rarity_count": int(selected_role_counts.get("RARITY", 0)),
        "selected_honest_count": int(selected_role_counts.get("HONEST", 0)),
        "selected_noise_count": int(selected_role_counts.get("NOISE", 0)),
        "selected_total": int(sum(selected_role_counts.values())),
        "fraud_survival_rate": fraud_selected / fraud_total if fraud_total else 0.0,
    }


def _build_adaptive_telemetry(
    *,
    round_id: int,
    env_round: EnvironmentRound,
    selected_role_counts: Counter,
    predicted_labels: list[str],
    before_main: ModelMetrics,
    after_main: ModelMetrics,
    before_corner: ModelMetrics,
    after_corner: ModelMetrics,
) -> RoundTelemetry:
    total = max(len(env_round.role_by_index), 1)
    fraud_rate = selected_role_counts.get("FRAUD", 0) / total
    rarity_rate = selected_role_counts.get("RARITY", 0) / total
    honest_rate = selected_role_counts.get("HONEST", 0) / total
    noise_rate = selected_role_counts.get("NOISE", 0) / total

    total_rarity = env_round.role_counts.get("RARITY", 0)
    retained_rarity = selected_role_counts.get("RARITY", 0)
    false_slash = sum(
        1
        for ground_truth, predicted in zip(env_round.role_by_index, predicted_labels)
        if ground_truth == "HONEST" and predicted == "FRAUD"
    )

    return RoundTelemetry(
        round_id=round_id,
        fraud_rate=fraud_rate,
        rarity_rate=rarity_rate,
        honest_rate=honest_rate,
        noise_rate=noise_rate,
        main_accuracy=after_main.accuracy,
        corner_accuracy=after_corner.accuracy,
        main_loss_delta_avg=after_main.loss - before_main.loss,
        corner_loss_delta_avg=after_corner.loss - before_corner.loss,
        false_slash_estimate=false_slash / total,
        rarity_retention_rate=retained_rarity / total_rarity if total_rarity else 1.0,
        golden_drift_score=_clamp(
            abs(after_main.loss - before_main.loss) * 1.4
            + fraud_rate * 0.6
            + noise_rate * 0.25,
            0.0,
            1.0,
        ),
        reject_rate_l3=0.0,
        cosine_outlier_ratio=sum(
            1 for predicted in predicted_labels if predicted in {"FRAUD", "RARITY", "NOISE"}
        ) / total,
        suspect_queue_length=0,
        audit_sample_size=len(predicted_labels),
        avg_sbt_score=0.0,
        new_vehicle_ratio=0.0,
        hash_mismatch_rate=0.0,
        recent_attack_pressure=_clamp(
            fraud_rate * 1.2
            + noise_rate * 0.5
            + env_round.role_counts.get("FRAUD", 0) / total * 0.2,
            0.0,
            1.0,
        ),
    )


def run_unified_benchmark(
    *,
    rounds: int = 24,
    cycle_rounds: int = 12,
    pretrain_epochs: int = 7,
    pretrain_batch_size: int = 64,
    pretrain_learning_rate: float = 1e-3,
    policy: Policy | None = None,
    krum_byzantine_budget: int = 10,
) -> dict[str, Any]:
    reference_policy = clone_policy(policy or DEFAULT_POLICY)
    initial_model, checkpoint_info = pretrain_initial_checkpoint(
        epochs=pretrain_epochs,
        batch_size=pretrain_batch_size,
        learning_rate=pretrain_learning_rate,
    )
    eval_bundle = build_eval_bundle()
    generator = _make_generator(
        initial_model=initial_model,
        eval_bundle=eval_bundle,
    )
    environment_rounds, overall_ground_truth_counts = _expand_environment(
        reference_policy,
        total_rounds=rounds,
        cycle_rounds=cycle_rounds,
        generator=generator,
    )
    base_cycle_counts = Counter()
    for env_round in environment_rounds[:cycle_rounds]:
        base_cycle_counts.update(env_round.role_counts)

    methods = [
        ("fedavg", "FedAvg"),
        ("geomed", "GeoMed"),
        ("krum", "Multi-Krum"),
        ("flpg_adaptive", "Full FLPG (adaptive)"),
        ("flpg_recheck", "Full FLPG (recheck p=0.10)"),
    ]

    results_by_method: dict[str, dict[str, Any]] = {}

    for method_id, method_label in methods:
        runtime = _make_runtime(initial_model, eval_bundle)
        current_policy = clone_policy(reference_policy)
        rule_engine = RuleEngine()
        safety_guard = SafetyGuard()
        round_records: list[dict[str, Any]] = []
        predicted_by_round: list[list[str]] | None = (
            [] if method_id in {"flpg_adaptive", "flpg_recheck"} else None
        )
        recheck_rng = random.Random(20260428 + sum(ord(char) for char in method_id))

        for round_index, env_round in enumerate(environment_rounds):
            before_main, before_corner = _evaluate_runtime(runtime)
            fraud_total = int(env_round.role_counts.get("FRAUD", 0))

            if method_id == "fedavg":
                aggregated_gradient = _mean_gradient(env_round.gradients)
                selected_indices = set(range(len(env_round.gradients)))
                predicted_counts = None

            elif method_id == "geomed":
                aggregated_gradient, _ = geometric_median(env_round.gradients)
                selected_indices = set(range(len(env_round.gradients)))
                predicted_counts = None

            elif method_id == "krum":
                selected_indices = set(
                    _multi_krum_indices(
                        env_round.gradients,
                        byzantine_budget=krum_byzantine_budget,
                    )
                )
                aggregated_gradient = _mean_gradient(
                    [env_round.gradients[idx] for idx in sorted(selected_indices)]
                )
                predicted_counts = None

            else:
                eval_policy = clone_policy(current_policy, round_id=reference_policy.round_id + round_index)
                if method_id == "flpg_recheck":
                    policy_payload = eval_policy.model_dump()
                    policy_payload["recheck_probability"] = max(
                        float(policy_payload["recheck_probability"]),
                        0.10,
                    )
                    eval_policy = Policy.model_validate(policy_payload)
                l1_result = filter_suspects(
                    env_round.gradients,
                    env_round.vehicle_ids,
                    threshold=eval_policy.cosine_filter_threshold,
                    recheck_probability=eval_policy.recheck_probability,
                    rng=recheck_rng,
                )
                suspect_indices = set(l1_result.suspect_indices)
                selected_indices = set(range(len(env_round.gradients))) - suspect_indices
                predicted_labels = ["HONEST" for _ in env_round.gradients]
                auditor = _make_auditor(
                    runtime.model,
                    main_dataset=eval_bundle.audit_main,
                    corner_dataset=eval_bundle.audit_corner,
                )
                auditor.apply_policy(eval_policy)

                for idx in suspect_indices:
                    audit = auditor.audit(env_round.vehicle_ids[idx], env_round.gradients[idx])
                    predicted_labels[idx] = audit.classification.value
                    if audit.include_in_aggregation:
                        selected_indices.add(idx)

                if selected_indices:
                    aggregated_gradient = _mean_gradient(
                        [env_round.gradients[idx] for idx in sorted(selected_indices)]
                    )
                else:
                    aggregated_gradient = np.zeros_like(env_round.gradients[0])

                predicted_counts = Counter(predicted_labels)
                assert predicted_by_round is not None
                predicted_by_round.append(predicted_labels)

            runtime.model = _apply_aggregated_gradient(
                runtime.model,
                aggregated_gradient,
                learning_rate=L2_LEARNING_RATE,
            )
            after_main, after_corner = _evaluate_runtime(runtime)
            selected_role_counts = _role_counts_for_indices(env_round, selected_indices)

            round_records.append(
                _round_record(
                    round_index=round_index,
                    env_round=env_round,
                    method_id=method_id,
                    method_label=method_label,
                    main_metrics=after_main,
                    corner_metrics=after_corner,
                    selected_role_counts=selected_role_counts,
                    predicted_role_counts=predicted_counts,
                    fraud_total=fraud_total,
                )
            )

            if method_id == "flpg_adaptive":
                telemetry = _build_adaptive_telemetry(
                    round_id=reference_policy.round_id + round_index,
                    env_round=env_round,
                    selected_role_counts=selected_role_counts,
                    predicted_labels=predicted_by_round[-1],
                    before_main=before_main,
                    after_main=after_main,
                    before_corner=before_corner,
                    after_corner=after_corner,
                )

                async def _next_policy() -> Policy:
                    rule_proposal = await rule_engine.propose(eval_policy, telemetry)
                    checked = await safety_guard.check(rule_proposal, telemetry)
                    if checked.safety_guard_passed:
                        return checked.proposed_policy
                    return clone_policy(eval_policy, round_id=eval_policy.round_id + 1)

                current_policy = asyncio.run(_next_policy())

        summary = {
            "main_accuracy_avg": mean(float(item["main_accuracy"]) for item in round_records),
            "corner_accuracy_avg": mean(float(item["corner_accuracy"]) for item in round_records),
            "fraud_survival_rate_avg": mean(float(item["fraud_survival_rate"]) for item in round_records),
            "fraud_survival_rate_overall": (
                sum(int(item["selected_fraud_count"]) for item in round_records)
                / max(sum(int(item["round_fraud_total"]) for item in round_records), 1)
            ),
            "selected_total_avg": mean(float(item["selected_total"]) for item in round_records),
        }
        classification_summary = _classification_summary(
            predicted_by_round=predicted_by_round,
            rounds=environment_rounds,
        )
        if classification_summary is not None:
            summary["fraud_precision"] = classification_summary["FRAUD"]["precision"]
            summary["fraud_recall"] = classification_summary["FRAUD"]["recall"]
            summary["rarity_precision"] = classification_summary["RARITY"]["precision"]
            summary["rarity_recall"] = classification_summary["RARITY"]["recall"]

        results_by_method[method_id] = {
            "id": method_id,
            "label": method_label,
            "summary": summary,
            "phase_breakdown": _phase_breakdown(round_records),
            "classification_summary": classification_summary,
            "round_records": round_records,
        }

    fedavg_summary = results_by_method["fedavg"]["summary"]
    comparison = {}
    for method_id, payload in results_by_method.items():
        if method_id == "fedavg":
            continue
        summary = payload["summary"]
        comparison[method_id] = {
            "main_accuracy_delta_pp": (summary["main_accuracy_avg"] - fedavg_summary["main_accuracy_avg"]) * 100,
            "corner_accuracy_delta_pp": (summary["corner_accuracy_avg"] - fedavg_summary["corner_accuracy_avg"]) * 100,
            "fraud_survival_delta_pp": (summary["fraud_survival_rate_overall"] - fedavg_summary["fraud_survival_rate_overall"]) * 100,
        }
        if "fraud_recall" in summary:
            comparison[method_id]["fraud_recall_pp"] = summary["fraud_recall"] * 100
        if "rarity_recall" in summary:
            comparison[method_id]["rarity_recall_pp"] = summary["rarity_recall"] * 100

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "rounds": rounds,
            "cycle_rounds": cycle_rounds,
            "clients_per_round": BATCH_SIZE,
            "vehicle_pool_size": VEHICLE_POOL_SIZE,
            "phase_cycle": BENCHMARK_PHASE_CYCLE,
            "krum_byzantine_budget": krum_byzantine_budget,
            "policy_source": "default_policy" if policy is None else "custom_policy",
            "initial_checkpoint": checkpoint_info,
            "d_proto_main_size": len(eval_bundle.proto_main),
            "d_proto_corner_size": len(eval_bundle.proto_corner),
            "d_audit_main_size": len(eval_bundle.audit_main),
            "d_audit_corner_size": len(eval_bundle.audit_corner),
            "d_oracle_main_size": len(eval_bundle.oracle_main),
            "d_oracle_corner_size": len(eval_bundle.oracle_corner),
            "first_cycle_ground_truth_counts": dict(base_cycle_counts),
            "overall_ground_truth_counts": dict(overall_ground_truth_counts),
        },
        "comparison_vs_fedavg": comparison,
        "methods": results_by_method,
    }

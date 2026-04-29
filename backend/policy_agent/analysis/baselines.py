"""Backend-driven baseline evaluation for the Data Analysis page."""

from __future__ import annotations

import copy
import random
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = PROJECT_ROOT / "backend"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

for candidate in (PROJECT_ROOT, BACKEND_DIR, SCRIPTS_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from common.config import L2_LEARNING_RATE  # noqa: E402
from common.schemas import DEFAULT_POLICY, Policy, RoundTelemetry  # noqa: E402
from generate_demo_data import DemoDataGenerator  # noqa: E402
from l1_linear_defense.aggregation import filter_suspects  # noqa: E402
from l2_dual_audit.classifier import DualChannelAuditor  # noqa: E402
from policy_agent.analysis.unified_benchmark import (  # noqa: E402
    EvalBundle,
    _make_generator,
    build_eval_bundle,
    pretrain_initial_checkpoint,
)
from policy_agent.constraints.safety_guard import SafetyGuard  # noqa: E402
from policy_agent.engine.rule_engine import RuleEngine  # noqa: E402


TARGET_LABELS = ("RARITY", "FRAUD")


@dataclass
class RoleMetrics:
    total: int = 0
    true_positive: int = 0
    predicted: int = 0

    @property
    def recall(self) -> float:
        return self.true_positive / self.total if self.total else 0.0

    @property
    def precision(self) -> float:
        return self.true_positive / self.predicted if self.predicted else 0.0


@dataclass
class EnvironmentRound:
    round_id: int
    round_index: int
    phase: str
    updates: list[dict[str, Any]]
    gradients: list[np.ndarray]
    vehicle_ids: list[str]
    planned_roles: Counter
    new_vehicle_count: int


@dataclass
class ModelMetrics:
    main_loss: float
    main_accuracy: float
    corner_loss: float
    corner_accuracy: float


@dataclass
class StrategyRuntime:
    model: nn.Module
    main_loader: torch.utils.data.DataLoader
    corner_loader: torch.utils.data.DataLoader
    criterion: nn.Module


def clone_policy(policy: Policy, round_id: int | None = None) -> Policy:
    """Create a detached copy and optionally update the round metadata."""
    payload = policy.model_dump()
    if round_id is not None:
        payload["round_id"] = round_id
        payload["effective_from_round"] = round_id
    return Policy.model_validate(payload)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _make_strategy_runtime(initial_model: nn.Module, eval_bundle: EvalBundle) -> StrategyRuntime:
    return StrategyRuntime(
        model=copy.deepcopy(initial_model),
        main_loader=torch.utils.data.DataLoader(
            eval_bundle.oracle_main,
            batch_size=32,
            shuffle=False,
        ),
        corner_loader=torch.utils.data.DataLoader(
            eval_bundle.oracle_corner,
            batch_size=32,
            shuffle=False,
        ),
        criterion=nn.CrossEntropyLoss(),
    )


def _build_environment_rounds(
    scenario_policy: Policy,
    rounds: int,
) -> tuple[nn.Module, EvalBundle, DemoDataGenerator, list[EnvironmentRound], dict[str, Any]]:
    initial_model, checkpoint_info = pretrain_initial_checkpoint(
        epochs=5,
        batch_size=64,
        learning_rate=1e-3,
    )
    eval_bundle = build_eval_bundle()
    generator = _make_generator(initial_model=initial_model, eval_bundle=eval_bundle)
    generator.ground_truth_mode = "archetype"
    environment: list[EnvironmentRound] = []

    for round_index in range(rounds):
        round_id = scenario_policy.round_id + round_index
        round_policy = clone_policy(scenario_policy, round_id=round_id)
        (
            updates,
            planned_roles,
            vehicle_addresses,
            new_vehicle_count,
            _preflight_counts,
            _l1_projection,
        ) = generator._build_batch(
            round_id=round_id,
            policy=round_policy,
            round_index=round_index,
        )
        gradients = [np.array(payload["gradient_data"], dtype=float) for payload in updates]
        environment.append(
            EnvironmentRound(
                round_id=round_id,
                round_index=round_index,
                phase=generator._phase_name(round_index),
                updates=updates,
                gradients=gradients,
                vehicle_ids=vehicle_addresses,
                planned_roles=planned_roles,
                new_vehicle_count=new_vehicle_count,
            )
        )

    return initial_model, eval_bundle, generator, environment, checkpoint_info


def _make_auditor(eval_bundle: EvalBundle, model: nn.Module) -> DualChannelAuditor:
    return DualChannelAuditor(
        model=copy.deepcopy(model),
        main_dataset=eval_bundle.audit_main,
        corner_dataset=eval_bundle.audit_corner,
    )


def _evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
) -> tuple[float, float]:
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
        return 0.0, 0.0

    return total_loss / total_samples, total_correct / total_samples


def _compute_model_metrics(runtime: StrategyRuntime) -> ModelMetrics:
    main_loss, main_accuracy = _evaluate_model(
        runtime.model,
        runtime.main_loader,
        runtime.criterion,
    )
    corner_loss, corner_accuracy = _evaluate_model(
        runtime.model,
        runtime.corner_loader,
        runtime.criterion,
    )
    return ModelMetrics(
        main_loss=main_loss,
        main_accuracy=main_accuracy,
        corner_loss=corner_loss,
        corner_accuracy=corner_accuracy,
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
        raise ValueError(
            "Aggregated gradient dimension does not match model parameter dimension"
        )

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


def _summarize_round(
    env_round: EnvironmentRound,
    included_indices: set[int],
    predicted_labels: dict[str, str],
    eval_policy: Policy,
    before_metrics: ModelMetrics,
    after_metrics: ModelMetrics,
) -> dict[str, float | int | str]:
    total = max(len(env_round.updates), 1)
    included_roles = Counter(
        env_round.updates[idx]["metadata"].get(
            "ground_truth_label",
            env_round.updates[idx]["metadata"].get(
                "ground_truth_role",
                env_round.updates[idx]["metadata"]["planned_role"],
            ),
        )
        for idx in included_indices
    )
    total_rarity = sum(
        1
        for payload in env_round.updates
        if payload["metadata"].get(
            "ground_truth_label",
            payload["metadata"].get("ground_truth_role", payload["metadata"]["planned_role"]),
        ) == "RARITY"
    )
    honest_false_slash = sum(
        1
        for payload in env_round.updates
        if payload["metadata"].get(
            "ground_truth_label",
            payload["metadata"].get("ground_truth_role", payload["metadata"]["planned_role"]),
        ) == "HONEST"
        and predicted_labels.get(payload["vehicle_address"]) == "FRAUD"
    )

    fraud_rate = included_roles.get("FRAUD", 0) / total
    rarity_rate = included_roles.get("RARITY", 0) / total
    honest_rate = included_roles.get("HONEST", 0) / total
    noise_rate = included_roles.get("NOISE", 0) / total

    return {
        "round_id": env_round.round_id,
        "phase": env_round.phase,
        "main_accuracy": after_metrics.main_accuracy,
        "corner_accuracy": after_metrics.corner_accuracy,
        "false_slash_estimate": honest_false_slash / total,
        "rarity_retention_rate": (
            included_roles.get("RARITY", 0) / total_rarity if total_rarity else 1.0
        ),
        "fraud_rate": fraud_rate,
        "rarity_rate": rarity_rate,
        "honest_rate": honest_rate,
        "noise_rate": noise_rate,
        "main_loss_delta_avg": after_metrics.main_loss - before_metrics.main_loss,
        "corner_loss_delta_avg": after_metrics.corner_loss - before_metrics.corner_loss,
        "theta_rare": eval_policy.theta_rare,
        "rarity_reward_multiplier": eval_policy.rarity_reward_multiplier,
        "slash_multiplier": eval_policy.slash_multiplier,
        "corner_weight": eval_policy.corner_weight,
    }


async def _evaluate_strategy(
    *,
    strategy_id: str,
    label: str,
    description: str,
    scenario_policy: Policy,
    rounds: list[EnvironmentRound],
    generator: DemoDataGenerator,
    initial_model: nn.Module,
    eval_bundle: EvalBundle,
) -> dict[str, Any]:
    metrics_by_role = {target: RoleMetrics() for target in TARGET_LABELS}
    round_results: list[dict[str, Any]] = []
    rule_engine = RuleEngine()
    safety_guard = SafetyGuard()
    runtime = _make_strategy_runtime(initial_model, eval_bundle)
    recheck_rng = random.Random(20260428 + sum(ord(char) for char in strategy_id))

    if strategy_id == "adaptive":
        current_policy = clone_policy(scenario_policy)
    else:
        current_policy = clone_policy(DEFAULT_POLICY)

    for env_round in rounds:
        if strategy_id == "adaptive":
            eval_policy = clone_policy(current_policy, round_id=env_round.round_id)
        elif strategy_id in {"l1_only", "static_l2"}:
            eval_policy = clone_policy(DEFAULT_POLICY, round_id=env_round.round_id)
        else:
            eval_policy = clone_policy(DEFAULT_POLICY, round_id=env_round.round_id)

        before_metrics = _compute_model_metrics(runtime)
        predicted_labels: dict[str, str] = {}
        included_indices: set[int] = set()
        aggregated_gradient: np.ndarray

        if strategy_id == "fedavg":
            for idx, vehicle_id in enumerate(env_round.vehicle_ids):
                predicted_labels[vehicle_id] = "ACCEPTED"
                included_indices.add(idx)
            aggregated_gradient = _mean_gradient(env_round.gradients)
        else:
            l1_result = filter_suspects(
                env_round.gradients,
                env_round.vehicle_ids,
                threshold=eval_policy.cosine_filter_threshold,
                recheck_probability=eval_policy.recheck_probability,
                rng=recheck_rng,
            )
            suspect_indices = set(l1_result.suspect_indices)
            clean_indices = set(range(len(env_round.updates))) - suspect_indices

            for idx in clean_indices:
                vehicle_id = env_round.vehicle_ids[idx]
                predicted_labels[vehicle_id] = "HONEST"
                included_indices.add(idx)

            if strategy_id == "l1_only":
                for idx in suspect_indices:
                    predicted_labels[env_round.vehicle_ids[idx]] = "FRAUD"
            else:
                auditor = _make_auditor(eval_bundle, runtime.model)
                auditor.apply_policy(eval_policy)
                for idx in suspect_indices:
                    audit = auditor.audit(env_round.vehicle_ids[idx], env_round.gradients[idx])
                    predicted_labels[env_round.vehicle_ids[idx]] = audit.classification.value
                    if audit.include_in_aggregation:
                        included_indices.add(idx)

            if included_indices:
                aggregated_gradient = _mean_gradient(
                    [env_round.gradients[idx] for idx in sorted(included_indices)]
                )
            elif strategy_id == "l1_only":
                aggregated_gradient = l1_result.aggregated_gradient.copy()
            else:
                aggregated_gradient = np.zeros_like(env_round.gradients[0])

        runtime.model = _apply_aggregated_gradient(
            runtime.model,
            aggregated_gradient,
            learning_rate=L2_LEARNING_RATE,
        )
        after_metrics = _compute_model_metrics(runtime)

        for payload in env_round.updates:
            vehicle_id = payload["vehicle_address"]
            ground_truth = payload["metadata"].get(
                "ground_truth_label",
                payload["metadata"].get("ground_truth_role", payload["metadata"]["planned_role"]),
            )
            predicted = predicted_labels.get(vehicle_id, "ACCEPTED")
            if ground_truth in TARGET_LABELS:
                metrics_by_role[ground_truth].total += 1
                if predicted == ground_truth:
                    metrics_by_role[ground_truth].true_positive += 1
            if predicted in TARGET_LABELS:
                metrics_by_role[predicted].predicted += 1

        round_result = _summarize_round(
            env_round=env_round,
            included_indices=included_indices,
            predicted_labels=predicted_labels,
            eval_policy=eval_policy,
            before_metrics=before_metrics,
            after_metrics=after_metrics,
        )
        round_results.append(round_result)

        if strategy_id == "adaptive":
            telemetry = RoundTelemetry(
                round_id=env_round.round_id,
                fraud_rate=float(round_result["fraud_rate"]),
                rarity_rate=float(round_result["rarity_rate"]),
                honest_rate=float(round_result["honest_rate"]),
                noise_rate=float(round_result["noise_rate"]),
                main_accuracy=float(round_result["main_accuracy"]),
                corner_accuracy=float(round_result["corner_accuracy"]),
                main_loss_delta_avg=float(round_result["main_loss_delta_avg"]),
                corner_loss_delta_avg=float(round_result["corner_loss_delta_avg"]),
                false_slash_estimate=float(round_result["false_slash_estimate"]),
                rarity_retention_rate=float(round_result["rarity_retention_rate"]),
                golden_drift_score=_clamp(
                    abs(float(round_result["main_loss_delta_avg"])) * 1.4
                    + float(round_result["fraud_rate"]) * 0.6
                    + float(round_result["noise_rate"]) * 0.25,
                    0.0,
                    1.0,
                ),
                reject_rate_l3=0.0,
                cosine_outlier_ratio=(
                    len(
                        [
                            predicted
                            for predicted in predicted_labels.values()
                            if predicted in {"FRAUD", "RARITY", "NOISE"}
                        ]
                    )
                    / max(len(env_round.updates), 1)
                ),
                suspect_queue_length=0,
                audit_sample_size=len(predicted_labels),
                avg_sbt_score=0.0,
                new_vehicle_ratio=env_round.new_vehicle_count / max(len(env_round.updates), 1),
                hash_mismatch_rate=0.0,
                recent_attack_pressure=_clamp(
                    float(round_result["fraud_rate"]) * 1.2
                    + float(round_result["noise_rate"]) * 0.5
                    + (
                        env_round.planned_roles.get("FRAUD", 0)
                        + env_round.planned_roles.get("FRAUD_CORNER_HARM", 0)
                    ) / max(len(env_round.updates), 1) * 0.2,
                    0.0,
                    1.0,
                ),
            )
            proposal = await rule_engine.propose(eval_policy, telemetry)
            proposal = await safety_guard.check(proposal, telemetry)
            if proposal.safety_guard_passed:
                current_policy = proposal.proposed_policy
            else:
                current_policy = clone_policy(eval_policy, round_id=env_round.round_id + 1)

    summary = {
        "main_accuracy_avg": mean(float(round_item["main_accuracy"]) for round_item in round_results),
        "corner_accuracy_avg": mean(float(round_item["corner_accuracy"]) for round_item in round_results),
        "false_slash_estimate_avg": mean(float(round_item["false_slash_estimate"]) for round_item in round_results),
        "rarity_retention_rate_avg": mean(float(round_item["rarity_retention_rate"]) for round_item in round_results),
        "rarity_precision": metrics_by_role["RARITY"].precision,
        "rarity_recall": metrics_by_role["RARITY"].recall,
        "fraud_precision": metrics_by_role["FRAUD"].precision,
        "fraud_recall": metrics_by_role["FRAUD"].recall,
    }

    return {
        "id": strategy_id,
        "label": label,
        "description": description,
        "summary": summary,
        "rounds": round_results,
    }


async def build_baseline_analysis(
    current_policy: Policy | None,
    *,
    rounds: int = 12,
) -> dict[str, Any]:
    """Build the backend-evaluated baseline comparison payload."""
    scenario_policy = clone_policy(current_policy or DEFAULT_POLICY)
    scenario_source = "live_policy" if current_policy is not None else "default_policy"
    initial_model, eval_bundle, generator, environment_rounds, checkpoint_info = (
        _build_environment_rounds(scenario_policy, rounds)
    )

    baselines = [
        await _evaluate_strategy(
            strategy_id="fedavg",
            label="FedAvg",
            description="Arithmetic mean over all client gradients on the same simulated rounds, with no screening or auditing.",
            scenario_policy=scenario_policy,
            rounds=environment_rounds,
            generator=generator,
            initial_model=initial_model,
            eval_bundle=eval_bundle,
        ),
        await _evaluate_strategy(
            strategy_id="l1_only",
            label="L1 only",
            description="Cosine filtering routes suspects out, but cannot recover beneficial rarity.",
            scenario_policy=scenario_policy,
            rounds=environment_rounds,
            generator=generator,
            initial_model=initial_model,
            eval_bundle=eval_bundle,
        ),
        await _evaluate_strategy(
            strategy_id="static_l2",
            label="L1 + L2 static policy",
            description="Fixed default thresholds with no round-to-round adaptation.",
            scenario_policy=scenario_policy,
            rounds=environment_rounds,
            generator=generator,
            initial_model=initial_model,
            eval_bundle=eval_bundle,
        ),
        await _evaluate_strategy(
            strategy_id="adaptive",
            label="Full FLPG (adaptive)",
            description="Round-by-round policy adjustment using the backend policy engine.",
            scenario_policy=scenario_policy,
            rounds=environment_rounds,
            generator=generator,
            initial_model=initial_model,
            eval_bundle=eval_bundle,
        ),
    ]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scenario_policy_source": scenario_source,
        "scenario_policy_round": scenario_policy.round_id,
        "classification_rounds": rounds,
        "ground_truth_mode": generator.ground_truth_mode,
        "dataset_isolation": {
            "D_proto_main": len(eval_bundle.proto_main),
            "D_proto_corner": len(eval_bundle.proto_corner),
            "D_audit_main": len(eval_bundle.audit_main),
            "D_audit_corner": len(eval_bundle.audit_corner),
            "D_oracle_main": len(eval_bundle.oracle_main),
            "D_oracle_corner": len(eval_bundle.oracle_corner),
        },
        "initial_checkpoint": checkpoint_info,
        "baselines": baselines,
    }

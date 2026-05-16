#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import copy
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

for candidate in (PROJECT_ROOT, BACKEND_DIR, SCRIPTS_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from common.config import L2_LEARNING_RATE  # noqa: E402
from common.schemas import DEFAULT_POLICY, Policy  # noqa: E402
from generate_demo_data import DemoDataGenerator  # noqa: E402
from l1_linear_defense.aggregation import (  # noqa: E402
    cosine_similarity,
    filter_suspects,
    geometric_median,
)
from l1_linear_defense.config import L1RouterConfig, make_l1_router_config  # noqa: E402
from policy_agent.analysis.unified_benchmark import (  # noqa: E402
    BENCHMARK_PHASE_CYCLE,
    PRETRAIN_SEED,
    _apply_aggregated_gradient,
    _evaluate_runtime,
    _make_auditor,
    _make_generator,
    _make_runtime,
    _mean_gradient,
    _multi_krum_indices,
    _role_counts_for_indices,
    _build_adaptive_telemetry,
    build_eval_bundle,
    clone_policy,
    pretrain_initial_checkpoint,
)
from policy_agent.constraints.safety_guard import SafetyGuard  # noqa: E402
from policy_agent.engine.rule_engine import RuleEngine  # noqa: E402


THETA_CORNER_HARM_PROXY = 0.0
THETA_CORNER_HARM_SOURCE = "analysis_proxy_not_runtime_policy"


@dataclass
class RoundBundle:
    round_id: int
    cycle_round_index: int
    phase: str
    updates: list[dict[str, Any]]
    gradients: list[np.ndarray]
    sample_counts: list[int]
    role_by_index: list[str]
    planned_role_by_index: list[str]
    vehicle_ids: list[str]
    rarity_generation_trace: list[dict[str, Any]]

    @property
    def role_counts(self) -> Counter:
        return Counter(self.role_by_index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export thesis-oriented FLPG benchmark artifacts as JSON/CSV/TXT."
    )
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--cycle-rounds", type=int, default=12)
    parser.add_argument("--pretrain-epochs", type=int, default=5)
    parser.add_argument("--pretrain-batch-size", type=int, default=64)
    parser.add_argument("--pretrain-learning-rate", type=float, default=1e-3)
    parser.add_argument("--krum-byzantine-budget", type=int, default=10)
    parser.add_argument(
        "--recheck-probability",
        type=float,
        default=None,
        help="Override policy recheck_probability for the FLPG artifact run.",
    )
    parser.add_argument(
        "--l1-router-mode",
        type=str,
        default="cosine_recheck",
        help="L1 router mode: cosine_recheck or dual_proxy_budgeted.",
    )
    parser.add_argument(
        "--l1-queue-budget-ratio",
        "--queue-budget-ratio",
        dest="l1_queue_budget_ratio",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--l1-random-recheck-ratio",
        "--random-recheck-ratio",
        dest="l1_random_recheck_ratio",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "thesis_artifacts",
    )
    return parser.parse_args()


def update_l1_client_state(
    states: dict[str, dict[str, Any]],
    *,
    vehicle_id: str,
    round_id: int,
    routed: bool,
    verdict: str,
) -> None:
    state = states.setdefault(
        vehicle_id,
        {
            "reputation": 1.0,
            "fraud_count": 0,
            "recent_fraud_count": 0,
            "last_audit_round": None,
        },
    )
    state["last_seen_round"] = round_id
    if not routed:
        return

    state["last_audit_round"] = round_id
    if verdict == "FRAUD":
        state["fraud_count"] = int(state.get("fraud_count", 0)) + 1
        state["recent_fraud_count"] = int(state.get("recent_fraud_count", 0)) + 1
        state["reputation"] = max(0.0, float(state.get("reputation", 1.0)) - 0.20)
    elif verdict == "NOISE":
        state["recent_fraud_count"] = max(0, int(state.get("recent_fraud_count", 0)) - 1)
        state["reputation"] = max(0.0, float(state.get("reputation", 1.0)) - 0.03)
    else:
        state["recent_fraud_count"] = max(0, int(state.get("recent_fraud_count", 0)) - 1)
        state["reputation"] = min(1.0, float(state.get("reputation", 1.0)) + 0.02)


def policy_with_recheck_probability(policy: Policy, probability: float) -> Policy:
    payload = policy.model_dump()
    payload["recheck_probability"] = probability
    return Policy.model_validate(payload)


def build_round_bundles(
    *,
    policy: Policy,
    total_rounds: int,
    cycle_rounds: int,
    generator: DemoDataGenerator,
) -> tuple[list[RoundBundle], Counter]:
    generator._phase_name = lambda round_index: BENCHMARK_PHASE_CYCLE[round_index % len(BENCHMARK_PHASE_CYCLE)]  # type: ignore[method-assign]
    generator.ground_truth_mode = "archetype"

    rounds: list[RoundBundle] = []
    overall_counts: Counter = Counter()
    _ = cycle_rounds
    for round_index in range(total_rounds):
        round_policy = clone_policy(policy, round_id=policy.round_id + round_index)
        (
            updates,
            planned_roles,
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
        planned_role_by_index = [str(payload["metadata"]["planned_role"]) for payload in updates]
        overall_counts.update(role_by_index)
        rounds.append(
            RoundBundle(
                round_id=round_policy.round_id,
                cycle_round_index=round_index % len(BENCHMARK_PHASE_CYCLE),
                phase=BENCHMARK_PHASE_CYCLE[round_index % len(BENCHMARK_PHASE_CYCLE)],
                updates=copy.deepcopy(updates),
                gradients=gradients,
                sample_counts=sample_counts,
                role_by_index=role_by_index,
                planned_role_by_index=planned_role_by_index,
                vehicle_ids=list(vehicle_addresses),
                rarity_generation_trace=copy.deepcopy(generator.current_rarity_generation_trace),
            )
        )
        _ = planned_roles, preflight_counts

    return rounds, overall_counts


def ordered_fieldnames(requested: list[str], rows: list[dict[str, Any]]) -> list[str]:
    seen = set(requested)
    extras = []
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                extras.append(key)
    return requested + extras


def write_csv(path: Path, requested_fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ordered_fieldnames(requested_fieldnames, rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def compute_round_cosines(round_bundle: RoundBundle) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    mean_vector = np.mean(np.stack(round_bundle.gradients), axis=0)
    geomed_vector, _ = geometric_median(round_bundle.gradients)
    grouped: dict[str, list[float]] = defaultdict(list)
    for role, gradient in zip(round_bundle.role_by_index, round_bundle.gradients):
        grouped[role].append(cosine_similarity(gradient, mean_vector))
    return mean_vector, geomed_vector, {
        "HONEST": safe_mean(grouped.get("HONEST", [])),
        "FRAUD": safe_mean(grouped.get("FRAUD", [])),
        "RARITY": safe_mean(grouped.get("RARITY", [])),
        "NOISE": safe_mean(grouped.get("NOISE", [])),
    }


def classify_shadow_audit(
    *,
    auditor,
    vehicle_id: str,
    gradient: np.ndarray,
    base_main_loss: float,
    base_corner_loss: float,
    audit_mode: str = "dual",
    theta_corner_harm_proxy: float | None = None,
) -> dict[str, Any]:
    corner_harm_threshold = (
        THETA_CORNER_HARM_PROXY
        if theta_corner_harm_proxy is None
        else float(theta_corner_harm_proxy)
    )
    updated_model = auditor.apply_gradient(gradient)
    main_after = auditor.compute_loss(updated_model, auditor.main_loader)
    corner_after = auditor.compute_loss(updated_model, auditor.corner_loader)
    delta_main = main_after - base_main_loss
    delta_corner = corner_after - base_corner_loss

    if audit_mode == "main_only":
        if delta_main <= auditor.fraud_threshold:
            verdict = "HONEST"
            include = True
            action = "accept_standard"
        else:
            verdict = "FRAUD"
            include = False
            action = "reject_fraud"
    elif audit_mode == "corner_only":
        if delta_corner <= auditor.rarity_threshold:
            verdict = "RARITY"
            include = True
            action = "accept_rarity"
        elif delta_corner > corner_harm_threshold:
            verdict = "FRAUD"
            include = False
            action = "reject_corner_harm"
        else:
            verdict = "NOISE"
            include = False
            action = "reject_noise"
    elif delta_main > auditor.fraud_threshold:
        verdict = "FRAUD"
        include = False
        action = "reject_fraud"
    elif delta_corner <= auditor.rarity_threshold and delta_main <= auditor.rarity_main_threshold:
        verdict = "RARITY"
        include = True
        action = "accept_rarity"
    elif delta_main <= 0 and delta_corner > corner_harm_threshold:
        verdict = "FRAUD"
        include = False
        action = "reject_corner_harm"
    elif delta_main < 0:
        verdict = "HONEST"
        include = True
        action = "accept_standard"
    else:
        verdict = "NOISE"
        include = False
        action = "reject_noise"

    final_score = abs(delta_main) + 0.5 * auditor.corner_weight * max(0.0, -delta_corner)
    return {
        "vehicle_id": vehicle_id,
        "delta_l_main": delta_main,
        "delta_l_corner": delta_corner,
        "verdict": verdict,
        "include_in_aggregation": include,
        "action": action,
        "final_score": final_score,
    }


def policy_update_reason(current: Policy, proposal, checked) -> str:
    reasons = list(checked.reasons or proposal.reasons or [])
    if checked.blocked_reasons:
        reasons.append("blocked: " + " | ".join(checked.blocked_reasons))
    diff_parts = []
    for field, (before, after) in checked.get_diff().items():
        diff_parts.append(f"{field}={before}->{after}")
    if diff_parts:
        reasons.append("changes: " + ", ".join(diff_parts))
    return " | ".join(reasons) if reasons else "no policy change"


async def propose_next_policy(current_policy: Policy, telemetry) -> tuple[Any, Any]:
    proposal = await RuleEngine().propose(current_policy, telemetry)
    checked = await SafetyGuard().check(proposal, telemetry)
    return proposal, checked


def run_flpg_with_artifacts(
    *,
    rounds: list[RoundBundle],
    initial_model,
    eval_bundle,
    reference_policy: Policy,
    fixed_recheck_probability: float | None = None,
    adapt_policy: bool = True,
    audit_mode: str = "dual",
    recheck_seed: int = 20260428,
    compute_oracle_drift: bool = True,
    l1_router_config: L1RouterConfig | None = None,
    theta_corner_harm_proxy: float | None = None,
    force_exhaustive_l2: bool = False,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    runtime = _make_runtime(initial_model, eval_bundle)
    current_policy = clone_policy(reference_policy)
    corner_harm_threshold = (
        THETA_CORNER_HARM_PROXY
        if theta_corner_harm_proxy is None
        else float(theta_corner_harm_proxy)
    )

    round_summary_rows: list[dict[str, Any]] = []
    l1_rows: list[dict[str, Any]] = []
    l2_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []

    blind_spot_observed = 0
    routed_non_honest = 0
    routed_total = 0
    routed_by_label = Counter()
    total_by_label = Counter()
    fraud_total_by_family = Counter()
    fraud_survival_by_family = Counter()
    fraud_caught_by_routing_reason = Counter()
    routed_by_reason = Counter()
    recheck_rng = random.Random(recheck_seed)
    l1_client_states: dict[str, dict[str, Any]] = {}

    for round_bundle in rounds:
        before_main, before_corner = _evaluate_runtime(runtime)
        eval_policy = clone_policy(current_policy, round_id=round_bundle.round_id)
        if fixed_recheck_probability is not None:
            eval_policy = policy_with_recheck_probability(
                eval_policy,
                fixed_recheck_probability,
            )
        mean_vector, geomed_vector, cosine_means = compute_round_cosines(round_bundle)

        l1_result = filter_suspects(
            round_bundle.gradients,
            round_bundle.vehicle_ids,
            threshold=eval_policy.cosine_filter_threshold,
            recheck_probability=eval_policy.recheck_probability,
            rng=recheck_rng,
            router_config=l1_router_config,
            client_states=l1_client_states,
            current_round=round_bundle.round_id,
        )
        if force_exhaustive_l2:
            l1_result.clean_indices = []
            l1_result.suspect_indices = list(range(len(round_bundle.gradients)))
            l1_result.routing_reasons = {
                idx: "exhaustive_l2_audit"
                for idx in l1_result.suspect_indices
            }
        suspect_indices = set(l1_result.suspect_indices)
        selected_indices = set(range(len(round_bundle.gradients))) - suspect_indices
        predicted_labels = ["HONEST" for _ in round_bundle.gradients]

        auditor = _make_auditor(
            runtime.model,
            main_dataset=eval_bundle.audit_main,
            corner_dataset=eval_bundle.audit_corner,
        )
        auditor.apply_policy(eval_policy)
        base_main_loss = auditor.compute_loss(auditor.model, auditor.main_loader)
        base_corner_loss = auditor.compute_loss(auditor.model, auditor.corner_loader)
        oracle_auditor = None
        oracle_base_main_loss = None
        oracle_base_corner_loss = None
        if compute_oracle_drift:
            oracle_auditor = _make_auditor(
                runtime.model,
                main_dataset=eval_bundle.oracle_main,
                corner_dataset=eval_bundle.oracle_corner,
            )
            oracle_auditor.apply_policy(eval_policy)
            oracle_base_main_loss = oracle_auditor.compute_loss(
                oracle_auditor.model,
                oracle_auditor.main_loader,
            )
            oracle_base_corner_loss = oracle_auditor.compute_loss(
                oracle_auditor.model,
                oracle_auditor.corner_loader,
            )

        actual_action_counts = Counter()
        honest_safe_count = 0
        fraud_survival_count = 0

        for idx, gradient in enumerate(round_bundle.gradients):
            vehicle_id = round_bundle.vehicle_ids[idx]
            true_label = round_bundle.role_by_index[idx]
            planned_role = round_bundle.planned_role_by_index[idx]
            metadata = round_bundle.updates[idx]["metadata"]
            attack_family = str(metadata.get("attack_family", "none"))
            rarity_subtype = str(metadata.get("rarity_subtype", ""))
            stress_experiment = str(metadata.get("stress_experiment", ""))
            stress_config = str(metadata.get("stress_config", ""))
            routed = idx in suspect_indices
            routing_reason = l1_result.routing_reasons.get(
                idx,
                "cosine_screening" if routed else "bypass",
            )
            l1_detail = l1_result.l1_score_details.get(idx, {})

            total_by_label[true_label] += 1
            if true_label == "FRAUD" and attack_family != "none":
                fraud_total_by_family[attack_family] += 1

            cosine_to_mean = cosine_similarity(gradient, mean_vector)
            cosine_to_geomed = cosine_similarity(gradient, geomed_vector)
            deviation_score = 1.0 - cosine_to_geomed
            if routed:
                routed_total += 1
                routed_by_reason[routing_reason] += 1
                routed_by_label[true_label] += 1
                if true_label != "HONEST":
                    routed_non_honest += 1

            l1_rows.append({
                "round_id": round_bundle.round_id,
                "client_id": vehicle_id,
                "true_label": true_label,
                "planned_role": planned_role,
                "rarity_subtype": rarity_subtype,
                "stress_experiment": stress_experiment,
                "stress_config": stress_config,
                "cosine_to_mean": cosine_to_mean,
                "cosine_to_geomed": cosine_to_geomed,
                "deviation_score_geomed": deviation_score,
                "tau_screen": eval_policy.cosine_filter_threshold,
                "recheck_probability": eval_policy.recheck_probability,
                "routed_to_l2": routed,
                "routing_reason": routing_reason,
                "l1_router_mode": l1_result.router_mode,
                "l1_risk_score": l1_detail.get("risk_score", ""),
                "l1_norm_mad_score": l1_detail.get("norm_mad_score", ""),
                "l1_sign_disagreement": l1_detail.get("sign_disagreement", ""),
                "l1_reputation_risk": l1_detail.get("reputation_risk", ""),
                "l1_audit_age_score": l1_detail.get("audit_age_score", ""),
                "phase_name": round_bundle.phase,
            })

            shadow = classify_shadow_audit(
                auditor=auditor,
                vehicle_id=vehicle_id,
                gradient=gradient,
                base_main_loss=base_main_loss,
                base_corner_loss=base_corner_loss,
                audit_mode=audit_mode,
                theta_corner_harm_proxy=corner_harm_threshold,
            )

            if routed:
                predicted_labels[idx] = shadow["verdict"]
                if shadow["include_in_aggregation"]:
                    selected_indices.add(idx)
            actual_verdict = shadow["verdict"] if routed else "L1_CLEAN"
            actual_action = shadow["action"] if routed else "accept_standard"
            actual_included = idx in selected_indices if routed else True
            update_l1_client_state(
                l1_client_states,
                vehicle_id=vehicle_id,
                round_id=round_bundle.round_id,
                routed=routed,
                verdict=actual_verdict,
            )

            if actual_included and true_label == "FRAUD":
                fraud_survival_count += 1
                if attack_family != "none":
                    fraud_survival_by_family[attack_family] += 1
            if routed and true_label == "FRAUD" and actual_verdict == "FRAUD":
                fraud_caught_by_routing_reason[routing_reason] += 1
            if actual_action == "accept_standard" and shadow["delta_l_main"] < 0 and shadow["delta_l_corner"] <= corner_harm_threshold:
                honest_safe_count += 1
            if actual_action == "accept_standard" and shadow["delta_l_main"] < 0 and shadow["delta_l_corner"] > corner_harm_threshold:
                blind_spot_observed += 1

            actual_action_counts[actual_action] += 1
            oracle_delta_main = ""
            oracle_delta_corner = ""
            if (
                routed
                and oracle_auditor is not None
                and oracle_base_main_loss is not None
                and oracle_base_corner_loss is not None
            ):
                oracle_updated_model = oracle_auditor.apply_gradient(gradient)
                oracle_main_after = oracle_auditor.compute_loss(
                    oracle_updated_model,
                    oracle_auditor.main_loader,
                )
                oracle_corner_after = oracle_auditor.compute_loss(
                    oracle_updated_model,
                    oracle_auditor.corner_loader,
                )
                oracle_delta_main = oracle_main_after - oracle_base_main_loss
                oracle_delta_corner = oracle_corner_after - oracle_base_corner_loss

            l2_rows.append({
                "round_id": round_bundle.round_id,
                "client_id": vehicle_id,
                "true_label": true_label,
                "planned_role": planned_role,
                "attack_family": attack_family,
                "rarity_subtype": rarity_subtype,
                "stress_experiment": stress_experiment,
                "stress_config": stress_config,
                "l_main_base": base_main_loss,
                "l_corner_base": base_corner_loss,
                "delta_l_main": shadow["delta_l_main"],
                "delta_l_corner": shadow["delta_l_corner"],
                "oracle_delta_l_main": oracle_delta_main,
                "oracle_delta_l_corner": oracle_delta_corner,
                "shadow_verdict": shadow["verdict"],
                "verdict": actual_verdict,
                "action": actual_action,
                "aggregation_weight": 1.0 if actual_included else 0.0,
                "audited_in_l2": routed,
                "routing_reason": routing_reason,
                "l1_router_mode": l1_result.router_mode,
                "l1_risk_score": l1_detail.get("risk_score", ""),
                "l1_norm_mad_score": l1_detail.get("norm_mad_score", ""),
                "l1_sign_disagreement": l1_detail.get("sign_disagreement", ""),
                "l1_reputation_risk": l1_detail.get("reputation_risk", ""),
                "l1_audit_age_score": l1_detail.get("audit_age_score", ""),
                "tau_screen": eval_policy.cosine_filter_threshold,
                "recheck_probability": eval_policy.recheck_probability,
                "theta_tol": eval_policy.theta_tol,
                "theta_rare": eval_policy.theta_rare,
                "theta_rarity_main_tol": eval_policy.theta_rarity_main_tol,
                "theta_corner_harm": corner_harm_threshold,
                "phase_name": round_bundle.phase,
            })

        if selected_indices:
            aggregated_gradient = _mean_gradient(
                [round_bundle.gradients[idx] for idx in sorted(selected_indices)]
            )
        else:
            aggregated_gradient = np.zeros_like(round_bundle.gradients[0])

        runtime.model = _apply_aggregated_gradient(
            runtime.model,
            aggregated_gradient,
            learning_rate=L2_LEARNING_RATE,
        )
        after_main, after_corner = _evaluate_runtime(runtime)
        selected_role_counts = _role_counts_for_indices(
            type("EnvRoundShim", (), {"role_by_index": round_bundle.role_by_index})(),
            selected_indices,
        )

        telemetry = _build_adaptive_telemetry(
            round_id=round_bundle.round_id,
            env_round=type(
                "EnvRoundShim",
                (),
                {
                    "role_by_index": round_bundle.role_by_index,
                    "role_counts": round_bundle.role_counts,
                },
            )(),
            selected_role_counts=selected_role_counts,
            predicted_labels=predicted_labels,
            before_main=before_main,
            after_main=after_main,
            before_corner=before_corner,
            after_corner=after_corner,
        )

        if adapt_policy:
            proposal, checked = asyncio.run(propose_next_policy(eval_policy, telemetry))
            update_reason = policy_update_reason(eval_policy, proposal, checked)
            blocked = checked.blocked
        else:
            proposal = None
            checked = None
            update_reason = "policy adaptation disabled for isolated synthetic ALG benchmark"
            blocked = False
        policy_rows.append({
            "round_id": round_bundle.round_id,
            "tau_screen": eval_policy.cosine_filter_threshold,
            "theta_tol": eval_policy.theta_tol,
            "theta_rare": eval_policy.theta_rare,
            "theta_rarity_main_tol": eval_policy.theta_rarity_main_tol,
            "theta_corner_harm": corner_harm_threshold,
            "theta_corner_harm_source": THETA_CORNER_HARM_SOURCE,
            "corner_weight": eval_policy.corner_weight,
            "reward_multiplier": eval_policy.rarity_reward_multiplier,
            "honest_reward_multiplier": eval_policy.honest_reward_multiplier,
            "slash_multiplier": eval_policy.slash_multiplier,
            "recheck_probability": eval_policy.recheck_probability,
            "update_reason": update_reason,
            "blocked": blocked,
            "phase_name": round_bundle.phase,
        })

        round_summary_rows.append({
            "round_id": round_bundle.round_id,
            "phase_name": round_bundle.phase,
            "participants": len(round_bundle.gradients),
            "queue_size_l1": len(suspect_indices),
            "queue_ratio_qt": len(suspect_indices) / max(len(round_bundle.gradients), 1),
            "l1_router_mode": l1_result.router_mode,
            "accepted_standard": actual_action_counts["accept_standard"],
            "accepted_rarity": actual_action_counts["accept_rarity"],
            "rejected_fraud": actual_action_counts["reject_fraud"],
            "rejected_corner_harm": actual_action_counts["reject_corner_harm"],
            "rejected_noise": actual_action_counts["reject_noise"],
            "honest_safe_count": honest_safe_count,
            "tau_screen": eval_policy.cosine_filter_threshold,
            "theta_tol": eval_policy.theta_tol,
            "theta_rare": eval_policy.theta_rare,
            "theta_rarity_main_tol": eval_policy.theta_rarity_main_tol,
            "theta_corner_harm": corner_harm_threshold,
            "theta_corner_harm_source": THETA_CORNER_HARM_SOURCE,
            "corner_weight": eval_policy.corner_weight,
            "recheck_probability": eval_policy.recheck_probability,
            "main_task_accuracy": after_main.accuracy,
            "corner_case_accuracy": after_corner.accuracy,
        })

        baseline_rows.append({
            "round_id": round_bundle.round_id,
            "method": "flpg_adaptive",
            "fraud_survival_count": fraud_survival_count,
            "corner_case_accuracy": after_corner.accuracy,
            "main_task_accuracy": after_main.accuracy,
            "mean_cosine_honest": cosine_means["HONEST"],
            "mean_cosine_fraud": cosine_means["FRAUD"],
            "mean_cosine_rarity": cosine_means["RARITY"],
            "mean_cosine_noise": cosine_means["NOISE"],
            "phase_name": round_bundle.phase,
        })

        if adapt_policy and checked is not None and checked.safety_guard_passed:
            current_policy = checked.proposed_policy
        else:
            current_policy = clone_policy(eval_policy, round_id=eval_policy.round_id + 1)
        if fixed_recheck_probability is not None:
            current_policy = policy_with_recheck_probability(
                current_policy,
                fixed_recheck_probability,
            )

    summary = {
        "l1_routed_total": routed_total,
        "l1_routing_precision_non_honest": routed_non_honest / routed_total if routed_total else 0.0,
        "l1_routing_recall": {
            label: routed_by_label[label] / total_by_label[label] if total_by_label[label] else 0.0
            for label in ("FRAUD", "RARITY", "NOISE", "HONEST")
        },
        "blind_spot_observed_count": blind_spot_observed,
        "l1_routed_by_reason": dict(routed_by_reason),
        "fraud_caught_by_routing_reason": dict(fraud_caught_by_routing_reason),
        "fraud_survival_by_family": {
            family: fraud_survival_by_family[family] / total
            for family, total in fraud_total_by_family.items()
            if total
        },
        "fraud_total_by_family": dict(fraud_total_by_family),
        "fraud_survived_by_family": dict(fraud_survival_by_family),
    }

    return round_summary_rows, l1_rows, l2_rows, policy_rows, baseline_rows, summary


def run_baseline_method(
    *,
    method_id: str,
    rounds: list[RoundBundle],
    initial_model,
    eval_bundle,
    krum_byzantine_budget: int,
) -> list[dict[str, Any]]:
    runtime = _make_runtime(initial_model, eval_bundle)
    rows: list[dict[str, Any]] = []

    for round_bundle in rounds:
        mean_vector, _geomed_vector, cosine_means = compute_round_cosines(round_bundle)
        _ = mean_vector
        fraud_total = round_bundle.role_counts.get("FRAUD", 0)
        family_totals: Counter = Counter()

        if method_id == "fedavg":
            selected_indices = set(range(len(round_bundle.gradients)))
            aggregated_gradient = _mean_gradient(round_bundle.gradients)
        elif method_id == "geomed":
            selected_indices = set(range(len(round_bundle.gradients)))
            aggregated_gradient, _ = geometric_median(round_bundle.gradients)
        elif method_id == "krum":
            selected_indices = set(
                _multi_krum_indices(
                    round_bundle.gradients,
                    byzantine_budget=krum_byzantine_budget,
                )
            )
            aggregated_gradient = _mean_gradient(
                [round_bundle.gradients[idx] for idx in sorted(selected_indices)]
            )
        else:
            raise ValueError(f"Unsupported baseline method: {method_id}")

        runtime.model = _apply_aggregated_gradient(
            runtime.model,
            aggregated_gradient,
            learning_rate=L2_LEARNING_RATE,
        )
        after_main, after_corner = _evaluate_runtime(runtime)
        selected_role_counts = Counter(round_bundle.role_by_index[idx] for idx in selected_indices)
        family_survived: Counter = Counter()
        for idx, true_label in enumerate(round_bundle.role_by_index):
            metadata = round_bundle.updates[idx]["metadata"]
            attack_family = str(metadata.get("attack_family", "none"))
            if true_label == "FRAUD" and attack_family != "none":
                family_totals[attack_family] += 1
                if idx in selected_indices:
                    family_survived[attack_family] += 1
        rows.append({
            "round_id": round_bundle.round_id,
            "method": method_id,
            "fraud_survival_count": int(selected_role_counts.get("FRAUD", 0)),
            "fraud_survival_rate": int(selected_role_counts.get("FRAUD", 0)) / max(fraud_total, 1),
            "sign_flip_proxy_total": int(family_totals.get("sign_flip_proxy", 0)),
            "sign_flip_proxy_survival_count": int(family_survived.get("sign_flip_proxy", 0)),
            "sign_flip_proxy_survival_rate": int(family_survived.get("sign_flip_proxy", 0))
            / max(int(family_totals.get("sign_flip_proxy", 0)), 1),
            "corner_harm_total": int(family_totals.get("corner_harm", 0)),
            "corner_harm_survival_count": int(family_survived.get("corner_harm", 0)),
            "corner_harm_survival_rate": int(family_survived.get("corner_harm", 0))
            / max(int(family_totals.get("corner_harm", 0)), 1),
            "corner_case_accuracy": after_corner.accuracy,
            "main_task_accuracy": after_main.accuracy,
            "mean_cosine_honest": cosine_means["HONEST"],
            "mean_cosine_fraud": cosine_means["FRAUD"],
            "mean_cosine_rarity": cosine_means["RARITY"],
            "mean_cosine_noise": cosine_means["NOISE"],
            "phase_name": round_bundle.phase,
        })

    return rows


def build_run_config(
    *,
    args: argparse.Namespace,
    initial_policy: Policy,
    eval_bundle,
    clients_total: int,
    clients_per_round: int,
) -> dict[str, Any]:
    return {
        "experiment_id": f"flpg_thesis_r{args.rounds}_c{args.cycle_rounds}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": PRETRAIN_SEED,
        "generator_seed": 20260318,
        "num_rounds": args.rounds,
        "clients_total": clients_total,
        "clients_per_round": clients_per_round,
        "server_lr_eta": L2_LEARNING_RATE,
        "local_epochs": None,
        "local_batch_size": None,
        "optimizer": "synthetic_gradient_generator; checkpoint pretrain uses Adam",
        "local_training_mode": "synthetic_gradients_no_client_optimizer",
        "tau_screen_init": initial_policy.cosine_filter_threshold,
        "theta_tol_init": initial_policy.theta_tol,
        "theta_rare_init": initial_policy.theta_rare,
        "theta_rarity_main_tol_init": initial_policy.theta_rarity_main_tol,
        "theta_corner_harm_init": THETA_CORNER_HARM_PROXY,
        "theta_corner_harm_source": THETA_CORNER_HARM_SOURCE,
        "recheck_probability_init": initial_policy.recheck_probability,
        "corner_weight_init": initial_policy.corner_weight,
        "d_proto_main_size": len(eval_bundle.proto_main),
        "d_proto_corner_size": len(eval_bundle.proto_corner),
        "d_audit_main_size": len(eval_bundle.audit_main),
        "d_audit_corner_size": len(eval_bundle.audit_corner),
        "d_oracle_main_size": len(eval_bundle.oracle_main),
        "d_oracle_corner_size": len(eval_bundle.oracle_corner),
        "pretrain_epochs": args.pretrain_epochs,
        "pretrain_batch_size": args.pretrain_batch_size,
        "pretrain_learning_rate": args.pretrain_learning_rate,
        "l1_reference_type": "geometric_median",
        "l1_router_mode": args.l1_router_mode,
        "l1_queue_budget_ratio": args.l1_queue_budget_ratio,
        "l1_random_recheck_ratio": args.l1_random_recheck_ratio,
    }


def build_rarity_generation_rows(rounds: list[RoundBundle]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for round_bundle in rounds:
        for row in round_bundle.rarity_generation_trace:
            cloned = dict(row)
            cloned["phase_name"] = round_bundle.phase
            rows.append(cloned)
    return rows


def build_blind_spot_coverage(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "covered_main_helpful_corner_harmful": True,
        "runtime_corner_harm_guard": True,
        "covered_proxy_bias_cases": False,
        "covered_low_confidence_cases": False,
        "notes": (
            "Runtime L2 now rejects main-helpful/corner-harmful updates and the benchmark "
            "constructs a dedicated FRAUD_CORNER_HARM archetype. Proxy-bias and "
            "low-confidence edge-case suites remain future-work coverage. "
            f"Observed shadow-audit main-helpful/corner-harmful count={summary['blind_spot_observed_count']}."
        ),
    }


def build_summary_metrics(
    *,
    run_config: dict[str, Any],
    round_summary_rows: list[dict[str, Any]],
    l1_rows: list[dict[str, Any]],
    l2_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    flpg_summary: dict[str, Any],
) -> dict[str, Any]:
    verdict_counts = Counter(row["verdict"] for row in l2_rows if row["audited_in_l2"])
    method_summary: dict[str, Any] = {}
    grouped_methods: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in baseline_rows:
        grouped_methods[str(row["method"])].append(row)
    for method, rows in grouped_methods.items():
        method_summary[method] = {
            "rounds": len(rows),
            "main_task_accuracy_avg": safe_mean([float(row["main_task_accuracy"]) for row in rows]),
            "corner_case_accuracy_avg": safe_mean([float(row["corner_case_accuracy"]) for row in rows]),
            "fraud_survival_count_avg": safe_mean([float(row["fraud_survival_count"]) for row in rows]),
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": run_config["experiment_id"],
        "l1": {
            "routed_total": flpg_summary["l1_routed_total"],
            "routing_precision_non_honest": flpg_summary["l1_routing_precision_non_honest"],
            "routing_recall": flpg_summary["l1_routing_recall"],
        },
        "l2": {
            "audited_total": sum(1 for row in l2_rows if row["audited_in_l2"]),
            "verdict_counts": dict(verdict_counts),
        },
        "round_summary": {
            "queue_size_avg": safe_mean([float(row["queue_size_l1"]) for row in round_summary_rows]),
            "queue_ratio_avg": safe_mean([float(row["queue_ratio_qt"]) for row in round_summary_rows]),
            "accepted_rarity_avg": safe_mean([float(row["accepted_rarity"]) for row in round_summary_rows]),
            "rejected_fraud_avg": safe_mean([float(row["rejected_fraud"]) for row in round_summary_rows]),
        },
        "blind_spot_observed_count": flpg_summary["blind_spot_observed_count"],
        "fraud_survival_by_family": flpg_summary.get("fraud_survival_by_family", {}),
        "fraud_total_by_family": flpg_summary.get("fraud_total_by_family", {}),
        "fraud_survived_by_family": flpg_summary.get("fraud_survived_by_family", {}),
        "fraud_caught_by_routing_reason": flpg_summary.get("fraud_caught_by_routing_reason", {}),
        "l1_routed_by_reason": flpg_summary.get("l1_routed_by_reason", {}),
        "methods": method_summary,
    }


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir

    reference_policy = clone_policy(DEFAULT_POLICY)
    if args.recheck_probability is not None:
        reference_policy = policy_with_recheck_probability(
            reference_policy,
            args.recheck_probability,
        )
    l1_router_config = make_l1_router_config(
        args.l1_router_mode,
        cos_deviation_threshold=reference_policy.cosine_filter_threshold,
        queue_budget_ratio=args.l1_queue_budget_ratio,
        random_recheck_ratio=args.l1_random_recheck_ratio,
    )
    initial_model, checkpoint_info = pretrain_initial_checkpoint(
        epochs=args.pretrain_epochs,
        batch_size=args.pretrain_batch_size,
        learning_rate=args.pretrain_learning_rate,
    )
    eval_bundle = build_eval_bundle()
    generator = _make_generator(initial_model=initial_model, eval_bundle=eval_bundle)
    rounds, overall_counts = build_round_bundles(
        policy=reference_policy,
        total_rounds=args.rounds,
        cycle_rounds=args.cycle_rounds,
        generator=generator,
    )

    run_config = build_run_config(
        args=args,
        initial_policy=reference_policy,
        eval_bundle=eval_bundle,
        clients_total=len(generator.vehicle_pool),
        clients_per_round=len(rounds[0].gradients) if rounds else 0,
    )
    run_config["initial_checkpoint"] = checkpoint_info
    run_config["overall_ground_truth_counts"] = dict(overall_counts)

    round_summary_rows, l1_rows, l2_rows, policy_rows, flpg_baseline_rows, flpg_summary = run_flpg_with_artifacts(
        rounds=rounds,
        initial_model=initial_model,
        eval_bundle=eval_bundle,
        reference_policy=reference_policy,
        fixed_recheck_probability=args.recheck_probability,
        l1_router_config=l1_router_config,
    )

    baseline_rows = list(flpg_baseline_rows)
    for method_id in ("fedavg", "geomed", "krum"):
        baseline_rows.extend(
            run_baseline_method(
                method_id=method_id,
                rounds=rounds,
                initial_model=initial_model,
                eval_bundle=eval_bundle,
                krum_byzantine_budget=args.krum_byzantine_budget,
            )
        )

    rarity_rows = build_rarity_generation_rows(rounds)
    blind_spot_coverage = build_blind_spot_coverage(flpg_summary)
    summary_metrics = build_summary_metrics(
        run_config=run_config,
        round_summary_rows=round_summary_rows,
        l1_rows=l1_rows,
        l2_rows=l2_rows,
        baseline_rows=baseline_rows,
        flpg_summary=flpg_summary,
    )

    write_json(output_dir / "run_config.json", run_config)
    write_csv(
        output_dir / "round_summary.csv",
        [
            "round_id",
            "phase_name",
            "participants",
            "queue_size_l1",
            "queue_ratio_qt",
            "l1_router_mode",
            "accepted_standard",
            "accepted_rarity",
            "rejected_fraud",
            "rejected_corner_harm",
            "rejected_noise",
            "honest_safe_count",
            "tau_screen",
            "theta_tol",
            "theta_rare",
            "theta_rarity_main_tol",
            "theta_corner_harm",
            "recheck_probability",
            "corner_weight",
        ],
        round_summary_rows,
    )
    write_csv(
        output_dir / "l1_routing.csv",
        [
            "round_id",
            "client_id",
            "true_label",
            "planned_role",
            "cosine_to_mean",
            "cosine_to_geomed",
            "deviation_score_geomed",
            "tau_screen",
            "recheck_probability",
            "routed_to_l2",
            "routing_reason",
            "l1_router_mode",
            "l1_risk_score",
            "l1_norm_mad_score",
            "l1_sign_disagreement",
            "l1_reputation_risk",
            "l1_audit_age_score",
            "phase_name",
        ],
        l1_rows,
    )
    write_csv(
        output_dir / "l2_audit.csv",
        [
            "round_id",
            "client_id",
            "true_label",
            "planned_role",
            "attack_family",
            "l_main_base",
            "l_corner_base",
            "delta_l_main",
            "delta_l_corner",
            "shadow_verdict",
            "verdict",
            "action",
            "aggregation_weight",
            "audited_in_l2",
            "routing_reason",
            "l1_router_mode",
            "l1_risk_score",
            "l1_norm_mad_score",
            "l1_sign_disagreement",
            "l1_reputation_risk",
            "l1_audit_age_score",
            "phase_name",
        ],
        l2_rows,
    )
    write_csv(
        output_dir / "rarity_generation.csv",
        [
            "round_id",
            "candidate_id",
            "corner_dominant_score",
            "main_suppression_score",
            "precheck_delta_l_main",
            "precheck_delta_l_corner",
            "passed_precheck",
            "accepted_as_rarity",
            "theta_tol_used",
            "theta_rare_used",
            "theta_rarity_main_tol_used",
        ],
        rarity_rows,
    )
    write_csv(
        output_dir / "policy_trace.csv",
        [
            "round_id",
            "tau_screen",
            "theta_tol",
            "theta_rare",
            "theta_rarity_main_tol",
            "theta_corner_harm",
            "corner_weight",
            "reward_multiplier",
            "slash_multiplier",
            "update_reason",
        ],
        policy_rows,
    )
    write_json(output_dir / "blind_spot_coverage.json", blind_spot_coverage)
    write_csv(
        output_dir / "baseline_diagnostics.csv",
        [
            "round_id",
            "method",
            "fraud_survival_count",
            "corner_case_accuracy",
            "main_task_accuracy",
            "mean_cosine_honest",
            "mean_cosine_fraud",
            "mean_cosine_rarity",
            "mean_cosine_noise",
        ],
        baseline_rows,
    )
    write_text(
        output_dir / "gradient_definition.txt",
        "g_i is a flattened synthetic true-gradient surrogate that is applied exactly as W' = W - eta * g_i; it is not a model delta (W_local - W) or pseudo-gradient.",
    )
    write_json(output_dir / "summary_metrics.json", summary_metrics)

    print(f"Exported thesis artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

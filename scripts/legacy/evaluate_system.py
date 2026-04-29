#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import copy
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common.schemas import DEFAULT_POLICY, Policy, RoundTelemetry
from generate_demo_data import DEMO_POLICY_PHASES, DemoDataGenerator
from l1_linear_defense.aggregation import filter_suspects
from policy_agent.constraints.safety_guard import SafetyGuard
from policy_agent.engine.glm_policy_engine import GLMPolicyEngine
from policy_agent.engine.policy_modes import build_policy_decision_context
from policy_agent.engine.rule_engine import RuleEngine


TARGET_LABELS = ("RARITY", "FRAUD")


@dataclass
class RoleMetrics:
    total: int = 0
    routed: int = 0
    true_positive: int = 0
    predicted: int = 0
    l1_clean_miss: int = 0
    wrong_label_miss: int = 0

    @property
    def routing_recall(self) -> float:
        return self.routed / self.total if self.total else 0.0

    @property
    def recall(self) -> float:
        return self.true_positive / self.total if self.total else 0.0

    @property
    def precision(self) -> float:
        return self.true_positive / self.predicted if self.predicted else 0.0


def clone_policy(policy: Policy, round_id: int | None = None) -> Policy:
    data = policy.model_dump()
    if round_id is not None:
        data["round_id"] = round_id
        data["effective_from_round"] = round_id
    return Policy.model_validate(data)


def fetch_live_policy(timeout: float = 2.0) -> Policy | None:
    url = os.getenv("POLICY_AGENT_URL", "http://127.0.0.1:8083").rstrip("/")
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.get(f"{url}/api/v1/policy/current", timeout=timeout)
        response.raise_for_status()
        return Policy.model_validate(response.json())
    except Exception:
        return None


def choose_reference_policy(source: str) -> tuple[Policy, str]:
    if source == "default":
        return clone_policy(DEFAULT_POLICY), "default_policy"
    if source == "live":
        live = fetch_live_policy()
        if live is None:
            raise RuntimeError("Could not fetch live policy from policy-agent")
        return live, "live_policy"

    live = fetch_live_policy()
    if live is not None:
        return live, "live_policy"
    return clone_policy(DEFAULT_POLICY), "default_policy"


def evaluate_classification(reference_policy: Policy, rounds: int) -> dict[str, Any]:
    generator = DemoDataGenerator()
    generator.ground_truth_mode = "archetype"
    metrics_by_role = {label: RoleMetrics() for label in TARGET_LABELS}
    confusion = Counter()
    l1_honest_false_positive = 0
    l1_honest_total = 0
    observed_rounds: list[dict[str, Any]] = []
    recheck_rng = random.Random(20260428)

    for round_index in range(rounds):
        eval_policy = clone_policy(reference_policy, round_id=reference_policy.round_id + round_index)
        (
            updates,
            planned_roles,
            _vehicles,
            _new_vehicle_count,
            preflight_counts,
            _l1_projection,
        ) = generator._build_batch(
            round_id=eval_policy.round_id,
            policy=eval_policy,
            round_index=round_index,
        )

        gradients = [np.array(payload["gradient_data"], dtype=float) for payload in updates]
        vehicle_ids = [payload["vehicle_address"] for payload in updates]
        l1_result = filter_suspects(
            gradients,
            vehicle_ids,
            threshold=eval_policy.cosine_filter_threshold,
            recheck_probability=eval_policy.recheck_probability,
            rng=recheck_rng,
        )
        suspect_indices = set(l1_result.suspect_indices)

        generator.auditor.apply_policy(eval_policy)
        final_labels: dict[str, str] = {}
        for idx, payload in enumerate(updates):
            vehicle_id = payload["vehicle_address"]
            if idx in suspect_indices:
                audit = generator.auditor.audit(vehicle_id, gradients[idx])
                final_labels[vehicle_id] = audit.classification.value
            else:
                final_labels[vehicle_id] = "L1_CLEAN"

        round_confusion = Counter()
        for idx, payload in enumerate(updates):
            vehicle_id = payload["vehicle_address"]
            ground_truth = payload["metadata"].get(
                "ground_truth_label",
                payload["metadata"].get(
                    "ground_truth_role",
                    payload["metadata"].get("planned_role", payload["metadata"]["preflight_role"]),
                ),
            )
            predicted = final_labels[vehicle_id]
            confusion[(ground_truth, predicted)] += 1
            round_confusion[(ground_truth, predicted)] += 1

            if ground_truth == "HONEST":
                l1_honest_total += 1
                if idx in suspect_indices:
                    l1_honest_false_positive += 1

            if ground_truth in TARGET_LABELS:
                role_metrics = metrics_by_role[ground_truth]
                role_metrics.total += 1
                if idx in suspect_indices:
                    role_metrics.routed += 1
                if predicted == ground_truth:
                    role_metrics.true_positive += 1
                elif predicted == "L1_CLEAN":
                    role_metrics.l1_clean_miss += 1
                else:
                    role_metrics.wrong_label_miss += 1

        for predicted in TARGET_LABELS:
            predicted_count = sum(
                1
                for value in final_labels.values()
                if value == predicted
            )
            metrics_by_role[predicted].predicted += predicted_count

        observed_rounds.append(
            {
                "round_index": round_index,
                "phase": DEMO_POLICY_PHASES[round_index % len(DEMO_POLICY_PHASES)]["name"],
                "planned_roles": dict(planned_roles),
                "preflight_roles": dict(preflight_counts),
                "suspects": len(suspect_indices),
                "round_confusion": {
                    f"{gt}->{pred}": count
                    for (gt, pred), count in sorted(round_confusion.items())
                },
            }
        )

    per_role = {
        role: {
            "total": role_metrics.total,
            "routing_recall": role_metrics.routing_recall,
            "recall": role_metrics.recall,
            "precision": role_metrics.precision,
            "l1_clean_miss": role_metrics.l1_clean_miss,
            "wrong_label_miss": role_metrics.wrong_label_miss,
        }
        for role, role_metrics in metrics_by_role.items()
    }

    return {
        "policy_round": reference_policy.round_id,
        "rounds": rounds,
        "per_role": per_role,
        "honest_l1_false_positive_rate": (
            l1_honest_false_positive / l1_honest_total if l1_honest_total else 0.0
        ),
        "confusion": {
            f"{gt}->{pred}": count
            for (gt, pred), count in sorted(confusion.items())
        },
        "observed_rounds": observed_rounds,
    }


def telemetry_scenarios(round_id: int) -> list[tuple[str, RoundTelemetry]]:
    base = {
        "round_id": round_id,
        "fraud_rate": 0.03,
        "rarity_rate": 0.06,
        "honest_rate": 0.84,
        "noise_rate": 0.07,
        "main_accuracy": 0.93,
        "corner_accuracy": 0.80,
        "main_loss_delta_avg": -0.003,
        "corner_loss_delta_avg": -0.008,
        "false_slash_estimate": 0.02,
        "rarity_retention_rate": 0.88,
        "golden_drift_score": 0.03,
        "reject_rate_l3": 0.0,
        "cosine_outlier_ratio": 0.18,
        "suspect_queue_length": 18,
        "avg_sbt_score": 5.0,
        "new_vehicle_ratio": 0.08,
        "hash_mismatch_rate": 0.0,
        "recent_attack_pressure": 0.05,
    }

    scenarios = []
    scenarios.append(
        (
            "fraud_pressure",
            RoundTelemetry.model_validate(
                {
                    **base,
                    "fraud_rate": 0.24,
                    "rarity_rate": 0.05,
                    "honest_rate": 0.58,
                    "noise_rate": 0.13,
                    "main_accuracy": 0.82,
                    "corner_accuracy": 0.72,
                    "main_loss_delta_avg": 0.018,
                    "corner_loss_delta_avg": 0.004,
                    "recent_attack_pressure": 0.36,
                }
            ),
        )
    )
    scenarios.append(
        (
            "rarity_under_recall",
            RoundTelemetry.model_validate(
                {
                    **base,
                    "fraud_rate": 0.03,
                    "rarity_rate": 0.01,
                    "honest_rate": 0.78,
                    "noise_rate": 0.18,
                    "main_accuracy": 0.90,
                    "corner_accuracy": 0.58,
                    "main_loss_delta_avg": -0.002,
                    "corner_loss_delta_avg": -0.022,
                    "rarity_retention_rate": 0.61,
                }
            ),
        )
    )
    scenarios.append(
        (
            "false_slash_risk",
            RoundTelemetry.model_validate(
                {
                    **base,
                    "fraud_rate": 0.04,
                    "rarity_rate": 0.04,
                    "honest_rate": 0.74,
                    "noise_rate": 0.18,
                    "corner_accuracy": 0.69,
                    "false_slash_estimate": 0.11,
                    "rarity_retention_rate": 0.73,
                }
            ),
        )
    )
    scenarios.append(
        (
            "novel_rarity_like_drift",
            RoundTelemetry.model_validate(
                {
                    **base,
                    "fraud_rate": 0.05,
                    "rarity_rate": 0.03,
                    "honest_rate": 0.72,
                    "noise_rate": 0.20,
                    "main_accuracy": 0.88,
                    "corner_accuracy": 0.69,
                    "main_loss_delta_avg": 0.003,
                    "corner_loss_delta_avg": -0.024,
                    "rarity_retention_rate": 0.76,
                    "golden_drift_score": 0.09,
                    "recent_attack_pressure": 0.10,
                }
            ),
        )
    )
    scenarios.append(
        (
            "harmful_shift_drift",
            RoundTelemetry.model_validate(
                {
                    **base,
                    "fraud_rate": 0.16,
                    "rarity_rate": 0.04,
                    "honest_rate": 0.62,
                    "noise_rate": 0.18,
                    "main_accuracy": 0.79,
                    "corner_accuracy": 0.71,
                    "main_loss_delta_avg": 0.021,
                    "corner_loss_delta_avg": -0.0002,
                    "golden_drift_score": 0.10,
                    "recent_attack_pressure": 0.24,
                }
            ),
        )
    )
    scenarios.append(
        (
            "ambiguous_overlap",
            RoundTelemetry.model_validate(
                {
                    **base,
                    "fraud_rate": 0.19,
                    "rarity_rate": 0.03,
                    "honest_rate": 0.60,
                    "noise_rate": 0.18,
                    "main_accuracy": 0.84,
                    "corner_accuracy": 0.67,
                    "main_loss_delta_avg": 0.006,
                    "corner_loss_delta_avg": -0.003,
                    "false_slash_estimate": 0.06,
                    "rarity_retention_rate": 0.79,
                    "golden_drift_score": 0.08,
                    "recent_attack_pressure": 0.27,
                }
            ),
        )
    )
    return scenarios


def proposal_delta(current: Policy, proposed: Policy) -> dict[str, float]:
    return {
        "theta_tol": proposed.theta_tol - current.theta_tol,
        "theta_rare": proposed.theta_rare - current.theta_rare,
        "theta_drift": proposed.theta_drift - current.theta_drift,
        "recheck_probability": proposed.recheck_probability - current.recheck_probability,
        "slash_multiplier": proposed.slash_multiplier - current.slash_multiplier,
        "rarity_reward_multiplier": proposed.rarity_reward_multiplier - current.rarity_reward_multiplier,
        "corner_weight": proposed.corner_weight - current.corner_weight,
    }


def score_proposal(current: Policy, telemetry: RoundTelemetry, proposed: Policy) -> tuple[float, list[str]]:
    context = build_policy_decision_context(telemetry)
    delta = proposal_delta(current, proposed)
    score = 0.0
    notes: list[str] = []

    if context.fraud_pressure_high:
        if delta["theta_tol"] < 0:
            score += 2.0
            notes.append("tightened theta_tol under fraud pressure")
        else:
            score -= 2.0
            notes.append("did not tighten theta_tol under fraud pressure")
        if delta["recheck_probability"] > 0:
            score += 2.0
            notes.append("raised recheck under fraud pressure")
        else:
            score -= 1.0
        if not context.rarity_under_recall and not context.false_slash_risk_high:
            if 0 < delta["slash_multiplier"] <= 0.10:
                score += 1.0
                notes.append("moderately raised slashing")
            elif delta["slash_multiplier"] > 0.10:
                score -= 0.5
        if delta["theta_rare"] < 0:
            score -= 2.0
            notes.append("tightened rarity threshold during fraud mode")

    if context.rarity_under_recall:
        if delta["theta_rare"] > 0:
            score += 3.0
            notes.append("relaxed theta_rare for rarity recovery")
        else:
            score -= 3.0
        if delta["rarity_reward_multiplier"] > 0:
            score += 2.0
            notes.append("raised rarity reward")
        else:
            score -= 1.0
        if delta["recheck_probability"] > 0:
            score += 1.0
        if delta["slash_multiplier"] > 0:
            score -= 2.0
            notes.append("raised slashing during rarity preservation")

    if context.false_slash_risk_high:
        if delta["slash_multiplier"] < 0:
            score += 3.0
            notes.append("reduced slashing under false-slash risk")
        else:
            score -= 3.0
        if delta["recheck_probability"] > 0:
            score += 2.0
        else:
            score -= 1.0

    if context.drift_warning:
        if delta["recheck_probability"] > 0:
            score += 1.0
        else:
            score -= 1.0

        if context.novel_rarity_like_drift:
            if delta["theta_drift"] < 0:
                score -= 2.0
                notes.append("punished novelty-like drift")
            else:
                score += 1.0
            if delta["theta_rare"] >= 0:
                score += 1.0
            if delta["slash_multiplier"] <= 0:
                score += 1.0
            else:
                score -= 2.0
                notes.append("increased slashing on novelty-like drift")
        elif context.harmful_shift_like_drift:
            if delta["theta_drift"] < 0:
                score += 2.0
                notes.append("tightened drift threshold for harmful shift")
            else:
                score -= 2.0
            if (
                context.fraud_pressure_high
                and not context.false_slash_risk_high
                and not context.rarity_under_recall
                and delta["slash_multiplier"] > 0
            ):
                score += 1.0
        elif context.drift_ambiguous:
            if delta["slash_multiplier"] <= 0:
                score += 1.0
            else:
                score -= 2.0
                notes.append("added punishment under ambiguous drift")
            if delta["theta_drift"] < 0 and delta["recheck_probability"] <= 0:
                score -= 1.0

    return score, notes


async def evaluate_policy_control(
    current_policy: Policy,
    skip_glm: bool,
) -> dict[str, Any]:
    safety_guard = SafetyGuard()
    rule_engine = RuleEngine()
    glm_engine = GLMPolicyEngine()

    scenario_results: list[dict[str, Any]] = []
    glm_scores: list[float] = []
    rule_scores: list[float] = []
    glm_remote_calls = 0
    glm_successful_improvements = 0

    for name, telemetry in telemetry_scenarios(current_policy.round_id):
        context = build_policy_decision_context(telemetry)

        rule_proposal = await rule_engine.propose(current_policy, telemetry)
        rule_proposal = await safety_guard.check(rule_proposal, telemetry)
        rule_score, rule_notes = score_proposal(
            current_policy,
            telemetry,
            rule_proposal.proposed_policy,
        )
        rule_scores.append(rule_score)

        result: dict[str, Any] = {
            "scenario": name,
            "needs_glm_reasoning": context.needs_glm_reasoning,
            "active_modes": list(context.active_modes),
            "rule": {
                "source_engine": rule_proposal.source_engine,
                "safety_guard_passed": rule_proposal.safety_guard_passed,
                "score": rule_score,
                "delta": proposal_delta(current_policy, rule_proposal.proposed_policy),
                "reasons": rule_proposal.reasons,
                "notes": rule_notes,
            },
        }

        if skip_glm:
            result["glm"] = {"skipped": True}
            scenario_results.append(result)
            continue

        glm_proposal = await glm_engine.propose(current_policy, telemetry)
        glm_proposal = await safety_guard.check(glm_proposal, telemetry)
        glm_score, glm_notes = score_proposal(
            current_policy,
            telemetry,
            glm_proposal.proposed_policy,
        )
        glm_scores.append(glm_score)
        if glm_proposal.llm_used:
            glm_remote_calls += 1
        if glm_score > rule_score:
            glm_successful_improvements += 1

        result["glm"] = {
            "source_engine": glm_proposal.source_engine,
            "llm_used": glm_proposal.llm_used,
            "safety_guard_passed": glm_proposal.safety_guard_passed,
            "score": glm_score,
            "delta": proposal_delta(current_policy, glm_proposal.proposed_policy),
            "reasons": glm_proposal.reasons,
            "notes": glm_notes,
            "score_gain_vs_rule": glm_score - rule_score,
        }
        scenario_results.append(result)

    summary = {
        "scenarios": scenario_results,
        "rule_average_score": mean(rule_scores) if rule_scores else 0.0,
    }

    if not skip_glm:
        summary.update(
            {
                "glm_average_score": mean(glm_scores) if glm_scores else 0.0,
                "glm_remote_calls": glm_remote_calls,
                "glm_better_than_rule_count": glm_successful_improvements,
            }
        )

    return summary


def print_classification_summary(report: dict[str, Any]) -> None:
    print("== Classification Evaluation ==")
    print(f"policy_round={report['policy_round']} rounds={report['rounds']}")
    for role, metrics in report["per_role"].items():
        print(
            f"{role}: total={metrics['total']} "
            f"routing_recall={metrics['routing_recall']:.3f} "
            f"recall={metrics['recall']:.3f} "
            f"precision={metrics['precision']:.3f} "
            f"l1_clean_miss={metrics['l1_clean_miss']} "
            f"wrong_label_miss={metrics['wrong_label_miss']}"
        )
    print(
        "HONEST L1 false-positive routing rate="
        f"{report['honest_l1_false_positive_rate']:.3f}"
    )
    print("Top confusion entries:")
    for key, count in list(report["confusion"].items())[:12]:
        print(f"  {key}: {count}")


def print_policy_summary(report: dict[str, Any], skip_glm: bool) -> None:
    print("\n== Policy Evaluation ==")
    print(f"rule_average_score={report['rule_average_score']:.3f}")
    if not skip_glm:
        print(f"glm_average_score={report['glm_average_score']:.3f}")
        print(f"glm_remote_calls={report['glm_remote_calls']}")
        print(f"glm_better_than_rule_count={report['glm_better_than_rule_count']}")

    for scenario in report["scenarios"]:
        print(
            f"\n[{scenario['scenario']}] needs_glm={scenario['needs_glm_reasoning']} "
            f"modes={scenario['active_modes']}"
        )
        print(
            f"  rule: source={scenario['rule']['source_engine']} "
            f"score={scenario['rule']['score']:.3f}"
        )
        if skip_glm:
            continue
        glm = scenario["glm"]
        if glm.get("skipped"):
            print("  glm: skipped")
        else:
            print(
                f"  glm: source={glm['source_engine']} llm_used={glm['llm_used']} "
                f"score={glm['score']:.3f} gain_vs_rule={glm['score_gain_vs_rule']:.3f}"
            )


async def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate FLPG classification and policy behavior")
    parser.add_argument("--classification-rounds", type=int, default=16)
    parser.add_argument("--policy-source", choices=("auto", "live", "default"), default="auto")
    parser.add_argument("--skip-glm", action="store_true")
    args = parser.parse_args()

    reference_policy, policy_source = choose_reference_policy(args.policy_source)
    print(f"Using {policy_source} (round={reference_policy.round_id})")

    classification_report = evaluate_classification(
        reference_policy=reference_policy,
        rounds=args.classification_rounds,
    )
    print_classification_summary(classification_report)

    policy_report = await evaluate_policy_control(
        current_policy=reference_policy,
        skip_glm=args.skip_glm,
    )
    print_policy_summary(policy_report, skip_glm=args.skip_glm)

    rarity_recall = classification_report["per_role"]["RARITY"]["recall"]
    fraud_recall = classification_report["per_role"]["FRAUD"]["recall"]
    rarity_precision = classification_report["per_role"]["RARITY"]["precision"]
    fraud_precision = classification_report["per_role"]["FRAUD"]["precision"]

    print("\n== Verdict ==")
    print(
        "classification_pass="
        f"{rarity_recall >= 0.60 and fraud_recall >= 0.80 and rarity_precision >= 0.80 and fraud_precision >= 0.80}"
    )
    if args.skip_glm:
        print("glm_gain_verdict=skipped")
    else:
        glm_remote_calls = policy_report["glm_remote_calls"]
        glm_gain = policy_report["glm_average_score"] - policy_report["rule_average_score"]
        if glm_remote_calls == 0:
            print("glm_gain_verdict=not_observed")
        else:
            print(f"glm_gain_verdict={glm_gain > 0}")
            print(f"glm_average_score_gain={glm_gain:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

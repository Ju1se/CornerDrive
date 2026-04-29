#!/usr/bin/env python3
"""CornerDrive demo data generator (improved benchmark version).

Key improvements over the previous version:

1. New archetype FRAUD_CORNER_HARM that keeps main-task loss roughly neutral
   while harming corner-case loss. This makes the dual-objective audit
   empirically necessary rather than a design assertion.

2. Removed _promote_rarity_visibility() which iteratively reshaped gradients
   until L1 produced a desired routing outcome. That mechanism made the
   benchmark's "L1 0% honest false positive / 100% fraud recall" numbers
   construction artefacts rather than discriminator behaviour.

3. Telemetry now decomposes fraud survival, L1 routing, and L2 verdicts by
   attack family / archetype, exposing per-family behaviour that overall
   averages would otherwise hide.

4. Optional generation trace persistence (one JSONL per round) for
   reproducibility audits.

5. Stub interface for periodic reference-gradient refresh; currently a no-op
   because the prototype model is frozen, but kept so that future work can
   wire it to the evolving global model state without restructuring.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests
import redis
import torch
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from common.demo_audit_assets import DEMO_GRADIENT_DIM, build_demo_audit_bundle
from common.schemas import Policy
from common.config import L2_AUDIT_QUEUE
from l1_linear_defense.aggregation import filter_suspects
from l2_dual_audit.classifier import Classification, DualChannelAuditor

load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Environment / configuration
# ---------------------------------------------------------------------------

API_KEY = os.getenv("DEMO_API_KEY", "dev_key_1")
L1_API_URL = os.getenv("L1_API_URL", "http://127.0.0.1:8081").rstrip("/")
L4_API_URL = os.getenv("L4_API_URL", "http://127.0.0.1:8082").rstrip("/")
POLICY_AGENT_URL = os.getenv("POLICY_AGENT_URL", "http://127.0.0.1:8083").rstrip("/")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

BASE_BATCH_SIZE = int(os.getenv("BASE_BATCH_SIZE", "32"))
SIMULATION_SCALE_FACTOR = max(1, int(os.getenv("SIMULATION_SCALE_FACTOR", "3")))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", str(BASE_BATCH_SIZE * SIMULATION_SCALE_FACTOR)))
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", str(max(8, 4 * SIMULATION_SCALE_FACTOR))))
VEHICLE_POOL_SIZE = int(os.getenv("VEHICLE_POOL_SIZE", str(max(128, BATCH_SIZE * 4))))

CONTINUOUS_MODE = os.getenv("CONTINUOUS_MODE", "true").lower() == "true"
ROUND_INTERVAL = float(os.getenv("ROUND_INTERVAL", "4.0"))
PROCESS_WAIT_SECONDS = float(os.getenv("PROCESS_WAIT_SECONDS", "16.0"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "60.0"))
AUTO_ACTIVATE_POLICY = os.getenv("AUTO_ACTIVATE_POLICY", "true").lower() == "true"
SAMPLE_COUNT_MIN = int(os.getenv("SAMPLE_COUNT_MIN", "1200"))
SAMPLE_COUNT_MAX = int(os.getenv("SAMPLE_COUNT_MAX", "12000"))
SUBMISSION_MAX_WORKERS = max(6, int(os.getenv("SUBMISSION_MAX_WORKERS", str(min(12, BATCH_SIZE)))))

POLICY_TRIGGER_FRAUD_RATE = float(os.getenv("POLICY_TRIGGER_FRAUD_RATE", "0.18"))
POLICY_TRIGGER_RARITY_RATE = float(os.getenv("POLICY_TRIGGER_RARITY_RATE", "0.03"))
POLICY_TRIGGER_CORNER_ACCURACY = float(os.getenv("POLICY_TRIGGER_CORNER_ACCURACY", "0.70"))
POLICY_TRIGGER_FALSE_SLASH = float(os.getenv("POLICY_TRIGGER_FALSE_SLASH", "0.08"))
POLICY_TRIGGER_DRIFT_SCORE = float(os.getenv("POLICY_TRIGGER_DRIFT_SCORE", "0.06"))
POLICY_TRIGGER_COOLDOWN_ROUNDS = int(os.getenv("POLICY_TRIGGER_COOLDOWN_ROUNDS", "2"))
ROLE_MATERIALIZE_RETRIES = int(os.getenv("ROLE_MATERIALIZE_RETRIES", "28"))
RARITY_TARGET_MARGIN = float(os.getenv("RARITY_TARGET_MARGIN", "0.002"))
SUSPECT_MARGIN = float(os.getenv("SUSPECT_MARGIN", "0.025"))
MIN_RARITY_PER_ROUND = int(os.getenv("MIN_RARITY_PER_ROUND", "1"))
MIN_RARITY_PER_CORNER_PHASE = int(os.getenv("MIN_RARITY_PER_CORNER_PHASE", "2"))
MIN_FRAUD_PER_ATTACK_PHASE = int(os.getenv("MIN_FRAUD_PER_ATTACK_PHASE", "2"))
MIN_NOISE_PER_STRESS_PHASE = int(os.getenv("MIN_NOISE_PER_STRESS_PHASE", "2"))
MIN_CORNER_HARM_PER_TARGET_PHASE = int(os.getenv("MIN_CORNER_HARM_PER_TARGET_PHASE", "1"))

# Periodic reference-gradient refresh (currently a no-op; see
# _maybe_refresh_reference_gradients for rationale).
GRADIENT_REFRESH_INTERVAL = int(os.getenv("GRADIENT_REFRESH_INTERVAL", "5"))

# Optional persistence of per-round generation traces. Empty string disables.
GENERATION_TRACE_DIR = os.getenv("GENERATION_TRACE_DIR", "")

# Phase-level role profiles. FRAUD has been split into two attack families:
#   - "FRAUD"             : sign-flip-and-amplify (the original family)
#   - "FRAUD_CORNER_HARM" : main-friendly but corner-harming attack
DEMO_ROLE_PROFILES_BY_PHASE = {
    "steady": {
        "HONEST": 0.84,
        "FRAUD": 0.03,
        "FRAUD_CORNER_HARM": 0.01,
        "RARITY": 0.08,
        "NOISE": 0.04,
    },
    "fraud_wave": {
        "HONEST": 0.72,
        "FRAUD": 0.07,
        "FRAUD_CORNER_HARM": 0.04,
        "RARITY": 0.09,
        "NOISE": 0.08,
    },
    "corner_gap": {
        "HONEST": 0.74,
        "FRAUD": 0.02,
        "FRAUD_CORNER_HARM": 0.04,
        "RARITY": 0.14,
        "NOISE": 0.06,
    },
    "false_slash_risk": {
        "HONEST": 0.78,
        "FRAUD": 0.02,
        "FRAUD_CORNER_HARM": 0.01,
        "RARITY": 0.10,
        "NOISE": 0.09,
    },
    "drift_burst": {
        "HONEST": 0.70,
        "FRAUD": 0.06,
        "FRAUD_CORNER_HARM": 0.04,
        "RARITY": 0.08,
        "NOISE": 0.12,
    },
}

DEMO_PHASE_SEQUENCE = [
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
    "steady",
]

# Archetype constants. Order matters for _expand_profile allocation.
ROLES_ORDER = ("HONEST", "FRAUD", "FRAUD_CORNER_HARM", "RARITY", "NOISE")
FRAUD_ROLES = {"FRAUD", "FRAUD_CORNER_HARM"}

# Mapping from archetype to attack_family metadata field.
ARCHETYPE_TO_ATTACK_FAMILY = {
    "HONEST": "none",
    "RARITY": "none",
    "NOISE": "none",
    "FRAUD": "sign_flip_proxy",
    "FRAUD_CORNER_HARM": "corner_harm",
}

# Mapping from archetype to the four-class ground-truth label that L2 verdicts
# are compared against.
ARCHETYPE_TO_GROUND_TRUTH_LABEL = {
    "HONEST": "HONEST",
    "RARITY": "RARITY",
    "NOISE": "NOISE",
    "FRAUD": "FRAUD",
    "FRAUD_CORNER_HARM": "FRAUD",
}

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("FLPGDemoGenerator")
LOCAL_AUDIT_LOGGER = logging.getLogger("l2_dual_audit.classifier")
LOCAL_L1_LOGGER = logging.getLogger("l1_linear_defense.aggregation")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        return vector.copy()
    return vector / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_deviation_score(vector: np.ndarray, reference: np.ndarray) -> float:
    return 1.0 - cosine_similarity(vector, reference)


@dataclass(frozen=True)
class CandidateMetrics:
    classification: Classification
    delta_main: float
    delta_corner: float
    deviation: float


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class DemoDataGenerator:
    def __init__(self, seed: int = 20260318):
        self.seed = seed
        self.random = random.Random(seed)
        self.numpy_rng = np.random.default_rng(seed)
        self.ground_truth_mode = os.getenv("DEMO_GROUND_TRUTH_MODE", "archetype")
        LOCAL_AUDIT_LOGGER.setLevel(max(LOCAL_AUDIT_LOGGER.level, logging.WARNING))
        LOCAL_L1_LOGGER.setLevel(max(LOCAL_L1_LOGGER.level, logging.ERROR))

        self.session = requests.Session()
        self.session.trust_env = False
        self.session.headers.update({
            "Content-Type": "application/json",
            "X-API-Key": API_KEY,
        })
        try:
            self.redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            self.redis_client.ping()
        except redis.RedisError:
            self.redis_client = None

        model, main_dataset, corner_dataset = build_demo_audit_bundle()
        self.prototype_model = model
        self.main_dataset = main_dataset
        self.corner_dataset = corner_dataset
        self.auditor = DualChannelAuditor(
            model=copy.deepcopy(model),
            main_dataset=main_dataset,
            corner_dataset=corner_dataset,
        )

        self.main_gradient = normalize(self._compute_dataset_gradient(main_dataset))
        self.corner_gradient = normalize(self._compute_dataset_gradient(corner_dataset))
        self.random_gradient = self._build_random_basis()

        self.vehicle_pool = [
            self._vehicle_address(f"vehicle_pool_{index}")
            for index in range(VEHICLE_POOL_SIZE)
        ]
        self.seen_vehicles: set[str] = set()
        self.synthetic_round_id: int | None = None
        self.last_policy_trigger_round: int | None = None
        self.last_recent_audit_source: str | None = None
        self.last_refresh_round: int = -1
        self.current_rarity_generation_trace: list[dict[str, Any]] = []
        self.current_attack_anchor_validation: dict[str, dict[str, Any]] = {}
        self.current_attack_anchor_scale_by_role: dict[str, float] = {}

        if GENERATION_TRACE_DIR:
            self.trace_dir: Path | None = Path(GENERATION_TRACE_DIR)
            self.trace_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.trace_dir = None

    # ------------------------------------------------------------------
    # Identity / hashing helpers
    # ------------------------------------------------------------------

    def _vehicle_address(self, raw_id: str) -> str:
        digest = hashlib.sha256(raw_id.encode()).hexdigest()
        return f"0x{digest[:40]}"

    def _compute_dataset_gradient(self, dataset) -> np.ndarray:
        model = copy.deepcopy(self.prototype_model)
        model.zero_grad(set_to_none=True)

        outputs = model(dataset.data)
        loss = torch.nn.functional.cross_entropy(outputs, dataset.targets)
        loss.backward()

        flat_parts = [
            parameter.grad.detach().view(-1)
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        return torch.cat(flat_parts).cpu().numpy()

    def _build_random_basis(self) -> np.ndarray:
        raw = self.numpy_rng.normal(0.0, 1.0, size=DEMO_GRADIENT_DIM)
        raw = raw - np.dot(raw, self.main_gradient) * self.main_gradient
        raw = raw - np.dot(raw, self.corner_gradient) * self.corner_gradient
        return normalize(raw)

    def _orthogonalize(self, raw: np.ndarray) -> np.ndarray:
        raw = raw - np.dot(raw, self.main_gradient) * self.main_gradient
        raw = raw - np.dot(raw, self.corner_gradient) * self.corner_gradient
        return normalize(raw)

    def _basis_from_key(self, key: str) -> np.ndarray:
        seed_key = key if self.seed == 20260318 else f"{self.seed}:{key}"
        seed = int(hashlib.sha256(seed_key.encode()).hexdigest()[:16], 16) % (2 ** 32)
        rng = np.random.default_rng(seed)
        return self._orthogonalize(rng.normal(0.0, 1.0, size=DEMO_GRADIENT_DIM))

    def _vehicle_style_vector(self, vehicle_address: str) -> np.ndarray:
        return self._basis_from_key(f"vehicle-style:{vehicle_address}")

    # ------------------------------------------------------------------
    # Phase / round helpers
    # ------------------------------------------------------------------

    def _phase_name(self, round_index: int) -> str:
        return DEMO_PHASE_SEQUENCE[round_index % len(DEMO_PHASE_SEQUENCE)]

    def _round_drift_vector(self, round_index: int) -> np.ndarray:
        phase_name = self._phase_name(round_index)
        phase_basis = self._basis_from_key(f"phase:{phase_name}")

        if phase_name == "fraud_wave":
            raw = -0.65 * self.main_gradient + 0.15 * self.corner_gradient + 0.75 * phase_basis
        elif phase_name == "corner_gap":
            raw = 0.55 * self.corner_gradient - 0.18 * self.main_gradient + 0.55 * phase_basis
        elif phase_name == "false_slash_risk":
            raw = 0.35 * self.corner_gradient + 0.60 * phase_basis
        elif phase_name == "drift_burst":
            raw = 0.22 * self.corner_gradient - 0.10 * self.main_gradient + 0.95 * phase_basis
        else:
            raw = 0.12 * self.main_gradient + 0.10 * self.corner_gradient + 0.35 * phase_basis

        return normalize(raw)

    def _sample_count_scale(self, sample_count: int) -> float:
        scale = float(np.sqrt(sample_count / 1800.0))
        return clamp(scale, 0.80, 1.20)

    def _profile_for_round(self, round_index: int) -> dict[str, float]:
        phase_name = self._phase_name(round_index)
        return DEMO_ROLE_PROFILES_BY_PHASE.get(
            phase_name,
            DEMO_ROLE_PROFILES_BY_PHASE["steady"],
        )

    def _minimum_role_counts(self, phase_name: str) -> dict[str, int]:
        minimums = {role: 0 for role in ROLES_ORDER}

        if BATCH_SIZE >= 12:
            minimums["RARITY"] = max(MIN_RARITY_PER_ROUND, 0)

        if phase_name in {"corner_gap", "false_slash_risk", "drift_burst"} and BATCH_SIZE >= 24:
            minimums["RARITY"] = max(minimums["RARITY"], MIN_RARITY_PER_CORNER_PHASE)

        if phase_name in {"fraud_wave", "drift_burst"} and BATCH_SIZE >= 12:
            minimums["FRAUD"] = max(MIN_FRAUD_PER_ATTACK_PHASE, 0)

        if phase_name in {"fraud_wave", "false_slash_risk", "drift_burst"} and BATCH_SIZE >= 16:
            minimums["NOISE"] = max(MIN_NOISE_PER_STRESS_PHASE, 0)

        # Ensure corner-harm shows up at least minimally in phases where it is
        # the most relevant test signal.
        if phase_name in {"corner_gap", "drift_burst", "fraud_wave"} and BATCH_SIZE >= 16:
            minimums["FRAUD_CORNER_HARM"] = max(MIN_CORNER_HARM_PER_TARGET_PHASE, 0)

        # Trim the minimums if they would exceed the batch size, in priority
        # order (NOISE is dropped first, FRAUD_CORNER_HARM last among fraud-like
        # categories so that the dual-objective signal is preserved).
        trim_order = ("NOISE", "FRAUD", "FRAUD_CORNER_HARM", "RARITY")
        while sum(minimums.values()) >= BATCH_SIZE:
            trimmed = False
            for role in trim_order:
                if minimums[role] > 0 and sum(minimums.values()) >= BATCH_SIZE:
                    minimums[role] -= 1
                    trimmed = True
            if not trimmed:
                break

        return minimums

    # ------------------------------------------------------------------
    # Reference-gradient refresh (interface stub; see method docstring)
    # ------------------------------------------------------------------

    def _maybe_refresh_reference_gradients(self, round_index: int) -> None:
        """Recompute main/corner reference gradients periodically.

        In the current implementation this is intentionally a no-op:
        the prototype model is frozen for the entire run, so recomputing
        ``main_gradient`` and ``corner_gradient`` would yield bit-identical
        vectors. The interface is preserved so that future work can wire the
        prototype to the evolving global model state without restructuring
        the round loop.
        """
        if round_index - self.last_refresh_round < GRADIENT_REFRESH_INTERVAL:
            return
        # NOTE(future): wire prototype_model to the auditor's current model
        # state to make refresh meaningful. Currently a no-op.
        self.last_refresh_round = round_index

    # ------------------------------------------------------------------
    # Policy interaction
    # ------------------------------------------------------------------

    def _fetch_current_policy(self) -> Policy:
        response = self.session.get(
            f"{POLICY_AGENT_URL}/api/v1/policy/current",
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return Policy.model_validate(response.json())

    def _select_round_id(self, current_policy: Policy) -> int:
        if self.synthetic_round_id is None:
            self.synthetic_round_id = current_policy.round_id
        return self.synthetic_round_id

    def _next_round(self) -> None:
        if self.synthetic_round_id is None:
            self.synthetic_round_id = 0
        self.synthetic_round_id += 1

    # ------------------------------------------------------------------
    # Candidate construction and evaluation
    # ------------------------------------------------------------------

    def _make_candidate(
        self,
        base_vector: np.ndarray,
        scale: float,
        jitter: float = 0.0,
    ) -> np.ndarray:
        candidate = normalize(base_vector) * scale
        if jitter > 0.0:
            candidate = normalize(
                candidate + self.numpy_rng.normal(0.0, jitter, size=DEMO_GRADIENT_DIM)
            ) * scale
        return candidate.astype(float)

    def _evaluate_candidate(
        self,
        vector: np.ndarray,
        honest_reference: np.ndarray | None,
    ) -> CandidateMetrics:
        result = self.auditor.audit("candidate", vector.copy())
        deviation = (
            cosine_deviation_score(vector, honest_reference)
            if honest_reference is not None
            else 0.0
        )
        return CandidateMetrics(
            classification=result.classification,
            delta_main=result.delta_loss_main,
            delta_corner=result.delta_loss_corner,
            deviation=deviation,
        )

    def _matches_target(
        self,
        role: str,
        metrics: CandidateMetrics,
        policy: Policy,
        vector: np.ndarray | None = None,
    ) -> bool:
        fraud_cutoff = policy.cosine_filter_threshold + max(SUSPECT_MARGIN * 0.20, 0.005)
        rarity_cutoff = policy.cosine_filter_threshold + max(SUSPECT_MARGIN * 0.35, 0.01)
        noise_cutoff = policy.cosine_filter_threshold + max(SUSPECT_MARGIN * 0.35, 0.012)

        if self.ground_truth_mode == "archetype" and vector is not None:
            main_alignment = cosine_similarity(vector, self.main_gradient)
            corner_alignment = cosine_similarity(vector, self.corner_gradient)
            magnitude = float(np.linalg.norm(vector))

            if role == "HONEST":
                return (
                    main_alignment > 0.55
                    and metrics.deviation < max(policy.cosine_filter_threshold - 0.05, 0.35)
                    and 0.6 <= magnitude <= 2.5
                )

            if role == "FRAUD":
                # Sign-flip family: accept only candidates that cause measured
                # main-task damage. Direction reversal alone is not enough.
                return (
                    main_alignment < -0.20
                    and metrics.delta_main >= max(policy.theta_tol * 1.2, 0.06)
                )

            if role == "FRAUD_CORNER_HARM":
                # Corner-harm family: measured corner damage with main-task
                # damage kept well below the fraud threshold.
                return (
                    main_alignment > 0.20
                    and metrics.delta_main <= max(policy.theta_tol * 0.5, 0.025)
                    and metrics.delta_corner >= max(abs(policy.theta_rare) * 1.2, 0.03)
                )

            if role == "RARITY":
                return (
                    corner_alignment > 0.45
                    and main_alignment < 0.35
                    and metrics.delta_corner <= policy.theta_rare - RARITY_TARGET_MARGIN
                    and metrics.delta_main <= policy.theta_tol * 0.5
                    and metrics.deviation > rarity_cutoff
                )

            # NOISE
            return (
                abs(main_alignment) < 0.55
                and abs(corner_alignment) < 0.65
                and metrics.deviation > noise_cutoff
            )

        # preflight ground-truth mode: defer to L2 classification
        if role == "HONEST":
            return (
                metrics.classification == Classification.HONEST
                and metrics.deviation < max(policy.cosine_filter_threshold - 0.05, 0.35)
            )

        if role == "FRAUD":
            return (
                metrics.classification == Classification.FRAUD
                and metrics.deviation > fraud_cutoff
            )

        if role == "FRAUD_CORNER_HARM":
            # In preflight mode we still expect L2 to flag this as FRAUD; the
            # difference from sign-flip FRAUD is downstream (attack_family
            # metadata + per-family metrics), not in the verdict label.
            return metrics.classification == Classification.FRAUD

        if role == "RARITY":
            return (
                metrics.classification == Classification.RARITY
                and metrics.delta_corner <= policy.theta_rare - RARITY_TARGET_MARGIN
                and metrics.delta_main < max(policy.theta_tol * 0.25, 0.01)
                and metrics.deviation > rarity_cutoff
            )

        return (
            metrics.classification == Classification.NOISE
            and metrics.deviation > noise_cutoff
        )

    def _candidate_score(
        self,
        role: str,
        metrics: CandidateMetrics,
        policy: Policy,
    ) -> float:
        fraud_cutoff = policy.cosine_filter_threshold + max(SUSPECT_MARGIN * 0.20, 0.005)
        rarity_cutoff = policy.cosine_filter_threshold + max(SUSPECT_MARGIN * 0.35, 0.01)
        noise_cutoff = policy.cosine_filter_threshold + max(SUSPECT_MARGIN * 0.35, 0.012)

        if role == "HONEST":
            return -metrics.delta_main - max(
                metrics.deviation - max(policy.cosine_filter_threshold - 0.05, 0.35),
                0.0,
            )

        if role == "FRAUD":
            return (
                (metrics.delta_main - policy.theta_tol) * 6.0
                + (metrics.deviation - fraud_cutoff) * 1.5
            )

        if role == "FRAUD_CORNER_HARM":
            # Reward solutions that hurt corner loss while keeping main neutral.
            return (
                metrics.delta_corner * 5.0
                - max(metrics.delta_main, 0.0) * 8.0
            )

        if role == "RARITY":
            return (
                (policy.theta_rare - metrics.delta_corner) * 5.0
                + max(-metrics.delta_main, -0.02) * 2.5
                + (metrics.deviation - rarity_cutoff)
            )

        return (
            (metrics.deviation - noise_cutoff) * 2.0
            - abs(metrics.delta_main) * 6.0
            - abs(metrics.delta_corner) * 2.5
        )

    def _rarity_precheck_passed(
        self,
        metrics: CandidateMetrics,
        policy: Policy,
    ) -> bool:
        """Independent semantic precheck before the stronger rarity acceptance rule."""
        return metrics.delta_main <= policy.theta_tol and metrics.delta_corner <= policy.theta_rare

    def _record_rarity_generation(
        self,
        *,
        round_id: int,
        candidate_id: str,
        stage: str,
        vector: np.ndarray,
        metrics: CandidateMetrics,
        policy: Policy,
        accepted_as_rarity: bool,
        vehicle_id: str | None = None,
    ) -> None:
        self.current_rarity_generation_trace.append({
            "round_id": round_id,
            "candidate_id": candidate_id,
            "stage": stage,
            "vehicle_id": vehicle_id,
            "corner_dominant_score": cosine_similarity(vector, self.corner_gradient),
            "main_suppression_score": -cosine_similarity(vector, self.main_gradient),
            "precheck_delta_l_main": metrics.delta_main,
            "precheck_delta_l_corner": metrics.delta_corner,
            "passed_precheck": self._rarity_precheck_passed(metrics, policy),
            "accepted_as_rarity": accepted_as_rarity,
            "theta_tol_used": policy.theta_tol,
            "theta_rare_used": policy.theta_rare,
            "rarity_cutoff_used": policy.cosine_filter_threshold + max(SUSPECT_MARGIN * 0.35, 0.01),
            "deviation_used": metrics.deviation,
            "ground_truth_mode": self.ground_truth_mode,
        })

    def _label_for_projection(self, payload: dict[str, Any]) -> str:
        if self.ground_truth_mode == "archetype":
            return str(
                payload["metadata"].get(
                    "ground_truth_role",
                    payload["metadata"].get("planned_role", payload["metadata"].get("preflight_role")),
                )
            )
        return str(payload["metadata"].get("preflight_role", payload["metadata"].get("planned_role")))

    # ------------------------------------------------------------------
    # Per-archetype anchor materialization
    # ------------------------------------------------------------------

    def _materialize_gradient(
        self,
        role: str,
        anchor: np.ndarray,
        policy: Policy,
        honest_reference: np.ndarray | None,
        vehicle_address: str,
        round_index: int,
        sample_count: int,
    ) -> tuple[np.ndarray, CandidateMetrics]:
        anchor_scale = float(np.linalg.norm(anchor))
        style_vector = self._vehicle_style_vector(vehicle_address)
        round_drift = self._round_drift_vector(round_index)
        count_scale = self._sample_count_scale(sample_count)

        # Role-specific client style jitter. Attack anchors are already
        # energy-calibrated; this step adds client-level variation without
        # redefining their ground truth.
        role_mix = {
            "HONEST":            (0.12, 0.10, 0.04, 0.95, 1.05),
            "FRAUD":             (0.10, 0.18, 0.08, 1.00, 1.24),
            "FRAUD_CORNER_HARM": (0.08, 0.10, 0.03, 0.95, 1.10),
            "RARITY":            (0.04, 0.05, 0.01, 0.98, 1.05),
            "NOISE":             (0.12, 0.18, 0.10, 1.00, 1.22),
        }
        style_weight, drift_weight, residual_weight, min_factor, max_factor = role_mix[role]

        anchor_direction = normalize(anchor)
        anchor_metrics = self._evaluate_candidate(anchor, honest_reference)
        last_metrics = anchor_metrics

        for attempt in range(ROLE_MATERIALIZE_RETRIES):
            residual = self._basis_from_key(
                f"{vehicle_address}:{round_index}:{role}:attempt:{attempt}"
            )
            direction = normalize(
                anchor_direction
                + style_weight * style_vector
                + drift_weight * round_drift
                + residual_weight * residual
            )
            scale_factor = self.random.uniform(min_factor, max_factor) * count_scale
            candidate = direction * anchor_scale * scale_factor
            metrics = self._evaluate_candidate(candidate, honest_reference)
            accepted = self._matches_target(role, metrics, policy, vector=candidate)
            if role == "RARITY":
                self._record_rarity_generation(
                    round_id=policy.round_id,
                    candidate_id=f"materialize:{vehicle_address}:{attempt}",
                    stage="materialize",
                    vector=candidate,
                    metrics=metrics,
                    policy=policy,
                    accepted_as_rarity=accepted,
                    vehicle_id=vehicle_address,
                )
            if accepted:
                return candidate.astype(float), metrics
            last_metrics = metrics

        return anchor.copy(), anchor_metrics

    def _update_preflight_metadata(
        self,
        payload: dict[str, Any],
        metrics: CandidateMetrics,
        is_l1_suspect: bool | None = None,
    ) -> None:
        payload["metadata"].update({
            "preflight_role": metrics.classification.value,
            "preflight_delta_main": metrics.delta_main,
            "preflight_delta_corner": metrics.delta_corner,
            "preflight_deviation": metrics.deviation,
        })
        if is_l1_suspect is not None:
            payload["metadata"]["l1_preflight_suspect"] = is_l1_suspect

    def _local_l1_summary(
        self,
        updates: list[dict[str, Any]],
        policy: Policy,
    ) -> tuple[Any, Counter]:
        gradients = [
            np.array(payload["gradient_data"], dtype=float)
            for payload in updates
        ]
        vehicles = [payload["vehicle_address"] for payload in updates]
        result = filter_suspects(
            gradients,
            vehicles,
            threshold=policy.cosine_filter_threshold,
        )
        suspect_roles = Counter(
            self._label_for_projection(updates[idx])
            for idx in result.suspect_indices
        )
        return result, suspect_roles

    def _refresh_l1_projection_flags(
        self,
        updates: list[dict[str, Any]],
        result: Any,
    ) -> None:
        suspect_indices = set(result.suspect_indices)
        for idx, payload in enumerate(updates):
            payload["metadata"]["l1_preflight_suspect"] = idx in suspect_indices

    # ------------------------------------------------------------------
    # Anchor banks
    # ------------------------------------------------------------------

    def _find_signflip_gradient(
        self,
        policy: Policy,
        honest_reference: np.ndarray,
        round_drift: np.ndarray,
    ) -> np.ndarray | None:
        """Search for a sign-flip fraud anchor that exceeds the main-loss target."""
        self.auditor.apply_policy(policy)

        target_delta_main = max(policy.theta_tol * 1.5, 0.075)
        base_directions = [
            -1.42 * self.main_gradient - 0.24 * self.corner_gradient + 0.30 * round_drift,
            -1.18 * self.main_gradient + 0.22 * self.corner_gradient
            + 0.46 * self._basis_from_key("fraud-sideband"),
            -(1.08 * self.main_gradient + 0.30 * self.corner_gradient)
            + 0.42 * round_drift + 0.26 * self.random_gradient,
            -1.25 * self.main_gradient + 0.40 * self._basis_from_key("fraud-drift"),
        ]
        scales = [10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0]

        fallback: np.ndarray | None = None
        fallback_delta_main = -float("inf")
        fallback_scale: float | None = None

        for base in base_directions:
            for scale in scales:
                candidate = self._make_candidate(base, scale)
                metrics = self._evaluate_candidate(candidate, honest_reference)
                main_alignment = cosine_similarity(candidate, self.main_gradient)

                if metrics.delta_main >= target_delta_main and main_alignment < -0.20:
                    logger.info(
                        "[FRAUD anchor] scale=%.1f dL_main=%.4f main_align=%.3f norm=%.2f",
                        scale,
                        metrics.delta_main,
                        main_alignment,
                        float(np.linalg.norm(candidate)),
                    )
                    self.current_attack_anchor_scale_by_role["FRAUD"] = scale
                    return candidate

                if metrics.delta_main > fallback_delta_main and main_alignment < -0.10:
                    fallback_delta_main = metrics.delta_main
                    fallback = candidate
                    fallback_scale = scale

        if fallback is not None:
            logger.warning(
                "[FRAUD anchor] no candidate hit target dL_main=%.4f; best fallback dL_main=%.4f",
                target_delta_main,
                fallback_delta_main,
            )
            if fallback_scale is not None:
                self.current_attack_anchor_scale_by_role["FRAUD"] = fallback_scale
        return fallback

    def _find_gradient_for_role(
        self,
        role: str,
        policy: Policy,
        honest_reference: np.ndarray | None = None,
        round_drift: np.ndarray | None = None,
    ) -> np.ndarray | None:
        self.auditor.apply_policy(policy)
        round_drift = round_drift if round_drift is not None else self.random_gradient

        if role == "HONEST":
            bases = [
                self.main_gradient + 0.12 * self.corner_gradient + 0.08 * round_drift,
                0.92 * self.main_gradient + 0.18 * self.corner_gradient + 0.10 * round_drift,
                self.main_gradient + 0.10 * self._basis_from_key("honest-microbatch"),
            ]
            scales = [0.80, 1.00, 1.20, 1.40, 1.80, 2.20]
        elif role == "FRAUD":
            if honest_reference is None:
                return None
            return self._find_signflip_gradient(policy, honest_reference, round_drift)
        elif role == "RARITY":
            bases = [
                1.85 * self.corner_gradient - 0.78 * self.main_gradient
                + 0.34 * self._basis_from_key("rarity-outlier-core"),
                1.65 * self.corner_gradient - 0.28 * self.main_gradient + 0.26 * round_drift,
                1.78 * self.corner_gradient - 0.38 * self.main_gradient
                + 0.30 * self.random_gradient + 0.24 * round_drift,
                1.42 * self.corner_gradient + 0.42 * self._basis_from_key("rarity-support")
                - 0.30 * self.main_gradient + 0.28 * round_drift,
            ]
            scales = [10.00, 12.00, 14.00, 16.00, 18.00, 20.00, 24.00, 28.00]
        else:  # NOISE
            bases = [
                1.10 * self.random_gradient + 0.35 * round_drift,
                -1.05 * self.random_gradient + 0.32 * round_drift,
                0.85 * self.random_gradient + 0.32 * self.corner_gradient + 0.42 * round_drift,
                self._basis_from_key("noise-sideband") + 0.18 * self.main_gradient + 0.34 * round_drift,
            ]
            scales = [2.00, 3.00, 4.00, 5.50, 7.00, 8.50, 10.00]

        fallback: np.ndarray | None = None
        fallback_score = float("-inf")
        rarity_cutoff = policy.cosine_filter_threshold + max(SUSPECT_MARGIN * 0.35, 0.01)

        for base_index, base in enumerate(bases):
            for scale in scales:
                candidate = self._make_candidate(base, scale)
                metrics = self._evaluate_candidate(candidate, honest_reference)
                accepted = self._matches_target(role, metrics, policy, vector=candidate)
                if role == "RARITY":
                    self._record_rarity_generation(
                        round_id=policy.round_id,
                        candidate_id=f"bank:{base_index}:scale:{scale:.2f}",
                        stage="bank_search",
                        vector=candidate,
                        metrics=metrics,
                        policy=policy,
                        accepted_as_rarity=accepted,
                    )
                if accepted:
                    return candidate
                if self.ground_truth_mode == "archetype" and role == "RARITY":
                    main_alignment = cosine_similarity(candidate, self.main_gradient)
                    corner_alignment = cosine_similarity(candidate, self.corner_gradient)
                    score = (
                        corner_alignment * 2.0
                        - max(main_alignment, 0.0) * 1.5
                        + (metrics.deviation - rarity_cutoff)
                    )
                elif role in {"FRAUD", "NOISE"}:
                    score = self._candidate_score(role, metrics, policy)
                elif metrics.classification == Classification[role]:
                    score = self._candidate_score(role, metrics, policy)
                else:
                    continue

                if score > fallback_score:
                    fallback = candidate
                    fallback_score = score

        return fallback

    def _find_corner_harm_gradient(
        self,
        policy: Policy,
        honest_reference: np.ndarray,
        round_drift: np.ndarray,
    ) -> np.ndarray | None:
        """Search for a corner-harm anchor with measured corner damage."""
        self.auditor.apply_policy(policy)

        target_delta_corner = max(abs(policy.theta_rare) * 1.5, 0.04)
        main_safe_threshold = policy.theta_tol * 0.5

        gamma_options = [0.6, 0.9, 1.2, 1.5, 1.8]
        main_alpha_options = [0.85, 1.00, 1.15]
        sideband_keys = ["corner-harm-A", "corner-harm-B", "corner-harm-C"]

        bases: list[np.ndarray] = []
        for main_alpha in main_alpha_options:
            for gamma in gamma_options:
                for sideband_key in sideband_keys:
                    bases.append(
                        main_alpha * self.main_gradient
                        - gamma * self.corner_gradient
                        + 0.15 * self._basis_from_key(sideband_key)
                        + 0.10 * round_drift
                    )

        scales = [2.0, 5.0, 10.0, 20.0, 40.0, 80.0]

        fallback: np.ndarray | None = None
        fallback_score = float("-inf")
        fallback_scale: float | None = None

        for base in bases:
            for scale in scales:
                candidate = self._make_candidate(base, scale)
                metrics = self._evaluate_candidate(candidate, honest_reference)
                main_alignment = cosine_similarity(candidate, self.main_gradient)

                corner_hurt = metrics.delta_corner >= target_delta_corner
                main_safe = metrics.delta_main <= main_safe_threshold
                main_alignment_ok = main_alignment > 0.20

                if corner_hurt and main_safe and main_alignment_ok:
                    logger.info(
                        "[CORNER_HARM anchor] scale=%.1f dL_main=%.4f dL_corner=%.4f norm=%.2f",
                        scale,
                        metrics.delta_main,
                        metrics.delta_corner,
                        float(np.linalg.norm(candidate)),
                    )
                    self.current_attack_anchor_scale_by_role["FRAUD_CORNER_HARM"] = scale
                    return candidate

                if main_safe and main_alignment_ok and metrics.delta_corner > fallback_score:
                    fallback_score = metrics.delta_corner
                    fallback = candidate
                    fallback_scale = scale

        if fallback is not None:
            logger.warning(
                "[CORNER_HARM anchor] no candidate hit target dL_corner=%.4f; "
                "best fallback dL_corner=%.4f",
                target_delta_corner,
                fallback_score,
            )
            if fallback_scale is not None:
                self.current_attack_anchor_scale_by_role["FRAUD_CORNER_HARM"] = fallback_scale

        return fallback

    def _build_gradient_bank(self, policy: Policy, round_index: int) -> dict[str, np.ndarray]:
        self.current_attack_anchor_validation = {}
        self.current_attack_anchor_scale_by_role = {}
        round_drift = self._round_drift_vector(round_index)
        honest = self._find_gradient_for_role("HONEST", policy, round_drift=round_drift)
        if honest is None:
            raise RuntimeError("Could not find an HONEST demo gradient for the current policy")

        bank: dict[str, np.ndarray] = {"HONEST": honest}

        fraud_anchor = self._find_signflip_gradient(policy, honest, round_drift)
        if fraud_anchor is None:
            logger.warning("falling back to HONEST gradient for missing FRAUD sample")
            fraud_anchor = honest
        bank["FRAUD"] = fraud_anchor
        fraud_metrics = self._evaluate_candidate(fraud_anchor, honest)
        fraud_target = max(policy.theta_tol * 1.5, 0.075)
        self.current_attack_anchor_validation["FRAUD"] = {
            "attack_family": ARCHETYPE_TO_ATTACK_FAMILY["FRAUD"],
            "target_condition": "delta_main >= max(1.5*theta_tol, 0.075)",
            "target_delta": fraud_target,
            "target_passed": (
                fraud_metrics.delta_main >= fraud_target
                and cosine_similarity(fraud_anchor, self.main_gradient) < -0.20
            ),
            "anchor_delta_main": fraud_metrics.delta_main,
            "anchor_delta_corner": fraud_metrics.delta_corner,
            "anchor_search_scale_used": self.current_attack_anchor_scale_by_role.get("FRAUD"),
            "anchor_scale_used": float(np.linalg.norm(fraud_anchor)),
        }

        for role in ("RARITY", "NOISE"):
            candidate = self._find_gradient_for_role(
                role,
                policy,
                honest_reference=honest,
                round_drift=round_drift,
            )
            if candidate is None:
                if role == "NOISE":
                    logger.info("reusing HONEST anchor for missing %s sample", role)
                else:
                    logger.warning("falling back to HONEST gradient for missing %s sample", role)
                candidate = honest
            bank[role] = candidate

        # Corner-harm anchor: stealth-magnitude attack that targets ΔL_corner.
        corner_harm = self._find_corner_harm_gradient(policy, honest, round_drift)
        if corner_harm is None:
            logger.warning(
                "no FRAUD_CORNER_HARM candidate satisfied target; falling back to FRAUD anchor"
            )
            corner_harm = bank["FRAUD"]
        bank["FRAUD_CORNER_HARM"] = corner_harm
        corner_harm_metrics = self._evaluate_candidate(corner_harm, honest)
        corner_harm_target = max(abs(policy.theta_rare) * 1.5, 0.04)
        corner_main_safe = policy.theta_tol * 0.5
        self.current_attack_anchor_validation["FRAUD_CORNER_HARM"] = {
            "attack_family": ARCHETYPE_TO_ATTACK_FAMILY["FRAUD_CORNER_HARM"],
            "target_condition": (
                "delta_corner >= max(1.5*abs(theta_rare), 0.04) "
                "and delta_main <= 0.5*theta_tol"
            ),
            "target_delta": corner_harm_target,
            "target_main_safe": corner_main_safe,
            "target_passed": (
                corner_harm_metrics.delta_corner >= corner_harm_target
                and corner_harm_metrics.delta_main <= corner_main_safe
            ),
            "anchor_delta_main": corner_harm_metrics.delta_main,
            "anchor_delta_corner": corner_harm_metrics.delta_corner,
            "anchor_search_scale_used": self.current_attack_anchor_scale_by_role.get(
                "FRAUD_CORNER_HARM"
            ),
            "anchor_scale_used": float(np.linalg.norm(corner_harm)),
        }

        return bank

    # ------------------------------------------------------------------
    # Profile expansion / vehicle selection
    # ------------------------------------------------------------------

    def _expand_profile(self, profile: dict[str, float], phase_name: str) -> list[str]:
        minimums = self._minimum_role_counts(phase_name)
        counts = {role: minimums.get(role, 0) for role in ROLES_ORDER}

        remaining = max(BATCH_SIZE - sum(counts.values()), 0)
        total_weight = sum(max(profile.get(role, 0.0), 0.0) for role in ROLES_ORDER)
        if total_weight <= 0.0:
            weights = {role: (1.0 if role == "HONEST" else 0.0) for role in ROLES_ORDER}
        else:
            weights = {
                role: max(profile.get(role, 0.0), 0.0) / total_weight
                for role in ROLES_ORDER
            }

        fractional_parts: dict[str, float] = {}
        for role in ROLES_ORDER:
            ideal = weights[role] * remaining
            base = int(np.floor(ideal))
            counts[role] += base
            fractional_parts[role] = ideal - base

        leftover = BATCH_SIZE - sum(counts.values())
        # Tie-break order favours the categories the benchmark most needs to
        # observe. RARITY first, then FRAUD_CORNER_HARM (rare and important),
        # then plain FRAUD, then NOISE, then HONEST.
        tie_priority = {
            "RARITY": 0,
            "FRAUD_CORNER_HARM": 1,
            "FRAUD": 2,
            "NOISE": 3,
            "HONEST": 4,
        }
        allocation_order = sorted(
            ROLES_ORDER,
            key=lambda role: (-fractional_parts[role], tie_priority[role]),
        )
        for role in allocation_order:
            if leftover <= 0:
                break
            counts[role] += 1
            leftover -= 1

        while sum(counts.values()) < BATCH_SIZE:
            counts["HONEST"] += 1

        # Drop excess in reverse-priority order (HONEST first, FRAUD_CORNER_HARM
        # last) so the rare categories survive truncation.
        drop_order = ("HONEST", "NOISE", "RARITY", "FRAUD", "FRAUD_CORNER_HARM")
        while sum(counts.values()) > BATCH_SIZE:
            shrunk = False
            for role in drop_order:
                if counts[role] > minimums.get(role, 0) and sum(counts.values()) > BATCH_SIZE:
                    counts[role] -= 1
                    shrunk = True
            if not shrunk:
                break

        roles: list[str] = []
        for role in ROLES_ORDER:
            roles.extend([role] * counts[role])
        self.random.shuffle(roles)
        return roles

    def _select_vehicles(self, round_index: int) -> tuple[list[str], int]:
        if self.ground_truth_mode == "archetype" and BATCH_SIZE <= len(self.vehicle_pool):
            addresses = self.random.sample(self.vehicle_pool, BATCH_SIZE)
            new_vehicle_count = sum(address not in self.seen_vehicles for address in addresses)
            self.seen_vehicles.update(addresses)
            return addresses, new_vehicle_count

        start = (round_index * BATCH_SIZE) % len(self.vehicle_pool)
        addresses = [
            self.vehicle_pool[(start + offset) % len(self.vehicle_pool)]
            for offset in range(BATCH_SIZE)
        ]
        new_vehicle_count = sum(address not in self.seen_vehicles for address in addresses)
        self.seen_vehicles.update(addresses)
        return addresses, new_vehicle_count

    # ------------------------------------------------------------------
    # Batch construction
    # ------------------------------------------------------------------

    def _build_batch(
        self,
        round_id: int,
        policy: Policy,
        round_index: int,
    ) -> tuple[list[dict[str, Any]], Counter, list[str], int, Counter, Counter]:
        self.current_rarity_generation_trace = []
        bank = self._build_gradient_bank(policy, round_index)
        phase_name = self._phase_name(round_index)
        roles = self._expand_profile(self._profile_for_round(round_index), phase_name)
        vehicle_addresses, new_vehicle_count = self._select_vehicles(round_index)

        updates: list[dict[str, Any]] = []
        role_counter: Counter = Counter()
        preflight_counter: Counter = Counter()
        honest_reference = bank["HONEST"]

        for index, (role, vehicle_address) in enumerate(zip(roles, vehicle_addresses)):
            sample_count = self.random.randint(SAMPLE_COUNT_MIN, SAMPLE_COUNT_MAX)
            gradient, metrics = self._materialize_gradient(
                role=role,
                anchor=bank[role],
                policy=policy,
                honest_reference=honest_reference,
                vehicle_address=vehicle_address,
                round_index=round_index,
                sample_count=sample_count,
            )

            role_counter[role] += 1
            preflight_counter[metrics.classification.value] += 1
            updates.append({
                "vehicle_address": vehicle_address,
                "gradient_data": gradient.tolist(),
                "data_sample_count": sample_count,
                "metadata": {
                    "round_id": round_id,
                    "planned_role": role,
                    "ground_truth_role": role,
                    "ground_truth_label": ARCHETYPE_TO_GROUND_TRUTH_LABEL[role],
                    "attack_family": ARCHETYPE_TO_ATTACK_FAMILY[role],
                    "gradient_dim": DEMO_GRADIENT_DIM,
                    "index": index,
                },
            })
            if role in self.current_attack_anchor_validation:
                updates[-1]["metadata"].update(
                    self.current_attack_anchor_validation[role]
                )
            self._update_preflight_metadata(updates[-1], metrics)

        # Read-only L1 projection. We compute what L1 would do for diagnostic
        # logging only; we never reshape gradients to satisfy L1's rule. This
        # is the explicit removal of the previous _promote_rarity_visibility
        # mechanism, which iteratively rewrote rarity and honest gradients
        # until L1 produced a target routing pattern. That made benchmark L1
        # statistics construction artefacts rather than discriminator output.
        result, l1_suspect_roles = self._local_l1_summary(updates, policy)
        self._refresh_l1_projection_flags(updates, result)

        refreshed_preflight = Counter(
            payload["metadata"]["preflight_role"]
            for payload in updates
        )
        return (
            updates,
            role_counter,
            vehicle_addresses,
            new_vehicle_count,
            refreshed_preflight,
            l1_suspect_roles,
        )

    # ------------------------------------------------------------------
    # Submission and audit polling
    # ------------------------------------------------------------------

    def _submit_gradient(self, payload: dict[str, Any]) -> bool:
        try:
            response = self.session.post(
                f"{L1_API_URL}/api/v1/gradients",
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            if response.status_code == 200:
                return True
            logger.error(
                "submit failed for %s: HTTP %s %s",
                payload["vehicle_address"][:12],
                response.status_code,
                response.text,
            )
        except Exception as exc:
            logger.error("submit failed for %s: %s", payload["vehicle_address"][:12], exc)
        return False

    def _wait_for_round_audits(
        self,
        started_at: datetime,
        vehicle_addresses: list[str],
        expected_suspects: int,
    ) -> list[dict[str, Any]]:
        deadline = time.time() + PROCESS_WAIT_SECONDS + 20.0
        relevant_addresses = set(vehicle_addresses)
        filtered: list[dict[str, Any]] = []

        while time.time() < deadline:
            try:
                audits, source, fallback_reason = self._fetch_recent_audits(limit=100)
                if source != self.last_recent_audit_source:
                    if fallback_reason:
                        logger.warning("recent audits source switched to %s: %s", source, fallback_reason)
                    logger.info("recent audits source: %s", source)
                    self.last_recent_audit_source = source
            except Exception as exc:
                logger.warning("failed to poll recent audits: %s", exc)
                time.sleep(2.0)
                continue

            filtered = []
            seen = set()
            for audit in audits:
                vehicle_id = audit.get("vehicle_id")
                timestamp = audit.get("timestamp")
                if vehicle_id not in relevant_addresses or not timestamp:
                    continue
                try:
                    audit_time = datetime.fromisoformat(timestamp)
                except ValueError:
                    continue
                if audit_time < started_at:
                    continue
                unique_key = (vehicle_id, timestamp)
                if unique_key in seen:
                    continue
                seen.add(unique_key)
                filtered.append(audit)

            unique_audited = {audit["vehicle_id"] for audit in filtered}
            if len(unique_audited) >= expected_suspects:
                return filtered

            time.sleep(2.0)

        return filtered

    def _fetch_recent_audits(self, limit: int) -> tuple[list[dict[str, Any]], str, str | None]:
        try:
            response = self.session.get(
                f"{L4_API_URL}/api/v1/recent-audits",
                params={"limit": limit},
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            return response.json(), "l4_api", None
        except Exception as api_exc:
            if not self.redis_client:
                raise RuntimeError(f"L4 API unavailable and Redis fallback is not configured: {api_exc}") from api_exc

            try:
                raw_audits = self.redis_client.lrange("recent_audits", 0, limit - 1)
                audits = [json.loads(item) for item in raw_audits]
                return audits, "redis", str(api_exc)
            except Exception as redis_exc:
                raise RuntimeError(
                    f"L4 API unavailable ({api_exc}); Redis fallback failed ({redis_exc})"
                ) from redis_exc

    def _read_suspect_queue_length(self) -> int:
        if not self.redis_client:
            return 0

        try:
            return max(int(self.redis_client.llen(L2_AUDIT_QUEUE) or 0), 0)
        except redis.RedisError as exc:
            logger.warning("failed to read suspect queue length: %s", exc)
            return 0

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def _build_telemetry(
        self,
        round_id: int,
        batch_size: int,
        audits: list[dict[str, Any]],
        planned_roles: Counter,
        new_vehicle_count: int,
        suspect_queue_length: int,
        updates: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        unique_audits: dict[str, dict[str, Any]] = {}
        for audit in audits:
            unique_audits[audit["vehicle_id"]] = audit

        counts = Counter(audit["classification"] for audit in unique_audits.values())
        audited_count = len(unique_audits)
        clean_honest = max(batch_size - audited_count, 0)

        fraud_count = counts.get("FRAUD", 0)
        rarity_count = counts.get("RARITY", 0)
        honest_count = counts.get("HONEST", 0) + clean_honest
        noise_count = counts.get("NOISE", 0)

        total = max(batch_size, 1)
        has_routing_reasons = any("routing_reason" in audit for audit in unique_audits.values())
        if has_routing_reasons:
            cosine_routed_count = sum(
                1
                for audit in unique_audits.values()
                if audit.get("routing_reason") == "cosine_screening"
            )
            recheck_routed_count = sum(
                1
                for audit in unique_audits.values()
                if audit.get("routing_reason") == "probabilistic_recheck"
            )
        else:
            cosine_routed_count = audited_count
            recheck_routed_count = 0
        delta_main_avg = (
            float(np.mean([audit["delta_loss_main"] for audit in unique_audits.values()]))
            if unique_audits
            else 0.0
        )
        delta_corner_avg = (
            float(np.mean([audit["delta_loss_corner"] for audit in unique_audits.values()]))
            if unique_audits
            else 0.0
        )

        sbt_total = sum(int(audit["sbt_points"]) for audit in unique_audits.values()) + clean_honest
        rarity_targets = max(planned_roles.get("RARITY", 0), 1)
        planned_fraud_total = sum(planned_roles.get(role, 0) for role in FRAUD_ROLES)

        fraud_rate = fraud_count / total
        rarity_rate = rarity_count / total
        honest_rate = honest_count / total
        noise_rate = noise_count / total

        telemetry: dict[str, Any] = {
            "round_id": round_id,
            "fraud_rate": fraud_rate,
            "rarity_rate": rarity_rate,
            "honest_rate": honest_rate,
            "noise_rate": noise_rate,
            "audit_sample_size": audited_count,
            "main_accuracy": clamp(
                0.88 + honest_rate * 0.08 + rarity_rate * 0.04 - fraud_rate * 0.25 - noise_rate * 0.12,
                0.45,
                0.99,
            ),
            "corner_accuracy": clamp(
                0.58 + rarity_rate * 0.30 + honest_rate * 0.05 - fraud_rate * 0.08 - noise_rate * 0.03,
                0.20,
                0.97,
            ),
            "main_loss_delta_avg": delta_main_avg,
            "corner_loss_delta_avg": delta_corner_avg,
            "false_slash_estimate": clamp(counts.get("HONEST", 0) / total, 0.0, 1.0),
            "rarity_retention_rate": clamp(rarity_count / rarity_targets, 0.0, 1.0),
            "golden_drift_score": clamp(
                abs(delta_main_avg) * 1.4 + fraud_rate * 0.6 + noise_rate * 0.25,
                0.0,
                1.0,
            ),
            "reject_rate_l3": 0.0,
            "cosine_outlier_ratio": cosine_routed_count / total,
            "l1_recheck_ratio": recheck_routed_count / total,
            "suspect_queue_length": max(suspect_queue_length, 0),
            "avg_sbt_score": max(0.0, sbt_total / total),
            "new_vehicle_ratio": new_vehicle_count / total,
            "hash_mismatch_rate": 0.0,
            "recent_attack_pressure": clamp(
                fraud_rate * 1.2 + noise_rate * 0.5 + planned_fraud_total / total * 0.2,
                0.0,
                1.0,
            ),
        }

        # ------------------------------------------------------------------
        # Per-attack-family decomposition (new in improved benchmark version).
        # ------------------------------------------------------------------

        updates = updates or []
        vehicle_to_family = {
            payload["vehicle_address"]: payload["metadata"].get("attack_family", "none")
            for payload in updates
        }
        vehicle_to_planned = {
            payload["vehicle_address"]: payload["metadata"]["planned_role"]
            for payload in updates
        }
        vehicle_to_gt_label = {
            payload["vehicle_address"]: payload["metadata"].get(
                "ground_truth_label",
                payload["metadata"]["planned_role"],
            )
            for payload in updates
        }

        # Fraud survival per family. A fraud update "survives" when its L2
        # verdict is not FRAUD, including the case where L1 never escalated it
        # so no audit was produced.
        family_stats: dict[str, dict[str, int]] = {}
        for family in set(vehicle_to_family.values()):
            if family == "none":
                continue
            family_stats[family] = {"survived": 0, "caught": 0, "total": 0}

        audited_vehicles = set(unique_audits.keys())
        fraud_caught_by_routing_reason: Counter = Counter()
        routed_by_reason: Counter = Counter()
        for vehicle_id, family in vehicle_to_family.items():
            if family == "none":
                continue
            family_stats[family]["total"] += 1
            audit = unique_audits.get(vehicle_id)
            if audit is None:
                family_stats[family]["survived"] += 1
            elif audit["classification"] == "FRAUD":
                family_stats[family]["caught"] += 1
                reason = audit.get("routing_reason", "unknown")
                fraud_caught_by_routing_reason[reason] += 1
            else:
                family_stats[family]["survived"] += 1

        for audit in unique_audits.values():
            routed_by_reason[audit.get("routing_reason", "unknown")] += 1

        telemetry["fraud_survival_by_family"] = {
            family: (
                stats["survived"] / stats["total"]
                if stats["total"] > 0 else 0.0
            )
            for family, stats in family_stats.items()
        }
        telemetry["fraud_caught_by_family"] = {
            family: stats["caught"] for family, stats in family_stats.items()
        }
        telemetry["fraud_total_by_family"] = {
            family: stats["total"] for family, stats in family_stats.items()
        }
        telemetry["fraud_caught_by_routing_reason"] = dict(fraud_caught_by_routing_reason)
        telemetry["l1_routed_by_reason"] = dict(routed_by_reason)

        # L1 routing recall per archetype (was a vehicle escalated to L2?).
        l1_routing_per_archetype: dict[str, dict[str, int]] = {}
        for vehicle_id, planned in vehicle_to_planned.items():
            bucket = l1_routing_per_archetype.setdefault(
                planned, {"routed": 0, "total": 0}
            )
            bucket["total"] += 1
            if vehicle_id in audited_vehicles:
                bucket["routed"] += 1

        telemetry["l1_routing_by_archetype"] = {
            role: (
                stats["routed"] / stats["total"]
                if stats["total"] > 0 else 0.0
            )
            for role, stats in l1_routing_per_archetype.items()
        }
        telemetry["l1_routed_count_by_archetype"] = {
            role: stats["routed"] for role, stats in l1_routing_per_archetype.items()
        }
        telemetry["l1_total_count_by_archetype"] = {
            role: stats["total"] for role, stats in l1_routing_per_archetype.items()
        }

        # L2 confusion matrix: ground-truth label (4 classes) vs verdict.
        confusion: dict[str, int] = {}
        for vehicle_id, audit in unique_audits.items():
            gt = vehicle_to_gt_label.get(vehicle_id, "UNKNOWN")
            verdict = audit["classification"]
            key = f"{gt}->{verdict}"
            confusion[key] = confusion.get(key, 0) + 1
        # Vehicles not audited are implicitly classified HONEST by L1 bypass.
        for vehicle_id, gt in vehicle_to_gt_label.items():
            if vehicle_id in audited_vehicles:
                continue
            key = f"{gt}->L1_BYPASS_HONEST"
            confusion[key] = confusion.get(key, 0) + 1
        telemetry["l2_confusion_matrix"] = confusion

        return telemetry

    def _apply_demo_phase(
        self,
        telemetry: dict[str, Any],
        round_index: int,
    ) -> tuple[dict[str, Any], str]:
        # Phase-specific role profiles and drift vectors already shape the
        # generated gradients upstream. Persist the observed telemetry as-is
        # so policy decisions reflect real L2/L4 outcomes instead of a
        # scripted overwrite.
        return telemetry.copy(), self._phase_name(round_index)

    def _save_telemetry(self, telemetry: dict[str, Any]) -> None:
        response = self.session.post(
            f"{POLICY_AGENT_URL}/api/v1/policy/telemetry",
            json=telemetry,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()

    def _should_trigger_policy(
        self,
        telemetry: dict[str, Any],
        round_id: int,
    ) -> tuple[bool, list[str], str | None]:
        reasons: list[str] = []

        if telemetry["fraud_rate"] >= POLICY_TRIGGER_FRAUD_RATE:
            reasons.append(
                f"fraud_rate={telemetry['fraud_rate']:.2f} is high, tighten security strategy"
            )

        if (
            telemetry["rarity_rate"] <= POLICY_TRIGGER_RARITY_RATE
            and telemetry["corner_accuracy"] <= POLICY_TRIGGER_CORNER_ACCURACY
        ):
            reasons.append(
                "rarity_rate is low and corner_accuracy is weak, relax theta_rare to admit more rare gradients"
            )

        if telemetry["false_slash_estimate"] >= POLICY_TRIGGER_FALSE_SLASH:
            reasons.append(
                "false_slash_estimate is high, reduce penalty strength to avoid harming honest vehicles"
            )

        if telemetry["golden_drift_score"] >= POLICY_TRIGGER_DRIFT_SCORE:
            reasons.append(
                "golden_drift_score is high, increase drift detection sensitivity"
            )

        if not reasons:
            return False, [], "telemetry is stable; no policy adjustment needed"

        if (
            self.last_policy_trigger_round is not None
            and round_id - self.last_policy_trigger_round < POLICY_TRIGGER_COOLDOWN_ROUNDS
        ):
            return (
                False,
                reasons,
                f"cooldown active ({POLICY_TRIGGER_COOLDOWN_ROUNDS} rounds between policy proposals)",
            )

        return True, reasons, None

    def _propose_policy(self, telemetry: dict[str, Any]) -> dict[str, Any]:
        response = self.session.post(
            f"{POLICY_AGENT_URL}/api/v1/policy/propose",
            json=telemetry,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    def _activate_policy(self, round_id: int) -> dict[str, Any]:
        response = self.session.post(
            f"{POLICY_AGENT_URL}/api/v1/policy/activate",
            params={"round_id": round_id},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # Generation trace persistence
    # ------------------------------------------------------------------

    def _persist_generation_trace(self, round_index: int) -> None:
        if self.trace_dir is None or not self.current_rarity_generation_trace:
            return
        out_file = self.trace_dir / f"trace_round_{round_index:04d}.jsonl"
        try:
            with out_file.open("w") as handle:
                for entry in self.current_rarity_generation_trace:
                    handle.write(json.dumps(entry) + "\n")
        except OSError as exc:
            logger.warning("failed to persist generation trace for round %s: %s", round_index, exc)

    # ------------------------------------------------------------------
    # Round driver
    # ------------------------------------------------------------------

    def run_round(self, round_index: int) -> None:
        self._maybe_refresh_reference_gradients(round_index)
        current_policy = self._fetch_current_policy()
        round_id = self._select_round_id(current_policy)

        (
            updates,
            planned_roles,
            vehicle_addresses,
            new_vehicle_count,
            preflight_counts,
            l1_projection,
        ) = self._build_batch(
            round_id=round_id,
            policy=current_policy,
            round_index=round_index,
        )

        logger.info("")
        logger.info(
            "Round %s | policy round %s | gradient_dim=%s",
            round_id,
            current_policy.round_id,
            DEMO_GRADIENT_DIM,
        )
        logger.info("planned composition: %s", dict(planned_roles))
        logger.info("preflight local audit: %s", dict(preflight_counts))
        logger.info("local L1 suspect projection: %s", dict(l1_projection))

        round_started_at = utc_now()
        with ThreadPoolExecutor(max_workers=min(SUBMISSION_MAX_WORKERS, BATCH_SIZE)) as executor:
            results = list(executor.map(self._submit_gradient, updates))

        success_count = sum(results)
        logger.info("submitted: %s/%s", success_count, len(updates))
        if success_count == 0:
            raise RuntimeError("all gradient submissions failed")

        observed_queue_length = self._read_suspect_queue_length()
        logger.info("observed suspect queue length after intake: %s", observed_queue_length)

        expected_suspects = (
            preflight_counts.get("FRAUD", 0)
            + preflight_counts.get("RARITY", 0)
            + preflight_counts.get("NOISE", 0)
        )
        time.sleep(PROCESS_WAIT_SECONDS)
        audits = self._wait_for_round_audits(
            started_at=round_started_at,
            vehicle_addresses=vehicle_addresses,
            expected_suspects=expected_suspects,
        )

        audit_counts = Counter(audit["classification"] for audit in audits)
        logger.info("audits observed: %s", dict(audit_counts))

        telemetry = self._build_telemetry(
            round_id=round_id,
            batch_size=BATCH_SIZE,
            audits=audits,
            planned_roles=planned_roles,
            new_vehicle_count=new_vehicle_count,
            suspect_queue_length=observed_queue_length,
            updates=updates,
        )
        telemetry, phase_name = self._apply_demo_phase(telemetry, round_index)
        self._save_telemetry(telemetry)
        logger.info("demo phase: %s", phase_name)

        # Surface per-family fraud survival in logs so issues are visible at
        # round granularity, not only after offline aggregation.
        survival = telemetry.get("fraud_survival_by_family", {})
        if survival:
            logger.info("fraud_survival_by_family: %s", survival)
        l1_routing = telemetry.get("l1_routing_by_archetype", {})
        if l1_routing:
            logger.info("l1_routing_by_archetype: %s", l1_routing)

        should_trigger, trigger_reasons, skip_reason = self._should_trigger_policy(telemetry, round_id)
        if should_trigger:
            logger.info("policy trigger fired: %s", trigger_reasons)
            proposal = self._propose_policy(telemetry)
            self.last_policy_trigger_round = round_id

            diff = proposal.get("proposed_policy", {})
            logger.info(
                "policy proposal r%s via %s | blocked=%s | reasons=%s",
                proposal.get("round_id"),
                proposal.get("source_engine"),
                not proposal.get("safety_guard_passed", False),
                proposal.get("reasons", []),
            )
            logger.info(
                "next params: theta_tol=%s theta_rare=%s theta_drift=%s recheck=%s slash=%s rarity_reward=%s corner_weight=%s",
                diff.get("theta_tol"),
                diff.get("theta_rare"),
                diff.get("theta_drift"),
                diff.get("recheck_probability"),
                diff.get("slash_multiplier"),
                diff.get("rarity_reward_multiplier"),
                diff.get("corner_weight"),
            )

            if AUTO_ACTIVATE_POLICY and proposal.get("safety_guard_passed", False):
                activation = self._activate_policy(proposal["round_id"])
                logger.info(
                    "activated policy for round %s (%s)",
                    activation["round_id"],
                    activation["policy_hash"][:12],
                )
            else:
                logger.info("proposal kept pending for round %s", proposal.get("round_id"))
        else:
            if trigger_reasons:
                logger.info("policy trigger skipped: %s | reasons=%s", skip_reason, trigger_reasons)
            else:
                logger.info("policy trigger skipped: %s", skip_reason)

        self._persist_generation_trace(round_index)
        self._next_round()

    def run(self) -> None:
        round_index = 0
        while True:
            self.run_round(round_index)
            round_index += 1

            if not CONTINUOUS_MODE and round_index >= NUM_ROUNDS:
                break

            logger.info("sleeping %.1fs before next round", ROUND_INTERVAL)
            time.sleep(ROUND_INTERVAL)


if __name__ == "__main__":
    try:
        DemoDataGenerator().run()
    except KeyboardInterrupt:
        logger.info("demo generation interrupted")
    except Exception as exc:
        logger.exception("demo generation failed: %s", exc)
        raise

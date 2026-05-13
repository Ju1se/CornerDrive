"""
L2: Dual-Purpose Audit - The Sniper
Classifies gradients as FRAUD, beneficial RARITY, NOISE, or HONEST.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging
import hashlib
import json
from datetime import datetime, timezone

from common.config import (
    L2_FRAUD_THRESHOLD,
    L2_RARITY_THRESHOLD,
    L2_LEARNING_RATE,
)
from common.schemas import Policy

logger = logging.getLogger(__name__)


class Classification(Enum):
    """L2 classification outcomes."""
    FRAUD = "FRAUD"      # ΔL_main > θ_tol → Malicious, slash stake
    RARITY = "RARITY"    # ΔL_corner ≤ theta_rare and ΔL_main ≤ theta_tol → Beneficial rare update
                         # theta_rare is negative; theta_tol bounds tolerated main-task drift.
    HONEST = "HONEST"    # ΔL_main < 0 → Helps main task
    NOISE = "NOISE"      # Otherwise → Negligible impact


@dataclass
class AuditResult:
    """Result of L2 audit."""
    vehicle_id: str
    classification: Classification
    delta_loss_main: float
    delta_loss_corner: float
    final_score: float
    include_in_aggregation: bool
    sbt_points: int
    fraud_proof: Optional[Dict[str, Any]] = None
    rarity_certificate: Optional[Dict[str, Any]] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


class DualChannelAuditor:
    """
    L2 Dual-Purpose Auditor - "The Sniper"

    Analyzes suspect gradients using dual loss channels:
    1. Main dataset: Checks if gradient helps/hurts primary task
    2. Corner cases: Checks if gradient provides rare valuable information
    """

    def __init__(
        self,
        model: nn.Module,
        main_dataset: torch.utils.data.Dataset,
        corner_dataset: torch.utils.data.Dataset,
        criterion: nn.Module = None,
        learning_rate: float = L2_LEARNING_RATE,
        fraud_threshold: float = L2_FRAUD_THRESHOLD,
        rarity_threshold: float = L2_RARITY_THRESHOLD,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.main_dataset = main_dataset
        self.corner_dataset = corner_dataset
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.lr = learning_rate
        self.fraud_threshold = fraud_threshold
        self.rarity_threshold = rarity_threshold
        self.device = device
        self.slash_multiplier = 1.0
        self.rarity_reward_multiplier = 1.0
        self.corner_weight = 1.0
        self.corner_harm_threshold = 0.0

        # Create data loaders
        self.main_loader = torch.utils.data.DataLoader(
            main_dataset, batch_size=32, shuffle=False
        )
        self.corner_loader = torch.utils.data.DataLoader(
            corner_dataset, batch_size=32, shuffle=False
        )

    def apply_policy(self, policy: Policy) -> None:
        """Sync dynamic thresholds and reward weights from the current policy."""
        self.fraud_threshold = policy.theta_tol
        self.rarity_threshold = policy.theta_rare
        self.slash_multiplier = policy.slash_multiplier
        self.rarity_reward_multiplier = policy.rarity_reward_multiplier
        self.corner_weight = policy.corner_weight

    def compute_loss(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
    ) -> float:
        """Compute average loss on a dataset."""
        model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

        return total_loss / max(total_samples, 1)

    def apply_gradient(
        self,
        gradient: np.ndarray,
        learning_rate: float = None,
    ) -> nn.Module:
        """
        Apply gradient to model: W_new = W - η * g

        Returns a copy of the model with gradient applied.
        """
        lr = learning_rate or self.lr

        # Create model copy
        model_copy = copy.deepcopy(self.model)
        model_copy = model_copy.to(self.device)

        # Flatten model parameters
        params = []
        for param in model_copy.parameters():
            params.append(param.data.view(-1))
        flat_params = torch.cat(params)

        # Apply gradient
        gradient_tensor = torch.tensor(gradient, dtype=torch.float32, device=self.device)

        if len(gradient_tensor) != len(flat_params):
            raise ValueError(
                f"Gradient size {len(gradient_tensor)} != model params {len(flat_params)}"
            )

        flat_params -= lr * gradient_tensor

        # Unflatten back to model
        idx = 0
        for param in model_copy.parameters():
            param_size = param.numel()
            param.data = flat_params[idx:idx + param_size].view(param.shape)
            idx += param_size

        return model_copy

    def compute_delta_loss(
        self,
        gradient: np.ndarray,
        dataloader: torch.utils.data.DataLoader,
    ) -> float:
        """
        Compute loss drift after applying gradient.

        Formula: ΔL = L(W - ηg; D) - L(W; D)

        Interpretation:
            ΔL < 0: Gradient improves performance
            ΔL > 0: Gradient degrades performance
        """
        # Current loss
        loss_before = self.compute_loss(self.model, dataloader)

        # Loss after applying gradient
        model_updated = self.apply_gradient(gradient)
        loss_after = self.compute_loss(model_updated, dataloader)

        return loss_after - loss_before

    def audit(self, vehicle_id: str, gradient: np.ndarray) -> AuditResult:
        """
        Perform dual-channel audit on a suspect gradient.

        Classification Logic:
            1. If ΔL_main > θ_tol → FRAUD (hurts main task)
            2. Elif ΔL_corner ≤ theta_rare and ΔL_main ≤ θ_tol → RARITY
               (helps corner cases significantly while staying within the
               main-task damage budget)
               theta_rare 是负值 (如 -0.03)，越负越严格
            3. Elif ΔL_main < 0 and ΔL_corner > theta_corner_harm → FRAUD
               (main-helpful but corner-harmful update)
            4. Elif ΔL_main < 0 → HONEST (helps main task without corner harm)
            5. Else → NOISE (negligible impact)
        """
        logger.info(f"Auditing gradient from vehicle {vehicle_id}")

        # Compute dual loss drifts
        delta_main = self.compute_delta_loss(gradient, self.main_loader)
        delta_corner = self.compute_delta_loss(gradient, self.corner_loader)

        logger.info(f"ΔL_main={delta_main:.6f}, ΔL_corner={delta_corner:.6f}")

        # Classification logic
        if delta_main > self.fraud_threshold:
            classification = Classification.FRAUD
            include = False
            sbt_points = int(round(-50 * self.slash_multiplier))
            fraud_proof = self._generate_fraud_proof(
                vehicle_id, gradient, delta_main, delta_corner
            )
            rarity_cert = None

        elif delta_corner <= self.rarity_threshold and delta_main <= self.fraud_threshold:
            classification = Classification.RARITY
            include = True
            sbt_points = int(round(10 * self.rarity_reward_multiplier))
            fraud_proof = None
            rarity_cert = self._generate_rarity_certificate(
                vehicle_id, gradient, delta_main, delta_corner
            )

        elif delta_main <= 0 and delta_corner > self.corner_harm_threshold:
            classification = Classification.FRAUD
            include = False
            sbt_points = int(round(-50 * self.slash_multiplier))
            fraud_proof = self._generate_fraud_proof(
                vehicle_id, gradient, delta_main, delta_corner
            )
            fraud_proof["proof_type"] = "CORNER_HARM"
            rarity_cert = None

        elif delta_main < 0:
            classification = Classification.HONEST
            include = True
            sbt_points = 1
            fraud_proof = None
            rarity_cert = None

        else:
            classification = Classification.NOISE
            include = False
            sbt_points = 0
            fraud_proof = None
            rarity_cert = None

        # Compute final score for settlement
        final_score = self._compute_final_score(delta_main, delta_corner)

        result = AuditResult(
            vehicle_id=vehicle_id,
            classification=classification,
            delta_loss_main=delta_main,
            delta_loss_corner=delta_corner,
            final_score=final_score,
            include_in_aggregation=include,
            sbt_points=sbt_points,
            fraud_proof=fraud_proof,
            rarity_certificate=rarity_cert,
        )

        logger.info(
            f"Audit complete: {vehicle_id} → {classification.value} "
            f"(SBT: {sbt_points:+d})"
        )

        return result

    def _compute_final_score(
        self,
        delta_main: float,
        delta_corner: float,
    ) -> float:
        """
        Compute final score for settlement priority.

        Formula: Score = |ΔL_main| + λ * max(0, -ΔL_corner)

        Higher score = more significant (fraud or rarity)
        """
        lambda_weight = 0.5 * self.corner_weight
        rarity_bonus = max(0, -delta_corner)
        return abs(delta_main) + lambda_weight * rarity_bonus

    def _generate_fraud_proof(
        self,
        vehicle_id: str,
        gradient: np.ndarray,
        delta_main: float,
        delta_corner: float,
    ) -> Dict[str, Any]:
        """Generate cryptographic fraud proof for L4 settlement."""
        proof_data = {
            "vehicle_id": vehicle_id,
            "delta_loss_main": delta_main,
            "delta_loss_corner": delta_corner,
            "fraud_threshold": self.fraud_threshold,
            "gradient_hash": hashlib.sha256(gradient.tobytes()).hexdigest(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Compute proof hash
        proof_string = json.dumps(proof_data, sort_keys=True)
        proof_hash = hashlib.sha256(proof_string.encode()).hexdigest()

        return {
            **proof_data,
            "proof_hash": proof_hash,
            "proof_type": "FRAUD",
        }

    def _generate_rarity_certificate(
        self,
        vehicle_id: str,
        gradient: np.ndarray,
        delta_main: float,
        delta_corner: float,
    ) -> Dict[str, Any]:
        """Generate rarity certificate for bonus rewards."""
        cert_data = {
            "vehicle_id": vehicle_id,
            "delta_loss_main": delta_main,
            "delta_loss_corner": delta_corner,
            "rarity_threshold": self.rarity_threshold,
            "corner_improvement": abs(delta_corner),
            "gradient_hash": hashlib.sha256(gradient.tobytes()).hexdigest(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Compute certificate hash
        cert_string = json.dumps(cert_data, sort_keys=True)
        cert_hash = hashlib.sha256(cert_string.encode()).hexdigest()

        return {
            **cert_data,
            "certificate_hash": cert_hash,
            "certificate_type": "RARITY",
        }

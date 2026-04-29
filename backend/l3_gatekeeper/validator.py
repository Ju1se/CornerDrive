"""
L3: Global Validation - The Gatekeeper
Validates aggregated model updates against golden dataset before commit.
"""

import copy
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from common.config import L3_DRIFT_THRESHOLD, L3_GOLDEN_DATASET_PATH
from common.schemas import Policy

logger = logging.getLogger(__name__)


class ValidationDecision(Enum):
    """L3 validation decisions."""
    APPROVE = "APPROVE"  # Model update accepted
    REJECT = "REJECT"    # Model update rejected


@dataclass
class ValidationResult:
    """Result of L3 validation."""
    decision: ValidationDecision
    drift: float
    drift_threshold: float
    loss_before: float
    loss_after: float
    model_version: int
    commit_hash: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


class GoldenDatasetManager:
    """Manages the golden dataset for L3 validation."""

    def __init__(
        self,
        dataset_path: str = L3_GOLDEN_DATASET_PATH,
        input_dim: int = 784,
        num_classes: int = 10,
    ):
        self.dataset_path = Path(dataset_path)
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.dataset = None
        self.dataset_source = "uninitialized"
        self._load_dataset()

    def _load_dataset(self):
        """Load golden dataset from disk."""
        if self._has_dataset_artifacts():
            logger.info(f"Loading golden dataset from {self.dataset_path}")
            self.dataset = self._load_dataset_from_path()
            self.dataset_source = "disk"
            return

        logger.warning(
            f"Golden dataset artifacts not found at {self.dataset_path}, using placeholder"
        )
        self._create_placeholder()

    def _has_dataset_artifacts(self) -> bool:
        """Return whether the configured path contains loadable dataset artifacts."""
        if self.dataset_path.is_file():
            return True

        if not self.dataset_path.is_dir():
            return False

        candidate_names = (
            "dataset.pt",
            "dataset.pth",
            "golden_dataset.pt",
            "golden_dataset.pth",
            "data.pt",
        )
        return any((self.dataset_path / candidate).exists() for candidate in candidate_names)

    def has_dataset_artifacts(self) -> bool:
        """Public wrapper used by status endpoints and diagnostics."""
        return self._has_dataset_artifacts()

    def _load_dataset_from_path(self) -> torch.utils.data.Dataset:
        """Load a golden dataset from a supported torch payload."""
        if self.dataset_path.is_file():
            payload = torch.load(self.dataset_path, map_location="cpu")
            return self._dataset_from_payload(payload)

        dataset_candidates = [
            self.dataset_path / "dataset.pt",
            self.dataset_path / "dataset.pth",
            self.dataset_path / "golden_dataset.pt",
            self.dataset_path / "golden_dataset.pth",
        ]
        for candidate in dataset_candidates:
            if candidate.exists():
                payload = torch.load(candidate, map_location="cpu")
                return self._dataset_from_payload(payload)

        data_path = self.dataset_path / "data.pt"
        target_path = self.dataset_path / "targets.pt"
        labels_path = self.dataset_path / "labels.pt"
        if data_path.exists() and (target_path.exists() or labels_path.exists()):
            payload = {
                "data": torch.load(data_path, map_location="cpu"),
                "targets": torch.load(
                    target_path if target_path.exists() else labels_path,
                    map_location="cpu",
                ),
            }
            return self._dataset_from_payload(payload)

        raise RuntimeError(f"Unsupported golden dataset layout at {self.dataset_path}")

    def _dataset_from_payload(self, payload: Any) -> torch.utils.data.Dataset:
        """Normalize common saved payload formats into a TensorDataset."""
        data: Any
        targets: Any

        if isinstance(payload, dict):
            if "data" not in payload:
                raise RuntimeError("Golden dataset payload is missing 'data'")
            targets = payload.get("targets", payload.get("labels"))
            if targets is None:
                raise RuntimeError("Golden dataset payload is missing 'targets' or 'labels'")
            data = payload["data"]
        elif isinstance(payload, (list, tuple)) and len(payload) == 2:
            data, targets = payload
        else:
            data = getattr(payload, "data", None)
            targets = getattr(payload, "targets", getattr(payload, "labels", None))
            if data is None or targets is None:
                raise RuntimeError("Unsupported golden dataset payload type")

        data_tensor = torch.as_tensor(data, dtype=torch.float32)
        target_tensor = torch.as_tensor(targets, dtype=torch.long)

        if data_tensor.shape[0] != target_tensor.shape[0]:
            raise RuntimeError("Golden dataset data/target length mismatch")
        if data_tensor.shape[0] == 0:
            raise RuntimeError("Golden dataset payload is empty")

        return torch.utils.data.TensorDataset(data_tensor, target_tensor)

    def _create_placeholder(self):
        """Create placeholder golden dataset for development."""
        class PlaceholderGoldenDataset(torch.utils.data.Dataset):
            def __init__(self, input_dim: int, num_classes: int, size=200):
                self.data = torch.randn(size, input_dim)
                self.targets = torch.randint(0, num_classes, (size,))

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx], self.targets[idx]

        self.dataset = PlaceholderGoldenDataset(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
        )
        self.dataset_source = "placeholder"

    def get_dataloader(self, batch_size: int = 32) -> torch.utils.data.DataLoader:
        """Get DataLoader for golden dataset."""
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
        )

    def describe(self) -> Dict[str, Any]:
        """Summarize the currently loaded golden dataset source and shape."""
        sample_count = len(self.dataset) if self.dataset is not None else 0
        sample_shape: Optional[list[int]] = None

        if sample_count > 0:
            sample, _ = self.dataset[0]
            sample_shape = list(torch.as_tensor(sample).shape)

        return {
            "dataset_path": str(self.dataset_path),
            "dataset_source": self.dataset_source,
            "dataset_artifacts_present": self._has_dataset_artifacts(),
            "sample_count": sample_count,
            "sample_shape": sample_shape,
        }


class Gatekeeper:
    """
    L3 Gatekeeper - Global Model Validator

    Validates aggregated model updates before committing to blockchain.
    Uses golden dataset to ensure model quality is maintained.
    """

    def __init__(
        self,
        model: nn.Module,
        golden_dataset_path: str = L3_GOLDEN_DATASET_PATH,
        drift_threshold: float = L3_DRIFT_THRESHOLD,
        criterion: nn.Module = None,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        input_dim, num_classes = self._infer_model_io()
        self.golden_manager = GoldenDatasetManager(
            golden_dataset_path,
            input_dim=input_dim,
            num_classes=num_classes,
        )
        self.drift_threshold = drift_threshold
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.device = device
        self.model_version = 0

        # Store model checkpoint for rollback
        self._checkpoint = None
        self._save_checkpoint()

    def _infer_model_io(self) -> tuple[int, int]:
        """Best-effort inference for placeholder golden dataset shape."""
        linear_layers = [module for module in self.model.modules() if isinstance(module, nn.Linear)]
        if linear_layers:
            return linear_layers[0].in_features, linear_layers[-1].out_features

        return 784, 10

    def _save_checkpoint(self):
        """Save current model state for potential rollback."""
        self._checkpoint = {
            'state_dict': copy.deepcopy(self.model.state_dict()),
            'version': self.model_version,
        }

    def _load_checkpoint(self):
        """Rollback to last checkpoint."""
        if self._checkpoint:
            self.model.load_state_dict(self._checkpoint['state_dict'])
            self.model_version = self._checkpoint['version']
            logger.info(f"Rolled back to model version {self.model_version}")

    def compute_loss(self, model: nn.Module) -> float:
        """Compute loss on golden dataset."""
        model.eval()
        dataloader = self.golden_manager.get_dataloader()

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

    def apply_update(
        self,
        aggregated_gradient: np.ndarray,
        learning_rate: float,
    ) -> nn.Module:
        """
        Apply aggregated gradient update to model.

        Formula: W_new = W_old - η * ΔW

        Returns copy of model with update applied.
        """
        # Create model copy
        model_copy = copy.deepcopy(self.model)
        model_copy = model_copy.to(self.device)

        # Flatten parameters
        params = []
        for param in model_copy.parameters():
            params.append(param.data.view(-1))
        flat_params = torch.cat(params)

        # Apply gradient
        gradient_tensor = torch.tensor(
            aggregated_gradient,
            dtype=torch.float32,
            device=self.device
        )

        flat_params -= learning_rate * gradient_tensor

        # Unflatten
        idx = 0
        for param in model_copy.parameters():
            param_size = param.numel()
            param.data = flat_params[idx:idx + param_size].view(param.shape)
            idx += param_size

        return model_copy

    def validate(
        self,
        aggregated_gradient: np.ndarray,
        learning_rate: float = 0.01,
        policy: Optional[Policy] = None,
    ) -> ValidationResult:
        """
        Validate an aggregated gradient update.

        Formula:
            Drift = L_gold(W_old - η*ΔW) - L_gold(W_old)

        Decision:
            if Drift < θ_drift: APPROVE
            else: REJECT
        """
        logger.info("Starting L3 validation...")

        if policy is not None:
            self.drift_threshold = policy.theta_drift

        # Compute loss before update
        loss_before = self.compute_loss(self.model)

        # Apply update to copy
        model_updated = self.apply_update(aggregated_gradient, learning_rate)

        # Compute loss after update
        loss_after = self.compute_loss(model_updated)

        # Compute drift
        drift = loss_after - loss_before

        logger.info(
            f"L3 Validation: loss_before={loss_before:.6f}, "
            f"loss_after={loss_after:.6f}, drift={drift:.6f}"
        )

        # Make decision
        if drift < self.drift_threshold:
            decision = ValidationDecision.APPROVE

            # Accept update
            self._save_checkpoint()  # Save current as backup
            self.model.load_state_dict(model_updated.state_dict())
            self.model_version += 1

            # Generate commit hash
            commit_hash = self._generate_commit_hash(aggregated_gradient)

            logger.info(f"✅ APPROVED: Model updated to version {self.model_version}")

        else:
            decision = ValidationDecision.REJECT
            commit_hash = None

            logger.warning(
                f"❌ REJECTED: Drift {drift:.6f} exceeds threshold {self.drift_threshold}"
            )

        return ValidationResult(
            decision=decision,
            drift=drift,
            drift_threshold=self.drift_threshold,
            loss_before=loss_before,
            loss_after=loss_after,
            model_version=self.model_version,
            commit_hash=commit_hash,
        )

    def _generate_commit_hash(self, gradient: np.ndarray) -> str:
        """Generate commit hash for blockchain."""
        data = {
            "gradient_hash": hashlib.sha256(gradient.tobytes()).hexdigest(),
            "model_version": self.model_version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        data_string = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_string.encode()).hexdigest()

    def rollback(self):
        """Rollback to previous model version."""
        self._load_checkpoint()

    def get_model_state(self) -> Dict[str, Any]:
        """Get current model state for L4 settlement."""
        return {
            "version": self.model_version,
            "state_dict_hash": hashlib.sha256(
                str(self.model.state_dict()).encode()
            ).hexdigest()[:16],
        }


# Simple placeholder model for development
class SimpleMLP(nn.Module):
    """Simple MLP for testing L3 validation."""
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

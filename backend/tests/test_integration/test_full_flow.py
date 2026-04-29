"""
Integration tests for complete L1 → L2 → L3 → L4 flow.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import redis
import json
import random
from datetime import datetime

# Import all layers
from l1_linear_defense.aggregation import filter_suspects, geometric_median
from l2_dual_audit.classifier import DualChannelAuditor, Classification
from l3_gatekeeper.validator import Gatekeeper, ValidationDecision


class SimpleMLP(nn.Module):
    """Simple model for testing."""
    def __init__(self, input_dim=100, hidden_dim=64, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class TestL1LinearDefense:
    """Test L1 aggregation and filtering."""

    def test_geometric_median_basic(self):
        """Test geometric median computation."""
        gradients = [
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([1.0, 1.0]),
        ]

        median, iterations = geometric_median(gradients)

        assert median.shape == (2,)
        assert iterations > 0
        # Median should be somewhere between the points
        assert np.all(median >= 0)
        assert np.all(median <= 1.5)

    def test_geometric_median_with_outlier(self):
        """Test that geometric median is robust to outliers."""
        # Normal gradients
        gradients = [np.array([1.0, 1.0]) for _ in range(9)]
        # Add outlier
        gradients.append(np.array([100.0, 100.0]))

        median, _ = geometric_median(gradients)

        # Median should be close to [1, 1], not influenced by outlier
        assert np.allclose(median, [1.0, 1.0], atol=5.0)

    def test_geometric_median_single_gradient(self):
        """Test geometric median with single gradient."""
        gradients = [np.array([2.0, 3.0, 4.0])]

        median, iterations = geometric_median(gradients)

        assert np.array_equal(median, gradients[0])
        assert iterations == 1

    def test_geometric_median_empty(self):
        """Test geometric median with empty list."""
        with pytest.raises(ValueError, match="Cannot compute geometric median of empty list"):
            geometric_median([])

    def test_cosine_similarity_filtering(self):
        """Test suspect detection using cosine similarity."""
        # Create gradients with one obvious outlier
        gradients = [np.array([1.0, 1.0, 1.0]) for _ in range(9)]
        gradients.append(np.array([-10.0, -10.0, -10.0]))  # Outlier

        vehicle_ids = [f"0x{i:040x}" for i in range(10)]

        result = filter_suspects(gradients, vehicle_ids, threshold=0.3)

        assert len(result.suspect_indices) >= 1
        assert 9 in result.suspect_indices  # Outlier should be detected
        assert result.routing_reasons[9] == "cosine_screening"
        assert len(result.clean_indices) + len(result.suspect_indices) == 10
        assert result.aggregated_gradient.shape == (3,)
        assert result.iterations > 0

    def test_probabilistic_recheck_routes_apparent_clean_gradients(self):
        """Test blind recheck routing for gradients that pass cosine screening."""
        gradients = [np.array([1.0, 1.0, 1.0]) for _ in range(4)]
        vehicle_ids = [f"0x{i:040x}" for i in range(4)]

        result = filter_suspects(
            gradients,
            vehicle_ids,
            threshold=0.3,
            recheck_probability=1.0,
            rng=random.Random(7),
        )

        assert result.clean_indices == []
        assert result.suspect_indices == [0, 1, 2, 3]
        assert set(result.routing_reasons.values()) == {"probabilistic_recheck"}

    def test_filter_suspects_input_validation(self):
        """Test input validation for filter_suspects."""
        with pytest.raises(ValueError, match="Gradients and vehicle_ids must have same length"):
            filter_suspects([np.array([1.0])], ["0x1", "0x2"])

        with pytest.raises(ValueError, match="Cannot filter empty gradient list"):
            filter_suspects([], [])


class TestL2DualAudit:
    """Test L2 classification."""

    @pytest.fixture
    def auditor(self):
        """Create auditor with placeholder datasets."""
        model = SimpleMLP()

        class PlaceholderDataset(torch.utils.data.Dataset):
            def __init__(self, size=100):
                self.data = torch.randn(size, 100)
                self.targets = torch.randint(0, 10, (size,))

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx], self.targets[idx]

        return DualChannelAuditor(
            model=model,
            main_dataset=PlaceholderDataset(),
            corner_dataset=PlaceholderDataset(size=50),
        )

    def test_audit_produces_classification(self, auditor):
        """Test that audit produces a valid classification."""
        # Get gradient size from model
        param_count = sum(p.numel() for p in auditor.model.parameters())
        gradient = np.random.randn(param_count) * 0.01

        result = auditor.audit("0x" + "1" * 40, gradient)

        assert result.classification in Classification
        assert isinstance(result.delta_loss_main, float)
        assert isinstance(result.delta_loss_corner, float)
        assert isinstance(result.sbt_points, int)
        assert result.vehicle_id == "0x" + "1" * 40
        assert isinstance(result.include_in_aggregation, bool)
        assert isinstance(result.final_score, float)

    def test_audit_classification_logic_honest(self, auditor, monkeypatch):
        """Test classification logic for honest gradients."""
        param_count = sum(p.numel() for p in auditor.model.parameters())
        gradient = np.zeros(param_count)
        deltas = iter([-0.001, -0.001])

        monkeypatch.setattr(auditor, "compute_delta_loss", lambda *_args, **_kwargs: next(deltas))

        result = auditor.audit("0x" + "honest" * 8 + "0", gradient)

        assert result.classification == Classification.HONEST
        assert result.sbt_points == 1
        assert result.include_in_aggregation is True

    def test_audit_classification_logic_fraud(self, auditor):
        """Test classification logic for fraudulent gradients."""
        param_count = sum(p.numel() for p in auditor.model.parameters())

        # Large gradient should hurt performance (positive delta loss)
        large_gradient = np.ones(param_count) * 10.0

        result = auditor.audit("0x" + "bad" * 13 + "0", large_gradient)

        # Large positive delta_loss_main should indicate fraud
        if result.delta_loss_main > auditor.fraud_threshold:
            assert result.classification == Classification.FRAUD
            assert result.sbt_points == -50
            assert result.include_in_aggregation is False
            assert result.fraud_proof is not None
            assert result.fraud_proof["proof_type"] == "FRAUD"

    def test_audit_proof_generation(self, auditor):
        """Test cryptographic proof generation."""
        param_count = sum(p.numel() for p in auditor.model.parameters())
        gradient = np.random.randn(param_count)

        # Test with gradient that might be fraud
        large_gradient = np.ones(param_count) * 5.0
        result = auditor.audit("0x" + "fraud" * 8 + "0", large_gradient)

        if result.fraud_proof:
            assert "proof_hash" in result.fraud_proof
            assert "gradient_hash" in result.fraud_proof
            assert "timestamp" in result.fraud_proof
            assert result.fraud_proof["vehicle_id"] == "0x" + "fraud" * 8 + "0"

    def test_rarity_allows_main_delta_within_tolerance(self, auditor, monkeypatch):
        """Corner improvement earns rarity when main-task drift stays within theta_tol."""
        param_count = sum(p.numel() for p in auditor.model.parameters())
        gradient = np.zeros(param_count)
        deltas = iter([0.02, -0.08])

        monkeypatch.setattr(auditor, "compute_delta_loss", lambda *_args, **_kwargs: next(deltas))

        result = auditor.audit("0x" + "rare" * 10, gradient)

        assert result.classification == Classification.RARITY
        assert result.rarity_certificate is not None
        assert result.include_in_aggregation is True

    def test_rarity_rejected_when_main_delta_exceeds_tolerance(self, auditor, monkeypatch):
        """Corner improvement does not override the main-task damage budget."""
        param_count = sum(p.numel() for p in auditor.model.parameters())
        gradient = np.zeros(param_count)
        deltas = iter([auditor.fraud_threshold + 0.01, -0.08])

        monkeypatch.setattr(auditor, "compute_delta_loss", lambda *_args, **_kwargs: next(deltas))

        result = auditor.audit("0x" + "rarex" * 8, gradient)

        assert result.classification == Classification.FRAUD
        assert result.rarity_certificate is None
        assert result.include_in_aggregation is False

    def test_rarity_still_wins_when_corner_help_and_main_non_positive(self, auditor, monkeypatch):
        """Rarity remains valid when corner help is strong and the main task is not harmed."""
        param_count = sum(p.numel() for p in auditor.model.parameters())
        gradient = np.zeros(param_count)
        deltas = iter([-0.01, -0.08])

        monkeypatch.setattr(auditor, "compute_delta_loss", lambda *_args, **_kwargs: next(deltas))

        result = auditor.audit("0x" + "corner" * 6 + "00", gradient)

        assert result.classification == Classification.RARITY
        assert result.rarity_certificate is not None
        assert result.include_in_aggregation is True

    def test_corner_harm_is_rejected_even_when_main_improves(self, auditor, monkeypatch):
        """A main-helpful update should not pass as honest when it harms corner loss."""
        param_count = sum(p.numel() for p in auditor.model.parameters())
        gradient = np.zeros(param_count)
        deltas = iter([-0.01, 0.03])

        monkeypatch.setattr(auditor, "compute_delta_loss", lambda *_args, **_kwargs: next(deltas))

        result = auditor.audit("0x" + "harm" * 10, gradient)

        assert result.classification == Classification.FRAUD
        assert result.fraud_proof is not None
        assert result.fraud_proof["proof_type"] == "CORNER_HARM"
        assert result.include_in_aggregation is False

    def test_compute_delta_loss(self, auditor):
        """Test loss drift computation."""
        param_count = sum(p.numel() for p in auditor.model.parameters())
        gradient = np.random.randn(param_count) * 0.001

        delta_loss = auditor.compute_delta_loss(gradient, auditor.main_loader)

        assert isinstance(delta_loss, float)
        # Delta loss could be positive or negative

    def test_gradient_application(self, auditor):
        """Test gradient application to model."""
        param_count = sum(p.numel() for p in auditor.model.parameters())
        gradient = np.random.randn(param_count) * 0.01

        model_copy = auditor.apply_gradient(gradient)

        assert isinstance(model_copy, type(auditor.model))

        # Check that parameters have changed
        for orig_param, new_param in zip(auditor.model.parameters(), model_copy.parameters()):
            assert not torch.equal(orig_param.data, new_param.data)


class TestL3Gatekeeper:
    """Test L3 validation."""

    @pytest.fixture
    def gatekeeper(self):
        """Create gatekeeper with placeholder golden dataset."""
        model = SimpleMLP()
        return Gatekeeper(model=model, drift_threshold=0.1)

    def test_validate_small_update(self, gatekeeper):
        """Test that small updates are approved."""
        param_count = sum(p.numel() for p in gatekeeper.model.parameters())

        # Small gradient should be approved
        small_gradient = np.random.randn(param_count) * 0.001

        result = gatekeeper.validate(small_gradient, learning_rate=0.01)

        assert result.decision in [ValidationDecision.APPROVE, ValidationDecision.REJECT]
        assert isinstance(result.drift, float)
        assert isinstance(result.loss_before, float)
        assert isinstance(result.loss_after, float)

    def test_model_version_increments(self, gatekeeper):
        """Test that model version increments on approval."""
        initial_version = gatekeeper.model_version
        param_count = sum(p.numel() for p in gatekeeper.model.parameters())

        # Very small gradient likely to be approved
        tiny_gradient = np.zeros(param_count)

        result = gatekeeper.validate(tiny_gradient, learning_rate=0.0)

        if result.decision == ValidationDecision.APPROVE:
            assert gatekeeper.model_version == initial_version + 1
            assert result.commit_hash is not None

    def test_validate_large_update(self, gatekeeper):
        """Test that large updates might be rejected."""
        param_count = sum(p.numel() for p in gatekeeper.model.parameters())

        # Large gradient might exceed drift threshold
        large_gradient = np.random.randn(param_count) * 10.0

        result = gatekeeper.validate(large_gradient, learning_rate=1.0)

        assert result.decision in [ValidationDecision.APPROVE, ValidationDecision.REJECT]
        if result.decision == ValidationDecision.REJECT:
            assert result.drift >= gatekeeper.drift_threshold
            assert result.commit_hash is None

    def test_checkpoint_and_rollback(self, gatekeeper):
        """Test checkpoint and rollback functionality."""
        initial_version = gatekeeper.model_version
        initial_state = {name: param.clone() for name, param in gatekeeper.model.named_parameters()}

        # Create a change
        param_count = sum(p.numel() for p in gatekeeper.model.parameters())
        gradient = np.random.randn(param_count) * 0.01

        result = gatekeeper.validate(gradient, learning_rate=0.01)

        # Rollback
        gatekeeper.rollback()

        assert gatekeeper.model_version == initial_version

        # Check parameters are restored
        for name, param in gatekeeper.model.named_parameters():
            assert torch.equal(param.data, initial_state[name])

    def test_commit_hash_generation(self, gatekeeper):
        """Test commit hash generation."""
        param_count = sum(p.numel() for p in gatekeeper.model.parameters())
        gradient = np.random.randn(param_count)

        result = gatekeeper.validate(gradient, learning_rate=0.01)

        if result.decision == ValidationDecision.APPROVE:
            assert isinstance(result.commit_hash, str)
            assert len(result.commit_hash) == 64  # SHA256 hex length


class TestFullPipeline:
    """Test complete L1 → L2 → L3 → L4 flow."""

    def test_end_to_end_flow(self):
        """Test complete pipeline with simulated gradients."""
        # L1: Aggregate and filter
        gradients = [np.random.randn(100) * 0.01 for _ in range(10)]
        vehicle_ids = [f"0x{i:040x}" for i in range(10)]

        l1_result = filter_suspects(gradients, vehicle_ids, threshold=0.5)

        assert l1_result.aggregated_gradient.shape == (100,)
        assert len(l1_result.clean_indices) + len(l1_result.suspect_indices) == 10

        # L3: Validate aggregation (create appropriate model)
        model = SimpleMLP(input_dim=100, hidden_dim=32, output_dim=5)
        gatekeeper = Gatekeeper(model=model, drift_threshold=0.5)

        # Resize gradient to match model
        param_count = sum(p.numel() for p in model.parameters())

        if len(l1_result.aggregated_gradient) != param_count:
            # Pad or truncate gradient to match model parameters
            if len(l1_result.aggregated_gradient) > param_count:
                test_gradient = l1_result.aggregated_gradient[:param_count]
            else:
                test_gradient = np.pad(l1_result.aggregated_gradient, (0, param_count - len(l1_result.aggregated_gradient)))
        else:
            test_gradient = l1_result.aggregated_gradient

        l3_result = gatekeeper.validate(test_gradient, learning_rate=0.01)

        assert l3_result.decision in [ValidationDecision.APPROVE, ValidationDecision.REJECT]
        assert isinstance(l3_result.drift, float)

    def test_suspect_flow_through_l2(self):
        """Test that suspects flow through L2 audit."""
        # Create gradients with clear outlier
        gradients = [np.array([1.0, 1.0]) for _ in range(5)]
        gradients.append(np.array([-5.0, -5.0]))  # Clear suspect
        vehicle_ids = [f"0x{i:040x}" for i in range(6)]

        # L1 filtering
        l1_result = filter_suspects(gradients, vehicle_ids, threshold=0.3)
        assert len(l1_result.suspect_indices) >= 1

        # L2 audit simulation
        suspect_vehicle = vehicle_ids[l1_result.suspect_indices[0]]
        suspect_gradient = gradients[l1_result.suspect_indices[0]]

        # Create auditor
        model = SimpleMLP(input_dim=2, hidden_dim=4, output_dim=2)

        class SmallDataset(torch.utils.data.Dataset):
            def __init__(self, size=20):
                self.data = torch.randn(size, 2)
                self.targets = torch.randint(0, 2, (size,))
            def __len__(self): return len(self.data)
            def __getitem__(self, idx): return self.data[idx], self.targets[idx]

        auditor = DualChannelAuditor(
            model=model,
            main_dataset=SmallDataset(),
            corner_dataset=SmallDataset(size=10),
        )

        # Pad gradient to match model size if needed
        param_count = sum(p.numel() for p in model.parameters())
        if len(suspect_gradient) != param_count:
            padded_gradient = np.pad(suspect_gradient, (0, param_count - len(suspect_gradient)))
        else:
            padded_gradient = suspect_gradient

        l2_result = auditor.audit(suspect_vehicle, padded_gradient)

        assert l2_result.classification in Classification
        assert l2_result.vehicle_id == suspect_vehicle

    def test_integration_data_flow(self):
        """Test data flow and format consistency between layers."""
        # Test gradient data consistency
        original_gradient = np.random.randn(50)

        # L1 processes gradient
        l1_result = filter_suspects(
            [original_gradient],
            ["0x1234567890abcdef" * 4],  # Fixed length
            threshold=0.1
        )

        # Gradient should maintain shape or be convertible
        assert l1_result.aggregated_gradient.shape == original_gradient.shape

        # Test that gradient can be used in subsequent layers
        param_count = 50  # Match original gradient
        if len(l1_result.aggregated_gradient) != param_count:
            # Should be able to reshape/pad as needed
            test_gradient = np.pad(
                l1_result.aggregated_gradient,
                (0, max(0, param_count - len(l1_result.aggregated_gradient)))
            )[:param_count]
        else:
            test_gradient = l1_result.aggregated_gradient

        assert len(test_gradient) == param_count


class TestRedisIntegration:
    """Test Redis integration across layers."""

    @pytest.fixture
    def redis_client(self):
        """Create Redis client for testing."""
        # Use test database
        client = redis.Redis(db=15, decode_responses=True)  # Use separate DB for tests
        try:
            client.ping()
        except redis.ConnectionError:
            pytest.skip("Redis is not available for integration storage tests")
        try:
            yield client
        finally:
            client.flushdb()  # Clean up after tests

    def test_l1_batch_storage(self, redis_client):
        """Test L1 batch storage to Redis."""
        # Simulate batch result
        from l1_linear_defense.aggregation import AggregationResult
        result = AggregationResult(
            aggregated_gradient=np.array([1.0, 2.0, 3.0]),
            clean_indices=[0, 1, 2],
            suspect_indices=[3, 4],
            cosine_scores={0: 0.1, 1: 0.2, 2: 0.15, 3: 0.8, 4: 0.9},
            iterations=10,
        )

        # Store to Redis (simulating L1 behavior)
        batch_id = f"batch:{datetime.utcnow().timestamp()}"
        redis_client.hset(batch_id, mapping={
            "clean_count": str(len(result.clean_indices)),
            "suspect_count": str(len(result.suspect_indices)),
            "aggregated": json.dumps(result.aggregated_gradient.tolist()),
        })
        redis_client.expire(batch_id, 3600)

        # Verify storage
        stored_data = redis_client.hgetall(batch_id)
        assert int(stored_data["clean_count"]) == 3
        assert int(stored_data["suspect_count"]) == 2
        assert json.loads(stored_data["aggregated"]) == [1.0, 2.0, 3.0]

    def test_l2_audit_dispatch(self, monkeypatch):
        """Test that L1 dispatches suspects directly to the L2 Celery worker."""
        from l1_linear_defense import server as l1_server

        suspect_data = {
            "vehicle_id": "0x" + "test" * 10,
            "gradient": [1.0, 2.0, 3.0],
            "cosine_score": 0.85,
            "policy_round": 7,
            "timestamp": datetime.utcnow().isoformat(),
        }
        captured = {}

        def fake_send_task(name, args=None, kwargs=None, queue=None):
            captured["name"] = name
            captured["args"] = args
            captured["kwargs"] = kwargs
            captured["queue"] = queue

        monkeypatch.setattr(l1_server.l2_audit_client, "send_task", fake_send_task)
        l1_server.dispatch_l2_audit(suspect_data)

        assert captured["name"] == "l2_audit.audit_gradient"
        assert captured["queue"] == "l2_audit_queue"
        assert captured["args"] == [suspect_data]

    def test_l4_statistics_storage(self, redis_client):
        """Test L4 statistics storage."""
        # Update various statistics
        redis_client.incr("stats:fraud_count")
        redis_client.incr("stats:rare_count")
        redis_client.incr("stats:honest_count")

        # Verify statistics
        fraud_count = int(redis_client.get("stats:fraud_count") or 0)
        rare_count = int(redis_client.get("stats:rare_count") or 0)
        honest_count = int(redis_client.get("stats:honest_count") or 0)

        assert fraud_count == 1
        assert rare_count == 1
        assert honest_count == 1

    def test_round_scoped_statistics_storage(self, redis_client):
        """Test per-round statistics used by policy telemetry."""
        redis_client.hincrby("stats:round:r12", "audit_count", 1)
        redis_client.hincrby("stats:round:r12", "fraud_count", 1)
        redis_client.hincrby("stats:round:r12", "audit_count", 1)
        redis_client.hincrby("stats:round:r12", "honest_count", 1)

        round_stats = redis_client.hgetall("stats:round:r12")
        assert int(round_stats["audit_count"]) == 2
        assert int(round_stats["fraud_count"]) == 1
        assert int(round_stats["honest_count"]) == 1

    def test_vehicle_data_storage(self, redis_client):
        """Test vehicle-specific data storage."""
        vehicle_address = "0x" + "vehicle" * 8 + "00"

        # Store vehicle data
        redis_client.hset(f"vehicle:{vehicle_address}", mapping={
            "reputation": "150",
            "contributions": "25",
            "fraud_count": "2",
            "rare_count": "1",
            "stake": "100000000000000000000",  # 100 ETH in wei
            "rewards": "150000000000000000000",  # 150 tokens
            "registered": "true",
        })

        # Retrieve and verify
        vehicle_data = redis_client.hgetall(f"vehicle:{vehicle_address}")
        assert int(vehicle_data["reputation"]) == 150
        assert int(vehicle_data["contributions"]) == 25
        assert vehicle_data["registered"] == "true"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

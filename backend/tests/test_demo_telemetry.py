import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from generate_demo_data import DemoDataGenerator


class TestDemoTelemetry:
    def test_build_telemetry_includes_observed_audit_sample_size_and_rates(self):
        generator = DemoDataGenerator.__new__(DemoDataGenerator)
        audits = [
            {
                "vehicle_id": "0x1",
                "classification": "FRAUD",
                "delta_loss_main": 0.05,
                "delta_loss_corner": 0.04,
                "sbt_points": -48,
            },
            {
                "vehicle_id": "0x2",
                "classification": "NOISE",
                "delta_loss_main": 0.001,
                "delta_loss_corner": -0.0005,
                "sbt_points": 0,
            },
            {
                "vehicle_id": "0x3",
                "classification": "FRAUD",
                "delta_loss_main": 0.06,
                "delta_loss_corner": 0.05,
                "sbt_points": -48,
            },
        ]

        telemetry = generator._build_telemetry(
            round_id=7,
            batch_size=5,
            audits=audits,
            planned_roles={"FRAUD": 2, "RARITY": 1, "NOISE": 1, "HONEST": 1},
            new_vehicle_count=2,
            suspect_queue_length=3,
        )

        assert telemetry["audit_sample_size"] == 3
        assert telemetry["fraud_rate"] == pytest.approx(0.4)
        assert telemetry["noise_rate"] == pytest.approx(0.2)
        assert telemetry["honest_rate"] == pytest.approx(0.4)

    def test_apply_demo_phase_preserves_observed_classification_metrics(self):
        generator = DemoDataGenerator.__new__(DemoDataGenerator)
        telemetry = {
            "round_id": 12,
            "fraud_rate": 0.4,
            "rarity_rate": 0.1,
            "honest_rate": 0.3,
            "noise_rate": 0.2,
            "audit_sample_size": 6,
            "main_accuracy": 0.77,
            "corner_accuracy": 0.64,
            "main_loss_delta_avg": 0.01,
            "corner_loss_delta_avg": -0.02,
            "false_slash_estimate": 0.02,
            "rarity_retention_rate": 0.5,
            "golden_drift_score": 0.12,
            "reject_rate_l3": 0.0,
            "cosine_outlier_ratio": 0.6,
            "suspect_queue_length": 4,
            "avg_sbt_score": 3.0,
            "new_vehicle_ratio": 0.2,
            "hash_mismatch_rate": 0.0,
            "recent_attack_pressure": 0.5,
        }

        shaped, phase_name = generator._apply_demo_phase(telemetry, round_index=1)

        assert phase_name == "fraud_wave"
        assert shaped == telemetry

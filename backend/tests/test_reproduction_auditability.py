import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from common.schemas import DEFAULT_POLICY
from l2_dual_audit.classifier import Classification, DualChannelAuditor
from policy_agent.analysis.real_gradient_benchmark import (
    REAL_GRADIENT_CALIBRATED_POLICY_UPDATES,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from export_thesis_artifacts import validate_synthetic_router_mode  # noqa: E402


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_minimal_table_inputs(root: Path) -> tuple[Path, Path]:
    real_dir = root / "real"
    synthetic_dir = root / "synthetic"
    _write_csv(
        real_dir / "real_gradient_reliability_summary.csv",
        [
            {
                "source": "mnist",
                "method": "CornerDrive",
                "main_accuracy_avg_mean": 0.5,
                "main_accuracy_avg_ci95": 0.01,
                "corner_accuracy_avg_mean": 0.7,
                "corner_accuracy_avg_ci95": 0.02,
                "fraud_survival_rate_avg_mean": 0.0,
                "fraud_survival_rate_avg_ci95": 0.0,
                "rarity_retention_rate_avg_mean": 0.4,
                "rarity_retention_rate_avg_ci95": 0.03,
                "l1_review_rate_avg_mean": 0.85,
                "l1_review_rate_avg_ci95": 0.0,
            }
        ],
    )
    (real_dir / "real_gradient_reliability_summary.json").write_text(
        json.dumps(
            {
                "sources": ["mnist"],
                "seeds": [1],
                "methods": ["cornerdrive"],
                "runs": [
                    {
                        "config": {
                            "rounds": 1,
                            "clients_per_round": 2,
                            "max_clients": 4,
                            "max_samples_per_client": 8,
                            "reference_split_fraction": 0.5,
                            "max_reference_samples": 16,
                            "max_evaluation_samples": 16,
                            "cornerdrive_l1_mode": "dual_proxy_budgeted",
                        }
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_csv(
        synthetic_dir / "synthetic_alg_main_result_table.csv",
        [
            {
                "method": "CornerDrive p=0.10",
                "main_accuracy_mean": 0.85,
                "main_accuracy_std": 0.01,
                "corner_accuracy_mean": 0.61,
                "corner_accuracy_std": 0.02,
                "rarity_recall_mean": 1.0,
                "sign_flip_survival_mean": 0.0,
                "corner_harm_survival_mean": 0.84,
            }
        ],
    )
    return real_dir, synthetic_dir


def test_make_paper_tables_fails_when_required_inputs_are_missing(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/make_paper_tables.py",
            "--real-dir",
            str(tmp_path / "missing_real"),
            "--synthetic-dir",
            str(tmp_path / "missing_synthetic"),
            "--output-dir",
            str(tmp_path / "out"),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Required table inputs are missing or empty" in result.stderr


def test_make_paper_tables_removes_stale_allowed_missing_outputs(tmp_path):
    real_dir, synthetic_dir = _write_minimal_table_inputs(tmp_path)
    output_dir = tmp_path / "tables"
    stale_appendix = output_dir / "appendix_rarity_overlap.csv"
    stale_appendix.parent.mkdir(parents=True, exist_ok=True)
    stale_appendix.write_text("old,stale\n1,1\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/make_paper_tables.py",
            "--real-dir",
            str(real_dir),
            "--synthetic-dir",
            str(synthetic_dir),
            "--stress-dir",
            str(tmp_path / "missing_stress"),
            "--divergence-dir",
            str(tmp_path / "missing_divergence"),
            "--corner-harm-dir",
            str(tmp_path / "missing_corner_harm"),
            "--output-dir",
            str(output_dir),
            "--allow-missing-appendix",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "table_5_1_real_gradient_macro.csv").exists()
    assert (output_dir / "alg_main_result_table.csv").exists()
    assert not stale_appendix.exists()
    provenance = json.loads((output_dir / "table_provenance.json").read_text())
    assert provenance["allow_missing_appendix"] is True
    assert provenance["inputs"]["stress_rarity_overlap"]["exists"] is False


def test_calibration_manifest_keeps_calibration_and_heldout_disjoint():
    manifest = json.loads(
        (PROJECT_ROOT / "configs" / "real_gradient_calibration_manifest.json").read_text()
    )
    calibration_seeds = set(manifest["calibration"]["seeds"])
    heldout_seeds = set(manifest["final_heldout"]["seeds"])

    assert calibration_seeds
    assert heldout_seeds
    assert calibration_seeds.isdisjoint(heldout_seeds)
    assert manifest["final_heldout"]["retuning_allowed"] is False
    assert manifest["selected_policy"] == REAL_GRADIENT_CALIBRATED_POLICY_UPDATES


def test_l2_rejects_corner_benefit_with_positive_main_conflict(monkeypatch):
    model = torch.nn.Linear(1, 2)
    dataset = torch.utils.data.TensorDataset(
        torch.zeros(2, 1),
        torch.zeros(2, dtype=torch.long),
    )
    auditor = DualChannelAuditor(model=model, main_dataset=dataset, corner_dataset=dataset)
    auditor.apply_policy(DEFAULT_POLICY)
    deltas = iter([0.01, -0.01])
    monkeypatch.setattr(auditor, "compute_delta_loss", lambda _gradient, _loader: next(deltas))

    gradient = np.zeros(sum(parameter.numel() for parameter in model.parameters()))
    result = auditor.audit("vehicle", gradient)

    assert result.classification == Classification.NOISE
    assert result.include_in_aggregation is False


def test_synthetic_exporters_reject_dual_proxy_router_mode():
    validate_synthetic_router_mode("cosine_recheck")
    with pytest.raises(SystemExit, match="Synthetic ALG artifact exporters"):
        validate_synthetic_router_mode("dual_proxy_budgeted")

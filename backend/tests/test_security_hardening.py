import pytest
import numpy as np
from fastapi import HTTPException
from pydantic import ValidationError

from common.security import is_valid_api_key
from l1_linear_defense.server import GradientSubmission, ensure_consistent_gradient_shapes


def test_api_key_validation_uses_exact_keys():
    assert is_valid_api_key("expected", ["expected"])
    assert not is_valid_api_key("expected ", ["expected"])
    assert not is_valid_api_key("", ["expected"])
    assert not is_valid_api_key(None, ["expected"])


def test_gradient_submission_rejects_malformed_addresses():
    with pytest.raises(ValidationError):
        GradientSubmission(
            vehicle_address="0x" + "z" * 40,
            gradient_data=[0.1, 0.2],
            data_sample_count=10,
        )


def test_gradient_submission_rejects_non_finite_values():
    with pytest.raises(ValidationError):
        GradientSubmission(
            vehicle_address="0x" + "1" * 40,
            gradient_data=[0.1, float("nan")],
            data_sample_count=10,
        )


def test_immediate_batch_rejects_mixed_gradient_shapes():
    with pytest.raises(HTTPException) as exc_info:
        ensure_consistent_gradient_shapes([
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0, 3.0]),
        ])
    assert "same dimension" in exc_info.value.detail

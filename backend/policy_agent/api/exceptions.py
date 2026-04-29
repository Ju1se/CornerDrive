"""
Custom exceptions for FLPG Policy Agent.

Defines specific exception types for better error handling and reporting.
"""

from typing import Optional
from fastapi import HTTPException, status


class PolicyAgentException(Exception):
    """Base exception for policy agent errors."""

    def __init__(
        self,
        message: str,
        code: str = "POLICY_AGENT_ERROR",
        details: Optional[dict] = None
    ):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)


class GLMEngineException(PolicyAgentException):
    """GLM engine related errors."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, code="GLM_ENGINE_ERROR", details=details)


class GLMAPIException(GLMEngineException):
    """GLM API call failures."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        details: Optional[dict] = None
    ):
        details = details or {}
        if status_code:
            details["status_code"] = status_code
        super().__init__(message, details=details)


class GLMParseException(GLMEngineException):
    """GLM response parsing failures."""

    def __init__(self, message: str, raw_response: Optional[str] = None):
        details = {}
        if raw_response:
            details["raw_response_preview"] = raw_response[:200]
        super().__init__(message, details=details)


class PolicyValidationException(PolicyAgentException):
    """Policy validation failures."""

    def __init__(self, message: str, violations: Optional[list] = None):
        details = {}
        if violations:
            details["violations"] = violations
        super().__init__(message, code="VALIDATION_ERROR", details=details)


class SafetyGuardException(PolicyAgentException):
    """Safety guard blocking."""

    def __init__(self, message: str, blocked_reasons: Optional[list] = None):
        details = {}
        if blocked_reasons:
            details["blocked_reasons"] = blocked_reasons
        super().__init__(message, code="SAFETY_GUARD_BLOCK", details=details)


class StorageException(PolicyAgentException):
    """Storage/Redis related errors."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, code="STORAGE_ERROR", details=details)


class PolicyNotFoundException(PolicyAgentException):
    """Policy not found errors."""

    def __init__(self, round_id: Optional[int] = None):
        message = f"Policy not found for round {round_id}" if round_id else "Policy not found"
        super().__init__(message, code="POLICY_NOT_FOUND", details={"round_id": round_id})


class TelemetryNotFoundException(PolicyAgentException):
    """Telemetry not found errors."""

    def __init__(self, round_id: Optional[int] = None):
        message = f"Telemetry not found for round {round_id}" if round_id else "Telemetry not found"
        super().__init__(message, code="TELEMETRY_NOT_FOUND", details={"round_id": round_id})


def http_exception_from_error(error: PolicyAgentException) -> HTTPException:
    """
    Convert a PolicyAgentException to HTTPException.

    Args:
        error: PolicyAgentException instance

    Returns:
        HTTPException with appropriate status code
    """
    status_code_map = {
        "POLICY_NOT_FOUND": status.HTTP_404_NOT_FOUND,
        "TELEMETRY_NOT_FOUND": status.HTTP_404_NOT_FOUND,
        "VALIDATION_ERROR": status.HTTP_400_BAD_REQUEST,
        "SAFETY_GUARD_BLOCK": status.HTTP_403_FORBIDDEN,
        "GLM_ENGINE_ERROR": status.HTTP_503_SERVICE_UNAVAILABLE,
        "STORAGE_ERROR": status.HTTP_503_SERVICE_UNAVAILABLE,
    }

    http_status = status_code_map.get(
        error.code,
        status.HTTP_500_INTERNAL_SERVER_ERROR
    )

    return HTTPException(
        status_code=http_status,
        detail={
            "code": error.code,
            "message": error.message,
            "details": error.details
        }
    )

"""
Unified response schemas for FLPG Policy Agent API.

This module defines standard response formats for all API endpoints
to ensure consistency across the service.
"""

from typing import Optional, Generic, TypeVar, Any
from pydantic import BaseModel, ConfigDict, Field


T = TypeVar("T")


class APIResponse(BaseModel, Generic[T]):
    """Standard API response wrapper."""

    success: bool = Field(description="Indicates if the request was successful")
    data: Optional[T] = Field(default=None, description="Response data")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    message: Optional[str] = Field(default=None, description="Additional information")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "data": {},
                "error": None,
                "message": "Request completed successfully"
            }
        },
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(description="Service status: ok, degraded, or error")
    service: str = Field(description="Service name")
    version: str = Field(description="Service version")
    redis: str = Field(description="Redis connection status")
    glm_enabled: bool = Field(default=False, description="GLM engine enabled status")


class ErrorResponse(BaseModel):
    """Error response."""

    success: bool = Field(default=False, description="Always false for errors")
    error: str = Field(description="Error type or code")
    detail: str = Field(description="Detailed error message")
    path: Optional[str] = Field(default=None, description="Request path")
    timestamp: Optional[str] = Field(default=None, description="Error timestamp")


class PolicyProposalResponse(BaseModel):
    """Policy proposal response."""

    round_id: int = Field(description="Round this proposal applies to")
    status: str = Field(description="Proposal status: proposed, approved, blocked")
    changes: int = Field(description="Number of parameter changes")
    blocked: bool = Field(description="Whether proposal was blocked")
    approved: bool = Field(description="Whether proposal was approved")
    llm_used: bool = Field(description="Whether LLM was used")
    reasons: list[str] = Field(default_factory=list, description="Reasons for changes")
    blocked_reasons: list[str] = Field(default_factory=list, description="Blocking reasons")
    policy_hash: str = Field(description="Policy content hash")


class GLMEngineStatusResponse(BaseModel):
    """GLM engine status response."""

    engine_type: str = Field(description="Engine type")
    enabled: bool = Field(description="Whether GLM engine is enabled")
    model: Optional[str] = Field(default=None, description="GLM model name")
    base_url: Optional[str] = Field(default=None, description="GLM API base URL")
    timeout: int = Field(description="API timeout in seconds")
    description: str = Field(description="Engine description")

"""
L1: Linear Defense - FastAPI Server
Receives gradients from vehicles, performs screening, routes suspects to L2.
"""

import asyncio
import json
import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from celery import Celery
import numpy as np
import redis
from fastapi import FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from common.config import (
    CELERY_BROKER_URL, L1_CORS_ALLOWED_ORIGINS,
    L1_HOST, L1_MAX_GRADIENT_ABS, L1_MAX_GRADIENT_DIM, L1_PORT, REDIS_URL,
    L1_BATCH_SIZE, L1_BATCH_TIMEOUT, L1_RECHECK_PROBABILITY, L1_SUSPECT_THRESHOLD,
    L2_AUDIT_QUEUE,
)
from common.policy_loader import load_current_policy
from common.security import verify_api_key
from .aggregation import filter_suspects, AggregationResult
from .config import l1_router_config_from_env

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="FLPG L1 - Linear Defense",
    description="Byzantine-robust gradient aggregation with outlier detection",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=L1_CORS_ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Redis client
try:
    redis_client = redis.from_url(REDIS_URL)
    redis_client.ping()
except redis.ConnectionError:
    logger.error(f"Cannot connect to Redis at {REDIS_URL}")
    redis_client = None

# L2 task bridge
l2_audit_client = Celery(
    "l1_l2_bridge",
    broker=CELERY_BROKER_URL,
    backend=REDIS_URL,
)

# ============ SCHEMAS ============

class GradientSubmission(BaseModel):
    """Schema for gradient submission from vehicles."""
    vehicle_address: str = Field(..., min_length=42, max_length=42)
    gradient_data: List[float] = Field(..., min_length=1, max_length=L1_MAX_GRADIENT_DIM)
    data_sample_count: int = Field(..., gt=0, le=100000)
    timestamp: Optional[datetime] = None

    @field_validator("vehicle_address")
    @classmethod
    def validate_address(cls, v: str) -> str:
        normalized = v.lower()
        if not normalized.startswith("0x"):
            raise ValueError("Address must start with 0x")
        try:
            int(normalized[2:], 16)
        except ValueError as exc:
            raise ValueError("Address must be 20-byte hexadecimal") from exc
        return normalized

    @field_validator("gradient_data")
    @classmethod
    def validate_gradient_data(cls, values: List[float]) -> List[float]:
        for value in values:
            if not math.isfinite(value):
                raise ValueError("Gradient values must be finite")
            if abs(value) > L1_MAX_GRADIENT_ABS:
                raise ValueError("Gradient value exceeds configured bound")
        return values


class SubmissionResponse(BaseModel):
    """Response for gradient submission."""
    status: str
    vehicle: str
    score: Optional[float] = None
    is_suspect: bool = False
    message: str


class BatchResult(BaseModel):
    """Result of batch processing."""
    total: int
    clean: int
    suspects: int
    suspect_vehicles: List[str]
    aggregation_complete: bool


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    layer: str = "L1"
    service: str = "linear_defense"
    timestamp: datetime
    checks: Dict[str, str]


# ============ BATCH PROCESSING ============

class GradientBatcher:
    """Collects gradients and processes in batches."""

    def __init__(self, batch_size: int = L1_BATCH_SIZE, timeout: float = L1_BATCH_TIMEOUT):
        self.batch_size = batch_size
        self.timeout = timeout
        self.gradients: List[np.ndarray] = []
        self.vehicle_ids: List[str] = []
        self.lock = asyncio.Lock()
        self._batch_task: Optional[asyncio.Task] = None
        self.redis_available = redis_client is not None
        self.last_result: Optional[AggregationResult] = None
        self.last_processed_at: Optional[datetime] = None
        self.client_states: Dict[str, Dict[str, Any]] = {}

    async def add(self, vehicle_id: str, gradient: np.ndarray) -> None:
        async with self.lock:
            self.gradients.append(gradient)
            self.vehicle_ids.append(vehicle_id)

            if len(self.gradients) >= self.batch_size:
                await self._process_batch()
            elif self._batch_task is None:
                self._batch_task = asyncio.create_task(self._timeout_process())

    async def _timeout_process(self) -> None:
        await asyncio.sleep(self.timeout)
        async with self.lock:
            if self.gradients:
                await self._process_batch()
            self._batch_task = None

    async def _process_batch(self) -> None:
        if not self.gradients:
            return

        current_policy = None
        policy_round = 0
        try:
            current_policy = await load_current_policy()
            threshold = current_policy.cosine_filter_threshold
            recheck_probability = current_policy.recheck_probability
            policy_round = current_policy.round_id
        except Exception as exc:
            logger.warning(f"Falling back to configured L1 threshold: {exc}")
            threshold = None
            recheck_probability = L1_RECHECK_PROBABILITY

        router_config = l1_router_config_from_env()

        # Process batch
        result = filter_suspects(
            self.gradients,
            self.vehicle_ids,
            threshold=threshold if threshold is not None else L1_SUSPECT_THRESHOLD,
            recheck_probability=recheck_probability,
            router_config=router_config,
            client_states=self.client_states,
            current_round=policy_round,
        )
        self.last_result = result
        self.last_processed_at = datetime.now(timezone.utc)

        # Store result in Redis if available
        if self.redis_available:
            try:
                batch_id = f"batch:{self.last_processed_at.timestamp()}"
                redis_client.hset(batch_id, mapping={
                    "clean_count": len(result.clean_indices),
                    "suspect_count": len(result.suspect_indices),
                    "router_mode": result.router_mode,
                    "aggregated": json.dumps(result.aggregated_gradient.tolist()),
                })
                redis_client.expire(batch_id, 3600)  # 1 hour TTL
            except Exception as e:
                logger.error(f"Redis error during batch processing: {e}")

        for idx in result.suspect_indices:
            details = result.l1_score_details.get(idx, {})
            self.client_states.setdefault(self.vehicle_ids[idx], {})[
                "last_audit_round"
            ] = policy_round
            suspect_data = {
                "vehicle_id": self.vehicle_ids[idx],
                "gradient": self.gradients[idx].tolist(),
                "cosine_score": result.cosine_scores[idx],
                "l1_risk_score": details.get("risk_score"),
                "l1_norm_mad_score": details.get("norm_mad_score"),
                "l1_sign_disagreement": details.get("sign_disagreement"),
                "l1_reputation_risk": details.get("reputation_risk"),
                "l1_audit_age_score": details.get("audit_age_score"),
                "l1_router_mode": result.router_mode,
                "routing_reason": result.routing_reasons.get(idx, "cosine_screening"),
                "policy_round": policy_round,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            dispatch_l2_audit(suspect_data)

        logger.info(f"Batch processed: {len(result.clean_indices)} clean, {len(result.suspect_indices)} suspects")

        # Clear batch
        self.gradients = []
        self.vehicle_ids = []


# Global batcher instance
batcher = GradientBatcher()


def dispatch_l2_audit(suspect_data: dict) -> None:
    """Send a suspect gradient directly to the L2 Celery worker."""
    try:
        l2_audit_client.send_task(
            "l2_audit.audit_gradient",
            args=[suspect_data],
            queue=L2_AUDIT_QUEUE,
        )
    except Exception as exc:
        logger.error(
            "Failed to dispatch suspect gradient to L2 for %s: %s",
            suspect_data.get("vehicle_id", "unknown"),
            exc,
        )


def gradient_array_from_submission(submission: GradientSubmission) -> np.ndarray:
    """Build a validated numeric gradient array from a submitted payload."""
    return np.array(submission.gradient_data, dtype=float)


def ensure_consistent_gradient_shapes(gradients: List[np.ndarray]) -> None:
    """Reject mixed-dimensional batches before reaching aggregation math."""
    shapes = {gradient.shape for gradient in gradients}
    if len(shapes) != 1:
        raise HTTPException(status_code=400, detail="All gradients in a batch must have the same dimension")


# ============ ENDPOINTS ============

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check for L1 server."""
    checks = {}
    status = "healthy"

    # Check Redis
    if redis_client:
        try:
            redis_client.ping()
            checks["redis"] = "connected"
        except Exception as e:
            checks["redis"] = f"error: {str(e)}"
            status = "degraded"
    else:
        checks["redis"] = "not_available"
        status = "degraded"

    checks["batcher"] = "ready"

    return HealthResponse(
        status=status,
        timestamp=datetime.now(timezone.utc),
        checks=checks,
    )


@app.post("/api/v1/gradients", response_model=SubmissionResponse)
@limiter.limit("100/minute")
async def submit_gradient(
    request: Request,
    submission: GradientSubmission,
    api_key: str = Security(verify_api_key),
):
    """
    Submit a gradient from a vehicle.

    The gradient will be:
    1. Added to the current batch
    2. Screened using geometric median + cosine deviation
    3. If suspect, queued for L2 audit
    """
    try:
        gradient = gradient_array_from_submission(submission)

        # Add to batch
        await batcher.add(submission.vehicle_address, gradient)

        return SubmissionResponse(
            status="accepted",
            vehicle=submission.vehicle_address,
            is_suspect=False,
            message="Gradient queued for batch processing",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing gradient: {e}")
        raise HTTPException(status_code=500, detail="Failed to process gradient")


@app.post("/api/v1/batches/process", response_model=BatchResult)
async def process_batch_now(
    gradients: List[GradientSubmission],
    api_key: str = Security(verify_api_key),
):
    """Process a batch of gradients immediately."""
    if not gradients:
        raise HTTPException(status_code=400, detail="Empty gradient list")

    gradient_arrays = [gradient_array_from_submission(g) for g in gradients]
    ensure_consistent_gradient_shapes(gradient_arrays)
    vehicle_ids = [g.vehicle_address for g in gradients]

    policy_round = 0
    threshold = L1_SUSPECT_THRESHOLD
    recheck_probability = L1_RECHECK_PROBABILITY
    try:
        current_policy = await load_current_policy()
        policy_round = current_policy.round_id
        threshold = current_policy.cosine_filter_threshold
        recheck_probability = current_policy.recheck_probability
    except Exception as exc:
        logger.warning(f"Could not load current policy round for immediate batch: {exc}")

    result = filter_suspects(
        gradient_arrays,
        vehicle_ids,
        threshold=threshold,
        recheck_probability=recheck_probability,
        router_config=l1_router_config_from_env(),
        current_round=policy_round,
    )

    # Queue suspects for L2
    suspect_vehicles = []
    for idx in result.suspect_indices:
        suspect_vehicles.append(vehicle_ids[idx])
        details = result.l1_score_details.get(idx, {})
        suspect_data = {
            "vehicle_id": vehicle_ids[idx],
            "gradient": gradient_arrays[idx].tolist(),
            "cosine_score": result.cosine_scores[idx],
            "l1_risk_score": details.get("risk_score"),
            "l1_norm_mad_score": details.get("norm_mad_score"),
            "l1_sign_disagreement": details.get("sign_disagreement"),
            "l1_reputation_risk": details.get("reputation_risk"),
            "l1_audit_age_score": details.get("audit_age_score"),
            "l1_router_mode": result.router_mode,
            "routing_reason": result.routing_reasons.get(idx, "cosine_screening"),
            "policy_round": policy_round,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        dispatch_l2_audit(suspect_data)

    return BatchResult(
        total=len(gradients),
        clean=len(result.clean_indices),
        suspects=len(result.suspect_indices),
        suspect_vehicles=suspect_vehicles,
        aggregation_complete=True,
    )


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    # TODO: Implement prometheus_client metrics
    return {"message": "Metrics endpoint"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=L1_HOST, port=L1_PORT)

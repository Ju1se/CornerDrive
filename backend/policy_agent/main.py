"""
FLPG Policy Agent Service - Main FastAPI Application

This service implements a bounded adaptive policy agent that:
1. Collects telemetry from L1/L2/L3/L4 at round close
2. Proposes next-round policy parameters using GLM engine (direct control)
3. Validates proposals against hard bounds and safety guards
4. Stores frozen policies in Redis for round-based execution
5. Optionally commits policy hashes to blockchain

The policy agent is a control-plane component only.
It does NOT make direct fraud/rarity/honest/noise classifications.

UPDATED: GLM now directly controls policy parameters (not advisory).
"""

import os
import logging
import time
from contextlib import asynccontextmanager
from typing import Callable
from fastapi import FastAPI, Request, Response, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CollectorRegistry
from fastapi.responses import Response

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.schemas import (
    Policy, RoundTelemetry, PolicyProposal,
    PolicyBounds, PolicyMaxStep, DEFAULT_POLICY
)
from common.config import CORS_ALLOWED_ORIGINS, REDIS_URL

# Import routes
from policy_agent.api.routes_policy import router as policy_router
from policy_agent.api.exceptions import PolicyAgentException, http_exception_from_error

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics (use a separate registry to avoid conflicts)
REGISTRY = CollectorRegistry()
policy_proposals_total = Counter(
    'policy_proposals_total',
    'Total number of policy proposals generated',
    registry=REGISTRY
)
policy_activations_total = Counter(
    'policy_activations_total',
    'Total number of policy activations',
    registry=REGISTRY
)
proposal_duration = Histogram(
    'policy_proposal_duration_seconds',
    'Time taken to generate policy proposal',
    registry=REGISTRY
)
current_policy_gauge = Gauge(
    'policy_current_theta_tol',
    'Current policy theta_tol value',
    registry=REGISTRY
)
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status'],
    registry=REGISTRY
)
http_request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint'],
    registry=REGISTRY
)


# Global Redis store (will be initialized at startup)
redis_store = None


# ============================================================================
# Middleware
# ============================================================================

async def request_logging_middleware(request: Request, call_next: Callable) -> Response:
    """
    Middleware to log all HTTP requests and track metrics.
    """
    start_time = time.time()
    method = request.method
    path = request.url.path

    # Skip logging for health checks to reduce noise
    if path == "/health":
        return await call_next(request)

    logger.info(f"Incoming request: {method} {path}")

    try:
        response = await call_next(request)
        status_code = response.status_code
        duration = time.time() - start_time

        # Track metrics
        http_requests_total.labels(
            method=method,
            endpoint=path,
            status=status_code
        ).inc()
        http_request_duration.labels(
            method=method,
            endpoint=path
        ).observe(duration)

        logger.info(
            f"Request completed: {method} {path} - "
            f"Status: {status_code} - Duration: {duration:.3f}s"
        )

        return response

    except Exception as e:
        duration = time.time() - start_time
        status_code = getattr(e, "status_code", 500)

        http_requests_total.labels(
            method=method,
            endpoint=path,
            status=status_code
        ).inc()
        http_request_duration.labels(
            method=method,
            endpoint=path
        ).observe(duration)

        logger.error(
            f"Request failed: {method} {path} - "
            f"Status: {status_code} - Duration: {duration:.3f}s - Error: {str(e)}"
        )
        raise


async def exception_handler(request: Request, call_next: Callable) -> Response:
    """
    Global exception handler for custom exceptions.
    """
    try:
        return await call_next(request)
    except PolicyAgentException as e:
        # Convert custom exceptions to HTTP responses
        http_exc = http_exception_from_error(e)
        raise http_exc
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    global redis_store

    # Startup
    logger.info("Starting FLPG Policy Agent Service...")

    from policy_agent.storage.redis_store import RedisPolicyStore

    try:
        # Initialize Redis store
        redis_store = RedisPolicyStore(redis_url=REDIS_URL)

        # Check health
        is_healthy = await redis_store.health_check()
        if not is_healthy:
            logger.warning("Redis health check failed, but continuing...")

        # Ensure default policy exists
        current = await redis_store.get_current_policy()
        if not current:
            await redis_store.set_current_policy(DEFAULT_POLICY)
            logger.info("Default policy initialized")

        logger.info("Policy Agent Service started successfully")
    except Exception as e:
        logger.error(f"Failed to start Policy Agent Service: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down Policy Agent Service...")
    if redis_store:
        await redis_store.close()


# Create FastAPI app
app = FastAPI(
    title="FLPG Policy Agent",
    description="Bounded adaptive policy agent for FLPG federated learning system",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Add custom middleware
app.middleware("http")(request_logging_middleware)
app.middleware("http")(exception_handler)

# Include policy routes
app.include_router(policy_router)


# ============================================================================
# Health and Info Endpoints
# ============================================================================

@app.get("/health")
async def health():
    """Health check endpoint."""
    is_healthy = await redis_store.health_check() if redis_store else False

    return {
        "status": "ok" if is_healthy else "degraded",
        "service": "flpg-policy-agent",
        "version": "1.0.0",
        "redis": "connected" if is_healthy else "disconnected"
    }


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "FLPG Policy Agent",
        "description": "Bounded adaptive policy agent for FLPG",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health",
            "policy": {
                "health": "GET /api/v1/policy/health",
                "current": "GET /api/v1/policy/current",
                "next": "GET /api/v1/policy/next",
                "history": "GET /api/v1/policy/history/{round_id}",
                "propose": "POST /api/v1/policy/propose",
                "activate": "POST /api/v1/policy/activate",
                "explanation": "GET /api/v1/policy/explanation/{round_id}",
                "proposal_latest": "GET /api/v1/policy/proposal/latest",
            },
            "telemetry": {
                "latest": "GET /api/v1/policy/telemetry/latest",
                "by_round": "GET /api/v1/policy/telemetry/{round_id}",
                "history": "GET /api/v1/policy/telemetry",
            },
            "llm_stats": "GET /api/v1/llm/stats",
            "metrics": "GET /metrics",
        },
        "guide": "FLPG_Adaptive_Policy_Agent_Claude_Code_Guide.md"
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(REGISTRY), media_type="text/plain")


# ============================================================================
# GLM Engine Statistics Endpoint
# ============================================================================

@app.get("/api/v1/llm/stats")
async def get_llm_stats():
    """
    Get GLM policy engine status.

    Returns information about GLM policy engine:
    - Enabled status
    - Model configuration
    - API availability
    """
    try:
        from policy_agent.engine.glm_policy_engine import GLMPolicyEngine

        engine = GLMPolicyEngine()

        return {
            "service": "FLPG Policy Agent - GLM Engine",
            "engine_type": "glm_policy_engine",
            "enabled": engine.enabled,
            "model": engine.model if engine.enabled else None,
            "base_url": engine.base_url if engine.enabled else None,
            "description": "GLM directly controls policy parameters (not advisory)"
        }
    except Exception as e:
        logger.error(f"Error getting GLM stats: {e}")
        return {
            "error": "Failed to retrieve GLM engine statistics",
            "detail": str(e)
        }


# ============================================================================
# Explanation Endpoint (from GLM reasons)
# ============================================================================

@app.post("/api/v1/policy/explain")
async def explain_policy(proposal: PolicyProposal) -> dict:
    """
    Generate a human-readable explanation of policy changes.

    Returns the explanation from the proposal (generated by GLM).
    """
    try:
        explanation = proposal.explanation or proposal.summary()

        return {
            "explanation": explanation,
            "source": proposal.source_engine,
            "round_id": proposal.round_id,
            "blocked": proposal.blocked,
            "reasons": proposal.reasons,
            "llm_used": proposal.llm_used
        }

    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        # Return fallback explanation
        return {
            "explanation": proposal.summary(),
            "source": "fallback",
            "round_id": proposal.round_id,
            "blocked": proposal.blocked,
            "reasons": proposal.reasons
        }


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("POLICY_AGENT_HOST", "127.0.0.1")
    port = int(os.getenv("POLICY_AGENT_PORT", "8083"))

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )

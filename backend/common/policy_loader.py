"""
Policy loader for L1/L2/L3/L4 services.

This module provides a common interface for all layers to load
the current frozen policy from the policy agent.
"""

import asyncio
import logging
import os
from typing import Optional

import httpx
from redis.asyncio import Redis

from common.schemas import Policy, DEFAULT_POLICY
from common.config import REDIS_URL

logger = logging.getLogger(__name__)


# Cache policy to avoid repeated lookups within a round
_cached_policy: Optional[Policy] = None
_cached_round: Optional[int] = None


class PolicyLoader:
    """
    Loads frozen policy for use by L1/L2/L3/L4 services.

    Policy is loaded from:
    1. Redis cache (preferred - direct access)
    2. Policy Agent API (fallback)

    The policy is cached locally to avoid repeated lookups
    within the same round.
    """

    def __init__(
        self,
        redis_url: str = REDIS_URL,
        policy_agent_url: str = None
    ):
        """
        Initialize policy loader.

        Args:
            redis_url: Redis connection URL
            policy_agent_url: Policy agent API URL (fallback)
        """
        self.redis_url = redis_url
        self.policy_agent_url = (
            policy_agent_url
            or os.getenv("POLICY_AGENT_URL")
            or "http://localhost:8083"
        ).rstrip("/")
        self._redis: Optional[Redis] = None
        self._http_client: Optional[httpx.AsyncClient] = None

    async def _get_redis(self) -> Redis:
        """Get or create Redis client."""
        if self._redis is None:
            self._redis = Redis.from_url(self.redis_url)
        return self._redis

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=5.0)
        return self._http_client

    async def close(self):
        """Close connections."""
        if self._redis:
            await self._redis.close()
            self._redis = None
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def load_policy(
        self,
        round_id: Optional[int] = None,
        use_cache: bool = True
    ) -> Policy:
        """
        Load the current frozen policy.

        Args:
            round_id: Expected round ID (for validation)
            use_cache: Whether to use cached policy if available

        Returns:
            Current frozen policy

        Raises:
            Exception: If policy cannot be loaded
        """
        global _cached_policy, _cached_round

        # Check cache first, but only when the caller asks for a specific round.
        # For round-less lookups we refresh from storage so long-lived workers do
        # not keep using a stale policy indefinitely.
        if use_cache and _cached_policy is not None and round_id is not None:
            if _cached_round == round_id:
                logger.debug(f"Using cached policy for round {_cached_round}")
                return _cached_policy

        if round_id is not None:
            policy = await self._load_policy_for_round(round_id)
        else:
            policy = await self._load_from_redis()

        # Fallback to policy agent API
        if policy is None:
            policy = await self._load_from_api()

        # Final fallback to default
        if policy is None:
            logger.warning("Could not load policy, using default")
            policy = DEFAULT_POLICY

        # Update cache
        _cached_policy = policy
        _cached_round = policy.round_id

        logger.info(f"Loaded policy for round {policy.round_id}")

        return policy

    async def _load_from_redis(self) -> Optional[Policy]:
        """Load policy directly from Redis."""
        try:
            redis = await self._get_redis()
            data = await redis.get("policy:current")

            if data:
                return Policy.model_validate_json(data)

        except Exception as e:
            logger.warning(f"Failed to load policy from Redis: {e}")

        return None

    async def _load_policy_for_round(self, round_id: int) -> Optional[Policy]:
        """Load the policy that was active for a specific round."""
        try:
            redis = await self._get_redis()
            current_data = await redis.get("policy:current")
            if current_data:
                current_policy = Policy.model_validate_json(current_data)
                if current_policy.round_id == round_id:
                    return current_policy

            history_data = await redis.get(f"policy:history:r{round_id}")
            if history_data:
                return Policy.model_validate_json(history_data)
        except Exception as e:
            logger.warning(f"Failed to load policy for round {round_id} from Redis: {e}")

        return None

    async def _load_from_api(self) -> Optional[Policy]:
        """Load policy from policy agent API."""
        try:
            client = await self._get_http_client()
            response = await client.get(
                f"{self.policy_agent_url}/api/v1/policy/current"
            )

            if response.status_code == 200:
                data = response.json()
                return Policy(**data)

        except Exception as e:
            logger.warning(f"Failed to load policy from API: {e}")

        return None

    async def refresh_policy(self) -> Policy:
        """
        Force refresh the policy from the source.

        Returns:
            Updated policy
        """
        global _cached_policy, _cached_round

        policy = await self.load_policy(use_cache=False)

        return policy

    def get_policy_value(self, policy: Policy, field: str, default=None):
        """
        Get a specific value from the policy.

        This is a convenience method for layers that only need
        specific values.

        Args:
            policy: Loaded policy
            field: Field name to retrieve
            default: Default value if field not found

        Returns:
            Field value or default
        """
        return getattr(policy, field, default)


# Singleton instance
_loader_instance: Optional[PolicyLoader] = None


def get_policy_loader() -> PolicyLoader:
    """Get the singleton policy loader instance."""
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = PolicyLoader()
    return _loader_instance


async def load_current_policy(round_id: Optional[int] = None) -> Policy:
    """
    Convenience function to load the current policy.

    Args:
        round_id: Expected round ID for validation

    Returns:
        Current frozen policy
    """
    loader = get_policy_loader()
    return await loader.load_policy(round_id=round_id)


# Convenience functions for specific parameters
async def get_theta_tol() -> float:
    """Get current theta_tol (fraud threshold)."""
    policy = await load_current_policy()
    return policy.theta_tol


async def get_theta_rare() -> float:
    """Get current theta_rare (rarity threshold)."""
    policy = await load_current_policy()
    return policy.theta_rare


async def get_cosine_filter_threshold() -> float:
    """Get current cosine filter threshold."""
    policy = await load_current_policy()
    return policy.cosine_filter_threshold


async def get_recheck_probability() -> float:
    """Get current recheck probability."""
    policy = await load_current_policy()
    return policy.recheck_probability


async def get_slash_multiplier() -> float:
    """Get current slash multiplier."""
    policy = await load_current_policy()
    return policy.slash_multiplier


async def get_rarity_reward_multiplier() -> float:
    """Get current rarity reward multiplier."""
    policy = await load_current_policy()
    return policy.rarity_reward_multiplier


async def get_corner_weight() -> float:
    """Get current corner aggregation weight."""
    policy = await load_current_policy()
    return policy.corner_weight


async def get_theta_drift() -> float:
    """Get current drift threshold."""
    policy = await load_current_policy()
    return policy.theta_drift


# Clear cache (call at round start)
async def clear_policy_cache():
    """Clear the policy cache (call at start of each round)."""
    global _cached_policy, _cached_round
    _cached_policy = None
    _cached_round = None
    logger.debug("Policy cache cleared")

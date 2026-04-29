"""
Redis-based storage for FLPG Policy Agent.

Implements:
- Current policy storage (policy:current)
- Next policy storage (policy:next)
- Policy history snapshots (policy:history:r{round_id})
- Round telemetry storage (telemetry:round:r{round_id})
- Telemetry latest (telemetry:latest)
- Policy explanations (policy:explanation:r{round_id})

Updated to match FLPG_Adaptive_Policy_Agent_Claude_Code_Guide.md
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional

import redis.asyncio as aioredis
from redis.asyncio.connection import ConnectionPool

from common.schemas import Policy, RoundTelemetry, PolicyProposal, DEFAULT_POLICY

logger = logging.getLogger(__name__)


# Redis key patterns matching the guide specification
KEY_POLICY_CURRENT = "policy:current"
KEY_POLICY_NEXT = "policy:next"
KEY_POLICY_HISTORY = "policy:history:r{round_id}"
KEY_POLICY_PROPOSAL = "policy:proposal:r{round_id}"
KEY_POLICY_PROPOSAL_LATEST = "policy:proposal:latest"
KEY_POLICY_EXPLANATION = "policy:explanation:r{round_id}"
KEY_TELEMETRY_ROUND = "telemetry:round:r{round_id}"
KEY_TELEMETRY_LATEST = "telemetry:latest"


def proposal_recency_key(proposal: PolicyProposal) -> tuple[float, int]:
    """
    Sort proposals by creation time first, then by round id.

    Redis can retain stale demo proposals with very large round ids. Using
    created_at keeps "latest" aligned with the live run rather than the
    numerically largest historical round.
    """
    created_at = proposal.created_at
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    return (created_at.timestamp(), proposal.round_id)


class RedisPolicyStore:
    """
    Redis-based storage layer for policy agent data.

    Key design principles (from guide):
    - Current policy is frozen for the duration of a round
    - Historical policies are immutable snapshots
    - Next policy is prepared but not activated until round start
    """

    def __init__(self, redis_url: str):
        """
        Initialize Redis store.

        Args:
            redis_url: Redis connection URL (redis://localhost:6379/0)
        """
        self.redis_url = redis_url
        self._pool: Optional[ConnectionPool] = None
        self._client: Optional[aioredis.Redis] = None

    async def _get_client(self) -> aioredis.Redis:
        """Get or create Redis client."""
        if self._client is None:
            self._pool = ConnectionPool.from_url(self.redis_url)
            self._client = aioredis.Redis(connection_pool=self._pool)

        return self._client

    async def close(self):
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None
        if self._pool:
            await self._pool.disconnect()
            self._pool = None

    # ========================================================================
    # Policy Storage (Current / Next / History)
    # ========================================================================

    async def get_current_policy(self) -> Optional[Policy]:
        """
        Get the current frozen policy.

        Returns:
            Current policy or None if not set
        """
        client = await self._get_client()
        data = await client.get(KEY_POLICY_CURRENT)

        if not data:
            return None

        return Policy.model_validate_json(data)

    async def set_current_policy(self, policy: Policy) -> None:
        """
        Set the current frozen policy.

        Args:
            policy: Policy to set as current
        """
        client = await self._get_client()
        await client.set(
            KEY_POLICY_CURRENT,
            policy.model_dump_json(),
            # No expiry - current policy persists until explicitly changed
        )
        logger.info(f"Current policy set for round {policy.round_id}")

    async def get_next_policy(self) -> Optional[Policy]:
        """
        Get the proposed next policy (not yet active).

        Returns:
            Next policy or None if not set
        """
        client = await self._get_client()
        data = await client.get(KEY_POLICY_NEXT)

        if not data:
            return None

        return Policy.model_validate_json(data)

    async def set_next_policy(self, policy: Policy) -> None:
        """
        Set the next policy (approved but not yet active).

        Args:
            policy: Policy to set as next
        """
        client = await self._get_client()
        await client.set(
            KEY_POLICY_NEXT,
            policy.model_dump_json(),
            # No expiry - persists until activated
        )
        logger.info(f"Next policy set for round {policy.round_id}")

    async def get_policy(self, round_id: int) -> Optional[Policy]:
        """
        Get policy for a specific round from history.

        Args:
            round_id: Round number

        Returns:
            Policy or None if not found
        """
        client = await self._get_client()
        key = KEY_POLICY_HISTORY.format(round_id=round_id)
        data = await client.get(key)

        if not data:
            return None

        return Policy.model_validate_json(data)

    async def save_policy_history(self, policy: Policy) -> None:
        """
        Save policy to history (immutable snapshot).

        Args:
            policy: Policy to save
        """
        client = await self._get_client()
        key = KEY_POLICY_HISTORY.format(round_id=policy.round_id)

        await client.set(
            key,
            policy.model_dump_json(),
            # Keep history for 30 days
            ex=30 * 24 * 3600
        )

        logger.debug(f"Policy saved to history for round {policy.round_id}")

    async def get_recent_policies(
        self,
        count: int = 10
    ) -> list[Policy]:
        """
        Get recent historical policies, newest first.

        Args:
            count: Maximum number of policies to return

        Returns:
            Sorted list of policies
        """
        client = await self._get_client()
        pattern = KEY_POLICY_HISTORY.format(round_id="*")
        keys = []

        async for key in client.scan_iter(match=pattern, count=count):
            keys.append(key)

        policies = []
        for key in keys:
            data = await client.get(key)
            if data:
                policies.append(Policy.model_validate_json(data))

        policies.sort(key=lambda policy: policy.round_id, reverse=True)
        return policies[:count]

    async def save_explanation(
        self,
        round_id: int,
        explanation: str,
        metadata: Optional[dict] = None
    ) -> None:
        """
        Save policy explanation for a round.

        Args:
            round_id: Round number
            explanation: Human-readable explanation
            metadata: Optional additional metadata
        """
        client = await self._get_client()
        key = KEY_POLICY_EXPLANATION.format(round_id=round_id)

        data = {
            "explanation": explanation,
            "round_id": round_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {}
        }

        await client.set(
            key,
            json.dumps(data),
            # Keep for 90 days
            ex=90 * 24 * 3600
        )

        logger.debug(f"Explanation saved for round {round_id}")

    async def get_explanation(self, round_id: int) -> Optional[dict]:
        """
        Get policy explanation for a round.

        Args:
            round_id: Round number

        Returns:
            Explanation dict or None
        """
        client = await self._get_client()
        key = KEY_POLICY_EXPLANATION.format(round_id=round_id)
        data = await client.get(key)

        if not data:
            return None

        return json.loads(data)

    # ========================================================================
    # Telemetry Storage
    # ========================================================================

    async def save_telemetry(self, telemetry: RoundTelemetry) -> None:
        """
        Save round telemetry.

        Args:
            telemetry: Telemetry data to save
        """
        client = await self._get_client()

        # Save to round-specific key
        round_key = KEY_TELEMETRY_ROUND.format(round_id=telemetry.round_id)
        await client.set(
            round_key,
            telemetry.model_dump_json(),
            # Keep telemetry for 30 days
            ex=30 * 24 * 3600
        )

        # Also save as latest
        await client.set(
            KEY_TELEMETRY_LATEST,
            telemetry.model_dump_json(),
            # Keep latest telemetry for 90 days
            ex=90 * 24 * 3600
        )

        logger.debug(f"Telemetry saved for round {telemetry.round_id}")

    async def get_telemetry(self, round_id: int) -> Optional[RoundTelemetry]:
        """
        Get telemetry for a specific round.

        Args:
            round_id: Round number

        Returns:
            Telemetry or None if not found
        """
        client = await self._get_client()
        key = KEY_TELEMETRY_ROUND.format(round_id=round_id)
        data = await client.get(key)

        if not data:
            return None

        return RoundTelemetry.model_validate_json(data)

    async def get_latest_telemetry(self) -> Optional[RoundTelemetry]:
        """
        Get the most recent telemetry.

        Returns:
            Latest telemetry or None
        """
        client = await self._get_client()
        data = await client.get(KEY_TELEMETRY_LATEST)

        if not data:
            return None

        return RoundTelemetry.model_validate_json(data)

    async def get_recent_telemetry(
        self,
        count: int = 10
    ) -> list[RoundTelemetry]:
        """
        Get recent telemetry entries.

        Args:
            count: Maximum number of entries to return

        Returns:
            List of recent telemetry, newest first
        """
        client = await self._get_client()

        # Scan for telemetry keys
        pattern = KEY_TELEMETRY_ROUND.format(round_id="*")
        keys = []
        async for key in client.scan_iter(match=pattern, count=count):
            keys.append(key)

        # Fetch data for each key
        telemetry_list = []
        for key in keys:
            data = await client.get(key)
            if data:
                telemetry_list.append(RoundTelemetry.model_validate_json(data))

        # Sort by round_id descending
        telemetry_list.sort(key=lambda t: t.round_id, reverse=True)

        return telemetry_list[:count]

    # ========================================================================
    # Proposal Storage
    # ========================================================================

    async def save_proposal(self, proposal: PolicyProposal) -> None:
        """
        Save a policy proposal.

        Args:
            proposal: PolicyProposal to save
        """
        client = await self._get_client()

        # Save by round ID
        key = KEY_POLICY_PROPOSAL.format(round_id=proposal.round_id)
        await client.set(
            key,
            proposal.model_dump_json(),
            # Keep for 30 days
            ex=30 * 24 * 3600
        )
        await client.set(
            KEY_POLICY_PROPOSAL_LATEST,
            proposal.model_dump_json(),
            ex=30 * 24 * 3600
        )

        logger.debug(f"Proposal saved for round {proposal.round_id}")

    async def get_proposal(self, round_id: int) -> Optional[PolicyProposal]:
        """
        Get proposal for a specific round.

        Args:
            round_id: Round number

        Returns:
            PolicyProposal or None
        """
        client = await self._get_client()
        key = KEY_POLICY_PROPOSAL.format(round_id=round_id)
        data = await client.get(key)

        if not data:
            return None

        return PolicyProposal.model_validate_json(data)

    async def get_recent_proposals(
        self,
        count: int = 10
    ) -> list[PolicyProposal]:
        """
        Get recent proposals, newest first.

        Args:
            count: Maximum number of proposals to return

        Returns:
            Sorted list of proposals
        """
        client = await self._get_client()
        pattern = KEY_POLICY_PROPOSAL.format(round_id="*")
        keys = []

        async for key in client.scan_iter(match=pattern, count=count):
            keys.append(key)

        proposals = []
        for key in keys:
            data = await client.get(key)
            if data:
                proposals.append(PolicyProposal.model_validate_json(data))

        proposals.sort(key=proposal_recency_key, reverse=True)
        return proposals[:count]

    async def get_latest_proposal(self) -> Optional[PolicyProposal]:
        """Get the most recent policy proposal."""
        client = await self._get_client()
        latest = await client.get(KEY_POLICY_PROPOSAL_LATEST)
        latest_pointer = PolicyProposal.model_validate_json(latest) if latest else None

        proposals = await self.get_recent_proposals(count=1)
        latest_scanned = proposals[0] if proposals else None

        if latest_pointer and latest_scanned:
            winner = max((latest_pointer, latest_scanned), key=proposal_recency_key)
            if winner != latest_pointer:
                await client.set(
                    KEY_POLICY_PROPOSAL_LATEST,
                    winner.model_dump_json(),
                    ex=30 * 24 * 3600
                )
            return winner

        return latest_pointer or latest_scanned

    # ========================================================================
    # Policy Lifecycle
    # ========================================================================

    async def activate_next_policy(self, round_id: int) -> Optional[Policy]:
        """
        Activate the next policy: move policy:next to policy:current.

        Args:
            round_id: Round that is starting

        Returns:
            The activated policy, or None if no next policy exists
        """
        next_policy = await self.get_next_policy()

        if next_policy is None:
            logger.warning(f"No next policy found for round {round_id}")
            return None

        # Archive current policy to history first
        current_policy = await self.get_current_policy()
        if current_policy:
            await self.save_policy_history(current_policy)

        # Set next as current
        await self.set_current_policy(next_policy)

        # Clear next policy
        client = await self._get_client()
        await client.delete(KEY_POLICY_NEXT)

        logger.info(f"Activated policy for round {round_id}: moved from next to current")

        return next_policy

    # ========================================================================
    # Health Check
    # ========================================================================

    async def health_check(self) -> bool:
        """
        Check if Redis connection is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            client = await self._get_client()
            await client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False

"""
Utility functions for policy hashing and signing.
"""

import hashlib
import json
from typing import Any, Optional


def compute_policy_hash(policy: Any) -> str:
    """
    Compute SHA256 hash of policy parameters.

    Args:
        policy: Policy object or dict

    Returns:
        Hex-encoded SHA256 hash
    """
    if hasattr(policy, 'model_dump'):
        policy_dict = policy.model_dump()
    elif isinstance(policy, dict):
        policy_dict = policy
    else:
        raise TypeError(f"Cannot compute hash for type {type(policy)}")

    # Remove fields that shouldn't affect the hash
    fields_to_remove = [
        'policy_hash',
        'approved',
        'signed_by',
        'created_at',
    ]

    for field in fields_to_remove:
        policy_dict.pop(field, None)

    # Sort keys for deterministic hashing
    canonical = json.dumps(policy_dict, sort_keys=True)

    return hashlib.sha256(canonical.encode()).hexdigest()


def compute_commitment(
    policy_hash: str,
    round_id: int,
    salt: Optional[str] = None
) -> str:
    """
    Compute a commitment hash for on-chain storage.

    Args:
        policy_hash: Hash of the policy
        round_id: Round number
        salt: Optional salt for commitment

    Returns:
        Hex-encoded commitment hash
    """
    commitment_data = {
        "policy_hash": policy_hash,
        "round_id": round_id,
    }

    if salt:
        commitment_data["salt"] = salt

    canonical = json.dumps(commitment_data, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def verify_commitment(
    commitment: str,
    policy_hash: str,
    round_id: int,
    salt: Optional[str] = None
) -> bool:
    """
    Verify a commitment hash.

    Args:
        commitment: The commitment hash to verify
        policy_hash: Policy hash
        round_id: Round number
        salt: Salt used in commitment (if any)

    Returns:
        True if commitment is valid
    """
    computed = compute_commitment(policy_hash, round_id, salt)
    return computed == commitment


def sign_policy_hash(
    policy_hash: str,
    private_key: str
) -> str:
    """
    Sign a policy hash with a private key.

    Note: This is a placeholder. In production, use proper cryptographic signing.

    Args:
        policy_hash: Hash to sign
        private_key: Private key for signing

    Returns:
        Signature (placeholder implementation)
    """
    # Placeholder: In production, use eth_account or similar
    import hmac

    signature = hmac.new(
        private_key.encode(),
        policy_hash.encode(),
        hashlib.sha256
    ).hexdigest()

    return signature

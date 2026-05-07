"""
L4: Settlement Dashboard API
Provides analytics and management interface for FLPG system.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import redis
from fastapi import FastAPI, HTTPException, Query, Security
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from web3 import Web3

from common.config import (
    L4_DASHBOARD_HOST,
    L4_DASHBOARD_PORT,
    REDIS_URL,
    GANACHE_URL,
    L4_CORS_ALLOWED_ORIGINS,
    L4_CONTRACT_ADDRESS,
    L4_ORACLE_PRIVATE_KEY,
    L3_DRIFT_THRESHOLD,
    L3_GOLDEN_DATASET_PATH,
)
from common.policy_loader import load_current_policy
from common.schemas import Policy
from common.security import verify_api_key

logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="FLPG L4 - Settlement Dashboard",
    description="Analytics and management for FLPG system",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=L4_CORS_ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis client
try:
    redis_client = redis.from_url(REDIS_URL)
    redis_client.ping()
    redis_available = True
except redis.ConnectionError:
    logger.error(f"Cannot connect to Redis at {REDIS_URL}")
    redis_client = None
    redis_available = False

# Web3 client
try:
    w3 = Web3(Web3.HTTPProvider(GANACHE_URL))
    blockchain_available = w3.is_connected()
except Exception as e:
    logger.error(f"Cannot connect to blockchain: {e}")
    w3 = None
    blockchain_available = False


# ============ SCHEMAS ============

class SystemStats(BaseModel):
    """System-wide statistics."""
    total_vehicles: int
    total_audits: int
    fraud_count: int
    rare_count: int
    honest_count: int
    noise_count: int
    fraud_rate: float
    total_rewards_distributed: float
    total_slashed: float


class VehicleStats(BaseModel):
    """Individual vehicle statistics."""
    address: str
    reputation: int
    tier: str
    tier_multiplier: float
    total_contributions: int
    fraud_count: int
    rare_count: int
    stake: float
    rewards_earned: float
    is_registered: bool


class TierDistribution(BaseModel):
    """Distribution of vehicles across tiers."""
    bronze: int
    silver: int
    gold: int
    platinum: int


class RecentAudit(BaseModel):
    """Recent audit record."""
    vehicle_id: str
    classification: str
    delta_loss_main: float
    delta_loss_corner: float
    sbt_points: int
    routing_reason: str = "unknown"
    timestamp: datetime


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    layer: str = "L4"
    service: str = "settlement_dashboard"
    timestamp: datetime
    checks: Dict[str, str]


class L3DatasetStatus(BaseModel):
    """Live view of L3's golden dataset configuration and source."""
    lifecycle: str
    dataset_source: str
    dataset_path: str
    dataset_artifacts_present: bool
    sample_count: Optional[int] = None
    sample_shape: Optional[List[int]] = None
    drift_threshold: float
    policy_round: Optional[int] = None
    detail: str


class SettlementBatch(BaseModel):
    """Batch settlement request."""
    round_id: int
    honest_vehicles: List[str]
    rarity_vehicles: List[str]
    fraud_vehicles: List[str]
    settlement_id: Optional[str] = None


# ============ HELPER FUNCTIONS ============

FLPG_AUDIT_ABI = [
    {
        "inputs": [
            {"internalType": "uint256", "name": "roundId", "type": "uint256"},
            {"internalType": "uint256", "name": "honestRewardMultiplierBps", "type": "uint256"},
            {"internalType": "uint256", "name": "slashMultiplierBps", "type": "uint256"},
            {"internalType": "uint256", "name": "rarityRewardMultiplierBps", "type": "uint256"},
        ],
        "name": "setRoundEconomicPolicy",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "roundId", "type": "uint256"},
            {"internalType": "address[]", "name": "honestVehicles", "type": "address[]"},
            {"internalType": "address[]", "name": "rarityVehicles", "type": "address[]"},
            {"internalType": "address[]", "name": "fraudVehicles", "type": "address[]"},
            {"internalType": "bytes32", "name": "settlementId", "type": "bytes32"},
        ],
        "name": "settleBatch",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "uint256", "name": "roundId", "type": "uint256"}],
        "name": "getRoundEconomicPolicy",
        "outputs": [
            {"internalType": "uint256", "name": "honestRewardMultiplierBps", "type": "uint256"},
            {"internalType": "uint256", "name": "slashMultiplierBps", "type": "uint256"},
            {"internalType": "uint256", "name": "rarityRewardMultiplierBps", "type": "uint256"},
            {"internalType": "bool", "name": "configured", "type": "bool"},
            {"internalType": "bool", "name": "settlementLocked", "type": "bool"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "bytes32", "name": "", "type": "bytes32"}],
        "name": "processedSettlementIds",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "address", "name": "vehicle", "type": "address"}],
        "name": "getVehicleStats",
        "outputs": [
            {"internalType": "int256", "name": "rep", "type": "int256"},
            {"internalType": "uint256", "name": "contribs", "type": "uint256"},
            {"internalType": "uint256", "name": "frauds", "type": "uint256"},
            {"internalType": "uint256", "name": "rarities", "type": "uint256"},
            {"internalType": "uint256", "name": "stake", "type": "uint256"},
            {"internalType": "string", "name": "tier", "type": "string"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "address", "name": "vehicle", "type": "address"}],
        "name": "isRegistered",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function",
    },
]

def get_tier_from_reputation(reputation: int) -> str:
    """Determine tier from reputation score."""
    if reputation >= 1000:
        return "PLATINUM"
    elif reputation >= 500:
        return "GOLD"
    elif reputation >= 100:
        return "SILVER"
    return "BRONZE"


def get_tier_multiplier(tier: str) -> float:
    """Get reward multiplier for tier."""
    multipliers = {
        "BRONZE": 1.0,
        "SILVER": 1.2,
        "GOLD": 1.5,
        "PLATINUM": 2.0,
    }
    return multipliers.get(tier, 1.0)


def get_contract():
    """Get contract instance."""
    if not w3 or not L4_CONTRACT_ADDRESS:
        return None

    return w3.eth.contract(
        address=Web3.to_checksum_address(L4_CONTRACT_ADDRESS),
        abi=FLPG_AUDIT_ABI,
    )


def submit_contract_transaction(contract_fn, account, nonce: int) -> str:
    """Sign and submit a single contract transaction."""
    tx = contract_fn.build_transaction({
        "from": account.address,
        "nonce": nonce,
        "gas": 300000,
        "gasPrice": w3.eth.gas_price,
        "chainId": w3.eth.chain_id,
    })
    signed_tx = w3.eth.account.sign_transaction(tx, L4_ORACLE_PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    if receipt.status != 1:
        raise RuntimeError(f"Transaction failed: {tx_hash.hex()}")
    return tx_hash.hex()


def multiplier_to_bps(multiplier: float) -> int:
    """Convert a floating-point multiplier into integer basis points."""
    return int(round(multiplier * 10_000))


def normalize_vehicle_list(vehicles: List[str]) -> List[str]:
    """Normalize vehicles to checksum format and stable ordering."""
    normalized = [Web3.to_checksum_address(vehicle) for vehicle in vehicles]
    return sorted(normalized, key=lambda address: address.lower())


def l3_dataset_artifacts_present(dataset_path: Path) -> bool:
    """Check whether the configured L3 dataset path contains supported artifacts."""
    if dataset_path.is_file():
        return True

    if not dataset_path.is_dir():
        return False

    candidate_names = (
        "dataset.pt",
        "dataset.pth",
        "golden_dataset.pt",
        "golden_dataset.pth",
        "data.pt",
    )
    return any((dataset_path / candidate).exists() for candidate in candidate_names)


def ensure_unique_vehicle_assignments(*groups: List[str]) -> None:
    """Reject duplicate settlement assignments within or across categories."""
    seen: set[str] = set()

    for group in groups:
        for vehicle in group:
            key = vehicle.lower()
            if key in seen:
                raise HTTPException(
                    status_code=400,
                    detail=f"Vehicle assigned multiple times in settlement batch: {vehicle}",
                )
            seen.add(key)


def resolve_settlement_id(
    round_id: int,
    honest_vehicles: List[str],
    rarity_vehicles: List[str],
    fraud_vehicles: List[str],
    settlement_id: Optional[str],
) -> tuple[bytes, str]:
    """Use a caller-provided id when available, otherwise hash the canonical payload."""
    if settlement_id:
        candidate = settlement_id.strip().lower()
        if candidate.startswith("0x") and len(candidate) == 66:
            try:
                raw_bytes = bytes.fromhex(candidate[2:])
                return raw_bytes, candidate
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="Invalid settlement_id hex") from exc

        raw_bytes = hashlib.sha256(candidate.encode("utf-8")).digest()
        return raw_bytes, f"0x{raw_bytes.hex()}"

    # Proof payloads are intentionally excluded here because on-chain replay safety
    # is keyed to the actual settlement effect: round + categorized vehicle lists.
    canonical_payload = json.dumps(
        {
            "round_id": round_id,
            "honest": [vehicle.lower() for vehicle in honest_vehicles],
            "rarity": [vehicle.lower() for vehicle in rarity_vehicles],
            "fraud": [vehicle.lower() for vehicle in fraud_vehicles],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    raw_bytes = hashlib.sha256(canonical_payload.encode("utf-8")).digest()
    return raw_bytes, f"0x{raw_bytes.hex()}"


async def load_policy_for_round(round_id: int) -> Policy:
    """Load the frozen policy for the settlement round."""
    if redis_available:
        current_payload = redis_client.get("policy:current")
        if current_payload:
            current_policy = Policy.model_validate_json(current_payload)
            if current_policy.round_id == round_id:
                return current_policy

        history_payload = redis_client.get(f"policy:history:r{round_id}")
        if history_payload:
            return Policy.model_validate_json(history_payload)

    policy = await load_current_policy(round_id=round_id)
    if policy.round_id != round_id:
        raise RuntimeError(
            f"Could not load frozen policy for round {round_id}; current policy is round {policy.round_id}"
        )
    return policy


def sync_round_policy_on_chain(
    contract,
    account,
    nonce: int,
    round_id: int,
    policy: Policy,
) -> tuple[Optional[str], int]:
    """Ensure the contract has the same economic multipliers frozen for this round."""
    target_slash_bps = multiplier_to_bps(policy.slash_multiplier)
    target_rarity_bps = multiplier_to_bps(policy.rarity_reward_multiplier)
    target_honest_bps = multiplier_to_bps(policy.honest_reward_multiplier)

    try:
        current_round_policy = contract.functions.getRoundEconomicPolicy(round_id).call()
    except Exception as exc:
        raise RuntimeError(
            "FLPGAudit contract does not expose round policy functions; redeploy the updated contract"
        ) from exc

    (
        current_honest_bps,
        current_slash_bps,
        current_rarity_bps,
        configured,
        settlement_locked,
    ) = current_round_policy

    if (
        configured
        and current_honest_bps == target_honest_bps
        and current_slash_bps == target_slash_bps
        and current_rarity_bps == target_rarity_bps
    ):
        return None, nonce

    if settlement_locked:
        raise RuntimeError(f"Round {round_id} is already locked on-chain with different economic policy")

    tx_hash = submit_contract_transaction(
        contract.functions.setRoundEconomicPolicy(
            round_id,
            target_honest_bps,
            target_slash_bps,
            target_rarity_bps,
        ),
        account,
        nonce,
    )
    return tx_hash, nonce + 1


# ============ ENDPOINTS ============

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check."""
    checks = {}
    status = "healthy"

    # Check Redis
    if redis_available:
        try:
            redis_client.ping()
            checks["redis"] = "connected"
        except Exception as e:
            checks["redis"] = f"error: {str(e)}"
            status = "degraded"
    else:
        checks["redis"] = "not_available"
        status = "degraded"

    # Check blockchain
    if blockchain_available:
        try:
            w3.eth.block_number
            checks["blockchain"] = "connected"
        except Exception as e:
            checks["blockchain"] = f"error: {str(e)}"
            status = "degraded"
    else:
        checks["blockchain"] = "not_available"
        status = "degraded"

    # Check contract
    contract = get_contract()
    if contract:
        checks["contract"] = "available"
    else:
        checks["contract"] = "not_available"
        status = "degraded"

    return HealthResponse(
        status=status,
        timestamp=datetime.now(timezone.utc),
        checks=checks,
    )


@app.get("/api/v1/stats", response_model=SystemStats)
async def get_system_stats():
    """Get system-wide statistics."""
    if not redis_available:
        raise HTTPException(status_code=503, detail="Redis not available")

    try:
        # Get counts from Redis
        fraud_count = int(redis_client.get("stats:fraud_count") or 0)
        rare_count = int(redis_client.get("stats:rare_count") or 0)
        honest_count = int(redis_client.get("stats:honest_count") or 0)
        noise_count = int(redis_client.get("stats:noise_count") or 0)

        total_audits = fraud_count + rare_count + honest_count + noise_count

        # Get vehicle count
        total_vehicles = int(redis_client.get("stats:total_vehicles") or 0)

        # Calculate fraud rate
        fraud_rate = fraud_count / max(total_audits, 1)

        # Get financial stats
        total_rewards = float(redis_client.get("stats:total_rewards") or 0)
        total_slashed = float(redis_client.get("stats:total_slashed") or 0)

        return SystemStats(
            total_vehicles=total_vehicles,
            total_audits=total_audits,
            fraud_count=fraud_count,
            rare_count=rare_count,
            honest_count=honest_count,
            noise_count=noise_count,
            fraud_rate=fraud_rate,
            total_rewards_distributed=total_rewards,
            total_slashed=total_slashed,
        )

    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/l3/status", response_model=L3DatasetStatus)
async def get_l3_status():
    """Report whether L3 is backed by a real golden dataset or a placeholder."""
    drift_threshold = L3_DRIFT_THRESHOLD
    policy_round = None

    try:
        policy = await load_current_policy()
        drift_threshold = policy.theta_drift
        policy_round = policy.round_id
    except Exception as exc:
        logger.warning(f"Unable to load active policy while reading L3 status: {exc}")

    dataset_path = Path(L3_GOLDEN_DATASET_PATH)
    artifacts_present = l3_dataset_artifacts_present(dataset_path)

    try:
        from l3_gatekeeper.validator import GoldenDatasetManager

        manager = GoldenDatasetManager(dataset_path=str(dataset_path))
        dataset_summary = manager.describe()
        dataset_source = dataset_summary["dataset_source"]

        if dataset_source == "disk":
            detail = "Golden dataset artifacts are loaded from disk for library-only L3 validation."
        else:
            detail = (
                "No supported golden dataset artifacts were found, so L3 is currently "
                "validating with a placeholder dataset."
            )

        return L3DatasetStatus(
            lifecycle="library_only",
            dataset_source=dataset_source,
            dataset_path=dataset_summary["dataset_path"],
            dataset_artifacts_present=dataset_summary["dataset_artifacts_present"],
            sample_count=dataset_summary["sample_count"],
            sample_shape=dataset_summary["sample_shape"],
            drift_threshold=drift_threshold,
            policy_round=policy_round,
            detail=detail,
        )

    except Exception as exc:
        logger.error(f"Unable to inspect L3 dataset status: {exc}")
        detail = (
            "Golden dataset artifacts exist but could not be loaded for L3 validation."
            if artifacts_present
            else "L3 dataset inspection is unavailable."
        )

        return L3DatasetStatus(
            lifecycle="library_only",
            dataset_source="invalid_artifacts" if artifacts_present else "inspection_error",
            dataset_path=str(dataset_path),
            dataset_artifacts_present=artifacts_present,
            sample_count=None,
            sample_shape=None,
            drift_threshold=drift_threshold,
            policy_round=policy_round,
            detail=f"{detail} {exc}",
        )


@app.get("/api/v1/vehicle/{address}", response_model=VehicleStats)
async def get_vehicle_stats(address: str):
    """Get statistics for a specific vehicle."""
    if not redis_available:
        raise HTTPException(status_code=503, detail="Redis not available")

    address = address.lower()

    try:
        # Get from Redis
        vehicle_data = redis_client.hgetall(f"vehicle:{address}")

        if not vehicle_data:
            raise HTTPException(status_code=404, detail="Vehicle not found")

        reputation = int(vehicle_data.get(b"reputation", 0))
        tier = get_tier_from_reputation(reputation)

        return VehicleStats(
            address=address,
            reputation=reputation,
            tier=tier,
            tier_multiplier=get_tier_multiplier(tier),
            total_contributions=int(vehicle_data.get(b"contributions", 0)),
            fraud_count=int(vehicle_data.get(b"fraud_count", 0)),
            rare_count=int(vehicle_data.get(b"rare_count", 0)),
            stake=float(vehicle_data.get(b"stake", 0)),
            rewards_earned=float(vehicle_data.get(b"rewards", 0)),
            is_registered=vehicle_data.get(b"registered", b"false").decode().lower() == "true",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting vehicle stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/tiers", response_model=TierDistribution)
async def get_tier_distribution():
    """Get distribution of vehicles across tiers."""
    if not redis_available:
        raise HTTPException(status_code=503, detail="Redis not available")

    try:
        bronze = int(redis_client.get("tier:bronze") or 0)
        silver = int(redis_client.get("tier:silver") or 0)
        gold = int(redis_client.get("tier:gold") or 0)
        platinum = int(redis_client.get("tier:platinum") or 0)

        return TierDistribution(
            bronze=bronze,
            silver=silver,
            gold=gold,
            platinum=platinum,
        )

    except Exception as e:
        logger.error(f"Error getting tier distribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/recent-audits", response_model=List[RecentAudit])
async def get_recent_audits(limit: int = Query(default=20, le=100)):
    """Get recent audit results."""
    if not redis_available:
        raise HTTPException(status_code=503, detail="Redis not available")

    try:
        # Get from Redis list
        audits = redis_client.lrange("recent_audits", 0, limit - 1)

        results = []
        for audit_data in audits:
            try:
                audit = json.loads(audit_data)
                results.append(RecentAudit(
                    vehicle_id=audit["vehicle_id"],
                    classification=audit["classification"],
                    delta_loss_main=audit["delta_loss_main"],
                    delta_loss_corner=audit["delta_loss_corner"],
                    sbt_points=audit["sbt_points"],
                    routing_reason=audit.get("routing_reason", "unknown"),
                    timestamp=datetime.fromisoformat(audit["timestamp"]),
                ))
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Skipping malformed audit record: {e}")
                continue

        return results

    except Exception as e:
        logger.error(f"Error getting recent audits: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/vehicles")
async def get_vehicles(
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=10, le=100),
    tier: Optional[str] = None,
):
    """Get paginated list of vehicles."""
    if not redis_available:
        raise HTTPException(status_code=503, detail="Redis not available")

    try:
        # Get all vehicle keys
        pattern = "vehicle:*"
        cursor = 0
        vehicles = []

        while True:
            cursor, keys = redis_client.scan(cursor, match=pattern, count=100)
            for key in keys:
                try:
                    data = redis_client.hgetall(key)
                    if data:
                        address = key.decode().split(":")[1]
                        rep = int(data.get(b"reputation", 0))
                        vehicle_tier = get_tier_from_reputation(rep)

                        if tier and vehicle_tier != tier.upper():
                            continue

                        vehicles.append({
                            "address": address,
                            "reputation": rep,
                            "tier": vehicle_tier,
                            "contributions": int(data.get(b"contributions", 0)),
                        })
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Skipping malformed vehicle data: {e}")
                    continue

            if cursor == 0:
                break

        # Sort by reputation descending
        vehicles.sort(key=lambda x: x["reputation"], reverse=True)

        # Paginate
        start = (page - 1) * limit
        end = start + limit

        return {
            "total": len(vehicles),
            "page": page,
            "limit": limit,
            "data": vehicles[start:end],
        }

    except Exception as e:
        logger.error(f"Error getting vehicles: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/settle/batch")
async def process_batch_settlement(
    batch: SettlementBatch,
    api_key: str = Security(verify_api_key),
):
    """Process a round-scoped settlement batch against the blockchain."""
    _ = api_key
    if not blockchain_available:
        raise HTTPException(status_code=503, detail="Blockchain not available")

    if not L4_ORACLE_PRIVATE_KEY:
        raise HTTPException(status_code=500, detail="Oracle private key not configured")

    try:
        contract = get_contract()
        if not contract:
            raise HTTPException(status_code=503, detail="Contract not available")

        honest_vehicles = normalize_vehicle_list(batch.honest_vehicles)
        rarity_vehicles = normalize_vehicle_list(batch.rarity_vehicles)
        fraud_vehicles = normalize_vehicle_list(batch.fraud_vehicles)
        ensure_unique_vehicle_assignments(honest_vehicles, rarity_vehicles, fraud_vehicles)

        settlement_id_bytes, settlement_id_hex = resolve_settlement_id(
            batch.round_id,
            honest_vehicles,
            rarity_vehicles,
            fraud_vehicles,
            batch.settlement_id,
        )

        try:
            already_processed = contract.functions.processedSettlementIds(settlement_id_bytes).call()
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail="Deployed FLPGAudit contract is missing settlement replay protection; redeploy required",
            ) from exc

        if already_processed:
            return {
                "status": "duplicate",
                "message": "Settlement batch was already processed on-chain",
                "round_id": batch.round_id,
                "settlement_id": settlement_id_hex,
                "submitted": {
                    "honest": len(honest_vehicles),
                    "rarity": len(rarity_vehicles),
                    "fraud": len(fraud_vehicles),
                },
            }

        policy = await load_policy_for_round(batch.round_id)
        account = w3.eth.account.from_key(L4_ORACLE_PRIVATE_KEY)
        nonce = w3.eth.get_transaction_count(account.address)
        policy_tx_hash, nonce = sync_round_policy_on_chain(
            contract,
            account,
            nonce,
            batch.round_id,
            policy,
        )
        settlement_tx_hash = submit_contract_transaction(
            contract.functions.settleBatch(
                batch.round_id,
                honest_vehicles,
                rarity_vehicles,
                fraud_vehicles,
                settlement_id_bytes,
            ),
            account,
            nonce,
        )

        return {
            "status": "success",
            "message": "Round settlement batch processed against FLPGAudit",
            "round_id": batch.round_id,
            "settlement_id": settlement_id_hex,
            "submitted": {
                "honest": len(honest_vehicles),
                "rarity": len(rarity_vehicles),
                "fraud": len(fraud_vehicles),
            },
            "policy": {
                "round_id": policy.round_id,
                "honest_reward_multiplier": policy.honest_reward_multiplier,
                "slash_multiplier": policy.slash_multiplier,
                "rarity_reward_multiplier": policy.rarity_reward_multiplier,
            },
            "tx_hashes": {
                "policy_sync": policy_tx_hash,
                "settlement": settlement_tx_hash,
            },
        }

    except Exception as e:
        logger.error(f"Error processing batch settlement: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    # TODO: Implement prometheus_client metrics
    return {"message": "Metrics endpoint"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=L4_DASHBOARD_HOST, port=L4_DASHBOARD_PORT)

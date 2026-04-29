# L4: On-Chain Settlement

## Purpose

L4 implements the blockchain-based settlement layer that manages incentives, enforces accountability, and maintains an immutable record of all federated learning activities. It connects the AI/ML system with cryptographic trust and economic incentives.

## Core Components

### Smart Contract System

#### Main Settlement Contract

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";

contract FLPGSettlement is ERC721, Ownable, ReentrancyGuard {
    struct Participant {
        address participant;
        uint256 sbtBalance;           // Soul-bound token balance
        SBTTier tier;                 // Current tier: Bronze, Silver, Gold, Platinum
        uint256 totalContributions;    // Total contributions made
        uint256 fraudCount;           // Number of fraud detections
        uint256 rarityCount;          // Number of rarity discoveries
        uint256 lastActivity;         // Last activity timestamp
        bool active;                  // Currently active participant
    }

    enum SBTTier {
        BRONZE,    // 0-99 points
        SILVER,    // 100-499 points
        GOLD,      // 500-999 points
        PLATINUM   // 1000+ points
    }

    enum AuditResult {
        HONEST,    // +1 SBT
        RARITY,    // +10 SBT
        NOISE,     // 0 SBT
        FRAUD      // -50 SBT
    }

    // Mappings
    mapping(address => Participant) public participants;
    mapping(bytes32 => bool) public processedProofs;  // Prevent double-spending
    mapping(SBTTier => uint256) public tierMultipliers;

    // Events
    event ContributionSettled(
        address indexed participant,
        bytes32 batchId,
        AuditResult result,
        int256 sbtChange,
        SBTTier newTier
    );

    event FraudProofSubmitted(
        address indexed participant,
        bytes32 proofHash,
        uint256 penalty
    );

    event RarityCertificateIssued(
        address indexed participant,
        bytes32 certificateHash,
        uint256 reward
    );

    event TierUpgraded(
        address indexed participant,
        SBTTier oldTier,
        SBTTier newTier
    );

    constructor() ERC721("FLPG Soul Bound Token", "FLPG-SBT") {
        // Initialize tier multipliers (basis points, 100 = 1x)
        tierMultipliers[SBTTier.BRONZE] = 100;
        tierMultipliers[SBTTier.SILVER] = 120;
        tierMultipliers[SBTTier.GOLD] = 150;
        tierMultipliers[SBTTier.PLATINUM] = 200;
    }

    // Settlement functions will be implemented below
}
```

### Settlement Logic

```python
class SettlerEngine:
    def __init__(self, web3_provider, contract_address, oracle_private_key):
        self.w3 = web3_provider
        self.contract = self.w3.eth.contract(
            address=contract_address,
            abi=self.get_contract_abi()
        )
        self.oracle_account = self.w3.eth.account.from_key(oracle_private_key)
        self.cache = SettlementCache()

    def settle_contribution(self, batch_result):
        """
        Settle a contribution batch based on L2 audit results

        Args:
            batch_result: Batch validation result from L2/L3

        Returns:
            Settlement transaction receipt
        """
        # Calculate total SBT changes
        total_sbt_change = self.calculate_sbt_changes(batch_result)

        # Prepare settlement data
        settlement_data = {
            'batchId': batch_result.batch_id,
            'timestamp': int(time.time()),
            'results': batch_result.audit_results,
            'totalChange': total_sbt_change,
            'multiplier': self.get_tier_multiplier(batch_result.participants)
        }

        # Create settlement transaction
        tx = self.contract.functions.settleBatch(
            settlement_data['batchId'],
            settlement_data['results'],
            settlement_data['totalChange']
        ).build_transaction({
            'from': self.oracle_account.address,
            'gas': 500000,
            'gasPrice': self.w3.eth.gas_price,
            'nonce': self.w3.eth.get_transaction_count(self.oracle_account.address)
        })

        # Sign and send transaction
        signed_tx = self.w3.eth.account.sign_transaction(
            tx, self.oracle_account.key
        )

        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

        # Update local cache
        self.cache.update_settlement_status(
            batch_result.batch_id,
            receipt.status,
            total_sbt_change
        )

        return receipt

    def calculate_sbt_changes(self, batch_result):
        """
        Calculate total SBT balance changes for the batch

        Args:
            batch_result: Results from L2 audit

        Returns:
            Dictionary of participant -> SBT change
        """
        sbt_changes = {}

        for audit_result in batch_result.audit_results:
            participant = audit_result.participant_address
            base_change = self.get_base_sbt_change(audit_result.classification)

            # Apply tier multiplier
            current_tier = self.get_participant_tier(participant)
            multiplier = self.get_tier_multiplier_value(current_tier)

            final_change = base_change * multiplier // 100
            sbt_changes[participant] = final_change

        return sbt_changes

    def get_base_sbt_change(self, classification):
        """Get base SBT change based on classification"""
        changes = {
            'HONEST': 1,
            'RARITY': 10,
            'NOISE': 0,
            'FRAUD': -50
        }
        return changes.get(classification, 0)
```

### SBT Tier Management

```python
class SBTTierManager:
    def __init__(self, settlement_engine):
        self.engine = settlement_engine
        self.tier_thresholds = {
            'BRONZE': 0,
            'SILVER': 100,
            'GOLD': 500,
            'PLATINUM': 1000
        }

    def update_tier(self, participant_address, new_sbt_balance):
        """
        Update participant's tier based on SBT balance

        Args:
            participant_address: Participant's wallet address
            new_sbt_balance: New SBT balance after settlement

        Returns:
            Tier change information
        """
        current_tier = self.engine.get_participant_tier(participant_address)
        new_tier = self.calculate_tier(new_sbt_balance)

        if new_tier != current_tier:
            # Emit tier upgrade event
            self.emit_tier_upgrade(participant_address, current_tier, new_tier)

            # Apply any tier-specific bonuses
            self.apply_tier_bonuses(participant_address, new_tier)

            return {
                'upgraded': True,
                'old_tier': current_tier,
                'new_tier': new_tier,
                'bonus_applied': self.get_tier_bonuses(new_tier)
            }

        return {'upgraded': False, 'current_tier': current_tier}

    def calculate_tier(self, sbt_balance):
        """Calculate tier based on SBT balance"""
        if sbt_balance >= self.tier_thresholds['PLATINUM']:
            return 'PLATINUM'
        elif sbt_balance >= self.tier_thresholds['GOLD']:
            return 'GOLD'
        elif sbt_balance >= self.tier_thresholds['SILVER']:
            return 'SILVER'
        else:
            return 'BRONZE'
```

### Dashboard API

```python
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="FLPG Settlement Dashboard API")
security = HTTPBearer()

class ParticipantStats(BaseModel):
    address: str
    sbt_balance: int
    tier: str
    total_contributions: int
    fraud_count: int
    rarity_count: int
    last_activity: str
    active: bool

class SettlementHistory(BaseModel):
    batch_id: str
    timestamp: str
    participant: str
    classification: str
    sbt_change: int
    tx_hash: str

class DashboardStats(BaseModel):
    total_participants: int
    active_participants: int
    total_sbt_distributed: int
    fraud_detections_24h: int
    rarity_discoveries_24h: int
    average_tier: str

@app.get("/api/dashboard/stats", response_model=DashboardStats)
async def get_dashboard_stats():
    """
    Get overall dashboard statistics
    """
    try:
        stats = await settlement_engine.get_dashboard_statistics()
        return DashboardStats(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/participants/{address}", response_model=ParticipantStats)
async def get_participant_stats(address: str):
    """
    Get detailed statistics for a specific participant
    """
    try:
        participant = await settlement_engine.get_participant(address)
        return ParticipantStats(**participant)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Participant not found: {e}")

@app.get("/api/participants/{address}/history", response_model=List[SettlementHistory])
async def get_participant_history(
    address: str,
    limit: Optional[int] = 100,
    offset: Optional[int] = 0
):
    """
    Get settlement history for a specific participant
    """
    try:
        history = await settlement_engine.get_participant_history(
            address, limit, offset
        )
        return [SettlementHistory(**item) for item in history]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/leaderboard")
async def get_leaderboard(limit: Optional[int] = 50):
    """
    Get leaderboard of top participants by SBT balance
    """
    try:
        leaderboard = await settlement_engine.get_leaderboard(limit)
        return leaderboard
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Proof-query endpoints were part of an earlier design sketch.
# The current stack does not expose on-chain fraud-proof or
# rarity-certificate read APIs because those artifacts are not
# persisted by the deployed settlement contract.
```

### Economic Model Implementation

```python
class EconomicModelAnalyzer:
    def __init__(self, settlement_engine):
        self.engine = settlement_engine
        self.historical_data = []

    def analyze_incentive_alignment(self, time_window_days=30):
        """
        Analyze if incentives are properly aligned

        Args:
            time_window_days: Analysis period

        Returns:
            Economic analysis results
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=time_window_days)

        # Get historical settlement data
        settlements = self.engine.get_settlements_by_date_range(start_date, end_date)

        # Calculate utilities
        utilities = self.calculate_utilities(settlements)

        # Check incentive alignment
        honest_utility = utilities['honest_participants']
        malicious_utility = utilities['malicious_actors']
        rarity_utility = utilities['rarity_discoverers']

        analysis = {
            'period_days': time_window_days,
            'total_settlements': len(settlements),
            'honest_participant_utility': honest_utility,
            'malicious_actor_utility': malicious_utility,
            'rarity_discovery_utility': rarity_utility,
            'incentive_alignment': honest_utility > malicious_utility < 0,
            'recommendations': self.generate_recommendations(utilities)
        }

        return analysis

    def calculate_utilities(self, settlements):
        """Calculate expected utilities for different strategies"""
        utilities = {
            'honest_participants': 0,
            'malicious_actors': 0,
            'rarity_discoverers': 0
        }

        total_honest = 0
        total_malicious = 0
        total_rarity = 0

        for settlement in settlements:
            if settlement.classification == 'HONEST':
                utilities['honest_participants'] += settlement.sbt_change
                total_honest += 1
            elif settlement.classification == 'FRAUD':
                utilities['malicious_actors'] += settlement.sbt_change
                total_malicious += 1
            elif settlement.classification == 'RARITY':
                utilities['rarity_discovery_utility'] += settlement.sbt_change
                total_rarity += 1

        # Calculate averages
        if total_honest > 0:
            utilities['honest_participants'] /= total_honest
        if total_malicious > 0:
            utilities['malicious_actors'] /= total_malicious
        if total_rarity > 0:
            utilities['rarity_discovery_utility'] /= total_rarity

        return utilities
```

### Monitoring & Alerts

```python
class SettlementMonitor:
    def __init__(self, settlement_engine):
        self.engine = settlement_engine
        self.alert_thresholds = {
            'fraud_rate': 0.1,      # 10% fraud rate triggers alert
            'settlement_failure_rate': 0.05,  # 5% settlement failure
            'gas_price_spike': 100,  # 100% increase in gas price
            'inactive_rate': 0.8     # 80% participants inactive
        }

    async def monitor_settlement_health(self):
        """Monitor settlement system health"""
        while True:
            try:
                # Get current metrics
                metrics = await self.get_current_metrics()

                # Check for alerts
                alerts = self.check_alert_conditions(metrics)

                # Send alerts if any
                for alert in alerts:
                    await self.send_alert(alert)

                # Sleep for monitoring interval
                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logging.error(f"Monitoring error: {e}")
                await asyncio.sleep(60)

    def check_alert_conditions(self, metrics):
        """Check if any alert conditions are met"""
        alerts = []

        # Check fraud rate
        if metrics['fraud_rate'] > self.alert_thresholds['fraud_rate']:
            alerts.append({
                'type': 'HIGH_FRAUD_RATE',
                'severity': 'HIGH',
                'message': f"Fraud rate {metrics['fraud_rate']:.2%} exceeds threshold",
                'data': metrics
            })

        # Check settlement failures
        if metrics['settlement_failure_rate'] > self.alert_thresholds['settlement_failure_rate']:
            alerts.append({
                'type': 'SETTLEMENT_FAILURES',
                'severity': 'MEDIUM',
                'message': f"Settlement failure rate {metrics['settlement_failure_rate']:.2%} exceeds threshold",
                'data': metrics
            })

        return alerts
```

## Integration Points

### Input from L3
- Receives validation results from Gatekeeper
- Processes approval/rejection decisions
- Handles rollback notifications

### Output to Blockchain
- Writes settlement transactions to smart contract
- Stores round-scoped settlement identifiers and categorized vehicle lists
- Maintains immutable audit trail

### Frontend Dashboard
- Provides REST API for dashboard
- Supplies real-time statistics
- Delivers participant analytics

### Economic Monitoring
- Tracks incentive alignment
- Analyzes system economics
- Generates policy recommendations

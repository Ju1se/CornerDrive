# L2: Dual-Purpose Audit "The Sniper"

## Purpose

L2 implements a sophisticated dual-purpose audit system that can simultaneously detect malicious updates (fraud detection) and identify valuable rare information (rarity discovery). This creates incentives for both honest participation and contribution of specialized knowledge.

## Core Concept: Dual-Purpose Evaluation

The system evaluates each gradient update against two different datasets:

1. **Main Dataset**: Primary task performance
2. **Corner Dataset**: Rare/specialized scenarios

### Mathematical Framework

```python
def dual_audit(gradient, main_loss, corner_loss, current_weights, learning_rate):
    """
    Perform dual-purpose audit on gradient update

    Args:
        gradient: Gradient tensor to audit
        main_loss: Loss function for main dataset
        corner_loss: Loss function for corner cases
        current_weights: Current model weights
        learning_rate: Learning rate η

    Returns:
        AuditResult with classification and scores
    """
    # Compute weight updates
    weight_update = -learning_rate * gradient
    candidate_weights = current_weights + weight_update

    # Calculate performance changes
    main_performance_old = main_loss(current_weights)
    main_performance_new = main_loss(candidate_weights)

    corner_performance_old = corner_loss(current_weights)
    corner_performance_new = corner_loss(candidate_weights)

    # Calculate deltas
    delta_main = main_performance_new - main_performance_old
    delta_corner = corner_performance_new - corner_performance_old

    # Classification logic
    if delta_main > FRAUD_THRESHOLD:
        classification = "FRAUD"
        action = "SLASH_STAKE"
        sbt_change = -50
    elif delta_corner <= RARITY_THRESHOLD and delta_main <= RARITY_MAIN_THRESHOLD:
        classification = "RARITY"
        action = "JACKPOT_REWARD"
        sbt_change = 10
    elif delta_main < 0:
        classification = "HONEST"
        action = "INCLUDE_AND_REWARD"
        sbt_change = 1
    else:
        classification = "NOISE"
        action = "DISCARD"
        sbt_change = 0

    return AuditResult(
        classification=classification,
        action=action,
        delta_main=delta_main,
        delta_corner=delta_corner,
        sbt_change=sbt_change,
        proof=generate_proof(gradient, delta_main, delta_corner)
    )
```

## Classification Rules

| Mathematical Trigger | Diagnosis | Action | SBT Points |
|---------------------|-----------|--------|------------|
| ΔL_main > θ_tol (Utility Violation) | ✗ FRAUD | SLASH STAKE | -50 |
| ΔL_corner ≤ θ_rare and ΔL_main ≤ θ_rarity-main (corner information gain within the stricter clean-rarity safety band) | ✓ RARITY | JACKPOT | +10 |
| ΔL_main < 0 (Helps Main) | ✓ HONEST | INCLUDE | +1 |
| Otherwise (Negligible Impact) | ~ NOISE | DISCARD | 0 |

Current boundary note: V4.1 separates the strong fraud threshold `θ_tol` from
the stricter clean-rarity main safety band `θ_rarity-main`. Updates that improve
corner loss but introduce positive main-task drift above this stricter band are
treated as conflict/noise rather than clean rarity.

## Configuration Parameters

```python
class L2Config:
    FRAUD_THRESHOLD = 0.05        # θ_tol - tolerance for main task
    RARITY_THRESHOLD = -0.03      # θ_rare - threshold for rare discoveries
    RARITY_MAIN_THRESHOLD = 0.005 # θ_rarity-main - strict main safety for RARITY
    LEARNING_RATE = 0.01          # η - learning rate
    AUDIT_QUEUE = "l2_audit_queue"  # Celery queue for direct L1 -> L2 dispatch
    CONCURRENT_AUDITS = 5         # Number of parallel audit workers
```

## Worker Architecture

### Celery Worker Setup

```python
from celery import Celery
from common.config import *

app = Celery('l2_dual_audit')
app.config_from_object('common.config')

# L1 sends suspect gradients directly into this Celery queue.

@app.task(bind=True, max_retries=3)
def audit_gradient_task(self, gradient_data, batch_id):
    """
    Celery task for auditing gradient updates
    """
    try:
        # Reconstruct gradient from received data
        gradient = reconstruct_gradient(gradient_data)

        # Perform dual audit
        audit_result = dual_audit(
            gradient=gradient,
            main_loss=load_main_loss(),
            corner_loss=load_corner_loss(),
            current_weights=load_current_weights(),
            learning_rate=L2_LEARNING_RATE
        )

        # Store audit result
        store_audit_result(batch_id, audit_result)

        return {
            "status": "completed",
            "batch_id": batch_id,
            "classification": audit_result.classification,
            "sbt_change": audit_result.sbt_change
        }

    except Exception as exc:
        # Retry logic
        if self.request.retries < self.max_retries:
            raise self.retry(countdown=60)
        else:
            # Log failure and notify L4
            notify_audit_failure(batch_id, str(exc))
            raise
```

## Proof Generation

### Fraud Proof

```python
def generate_fraud_proof(gradient, delta_main, evidence_data):
    """
    Generate cryptographic proof of fraud detection

    Args:
        gradient: Malicious gradient
        delta_main: Performance degradation
        evidence_data: Supporting evidence

    Returns:
        Fraud proof object
    """
    proof = {
        "type": "FRAUD_PROOF",
        "timestamp": datetime.utcnow().isoformat(),
        "gradient_hash": hash_tensor(gradient),
        "delta_main": delta_main,
        "evidence": evidence_data,
        "validator_signature": sign_validator_data(proof_data)
    }

    return FraudProof(**proof)
```

### Rarity Certificate

```python
def generate_rarity_certificate(gradient, delta_corner, discovered_cases):
    """
    Generate certificate for rare information discovery

    Args:
        gradient: Gradient with rare knowledge
        delta_corner: Performance improvement on corner cases
        discovered_cases: List of newly discovered corner cases

    Returns:
        Rarity certificate object
    """
    certificate = {
        "type": "RARITY_CERTIFICATE",
        "timestamp": datetime.utcnow().isoformat(),
        "gradient_hash": hash_tensor(gradient),
        "delta_corner": delta_corner,
        "discovered_cases": discovered_cases,
        "validator_signature": sign_validator_data(cert_data)
    }

    return RarityCertificate(**certificate)
```

## Economic Model

### Incentive Alignment

The dual-purpose design creates strategic incentives:

1. **Honest Participants**: Contribute helpful gradients → +1 SBT
2. **Malicious Actors**: Submit harmful gradients → -50 SBT
3. **Knowledge Contributors**: Find rare cases → +10 SBT jackpot

### Expected Utility Analysis

```python
def expected_utility_analysis():
    """
    Analyze expected utility for different strategies

    Expected: E[U_honest] > E[U_malicious] < 0
    """
    # Honest participant utility
    p_success = 0.95  # High success rate for honest gradients
    u_honest = p_success * SBT_HONEST_REWARD + (1 - p_success) * 0

    # Malicious actor utility
    p_detection = 0.9   # High detection rate
    u_malicious = p_detection * SBT_FRAUD_PENALTY + (1 - p_detection) * 0

    # Knowledge discovery utility (low probability, high reward)
    p_rarity = 0.01     # 1% chance of discovering rare case
    u_rarity = p_rarity * SBT_RARITY_REWARD + (1 - p_rarity) * 0

    return {
        "honest_utility": u_honest,
        "malicious_utility": u_malicious,
        "rarity_utility": u_rarity,
        "incentive_alignment": u_honest > u_malicious < 0
    }
```

## Performance Optimization

### Parallel Auditing
- Multiple Celery workers for concurrent audits
- Queue prioritization for urgent cases
- Batch processing for efficiency

### Efficient Loss Computation
- Pre-computed dataset embeddings
- Cached loss calculations
- Approximate methods for large datasets

### Memory Management
- Gradient compression during audit
- Efficient tensor operations
- Memory pooling for frequent allocations

## Security & Privacy

### Audit Integrity
- Cryptographic proofs for all classifications
- Immutable audit logs
- Validator signatures for authenticity

### Gradient Privacy
- No gradient reconstruction during audit
- Secure storage of audit results
- Access controls for audit data

### Robustness
- Fault tolerance with retry mechanisms
- Graceful degradation under load
- Alert system for anomalous patterns

## Integration Points

- **L1**: Receives suspect gradients for detailed audit
- **L3**: Forwards classified gradients to global validation
- **L4**: Reports audit results and SBT changes
- **Storage**: Secure audit log storage

# L3: Global Validation "Gatekeeper"

## Purpose

L3 serves as the gatekeeper layer that performs final validation on gradients that have passed through L1 screening and L2 audit. It ensures global model stability and prevents harmful updates from being incorporated into the federated model.

Current repository note:
- L3 exists as a library, not a standalone deployed service.
- The running stack exposes its live dataset source through `GET /api/v1/l3/status` on the L4 dashboard service.
- If no supported artifacts are found under `L3_GOLDEN_DATASET_PATH`, the implementation now reports an explicit placeholder fallback instead of silently implying a real golden dataset.

## Core Functionality

### Golden Dataset Validation

The gatekeeper maintains a trusted "golden dataset" that represents high-quality, diverse examples from the IoV domain.

```python
class GoldenDatasetValidator:
    def __init__(self, golden_dataset_path):
        self.golden_data = self.load_golden_dataset(golden_dataset_path)
        self.validation_metrics = {
            "accuracy": [],
            "loss": [],
            "drift": []
        }

    def validate_model_update(self, current_weights, candidate_weights):
        """
        Validate model update using golden dataset

        Args:
            current_weights: Current global model weights
            candidate_weights: Updated weights after gradient application

        Returns:
            ValidationResult with drift score and approval decision
        """
        # Calculate performance on golden dataset
        current_performance = self.evaluate_model(current_weights)
        candidate_performance = self.evaluate_model(candidate_weights)

        # Compute drift score
        drift_score = candidate_performance.loss - current_performance.loss

        # Determine approval based on drift threshold
        approved = drift_score <= DRIFT_THRESHOLD

        return ValidationResult(
            drift_score=drift_score,
            current_loss=current_performance.loss,
            candidate_loss=candidate_performance.loss,
            approved=approved,
            validation_timestamp=datetime.utcnow()
        )

    def evaluate_model(self, weights):
        """
        Evaluate model on golden dataset

        Args:
            weights: Model weights to evaluate

        Returns:
            ModelPerformance metrics
        """
        model = load_model_with_weights(weights)

        total_loss = 0
        correct_predictions = 0
        total_samples = len(self.golden_data)

        for batch in self.golden_data:
            inputs, targets = batch
            predictions = model(inputs)

            loss = compute_loss(predictions, targets)
            accuracy = compute_accuracy(predictions, targets)

            total_loss += loss.item()
            correct_predictions += accuracy * len(targets)

        avg_loss = total_loss / len(self.golden_data)
        avg_accuracy = correct_predictions / total_samples

        return ModelPerformance(
            loss=avg_loss,
            accuracy=avg_accuracy
        )
```

### Drift Detection Algorithm

```python
def compute_drift_score(old_weights, new_weights, golden_dataset):
    """
    Compute drift score for model update

    Drift = L_gold(W_old - ηΔW_cand) - L_gold(W_old)

    Args:
        old_weights: Current global model weights
        new_weights: Candidate updated weights
        golden_dataset: Validation dataset

    Returns:
        Drift score (positive = performance degradation)
    """
    # Ensure both weight tensors have same shape
    weight_diff = new_weights - old_weights

    # Apply learning rate adjustment
    eta = L3_LEARNING_RATE if hasattr(L3_LEARNING_RATE, '__call__') else 0.01
    adjusted_diff = -eta * weight_diff

    # Create test weights for drift computation
    test_weights = old_weights + adjusted_diff

    # Compute loss on golden dataset
    old_loss = evaluate_loss(old_weights, golden_dataset)
    test_loss = evaluate_loss(test_weights, golden_dataset)

    drift_score = test_loss - old_loss

    return drift_score
```

## Configuration Parameters

```python
class L3Config:
    DRIFT_THRESHOLD = 0.05               # Maximum allowed performance drift
    GOLDEN_DATASET_PATH = "data/validation/golden"
    VALIDATION_BATCH_SIZE = 32
    MAX_VALIDATION_TIME = 60              # seconds
    MIN_UPDATE_FREQUENCY = 10             # minutes between updates
    EMERGENCY_ROLLBACK_THRESHOLD = 0.15   # Trigger rollback if drift exceeds
```

## Batch Validation Process

### Multi-Gradient Aggregation

```python
def validate_gradient_batch(candidate_gradients, current_weights):
    """
    Validate a batch of gradients before aggregation

    Args:
        candidate_gradients: List of validated gradients from L1/L2
        current_weights: Current global model weights

    Returns:
        BatchValidationResult
    """
    # Compute aggregated update
    aggregated_gradient = aggregate_gradients(candidate_gradients)

    # Apply gradient to get candidate weights
    learning_rate = L2_LEARNING_RATE
    candidate_weights = current_weights - learning_rate * aggregated_gradient

    # Validate with golden dataset
    validator = GoldenDatasetValidator(GOLDEN_DATASET_PATH)
    validation_result = validator.validate_model_update(
        current_weights, candidate_weights
    )

    # Additional safety checks
    safety_checks = perform_safety_checks(
        current_weights, candidate_weights, candidate_gradients
    )

    final_approval = (
        validation_result.approved and
        safety_checks.passed and
        check_update_frequency()
    )

    return BatchValidationResult(
        approved=final_approval,
        drift_score=validation_result drift_score,
        validation_loss=validation_result.candidate_loss,
        safety_checks=safety_checks,
        aggregated_gradient=aggregated_gradient
    )
```

### Safety Checks

```python
def perform_safety_checks(old_weights, new_weights, gradients):
    """
    Perform additional safety validations

    Args:
        old_weights: Current model weights
        new_weights: Candidate model weights
        gradients: Input gradients being aggregated

    Returns:
        SafetyCheckResult
    """
    checks = []

    # 1. Weight magnitude check
    weight_change = torch.norm(new_weights - old_weights)
    magnitude_ok = weight_change < MAX_WEIGHT_CHANGE_THRESHOLD
    checks.append(("weight_magnitude", magnitude_ok, weight_change))

    # 2. Gradient norm check
    for i, grad in enumerate(gradients):
        grad_norm = torch.norm(grad)
        grad_ok = grad_norm < MAX_GRADIENT_NORM
        checks.append((f"gradient_norm_{i}", grad_ok, grad_norm))

    # 3. Numerical stability check
    numerical_stable = check_numerical_stability(new_weights)
    checks.append(("numerical_stability", numerical_stable, None))

    # 4. Feature drift check (if applicable)
    feature_drift = check_feature_drift(old_weights, new_weights)
    feature_ok = feature_drift < MAX_FEATURE_DRIFT
    checks.append(("feature_drift", feature_ok, feature_drift))

    all_passed = all(check[1] for check in checks)

    return SafetyCheckResult(
        passed=all_passed,
        checks=checks
    )
```

## Emergency Procedures

### Rollback Mechanism

```python
def emergency_rollback(current_weights, safe_weights, rollback_reason):
    """
    Emergency rollback to previous safe model state

    Args:
        current_weights: Current potentially corrupted weights
        safe_weights: Last known good weights
        rollback_reason: Reason for rollback
    """
    # Log emergency event
    log_emergency_event(
        timestamp=datetime.utcnow(),
        reason=rollback_reason,
        drift_score=compute_drift_score(current_weights, safe_weights),
        severity="HIGH"
    )

    # Perform rollback
    restore_model_weights(safe_weights)

    # Notify L4 for settlement adjustment
    notify_settlement_layer(
        event_type="EMERGENCY_ROLLBACK",
        timestamp=datetime.utcnow(),
        reason=rollback_reason,
        affected_batches=get_affected_batches()
    )

    # Trigger additional validation
    trigger_enhanced_validation()
```

### Circuit Breaker Pattern

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=300):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def validate_with_circuit_breaker(self, validation_func, *args, **kwargs):
        """
        Execute validation with circuit breaker protection
        """
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise CircuitBreakerOpenException("Circuit breaker is OPEN")

        try:
            result = validation_func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result

        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"

            raise e
```

## Monitoring & Metrics

### Key Performance Indicators

```python
class GatekeeperMetrics:
    def __init__(self):
        self.metrics = {
            "validation_success_rate": [],
            "average_drift_score": [],
            "validation_latency": [],
            "rollback_frequency": [],
            "throughput": []
        }

    def record_validation(self, validation_result, processing_time):
        """Record validation metrics"""
        self.metrics["validation_success_rate"].append(
            1.0 if validation_result.approved else 0.0
        )
        self.metrics["average_drift_score"].append(
            validation_result.drift_score
        )
        self.metrics["validation_latency"].append(processing_time)

    def get_health_status(self):
        """Get overall system health status"""
        recent_validations = self.metrics["validation_success_rate"][-100:]
        success_rate = sum(recent_validations) / len(recent_validations) if recent_validations else 1.0

        if success_rate >= 0.95:
            return "HEALTHY"
        elif success_rate >= 0.8:
            return "WARNING"
        else:
            return "CRITICAL"
```

## Integration Points

### Input from L2
- Receives validated gradients from dual-purpose audit
- Processes audit results and classifications
- Consumes classified updates before approve/reject validation

### Output to L4
- Sends approval/rejection decisions for settlement
- Provides validation decisions and drift measurements to downstream settlement logic
- Reports rollback events and system health

### Data Storage
- Maintains golden dataset
- Stores validation history
- Logs audit trails

### Monitoring
- Real-time health monitoring
- Performance metrics collection
- Alert generation for anomalies

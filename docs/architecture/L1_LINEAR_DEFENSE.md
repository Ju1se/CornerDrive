# L1: Linear Defense & Visibility Routing

## Purpose

L1 is a low-cost visibility router. It increases L2 audit coverage for suspicious
or under-observed updates, but it does not assign final Fraud/Rarity/Noise
verdicts, reject clients, slash clients, or settle rewards. L2 owns the
evidence-backed verdict; L4 turns verdicts into reputation and settlement.

The default mode remains the V2.5 cosine router for reproducible thesis
baselines. CornerDrive-L1V3 adds a budgeted multi-signal router that can be
enabled explicitly.

## Core Algorithms

### 1. Geometric Median Computation

The geometric median is robust against outliers and provides a more stable aggregation than simple averaging.

```python
def geometric_median(gradients, max_iter=100, eps=1e-6):
    """
    Compute geometric median of gradients

    Args:
        gradients: List of gradient tensors
        max_iter: Maximum iterations for Weiszfeld's algorithm
        eps: Convergence threshold

    Returns:
        Geometric median tensor
    """
    median = gradients[0]
    for _ in range(max_iter):
        weights = [1.0 / max(torch.norm(g - median), eps) for g in gradients]
        weight_sum = sum(weights)
        new_median = sum(w * g for w, g in zip(weights, gradients)) / weight_sum

        if torch.norm(new_median - median) < eps:
            break
        median = new_median

    return median
```

### 2. V2.5 Cosine Deviation Scoring

Identifies gradients that deviate significantly from the computed median.

```python
def cosine_deviation(gradient, median):
    """
    Compute cosine deviation score

    Args:
        gradient: Individual gradient tensor
        median: Geometric median gradient

    Returns:
        Cosine deviation score (0-1, higher = more deviant)
    """
    dot_product = torch.dot(gradient.flatten(), median.flatten())
    grad_norm = torch.norm(gradient)
    median_norm = torch.norm(median)

    cosine_sim = dot_product / (grad_norm * median_norm + 1e-8)
    deviation = 1 - cosine_sim

    return max(0, deviation)
```

### 3. CornerDrive-L1V3: Budgeted Multi-Signal Visibility Router

L1V3 computes a cheap per-update risk score:

```text
risk_i = w_cos  * R(d_cos_i)
       + w_norm * R(z_norm_i)
       + w_sign * R(d_sign_i)
       + w_rep  * R(rep_risk_i)
       + w_age  * R(audit_age_i)
```

Where `R(.)` is within-round percentile-rank normalization. This keeps cosine,
norm, sign, reputation, and audit-age features on comparable scales.

Signals:

- `d_cos_i`: `1 - cosine_similarity(g_i, g_med)`
- `z_norm_i`: robust MAD score of log update norm
- `d_sign_i`: sign disagreement on top-k reference coordinates
- `rep_risk_i`: low-reputation or recent-fraud risk
- `audit_age_i`: normalized rounds since last L2 audit

Available modes:

```text
v25_cosine_fixed       # M0: original cosine + fixed recheck
v3_m1_norm_fixed      # M1: cosine + norm + fixed recheck
v3_m2_norm_sign_fixed # M2: cosine + norm + sign + fixed recheck
v3_m3_budgeted        # M3: top-B risk + stratified random recheck
v3_m4_reputation_age  # M4: full L1V3 with reputation and audit age
```

For budgeted modes, L1 routes:

```text
threshold hits + top-B risk updates + stratified random recheck
```

The output is only:

```text
route_to_l2 = true/false
routing_reason = cosine_screening | norm_mad_screening | sign_screening
               | risk_topB | stratified_random | probabilistic_recheck | bypass
```

## Batch Processing

### Configuration Parameters

```python
class L1Config:
    BATCH_SIZE = 10              # Number of gradients per batch
    BATCH_TIMEOUT = 5.0          # Timeout in seconds
    SUSPECT_THRESHOLD = 0.3      # Legacy cosine-deviation threshold
    L1_ROUTER_MODE = "v25_cosine_fixed"
    L1_QUEUE_BUDGET_RATIO = 0.35
    L1_RANDOM_RECHECK_RATIO = 0.05
    GEOMETRIC_MEDIAN_MAX_ITER = 100
    GEOMETRIC_MEDIAN_EPS = 1e-6
```

### Processing Flow

1. **Collect Batch**: Wait for `BATCH_SIZE` gradients or timeout
2. **Compute Median**: Calculate geometric median of batch
3. **Score Gradients**: Compute cosine deviation, and optionally norm/sign/history scores
4. **Route**: Select updates for L2 under the configured visibility policy
5. **Forward**: Send routed updates to L2 and aggregate the non-routed path

## API Endpoints

### POST /gradient
Submit a gradient update to L1 for screening.

**Request:**
```json
{
  "gradient": "base64_encoded_tensor",
  "client_id": "vehicle_123",
  "timestamp": "2024-01-01T00:00:00Z",
  "signature": "digital_signature"
}
```

**Response:**
```json
{
  "status": "accepted|rejected|suspect",
  "score": 0.15,
  "batch_id": "batch_123",
  "message": "Gradient processed successfully"
}
```

### GET /batch/{batch_id}
Retrieve batch processing status.

### GET /health
Health check endpoint.

## Performance Optimizations

### 1. Efficient Gradient Storage
- Use numpy arrays instead of Python lists
- Implement gradient compression techniques
- Cache frequently accessed gradients

### 2. Parallel Processing
- Use multi-threading for median computation
- Batch multiple gradient operations
- Implement vectorized cosine similarity

### 3. Memory Management
- Implement gradient eviction policies
- Use streaming for large gradients
- Monitor memory usage and optimize

## Security Considerations

### 1. Input Validation
- Validate gradient dimensions and ranges
- Check digital signatures
- Rate limiting per client

### 2. Privacy Protection
- No gradient reconstruction from scores
- Secure storage of sensitive data
- Access controls for audit logs

### 3. Robustness
- Handle malformed inputs gracefully
- Implement retry mechanisms
- Fallback strategies for failures

## Monitoring & Metrics

### Key Metrics
- **Processing Latency**: Time per batch
- **Throughput**: Gradients processed per second
- **Suspect Rate**: Percentage of gradients flagged
- **Routing Reason Mix**: Why updates entered L2
- **Risk Score Distribution**: L1V3 risk telemetry by archetype
- **Audit Age**: Time since each client last received L2 visibility
- **Median Convergence**: Iterations needed

### Alerts
- High suspect rate (> 20%)
- Processing delays (> timeout)
- Memory usage warnings
- Failed batch processing

## Integration Points

- **L0**: Receives pre-processed gradients from vehicles
- **L2**: Forwards suspect gradients for detailed audit
- **L3**: Sends non-suspect gradients for global validation
- **L4**: Reports batch processing results for settlement

# FLPG System Architecture

## Overview

The Federated Learning Privacy Game (FLPG) implements a 5-layer security architecture for protecting federated learning systems in Internet of Vehicles (IoV) environments.

## Architecture Layers

### L0: Client Compliance (Vehicle Side - Out of Scope)
- **Norm Clipping**: `g_clip = g / max(1, ||g||₂/C)`
- **Local Differential Privacy**: `g̃ = g_clip + N(0, σ²I)`
- **Top-k Sparsification**: `ĝ = Φ_k(g̃, k%)`
- **Digital Signature** for authentication

### L1: Linear Defense & Screening
- **Geometric Median**: `w* = argmin_w Σ||w - ĝᵢ||₂`
- **Cosine Deviation**: `Score_i = 1 - (ĝᵢ·w*)/(||ĝᵢ||₂·||w*||₂)`
- **Batch Processing** with configurable thresholds
- **Suspect List Generation** for L2

### L2: Dual-Purpose Audit "The Sniper"
- **Main Task Impact**: `ΔL_main = L(W-ηg; D_main) - L(W; D_main)`
- **Corner Case Impact**: `ΔL_corner = L(W-ηg; D_corner) - L(W; D_corner)`
- **Classification Rules**:
  - FRAUD: `ΔL_main > θ_tol` (Utility Violation)
  - RARITY: `ΔL_corner ≤ θ_rare` and `ΔL_main ≤ θ_tol` (corner information gain within the main-task damage budget)
  - NOISE: Negligible Impact
- **Fraud Proof & Rarity Certificate** generation

### L3: Global Validation "Gatekeeper"
- **Golden Dataset Validation**
- **Drift Detection**: `Drift = L_gold(W_old - ηΔW_cand) - L_gold(W_old)`
- **Approve/Reject Decision** based on drift threshold
- **Commit Signal** to L4 for settlement

### L4: On-Chain Settlement
- **Smart Contract** with optimistic settlement
- **SBT Credit System**: Bronze/Silver/Gold/Platinum tiers
- **Incentive Alignment**: `E[U_honest] > E[U_malicious]`
- **Round-Scoped Settlement IDs** and categorized vehicle lists for replay-safe on-chain settlement

## Security Guarantees

1. **Economic Incentives**: Honest participants earn more than malicious ones
2. **Privacy Protection**: Local DP and sparsification at L0
3. **Robustness**: Geometric median and dual-purpose audit
4. **Accountability**: On-chain settlement and proof storage

## Performance Considerations

- **Batch Processing** for efficient L1 operations
- **Async Task Queue** for L2 audits
- **Caching** for frequently accessed data
- **Parallel Processing** where applicable

## Deployment Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Vehicles      │───▶│   L1 Server     │───▶│   L2 Workers    │
│   (L0)          │    │   (Screening)   │    │   (Audit)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Dashboard     │◀───│   L4 API        │◀───│   L3 Validator  │
│   (Frontend)    │    │   (Settlement)  │    │   (Gatekeeper)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

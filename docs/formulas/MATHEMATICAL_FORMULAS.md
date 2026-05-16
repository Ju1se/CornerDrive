# FLPG Mathematical Formulas

This document contains all the mathematical formulas used in the FLPG system, organized by layer.

## L0: Client Compliance (Vehicle Side - Reference Only)

### 1. Gradient Norm Clipping

**Purpose**: Limit gradient magnitude to prevent outliers.

**Formula**:
```
g_clip = g / max(1, ||g||₂/C)
```

**Variables**:
- `g`: Original gradient tensor
- `g_clip`: Clipped gradient tensor
- `||·||₂`: L2 norm (Euclidean norm)
- `C`: Clipping threshold (hyperparameter)

### 2. Local Differential Privacy

**Purpose**: Add noise to protect individual privacy.

**Formula**:
```
g̃ = g_clip + N(0, σ²I)
```

**Variables**:
- `g̃`: Noisy gradient
- `N(0, σ²I)`: Gaussian noise with mean 0, variance σ²
- `I`: Identity matrix
- `σ`: Noise scale (privacy parameter)

### 3. Top-k Sparsification

**Purpose**: Keep only the k largest gradient components.

**Formula**:
```
ĝ = Φ_k(g̃, k%)
```

**Variables**:
- `ĝ`: Sparsified gradient
- `Φ_k`: Top-k selection operator
- `k%`: Percentage of components to keep

## L1: Linear Defense & Screening

### 1. Geometric Median

**Purpose**: Robust aggregation of gradients using Weiszfeld's algorithm.

**Formula**:
```
w* = argmin_w Σ_{i=1}^n ||w - ĝᵢ||₂
```

**Algorithm**:
```
Initialize: w_0 = ĝ₁
Iterate: w_{t+1} = Σ_{i=1}^n (ĝᵢ / ||w_t - ĝᵢ||₂) / Σ_{j=1}^n (1 / ||w_t - ĝⱼ||₂)
Converge: ||w_{t+1} - w_t||₂ < ε
```

**Variables**:
- `w*`: Geometric median
- `ĝᵢ`: i-th sparsified gradient
- `n`: Number of gradients
- `ε`: Convergence threshold

### 2. Cosine Deviation Score

**Purpose**: Measure angular deviation from median.

**Formula**:
```
Score_i = 1 - (ĝᵢ · w*) / (||ĝᵢ||₂ · ||w*||₂)
```

**Variables**:
- `Score_i`: Deviation score for gradient i (0-1, higher = more deviant)
- `ĝᵢ · w*`: Dot product between gradient and median
- `||·||₂`: L2 norm

## L2: Dual-Purpose Audit

### 1. Main Task Impact

**Purpose**: Measure impact on primary task performance.

**Formula**:
```
ΔL_main = L(W - ηg; D_main) - L(W; D_main)
```

**Variables**:
- `ΔL_main`: Change in main task loss
- `L(·)`: Loss function
- `W`: Current model weights
- `g`: Gradient to audit
- `η`: Learning rate
- `D_main`: Main training dataset

### 2. Corner Case Impact

**Purpose**: Measure impact on rare/important cases.

**Formula**:
```
ΔL_corner = L(W - ηg; D_corner) - L(W; D_corner)
```

**Variables**:
- `ΔL_corner`: Change in corner case loss
- `D_corner`: Corner case dataset

### 3. Classification Rules

**Fraud Detection**:
```
IF ΔL_main > θ_tol THEN classify as FRAUD
```

**Rarity Discovery**:
```
IF (ΔL_corner ≤ θ_rare) AND (ΔL_main ≤ θ_rarity_main_tol) THEN classify as RARITY
```

**Honest Contribution**:
```
IF ΔL_main < 0 THEN classify as HONEST
```

**Boundary Note**:
This is the current V4.1 implementation rule. Clean rarity uses the stricter
main-task safety band `θ_rarity_main_tol`, while `θ_tol` remains the stronger
fraud threshold. Main-helpful but corner-harmful updates are handled by the
explicit corner-harm guard before being accepted as HONEST.

**Noise**:
```
IF (θ_rarity_main_tol < ΔL_main ≤ θ_tol AND ΔL_corner ≤ θ_rare)
OR (θ_tol ≥ ΔL_main ≥ 0 AND ΔL_corner > θ_rare)
THEN classify as NOISE
```

## L3: Global Validation

### 1. Golden Dataset Drift

**Purpose**: Measure model performance drift on trusted data.

**Formula**:
```
Drift = L_gold(W_old - ηΔW_cand) - L_gold(W_old)
```

**Variables**:
- `Drift`: Performance drift score
- `L_gold`: Loss on golden dataset
- `W_old`: Current global weights
- `ΔW_cand`: Candidate weight update
- `η`: Learning rate

### 2. Weight Update Validation

**Purpose**: Test candidate weights before committing.

**Formula**:
```
W_candidate = W_current - η · Σ_{i ∈ approved} g_i / |approved|
```

**Variables**:
- `W_candidate`: Candidate updated weights
- `W_current`: Current weights
- `g_i`: i-th approved gradient
- `approved`: Set of approved gradients

### 3. Statistical Validation

**Purpose**: Statistical test for model performance change.

**Formula** (t-test for performance difference):
```
t = (μ_old - μ_new) / √(σ²_old / n_old + σ²_new / n_new)
```

**Variables**:
- `t`: t-statistic
- `μ_old, μ_new`: Mean performance (old vs new)
- `σ²_old, σ²_new`: Performance variance
- `n_old, n_new`: Sample sizes

## L4: Economic Model

### 1. Expected Utility Analysis

**Purpose**: Ensure honest participation is more profitable than malicious behavior.

**Formula**:
```
E[U_honest] > E[U_malicious] < 0
```

**Honest Participant Expected Utility**:
```
E[U_honest] = p_honest · R_honest + (1 - p_honest) · 0
```

**Malicious Actor Expected Utility**:
```
E[U_malicious] = p_detection · P_fraud + (1 - p_detection) · R_malicious
```

**Variables**:
- `p_honest`: Probability of honest contribution being accepted
- `R_honest`: Reward for honest contribution (+1 SBT)
- `p_detection`: Probability of fraud detection
- `P_fraud`: Penalty for fraud (-50 SBT)
- `R_malicious`: Reward for successful malicious update

### 2. Rarity Discovery Expected Utility

**Formula**:
```
E[U_rarity] = p_rarity · R_rarity + (1 - p_rarity) · U_honest
```

**Variables**:
- `p_rarity`: Probability of discovering rare case
- `R_rarity`: Reward for rarity discovery (+10 SBT)
- `U_honest`: Baseline honest utility

### 3. Tier Multiplier System

**Formula**:
```
Final_Reward = Base_Reward × Tier_Multiplier
```

**Tier Multipliers**:
- Bronze: 1.0x (100 basis points)
- Silver: 1.2x (120 basis points)
- Gold: 1.5x (150 basis points)
- Platinum: 2.0x (200 basis points)

## System Performance Metrics

### 1. Convergence Rate

**Formula**:
```
Convergence_Rate = ||W_t - W_optimal||₂ / ||W_0 - W_optimal||₂
```

### 2. Throughput

**Formula**:
```
Throughput = N_processed / T_elapsed
```

### 3. Detection Rate

**Formula**:
```
Detection_Rate = N_detected / N_malicious
```

### 4. False Positive Rate

**Formula**:
```
FPR = N_false_positives / N_honest
```

## Privacy Guarantees

### 1. Differential Privacy Budget

**Formula**:
```
ε_total = Σ_{t=1}^T ε_t
```

**Where ε_t is the privacy cost at iteration t**

### 2. Sensitivity Analysis

**Formula**:
```
Δf = max_{D,D': |D∩D'| ≤ 1} ||f(D) - f(D')||₁
```

## Optimization Theory

### 1. Convergence Conditions

**Geometric Median Convergence**:
- **Convexity**: Objective function is convex
- **Lipschitz Continuity**: ∇f is Lipschitz continuous with constant L
- **Step Size**: Learning rate η < 1/L

### 2. Incentive Compatibility

**Individual Rationality (IR)**:
```
E[U_participate] ≥ E[U_not_participate] = 0
```

**Incentive Compatibility (IC)**:
```
E[U_honest] ≥ E[U_malicious] AND E[U_honest] ≥ E[U_random]
```

## Information Theory

### 1. Mutual Information

**Purpose**: Measure information gain in rarity discovery.

**Formula**:
```
I(G; D_corner) = H(G) - H(G|D_corner)
```

### 2. Entropy Measures

**Purpose**: Quantify uncertainty in model updates.

**Formula**:
```
H(G) = -Σ p(g) log p(g)
```

## Risk Management

### 1. Value at Risk (VaR)

**Purpose**: Measure potential losses in federated learning.

**Formula**:
```
VaR_α = inf{x : P(Loss ≤ x) ≥ α}
```

### 2. Expected Shortfall

**Purpose**: Expected loss beyond VaR threshold.

**Formula**:
```
ES_α = E[Loss | Loss > VaR_α]
```

## Game Theory

### 1. Nash Equilibrium Conditions

**Purpose**: Find stable strategy profiles.

**Formula**:
```
u_i(s_i*, s_{-i}*) ≥ u_i(s_i, s_{-i}*) ∀s_i
```

### 2. Mechanism Design

**Purpose**: Design incentive-compatible mechanisms.

**Formula**:
```
Social_Welfare = Σ_i v_i - c_i - p_i
```

## References

1. **Robust Aggregation**: P. M. D. O. Geometric Median (Weiszfeld, 1937)
2. **Differential Privacy**: Dwork et al. (2006)
3. **Game Theory in FL**: Zhang et al. (2020)
4. **Federated Learning Convergence**: Li et al. (2020)
5. **Mechanism Design**: Myerson (1981)

---

**Note**: This document is continuously updated as the system evolves. All formulas should be empirically validated during system deployment.

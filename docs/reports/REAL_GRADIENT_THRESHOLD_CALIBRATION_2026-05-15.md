# Real-Gradient Threshold Calibration

Date: 2026-05-15

## Question

The held-out real-gradient benchmark increased CornerDrive fraud survival, especially on
MNIST and FashionMNIST. The calibration question was whether this was mainly caused
by the L2 fraud threshold (`theta_tol`) or by L1 visibility.

## Protocol

The sweep reused the leakage-safe real-gradient benchmark protocol:

- sources: MNIST, FashionMNIST, FEMNIST;
- calibration seeds: `20260507`, `20260508`;
- holdout seed: `20260509`;
- scale: 120 clients, 20 clients per round, 10 rounds, 48 samples per client;
- fixed L2 rarity threshold: `theta_rare = -0.005`;
- reported metrics are macro means over source/seed runs.

The sweep is a calibration/sensitivity study. It should be reported separately from the
main benchmark unless the thesis explicitly states that the real-gradient profile was
recalibrated after the held-out audit.

## Results

| Profile | Runs | Fraud survival | Rarity retention | Corner acc. | Main acc. | L1 review |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current | 9 | 0.3444 | 0.5763 | 0.6611 | 0.4721 | 0.7428 |
| L2 strict only | 9 | 0.3356 | 0.5183 | 0.6405 | 0.4660 | 0.7411 |
| Recheck 0.50 | 9 | 0.2356 | 0.5383 | 0.6974 | 0.4709 | 0.8411 |
| L1 aggressive | 9 | 0.2333 | 0.5388 | 0.6940 | 0.4709 | 0.8189 |
| Balanced strict | 9 | 0.1533 | 0.4312 | 0.6966 | 0.4677 | 0.8872 |
| M3 risk budget 0.80 | 9 | 0.2267 | 0.5465 | 0.7155 | 0.4702 | 0.8500 |
| M3 sign-heavy 0.80 | 9 | 0.2556 | 0.5415 | 0.7086 | 0.4693 | 0.8500 |

## Interpretation

Tightening `theta_tol` alone has little effect on fraud survival and lowers rarity
retention, so L2 thresholding is not the main bottleneck. Increasing L1 visibility is more
effective: both `recheck50` and `l1_aggressive` reduce fraud survival by about one third
while preserving substantially more rarity than the strict low-fraud profile.

The M3 risk-budget router is the best current capability-oriented update. It does
not merely lower a threshold: it ranks candidates by combined cosine, norm-MAD,
and sign-disagreement risk, sends the highest-risk budget to L2, and keeps a small
stratified random audit slice. Relative to L1 aggressive, it slightly lowers fraud
survival (`0.2333 -> 0.2267`), improves rarity retention (`0.5388 -> 0.5465`), and
improves corner accuracy (`0.6940 -> 0.7155`) at a modestly higher L1 review rate
(`0.8189 -> 0.8500`).

The sign-heavy variant is rejected. Although it catches one additional corner-harm
fraud instance, it increases sign-flip survival (`0.2694 -> 0.3083`) and lowers the
macro result. This suggests real non-IID client drift also produces sign disagreement,
so sign should remain a supporting signal rather than dominate the risk score.

The recommended default is the M3 risk-budget profile:

```text
theta_tol = 0.02
theta_rare = -0.005
cosine_filter_threshold = 0.50
recheck_probability = 0.25
cornerdrive_l1_mode = v3_m3_budgeted
queue_budget_ratio = 0.80
random_recheck_ratio = 0.05
cos_weight = 0.35
norm_weight = 0.20
sign_weight = 0.15
norm_mad_threshold = 1.5
sign_threshold = 0.40
```

This profile keeps the L2 utility semantics unchanged and improves the main weakness
shown by the held-out benchmark: sign-flip fraud visibility under real non-IID gradients.

`recheck_probability = 0.50` is the minimal-change alternative. It performs similarly but
requires more L1 review. The `balanced_strict` profile is useful as a low-fraud upper
bound, but it should not be the default because rarity retention drops too much.

Attack-family decomposition for the two best non-strict profiles:

| Profile | Sign-flip survival | Corner-harm survival | Selected sign-flip / total | Selected corner-harm / total |
| --- | ---: | ---: | ---: | ---: |
| L1 aggressive | 0.2750 | 0.0667 | 99 / 360 | 6 / 90 |
| M3 risk budget 0.80 | 0.2694 | 0.0556 | 97 / 360 | 5 / 90 |

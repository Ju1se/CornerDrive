## Scenario: mixed

5 seeds (20261110, 20261111, 20261112, 20261113, 20261114), 100 rounds, 20 updates per round, model `r50head`. Accuracy is the mean of the last ten rounds on the held-out evaluation set.

### Method comparison

| Method | Normal acc | Rare acc | All acc | Rarity kept | Rarity share | Honest kept | Sign-flip share | Label-flip share | Corner-harm share | Fraud kept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Oracle | 90.4 ± 0.4 | 87.5 ± 1.1 | 89.5 ± 0.2 | 100.0 ± 0.0 | 1.43 ± 0.04 | 100.0 ± 0.0 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.0 ± 0.0 |
| FedAvg | 63.8 ± 11.0 | 61.1 ± 16.0 | 63.0 ± 11.7 | 100.0 ± 0.0 | 1.00 ± 0.00 | 100.0 ± 0.0 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 100.0 ± 0.0 |
| GeoMed | 89.7 ± 0.4 | 81.1 ± 4.5 | 87.1 ± 1.3 | 100.0 ± 0.0 | 1.01 ± 0.06 | 100.0 ± 0.0 | 0.30 ± 0.02 | 0.37 ± 0.02 | 0.66 ± 0.05 | 100.0 ± 0.0 |
| Multi-Krum | 90.5 ± 0.4 | 82.5 ± 3.3 | 88.1 ± 0.9 | 61.6 ± 7.0 | 0.95 ± 0.11 | 95.4 ± 1.7 | 0.02 ± 0.03 | 0.00 ± 0.00 | 0.39 ± 0.17 | 6.0 ± 3.0 |
| FLTrust | 89.2 ± 0.2 | 91.0 ± 0.6 | 89.8 ± 0.2 | 87.3 ± 2.7 | 2.43 ± 0.29 | 64.2 ± 1.9 | 0.45 ± 0.12 | 0.53 ± 0.03 | 0.24 ± 0.07 | 34.4 ± 3.6 |
| Zeno | 89.2 ± 0.7 | 86.9 ± 2.6 | 88.5 ± 0.9 | 99.9 ± 0.2 | 1.33 ± 0.00 | 94.6 ± 1.0 | 0.39 ± 0.06 | 0.05 ± 0.02 | 0.08 ± 0.07 | 14.6 ± 3.0 |
| Zeno++ | 89.3 ± 0.5 | 89.9 ± 1.4 | 89.5 ± 0.6 | 93.3 ± 2.3 | 1.79 ± 0.18 | 53.6 ± 3.2 | 0.71 ± 0.16 | 0.97 ± 0.04 | 0.10 ± 0.08 | 35.1 ± 4.5 |
| FedVal | 89.0 ± 0.6 | 87.2 ± 1.2 | 88.5 ± 0.7 | 99.8 ± 0.6 | 1.71 ± 0.20 | 94.9 ± 2.3 | 0.40 ± 0.04 | 0.26 ± 0.08 | 0.19 ± 0.07 | 45.4 ± 7.7 |
| CornerDrive (L1+L2) | 90.3 ± 0.4 | 87.4 ± 1.7 | 89.4 ± 0.3 | 99.7 ± 0.7 | 1.47 ± 0.02 | 91.2 ± 2.3 | 0.10 ± 0.02 | 0.05 ± 0.02 | 0.18 ± 0.08 | 7.5 ± 0.9 |
| CornerDrive (exhaustive L2) | 90.4 ± 0.5 | 87.9 ± 1.2 | 89.6 ± 0.2 | 99.5 ± 0.8 | 1.50 ± 0.01 | 89.9 ± 3.3 | 0.03 ± 0.03 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.9 ± 0.9 |
| Main-only audit | 90.5 ± 0.5 | 81.1 ± 5.3 | 87.7 ± 1.3 | 81.6 ± 4.9 | 1.21 ± 0.09 | 90.5 ± 2.3 | 0.00 ± 0.01 | 0.00 ± 0.00 | 0.71 ± 0.29 | 10.5 ± 4.2 |
| Corner-only audit | 36.8 ± 14.8 | 88.8 ± 2.5 | 52.4 ± 11.0 | 77.7 ± 5.2 | 2.49 ± 0.24 | 33.0 ± 8.0 | 1.37 ± 0.23 | 1.46 ± 0.27 | 0.00 ± 0.01 | 35.6 ± 3.2 |
| CornerDrive (embedding space) | 90.6 ± 0.4 | 86.0 ± 2.3 | 89.2 ± 0.7 | 77.2 ± 24.1 | 1.15 ± 0.34 | 93.0 ± 0.8 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.1 ± 0.1 |

Share = mean aggregation weight relative to an equal share (FedAvg = 1.00). Kept = fraction of updates with non-zero weight.

### Seed-paired accuracy difference: CornerDrive (L1+L2) minus each method

| Compared with | Δ all acc | Δ rare acc | Δ normal acc |
|---|---:|---:|---:|
| Oracle | -0.1 ± 0.2 | -0.1 ± 0.9 | -0.1 ± 0.1 |
| FedAvg | +26.4 ± 11.5 * | +26.2 ± 14.8 * | +26.5 ± 11.1 * |
| GeoMed | +2.3 ± 1.0 * | +6.3 ± 3.2 * | +0.6 ± 0.4 * |
| Multi-Krum | +1.3 ± 0.6 * | +4.9 ± 2.3 * | -0.2 ± 0.1 * |
| FLTrust | -0.3 ± 0.3 * | -3.6 ± 1.2 * | +1.0 ± 0.4 * |
| Zeno | +0.9 ± 0.6 * | +0.5 ± 1.4 | +1.1 ± 0.6 * |
| Zeno++ | -0.0 ± 0.5 | -2.5 ± 1.6 * | +1.0 ± 0.5 * |
| FedVal | +0.9 ± 0.5 * | +0.2 ± 1.3 | +1.3 ± 0.6 * |
| CornerDrive (exhaustive L2) | -0.2 ± 0.2 * | -0.5 ± 0.5 | -0.1 ± 0.1 * |
| Main-only audit | +1.7 ± 1.0 * | +6.3 ± 3.5 * | -0.2 ± 0.1 * |
| Corner-only audit | +37.0 ± 10.7 * | -1.4 ± 1.4 * | +53.5 ± 15.0 * |
| CornerDrive (embedding space) | +0.2 ± 0.8 | +1.4 ± 2.5 | -0.3 ± 0.1 * |

Points, mean ± half-width of the 95% t-interval over paired seeds; * marks an interval that excludes zero.

### Rarity discovery by the dual-dataset test (each method's own run)

| Audit | Review rate | Recognition | Precision | Said Rarity | of which fraud | of which corner-harm | of which honest/noise |
|---|---:|---:|---:|---:|---:|---:|---:|
| CornerDrive (L1+L2) | 51.3 ± 1.0 | 37.5 ± 3.3 | 83.8 ± 9.1 | 472 | 11 | 0 | 67 |
| CornerDrive (exhaustive L2) | 100.0 ± 0.0 | 98.0 ± 2.1 | 86.2 ± 9.2 | 1200 | 19 | 0 | 147 |
| CornerDrive (embedding space) | 100.0 ± 0.0 | 72.8 ± 22.4 | 79.0 ± 18.1 | 966 | 1 | 0 | 184 |
| Corner-only audit | 100.0 ± 0.0 | 52.6 ± 8.6 | 41.4 ± 8.7 | 1345 | 662 | 1 | 129 |
| Main-only audit | 100.0 ± 0.0 | 0.0 ± 0.0 | – | 0 | 0 | 0 | 0 |

### Update-level confusion, CornerDrive (L1+L2) (pooled over seeds)

| True type | Total | Fraud | Rarity | Honest | Noise | Not audited |
|---|---:|---:|---:|---:|---:|---:|
| Honest | 5907 | 55 | 52 | 1396 | 467 | 3937 |
| Rarity (rare-rich honest) | 1055 | 3 | 394 | 2 | 0 | 656 |
| Noise | 492 | 4 | 15 | 171 | 204 | 98 |
| Fraud: sign-flip | 1017 | 935 | 11 | 4 | 5 | 62 |
| Fraud: label-flip | 983 | 942 | 0 | 0 | 0 | 41 |
| Fraud: corner-harm | 546 | 472 | 0 | 0 | 0 | 74 |

### Update-level confusion, CornerDrive (exhaustive L2) (pooled over seeds)

| True type | Total | Fraud | Rarity | Honest | Noise | Not audited |
|---|---:|---:|---:|---:|---:|---:|
| Honest | 5907 | 147 | 132 | 5181 | 447 | 0 |
| Rarity (rare-rich honest) | 1055 | 5 | 1034 | 16 | 0 | 0 |
| Noise | 492 | 1 | 15 | 261 | 215 | 0 |
| Fraud: sign-flip | 1017 | 991 | 19 | 3 | 4 | 0 |
| Fraud: label-flip | 983 | 983 | 0 | 0 | 0 | 0 |
| Fraud: corner-harm | 546 | 546 | 0 | 0 | 0 | 0 |

### L1 routing by type and reason, CornerDrive (L1+L2), pooled

| True type | Total | Cosine | Norm-MAD | Sign | Recheck | Bypass | Routed % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Honest | 5907 | 371 | 191 | 75 | 1333 | 3937 | 33.4 |
| Rarity (rare-rich honest) | 1055 | 99 | 63 | 22 | 215 | 656 | 37.8 |
| Noise | 492 | 0 | 371 | 0 | 23 | 98 | 80.1 |
| Fraud: sign-flip | 1017 | 160 | 774 | 1 | 20 | 62 | 93.9 |
| Fraud: label-flip | 983 | 51 | 873 | 0 | 18 | 41 | 95.8 |
| Fraud: corner-harm | 546 | 45 | 399 | 0 | 28 | 74 | 86.4 |

### Golden-set signature of every update type (Oracle trajectory, relative drift, median [p5, p95])

| True type | n | Loss drift, normal set | Loss drift, rare set | Embedding drift, normal set | Embedding drift, rare set |
|---|---:|---:|---:|---:|---:|
| Honest | 5907 | +0.016 [-0.026, +0.078] | +0.036 [-0.013, +0.098] | +0.003 [-0.020, +0.026] | +0.011 [-0.010, +0.033] |
| Rarity (rare-rich honest) | 1055 | +0.041 [+0.009, +0.092] | -0.177 [-0.333, -0.057] | +0.023 [+0.005, +0.047] | -0.061 [-0.180, -0.013] |
| Noise | 492 | +0.012 [-0.031, +0.066] | +0.046 [-0.015, +0.098] | +0.002 [-0.024, +0.026] | +0.014 [-0.010, +0.037] |
| Fraud: sign-flip | 1017 | +0.395 [+0.120, +0.970] | -0.008 [-0.134, +0.169] | +0.154 [+0.067, +0.296] | +0.058 [-0.013, +0.147] |
| Fraud: label-flip | 983 | +0.593 [+0.373, +0.910] | -0.196 [-0.418, -0.008] | +0.382 [+0.179, +0.606] | +0.200 [+0.042, +0.454] |
| Fraud: corner-harm | 546 | +0.007 [-0.054, +0.099] | +1.563 [+0.572, +2.282] | +0.128 [+0.076, +0.175] | +0.200 [+0.042, +0.367] |

### Detectors on the same update stream (Oracle trajectory, AUROC per seed)

Is a rare-beneficial update valued above the others?

| Score | Rarity vs honest | Rarity vs corner-harm | Rarity vs label-flip | Rarity vs sign-flip | Rarity vs all others |
|---|---:|---:|---:|---:|---:|
| Rare-set loss drift alone (corner-only) | 0.999 ± 0.002 | 1.000 ± 0.000 | 0.454 ± 0.116 | 0.944 ± 0.040 | 0.932 ± 0.019 |
| Dual test, loss space (rare-set gain if both harm tests pass) | 0.993 ± 0.006 | 0.998 ± 0.004 | 0.998 ± 0.004 | 0.993 ± 0.005 | 0.994 ± 0.005 |
| Rare-set embedding drift alone | 0.986 ± 0.011 | 0.995 ± 0.011 | 0.999 ± 0.001 | 0.987 ± 0.009 | 0.988 ± 0.008 |
| Dual test, embedding space (rare-set gain if both harm tests pass) | 0.854 ± 0.121 | 0.936 ± 0.060 | 0.936 ± 0.060 | 0.935 ± 0.060 | 0.877 ± 0.104 |
| Zeno score | 0.946 ± 0.028 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.962 ± 0.021 |
| Zeno++ score | 0.957 ± 0.012 | 1.000 ± 0.000 | 0.259 ± 0.082 | 0.825 ± 0.091 | 0.868 ± 0.024 |
| FLTrust cosine | 0.949 ± 0.006 | 1.000 ± 0.001 | 0.858 ± 0.041 | 0.958 ± 0.017 | 0.945 ± 0.007 |
| Multi-Krum score | 0.412 ± 0.028 | 0.997 ± 0.003 | 1.000 ± 0.000 | 0.999 ± 0.002 | 0.611 ± 0.018 |
| FedVal score | 0.881 ± 0.023 | 1.000 ± 0.000 | 0.994 ± 0.009 | 0.991 ± 0.009 | 0.914 ± 0.015 |

Are benign updates trusted above fraud?

| Score | Benign vs sign-flip | Benign vs label-flip | Benign vs corner-harm | Benign vs all fraud | Rarity vs honest (0.5 = no bias) |
|---|---:|---:|---:|---:|---:|
| Dual test, loss space (larger of the two drifts / tolerance) | 0.993 ± 0.006 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.997 ± 0.003 | 0.535 ± 0.083 |
| Dual test, embedding space (larger of the two drifts / tolerance) | 0.998 ± 0.003 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.999 ± 0.001 | 0.237 ± 0.114 |
| Zeno score | 0.994 ± 0.008 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.998 ± 0.003 | 0.946 ± 0.028 |
| Zeno++ score | 0.508 ± 0.091 | 0.056 ± 0.025 | 0.999 ± 0.002 | 0.439 ± 0.037 | 0.957 ± 0.012 |
| FLTrust cosine | 0.555 ± 0.032 | 0.311 ± 0.067 | 0.963 ± 0.019 | 0.549 ± 0.027 | 0.949 ± 0.006 |
| Multi-Krum score | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.999 ± 0.001 | 1.000 ± 0.000 | 0.412 ± 0.028 |
| FedVal score | 0.925 ± 0.042 | 0.987 ± 0.015 | 0.999 ± 0.001 | 0.964 ± 0.022 | 0.881 ± 0.023 |

### Frozen dual test applied to every update (Oracle trajectory)

| Dual test on the reference stream | Recognition | Precision | Honest named Rarity | Fraud named Rarity | Sign-flip passed | Label-flip passed | Corner-harm passed | Honest rejected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| loss space | 98.3 ± 1.7 | 82.5 ± 10.2 | 2.7 ± 2.1 | 1.8 ± 2.1 | 5.8 ± 5.6 | 0.0 ± 0.0 | 0.0 ± 0.0 | 1.8 ± 1.5 |
| embedding space | 80.5 ± 11.7 | 83.8 ± 10.5 | 2.3 ± 1.1 | 0.1 ± 0.1 | 0.6 ± 0.9 | 0.1 ± 0.3 | 0.0 ± 0.0 | 0.8 ± 0.3 |

### Golden-set signature of every update type (FedAvg trajectory, relative drift, median [p5, p95])

| True type | n | Loss drift, normal set | Loss drift, rare set | Embedding drift, normal set | Embedding drift, rare set |
|---|---:|---:|---:|---:|---:|
| Honest | 5907 | -0.063 [-0.224, +0.041] | +0.089 [-0.013, +0.237] | -0.040 [-0.077, -0.010] | -0.013 [-0.041, +0.017] |
| Rarity (rare-rich honest) | 1055 | +0.025 [-0.116, +0.102] | -0.338 [-0.457, -0.195] | -0.035 [-0.067, -0.006] | -0.119 [-0.204, -0.074] |
| Noise | 492 | -0.071 [-0.240, +0.016] | +0.087 [-0.016, +0.247] | -0.041 [-0.081, -0.011] | -0.011 [-0.045, +0.016] |
| Fraud: sign-flip | 1017 | +1.692 [+0.561, +3.427] | +0.038 [-0.237, +0.784] | +0.262 [+0.078, +0.438] | +0.170 [+0.055, +0.325] |
| Fraud: label-flip | 983 | +0.603 [+0.314, +0.905] | -0.154 [-0.350, +0.041] | +0.067 [-0.025, +0.233] | +0.094 [+0.023, +0.208] |
| Fraud: corner-harm | 546 | -0.066 [-0.196, +0.023] | +1.172 [+0.498, +1.780] | +0.041 [-0.007, +0.134] | +0.078 [-0.011, +0.186] |

### Detectors on the same update stream (FedAvg trajectory, AUROC per seed)

Is a rare-beneficial update valued above the others?

| Score | Rarity vs honest | Rarity vs corner-harm | Rarity vs label-flip | Rarity vs sign-flip | Rarity vs all others |
|---|---:|---:|---:|---:|---:|
| Rare-set loss drift alone (corner-only) | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.892 ± 0.029 | 0.983 ± 0.009 | 0.986 ± 0.002 |
| Dual test, loss space (rare-set gain if both harm tests pass) | 0.991 ± 0.019 | 0.994 ± 0.012 | 0.994 ± 0.012 | 0.994 ± 0.012 | 0.992 ± 0.017 |
| Rare-set embedding drift alone | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Dual test, embedding space (rare-set gain if both harm tests pass) | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Zeno score | 0.827 ± 0.039 | 1.000 ± 0.001 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.874 ± 0.031 |
| Zeno++ score | 0.790 ± 0.048 | 0.952 ± 0.047 | 0.983 ± 0.007 | 0.999 ± 0.002 | 0.846 ± 0.035 |
| FLTrust cosine | 0.901 ± 0.025 | 0.995 ± 0.006 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.934 ± 0.017 |
| Multi-Krum score | 0.398 ± 0.102 | 0.727 ± 0.127 | 0.938 ± 0.063 | 0.998 ± 0.006 | 0.576 ± 0.072 |
| FedVal score | 0.772 ± 0.079 | 0.959 ± 0.048 | 0.973 ± 0.035 | 0.999 ± 0.001 | 0.831 ± 0.063 |

Are benign updates trusted above fraud?

| Score | Benign vs sign-flip | Benign vs label-flip | Benign vs corner-harm | Benign vs all fraud | Rarity vs honest (0.5 = no bias) |
|---|---:|---:|---:|---:|---:|
| Dual test, loss space (larger of the two drifts / tolerance) | 1.000 ± 0.000 | 0.998 ± 0.003 | 0.999 ± 0.001 | 0.999 ± 0.001 | 0.805 ± 0.057 |
| Dual test, embedding space (larger of the two drifts / tolerance) | 1.000 ± 0.000 | 0.997 ± 0.001 | 0.983 ± 0.017 | 0.995 ± 0.004 | 0.891 ± 0.111 |
| Zeno score | 1.000 ± 0.001 | 1.000 ± 0.001 | 0.998 ± 0.004 | 0.999 ± 0.001 | 0.827 ± 0.039 |
| Zeno++ score | 0.986 ± 0.014 | 0.890 ± 0.043 | 0.915 ± 0.058 | 0.934 ± 0.027 | 0.790 ± 0.048 |
| FLTrust cosine | 0.998 ± 0.002 | 0.991 ± 0.004 | 0.965 ± 0.032 | 0.989 ± 0.008 | 0.901 ± 0.025 |
| Multi-Krum score | 0.996 ± 0.010 | 0.932 ± 0.067 | 0.762 ± 0.074 | 0.922 ± 0.043 | 0.398 ± 0.102 |
| FedVal score | 0.993 ± 0.005 | 0.938 ± 0.050 | 0.904 ± 0.059 | 0.953 ± 0.031 | 0.772 ± 0.079 |

### Frozen dual test applied to every update (FedAvg trajectory)

| Dual test on the reference stream | Recognition | Precision | Honest named Rarity | Fraud named Rarity | Sign-flip passed | Label-flip passed | Corner-harm passed | Honest rejected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| loss space | 98.9 ± 2.4 | 81.1 ± 5.2 | 3.8 ± 1.3 | 0.0 ± 0.1 | 0.0 ± 0.0 | 0.1 ± 0.3 | 0.0 ± 0.0 | 30.6 ± 11.9 |
| embedding space | 100.0 ± 0.0 | 29.0 ± 11.4 | 44.0 ± 18.7 | 0.7 ± 1.3 | 0.1 ± 0.3 | 17.3 ± 4.6 | 26.6 ± 24.2 | 0.1 ± 0.1 |

## Scenario: no_attack

5 seeds (20261110, 20261111, 20261112, 20261113, 20261114), 100 rounds, 20 updates per round, model `r50head`. Accuracy is the mean of the last ten rounds on the held-out evaluation set.

### Method comparison

| Method | Normal acc | Rare acc | All acc | Rarity kept | Rarity share | Honest kept |
|---|---:|---:|---:|---:|---:|---:|
| Oracle | 90.6 ± 0.4 | 88.4 ± 0.7 | 90.0 ± 0.2 | 100.0 ± 0.0 | 1.00 ± 0.00 | 100.0 ± 0.0 |
| FedAvg | 90.6 ± 0.4 | 88.4 ± 0.7 | 90.0 ± 0.2 | 100.0 ± 0.0 | 1.00 ± 0.00 | 100.0 ± 0.0 |
| GeoMed | 90.6 ± 0.4 | 87.7 ± 0.8 | 89.7 ± 0.2 | 100.0 ± 0.0 | 0.88 ± 0.04 | 100.0 ± 0.0 |
| Multi-Krum | 90.5 ± 0.5 | 74.1 ± 0.9 | 85.6 ± 0.5 | 1.2 ± 0.9 | 0.02 ± 0.01 | 77.2 ± 1.2 |
| FLTrust | 89.8 ± 0.2 | 91.8 ± 0.7 | 90.4 ± 0.2 | 83.6 ± 3.5 | 1.90 ± 0.12 | 61.7 ± 1.8 |
| Zeno | 90.3 ± 0.3 | 89.8 ± 0.9 | 90.1 ± 0.1 | 98.7 ± 1.5 | 1.32 ± 0.02 | 70.5 ± 0.5 |
| Zeno++ | 90.1 ± 0.3 | 91.6 ± 0.6 | 90.5 ± 0.1 | 91.7 ± 3.1 | 1.72 ± 0.13 | 48.4 ± 2.4 |
| FedVal | 90.3 ± 0.3 | 90.6 ± 0.7 | 90.4 ± 0.2 | 98.8 ± 1.6 | 1.82 ± 0.22 | 80.6 ± 2.4 |
| CornerDrive (L1+L2) | 90.5 ± 0.3 | 89.3 ± 0.8 | 90.1 ± 0.1 | 99.0 ± 1.2 | 1.15 ± 0.03 | 83.9 ± 2.2 |
| CornerDrive (exhaustive L2) | 90.4 ± 0.4 | 89.5 ± 1.0 | 90.1 ± 0.1 | 99.0 ± 1.2 | 1.18 ± 0.04 | 81.7 ± 3.9 |
| Main-only audit | 90.8 ± 0.5 | 81.1 ± 1.7 | 87.9 ± 0.5 | 14.3 ± 8.3 | 0.18 ± 0.10 | 85.9 ± 1.2 |
| Corner-only audit | 90.3 ± 0.3 | 89.7 ± 0.7 | 90.1 ± 0.1 | 99.7 ± 0.4 | 1.24 ± 0.02 | 77.5 ± 1.8 |
| CornerDrive (embedding space) | 90.6 ± 0.4 | 87.8 ± 1.7 | 89.7 ± 0.4 | 79.8 ± 15.8 | 0.92 ± 0.18 | 87.1 ± 1.3 |

Share = mean aggregation weight relative to an equal share (FedAvg = 1.00). Kept = fraction of updates with non-zero weight.

### Seed-paired accuracy difference: CornerDrive (L1+L2) minus each method

| Compared with | Δ all acc | Δ rare acc | Δ normal acc |
|---|---:|---:|---:|
| Oracle | +0.2 ± 0.1 * | +0.9 ± 0.2 * | -0.2 ± 0.1 * |
| FedAvg | +0.2 ± 0.1 * | +0.9 ± 0.2 * | -0.2 ± 0.1 * |
| GeoMed | +0.4 ± 0.1 * | +1.6 ± 0.2 * | -0.1 ± 0.1 * |
| Multi-Krum | +4.5 ± 0.4 * | +15.2 ± 1.1 * | -0.0 ± 0.4 |
| FLTrust | -0.3 ± 0.1 * | -2.5 ± 0.3 * | +0.6 ± 0.3 * |
| Zeno | -0.0 ± 0.2 | -0.5 ± 0.2 * | +0.2 ± 0.2 * |
| Zeno++ | -0.4 ± 0.1 * | -2.3 ± 0.3 * | +0.4 ± 0.2 * |
| FedVal | -0.3 ± 0.3 | -1.2 ± 0.5 * | +0.1 ± 0.2 |
| CornerDrive (exhaustive L2) | -0.0 ± 0.1 | -0.2 ± 0.3 | +0.0 ± 0.1 |
| Main-only audit | +2.3 ± 0.5 * | +8.2 ± 1.7 * | -0.3 ± 0.2 * |
| Corner-only audit | +0.0 ± 0.1 | -0.4 ± 0.3 * | +0.2 ± 0.2 * |
| CornerDrive (embedding space) | +0.4 ± 0.3 * | +1.5 ± 1.0 * | -0.1 ± 0.1 |

Points, mean ± half-width of the 95% t-interval over paired seeds; * marks an interval that excludes zero.

### Rarity discovery by the dual-dataset test (each method's own run)

| Audit | Review rate | Recognition | Precision | Said Rarity | of which fraud | of which corner-harm | of which honest/noise |
|---|---:|---:|---:|---:|---:|---:|---:|
| CornerDrive (L1+L2) | 42.3 ± 1.5 | 49.6 ± 3.0 | 92.8 ± 7.1 | 856 | 0 | 0 | 64 |
| CornerDrive (exhaustive L2) | 100.0 ± 0.0 | 96.2 ± 2.1 | 91.8 ± 5.7 | 1679 | 0 | 0 | 140 |
| CornerDrive (embedding space) | 100.0 ± 0.0 | 74.0 ± 12.7 | 84.8 ± 11.1 | 1396 | 0 | 0 | 203 |
| Corner-only audit | 100.0 ± 0.0 | 96.7 ± 1.8 | 92.7 ± 4.5 | 1670 | 0 | 0 | 123 |
| Main-only audit | 100.0 ± 0.0 | 0.0 ± 0.0 | – | 0 | 0 | 0 | 0 |

### Update-level confusion, CornerDrive (L1+L2) (pooled over seeds)

| True type | Total | Fraud | Rarity | Honest | Noise | Not audited |
|---|---:|---:|---:|---:|---:|---:|
| Honest | 8399 | 159 | 64 | 2000 | 1189 | 4987 |
| Rarity (rare-rich honest) | 1601 | 12 | 792 | 10 | 3 | 784 |

### Update-level confusion, CornerDrive (exhaustive L2) (pooled over seeds)

| True type | Total | Fraud | Rarity | Honest | Noise | Not audited |
|---|---:|---:|---:|---:|---:|---:|
| Honest | 8399 | 350 | 140 | 6724 | 1185 | 0 |
| Rarity (rare-rich honest) | 1601 | 13 | 1539 | 46 | 3 | 0 |

### L1 routing by type and reason, CornerDrive (L1+L2), pooled

| True type | Total | Cosine | Norm-MAD | Sign | Recheck | Bypass | Routed % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Honest | 8399 | 401 | 1306 | 68 | 1637 | 4987 | 40.6 |
| Rarity (rare-rich honest) | 1601 | 130 | 392 | 30 | 265 | 784 | 51.0 |

## Stress tests (mixed scenario, frozen policy unchanged)

| Condition | Method | Rare acc | All acc | Rarity kept | Recognition | Precision | Corner-harm share | Fraud kept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Default | Oracle | 87.5 ± 1.1 | 89.5 ± 0.2 | 100.0 ± 0.0 | – | – | 0.00 ± 0.00 | 0.0 ± 0.0 |
| Default | Multi-Krum | 82.5 ± 3.3 | 88.1 ± 0.9 | 61.6 ± 7.0 | – | – | 0.39 ± 0.17 | 6.0 ± 3.0 |
| Default | Zeno | 86.9 ± 2.6 | 88.5 ± 0.9 | 99.9 ± 0.2 | – | – | 0.08 ± 0.07 | 14.6 ± 3.0 |
| Default | FedVal | 87.2 ± 1.2 | 88.5 ± 0.7 | 99.8 ± 0.6 | – | – | 0.19 ± 0.07 | 45.4 ± 7.7 |
| Default | CornerDrive (L1+L2) | 87.4 ± 1.7 | 89.4 ± 0.3 | 99.7 ± 0.7 | 37.5 ± 3.3 | 83.8 ± 9.1 | 0.18 ± 0.08 | 7.5 ± 0.9 |
| Default | CornerDrive (exhaustive L2) | 87.9 ± 1.2 | 89.6 ± 0.2 | 99.5 ± 0.8 | 98.0 ± 2.1 | 86.2 ± 9.2 | 0.00 ± 0.00 | 0.9 ± 0.9 |
| Weak rarity (35% rare) | Oracle | 85.4 ± 1.1 | 89.0 ± 0.5 | 100.0 ± 0.0 | – | – | 0.00 ± 0.00 | 0.0 ± 0.0 |
| Weak rarity (35% rare) | Multi-Krum | 80.3 ± 2.7 | 87.4 ± 0.7 | 62.9 ± 6.6 | – | – | 0.41 ± 0.19 | 6.3 ± 3.0 |
| Weak rarity (35% rare) | Zeno | 83.1 ± 2.2 | 87.6 ± 0.9 | 99.9 ± 0.3 | – | – | 0.22 ± 0.06 | 14.6 ± 3.0 |
| Weak rarity (35% rare) | FedVal | 84.3 ± 1.7 | 87.8 ± 1.0 | 98.5 ± 2.9 | – | – | 0.22 ± 0.07 | 44.2 ± 10.1 |
| Weak rarity (35% rare) | CornerDrive (L1+L2) | 84.5 ± 1.2 | 88.6 ± 0.4 | 99.7 ± 0.6 | 33.7 ± 2.6 | 79.4 ± 14.6 | 0.28 ± 0.13 | 8.9 ± 1.7 |
| Weak rarity (35% rare) | CornerDrive (exhaustive L2) | 85.5 ± 1.0 | 89.0 ± 0.4 | 99.4 ± 1.7 | 97.2 ± 3.2 | 83.2 ± 10.1 | 0.01 ± 0.01 | 1.1 ± 0.7 |
| Partial rare set (no truck) | Oracle | 87.5 ± 1.1 | 89.5 ± 0.2 | 100.0 ± 0.0 | – | – | 0.00 ± 0.00 | 0.0 ± 0.0 |
| Partial rare set (no truck) | Multi-Krum | 82.5 ± 3.3 | 88.1 ± 0.9 | 61.6 ± 7.0 | – | – | 0.39 ± 0.17 | 6.0 ± 3.0 |
| Partial rare set (no truck) | Zeno | 86.9 ± 2.6 | 88.5 ± 0.9 | 99.9 ± 0.2 | – | – | 0.08 ± 0.07 | 14.6 ± 3.0 |
| Partial rare set (no truck) | FedVal | 87.2 ± 1.2 | 88.5 ± 0.7 | 99.8 ± 0.6 | – | – | 0.19 ± 0.07 | 45.4 ± 7.7 |
| Partial rare set (no truck) | CornerDrive (L1+L2) | 86.8 ± 2.3 | 89.2 ± 0.4 | 98.9 ± 0.8 | 34.8 ± 3.7 | 74.7 ± 8.6 | 0.19 ± 0.04 | 7.6 ± 1.4 |
| Partial rare set (no truck) | CornerDrive (exhaustive L2) | 88.0 ± 1.3 | 89.6 ± 0.3 | 98.3 ± 0.6 | 87.0 ± 2.3 | 79.6 ± 9.4 | 0.00 ± 0.00 | 0.6 ± 0.6 |
| Random normal slice as rare set | Oracle | 87.5 ± 1.1 | 89.5 ± 0.2 | 100.0 ± 0.0 | – | – | 0.00 ± 0.00 | 0.0 ± 0.0 |
| Random normal slice as rare set | Multi-Krum | 82.5 ± 3.3 | 88.1 ± 0.9 | 61.6 ± 7.0 | – | – | 0.39 ± 0.17 | 6.0 ± 3.0 |
| Random normal slice as rare set | Zeno | 86.9 ± 2.6 | 88.5 ± 0.9 | 99.9 ± 0.2 | – | – | 0.08 ± 0.07 | 14.6 ± 3.0 |
| Random normal slice as rare set | FedVal | 87.2 ± 1.2 | 88.5 ± 0.7 | 99.8 ± 0.6 | – | – | 0.19 ± 0.07 | 45.4 ± 7.7 |
| Random normal slice as rare set | CornerDrive (L1+L2) | 80.2 ± 6.8 | 87.3 ± 1.8 | 81.6 ± 5.9 | 0.1 ± 0.2 | 0.3 ± 0.8 | 0.86 ± 0.27 | 18.1 ± 4.7 |
| Random normal slice as rare set | CornerDrive (exhaustive L2) | 79.5 ± 6.3 | 87.2 ± 1.6 | 81.3 ± 6.1 | 0.4 ± 0.7 | 0.5 ± 1.0 | 0.85 ± 0.25 | 12.8 ± 4.0 |
| Small golden draw (10 per class) | Oracle | 87.5 ± 1.1 | 89.5 ± 0.2 | 100.0 ± 0.0 | – | – | 0.00 ± 0.00 | 0.0 ± 0.0 |
| Small golden draw (10 per class) | Multi-Krum | 82.5 ± 3.3 | 88.1 ± 0.9 | 61.6 ± 7.0 | – | – | 0.39 ± 0.17 | 6.0 ± 3.0 |
| Small golden draw (10 per class) | Zeno | 86.8 ± 1.9 | 88.5 ± 0.8 | 99.7 ± 0.5 | – | – | 0.10 ± 0.07 | 14.8 ± 2.9 |
| Small golden draw (10 per class) | FedVal | 87.1 ± 1.5 | 88.4 ± 0.8 | 98.8 ± 1.7 | – | – | 0.24 ± 0.07 | 41.6 ± 12.1 |
| Small golden draw (10 per class) | CornerDrive (L1+L2) | 87.1 ± 1.8 | 89.3 ± 0.4 | 97.1 ± 2.0 | 35.0 ± 3.9 | 65.5 ± 5.9 | 0.18 ± 0.07 | 9.8 ± 1.3 |
| Small golden draw (10 per class) | CornerDrive (exhaustive L2) | 87.9 ± 1.0 | 89.6 ± 0.3 | 92.8 ± 4.0 | 87.9 ± 2.7 | 68.6 ± 5.0 | 0.00 ± 0.00 | 3.0 ± 1.7 |

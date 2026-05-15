# Reviewer-Facing Audit Fixes

Date: 2026-05-15

This note records the code changes made after a reviewer-style audit for hard
coded results, data leakage, oracle tuning, expected-CSV misuse, seed/config
mismatch, and table-copying risk.

## What Changed

- Real-gradient benchmarks now build a `RealGradientDataBundle` with separate
  client/update, audit/reference, and final evaluation surfaces.
- MNIST and FashionMNIST use the training split for pseudo-client gradients and
  deterministically split the official test set into audit/reference and final
  evaluation subsets.
- LEAF/FEMNIST loads `train/` shards for client gradients and uses `test/` shards
  for audit/reference and final evaluation clients. The loader no longer scans
  test shards by default when building the client update pool.
- The reliability exporter defaults now match the thesis reproduction manifest:
  120 clients, 48 samples per client, 20 clients per round, 10 rounds, 50
  pretrain steps, three seeds, and `results/real_gradient_reliability_medium`.
- `configs/real_gradient_reliability.yaml` now records `download`,
  `pretrain_steps`, and reference/evaluation split parameters.
- Paper-facing real-gradient tables and `results/expected_results.csv` were
  regenerated under the held-out protocol.

## Updated Held-Out Result

The leakage-safe run changes the interpretation of the real-gradient claim. The
old same-surface result should no longer be used as the headline result.

| Method | Main accuracy | Corner accuracy | Fraud survival | Rarity retention |
| --- | ---: | ---: | ---: | ---: |
| Multi-Krum | 0.4617 | 0.6354 | 0.5044 | 0.6956 |
| FLTrust | 0.4998 | 0.7114 | 0.0289 | 0.6250 |
| Zeno | 0.5054 | 0.6774 | 0.2289 | 0.9246 |
| Zeno++ | 0.4966 | 0.6696 | 0.0000 | 0.2239 |
| CornerDrive | 0.4721 | 0.6611 | 0.3444 | 0.5763 |

This is less flattering for CornerDrive than the previous table, but it is the
right result to expose in a reviewer-facing reproduction package. Any future
improvement to `real_data_adaptive` should tune on a separate calibration split
and report on untouched seeds or datasets.

## Checks

- `python -m pytest backend/tests/test_real_gradient_bdd100k.py -q`
- `bash scripts/reproduce_all.sh real-gradient`

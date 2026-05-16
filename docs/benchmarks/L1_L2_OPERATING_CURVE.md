# L1/L2 Operating-Curve Benchmark

This benchmark complements the existing L1V3 ablation and exhaustive L2 audit.
Instead of comparing one fixed router setting at a time, it exports two
operating curves on the same generated rounds:

1. **L1 visibility-cost frontier**: varies router mode, recheck probability,
   and queue budget ratio while holding the L2 thresholds fixed.
2. **L2 threshold grid**: varies `theta_tol` and `theta_rare` while holding a
   selected L1 operating point fixed.

The generated client rounds are fixed per seed with `DEFAULT_POLICY`. This keeps
the underlying honest, rarity, fraud, and noise samples constant while evaluating
different L1/L2 settings, which avoids changing the benchmark population during
threshold calibration.

## Usage

Quick smoke run:

```bash
python scripts/export_l1_l2_operating_curve.py \
  --rounds 2 \
  --seeds 20260318 \
  --pretrain-epochs 1 \
  --skip-oracle-drift \
  --output-dir results/l1_l2_operating_curve_smoke
```

Thesis-scale run:

```bash
python scripts/export_l1_l2_operating_curve.py \
  --rounds 24 \
  --cycle-rounds 12 \
  --pretrain-epochs 5 \
  --seeds 20260318,20260319,20260320 \
  --l1-modes m0,m4 \
  --p-recheck-values 0.0,0.05,0.10 \
  --budget-values 0.20,0.35,0.50 \
  --theta-tol-values 0.025,0.05,0.075 \
  --theta-rare-values=-0.01,-0.03,-0.05 \
  --output-dir results/l1_l2_operating_curve
```

To isolate just one axis:

```bash
python scripts/export_l1_l2_operating_curve.py --sweep frontier
python scripts/export_l1_l2_operating_curve.py --sweep threshold
```

To generate the paper-facing audit cost frontier from the existing V2.5
recheck sweep:

```bash
python scripts/export_cost_performance_frontier.py \
  --input results/audit_reproduction/v25_artifacts_b24/v25_recheck_sweep_table.csv \
  --output-dir results/cost_performance_frontier
```

## Outputs

- `l1_l2_operating_curve_config.json`: run configuration and fixed generation
  policy note.
- `l1_l2_operating_curve_by_seed.csv`: one row per seed/configuration.
- `l1_l2_operating_curve_summary.csv`: mean/std summary grouped by L1/L2
  operating point.
- `cost_performance_frontier.csv`: compact table with audit queue rate,
  corner-harm survival, corner accuracy, main accuracy, and rarity recall.
- `audit_cost_frontier.svg`: dependency-free plot of L1 review cost against
  corner-harm survival and corner accuracy.

Key columns:

- `audit_queue_ratio`, `l2_evals`: L2 workload/cost proxy.
- `l1_fraud_recall`, `l1_rarity_recall`, `l1_precision_non_honest`: L1 routing
  quality.
- `l1_high_cosine_dropped`: high-deviation updates missed because of budgeted
  routing.
- `l2_fraud_precision_cond`, `l2_fraud_recall_cond`: L2 quality conditional on
  an update reaching L2.
- `e2e_fraud_catch_rate`, `fraud_survival_rate`: end-to-end security outcome.
- `e2e_rarity_recognition`, `rarity_retention`, `false_rarity_rate`: rarity
  preservation and false-positive trade-off.
- `main_accuracy`, `corner_accuracy`: downstream utility.

## What This Improves

The existing `export_l1v3_ablation.py` is useful for M0-M4 router comparison,
but it reports fixed operating points. This script shows the budget frontier:
how much L2 cost is needed to improve fraud catch and rarity retention.

The existing stress-test exporter includes threshold perturbations, but those
stress cases are not a compact L1/L2 calibration table. This script adds a
dedicated `theta_tol x theta_rare` grid so L2 can be tuned separately from the
client sample generator.

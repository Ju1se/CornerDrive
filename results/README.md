# Results Directory

Generated experiment outputs are written here. Most files under `results/` are
ignored by Git because they can be large and should be regenerated from scripts.

Tracked files in this directory:

- `README.md`: this guide.
- `expected_results.csv`: compact expected values for the thesis reproduction
  smoke check. It is validated by `scripts/check_expected_results.py`.

Typical generated outputs:

```text
results/real_gradient_reliability_calibrated_holdout/
results/audit_reproduction/synthetic_alg_benchmark_b24/
results/audit_reproduction/synthetic_stress_tests_b24/
results/audit_reproduction/corner_family_divergence_b24/
results/audit_reproduction/corner_harm_threshold_calibration_b24/
results/cost_performance_frontier/
results/real_gradient_calibration_split/
results/real_gradient_learning_curve/
```

To regenerate thesis tables:

```bash
bash scripts/reproduce_all.sh main
bash scripts/reproduce_all.sh real-gradient
bash scripts/reproduce_all.sh appendix
python scripts/make_paper_tables.py
python scripts/check_expected_results.py
```

The generated paper-facing tables are written to `artifacts/tables/`.
Older local directories such as `v25_*` are historical generated outputs, not
current reproduction targets.

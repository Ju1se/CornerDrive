# Results Directory

Generated experiment outputs are written here. Most files under `results/` are
ignored by Git because they can be large and should be regenerated from scripts.

Tracked files in this directory:

- `README.md`: this guide.
- `expected_results.csv`: compact expected values for the thesis reproduction
  smoke check.

Typical generated outputs:

```text
results/real_gradient_reliability_medium/
results/audit_reproduction/v25_artifacts_b24/
results/audit_reproduction/v25_stress_tests_b24/
results/audit_reproduction/corner_family_divergence_b24/
results/audit_reproduction/corner_harm_threshold_calibration_b24/
```

To regenerate thesis tables:

```bash
bash scripts/reproduce_all.sh main
bash scripts/reproduce_all.sh appendix
python scripts/make_paper_tables.py
```

The generated paper-facing tables are written to `artifacts/tables/`.

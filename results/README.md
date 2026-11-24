# Results Directory

Generated experiment outputs are written here. Most files under `results/` are
ignored by Git because they can be large and should be regenerated from scripts.

Tracked files in this directory:

- `README.md`: this guide.
- `expected_results.csv`: compact expected values for the thesis reproduction
  smoke check. It is validated by `scripts/check_expected_results.py`.
- `cifar10_dual_dataset/`: raw results of the CIFAR-10 dual-dataset rarity
  benchmark (see below).

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

## CIFAR-10 dual-dataset benchmark

`results/cifar10_dual_dataset/` holds every run behind
`docs/benchmarks/CIFAR10_DUAL_DATASET_BENCHMARK.md`: one
`unit_<scenario>_<method>_s<seed>.npz` per trajectory, plus the `run_args.json`
of each stage.

| Stage | Seeds | Content |
|---|---|---|
| `devr50_lr0.003` ... `devr50_lr0.1` | 20261101 | head learning-rate grid (oracle, FedAvg) |
| `devr50` | 20261101, 20261102 | development seeds; the oracle units are what the frozen policy was calibrated on |
| `confirm_r50` | 20261110-20261114 | confirmatory run: 13 methods, `mixed` and `no_attack`; `REPORT.md` and `summary_per_unit.csv` are built from it |
| `confirm_r50_reference` | 20261110-20261114 | the same oracle and FedAvg trajectories with the drift at the frozen replica step logged |
| `stress_weak`, `stress_partial`, `stress_random`, `stress_small` | 20261110-20261114 | stress tests |
| `nowarm_r50` | 20261110-20261114 | robustness check without the warm-up |
| `dev` | 20261101, 20261102 | CNN-from-scratch diagnostic (oracle, FedAvg) |
| `leakage_audit` | - | `leakage_audit.json` from `scripts/audit_cifar10_dual_dataset_leakage.py` |

A unit file is a NumPy archive. Arrays named `u_*` have one row per round and
one column per sampled update (`u_role`, `u_truth`, `u_weight`, `u_verdict`,
`u_routed`, `u_d_main`, `u_d_corner`, the baseline scores `u_score_*`, ...);
arrays named `r_*` have one value per round (`r_acc_normal`, `r_acc_rare`,
`r_base_main`, ...). `meta` is a JSON string with the seed, scenario, method,
engine and federation settings, the policy, and the code tables for roles,
truths, verdicts and routing reasons. `dual_dataset_benchmark.analysis.load_unit`
reads one.

Not tracked: the warm-up checkpoints (`warmup_*.pt`, rebuilt by the launcher),
the run logs, the CIFAR-10 batches and the ResNet-50 embeddings (under
`data/real/`, rebuilt by `scripts/extract_cifar10_embeddings.py`).

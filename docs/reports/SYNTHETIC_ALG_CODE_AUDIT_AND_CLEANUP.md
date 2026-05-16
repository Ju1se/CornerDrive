# synthetic ALG Code Audit and Cleanup

Generated: 2026-04-28

## Scope

This audit checked the synthetic benchmark and artifact-export paths for:

- labels derived from auditor verdicts instead of archetypes
- generator/auditor/oracle split leakage
- replayed or cloned round templates
- scripts that could be mistaken for current thesis evidence
- runtime artifacts or logs that could leak into the repository

## Current Canonical Path

Use `scripts/export_synthetic_alg_benchmark.py` for Chapter 4 evidence. It exports:

- `synthetic_alg_main_result_table.csv`
- `synthetic_alg_archetype_generation_counts.csv`
- `synthetic_alg_l1_routing_by_archetype_reason.csv`
- `synthetic_alg_l2_confusion_matrix.csv`
- `synthetic_alg_rarity_discovery_metrics.csv`
- `synthetic_alg_fraud_survival_by_family.csv`
- `synthetic_alg_energy_attack_validation.csv`
- `synthetic_alg_dataset_isolation_config.json`
- `synthetic_alg_run_config.json`

Canonical command:

```bash
BATCH_SIZE=24 VEHICLE_POOL_SIZE=128 python scripts/export_synthetic_alg_benchmark.py \
  --rounds 24 \
  --cycle-rounds 12 \
  --pretrain-epochs 5 \
  --seeds 20260318,20260319,20260320,20260321,20260322 \
  --recheck-values 0.00,0.05,0.10,0.20,0.30 \
  --output-dir results/synthetic_alg_artifacts
```

The `BATCH_SIZE` and `VEHICLE_POOL_SIZE` environment variables are part of the
paper configuration. Omitting them uses the interactive demo defaults rather
than the thesis benchmark scale.

## Findings and Actions

### Ground Truth

Status: fixed.

`DemoDataGenerator` now defaults to `ground_truth_mode="archetype"`. Each generated update carries an archetype-derived `ground_truth_label`; `preflight_role` remains only diagnostic metadata from a local audit pass.

The old `preflight` mode still exists behind `DEMO_GROUND_TRUTH_MODE=preflight` for debugging, but it is not the default and must not be used for thesis metrics.

### Split Isolation

Status: fixed in the synthetic ALG artifact path and dashboard baseline path.

The synthetic ALG path uses:

- `D_proto_*` for generator directions
- `D_audit_*` for L2 audit decisions
- `D_oracle_*` for reported accuracies

The backend Data Analysis baseline path was also moved onto the same proto/audit/oracle split layout so it no longer evaluates and audits on the generator's own datasets.

### Oracle Feedback Leakage

Status: fixed in the synthetic ALG artifact path.

`scripts/export_synthetic_alg_benchmark.py` now disables policy adaptation while producing Chapter 4 evidence. Oracle splits are report-only and do not feed threshold updates. This keeps `p=0.0` and `p=0.10` comparisons focused on audit visibility and L2 verdict behavior.

### Round Replay

Status: fixed in current exporters.

The active artifact builders regenerate every requested round. `cycle_rounds` is retained as a phase-cycle/reporting parameter, not as a clone template size.

### Runtime Artifacts

Status: fixed.

`.gitignore` now excludes generated experiment outputs, logs, runtime state, and Redis dump files:

- `results/`
- `logs/`
- `backend/logs/`
- `runtime_state/`
- `dump.rdb`

## Legacy Code to Treat Carefully

These files are useful for development or historical comparison, but they are not the canonical Chapter 4 evidence path:

- `scripts/legacy/run_unified_benchmark.py`
- `scripts/legacy/run_fedavg_baseline.py`
- `scripts/legacy/evaluate_system.py`
- old folders under `results/thesis_artifacts*`
- old `results/unified_benchmark_*` and `results/fedavg_baseline_*` JSON files

If thesis tables are regenerated, prefer `results/synthetic_alg_artifacts` and do not mix it with the old result folders.

## No Confirmed Falsification Found

I did not find scripts that directly overwrite final metrics with fabricated constants or relabel benchmark ground truth from L2 verdicts in the current synthetic ALG path. The benchmark is still synthetic by design, so thesis wording should present it as a controlled server-side gradient-auditing benchmark rather than a real-client federated-learning benchmark.

## Remaining Development Placeholders

These are disclosed development placeholders rather than hidden data leaks:

- `backend/l3_gatekeeper/validator.py` creates a placeholder golden dataset when no artifacts exist, and the API reports that status.
- `docker-compose.yml` includes development credentials such as the local Grafana admin password and Ganache mnemonic. These are acceptable for local demo use but should not be used for deployment.

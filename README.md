# CornerDrive

CornerDrive is a reproducibility repository for a federated-learning thesis on
rarity-preserving update auditing in Internet of Vehicles settings. The core
method combines:

- L1 routing: cheap gradient screening before expensive audit.
- L2 dual-loss audit: classify suspect updates with main-task and corner-case
  loss drift.
- Controlled synthetic ALG simulations plus real-data client-gradient benchmarks.

This repository is intended to let a reader clone the project, install the
environment, run the documented commands, and regenerate the main thesis tables.

## What Can Be Reproduced

| Thesis item | Reproduction output |
|---|---|
| Real-gradient method comparison, Table 5.1 | `artifacts/tables/table_5_1_real_gradient_macro.csv` |
| CornerDrive real-gradient dataset breakdown, Table 5.2 | `artifacts/tables/table_5_2_cornerdrive_real_gradient_by_dataset.csv` |
| Synthetic ALG main result and recheck sweep | `results/audit_reproduction/synthetic_alg_benchmark_b24/*.csv` |
| L1 routing and L2 confusion appendix tables | `results/audit_reproduction/synthetic_alg_benchmark_b24/*.csv` |
| Rarity-overlap and proxy stress tests | `artifacts/tables/appendix_rarity_overlap.csv`, `artifacts/tables/appendix_proxy_sensitivity.csv` |
| Corner-family divergence stress test | `artifacts/tables/appendix_corner_family_divergence.csv` |
| Corner-harm threshold calibration | `artifacts/tables/appendix_corner_harm_threshold_calibration.csv` |
| Audit cost frontier | `results/cost_performance_frontier/audit_cost_frontier.svg` |
| Real-gradient dev/test calibration split | `results/real_gradient_calibration_split/real_gradient_calibration_split_summary.csv` |
| 50-round real-gradient FL learning curve | `results/real_gradient_learning_curve/real_gradient_learning_curve.svg` |

The compact expected-value manifest is `results/expected_results.csv`.

## Environment

Tested environment:

- Python 3.12
- PyTorch 2.8.0
- torchvision 0.23.0
- CPU execution is sufficient for the thesis reproduction scripts.

Install with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

or with conda:

```bash
conda env create -f environment.yml
conda activate cornerdrive-repro
```

If you want CUDA acceleration, install the PyTorch wheel matching your CUDA
runtime before running the experiments.

## Dataset

Data files are not tracked in Git. Runtime datasets belong under `data/real/`.

Prepare MNIST and FashionMNIST:

```bash
python scripts/prepare_data.py --download-torchvision
```

FEMNIST/LEAF is expected under:

```text
data/real/femnist/train/
data/real/femnist/test/
```

See `data/README.md` for details about MNIST, FashionMNIST, FEMNIST/LEAF,
optional BDD100K calibration, and how `D_main` and `D_corner` are constructed.
The real-gradient benchmark keeps client/update data, audit/reference data, and
final evaluation data on deterministic separate surfaces where the source
provides enough data.

## Reproduce Main Results

### Quick thesis smoke check

This regenerates the synthetic ALG main result, recheck sweep, and paper-facing CSV
tables. It is the fastest thesis-critical check.

```bash
bash scripts/reproduce_all.sh main
```

Important: the thesis synthetic ALG setting is:

```bash
BATCH_SIZE=24
VEHICLE_POOL_SIZE=128
```

`scripts/reproduce_all.sh` sets those defaults explicitly. Running
`scripts/export_synthetic_alg_benchmark.py` without them uses the demo defaults
(`BATCH_SIZE=96`, `VEHICLE_POOL_SIZE=384`) and will not reproduce the thesis
numbers.

Equivalent explicit command:

```bash
BATCH_SIZE=24 VEHICLE_POOL_SIZE=128 python scripts/export_synthetic_alg_benchmark.py \
  --seeds 20260318,20260319,20260320,20260321,20260322 \
  --recheck-values 0.00,0.05,0.10,0.20,0.30 \
  --output-dir results/audit_reproduction/synthetic_alg_benchmark_b24
```

### Appendix stress tables

```bash
bash scripts/reproduce_all.sh appendix
```

This runs:

- `scripts/export_synthetic_stress_tests.py`
- `scripts/export_corner_family_divergence.py`
- `scripts/export_corner_harm_threshold_calibration.py`

### Real-gradient reliability benchmark

```bash
bash scripts/reproduce_all.sh real-gradient
```

Equivalent explicit command:

```bash
python scripts/export_real_gradient_reliability_benchmark.py \
  --sources mnist,fashionmnist,femnist \
  --seeds 20260527,20260528,20260529,20260530,20260531,20260532,20260533,20260534,20260535,20260536,20260537,20260538,20260539,20260540,20260541,20260542,20260543,20260544,20260545,20260546 \
  --download \
  --max-clients 120 \
  --min-samples-per-client 8 \
  --max-samples-per-client 48 \
  --clients-per-round 20 \
  --rounds 10 \
  --pretrain-steps 50 \
  --local-batch-size 16 \
  --reference-split-fraction 0.50 \
  --max-reference-samples 4096 \
  --max-evaluation-samples 4096 \
  --output-dir results/real_gradient_reliability_calibrated_holdout
```

If FEMNIST is not prepared yet, run only the torchvision sources first:

```bash
REAL_SOURCES=mnist,fashionmnist bash scripts/reproduce_all.sh real-gradient
```

### Journal extension artifacts

These are slower or more appendix-oriented than the core thesis reproduction,
so they are explicit modes rather than part of the default `all` run.

```bash
bash scripts/reproduce_all.sh frontiers
bash scripts/reproduce_all.sh learning-curve
```

`frontiers` regenerates the audit cost frontier and the real-gradient dev/test
calibration split. `learning-curve` regenerates the 50-round end-to-end
real-gradient FL proxy with five seeds by default. Override
`LEARNING_CURVE_ROUNDS` and `LEARNING_CURVE_SEEDS` for larger journal-scale
sweeps.

### Full reproduction

```bash
bash scripts/reproduce_all.sh all
```

This can take substantially longer because it reruns real-gradient, ALG main,
stress tests, divergence, threshold calibration, and table generation.

## Expected Results

Representative expected values:

| Item | Expected |
|---|---:|
| Real-gradient calibrated CornerDrive macro fraud survival | 0.0013 |
| Real-gradient calibrated CornerDrive macro rarity retention | 0.3767 |
| Real-gradient calibrated CornerDrive macro corner accuracy | 0.7130 |
| ALG CornerDrive p=0.10 main accuracy | 85.58% ± 0.55 |
| ALG CornerDrive p=0.10 corner accuracy | 61.24% ± 0.53 |
| ALG CornerDrive p=0.10 sign-flip survival | 0.00% ± 0.00 |
| ALG CornerDrive p=0.10 corner-harm survival | 84.00% ± 5.48 |
| Rarity-overlap baseline recognition | 100.00% |
| Random main proxy rarity recognition | 0.00% |

See `results/expected_results.csv` for tolerances and source files.

## Generate Paper Tables From Existing Results

If result CSVs already exist, regenerate paper-facing tables without rerunning
the experiments:

```bash
python scripts/make_paper_tables.py
```

Outputs are written to `artifacts/tables/`.

## Repository Structure

| Path | Purpose |
|---|---|
| `backend/common` | shared schemas, config, and utilities |
| `backend/l1_linear_defense` | L1 gradient screening and routing |
| `backend/l2_dual_audit` | L2 dual-loss audit logic |
| `backend/l3_gatekeeper` | validation library code |
| `backend/l4_settlement` | settlement API/dashboard code |
| `backend/policy_agent` | policy service and benchmark analysis |
| `backend/tests` | backend and integration tests |
| `configs` | thesis reproduction parameter manifests |
| `data` | dataset instructions; generated data is ignored |
| `results` | generated experiment outputs; mostly ignored |
| `artifacts/tables` | regenerated paper-facing CSV tables |
| `scripts` | data prep, reproduction, exporters, and table builders |
| `docs` | architecture, formulas, reports, and benchmark notes |
| `contracts` | Solidity settlement/policy commitment sources |
| `frontend` | React dashboard for the full FLPG demo |

## Tests

Run the core tests used for the reproduction sanity check:

```bash
python -m pytest \
  backend/tests/test_baseline_analysis.py \
  backend/tests/test_l1v4_router.py \
  backend/tests/test_real_gradient_bdd100k.py \
  -q
```

## Notes on Baselines and Ablations

The main exporters include FedAvg, GeoMed, Multi-Krum, FLTrust, Zeno, Zeno++,
CornerDrive, main-only audit, corner-only audit, exhaustive L2, L1 ablations,
and operating-curve scripts. See:

- `scripts/export_synthetic_alg_benchmark.py`
- `scripts/export_exhaustive_l2_audit.py`
- `scripts/export_l1_l2_operating_curve.py`
- `scripts/export_layer_cost_profile.py`

## Citation

See `CITATION.cff`.

## License

MIT. See `LICENSE`.

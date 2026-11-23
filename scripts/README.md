# Scripts Index

This directory contains the command-line entry points for local operation,
thesis reproduction, and diagnostic exports. Prefer the wrappers in the
top-level sections before running individual exporters directly.

## Local Operation

| Script | Purpose |
| --- | --- |
| `setup.sh` | Create local Python, frontend, and contract dependency environments. Keeps an existing `.env` intact. |
| `run_demo.sh` | Start the local demo stack and simulated gradient stream. |
| `verify_all.sh` | Run the full project verification suite: backend tests, expected-result check, frontend build/lint, production dependency audit, and contract tests. |
| `prepare_data.py` | Download/check public datasets under `data/real/`. |

## Reproduction Wrappers

| Script | Purpose |
| --- | --- |
| `reproduce_all.sh` | Canonical thesis reproduction wrapper. Use its modes (`main`, `appendix`, `real-gradient`, `frontiers`, `learning-curve`, `all`) rather than hand-assembling long commands. |
| `make_paper_tables.py` | Rebuild paper-facing CSV tables in `artifacts/tables/` from existing result directories. |
| `check_expected_results.py` | Validate regenerated paper-facing tables against `results/expected_results.csv` tolerances. |

## Current Exporters

| Script | Category | Purpose |
| --- | --- | --- |
| `export_real_gradient_reliability_benchmark.py` | Primary real-gradient evidence | Multi-source, multi-seed reliability benchmark used for the main thesis tables. |
| `export_real_gradient_benchmark.py` | Real-gradient diagnostic | Single benchmark run/export path for focused calibration or smoke runs. |
| `export_real_gradient_calibration_split.py` | Calibration evidence | Converts threshold sweeps into dev/test split tables. |
| `export_real_gradient_learning_curve.py` | Extended evidence | Multi-round real-gradient FL learning curve artifacts. |
| `export_synthetic_alg_benchmark.py` | Mechanism evidence | Canonical synthetic ALG benchmark for controlled archetype behavior. |
| `export_synthetic_stress_tests.py` | Appendix evidence | Rarity-overlap and proxy-sensitivity stress tables. |
| `export_corner_family_divergence.py` | Appendix evidence | Corner-family divergence stress table. |
| `export_corner_harm_threshold_calibration.py` | Appendix evidence | Corner-harm threshold calibration table. |
| `export_l1_l2_operating_curve.py` | Diagnostic | L1 review-cost and L2 threshold operating curves. |
| `export_exhaustive_l2_audit.py` | Diagnostic | Exhaustive L2 audit comparison on fixed generated rounds. |
| `export_cost_performance_frontier.py` | Diagnostic | Cost/performance SVG and table from the recheck sweep. |
| `export_layer_cost_profile.py` | Diagnostic | Layer-by-layer runtime/cost profiling. |
| `export_synthetic_real_gradient_calibration.py` | Diagnostic | Scalar comparison between synthetic ALG and real client gradients. |
| `export_reputation_accumulation_simulation.py` | Design validation | L4 reputation accumulation simulation table. |

## CIFAR-10 Dual-Dataset Rarity Benchmark

A self-contained GPU benchmark (`backend/dual_dataset_benchmark/`, torch and numpy only).
Design and results: `docs/benchmarks/CIFAR10_DUAL_DATASET_BENCHMARK.md`.

| Script | Purpose |
| --- | --- |
| `annotate_cifar10_golden.py` | Annotate a quarter of CIFAR-10 as the golden dataset (normal set and rare set); the rest stays unannotated for testing. |
| `extract_cifar10_embeddings.py` | Embed all images with a frozen ImageNet ResNet-50 (downloads the torchvision weights on first use). |
| `run_cifar10_dual_dataset.py` | Resumable launcher: one unit per (seed, scenario, method). |
| `calibrate_cifar10_dual_dataset.py` | Freeze the CornerDrive thresholds from development seeds. |
| `analyze_cifar10_dual_dataset.py` | Method comparison, rarity discovery, confusion, routing and detector tables. |
| `plot_cifar10_dual_dataset_plane.py` | SVG of the verdict plane, one panel per update type. |
| `audit_cifar10_dual_dataset_leakage.py` | Leakage audit: disjoint surfaces, identical and near-duplicate images across surfaces, clean client shards. |

## Shared Helpers

| Script | Role |
| --- | --- |
| `generate_demo_data.py` | Demo-data and synthetic-gradient generation engine used by the live demo and several exporters. |
| `export_thesis_artifacts.py` | Shared synthetic export utilities and lower-level artifact builder. Prefer `export_synthetic_alg_benchmark.py` for current thesis reproduction. |

## Legacy

The `legacy/` directory keeps older benchmark CLIs for historical comparison.
They are not current thesis evidence and should not be used in new reproduction
instructions.

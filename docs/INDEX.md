# FLPG Documentation Index

This directory is the canonical home for project documentation.

Use the root `README.md` for quick onboarding. Use the documents here for architecture, API, implementation, and formula details.

## Start Here

- [Project README](../README.md) - quick start, repository map, and primary navigation
- [Thesis PDF](thesis/SAT301_FYP_Dissertation_CornerDrive.pdf) - submitted dissertation
- [Thesis Index](thesis/README.md) - dissertation metadata and related revision notes
- [Implementation Guide](implementation/IMPLEMENTATION.md) - what is currently implemented and how to run it locally
- [Ports and APIs](api/PORTS_AND_APIS.md) - canonical service ports and endpoint paths
- [Scripts Index](../scripts/README.md) - canonical, diagnostic, and legacy command-line entry points
- [Maintenance Audit](MAINTENANCE_AUDIT.md) - repository hygiene findings and cleanup rules
- [Reproducibility Audit](REPRODUCIBILITY_AUDIT.md) - reviewer protocol, result checks, and remaining reproducibility risks

## Architecture

- [System Architecture](architecture/ARCHITECTURE.md)
- [L1: Linear Defense](architecture/L1_LINEAR_DEFENSE.md)
- [L2: Dual Audit](architecture/L2_DUAL_AUDIT.md)
- [L3: Gatekeeper](architecture/L3_GATEKEEPER.md)
- [L4: Settlement](architecture/L4_SETTLEMENT.md)

## API Reference

- [Ports and APIs](api/PORTS_AND_APIS.md)

## Implementation Notes

- [Implementation Guide](implementation/IMPLEMENTATION.md)
- [Maintenance Audit](MAINTENANCE_AUDIT.md)
- [Reproducibility Audit](REPRODUCIBILITY_AUDIT.md)
- [L1/L2 Operating-Curve Benchmark](benchmarks/L1_L2_OPERATING_CURVE.md)
- [Real-Data Gradient Benchmark](benchmarks/REAL_GRADIENT_BENCHMARK.md)
- [Real-Gradient Calibration Split](benchmarks/REAL_GRADIENT_CALIBRATION_SPLIT.md)
- [Real-Gradient FL Learning Curve](benchmarks/REAL_GRADIENT_FL_LEARNING_CURVE.md)
- [Synthetic-Real Gradient Calibration](benchmarks/SYNTHETIC_REAL_GRADIENT_CALIBRATION.md)
- [CIFAR-10 Dual-Dataset Rarity Benchmark](benchmarks/CIFAR10_DUAL_DATASET_BENCHMARK.md) - golden-dataset annotation, FedVal baseline, embedding model in the loop

## Math and Theory

- [Mathematical Formulas](formulas/MATHEMATICAL_FORMULAS.md)

## Thesis

- [Submitted Dissertation PDF](thesis/SAT301_FYP_Dissertation_CornerDrive.pdf)
- [Thesis Document Index](thesis/README.md)
- [Real-Gradient Thesis Revision Plan](thesis/REAL_GRADIENT_THESIS_REVISION_PLAN.md)
- [Revised Thesis Audit](thesis/REVISED_THESIS_AUDIT.md)

## Reports

- [Reports Index](reports/README.md)
- [Synthetic ALG Code Audit and Cleanup](reports/SYNTHETIC_ALG_CODE_AUDIT_AND_CLEANUP.md)
- [Legacy Reports](reports/legacy/README.md) - historical notes only, not current thesis evidence

## Operations

- [Scripts Index](../scripts/README.md)
- [Ports and APIs](api/PORTS_AND_APIS.md)
- [Implementation Guide](implementation/IMPLEMENTATION.md)

## Documentation Rules

- Root `README.md` stays short and onboarding-focused.
- Canonical HTTP paths live in `docs/api/`.
- Layer behavior and design rationale live in `docs/architecture/`.
- Operational setup and implementation status live in `docs/implementation/`.
- New scripts and dated reports should be added to their local index files.

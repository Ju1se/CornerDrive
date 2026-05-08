# FLPG Documentation Index

This directory is the canonical home for project documentation.

Use the root `README.md` for quick onboarding. Use the documents here for architecture, API, implementation, and formula details.

## Start Here

- [Project README](../README.md) - quick start, repository map, and primary navigation
- [Implementation Guide](implementation/IMPLEMENTATION.md) - what is currently implemented and how to run it locally
- [Ports and APIs](api/PORTS_AND_APIS.md) - canonical service ports and endpoint paths

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
- [L1/L2 Operating-Curve Benchmark](benchmarks/L1_L2_OPERATING_CURVE.md)
- [Real-Data Gradient Benchmark](benchmarks/REAL_GRADIENT_BENCHMARK.md)

## Math and Theory

- [Mathematical Formulas](formulas/MATHEMATICAL_FORMULAS.md)

## Reports

- [V2.5 Code Audit and Cleanup](reports/V25_CODE_AUDIT_AND_CLEANUP.md)
- [Legacy Reports](reports/legacy/README.md) - historical notes only, not current thesis evidence

## Documentation Rules

- Root `README.md` stays short and onboarding-focused.
- Canonical HTTP paths live in `docs/api/`.
- Layer behavior and design rationale live in `docs/architecture/`.
- Operational setup and implementation status live in `docs/implementation/`.

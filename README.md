# Federated Learning Privacy Game (FLPG)

FLPG is a layered federated-learning security stack for Internet of Vehicles scenarios. The repository contains backend services for screening and audit, a policy control plane, a React dashboard, and Solidity contracts for settlement and policy commitment.

## Quick Start

```bash
cp .env.example .env
./scripts/setup.sh
./scripts/run_demo.sh
```

Default local endpoints:

- Frontend: `http://localhost:3000`
- L1 Linear Defense: `http://localhost:8081`
- L4 Settlement Dashboard: `http://localhost:8082`
- Policy Agent: `http://localhost:8083`

## Repository Map

| Path | Purpose |
| --- | --- |
| `backend/common` | shared config, schemas, helpers |
| `backend/l1_linear_defense` | gradient intake, batching, suspect filtering |
| `backend/l2_dual_audit` | async audit worker and classification |
| `backend/l3_gatekeeper` | validation library code |
| `backend/l4_settlement` | dashboard and settlement API |
| `backend/policy_agent` | adaptive policy service |
| `backend/tests` | backend and integration tests |
| `frontend/src` | React application |
| `contracts/contracts` | Solidity sources |
| `contracts/scripts` | deployment scripts |
| `docs` | architecture, API, implementation, and formulas |
| `scripts` | setup, unified local startup, demo generation, and V2.5 artifact export |

Local artifact directories such as `backend/venv`, `frontend/node_modules`, and `frontend/dist` are runtime or build outputs, not primary source roots.

## Documentation

- [Documentation Index](docs/INDEX.md)
- [Implementation Guide](docs/implementation/IMPLEMENTATION.md)
- [Ports and APIs](docs/api/PORTS_AND_APIS.md)
- [System Architecture](docs/architecture/ARCHITECTURE.md)
- [Mathematical Formulas](docs/formulas/MATHEMATICAL_FORMULAS.md)
- [V2.5 Code Audit and Cleanup](docs/reports/V25_CODE_AUDIT_AND_CLEANUP.md)

## Thesis Artifacts

Use the V2.5 exporter for Chapter 4 benchmark evidence:

```bash
python scripts/export_v25_artifacts.py --rounds 24 --cycle-rounds 12 --pretrain-epochs 5 --output-dir results/v25_artifacts
```

Additional reproducibility exporters:

| Script | Purpose |
| --- | --- |
| `scripts/export_l1v3_ablation.py` | M0-M4 L1 visibility-router ablation |
| `scripts/export_v25_stress_tests.py` | rarity/proxy/threshold stress-test tables |
| `scripts/export_corner_family_divergence.py` | corner-family divergence rho sweep |
| `scripts/export_exhaustive_l2_audit.py` | full-visibility L2 upper-bound ablation |
| `scripts/export_layer_cost_profile.py` | L1+L2 vs Exhaustive L2 cost profile |
| `scripts/export_reputation_accumulation_simulation.py` | L4 reputation accumulation simulation |
| `scripts/export_corner_harm_threshold_calibration.py` | corner-harm threshold robustness table |

Older benchmark scripts and reports live under `scripts/legacy/` and
`docs/reports/legacy/`; they are retained for design history only.

Layer-specific references:

- [L1: Linear Defense](docs/architecture/L1_LINEAR_DEFENSE.md)
- [L2: Dual Audit](docs/architecture/L2_DUAL_AUDIT.md)
- [L3: Gatekeeper](docs/architecture/L3_GATEKEEPER.md)
- [L4: Settlement](docs/architecture/L4_SETTLEMENT.md)

## Notes

- `L3` is implemented as library code and is not the default live service in the current stack.
- Canonical endpoint naming now lives under `docs/api/` so API paths have a single reference point.

## License

MIT License

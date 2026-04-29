# FLPG Implementation Guide

This document describes the implemented parts of the current FLPG stack and how to run them locally.

For architecture details, use `../architecture/`.
For canonical endpoint names, use `../api/PORTS_AND_APIS.md`.

## Current Scope

Implemented components:

- `L1` linear defense service for gradient intake, screening, and suspect routing
- `L2` dual-purpose audit worker for fraud, beneficial rarity, honest, and noise classification
- `L3` validation logic as library code, not a default live service
- `L4` dashboard and settlement API
- `Policy Agent` for policy proposal, activation, and telemetry history
- `Frontend` dashboard for system views and policy controls
- `Contracts` for audit settlement and policy commitment

## Local Startup

### Prerequisites

- Python `3.11+`
- Node.js `20+`
- Redis
- Ganache CLI or Docker

### Setup

```bash
cp .env.example .env
./scripts/setup.sh
```

### Start the stack

```bash
./scripts/run_demo.sh
```

This is now the single local startup entrypoint. It starts the services and the simulated gradient stream together.

## Service Entry Points

- Frontend: `http://localhost:3000`
- L1 Linear Defense: `http://localhost:8081`
- L4 Settlement Dashboard: `http://localhost:8082`
- Policy Agent: `http://localhost:8083`
- Ganache RPC: `http://localhost:8545`
- Grafana: `http://localhost:3001`
- Prometheus: `http://localhost:9090`

## Repository Mapping

| Path | Role |
| --- | --- |
| `backend/common` | shared config, schemas, and policy loading |
| `backend/l1_linear_defense` | intake API and aggregation logic |
| `backend/l2_dual_audit` | async audit worker and classification |
| `backend/l3_gatekeeper` | validation library and model drift checks |
| `backend/l4_settlement` | dashboard and settlement API |
| `backend/policy_agent` | adaptive policy control plane |
| `backend/tests` | backend and integration tests |
| `frontend/src` | React UI |
| `contracts/contracts` | Solidity contracts |
| `contracts/scripts` | deployment scripts |
| `scripts` | setup, unified local startup, demo generation, and evaluation helpers |

## Layer Summary

### L1

- Uses geometric median and cosine deviation scoring
- Accepts gradient submissions through FastAPI
- Queues suspect updates for L2 audit

### L2

- Evaluates suspect gradients against main and corner datasets
- Produces `FRAUD`, `RARITY`, `HONEST`, or `NOISE`
- Stores audit outcomes in Redis for L4 and dashboard use

### L3

- Applies aggregated updates to a copy of the model
- Validates drift on a golden dataset
- Approves or rejects updates before commit

### L4

- Reads system statistics, vehicle views, and recent audits
- Exposes the live `L3` dataset-status read surface
- Fans batch settlement requests into per-vehicle contract calls

### Policy Agent

- Stores current and next policy in Redis
- Proposes next policy from telemetry using GLM or rule fallback
- Exposes proposal history, explanations, telemetry, and diffs

## Testing

Run the backend test suite from `backend/`:

```bash
cd backend
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$(pwd) python3 -m pytest tests/test_policy_agent.py -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$(pwd) python3 -m pytest tests/test_integration/test_full_flow.py -q
```

## Operational Helpers

- `./scripts/setup.sh` installs local dependencies
- `./scripts/run_demo.sh` starts the local stack and generates simulated gradients in one command

## Notes

- `L3` is implemented as reusable logic and may be embedded instead of run as a separate service.
- `frontend/dist`, `frontend/node_modules`, and `backend/venv` are local build/runtime artifacts, not primary source directories.
- Root documentation files are intentionally kept short; detailed guidance lives under `docs/`.

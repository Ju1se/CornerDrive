# Ports and APIs

This document defines the current FLPG port map and the canonical interface rules for the running stack.

This file is the single source of truth for service ports, browser proxy paths, and canonical endpoint names.

## Port Map

| Service | Default Port | Purpose |
| --- | --- | --- |
| Frontend (Vite) | `3000` | Operator UI and browser-side API proxy |
| Redis | `6379` | Shared state store and Celery broker backend |
| Ganache | `8545` | Local EVM JSON-RPC |
| L1 Linear Defense | `8081` | Gradient intake and suspect screening |
| L4 Settlement Dashboard | `8082` | Read APIs, vehicle views, settlement fan-out |
| Policy Agent | `8083` | Policy control plane and GLM/rule proposals |
| Grafana | `3001` | Optional monitoring UI |
| Prometheus | `9090` | Optional monitoring backend |

## Canonical Interface Rules

1. Service health and ops endpoints live at the service root:
   - `GET /health`
   - `GET /metrics`
   - `GET /docs`
2. Resource APIs live under `/api/v1/...`.
3. Frontend never talks to backend ports directly in browser mode. It uses these proxy prefixes:
   - `/api/l1` -> `http://localhost:8081`
   - `/api/l4` -> `http://localhost:8082`
   - `/api/policy` -> `http://localhost:8083/api/v1/policy`
4. Canonical routes below are the supported interfaces for the current stack.

## Frontend Proxy Map

| Browser Path | Target Service Path |
| --- | --- |
| `/api/l1/health` | `GET http://localhost:8081/health` |
| `/api/l4/health` | `GET http://localhost:8082/health` |
| `/api/l4/api/v1/...` | `GET/POST http://localhost:8082/api/v1/...` |
| `/api/policy/health` | `GET http://localhost:8083/api/v1/policy/health` |
| `/api/policy/current` | `GET http://localhost:8083/api/v1/policy/current` |
| `/api/policy/...` | `GET/POST http://localhost:8083/api/v1/policy/...` |

## L1 Linear Defense (`8081`)

Ops endpoints:
- `GET /health`
- `GET /metrics`

Business endpoints:
- `POST /api/v1/gradients`
- `POST /api/v1/batches/process`

Notes:
- This service is intake-oriented and protected by `X-API-Key`.
- Gradient submissions reject malformed vehicle addresses, non-finite values, oversized vectors, and mixed-dimensional immediate batches before aggregation.

## L4 Settlement Dashboard (`8082`)

Ops endpoints:
- `GET /health`
- `GET /metrics`

Canonical API endpoints:
- `GET /api/v1/stats`
- `GET /api/v1/l3/status`
- `GET /api/v1/tiers`
- `GET /api/v1/recent-audits`
- `GET /api/v1/vehicles`
- `GET /api/v1/vehicle/{address}`
- `POST /api/v1/settle/batch`

Notes:
- `POST /api/v1/settle/batch` is protected by `X-API-Key` because it signs and submits settlement transactions through the configured oracle key.
- `GET /api/v1/l3/status` is the live status surface for the library-only L3 gatekeeper. It reports whether the golden dataset is loaded from disk or falling back to placeholder data.

## Policy Agent (`8083`)

Ops endpoints:
- `GET /health`
- `GET /metrics`
- `GET /api/v1/llm/stats`

Canonical policy endpoints:
- `GET /api/v1/policy/health`
- `GET /api/v1/policy/current`
- `GET /api/v1/policy/next`
- `GET /api/v1/policy/proposal/latest`
- `GET /api/v1/policy/proposal/{round_id}`
- `GET /api/v1/policy/history`
- `GET /api/v1/policy/history/{round_id}`
- `GET /api/v1/policy/explanation/{round_id}`
- `POST /api/v1/policy/propose`
- `POST /api/v1/policy/activate`
- `GET /api/v1/policy/telemetry/latest`
- `GET /api/v1/policy/telemetry/{round_id}`
- `GET /api/v1/policy/telemetry`
- `POST /api/v1/policy/telemetry`
- `GET /api/v1/policy/diff/{round_a}/{round_b}`
- `GET /api/v1/policy/glm-decisions`

Compatibility endpoint:
- `POST /api/v1/policy/explain`

Notes:
- Use `/health` for service uptime checks.
- Use `/api/v1/policy/health` when the caller is already operating inside the policy namespace.
- Policy writes (`POST /propose`, `POST /activate`, and `POST /telemetry`) require `X-API-Key`; read endpoints remain unauthenticated for dashboard use.

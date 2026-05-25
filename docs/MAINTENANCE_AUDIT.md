# Maintenance Audit

Date: 2026-05-25

This audit focused on repository hygiene rather than experiment validity. The
goal was to identify stale documentation, duplicate entry points, generated
artifacts in the working tree, and script surfaces that made the project harder
to navigate.

## Current Structure

| Area | Canonical Location | Notes |
| --- | --- | --- |
| Quick onboarding | `README.md` | Short project map, setup, reproduction, verification, and monitoring notes. |
| Detailed implementation | `docs/implementation/IMPLEMENTATION.md` | Current stack scope and local startup details. |
| API reference | `docs/api/PORTS_AND_APIS.md` | Single source of truth for ports and service paths. |
| Script catalog | `scripts/README.md` | Canonical vs diagnostic vs legacy script map. |
| Reviewer reports | `docs/reports/README.md` | Current report map and supersession notes. |
| Full verification | `scripts/verify_all.sh` | Backend, frontend, production dependency audit, and contract checks. |
| Lightweight tree check | `verify_implementation.py` | Fast syntax/file-presence check only; not a full verification replacement. |

## Findings

### Documentation

- `docs/architecture/PORTS_AND_APIS.md` duplicates the old location of the API
  reference, but it already acts as a compatibility pointer. Keep it as a thin
  redirect so older links do not break.
- `README_FLPG_IMPLEMENTATION.md` is also a compatibility pointer to
  `docs/implementation/IMPLEMENTATION.md`. It is redundant by design and should
  stay short.
- `docs/reports/legacy/README.md` referenced outdated V25 paths that no longer
  match the current report names. It was updated to point readers to the current
  reports index and current synthetic audit note.
- `docs/reports/` lacked an index, which made dated reports look equally
  current. `docs/reports/README.md` now separates current reviewer-facing
  evidence from historical notes.
- `docs/INDEX.md` did not expose the scripts index, reports index, monitoring
  notes, or this maintenance audit. It now serves as the top-level map.

### Scripts

- `scripts/legacy/` is intentionally retained for historical comparison.
  Current reproduction should go through `scripts/reproduce_all.sh` or the
  current exporters listed in `scripts/README.md`.
- `scripts/export_thesis_artifacts.py` is a shared low-level helper. The
  canonical synthetic thesis entry point is `scripts/export_synthetic_alg_benchmark.py`.
- `verify_implementation.py` overlaps with `scripts/verify_all.sh`, but the two
  have different scopes: quick source-tree sanity check vs full verification.
  Documentation now names `scripts/verify_all.sh` as the canonical full check.

### Generated Files

The following are generated artifacts and should remain out of version control:

- Python `__pycache__/` directories under `backend/` and `scripts/`
- `.pytest_cache/`
- `frontend/dist/`
- `contracts/cache/`

These paths are already covered by `.gitignore`; they were removed from the
working tree during this cleanup.

### Remaining Debt

- Frontend production dependency audit is clean, but development tooling still
  reports audit findings in the Vite/ESLint dependency chain. Fixing that
  cleanly requires a separate Vite and ESLint major-version migration.
- Several architecture documents contain illustrative code snippets that are
  broader than the current implementation. They are useful for design context,
  but the implementation guide should remain the authority for what is actually
  deployed.
- Large local `data/` and `results/` directories remain intentionally outside
  source control. For long-term archival, consider a DVC or manifest-based data
  package rather than committing generated results.

## Cleanup Rules Going Forward

1. Add new runnable workflows to `scripts/README.md`.
2. Add new reviewer-facing notes to `docs/reports/README.md` and state whether
   they supersede earlier reports.
3. Keep root-level documents short; place detailed material under `docs/`.
4. Prefer `scripts/verify_all.sh` before sharing or submitting the project.
5. Do not cite files under `scripts/legacy/` or `docs/reports/legacy/` as current
   evidence without a newer report explicitly elevating them.

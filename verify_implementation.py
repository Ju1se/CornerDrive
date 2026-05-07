#!/usr/bin/env python3
"""
Lightweight verification for the current FLPG project layout.

This script avoids importing heavy runtime dependencies and instead verifies:
1. Expected source files exist
2. Python modules compile syntactically
3. Frontend and contract entrypoints are present
"""

from __future__ import annotations

import py_compile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


REQUIRED_FILES = [
    "backend/common/config.py",
    "backend/common/policy_loader.py",
    "backend/l1_linear_defense/aggregation.py",
    "backend/l1_linear_defense/server.py",
    "backend/l2_dual_audit/classifier.py",
    "backend/l2_dual_audit/worker.py",
    "backend/l3_gatekeeper/validator.py",
    "backend/l4_settlement/dashboard_api.py",
    "backend/policy_agent/main.py",
    "backend/policy_agent/api/routes_policy.py",
    "frontend/src/App.tsx",
    "frontend/src/pages/PolicyDashboard.tsx",
    "contracts/contracts/FLPGAudit.sol",
    "contracts/contracts/PolicyRegistry.sol",
    "contracts/scripts/deploy.js",
    "contracts/scripts/deploy_policy_registry.js",
    "docker-compose.yml",
    "scripts/generate_demo_data.py",
]


PYTHON_FILES = [
    "backend/common/config.py",
    "backend/common/policy_loader.py",
    "backend/common/schemas/policy.py",
    "backend/common/schemas/telemetry.py",
    "backend/common/schemas/proposal.py",
    "backend/l1_linear_defense/aggregation.py",
    "backend/l1_linear_defense/server.py",
    "backend/l2_dual_audit/classifier.py",
    "backend/l2_dual_audit/worker.py",
    "backend/l3_gatekeeper/validator.py",
    "backend/l4_settlement/dashboard_api.py",
    "backend/policy_agent/main.py",
    "backend/policy_agent/api/routes_policy.py",
    "backend/policy_agent/tasks/round_close.py",
]


def check_required_files() -> bool:
    print("Checking required files...")
    all_ok = True

    for relative_path in REQUIRED_FILES:
        file_path = PROJECT_ROOT / relative_path
        if file_path.exists():
            print(f"  OK  {relative_path}")
        else:
            print(f"  MISSING  {relative_path}")
            all_ok = False

    return all_ok


def check_python_syntax() -> bool:
    print("\nCompiling Python sources...")
    all_ok = True

    for relative_path in PYTHON_FILES:
        file_path = PROJECT_ROOT / relative_path
        try:
            py_compile.compile(str(file_path), doraise=True)
            print(f"  OK  {relative_path}")
        except py_compile.PyCompileError as exc:
            print(f"  ERROR  {relative_path}: {exc.msg}")
            all_ok = False

    return all_ok


def main() -> int:
    files_ok = check_required_files()
    syntax_ok = check_python_syntax()

    print("\nSummary")
    print("  Files:", "OK" if files_ok else "FAIL")
    print("  Python syntax:", "OK" if syntax_ok else "FAIL")

    if files_ok and syntax_ok:
        print("\nFLPG source tree looks consistent.")
        return 0

    print("\nFLPG verification failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

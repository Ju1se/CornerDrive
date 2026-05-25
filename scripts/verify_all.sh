#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

INSTALL_DEPS=false
RUN_BACKEND=true
RUN_FRONTEND=true
RUN_CONTRACTS=true

usage() {
  cat <<'EOF'
Usage: bash scripts/verify_all.sh [options]

Runs the project verification suite:
  - backend pytest suite
  - expected reproduction-result tolerance check
  - frontend TypeScript/Vite build and ESLint
  - Hardhat contract tests

Options:
  --install       Install missing Python and npm dependencies before verifying.
  --backend-only  Run only backend verification.
  --frontend-only Run only frontend verification.
  --contracts-only
                  Run only contract verification.
  -h, --help      Show this help text.

Examples:
  bash scripts/verify_all.sh
  bash scripts/verify_all.sh --install
EOF
}

for arg in "$@"; do
  case "$arg" in
    --install)
      INSTALL_DEPS=true
      ;;
    --backend-only)
      RUN_BACKEND=true
      RUN_FRONTEND=false
      RUN_CONTRACTS=false
      ;;
    --frontend-only)
      RUN_BACKEND=false
      RUN_FRONTEND=true
      RUN_CONTRACTS=false
      ;;
    --contracts-only)
      RUN_BACKEND=false
      RUN_FRONTEND=false
      RUN_CONTRACTS=true
      ;;
    -h|--help|help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $arg" >&2
      usage
      exit 2
      ;;
  esac
done

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

resolve_python() {
  if [[ -x ".venv/bin/python" ]]; then
    echo ".venv/bin/python"
  elif [[ -x "backend/venv/bin/python" ]]; then
    echo "backend/venv/bin/python"
  elif command_exists python3; then
    echo "python3"
  elif command_exists python; then
    echo "python"
  else
    echo "Python is required but was not found." >&2
    exit 1
  fi
}

ensure_python_deps() {
  local python_bin="$1"

  if "$python_bin" -m pytest --version >/dev/null 2>&1; then
    return
  fi

  if [[ "$INSTALL_DEPS" != true ]]; then
    cat >&2 <<'EOF'
pytest is not available for the selected Python environment.
Run `bash scripts/verify_all.sh --install` or install `requirements.txt` first.
EOF
    exit 1
  fi

  if [[ "$python_bin" == python || "$python_bin" == python3 ]]; then
    python3 -m venv .venv
    python_bin=".venv/bin/python"
  fi

  "$python_bin" -m pip install --upgrade pip
  "$python_bin" -m pip install -r requirements.txt
}

ensure_node_deps() {
  local dir="$1"

  if [[ -d "$dir/node_modules" ]]; then
    return
  fi

  if [[ "$INSTALL_DEPS" != true ]]; then
    cat >&2 <<EOF
Missing npm dependencies in $dir.
Run \`bash scripts/verify_all.sh --install\` or run \`npm ci\` in $dir first.
EOF
    exit 1
  fi

  (cd "$dir" && npm ci)
}

run_backend() {
  local python_bin
  python_bin="$(resolve_python)"
  ensure_python_deps "$python_bin"
  if [[ -x ".venv/bin/python" ]]; then
    python_bin=".venv/bin/python"
  fi

  echo "[verify] Backend tests"
  PYTHONPATH="$ROOT/backend:$ROOT/scripts" "$python_bin" -m pytest backend/tests -q

  echo "[verify] Expected reproduction results"
  "$python_bin" scripts/check_expected_results.py --quiet
}

run_frontend() {
  if ! command_exists npm; then
    echo "npm is required for frontend verification." >&2
    exit 1
  fi

  ensure_node_deps "frontend"

  echo "[verify] Frontend build"
  (cd frontend && npm run build)

  echo "[verify] Frontend lint"
  (cd frontend && npm run lint)

  echo "[verify] Frontend production dependency audit"
  (cd frontend && npm audit --omit=dev --audit-level=high)
}

run_contracts() {
  if ! command_exists npm; then
    echo "npm is required for contract verification." >&2
    exit 1
  fi

  ensure_node_deps "contracts"

  echo "[verify] Contract tests"
  (cd contracts && npm test)
}

if [[ "$RUN_BACKEND" == true ]]; then
  run_backend
fi

if [[ "$RUN_FRONTEND" == true ]]; then
  run_frontend
fi

if [[ "$RUN_CONTRACTS" == true ]]; then
  run_contracts
fi

echo "[verify] All requested checks passed"

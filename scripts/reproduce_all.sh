#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON:-python}"
if [[ "${PYTHON_BIN}" == "python" && -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
fi

export BATCH_SIZE="${BATCH_SIZE:-24}"
export VEHICLE_POOL_SIZE="${VEHICLE_POOL_SIZE:-128}"

ALG_SEEDS="${ALG_SEEDS:-20260318,20260319,20260320,20260321,20260322}"
REAL_SEEDS="${REAL_SEEDS:-20260527,20260528,20260529,20260530,20260531,20260532,20260533,20260534,20260535,20260536,20260537,20260538,20260539,20260540,20260541,20260542,20260543,20260544,20260545,20260546}"
REAL_SOURCES="${REAL_SOURCES:-mnist,fashionmnist,femnist}"
RECHECK_VALUES="${RECHECK_VALUES:-0.00,0.05,0.10,0.20,0.30}"
LEARNING_CURVE_SEEDS="${LEARNING_CURVE_SEEDS:-20260507,20260508,20260509,20260510,20260511}"
LEARNING_CURVE_ROUNDS="${LEARNING_CURVE_ROUNDS:-50}"

run_alg_main() {
  echo "[CornerDrive] Reproducing ALG/V2.5 main and recheck tables"
  "${PYTHON_BIN}" scripts/export_v25_artifacts.py \
    --seeds "${ALG_SEEDS}" \
    --recheck-values "${RECHECK_VALUES}" \
    --output-dir results/audit_reproduction/v25_artifacts_b24
}

run_stress() {
  echo "[CornerDrive] Reproducing rarity/proxy/threshold stress tables"
  "${PYTHON_BIN}" scripts/export_v25_stress_tests.py \
    --seeds "${ALG_SEEDS}" \
    --threshold-seeds "${ALG_SEEDS}" \
    --output-dir results/audit_reproduction/v25_stress_tests_b24
}

run_divergence() {
  echo "[CornerDrive] Reproducing corner-family divergence table"
  "${PYTHON_BIN}" scripts/export_corner_family_divergence.py \
    --seeds "${ALG_SEEDS}" \
    --output-dir results/audit_reproduction/corner_family_divergence_b24
}

run_corner_harm_threshold() {
  echo "[CornerDrive] Reproducing corner-harm threshold calibration table"
  "${PYTHON_BIN}" scripts/export_corner_harm_threshold_calibration.py \
    --skip-oracle-drift \
    --seeds "${ALG_SEEDS}" \
    --output-dir results/audit_reproduction/corner_harm_threshold_calibration_b24
}

run_real_gradient() {
  echo "[CornerDrive] Reproducing real-gradient reliability tables"
  "${PYTHON_BIN}" scripts/export_real_gradient_reliability_benchmark.py \
    --sources "${REAL_SOURCES}" \
    --seeds "${REAL_SEEDS}" \
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
    --output-dir results/real_gradient_reliability_v41_best_holdout_20260527_20260546
}

run_frontiers() {
  echo "[CornerDrive] Reproducing audit cost frontier and calibration split tables"
  "${PYTHON_BIN}" scripts/export_cost_performance_frontier.py \
    --input results/audit_reproduction/v25_artifacts_b24/v25_recheck_sweep_table.csv \
    --output-dir results/cost_performance_frontier
  "${PYTHON_BIN}" scripts/export_real_gradient_calibration_split.py \
    --input-root results/real_gradient_threshold_sweep \
    --output-dir results/real_gradient_calibration_split
}

run_learning_curve() {
  echo "[CornerDrive] Reproducing real-gradient FL learning curve"
  "${PYTHON_BIN}" scripts/export_real_gradient_learning_curve.py \
    --sources "${REAL_SOURCES}" \
    --seeds "${LEARNING_CURVE_SEEDS}" \
    --rounds "${LEARNING_CURVE_ROUNDS}" \
    --download \
    --max-clients 120 \
    --min-samples-per-client 8 \
    --max-samples-per-client 48 \
    --clients-per-round 20 \
    --pretrain-steps 50 \
    --local-batch-size 16 \
    --reference-split-fraction 0.50 \
    --max-reference-samples 4096 \
    --max-evaluation-samples 4096 \
    --output-dir results/real_gradient_learning_curve
}

make_tables() {
  echo "[CornerDrive] Building paper-facing CSV tables"
  "${PYTHON_BIN}" scripts/make_paper_tables.py
}

usage() {
  cat <<'EOF'
Usage: bash scripts/reproduce_all.sh [main|appendix|real-gradient|frontiers|learning-curve|journal|all|tables]

Modes:
  main          Reproduce ALG/V2.5 main result and recheck tables.
  appendix      Reproduce stress, divergence, and threshold appendix tables.
  real-gradient Reproduce MNIST/FashionMNIST/FEMNIST real-gradient tables.
  frontiers     Reproduce cost frontier and dev/test calibration split tables.
  learning-curve
                Reproduce the 50-round real-gradient FL learning curve.
  journal       Run real-gradient, frontiers, learning-curve, then build tables.
  all           Run real-gradient, main, appendix, then build CSV tables.
  tables        Build artifacts/tables/*.csv from existing results.

Environment overrides:
  PYTHON=.venv/bin/python
  BATCH_SIZE=24
  VEHICLE_POOL_SIZE=128
  ALG_SEEDS=20260318,20260319,20260320,20260321,20260322
  REAL_SOURCES=mnist,fashionmnist,femnist
  REAL_SEEDS=20260527,20260528,20260529,20260530,20260531,20260532,20260533,20260534,20260535,20260536,20260537,20260538,20260539,20260540,20260541,20260542,20260543,20260544,20260545,20260546
  LEARNING_CURVE_SEEDS=20260507,20260508,20260509,20260510,20260511
  LEARNING_CURVE_ROUNDS=50
EOF
}

MODE="${1:-main}"
case "${MODE}" in
  main)
    run_alg_main
    make_tables
    ;;
  appendix)
    run_stress
    run_divergence
    run_corner_harm_threshold
    make_tables
    ;;
  real-gradient)
    run_real_gradient
    make_tables
    ;;
  frontiers)
    run_frontiers
    ;;
  learning-curve)
    run_learning_curve
    ;;
  journal)
    run_real_gradient
    run_frontiers
    run_learning_curve
    make_tables
    ;;
  all)
    run_real_gradient
    run_alg_main
    run_stress
    run_divergence
    run_corner_harm_threshold
    make_tables
    ;;
  tables)
    make_tables
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage
    exit 2
    ;;
esac

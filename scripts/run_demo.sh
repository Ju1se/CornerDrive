#!/bin/bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
readonly BACKEND_DIR="$PROJECT_ROOT/backend"
readonly FRONTEND_DIR="$PROJECT_ROOT/frontend"
readonly CONTRACTS_DIR="$PROJECT_ROOT/contracts"
readonly LOG_DIR="$PROJECT_ROOT/logs"
readonly STATE_DIR="$PROJECT_ROOT/runtime_state"
readonly REDIS_STATE_DIR="$STATE_DIR/redis"
readonly GANACHE_STATE_DIR="$STATE_DIR/ganache"
readonly ENV_FILE="$PROJECT_ROOT/.env"
readonly GENERATOR_SCRIPT="$SCRIPT_DIR/generate_demo_data.py"
readonly PYTHON_BIN="$BACKEND_DIR/venv/bin/python"
readonly CELERY_BIN="$BACKEND_DIR/venv/bin/celery"

mkdir -p "$LOG_DIR"
mkdir -p "$REDIS_STATE_DIR" "$GANACHE_STATE_DIR"

LOCAL_REDIS_STARTED=false
LOCAL_GANACHE_STARTED=false
DOCKER_REDIS_STARTED=false
DOCKER_GANACHE_STARTED=false
STARTED_PIDFILES=()

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

compose_available() {
    command_exists docker-compose || (command_exists docker && docker compose version >/dev/null 2>&1)
}

docker_compose() {
    if command_exists docker-compose; then
        docker-compose "$@"
        return
    fi

    docker compose "$@"
}

docker_ready() {
    command_exists docker && docker info >/dev/null 2>&1
}

port_in_use() {
    lsof -Pi :"$1" -sTCP:LISTEN -t >/dev/null 2>&1
}

record_pidfile() {
    STARTED_PIDFILES+=("$1")
}

wait_for_redis() {
    for _ in {1..20}; do
        if redis-cli -p 6379 ping >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    return 1
}

wait_for_ganache() {
    local response
    for _ in {1..20}; do
        response=$(curl -sf http://127.0.0.1:8545 \
            -X POST \
            -H "Content-Type: application/json" \
            -d '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' 2>/dev/null || true)
        if [[ -n "$response" && "$response" == *"\"result\""* ]]; then
            return 0
        fi
        sleep 1
    done
    return 1
}

wait_for_url() {
    local name=$1
    local url=$2

    for _ in {1..20}; do
        if curl -sf "$url" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done

    echo "⚠️  $name did not become ready in time: $url"
    return 1
}

amqp_broker_reachable() {
    "$PYTHON_BIN" - "$CELERY_BROKER_URL" <<'PY'
import socket
import sys
from urllib.parse import urlparse

url = urlparse(sys.argv[1])
host = url.hostname or "127.0.0.1"
port = url.port or 5672

sock = socket.socket()
sock.settimeout(1.0)
try:
    sock.connect((host, port))
except OSError:
    raise SystemExit(1)
finally:
    sock.close()
PY
}

normalize_local_celery_broker() {
    local fallback_url="${CELERY_REDIS_FALLBACK_URL:-redis://localhost:6379/1}"

    case "$CELERY_BROKER_URL" in
        amqp://*|pyamqp://*)
            if amqp_broker_reachable; then
                echo "✓ Using configured AMQP broker: $CELERY_BROKER_URL"
            else
                echo "⚠️  CELERY_BROKER_URL points to RabbitMQ, but no AMQP broker is reachable."
                echo "   Falling back to Redis broker for local development: $fallback_url"
                export CELERY_BROKER_URL="$fallback_url"
            fi
            ;;
    esac
}

load_env_file() {
    if [[ -f "$ENV_FILE" ]]; then
        set -a
        # shellcheck disable=SC1090
        source "$ENV_FILE"
        set +a
    fi
}

update_env_value() {
    local key=$1
    local value=$2

    "$PYTHON_BIN" - "$ENV_FILE" "$key" "$value" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
key = sys.argv[2]
value = sys.argv[3]

lines = path.read_text().splitlines() if path.exists() else []
updated = False

for index, line in enumerate(lines):
    if line.startswith(f"{key}="):
        lines[index] = f"{key}={value}"
        updated = True
        break

if not updated:
    if lines and lines[-1] != "":
        lines.append("")
    lines.append(f"{key}={value}")

path.write_text("\n".join(lines) + "\n")
PY
}

contract_available() {
    local address=$1
    local response

    if [[ -z "$address" ]]; then
        return 1
    fi

    response=$(curl -sf "$GANACHE_URL" \
        -X POST \
        -H "Content-Type: application/json" \
        -d "{\"jsonrpc\":\"2.0\",\"method\":\"eth_getCode\",\"params\":[\"$address\",\"latest\"],\"id\":1}" 2>/dev/null || true)

    [[ -n "$response" && "$response" == *"\"result\""* && "$response" != *"\"result\":\"0x\""* ]]
}

deploy_l4_contract() {
    local deploy_output
    local deployed_address

    echo "Deploying FLPGAudit contract..."
    if ! deploy_output=$(
        cd "$CONTRACTS_DIR" && \
        GANACHE_URL="$GANACHE_URL" \
        ORACLE_PRIVATE_KEY="${L4_ORACLE_PRIVATE_KEY:-}" \
        npx hardhat run scripts/deploy.js --network localhost 2>&1
    ); then
        printf '%s\n' "$deploy_output" | tee "$LOG_DIR/contract_deploy.log"
        echo "FLPGAudit deployment failed"
        exit 1
    fi

    printf '%s\n' "$deploy_output" | tee "$LOG_DIR/contract_deploy.log"
    deployed_address=$(printf '%s\n' "$deploy_output" | sed -n 's/^L4_CONTRACT_ADDRESS=//p' | tail -n 1)

    if [[ -z "$deployed_address" ]]; then
        echo "Unable to parse deployed FLPGAudit address"
        exit 1
    fi

    export L4_CONTRACT_ADDRESS="$deployed_address"
    update_env_value "L4_CONTRACT_ADDRESS" "$deployed_address"
    echo "✓ FLPGAudit ready at $deployed_address"
}

deploy_policy_registry_if_needed() {
    local deploy_output
    local deployed_address

    if [[ "${POLICY_CHAIN_COMMIT_ENABLED:-false}" != "true" ]]; then
        return
    fi

    if contract_available "${POLICY_REGISTRY_ADDRESS:-}"; then
        echo "✓ Using configured PolicyRegistry contract: $POLICY_REGISTRY_ADDRESS"
        return
    fi

    echo "Deploying PolicyRegistry contract..."
    if ! deploy_output=$(
        cd "$CONTRACTS_DIR" && \
        GANACHE_URL="$GANACHE_URL" \
        POLICY_ORACLE_PRIVATE_KEY="${POLICY_ORACLE_PRIVATE_KEY:-}" \
        npx hardhat run scripts/deploy_policy_registry.js --network localhost 2>&1
    ); then
        printf '%s\n' "$deploy_output" | tee "$LOG_DIR/policy_registry_deploy.log"
        echo "PolicyRegistry deployment failed"
        exit 1
    fi

    printf '%s\n' "$deploy_output" | tee "$LOG_DIR/policy_registry_deploy.log"
    deployed_address=$(printf '%s\n' "$deploy_output" | sed -n 's/^POLICY_REGISTRY_ADDRESS=//p' | tail -n 1)

    if [[ -z "$deployed_address" ]]; then
        echo "Unable to parse deployed PolicyRegistry address"
        exit 1
    fi

    export POLICY_REGISTRY_ADDRESS="$deployed_address"
    update_env_value "POLICY_REGISTRY_ADDRESS" "$deployed_address"
    echo "✓ PolicyRegistry ready at $deployed_address"
}

stop_pidfile() {
    local pidfile=$1

    if [[ ! -f "$pidfile" ]]; then
        return
    fi

    local pid
    pid=$(cat "$pidfile")

    if ps -p "$pid" >/dev/null 2>&1; then
        kill "$pid" 2>/dev/null || true
        sleep 1
        if ps -p "$pid" >/dev/null 2>&1; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    fi

    rm -f "$pidfile"
}

cleanup() {
    for (( index=${#STARTED_PIDFILES[@]}-1; index>=0; index-- )); do
        stop_pidfile "${STARTED_PIDFILES[index]}"
    done

    if [[ "$LOCAL_REDIS_STARTED" == true ]]; then
        redis-cli -p 6379 shutdown >/dev/null 2>&1 || true
    fi

    if compose_available && docker_ready; then
        if [[ "$DOCKER_REDIS_STARTED" == true ]]; then
            (cd "$PROJECT_ROOT" && docker_compose stop redis >/dev/null 2>&1) || true
        fi
        if [[ "$DOCKER_GANACHE_STARTED" == true ]]; then
            (cd "$PROJECT_ROOT" && docker_compose stop ganache >/dev/null 2>&1) || true
        fi
    fi
}

trap cleanup EXIT INT TERM

echo "Starting FLPG stack and simulated gradient stream..."

if [[ ! -x "$PYTHON_BIN" || ! -x "$CELERY_BIN" ]]; then
    echo "Backend virtual environment is missing. Run ./scripts/setup.sh first."
    exit 1
fi

if [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
    echo "Frontend dependencies are missing. Run ./scripts/setup.sh first."
    exit 1
fi

if [[ ! -d "$CONTRACTS_DIR/node_modules" ]]; then
    echo "Contract dependencies are missing. Run ./scripts/setup.sh first."
    exit 1
fi

if [[ ! -f "$GENERATOR_SCRIPT" ]]; then
    echo "Demo generator script is missing: $GENERATOR_SCRIPT"
    exit 1
fi

load_env_file

export CONTINUOUS_MODE="${CONTINUOUS_MODE:-true}"
export SIMULATION_SCALE_FACTOR="${SIMULATION_SCALE_FACTOR:-3}"
export BASE_BATCH_SIZE="${BASE_BATCH_SIZE:-32}"
export BATCH_SIZE="${BATCH_SIZE:-$((BASE_BATCH_SIZE * SIMULATION_SCALE_FACTOR))}"
export NUM_ROUNDS="${NUM_ROUNDS:-$((4 * SIMULATION_SCALE_FACTOR))}"
export VEHICLE_POOL_SIZE="${VEHICLE_POOL_SIZE:-$((BATCH_SIZE * 4))}"
export SAMPLE_COUNT_MIN="${SAMPLE_COUNT_MIN:-1200}"
export SAMPLE_COUNT_MAX="${SAMPLE_COUNT_MAX:-12000}"
export PROCESS_WAIT_SECONDS="${PROCESS_WAIT_SECONDS:-28}"
export SUBMISSION_MAX_WORKERS="${SUBMISSION_MAX_WORKERS:-12}"
export L1_BATCH_SIZE="${L1_BATCH_SIZE:-$BATCH_SIZE}"

export REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}"
export CELERY_BROKER_URL="${CELERY_BROKER_URL:-redis://localhost:6379/1}"
export GANACHE_URL="${GANACHE_URL:-http://127.0.0.1:8545}"
export POLICY_AGENT_URL="${POLICY_AGENT_URL:-http://127.0.0.1:8083}"
export L1_URL="${L1_URL:-http://127.0.0.1:8081}"
export L4_URL="${L4_URL:-http://127.0.0.1:8082}"

if wait_for_redis; then
    echo "✓ Redis is already running"
elif compose_available && docker_ready; then
    echo "Starting Redis with Docker Compose..."
    (cd "$PROJECT_ROOT" && docker_compose up -d redis >/dev/null)
    DOCKER_REDIS_STARTED=true
    wait_for_redis || {
        echo "Redis failed to start"
        exit 1
    }
    echo "✓ Redis started via Docker"
elif command_exists redis-server; then
    echo "Starting local Redis..."
    redis-server \
        --daemonize yes \
        --port 6379 \
        --dir "$REDIS_STATE_DIR" \
        --dbfilename dump.rdb \
        --appendonly yes \
        --appendfilename appendonly.aof
    LOCAL_REDIS_STARTED=true
    wait_for_redis || {
        echo "Redis failed to start"
        exit 1
    }
    echo "✓ Redis started locally"
else
    echo "Redis is not running and redis-server is unavailable"
    exit 1
fi

normalize_local_celery_broker

if wait_for_ganache; then
    echo "✓ Ganache is already running"
elif compose_available && docker_ready; then
    echo "Starting Ganache with Docker Compose..."
    (cd "$PROJECT_ROOT" && docker_compose up -d ganache >/dev/null)
    DOCKER_GANACHE_STARTED=true
    wait_for_ganache || {
        echo "Ganache failed to start"
        exit 1
    }
    echo "✓ Ganache started via Docker"
elif command_exists npx; then
    echo "Starting local Ganache..."
    (
        cd "$PROJECT_ROOT"
        nohup npx ganache \
            --host 0.0.0.0 \
            --port 8545 \
            --chain.chainId 1337 \
            --database.dbPath "$GANACHE_STATE_DIR" \
            --wallet.mnemonic "flpg test account security word seed" \
            --wallet.totalAccounts 10 \
            --miner.blockTime 2 \
            > "$LOG_DIR/ganache.log" 2>&1 &
        echo $! > "$LOG_DIR/ganache.pid"
    )
    LOCAL_GANACHE_STARTED=true
    record_pidfile "$LOG_DIR/ganache.pid"
    wait_for_ganache || {
        echo "Ganache failed to start"
        exit 1
    }
    echo "✓ Ganache started locally"
else
    echo "Ganache is not running and npx is unavailable"
    exit 1
fi

if contract_available "${L4_CONTRACT_ADDRESS:-}"; then
    echo "✓ Using configured FLPGAudit contract: $L4_CONTRACT_ADDRESS"
else
    deploy_l4_contract
fi

deploy_policy_registry_if_needed

cd "$BACKEND_DIR"

if port_in_use 8081; then
    echo "⚠️  L1 API already running on 8081"
else
    "$PYTHON_BIN" -m l1_linear_defense.server > "$LOG_DIR/l1_server.log" 2>&1 &
    echo $! > "$LOG_DIR/l1.pid"
    record_pidfile "$LOG_DIR/l1.pid"
fi

if pgrep -f "celery.*l2_dual_audit\.worker:celery_app worker" >/dev/null 2>&1; then
    echo "⚠️  L2 worker already running"
else
    "$CELERY_BIN" -A l2_dual_audit.worker:celery_app worker --loglevel=info --queues=l2_audit_queue,celery > "$LOG_DIR/l2_worker.log" 2>&1 &
    echo $! > "$LOG_DIR/l2.pid"
    record_pidfile "$LOG_DIR/l2.pid"
fi

if port_in_use 8082; then
    echo "⚠️  L4 API already running on 8082"
else
    "$PYTHON_BIN" -m l4_settlement.dashboard_api > "$LOG_DIR/l4_dashboard.log" 2>&1 &
    echo $! > "$LOG_DIR/l4.pid"
    record_pidfile "$LOG_DIR/l4.pid"
fi

if port_in_use 8083; then
    echo "⚠️  Policy Agent already running on 8083"
else
    "$PYTHON_BIN" -m policy_agent.main > "$LOG_DIR/policy_agent.log" 2>&1 &
    echo $! > "$LOG_DIR/policy_agent.pid"
    record_pidfile "$LOG_DIR/policy_agent.pid"
fi

if pgrep -f "celery.*policy_agent\.tasks\.round_close:celery_app beat" >/dev/null 2>&1; then
    echo "⚠️  Policy beat already running"
else
    "$CELERY_BIN" -A policy_agent.tasks.round_close:celery_app beat --loglevel=info --schedule "$BACKEND_DIR/celerybeat-policy-schedule" > "$LOG_DIR/policy_beat.log" 2>&1 &
    echo $! > "$LOG_DIR/policy_beat.pid"
    record_pidfile "$LOG_DIR/policy_beat.pid"
fi

cd "$FRONTEND_DIR"
if port_in_use 3000; then
    echo "⚠️  Frontend already running on 3000"
else
    npm run dev -- --host 0.0.0.0 > "$LOG_DIR/frontend.log" 2>&1 &
    echo $! > "$LOG_DIR/frontend.pid"
    record_pidfile "$LOG_DIR/frontend.pid"
fi

wait_for_url "L1 API" "http://127.0.0.1:8081/health" || true
wait_for_url "L4 API" "http://127.0.0.1:8082/health" || true
wait_for_url "Policy Agent" "http://127.0.0.1:8083/health" || true
wait_for_url "Frontend" "http://127.0.0.1:3000" || true

echo ""
echo "FLPG services are up."
echo "  Frontend:     http://localhost:3000"
echo "  L1 API:       http://localhost:8081"
echo "  L4 API:       http://localhost:8082"
echo "  Policy Agent: http://localhost:8083"
echo "  Ganache RPC:  http://localhost:8545"
echo ""
echo "Logs:"
echo "  $LOG_DIR/l1_server.log"
echo "  $LOG_DIR/l2_worker.log"
echo "  $LOG_DIR/l4_dashboard.log"
echo "  $LOG_DIR/policy_agent.log"
echo "  $LOG_DIR/policy_beat.log"
echo "  $LOG_DIR/frontend.log"
echo ""
echo "Starting simulated gradient stream..."
echo "Press Ctrl+C to stop the stack and the generator together."
echo ""

echo "Generator config:"
echo "  scale factor:      ${SIMULATION_SCALE_FACTOR}x"
echo "  batch size:        $BATCH_SIZE"
echo "  L1 batch size:     $L1_BATCH_SIZE"
echo "  vehicle pool:      $VEHICLE_POOL_SIZE"
echo "  sample count span: $SAMPLE_COUNT_MIN-$SAMPLE_COUNT_MAX"
echo "  submit workers:    $SUBMISSION_MAX_WORKERS"
echo "  audit wait:        ${PROCESS_WAIT_SECONDS}s"
echo ""
set +e
"$PYTHON_BIN" "$GENERATOR_SCRIPT"
GENERATOR_EXIT=$?
set -e

if [[ "$GENERATOR_EXIT" -ne 0 ]]; then
    echo "Demo generator exited with status $GENERATOR_EXIT. Stopping the stack."
    exit "$GENERATOR_EXIT"
fi

echo "Demo generator finished. Stopping the stack."

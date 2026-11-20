#!/usr/bin/env bash

set -Eeo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Reuse a dedicated compilation cache for this script across runs.
export VLLM_CACHE_ROOT="${SCRIPT_DIR}/.vllm-cache-ycp"
(umask 077 && mkdir -p -- "$VLLM_CACHE_ROOT")
echo "vLLM cache: $VLLM_CACHE_ROOT"

# Personal launcher: no Git checks, switching, or restoring.
export VLLM_VERSION="${VLLM_VERSION:-0.27.0}"
export MODEL="${MODEL:-/home/y00906461/models/Qwen3.5-35B-A3B}"
export SERVED_MODEL="${SERVED_MODEL:-qwen3_5}"
export HOST_IP="${HOST_IP:-61.28.30.29}"
export VLLM_ROOT="/workspace/ycp/vllm"
export ASCEND_ROOT="/workspace/ycp/vllm-ascend"

EXPERIMENT_LABEL="${EXPERIMENT_LABEL:-mtp3-dyntra-lb-performance}"
if [[ ! "${EXPERIMENT_LABEL}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: EXPERIMENT_LABEL may contain only letters, numbers, dot, underscore, and dash." >&2
    exit 2
fi

export PROXY_SCRIPT="${PROXY_SCRIPT:-${SCRIPT_DIR}/run_proxy_server.sh}"
ORIGINAL_PREFILLER="${SCRIPT_DIR}/run_vllm_serve_8p8d_mtp_p.sh"
export DECODER_SCRIPT_BL="${DECODER_SCRIPT_BL:-${SCRIPT_DIR}/run_vllm_serve_8p8d_mtp_d_bl.sh}"
export DECODER_SCRIPT_LB="${DECODER_SCRIPT_LB:-${SCRIPT_DIR}/run_vllm_serve_8p8d_mtp_d_lb.sh}"
export DECODER_SCRIPT_DYNAM="${DECODER_SCRIPT_DYNAM:-${SCRIPT_DIR}/run_vllm_serve_8p8d_mtp_d_dynam.sh}"
export BENCHMARK_SCRIPT="${BENCHMARK_SCRIPT:-${SCRIPT_DIR}/run_test_proxy_benchmark_manualPrompts_multiRound.sh}"

export BENCHMARK_MODEL="${BENCHMARK_MODEL:-${SERVED_MODEL}}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL}}"
export DATASET_PATH="${DATASET_PATH:-${SCRIPT_DIR}/Dataset-20260729-1733.jsonl}"

export TEST_MODES="${TEST_MODES:-bl,lb,dynam}"
export NUM_ROUNDS="${NUM_ROUNDS:-1}"
export NUM_PROMPTS="${NUM_PROMPTS:-1000}"
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-128}"
export REQUEST_RATE="${REQUEST_RATE:-400}"
export DISABLE_SHUFFLE="${DISABLE_SHUFFLE:-true}"
export ENABLE_DIAGNOSTICS="${ENABLE_DIAGNOSTICS:-true}"
export RECOMPUTE_SCHEDULER_ENABLE="${RECOMPUTE_SCHEDULER_ENABLE:-false}"

export INITIAL_STARTUP_DELAY="${INITIAL_STARTUP_DELAY:-10}"
export READY_SETTLE_DELAY="${READY_SETTLE_DELAY:-3}"
export DECODER_DRAIN_DELAY="${DECODER_DRAIN_DELAY:-5}"
export RESOURCE_RELEASE_DELAY="${RESOURCE_RELEASE_DELAY:-5}"
export ROUND_INTERVAL_DELAY="${ROUND_INTERVAL_DELAY:-0}"
export LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/log/${EXPERIMENT_LABEL}}"
export RESULT_DIR="${RESULT_DIR:-${SCRIPT_DIR}/results/${EXPERIMENT_LABEL}}"
export SKIP_INITIAL_PROCESS_CLEANUP=1

# Validate before starting any processes.
for required in "${VLLM_ROOT}/vllm" "${ASCEND_ROOT}/vllm_ascend"; do
    if [[ ! -d "$required" ]]; then
        echo "ERROR: source directory not found: $required" >&2
        echo "Edit VLLM_ROOT or ASCEND_ROOT near the top of this script." >&2
        exit 1
    fi
done
for required in "${MODEL}/config.json" "$DATASET_PATH" "$ORIGINAL_PREFILLER" "${SCRIPT_DIR}/automatic_benchmark.sh"; do
    if [[ ! -f "$required" ]]; then
        echo "ERROR: required file not found: $required" >&2
        exit 1
    fi
done

# Create a private prefiller copy, omitting the broad /dev/shm cleanup call.
# Refuse to proceed if the upstream script no longer matches the known line.
if [[ $(grep -Fxc 'cleanup_stale_vllm_shm || exit 1' "$ORIGINAL_PREFILLER" || true) != 1 ]]; then
    echo "ERROR: Prefill script differs from the expected version; please inspect its /dev/shm cleanup." >&2
    exit 1
fi
PRIVATE_PREFILL_DIR=$(mktemp -d "${TMPDIR:-/tmp}/dyntralb-prefill.XXXXXX")
export PREFILLER_SCRIPT="${PRIVATE_PREFILL_DIR}/prefill.sh"
sed '/^cleanup_stale_vllm_shm || exit 1$/d' "$ORIGINAL_PREFILLER" > "$PREFILLER_SCRIPT"
bash -n "$PREFILLER_SCRIPT"

# Add a bounded wait to a private copy of the benchmark controller.
export STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"
if [[ ! "$STARTUP_TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: STARTUP_TIMEOUT must be a positive integer (seconds)." >&2
    exit 1
fi
export DYNTRALB_AVERAGE_SCRIPT="${SCRIPT_DIR}/calculate_average.py"
if [[ ! -f "$DYNTRALB_AVERAGE_SCRIPT" ]]; then
    echo "ERROR: missing average calculator: $DYNTRALB_AVERAGE_SCRIPT" >&2
    exit 1
fi
PRIVATE_BENCHMARK="${PRIVATE_PREFILL_DIR}/automatic_benchmark.sh"
awk '
/^wait_for_ready\(\) \{$/ { in_function=1; functions++ }
in_function && $0 == "    local pid=$4" {
    print
    print "    local deadline=$((SECONDS + STARTUP_TIMEOUT))"
    declarations++
    next
}
in_function && $0 == "    while true; do" {
    print
    print "        if (( SECONDS >= deadline )); then"
    print "            echo \"ERROR: ${module_name} startup timed out after ${STARTUP_TIMEOUT}s\" >&2"
    print "            return 1"
    print "        fi"
    loops++
    next
}
in_function && /^}/ { in_function=0 }
/^run_benchmark_mode\(\) \{$/ { in_benchmark=1 }
in_benchmark && $0 == "    set_mode_status \"$mode\" \"测试完成\"" {
    print "    if ! python3 \"$DYNTRALB_AVERAGE_SCRIPT\" \"$result_file\" --append --expected \"$NUM_ROUNDS\"; then"
    print "        log_event \"WARNING: average calculation failed; check $result_file\""
    print "    fi"
    averages++
}
in_benchmark && /^}/ { in_benchmark=0 }
{ print }
END { if (functions != 1 || declarations != 1 || loops != 1 || averages != 1) exit 2 }
' "${SCRIPT_DIR}/automatic_benchmark.sh" > "$PRIVATE_BENCHMARK" || {
    echo "ERROR: automatic_benchmark.sh differs from the expected version; no service was started." >&2
    exit 1
}
bash -n "$PRIVATE_BENCHMARK"
export ETA_HISTORY_SCRIPT="${ETA_HISTORY_SCRIPT:-${SCRIPT_DIR}/benchmark_eta_history.py}"

export PYTHONPATH="${VLLM_ROOT}:${ASCEND_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1

echo "vLLM source:        $VLLM_ROOT"
echo "vLLM-Ascend source: $ASCEND_ROOT"
echo "Prefill copy:       $PREFILLER_SCRIPT"
echo "Modes:              $TEST_MODES"
echo "Startup wait limit: ${STARTUP_TIMEOUT}s per wait"

cd "$SCRIPT_DIR"
exec bash "$PRIVATE_BENCHMARK" "$@"

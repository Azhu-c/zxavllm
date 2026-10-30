#!/usr/bin/env bash

set -m

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# ================= 配置区 =================
PROXY_SCRIPT="${PROXY_SCRIPT:-./run_proxy_server.sh}"
PREFILLER_SCRIPT="${PREFILLER_SCRIPT:-./run_vllm_serve_8p8d_asyncSched_p.sh}"
DECODER_SCRIPT_BL="${DECODER_SCRIPT_BL:-./run_vllm_serve_8p8d_asyncSched_d_bl.sh}"
DECODER_SCRIPT_LB="${DECODER_SCRIPT_LB:-./run_vllm_serve_8p8d_asyncSched_d_lb.sh}"
DECODER_SCRIPT_DYNAM="${DECODER_SCRIPT_DYNAM:-./run_vllm_serve_8p8d_asyncSched_d_dynam.sh}"
BENCHMARK_SCRIPT="${BENCHMARK_SCRIPT:-./run_test_proxy_benchmark_manualPrompts_multiRound.sh}"

NUM_ROUNDS="${NUM_ROUNDS:-10}"
NUM_PROMPTS="${NUM_PROMPTS:-1000}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-128}"
REQUEST_RATE="${REQUEST_RATE:-400}"
DISABLE_SHUFFLE="${DISABLE_SHUFFLE:-false}"
TEST_MODES="${TEST_MODES:-bl,lb,dynam}"
DATASET_PATH="${DATASET_PATH:-./mixed_prompts_lognormal.jsonl}"
BENCHMARK_TZ="${BENCHMARK_TZ:-UTC-8}"
INTERNAL_NO_PROXY_HOSTS="${INTERNAL_NO_PROXY_HOSTS:-61.28.30.29,127.0.0.1,localhost}"
INITIAL_STARTUP_DELAY="${INITIAL_STARTUP_DELAY:-10}"
DECODER_DRAIN_DELAY="${DECODER_DRAIN_DELAY:-30}"
RESOURCE_RELEASE_DELAY="${RESOURCE_RELEASE_DELAY:-5}"
READY_SETTLE_DELAY="${READY_SETTLE_DELAY:-3}"
ROUND_INTERVAL_DELAY="${ROUND_INTERVAL_DELAY:-5}"
ETA_PROGRESS_MIN_PERCENT="${ETA_PROGRESS_MIN_PERCENT:-10}"
SKIP_INITIAL_PROCESS_CLEANUP="${SKIP_INITIAL_PROCESS_CLEANUP:-0}"
RECOMPUTE_SCHEDULER_ENABLE="${RECOMPUTE_SCHEDULER_ENABLE:-false}"

LOG_DIR="${LOG_DIR:-./log}"
ETA_HISTORY_FILE="${ETA_HISTORY_FILE:-${LOG_DIR}/automatic_benchmark_eta_history.json}"
ETA_HISTORY_SCRIPT="${ETA_HISTORY_SCRIPT:-${SCRIPT_DIR}/benchmark_eta_history.py}"
READY_STR_PROXY="${READY_STR_PROXY:-Application startup complete}"
READY_STR_PREFILLER="${READY_STR_PREFILLER:-Application startup complete}"
READY_STR_DECODER="${READY_STR_DECODER:-Application startup complete}"
export TZ="$BENCHMARK_TZ"
export no_proxy="${no_proxy:+${no_proxy},}${INTERNAL_NO_PROXY_HOSTS}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${INTERNAL_NO_PROXY_HOSTS}"
# ==========================================

usage() {
    cat <<'EOF'
用法:
  ./automatic_benchmark.sh [轮数] [模式列表] [数据集路径] [每轮请求数]
  ./automatic_benchmark.sh --rounds 10 --num-prompts 1000 --modes bl,lb,dynam --dataset-path Dataset-20260717-0930.jsonl
  ./automatic_benchmark.sh --rounds 2 --num-prompts 100 --modes bl --dataset-path Dataset-20260717-0930.jsonl --max-concurrency 1 --request-rate 1 --disable-shuffle

选项:
  -n, --rounds N          每种模式执行的 benchmark 轮数，默认 10
  -p, --num-prompts N     每轮 benchmark 的 request 数，默认 1000
  -m, --modes LIST        要测试的模式，逗号分隔；支持 bl、lb、dynam
  -d, --dataset-path P    数据集 JSONL 路径
      --max-concurrency N 最大并发请求数，默认 128
      --request-rate RATE 每秒请求数，默认 400；也支持 inf
      --disable-shuffle   禁止 benchmark 打乱数据集
      --recompute         开启 Decoder recompute scheduler
      --no-recompute      关闭 Decoder recompute scheduler（默认）
      --drain-seconds N   Benchmark 完成后停止 Decoder 前的 drain 秒数，默认 30
      --round-interval N  多轮 benchmark 之间的等待秒数，默认 5
  -h, --help              显示帮助

示例:
  ./automatic_benchmark.sh 5
  ./automatic_benchmark.sh 10 bl,lb Dataset-20260717-0930.jsonl 2000
  ./automatic_benchmark.sh --rounds 3 --num-prompts 2000 --modes bl,dynam --dataset-path Dataset-20260717-0930.jsonl

也可以通过环境变量 NUM_ROUNDS、NUM_PROMPTS、MAX_CONCURRENCY、REQUEST_RATE、
DISABLE_SHUFFLE、TEST_MODES、DATASET_PATH、RECOMPUTE_SCHEDULER_ENABLE、
BENCHMARK_TZ、INTERNAL_NO_PROXY_HOSTS
以及 DECODER_DRAIN_DELAY、ROUND_INTERVAL_DELAY 和 ETA_HISTORY_FILE 配置。
EOF
}

POSITIONAL_INDEX=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        -n|--rounds)
            [ "$#" -ge 2 ] || {
                echo "❌ $1 缺少参数" >&2
                exit 2
            }
            NUM_ROUNDS=$2
            shift 2
            ;;
        -p|--num-prompts)
            [ "$#" -ge 2 ] || {
                echo "❌ $1 缺少参数" >&2
                exit 2
            }
            NUM_PROMPTS=$2
            shift 2
            ;;
        -m|--modes)
            [ "$#" -ge 2 ] || {
                echo "❌ $1 缺少参数" >&2
                exit 2
            }
            TEST_MODES=$2
            shift 2
            ;;
        -d|--dataset-path)
            [ "$#" -ge 2 ] || {
                echo "❌ $1 缺少参数" >&2
                exit 2
            }
            DATASET_PATH=$2
            shift 2
            ;;
        --max-concurrency)
            [ "$#" -ge 2 ] || {
                echo "❌ $1 缺少参数" >&2
                exit 2
            }
            MAX_CONCURRENCY=$2
            shift 2
            ;;
        --request-rate)
            [ "$#" -ge 2 ] || {
                echo "❌ $1 缺少参数" >&2
                exit 2
            }
            REQUEST_RATE=$2
            shift 2
            ;;
        --disable-shuffle)
            DISABLE_SHUFFLE=true
            shift
            ;;
        --recompute)
            RECOMPUTE_SCHEDULER_ENABLE=true
            shift
            ;;
        --no-recompute)
            RECOMPUTE_SCHEDULER_ENABLE=false
            shift
            ;;
        --drain-seconds)
            [ "$#" -ge 2 ] || {
                echo "❌ $1 缺少参数" >&2
                exit 2
            }
            DECODER_DRAIN_DELAY=$2
            shift 2
            ;;
        --round-interval)
            [ "$#" -ge 2 ] || {
                echo "❌ $1 缺少参数" >&2
                exit 2
            }
            ROUND_INTERVAL_DELAY=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "❌ 未知选项: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            if [ "$POSITIONAL_INDEX" -eq 0 ]; then
                NUM_ROUNDS=$1
            elif [ "$POSITIONAL_INDEX" -eq 1 ]; then
                TEST_MODES=$1
            elif [ "$POSITIONAL_INDEX" -eq 2 ]; then
                DATASET_PATH=$1
            elif [ "$POSITIONAL_INDEX" -eq 3 ]; then
                NUM_PROMPTS=$1
            else
                echo "❌ 多余参数: $1" >&2
                usage >&2
                exit 2
            fi
            POSITIONAL_INDEX=$((POSITIONAL_INDEX + 1))
            shift
            ;;
    esac
done

if ! [[ "$NUM_ROUNDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "❌ 轮数必须是正整数，当前值: $NUM_ROUNDS" >&2
    exit 2
fi

if ! [[ "$NUM_PROMPTS" =~ ^[1-9][0-9]*$ ]]; then
    echo "❌ 每轮 request 数必须是正整数，当前值: $NUM_PROMPTS" >&2
    exit 2
fi

if ! [[ "$MAX_CONCURRENCY" =~ ^[1-9][0-9]*$ ]]; then
    echo "❌ 最大并发请求数必须是正整数，当前值: $MAX_CONCURRENCY" >&2
    exit 2
fi

if ! [[ "$REQUEST_RATE" =~ ^([1-9][0-9]*([.][0-9]+)?|0[.][0-9]*[1-9][0-9]*|inf)$ ]]; then
    echo "❌ request rate 必须是正数或 inf，当前值: $REQUEST_RATE" >&2
    exit 2
fi

case "$DISABLE_SHUFFLE" in
    true|false)
        ;;
    *)
        echo "❌ DISABLE_SHUFFLE 必须是 true 或 false，当前值: $DISABLE_SHUFFLE" >&2
        exit 2
        ;;
esac

case "$RECOMPUTE_SCHEDULER_ENABLE" in
    true|false)
        ;;
    *)
        echo "❌ RECOMPUTE_SCHEDULER_ENABLE 必须是 true 或 false，当前值: $RECOMPUTE_SCHEDULER_ENABLE" >&2
        exit 2
        ;;
esac
export RECOMPUTE_SCHEDULER_ENABLE

if ! [[ "$DECODER_DRAIN_DELAY" =~ ^[0-9]+$ ]]; then
    echo "❌ Decoder drain 秒数必须是非负整数，当前值: $DECODER_DRAIN_DELAY" >&2
    exit 2
fi

if ! [[ "$INITIAL_STARTUP_DELAY" =~ ^[0-9]+$ ]]; then
    echo "❌ 基础服务启动缓冲必须是非负整数，当前值: $INITIAL_STARTUP_DELAY" >&2
    exit 2
fi

if ! [[ "$RESOURCE_RELEASE_DELAY" =~ ^[0-9]+$ ]]; then
    echo "❌ 资源释放等待必须是非负整数，当前值: $RESOURCE_RELEASE_DELAY" >&2
    exit 2
fi

if ! [[ "$READY_SETTLE_DELAY" =~ ^[0-9]+$ ]]; then
    echo "❌ Ready settle 秒数必须是非负整数，当前值: $READY_SETTLE_DELAY" >&2
    exit 2
fi

if ! [[ "$ROUND_INTERVAL_DELAY" =~ ^[0-9]+$ ]]; then
    echo "❌ 轮间等待秒数必须是非负整数，当前值: $ROUND_INTERVAL_DELAY" >&2
    exit 2
fi

if ! [[ "$ETA_PROGRESS_MIN_PERCENT" =~ ^([1-9]|[1-9][0-9])$ ]]; then
    echo "❌ ETA_PROGRESS_MIN_PERCENT 必须是 1 到 99 的整数，当前值: $ETA_PROGRESS_MIN_PERCENT" >&2
    exit 2
fi

if [ ! -f "$ETA_HISTORY_SCRIPT" ]; then
    echo "❌ ETA 历史统计脚本不存在: $ETA_HISTORY_SCRIPT" >&2
    exit 2
fi

IFS=',' read -r -a RAW_MODES <<< "$TEST_MODES"
MODES=()
for raw_mode in "${RAW_MODES[@]}"; do
    mode=$(printf '%s' "$raw_mode" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')
    case "$mode" in
        bl|lb|dynam)
            duplicate=0
            for existing_mode in "${MODES[@]}"; do
                [ "$existing_mode" = "$mode" ] && duplicate=1
            done
            [ "$duplicate" -eq 1 ] || MODES+=("$mode")
            ;;
        "")
            ;;
        *)
            echo "❌ 不支持的模式: ${raw_mode}；仅支持 bl、lb、dynam" >&2
            exit 2
            ;;
    esac
done

if [ "${#MODES[@]}" -eq 0 ]; then
    echo "❌ 至少需要选择一种测试模式" >&2
    exit 2
fi

if [ ! -f "$DATASET_PATH" ]; then
    echo "❌ 数据集文件不存在: $DATASET_PATH" >&2
    exit 2
fi

mkdir -p "$LOG_DIR"
RUN_TIMESTAMP=$(date +'%Y%m%d_%H%M')
DATASET_FILENAME="${DATASET_PATH##*/}"
DATASET_NAME="${DATASET_FILENAME%.*}"
SCRIPT_START_EPOCH=$(date +%s)

PROGRESS_BL=0
PROGRESS_LB=0
PROGRESS_DYNAM=0
ROUND_REQUESTS_BL=0
ROUND_REQUESTS_LB=0
ROUND_REQUESTS_DYNAM=0
ROUND_REQUESTS_TOTAL_BL=0
ROUND_REQUESTS_TOTAL_LB=0
ROUND_REQUESTS_TOTAL_DYNAM=0
TOTAL_DURATION_BL=0
TOTAL_DURATION_LB=0
TOTAL_DURATION_DYNAM=0
ROUND_START_BL=0
ROUND_START_LB=0
ROUND_START_DYNAM=0
STATUS_BL="等待"
STATUS_LB="等待"
STATUS_DYNAM="等待"
DECODER_STARTED_BL=0
DECODER_STARTED_LB=0
DECODER_STARTED_DYNAM=0
DECODER_START_BL=0
DECODER_START_LB=0
DECODER_START_DYNAM=0
STARTUP_DONE_BL=0
STARTUP_DONE_LB=0
STARTUP_DONE_DYNAM=0
DRAIN_START_BL=0
DRAIN_START_LB=0
DRAIN_START_DYNAM=0
MODE_DONE_BL=0
MODE_DONE_LB=0
MODE_DONE_DYNAM=0
ROUND_GAP_START_BL=0
ROUND_GAP_START_LB=0
ROUND_GAP_START_DYNAM=0
ROUND_GAP_DURATION_BL=0
ROUND_GAP_DURATION_LB=0
ROUND_GAP_DURATION_DYNAM=0

BASE_STARTUP_DONE=0
TRANSITION_TOTAL=$((${#MODES[@]} - 1))
TRANSITION_COMPLETED=0
TRANSITION_START_EPOCH=0

PROXY_PID=""
PREFILLER_PID=""
DECODER_PID=""
BENCHMARK_PID=""
DASHBOARD_ENABLED=0
DASHBOARD_STOPPED=0

mode_label() {
    case "$1" in
        bl) printf 'BL' ;;
        lb) printf 'LB' ;;
        dynam) printf 'DYNAM' ;;
    esac
}

modes_display() {
    local display=""
    local mode
    for mode in "${MODES[@]}"; do
        if [ -n "$display" ]; then
            display="${display} -> "
        fi
        display="${display}$(mode_label "$mode")"
    done
    printf '%s' "$display"
}

mode_decoder_script() {
    case "$1" in
        bl) printf '%s' "$DECODER_SCRIPT_BL" ;;
        lb) printf '%s' "$DECODER_SCRIPT_LB" ;;
        dynam) printf '%s' "$DECODER_SCRIPT_DYNAM" ;;
    esac
}

mode_result_file() {
    printf '%s/%s_%s_rounds%s_%s.txt' \
        "$LOG_DIR" "$RUN_TIMESTAMP" "$DATASET_NAME" "$NUM_ROUNDS" "$1"
}

get_mode_progress() {
    case "$1" in
        bl) printf '%s' "$PROGRESS_BL" ;;
        lb) printf '%s' "$PROGRESS_LB" ;;
        dynam) printf '%s' "$PROGRESS_DYNAM" ;;
    esac
}

get_mode_average_duration() {
    local total_duration
    local completed_rounds

    case "$1" in
        bl)
            total_duration=$TOTAL_DURATION_BL
            completed_rounds=$PROGRESS_BL
            ;;
        lb)
            total_duration=$TOTAL_DURATION_LB
            completed_rounds=$PROGRESS_LB
            ;;
        dynam)
            total_duration=$TOTAL_DURATION_DYNAM
            completed_rounds=$PROGRESS_DYNAM
            ;;
    esac

    if [ "$completed_rounds" -gt 0 ]; then
        printf '%s' "$(((total_duration + completed_rounds / 2) / completed_rounds))"
    else
        printf '0'
    fi
}

get_mode_status() {
    case "$1" in
        bl) printf '%s' "$STATUS_BL" ;;
        lb) printf '%s' "$STATUS_LB" ;;
        dynam) printf '%s' "$STATUS_DYNAM" ;;
    esac
}

get_mode_round_start() {
    case "$1" in
        bl) printf '%s' "$ROUND_START_BL" ;;
        lb) printf '%s' "$ROUND_START_LB" ;;
        dynam) printf '%s' "$ROUND_START_DYNAM" ;;
    esac
}

set_mode_progress() {
    case "$1" in
        bl) PROGRESS_BL=$2 ;;
        lb) PROGRESS_LB=$2 ;;
        dynam) PROGRESS_DYNAM=$2 ;;
    esac
}

get_mode_round_requests() {
    case "$1" in
        bl) printf '%s' "$ROUND_REQUESTS_BL" ;;
        lb) printf '%s' "$ROUND_REQUESTS_LB" ;;
        dynam) printf '%s' "$ROUND_REQUESTS_DYNAM" ;;
    esac
}

set_mode_round_requests() {
    case "$1" in
        bl) ROUND_REQUESTS_BL=$2 ;;
        lb) ROUND_REQUESTS_LB=$2 ;;
        dynam) ROUND_REQUESTS_DYNAM=$2 ;;
    esac
}

get_mode_round_requests_total() {
    case "$1" in
        bl) printf '%s' "$ROUND_REQUESTS_TOTAL_BL" ;;
        lb) printf '%s' "$ROUND_REQUESTS_TOTAL_LB" ;;
        dynam) printf '%s' "$ROUND_REQUESTS_TOTAL_DYNAM" ;;
    esac
}

set_mode_round_requests_total() {
    case "$1" in
        bl) ROUND_REQUESTS_TOTAL_BL=$2 ;;
        lb) ROUND_REQUESTS_TOTAL_LB=$2 ;;
        dynam) ROUND_REQUESTS_TOTAL_DYNAM=$2 ;;
    esac
}

add_mode_duration() {
    case "$1" in
        bl) TOTAL_DURATION_BL=$((TOTAL_DURATION_BL + $2)) ;;
        lb) TOTAL_DURATION_LB=$((TOTAL_DURATION_LB + $2)) ;;
        dynam) TOTAL_DURATION_DYNAM=$((TOTAL_DURATION_DYNAM + $2)) ;;
    esac
}

set_mode_status() {
    case "$1" in
        bl) STATUS_BL=$2 ;;
        lb) STATUS_LB=$2 ;;
        dynam) STATUS_DYNAM=$2 ;;
    esac
}

set_mode_round_start() {
    case "$1" in
        bl) ROUND_START_BL=$2 ;;
        lb) ROUND_START_LB=$2 ;;
        dynam) ROUND_START_DYNAM=$2 ;;
    esac
}

set_mode_decoder_started() {
    case "$1" in
        bl) DECODER_STARTED_BL=$2 ;;
        lb) DECODER_STARTED_LB=$2 ;;
        dynam) DECODER_STARTED_DYNAM=$2 ;;
    esac
}

set_mode_decoder_start() {
    case "$1" in
        bl) DECODER_START_BL=$2 ;;
        lb) DECODER_START_LB=$2 ;;
        dynam) DECODER_START_DYNAM=$2 ;;
    esac
}

set_mode_startup_done() {
    case "$1" in
        bl) STARTUP_DONE_BL=$2 ;;
        lb) STARTUP_DONE_LB=$2 ;;
        dynam) STARTUP_DONE_DYNAM=$2 ;;
    esac
}

set_mode_drain_start() {
    case "$1" in
        bl) DRAIN_START_BL=$2 ;;
        lb) DRAIN_START_LB=$2 ;;
        dynam) DRAIN_START_DYNAM=$2 ;;
    esac
}

set_mode_done() {
    case "$1" in
        bl) MODE_DONE_BL=$2 ;;
        lb) MODE_DONE_LB=$2 ;;
        dynam) MODE_DONE_DYNAM=$2 ;;
    esac
}

set_mode_gap_start() {
    case "$1" in
        bl) ROUND_GAP_START_BL=$2 ;;
        lb) ROUND_GAP_START_LB=$2 ;;
        dynam) ROUND_GAP_START_DYNAM=$2 ;;
    esac
}

set_mode_gap_duration() {
    case "$1" in
        bl) ROUND_GAP_DURATION_BL=$2 ;;
        lb) ROUND_GAP_DURATION_LB=$2 ;;
        dynam) ROUND_GAP_DURATION_DYNAM=$2 ;;
    esac
}

mode_eta_state() {
    case "$1" in
        bl)
            printf 'bl:%s:%s:%s:%s:%s:%s:%s:%s:%s:%s:%s:1' \
                "$PROGRESS_BL" "$ROUND_START_BL" \
                "$ROUND_REQUESTS_BL" "$ROUND_REQUESTS_TOTAL_BL" \
                "$DECODER_STARTED_BL" "$DECODER_START_BL" \
                "$STARTUP_DONE_BL" "$DRAIN_START_BL" "$MODE_DONE_BL" \
                "$ROUND_GAP_START_BL" "$ROUND_GAP_DURATION_BL"
            ;;
        lb)
            printf 'lb:%s:%s:%s:%s:%s:%s:%s:%s:%s:%s:%s:1' \
                "$PROGRESS_LB" "$ROUND_START_LB" \
                "$ROUND_REQUESTS_LB" "$ROUND_REQUESTS_TOTAL_LB" \
                "$DECODER_STARTED_LB" "$DECODER_START_LB" \
                "$STARTUP_DONE_LB" "$DRAIN_START_LB" "$MODE_DONE_LB" \
                "$ROUND_GAP_START_LB" "$ROUND_GAP_DURATION_LB"
            ;;
        dynam)
            printf 'dynam:%s:%s:%s:%s:%s:%s:%s:%s:%s:%s:%s:1' \
                "$PROGRESS_DYNAM" "$ROUND_START_DYNAM" \
                "$ROUND_REQUESTS_DYNAM" "$ROUND_REQUESTS_TOTAL_DYNAM" \
                "$DECODER_STARTED_DYNAM" "$DECODER_START_DYNAM" \
                "$STARTUP_DONE_DYNAM" "$DRAIN_START_DYNAM" "$MODE_DONE_DYNAM" \
                "$ROUND_GAP_START_DYNAM" "$ROUND_GAP_DURATION_DYNAM"
            ;;
    esac
}

record_eta_sample() {
    local metric=$1
    local mode=$2
    local value=$3
    local args=(
        record
        --history-file "$ETA_HISTORY_FILE"
        --dataset-name "$DATASET_NAME"
        --num-prompts "$NUM_PROMPTS"
        --max-concurrency "$MAX_CONCURRENCY"
        --request-rate "$REQUEST_RATE"
        --disable-shuffle "$DISABLE_SHUFFLE"
        --initial-startup-delay "$INITIAL_STARTUP_DELAY"
        --ready-settle-delay "$READY_SETTLE_DELAY"
        --metric "$metric"
        --value "$value"
    )

    if [ -n "$mode" ]; then
        args+=(--mode "$mode")
    fi
    python3 "$ETA_HISTORY_SCRIPT" "${args[@]}" >/dev/null
}

format_duration() {
    local total_seconds=${1:-0}
    local hours=$((total_seconds / 3600))
    local minutes=$(((total_seconds % 3600) / 60))
    local seconds=$((total_seconds % 60))

    if [ "$hours" -gt 0 ]; then
        printf '%dh%02dm%02ds' "$hours" "$minutes" "$seconds"
    elif [ "$minutes" -gt 0 ]; then
        printf '%dm%02ds' "$minutes" "$seconds"
    else
        printf '%ds' "$seconds"
    fi
}

format_epoch() {
    local epoch=$1
    if date -d "@$epoch" +'%Y-%m-%d %H:%M:%S' >/dev/null 2>&1; then
        date -d "@$epoch" +'%Y-%m-%d %H:%M:%S'
    else
        date -r "$epoch" +'%Y-%m-%d %H:%M:%S'
    fi
}

make_progress_bar() {
    local completed=$1
    local total=$2
    local width=28
    local filled=$((completed * width / total))
    local empty=$((width - filled))
    local bar=""
    local i

    for ((i = 0; i < filled; i++)); do
        bar="${bar}#"
    done
    for ((i = 0; i < empty; i++)); do
        bar="${bar}-"
    done
    printf '%s' "$bar"
}

mode_progress_units() {
    local mode=$1
    local completed_rounds
    local round_requests
    local round_requests_total
    local units

    completed_rounds=$(get_mode_progress "$mode")
    round_requests=$(get_mode_round_requests "$mode")
    round_requests_total=$(get_mode_round_requests_total "$mode")
    units=$((completed_rounds * 1000))

    if [ "$round_requests_total" -gt 0 ]; then
        units=$((units + round_requests * 1000 / round_requests_total))
    fi

    printf '%s' "$units"
}

mode_progress_percent() {
    local units
    local total_units
    local tenths

    units=$(mode_progress_units "$1")
    total_units=$((NUM_ROUNDS * 1000))
    tenths=$((units * 1000 / total_units))
    printf '%d.%d%%' $((tenths / 10)) $((tenths % 10))
}

eta_text() {
    local now
    local eta_epoch
    local remaining_seconds
    local confidence
    local min_progress
    local mode
    local output
    local args
    now=$(date +%s)
    min_progress=$(printf '0.%02d' "$ETA_PROGRESS_MIN_PERCENT")
    args=(
        estimate
        --history-file "$ETA_HISTORY_FILE"
        --dataset-name "$DATASET_NAME"
        --num-prompts "$NUM_PROMPTS"
        --max-concurrency "$MAX_CONCURRENCY"
        --request-rate "$REQUEST_RATE"
        --disable-shuffle "$DISABLE_SHUFFLE"
        --initial-startup-delay "$INITIAL_STARTUP_DELAY"
        --ready-settle-delay "$READY_SETTLE_DELAY"
        --now "$now"
        --script-start "$SCRIPT_START_EPOCH"
        --base-startup-done "$BASE_STARTUP_DONE"
        --num-rounds "$NUM_ROUNDS"
        --round-interval-delay "$ROUND_INTERVAL_DELAY"
        --drain-delay "$DECODER_DRAIN_DELAY"
        --resource-release-delay "$RESOURCE_RELEASE_DELAY"
        --transition-total "$TRANSITION_TOTAL"
        --transition-completed "$TRANSITION_COMPLETED"
        --transition-start "$TRANSITION_START_EPOCH"
        --min-progress "$min_progress"
    )
    for mode in "${MODES[@]}"; do
        args+=(--mode-state "$(mode_eta_state "$mode")")
    done

    if ! output=$(python3 "$ETA_HISTORY_SCRIPT" "${args[@]}" 2>/dev/null); then
        printf '预计完成时间: ETA 计算失败'
        return
    fi
    IFS='|' read -r eta_epoch remaining_seconds confidence <<< "$output"
    if [ "${eta_epoch:-0}" -le 0 ]; then
        printf '预计完成时间: 等待首轮达到 %s%% 或读取到同配置历史数据' \
            "$ETA_PROGRESS_MIN_PERCENT"
        return
    fi
    printf '预计完成时间: %s | 剩余约 %s | 置信度: %s' \
        "$(format_epoch "$eta_epoch")" \
        "$(format_duration "$remaining_seconds")" \
        "$confidence"
}

dashboard_line() {
    printf '\033[2K%s\n' "$1"
}

render_dashboard() {
    [ "$DASHBOARD_ENABLED" -eq 1 ] || return

    local mode
    local completed_rounds
    local duration
    local duration_text
    local round_start
    local current_duration_text
    local status
    local modes_text
    local progress_units
    local progress_percent
    local total_progress_units
    modes_text=$(modes_display)
    total_progress_units=$((NUM_ROUNDS * 1000))

    printf '\0337\033[1;1H'
    dashboard_line "🚀 全自动化流水线测试：${modes_text}"
    dashboard_line "📄 当前数据集: ${DATASET_PATH}"
    dashboard_line \
        "📂 每种模式 ${NUM_ROUNDS} 轮 | 配置每轮 ${NUM_PROMPTS} requests | 日志目录: ${LOG_DIR} | 启动时间: $(format_epoch "$SCRIPT_START_EPOCH")"

    for mode in "${MODES[@]}"; do
        completed_rounds=$(get_mode_progress "$mode")
        duration=$(get_mode_average_duration "$mode")
        round_start=$(get_mode_round_start "$mode")
        status=$(get_mode_status "$mode")
        progress_units=$(mode_progress_units "$mode")
        progress_percent=$(mode_progress_percent "$mode")
        if [ "$completed_rounds" -gt 0 ]; then
            duration_text="$(format_duration "$duration")"
        else
            duration_text="--"
        fi
        if [ "$round_start" -gt 0 ]; then
            current_duration_text="$(format_duration $(($(date +%s) - round_start)))"
        else
            current_duration_text="--"
        fi
        dashboard_line \
            "$(printf '%-5s' "$(mode_label "$mode")") [$(make_progress_bar "$progress_units" "$total_progress_units")] [${progress_percent}] 已完成任务平均耗时: ${duration_text} | 本轮已运行: ${current_duration_text} | ${status}"
    done

    dashboard_line "⏱️  $(eta_text)"
    dashboard_line ""
    dashboard_line ""
    dashboard_line "============================================================"
    printf '\0338'
}

init_dashboard() {
    local required_lines
    required_lines=$((${#MODES[@]} + 7))

    if [ -t 1 ] && [ "${TERM:-dumb}" != "dumb" ] && command -v tput >/dev/null 2>&1; then
        TERM_ROWS=$(tput lines 2>/dev/null || printf '24')
        if [ "$TERM_ROWS" -gt $((required_lines + 4)) ]; then
            DASHBOARD_ENABLED=1
            DASHBOARD_SCROLL_START=$((required_lines + 1))
            printf '\033[2J\033[H\033[?25l'
            printf '\033[%d;%dr' "$DASHBOARD_SCROLL_START" "$TERM_ROWS"
            render_dashboard
            printf '\033[%d;1H' "$DASHBOARD_SCROLL_START"
            return
        fi
    fi

    echo "🚀 全自动化流水线测试：$(modes_display)"
    echo "📄 当前数据集: ${DATASET_PATH}"
    echo \
        "📂 每种模式 ${NUM_ROUNDS} 轮 | 配置每轮 ${NUM_PROMPTS} requests | 日志目录: ${LOG_DIR} | 启动时间: $(format_epoch "$SCRIPT_START_EPOCH")"
}

stop_dashboard() {
    [ "$DASHBOARD_STOPPED" -eq 0 ] || return
    DASHBOARD_STOPPED=1

    if [ "$DASHBOARD_ENABLED" -eq 1 ]; then
        render_dashboard
        printf '\033[r\033[%d;1H\033[?25h' "$TERM_ROWS"
    fi
}

log_event() {
    printf '%s %s\n' "$(date +'%Y%m%d_%H%M%S')" "$*"
}

cleanup() {
    local exit_code=$1
    trap - EXIT

    if [ "$exit_code" -eq 0 ]; then
        log_event "🧹 测试结束，正在释放所有组件资源..."
    else
        log_event "🧹 流程异常退出（code=${exit_code}），正在释放所有组件资源..."
    fi

    [ -n "$BENCHMARK_PID" ] && kill -9 -"$BENCHMARK_PID" 2>/dev/null
    [ -n "$DECODER_PID" ] && kill -9 -"$DECODER_PID" 2>/dev/null
    [ -n "$PROXY_PID" ] && kill -9 -"$PROXY_PID" 2>/dev/null
    [ -n "$PREFILLER_PID" ] && kill -9 -"$PREFILLER_PID" 2>/dev/null

    log_event "✅ 全局清理完成。"
    stop_dashboard
    exit "$exit_code"
}

trap 'cleanup $?' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_ready() {
    local log_file=$1
    local ready_str=$2
    local module_name=$3
    local pid=$4

    log_event "⏳ 等待 ${module_name} 就绪（关键字: '${ready_str}'）..."
    while true; do
        if grep -q "$ready_str" "$log_file" 2>/dev/null; then
            log_event "✅ ${module_name} 已就绪。"
            break
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            log_event "❌ ${module_name} 进程意外退出，请检查日志: ${log_file}"
            return 1
        fi
        render_dashboard
        sleep 2
    done
}

process_progress_file() {
    local progress_file=$1
    local target_mode=$2
    local current_line=0
    local event
    local event_mode
    local round
    local total
    local event_value1
    local event_value2

    [ -f "$progress_file" ] || return

    while IFS='|' read -r event event_mode round total event_value1 event_value2; do
        current_line=$((current_line + 1))
        [ "$current_line" -le "$PROCESSED_PROGRESS_LINES" ] && continue
        [ "$event_mode" = "$target_mode" ] || continue

        case "$event" in
            ROUND_START)
                set_mode_round_start "$target_mode" "$event_value1"
                set_mode_gap_start "$target_mode" 0
                set_mode_gap_duration "$target_mode" 0
                set_mode_round_requests "$target_mode" 0
                set_mode_round_requests_total "$target_mode" 0
                set_mode_status "$target_mode" "第 ${round}/${total} 轮运行中"
                ;;
            ROUND_PROGRESS)
                if [[ "$event_value1" =~ ^[0-9]+$ ]] && \
                    [[ "$event_value2" =~ ^[1-9][0-9]*$ ]] && \
                    [ "$event_value1" -le "$event_value2" ]; then
                    set_mode_round_requests "$target_mode" "$event_value1"
                    set_mode_round_requests_total "$target_mode" "$event_value2"
                    set_mode_status "$target_mode" \
                        "第 ${round}/${total} 轮 ${event_value1}/${event_value2} requests"
                fi
                ;;
            ROUND_END)
                set_mode_round_start "$target_mode" 0
                if [ "$event_value2" -eq 0 ]; then
                    record_eta_sample "round" "$target_mode" "$event_value1"
                    add_mode_duration "$target_mode" "$event_value1"
                    set_mode_progress "$target_mode" "$round"
                    set_mode_round_requests "$target_mode" 0
                    set_mode_round_requests_total "$target_mode" 0
                    set_mode_status "$target_mode" "第 ${round}/${total} 轮完成"
                    if [ "$DASHBOARD_ENABLED" -eq 0 ]; then
                        log_event \
                            "📈 $(mode_label "$target_mode") [${round}/${total}] 已完成任务平均耗时: $(format_duration "$(get_mode_average_duration "$target_mode")") | 本轮已运行: -- | $(eta_text)"
                    fi
                else
                    set_mode_status "$target_mode" "第 ${round}/${total} 轮失败"
                fi
                ;;
            ROUND_GAP_START)
                set_mode_gap_start "$target_mode" "$event_value1"
                set_mode_gap_duration "$target_mode" "$event_value2"
                set_mode_status "$target_mode" \
                    "第 ${round}/${total} 轮后等待 ${event_value2}s"
                ;;
            ROUND_GAP_END)
                set_mode_gap_start "$target_mode" 0
                set_mode_gap_duration "$target_mode" 0
                ;;
        esac
    done < "$progress_file"

    PROCESSED_PROGRESS_LINES=$current_line
}

append_round_averages() {
    local result_file=$1
    local expected_rounds=$2

    python3 - "$result_file" "$expected_rounds" >> "$result_file" <<'PY'
import re
import sys
from decimal import Decimal
from pathlib import Path

result_file = Path(sys.argv[1])
expected_rounds = int(sys.argv[2])
header = re.compile(r"=+\s*Serving Benchmark Result\s*=+", re.IGNORECASE)
footer = re.compile(r"={10,}")
metric = re.compile(
    r"^([^:]+):\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
)
ansi_escape = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

rounds = []
current = None
for raw_line in result_file.read_text(encoding="utf-8", errors="replace").splitlines():
    line = ansi_escape.sub("", raw_line).strip()
    if header.search(line):
        current = {}
        rounds.append(current)
        continue
    if current is None:
        continue
    if footer.fullmatch(line):
        current = None
        continue
    match = metric.fullmatch(line)
    if match:
        value = Decimal(match.group(2))
        if value.is_finite():
            current[match.group(1).strip()] = value

print()
print(f"========== {expected_rounds} 轮结果平均值 ==========")
if len(rounds) != expected_rounds:
    print(f"无法汇总：找到 {len(rounds)} 轮完整结果，预期 {expected_rounds} 轮。")
    sys.exit(1)

labels = [label for label in rounds[0] if all(label in item for item in rounds)]
if not labels:
    print("无法汇总：各轮没有共同的数值指标。")
    sys.exit(1)

width = max(len(label) for label in labels) + 2
for label in labels:
    average = sum(item[label] for item in rounds) / expected_rounds
    print(f"{label + ':':<{width + 1}} {average:.2f}")

print("说明：这里是各轮指标的算术平均；P99 等百分位的平均不是所有请求合并后的百分位。")
PY
}

run_benchmark_mode() {
    local mode=$1
    local result_file
    local progress_file
    local benchmark_status

    result_file=$(mode_result_file "$mode")
    progress_file="${TMPDIR:-/tmp}/automatic_benchmark_${RUN_TIMESTAMP}_${mode}_$$.progress"
    : > "$progress_file"
    PROCESSED_PROGRESS_LINES=0

    set_mode_status "$mode" "Benchmark 启动中"
    render_dashboard
    log_event "🚀 开始运行 $(mode_label "$mode") Benchmark，共 ${NUM_ROUNDS} 轮，每轮 ${NUM_PROMPTS} 个请求，并发 ${MAX_CONCURRENCY}，request rate ${REQUEST_RATE}，disable shuffle ${DISABLE_SHUFFLE}。"

    DATASET_PATH="$DATASET_PATH" \
        MODE="$mode" \
        NUM_PROMPTS="$NUM_PROMPTS" \
        MAX_CONCURRENCY="$MAX_CONCURRENCY" \
        REQUEST_RATE="$REQUEST_RATE" \
        DISABLE_SHUFFLE="$DISABLE_SHUFFLE" \
        ROUND_INTERVAL_DELAY="$ROUND_INTERVAL_DELAY" \
        BENCHMARK_PROGRESS_FILE="$progress_file" \
        bash "$BENCHMARK_SCRIPT" \
            --mode "$mode" \
            --rounds "$NUM_ROUNDS" \
            --num-prompts "$NUM_PROMPTS" \
            --max-concurrency "$MAX_CONCURRENCY" \
            --request-rate "$REQUEST_RATE" \
            --dataset-path "$DATASET_PATH" \
        > "$result_file" 2>&1 &
    BENCHMARK_PID=$!

    while kill -0 "$BENCHMARK_PID" 2>/dev/null; do
        process_progress_file "$progress_file" "$mode"
        render_dashboard
        sleep 1
    done

    wait "$BENCHMARK_PID"
    benchmark_status=$?
    BENCHMARK_PID=""
    process_progress_file "$progress_file" "$mode"
    rm -f "$progress_file"

    if [ "$benchmark_status" -ne 0 ]; then
        set_mode_status "$mode" "Benchmark 失败"
        render_dashboard
        log_event "❌ $(mode_label "$mode") Benchmark 失败（code=${benchmark_status}），请检查: ${result_file}"
        return "$benchmark_status"
    fi

    if ! append_round_averages "$result_file" "$NUM_ROUNDS"; then
        log_event "⚠️ $(mode_label "$mode") 每轮测试已完成，但平均值汇总失败，请检查: ${result_file}"
    fi

    set_mode_status "$mode" "测试完成"
    render_dashboard
    log_event "📊 $(mode_label "$mode") 测试完成，结果已保存至: ${result_file}"
}

release_decoder() {
    local mode=$1

    if [ "$DECODER_DRAIN_DELAY" -gt 0 ]; then
        set_mode_drain_start "$mode" "$(date +%s)"
        set_mode_status "$mode" "Drain 等待 ${DECODER_DRAIN_DELAY}s"
        render_dashboard
        log_event "⏳ $(mode_label "$mode") Benchmark 已结束，保留 Decoder ${DECODER_DRAIN_DELAY}s，等待在途 KV 传输和完成通知收尾..."
        sleep "$DECODER_DRAIN_DELAY"
    fi
    set_mode_drain_start "$mode" 0

    set_mode_status "$mode" "释放资源"
    render_dashboard
    log_event "⏳ 正在终止 Decoder ($(mode_label "$mode")) 进程 [PID: ${DECODER_PID}]..."
    kill -9 -"$DECODER_PID" 2>/dev/null
    wait "$DECODER_PID" 2>/dev/null
    DECODER_PID=""
    set_mode_done "$mode" 1
    set_mode_status "$mode" "已完成"
    render_dashboard
    log_event "✅ Decoder ($(mode_label "$mode")) 资源已释放。"
}

init_dashboard

if [ "$SKIP_INITIAL_PROCESS_CLEANUP" -eq 1 ]; then
    log_event "🧪 已跳过启动前残留进程清理。"
else
    log_event "🧹 清理可能残留的 VLLM、python 和 vllm 进程..."
    pkill -9 VLLM 2>/dev/null || true
    pkill -9 python 2>/dev/null || true
    pkill -9 vllm 2>/dev/null || true
fi

log_event "▶️ 启动 Proxy Server..."
bash "$PROXY_SCRIPT" > "${LOG_DIR}/proxy.log" 2>&1 &
PROXY_PID=$!
wait_for_ready "${LOG_DIR}/proxy.log" "$READY_STR_PROXY" "Proxy" "$PROXY_PID" || exit 1

log_event "▶️ 启动 Prefiller..."
bash "$PREFILLER_SCRIPT" > "${LOG_DIR}/prefiller.log" 2>&1 &
PREFILLER_PID=$!

first_mode=1
mode_index=0
for mode in "${MODES[@]}"; do
    mode_index=$((mode_index + 1))
    decoder_script=$(mode_decoder_script "$mode")
    decoder_log="${LOG_DIR}/decoder_${mode}.log"

    set_mode_status "$mode" "拉起服务"
    render_dashboard

    if [ "$first_mode" -eq 1 ] && [ "$INITIAL_STARTUP_DELAY" -gt 0 ]; then
        log_event "⏳ 基础服务启动缓冲 ${INITIAL_STARTUP_DELAY}s..."
        sleep "$INITIAL_STARTUP_DELAY"
    fi

    if [ "$BASE_STARTUP_DONE" -eq 0 ]; then
        BASE_STARTUP_DONE=1
        record_eta_sample "base_startup" "" \
            "$(($(date +%s) - SCRIPT_START_EPOCH))"
    fi

    log_event "▶️ 启动 Decoder ($(mode_label "$mode"))..."
    decoder_start_epoch=$(date +%s)
    set_mode_decoder_started "$mode" 1
    set_mode_decoder_start "$mode" "$decoder_start_epoch"
    bash "$decoder_script" > "$decoder_log" 2>&1 &
    DECODER_PID=$!

    if [ "$first_mode" -eq 1 ]; then
        wait_for_ready "${LOG_DIR}/prefiller.log" "$READY_STR_PREFILLER" "Prefiller" "$PREFILLER_PID" || exit 1
        first_mode=0
    fi
    wait_for_ready "$decoder_log" "$READY_STR_DECODER" "Decoder_$(mode_label "$mode")" "$DECODER_PID" || exit 1
    if [ "$READY_SETTLE_DELAY" -gt 0 ]; then
        sleep "$READY_SETTLE_DELAY"
    fi
    startup_duration=$(($(date +%s) - decoder_start_epoch))
    set_mode_startup_done "$mode" 1
    record_eta_sample "decoder_startup" "$mode" "$startup_duration"

    run_benchmark_mode "$mode" || exit $?
    release_decoder "$mode"

    if [ "$mode_index" -lt "${#MODES[@]}" ] && [ "$RESOURCE_RELEASE_DELAY" -gt 0 ]; then
        TRANSITION_START_EPOCH=$(date +%s)
        log_event "⏳ 等待底层硬件资源完全释放（${RESOURCE_RELEASE_DELAY}s）..."
        render_dashboard
        sleep "$RESOURCE_RELEASE_DELAY"
        TRANSITION_START_EPOCH=0
        TRANSITION_COMPLETED=$((TRANSITION_COMPLETED + 1))
    fi
done

log_event "🎉 所有自动化测试流程执行完毕！"

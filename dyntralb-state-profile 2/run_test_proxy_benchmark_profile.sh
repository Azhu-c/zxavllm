#!/bin/bash

MODE="${MODE:-${NOTE:-}}"
NUM_ROUNDS="${NUM_ROUNDS:-1}"
NUM_PROMPTS="${NUM_PROMPTS:-1000}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-128}"
REQUEST_RATE="${REQUEST_RATE:-400}"
DISABLE_SHUFFLE="${DISABLE_SHUFFLE:-false}"
ROUND_INTERVAL_DELAY="${ROUND_INTERVAL_DELAY:-5}"
DATASET_PATH="${DATASET_PATH:-./mixed_prompts_lognormal.jsonl}"
BENCHMARK_MODEL="${BENCHMARK_MODEL:-qwen3_30B}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/home/y00906461/models/Qwen3-30B-A3B-Instruct-2507}"
BENCHMARK_TZ="${BENCHMARK_TZ:-UTC-8}"
PROGRESS_FILE=${BENCHMARK_PROGRESS_FILE:-}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROGRESS_FILTER="${SCRIPT_DIR}/benchmark_progress_filter_profile.py"
RESULT_DIR="${RESULT_DIR:-${SCRIPT_DIR}/results}"
export TZ="$BENCHMARK_TZ"

usage() {
    cat <<'EOF'
用法:
  ./run_test_proxy_benchmark_manualPrompts_multiRound.sh [模式] [轮数] [数据集路径] [每轮请求数]
  ./run_test_proxy_benchmark_manualPrompts_multiRound.sh --mode bl --rounds 10 --num-prompts 1000 --dataset-path Dataset-20260717-0930.jsonl
  ./run_test_proxy_benchmark_manualPrompts_multiRound.sh --mode bl --rounds 2 --num-prompts 100 --dataset-path Dataset-20260717-0930.jsonl --max-concurrency 1 --request-rate 1 --disable-shuffle

选项:
  -m, --mode MODE           测试模式；支持 bl、lb、dynam
  -n, --rounds N            运行轮数，默认 1
  -p, --num-prompts N       每轮 benchmark 的 request 数，默认 1000
  -d, --dataset-path P      数据集 JSONL 路径
      --max-concurrency N   最大并发请求数，默认 128
      --request-rate RATE   每秒请求数，默认 400；也支持 inf
      --disable-shuffle     禁止 benchmark 打乱数据集
      --round-interval N    多轮 benchmark 之间的等待秒数，默认 5
  -h, --help                显示帮助

结果默认保存在脚本目录下的 results 文件夹。
也可以通过 MODE、NUM_ROUNDS、NUM_PROMPTS、MAX_CONCURRENCY、REQUEST_RATE、
DISABLE_SHUFFLE、ROUND_INTERVAL_DELAY、DATASET_PATH、RESULT_DIR 和
BENCHMARK_MODEL、TOKENIZER_PATH、BENCHMARK_TZ 环境变量配置。
EOF
}

POSITIONAL_INDEX=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        -m|--mode|--note)
            [ "$#" -ge 2 ] || { echo "❌ $1 缺少参数" >&2; exit 2; }
            MODE=$2
            shift 2
            ;;
        -n|--rounds)
            [ "$#" -ge 2 ] || { echo "❌ $1 缺少参数" >&2; exit 2; }
            NUM_ROUNDS=$2
            shift 2
            ;;
        -p|--num-prompts)
            [ "$#" -ge 2 ] || { echo "❌ $1 缺少参数" >&2; exit 2; }
            NUM_PROMPTS=$2
            shift 2
            ;;
        -d|--dataset-path)
            [ "$#" -ge 2 ] || { echo "❌ $1 缺少参数" >&2; exit 2; }
            DATASET_PATH=$2
            shift 2
            ;;
        --max-concurrency)
            [ "$#" -ge 2 ] || { echo "❌ $1 缺少参数" >&2; exit 2; }
            MAX_CONCURRENCY=$2
            shift 2
            ;;
        --request-rate)
            [ "$#" -ge 2 ] || { echo "❌ $1 缺少参数" >&2; exit 2; }
            REQUEST_RATE=$2
            shift 2
            ;;
        --disable-shuffle)
            DISABLE_SHUFFLE=true
            shift
            ;;
        --round-interval)
            [ "$#" -ge 2 ] || { echo "❌ $1 缺少参数" >&2; exit 2; }
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
            case "$POSITIONAL_INDEX" in
                0) MODE=$1 ;;
                1) NUM_ROUNDS=$1 ;;
                2) DATASET_PATH=$1 ;;
                3) NUM_PROMPTS=$1 ;;
                *)
                    echo "❌ 多余参数: $1" >&2
                    usage >&2
                    exit 2
                    ;;
            esac
            POSITIONAL_INDEX=$((POSITIONAL_INDEX + 1))
            shift
            ;;
    esac
done

MODE=$(printf '%s' "$MODE" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')
case "$MODE" in
    bl|lb|dynam)
        ;;
    *)
        echo "❌ 测试模式必须是 bl、lb 或 dynam，当前值: ${MODE:-<空>}" >&2
        exit 2
        ;;
esac

if ! [[ "$NUM_ROUNDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "❌ 运行轮数必须是正整数，当前值: $NUM_ROUNDS" >&2
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

if ! [[ "$ROUND_INTERVAL_DELAY" =~ ^[0-9]+$ ]]; then
    echo "❌ 轮间等待秒数必须是非负整数，当前值: $ROUND_INTERVAL_DELAY" >&2
    exit 2
fi

if [ ! -f "$DATASET_PATH" ]; then
    echo "❌ 数据集文件不存在: $DATASET_PATH" >&2
    exit 2
fi

emit_progress() {
    [ -n "$PROGRESS_FILE" ] || return
    printf '%s|%s|%s|%s|%s|%s\n' \
        "$1" "$MODE" "$2" "$NUM_ROUNDS" "${3:-0}" "${4:-0}" \
        >> "$PROGRESS_FILE"
}

# 1. 环境准备
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/home/y00906461/20260228_vllm_ascend/20260713_pr_vllm_ascend/vllm/

# 2. 代理免疫：确保请求不经过 HIS Proxy
export no_proxy=$no_proxy,61.28.30.29
export NO_PROXY=$NO_PROXY,61.28.30.29

LOG_DIR="./log"
mkdir -p "${LOG_DIR}"
mkdir -p "${RESULT_DIR}"
DATASET_FILENAME="${DATASET_PATH##*/}"
DATASET_NAME="${DATASET_FILENAME%.*}"

echo "🚀 计划连续执行 ${NUM_ROUNDS} 轮压测..."
echo "📄 数据集: ${DATASET_PATH}"
echo "🧪 测试模式: ${MODE}"
echo "📦 每轮请求数: ${NUM_PROMPTS}"
echo "🔀 最大并发数: ${MAX_CONCURRENCY}"
echo "⏱️  Request rate: ${REQUEST_RATE}"
echo "🧷 禁止数据集打乱: ${DISABLE_SHUFFLE}"
echo "🤖 服务模型名: ${BENCHMARK_MODEL}"
echo "🔤 Tokenizer: ${TOKENIZER_PATH}"
echo "💾 结果目录: ${RESULT_DIR}"

SHUFFLE_ARGS=()
if [ "$DISABLE_SHUFFLE" = "true" ]; then
    SHUFFLE_ARGS+=(--disable-shuffle)
fi

# 外层循环控制多轮执行
for (( i=1; i<=NUM_ROUNDS; i++ )); do
    ROUND_START_EPOCH=$(date +%s)
    emit_progress "ROUND_START" "$i" "$ROUND_START_EPOCH"

    # 3. 获取当前时间戳（精确到秒，放在循环内部确保每轮时间不同）
    TIMESTAMP=$(date +"%Y%m%d_%H%M")

    # 4. 定义【压测客户端】专属的日志文件名（加入 round 轮次标识）
    BENCH_LOG="${LOG_DIR}/${TIMESTAMP}_vllm_bench_result_${MODE}_round${i}.log"
    RESULT_FILENAME="${TIMESTAMP}_${DATASET_NAME}_${MODE}_round${i}.json"

    echo "----------------------------------------------------"
    echo "${TIMESTAMP} ▶️ 开始第 ${i}/${NUM_ROUNDS} 轮压测"

    # 5. 启动压测
    BENCHMARK_CMD=(
        vllm bench serve
        --backend openai-chat \
        --host 61.28.30.29 \
        --port 8080 \
        --endpoint /v1/chat/completions \
        --dataset-name custom \
        --dataset-path "$DATASET_PATH" \
        "${SHUFFLE_ARGS[@]}" \
        --custom-output-len -1 \
        --max-concurrency "$MAX_CONCURRENCY" \
        --num-prompts "$NUM_PROMPTS" \
        --model "$BENCHMARK_MODEL" \
        --tokenizer "$TOKENIZER_PATH" \
        --save-result \
        --save-detailed \
        --result-dir "$RESULT_DIR" \
        --result-filename "$RESULT_FILENAME" \
        --temperature 0.0 \
        --metric-percentiles "50,90,99" \
        --request-rate "$REQUEST_RATE"
    )

    if [ -n "$PROGRESS_FILE" ]; then
        "${BENCHMARK_CMD[@]}" 2>&1 | \
            python3 -u "$PROGRESS_FILTER" \
                --progress-file "$PROGRESS_FILE" \
                --mode "$MODE" \
                --round "$i" \
                --total-rounds "$NUM_ROUNDS"
        PIPELINE_STATUS=("${PIPESTATUS[@]}")
        ROUND_STATUS=${PIPELINE_STATUS[0]}
        FILTER_STATUS=${PIPELINE_STATUS[1]}
        if [ "$FILTER_STATUS" -ne 0 ]; then
            echo "⚠️ Benchmark 进度解析器异常退出（code=${FILTER_STATUS}）。" >&2
        fi
    else
        "${BENCHMARK_CMD[@]}"
        ROUND_STATUS=$?
    fi

    ROUND_END_EPOCH=$(date +%s)
    ROUND_DURATION=$((ROUND_END_EPOCH - ROUND_START_EPOCH))
    emit_progress "ROUND_END" "$i" "$ROUND_DURATION" "$ROUND_STATUS"

    if [ "$ROUND_STATUS" -ne 0 ]; then
        echo "❌ 第 ${i}/${NUM_ROUNDS} 轮压测失败，退出码: ${ROUND_STATUS}" >&2
        exit "$ROUND_STATUS"
    fi

    # 6. （可选）如果有多轮，在每轮之间给服务端一点缓冲时间释放连接和资源
    if [ "$i" -lt "$NUM_ROUNDS" ]; then
        GAP_START_EPOCH=$(date +%s)
        emit_progress "ROUND_GAP_START" "$i" \
            "$GAP_START_EPOCH" "$ROUND_INTERVAL_DELAY"
        echo "⏳ 第 ${i} 轮执行完毕，等待 ${ROUND_INTERVAL_DELAY} 秒后开始下一轮..."
        sleep "$ROUND_INTERVAL_DELAY"
        emit_progress "ROUND_GAP_END" "$i"
    fi
done
echo "${TIMESTAMP} ✅ 所有 ${NUM_ROUNDS} 轮压测已全部执行完毕！"

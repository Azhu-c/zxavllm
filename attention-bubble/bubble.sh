#!/usr/bin/env bash
# One entry point for the controlled Attention bubble experiment.
set -euo pipefail

BUBBLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$BUBBLE_DIR"
ACTION="${1:-help}"

if [[ "$ACTION" == help || "$ACTION" == --help || "$ACTION" == -h ]]; then
  echo "用法: bash bubble.sh prepare | prefill | decode | proxy | run r1 | report r1"
  exit 0
fi

if [[ "$ACTION" == prepare ]]; then
  python3 "$BUBBLE_DIR/prepare_reference.py" \
    --scripts-dir "$BUBBLE_DIR/.." --output-dir "$BUBBLE_DIR"
  echo "准备完成。请先检查 bubble_config.sh，再按 README 启动服务。"
  exit 0
fi

if [[ ! -f "$BUBBLE_DIR/bubble_config.sh" ]]; then
  echo "找不到 bubble_config.sh，请把完整实验文件复制到当前目录。" >&2
  exit 1
fi
# User edits these values once; they are scoped to this process and its children.
source "$BUBBLE_DIR/bubble_config.sh"
MODEL_PATH="${MODEL_PATH:-}"
HOST_IP="${HOST_IP:-}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3_30B}"
if [[ -z "$MODEL_PATH" || "$MODEL_PATH" == /path/to/your/model || ! -d "$MODEL_PATH" ]]; then
  echo "请在 bubble_config.sh 的 MODEL_PATH 填写服务器上的模型目录。" >&2
  exit 1
fi
if [[ -z "$HOST_IP" || "$HOST_IP" == 服务器的实际IP ]]; then
  echo "请在 bubble_config.sh 的 HOST_IP 填写服务器 IP。" >&2
  exit 1
fi
export MODEL_PATH HOST_IP SERVED_MODEL_NAME
export EXPERIMENT_PYTHON_DIR="$BUBBLE_DIR"
export BUBBLE_STATE_DIR="$BUBBLE_DIR/state"
export HCCL_IF_IP="$HOST_IP"

case "$ACTION" in
  prefill)
    [[ -f "$BUBBLE_DIR/run_vllm_serve_8p8d_asyncSched_p_bubble.sh" ]] || { echo "请先运行 bash bubble.sh prepare" >&2; exit 1; }
    exec bash "$BUBBLE_DIR/run_vllm_serve_8p8d_asyncSched_p_bubble.sh"
    ;;
  decode)
    [[ -f "$BUBBLE_DIR/run_vllm_serve_8p8d_asyncSched_d_local_pairs.sh" ]] || { echo "请先运行 bash bubble.sh prepare" >&2; exit 1; }
    exec bash "$BUBBLE_DIR/run_vllm_serve_8p8d_asyncSched_d_local_pairs.sh"
    ;;
  proxy)
    [[ -f "$BUBBLE_DIR/load_balance_proxy_server_ranked.py" ]] || { echo "请先运行 bash bubble.sh prepare" >&2; exit 1; }
    exec python3 "$BUBBLE_DIR/load_balance_proxy_server_ranked.py" \
      --host "$HOST_IP" --port 8080 \
      --prefiller-hosts "$HOST_IP" --prefiller-ports 13700 \
      --decoder-hosts "$HOST_IP" --decoder-ports 13701
    ;;
  run)
    RUN_ID="${2:-}"
    if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]]; then
      echo "请指定一个运行编号，例如：bash bubble.sh run r1" >&2
      exit 1
    fi
    for URL in "http://$HOST_IP:13700/v1/models" "http://$HOST_IP:13701/v1/models" "http://$HOST_IP:8080/healthcheck"; do
      if ! curl --connect-timeout 5 --max-time 10 -fsS "$URL" > /dev/null; then
        echo "服务尚未就绪: $URL。请检查前三个窗口。" >&2
        exit 1
      fi
    done
    MANIFEST="$BUBBLE_DIR/local-workload/local_pairs.jsonl"
    MODEL_RECORD="$BUBBLE_DIR/local-workload/model_path.txt"
    if [[ ! -f "$MANIFEST" ]]; then
      echo "第一次运行：正在生成实验请求……"
      python3 "$BUBBLE_DIR/local_pairs.py" generate \
        --tokenizer "$MODEL_PATH" --output-dir "$BUBBLE_DIR/local-workload" \
        --short-tokens 2048 --long-tokens 8192 --output-tokens 4096
      printf '%s\n' "$MODEL_PATH" > "$MODEL_RECORD"
    elif [[ ! -f "$MODEL_RECORD" || "$(cat "$MODEL_RECORD")" != "$MODEL_PATH" ]]; then
      echo "现有 workload 的模型路径与 bubble_config.sh 不一致或记录缺失。请使用一个新的实验文件夹。" >&2
      exit 1
    fi
    for MODE in timing profile; do
      for CONDITION in skewed balanced; do
        echo "正在运行 ${CONDITION} / ${MODE} / ${RUN_ID}；本组完成后继续下一组。"
        CMD=(python3 "$BUBBLE_DIR/local_pairs.py" run \
          --manifest "$MANIFEST" --condition "$CONDITION" --mode "$MODE" --run-id "$RUN_ID" \
          --proxy-url "http://$HOST_IP:8080" --decoder-url "http://$HOST_IP:13701" \
          --model "$SERVED_MODEL_NAME" --state-dir "$BUBBLE_STATE_DIR" \
          --output-dir "$BUBBLE_DIR/local-results")
        if [[ "$MODE" == profile ]]; then
          CMD+=(--profile-dir "$BUBBLE_DIR/profile_attention_bubble")
        fi
        "${CMD[@]}"
      done
    done
    echo "四组已完成。查看结果：bash bubble.sh report $RUN_ID"
    ;;
  report)
    RUN_ID="${2:-}"
    if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]]; then
      echo "请指定运行编号，例如：bash bubble.sh report r1" >&2
      exit 1
    fi
    exec python3 "$BUBBLE_DIR/local_pairs.py" report \
      "$BUBBLE_DIR/local-results/skewed_timing_${RUN_ID}.json" \
      "$BUBBLE_DIR/local-results/balanced_timing_${RUN_ID}.json"
    ;;
  *)
    echo "未知命令。运行 bash bubble.sh 查看用法。" >&2
    exit 1
    ;;
esac

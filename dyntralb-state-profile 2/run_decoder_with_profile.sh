#!/usr/bin/env bash
# Run the existing Decoder script while adding profiler flags to vllm serve.
if [ "$#" -ne 1 ] || [ ! -f "$1" ]; then
    echo "用法: bash run_decoder_with_profile.sh 现有Decoder脚本" >&2
    exit 2
fi
PROFILE_ENABLE="${PROFILE_ENABLE:-1}"
PROFILE_DIR="${PROFILE_DIR:-$(pwd -P)/profiling/manual}"
case "$PROFILE_ENABLE" in 0|1) ;; *) echo "PROFILE_ENABLE 只能为 0 或 1" >&2; exit 2 ;; esac
if [ "$PROFILE_ENABLE" = 1 ]; then
    mkdir -p "$PROFILE_DIR" || exit 1
fi
export PROFILE_ENABLE PROFILE_DIR

# Capture scheduler data for this Decoder process and its children.
STATE_CAPTURE_ENABLE="${STATE_CAPTURE_ENABLE:-1}"
case "$STATE_CAPTURE_ENABLE" in 0|1) ;; *) echo "STATE_CAPTURE_ENABLE 只能为 0 或 1" >&2; exit 2 ;; esac
if [ "$STATE_CAPTURE_ENABLE" = 1 ]; then
    STATE_CAPTURE_HOME="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
    STATE_CAPTURE_ROOT="${STATE_CAPTURE_ROOT:-$(pwd -P)/state_dataset}"
    mkdir -p "$STATE_CAPTURE_ROOT" || exit 1
    STATE_CAPTURE_ROOT="$(cd -- "$STATE_CAPTURE_ROOT" && pwd -P)"
    DYNTRA_DATA_DIR="$(mktemp -d "$STATE_CAPTURE_ROOT/run-XXXXXXXX")" || exit 1
    DYNTRA_RUN_ID="$(basename "$DYNTRA_DATA_DIR")"
    export STATE_CAPTURE_HOME DYNTRA_DATA_DIR DYNTRA_RUN_ID STATE_CAPTURE_ENABLE
    echo "调度状态采集目录: $DYNTRA_DATA_DIR"
fi

vllm() {
    # Decoder script may set PYTHONPATH; prepend capture path at the last moment.
    if [ "${STATE_CAPTURE_ENABLE:-0}" = 1 ]; then
        export PYTHONPATH="$STATE_CAPTURE_HOME${PYTHONPATH:+:$PYTHONPATH}"
        export DYNTRA_DATA_DIR DYNTRA_RUN_ID
    fi
    if [ "${1:-}" = serve ] && [ "$PROFILE_ENABLE" = 1 ]; then
        local arg
        for arg in "$@"; do
            case "$arg" in
                --profiler-config|--profiler-config=*)
                    echo "Decoder 脚本已有 --profiler-config，请先移除重复配置。" >&2
                    return 2
                    ;;
                --profiler-config.*)
                    echo "Decoder 脚本已有 profiler 参数，请先移除重复配置。" >&2
                    return 2
                    ;;
            esac
        done
        shift
        if [ "$#" -eq 0 ] || [[ "$1" == -* ]]; then
            echo "vllm serve 后需要紧跟模型路径，请检查 Decoder 脚本。" >&2
            return 2
        fi
        local model=$1
        shift
        command vllm serve "$model" \
            --profiler-config.profiler=torch \
            "--profiler-config.torch_profiler_dir=$PROFILE_DIR" \
            --profiler-config.torch_profiler_with_stack=false \
            --profiler-config.ignore_frontend=true \
            "$@"
    else
        command vllm "$@"
    fi
}
export -f vllm
exec bash "$1"

# 动机实验：本地候选请求与 Attention 空泡

本目录提供两组配对实验。每个 Decode rank 固定拥有 4 个长请求和 4 个短请求，每次同时运行其中 4 个；请求一直属于原来的 rank。

| 条件 | rank 0–3 当前运行 | rank 4–7 当前运行 | 本地等待 |
| --- | --- | --- | --- |
| `skewed` | 每 rank 4 长 | 每 rank 4 短 | 每 rank 另外 4 个请求 |
| `balanced` | 每 rank 2 长 + 2 短 | 每 rank 2 长 + 2 短 | 每 rank 另外 4 个请求 |

两组运行相同的 64 个请求；同一种长度的 prompt 完全相同，前缀缓存必须关闭。脚本通过先后提交两批请求构造这两种状态；`balanced` 是人工指定的参考组合，不是 DyntraLB 实际运行的结果，也不包含运行中的真实换入换出操作。等待请求进入 Decode 等待队列并不代表 KV 已就绪。

实验有两种模式：`profile` 采集稳定窗口内的 Decode trace；`timing` 不开启 profiler，记录全部请求完成的时间。两种模式都会读取 Decode 端的调度快照，核对 8 个 rank 的运行请求和等待请求。快照采集会带来少量额外开销，两组保持相同配置。

**本实验代码尚未在真实 NPU 服务器验证。**参考启动脚本默认 P 使用设备 8–15、D 使用设备 0–7，即总共 16 个设备；实际资源、P/D 地址、设备编号和 P 脚本中写死的 IP 请按服务器配置修改。以下流程要求实验客户端与 Decode 在同一台主机，且访问相同的绝对路径 `BUBBLE_STATE_DIR`。

旧的 32 请求、无等待候选的实验见 [README-legacy.md](README-legacy.md)。

## 0. 准备：只做一次

在 NPU 主机上克隆 [DyntraLB](https://github.com/Ydr-pku/DyntraLB)，使用该仓库配套的 vLLM/vLLM-Ascend 环境；已提交的参考子模块是 vLLM `0351e9aa1fdf1a51329d1906881528dfe61fc88e` 和 vLLM-Ascend `ba795ee1e12604ff9d8454be570f8357c73dca3a`。不同版本的调度接口可能不兼容。

将本目录中的以下文件放到 NPU 主机的同一个目录，例如 `/path/to/DyntraLB/motivation1`：

- `prepare_reference.py`：从仓库原始启动脚本生成实验副本。
- `motivation1.py`：提供请求生成所用函数。
- `local_pairs.py`：生成 64 条请求、运行配对实验、汇总完成时间。
- `scheduler_probe.py`：Decode 调度快照，记录实际运行队列和等待队列，不改变调度决策。
- `trace_metrics.py`：汇总人工核对并导出的 trace 事件。

Python 环境需要 `httpx` 和 `transformers`，以及运行参考代理需要的 `fastapi`、`uvicorn` 等依赖。启动服务前执行：

```bash
export REPO=/path/to/DyntraLB
export EXPERIMENT_DIR="$REPO/motivation1"
python3 "$EXPERIMENT_DIR/prepare_reference.py" --repo "$REPO"
```

脚本会在 `scripts/` 生成排名路由代理、旧 profiling 启动脚本和本实验的 Decode 启动脚本。**原脚本不会修改。**如果之前已经运行过生成命令，需在末尾加 `--force` 来更新这三个已生成的副本。若参考脚本发生变化，生成器会报错并指出不匹配的位置。

## 1. 窗口 1：启动 Prefill，保持运行

把下面示例中的模型路径、IP、设备编号换成服务器实际值。`MODEL_PATH` 是模型文件所在的**目录**，不是模型名称。

```bash
export REPO=/path/to/DyntraLB
export MODEL_PATH=/path/to/model
export SERVED_MODEL_NAME=qwen3_30B
cd "$REPO/scripts"
bash run_vllm_serve_8p8d_asyncSched_p.sh
```

参考 P 脚本中的 IP 和设备需要检查并按实际环境配置。P、D 和代理都是常驻服务：看到服务启动后就保留这个窗口，**不需要等命令退出**。先启动 P：参考 P 脚本启动时会检查已有 vLLM 进程。

## 2. 窗口 2：启动 Decode，保持运行

```bash
export REPO=/path/to/DyntraLB
export MODEL_PATH=/path/to/model
export SERVED_MODEL_NAME=qwen3_30B
export HOST_IP=服务器的实际IP
export EXPERIMENT_PYTHON_DIR="$REPO/motivation1"
export BUBBLE_STATE_DIR="$EXPERIMENT_PYTHON_DIR/state"
cd "$REPO/scripts"
bash run_vllm_serve_8p8d_asyncSched_d_local_pairs.sh
```

`HOST_IP` 是服务器网络接口的地址；不要直接使用上面的中文占位符。此实验脚本设置每个 D rank 最多 4 个运行请求，关闭前缀缓存，启用与参考异步配置相同的 `AsyncScheduler` 子类，并启用 profiler 接口。它要求 `BUBBLE_STATE_DIR` 是绝对路径；调度快照会写在此目录中。

如果报 `scheduler_probe` 找不到，检查 `EXPERIMENT_PYTHON_DIR`；如果报 vLLM 调度器接口错误，检查实际安装的 vLLM 版本。不要在同一组服务上并行运行多个本实验客户端。

## 3. 窗口 3：启动代理，保持运行

```bash
export REPO=/path/to/DyntraLB
export HOST_IP=服务器的实际IP
cd "$REPO/scripts"
python3 load_balance_proxy_server_ranked.py \
  --host "$HOST_IP" --port 8080 \
  --prefiller-hosts "$HOST_IP" --prefiller-ports 13700 \
  --decoder-hosts "$HOST_IP" --decoder-ports 13701
```

如果 P 和 D 在不同机器，`--prefiller-hosts` 和 `--decoder-hosts` 分别填对应地址；客户端和 D 仍需要共享调度快照路径。

## 4. 窗口 4：检查服务、生成请求

三个服务启动后，在第四个终端设置变量。终端之间的 `export` 不会自动共享，所以需要分别设置。

```bash
export REPO=/path/to/DyntraLB
export EXPERIMENT_DIR="$REPO/motivation1"
export BUBBLE_STATE_DIR="$EXPERIMENT_DIR/state"
export MODEL_PATH=/path/to/model
export SERVED_MODEL_NAME=qwen3_30B
export HOST_IP=服务器的实际IP

curl -fsS "http://$HOST_IP:13700/v1/models"
curl -fsS "http://$HOST_IP:13701/v1/models"
curl -fsS "http://$HOST_IP:8080/healthcheck"
```

三个检查均成功后，生成一次请求。默认短上下文约 2K、长上下文约 8K，每个请求生成 4096 tokens，使用模型 tokenizer 计数。

```bash
cd "$EXPERIMENT_DIR"
python3 local_pairs.py generate \
  --tokenizer "$MODEL_PATH" \
  --output-dir "$EXPERIMENT_DIR/local-workload" \
  --short-tokens 2048 --long-tokens 8192 \
  --output-tokens 4096
```

默认最长输入加输出需要模型支持至少 12288 tokens。`--output-tokens` 是每条请求的输出长度，不是全部请求的总长度。若准备等待请求的时间太长导致第一批提前结束，可以提高这个值；总长度仍不能超过 D 端 `--max-model-len`。生成器不会覆盖同名 `local_pairs.jsonl`；需要新配置时换一个输出目录。

## 5. 窗口 4：依次运行配对实验

先运行不采集 profiler 的完成时间对比。每条命令运行结束、终端提示符回来后，才能运行下一条；窗口 1–3 始终保持开启。

```bash
python3 local_pairs.py run \
  --manifest "$EXPERIMENT_DIR/local-workload/local_pairs.jsonl" \
  --condition skewed --mode timing --run-id r1 \
  --proxy-url "http://$HOST_IP:8080" \
  --decoder-url "http://$HOST_IP:13701" \
  --model "$SERVED_MODEL_NAME" \
  --state-dir "$BUBBLE_STATE_DIR" \
  --output-dir "$EXPERIMENT_DIR/local-results"

python3 local_pairs.py run \
  --manifest "$EXPERIMENT_DIR/local-workload/local_pairs.jsonl" \
  --condition balanced --mode timing --run-id r1 \
  --proxy-url "http://$HOST_IP:8080" \
  --decoder-url "http://$HOST_IP:13701" \
  --model "$SERVED_MODEL_NAME" \
  --state-dir "$BUBBLE_STATE_DIR" \
  --output-dir "$EXPERIMENT_DIR/local-results"
```

然后运行两组 profiler。`--profile-dir` 应与 Decode 启动脚本中 `torch_profiler_dir` 的实际位置一致；下面使用脚本默认目录。

```bash
python3 local_pairs.py run \
  --manifest "$EXPERIMENT_DIR/local-workload/local_pairs.jsonl" \
  --condition skewed --mode profile --run-id r1 \
  --proxy-url "http://$HOST_IP:8080" \
  --decoder-url "http://$HOST_IP:13701" \
  --model "$SERVED_MODEL_NAME" \
  --state-dir "$BUBBLE_STATE_DIR" \
  --profile-dir "$REPO/scripts/profile_attention_bubble" \
  --output-dir "$EXPERIMENT_DIR/local-results"

python3 local_pairs.py run \
  --manifest "$EXPERIMENT_DIR/local-workload/local_pairs.jsonl" \
  --condition balanced --mode profile --run-id r1 \
  --proxy-url "http://$HOST_IP:8080" \
  --decoder-url "http://$HOST_IP:13701" \
  --model "$SERVED_MODEL_NAME" \
  --state-dir "$BUBBLE_STATE_DIR" \
  --profile-dir "$REPO/scripts/profile_attention_bubble" \
  --output-dir "$EXPERIMENT_DIR/local-results"
```

每次运行流程：提交选中组 32 条请求 → 等待它们开始输出 → 提交另外 32 条候选 → 核对每个 rank 实际有 4 条指定运行请求和 4 条等待请求 → 持续核对 3 秒稳定窗口 → 等待全部 64 条请求完成。校验要求观察到每个运行请求在快照对应的一步各调度 1 个 token，且没有检测到 preemption。运行开始和结束时的快照也会保存到结果 JSON。

如果第一批提前完成、某个请求失败、调度状态不匹配，或输出 token 数无法核实，运行失败并记录错误。增大输出长度需要重新生成 workload，两个条件必须使用同一个新 workload。两个条件及不同重复均应单独运行；使用 `r2`、`r3` 等新的 `--run-id`，原有结果不会被覆盖。建议交替运行条件并记录实际的软件版本与设备配置。

## 6. 查看整体时间

```bash
python3 local_pairs.py report \
  "$EXPERIMENT_DIR/local-results/skewed_timing_r1.json" \
  "$EXPERIMENT_DIR/local-results/balanced_timing_r1.json"
```

`seconds` 从开始提交第一批请求到最后一条请求完成；`tokens_per_second` 使用 64 条实际完成请求的输出 token 总数除以这个时间。**第二批是在第一批开始输出后提交的**，所以该时间包含两阶段提交和 P/D 处理，用于比较本受控实验中的整批完成时间；不要把它描述成所有请求同时到达的在线负载。程序核对两次记录使用的 workload 内容哈希和模型名称；软件版本、硬件环境和其余服务参数仍需人工保持一致。

汇总还会显示稳定窗口开始时每个 rank 的运行请求输入 token 总数，以及这些请求已生成 token 数的范围。两组的请求到达和预填充完成时间可能不同，采样时实际上下文长度也可能随已生成 token 增长；解读 trace 前要比较这些进度和 JSON 快照中的 `computed_tokens`，不能假设两组在完全相同的生成进度上采样。

`profile` 结果不进入这张时间汇总，避免 profiler 扰动。重复运行的多个 `timing` JSON 可以继续追加到 `report` 命令中，程序会列出每次结果和各条件的中位数。

## 7. 查看 Attention 与通信时间线

`profile` 运行的结果 JSON 中 `new_profile_dirs` 会列出采样期间新增的 `*ascend_pt` 目录；如果为空，请在 Decode 的 `profile_attention_bubble` 路径下检查实际输出、服务日志和 profiler 配置。按 [vLLM-Ascend profiling 指南](https://docs.vllm.ai/projects/ascend/en/main/developer_guide/performance_and_debug/service_profiling_guide.html)对每个目录分析，例如：

```bash
python3 - /path/to/one_ascend_pt <<'PY'
import sys
from torch_npu.profiler.profiler import analyse
analyse(sys.argv[1])
PY
```

分析后查看 `ASCEND_PROFILER_OUTPUT/trace_view.json`。需要先在真实 trace 中确认：各文件对应哪个 rank；哪些事件是同一 decode step、同一层的 Attention 和其后的 EP 通信；以及不同 rank 的时间戳能否对齐。调度快照时间是 CPU 时间，不能当成 NPU Attention 或通信时间。

若确认了这些信息，可将事件整理成 CSV，每行一个 `step,layer,rank`，时间单位为微秒且使用已经核实对齐的公共时钟：

```text
step,layer,rank,attention_start_us,attention_end_us,collective_ready_us,step_start_us,step_end_us
```

`collective_ready_us` 指所选通信的可比较就绪点；不要未经验证就把 CPU 端调用时间或整个通信算子时长当成实际等待。`step_start_us`、`step_end_us` 是同一 rank 的同一 decode step 边界，在同一 step 的各 layer 行中保持一致。然后运行：

```bash
python3 trace_metrics.py /path/to/verified-events.csv \
  --ranks 8 --output "$EXPERIMENT_DIR/local-results/skewed-trace-summary.json"
```

脚本检查每组是否包含 8 个 rank，计算逐 rank Attention 耗时、最早到最晚的就绪时间差、各 rank 与最晚就绪点的差距，以及 trace 中跨 rank 的 step 时间跨度。这些差距是**到达偏差**，不能直接当作实测设备空闲时间或纯通信耗时。事件提取、步数/层数对应和跨设备时钟对齐需要基于真实 trace 核验；本代码不会自动识别它们。

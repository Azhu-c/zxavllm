# 实验 1：长短上下文导致的 Attention 跨 rank 等待

这个目录有两个脚本：`motivation1.py` 生成同一批混合请求并按指定 DP rank 发送；`prepare_reference.py` 根据 [DyntraLB 的 scripts](https://github.com/Ydr-pku/DyntraLB/tree/main/scripts) 生成可固定路由、可采集 Decode profiler 的脚本副本。原始文件不会改动。

## 实验变量

默认生成 32 个请求：16 个约 2K token 的短上下文、16 个约 8K token 的长上下文，所有请求生成 1024 token。

| 条件 | 每个 Decode rank 的请求 |
| --- | --- |
| balanced | 2 长 + 2 短 |
| skewed | rank 0–3 各 4 长；rank 4–7 各 4 短 |

两个条件复用完全相同的 prompt 和输出长度，改变的只有目标 Decode rank。生成器使用实际 tokenizer 计数，并保存长度与 prompt 哈希。`mixed_prompts.jsonl` 兼容参考仓库的 custom dataset 格式；它不含 rank 路由信息，正式实验请使用本目录客户端读取 `placement.jsonl`。

## 运行顺序：四个终端窗口

**Prefill、Decode、代理都是常驻服务。启动后保留各自的终端窗口，不需要等命令结束。** 第四个窗口运行请求客户端；它跑完一组实验后会退出。以下命令以参考脚本中的 `61.28.30.29` 为例，若 NPU 机器地址不同，四个窗口都要使用实际地址。

### 准备一次

在 NPU 机器上保留 [DyntraLB 仓库](https://github.com/Ydr-pku/DyntraLB)的本地副本，并把本目录的 `motivation1.py`、`prepare_reference.py` 复制过去。客户端需要 `httpx`，生成请求还需要 `transformers`。先执行：

```bash
python3 prepare_reference.py --repo /path/to/DyntraLB
```

它在仓库 `scripts/` 下生成 `load_balance_proxy_server_ranked.py` 和 `run_vllm_serve_8p8d_asyncSched_d_profile.sh`，原始脚本仍在。若要重新生成，添加 `--force`。

### 窗口 1：启动 Prefill，保持运行

```bash
cd /path/to/DyntraLB/scripts
export MODEL_PATH=/path/to/model
export SERVED_MODEL_NAME=qwen3_30B
bash run_vllm_serve_8p8d_asyncSched_p.sh
```

参考 Prefill 脚本把 IP `61.28.30.29` 和设备 `8–15` 写在文件中。若机器配置不同，先改好脚本中所有对应的 IP 和设备设置。**先启动 Prefill**：参考脚本会在启动前检查是否已有 vLLM 进程。

### 窗口 2：启动 Decode，保持运行

```bash
cd /path/to/DyntraLB/scripts
export MODEL_PATH=/path/to/model
export SERVED_MODEL_NAME=qwen3_30B
export HOST_IP=61.28.30.29
bash run_vllm_serve_8p8d_asyncSched_d_profile.sh
```

这个脚本以原始无 DyntraLB 的 Decode 配置为基础，增加了 profiler 配置；默认使用设备 `0–7`。本实验先不启用 MTP，以保持每请求每步一个 decode token。

### 窗口 3：启动代理，保持运行

```bash
cd /path/to/DyntraLB/scripts
export HOST_IP=61.28.30.29
python3 load_balance_proxy_server_ranked.py \
  --host "$HOST_IP" --port 8080 \
  --prefiller-hosts "$HOST_IP" --prefiller-ports 13700 \
  --decoder-hosts "$HOST_IP" --decoder-ports 13701
```

**看到代理启动并持续占用终端是正常的，不要等它“跑完”。** 参考仓库的 `run_proxy_server.sh` 使用 `--prefiller-port`，而代理参数名是 `--prefiller-ports`；这里已使用正确参数。若 P、D 分布在不同主机，分别填写它们可达的地址。

### 窗口 4：检查服务、生成请求、运行实验

```bash
export HOST_IP=61.28.30.29
export MODEL_PATH=/path/to/model
export SERVED_MODEL_NAME=qwen3_30B

curl -fsS "http://$HOST_IP:13700/v1/models"
curl -fsS "http://$HOST_IP:13701/v1/models"
curl -fsS "http://$HOST_IP:8080/healthcheck"
```

三个检查都成功后，在保存 `motivation1.py` 的目录运行：

```bash
python3 motivation1.py generate \
  --tokenizer "$MODEL_PATH" \
  --output-dir ./motivation1-workload \
  --short-tokens 2048 --long-tokens 8192 \
  --output-tokens 1024

python3 motivation1.py run \
  --manifest ./motivation1-workload/placement.jsonl \
  --condition balanced \
  --proxy-url "http://$HOST_IP:8080" \
  --decoder-url "http://$HOST_IP:13701" \
  --model "$SERVED_MODEL_NAME" \
  --run-id r1 \
  --output-dir ./motivation1-results

python3 motivation1.py run \
  --manifest ./motivation1-workload/placement.jsonl \
  --condition skewed \
  --proxy-url "http://$HOST_IP:8080" \
  --decoder-url "http://$HOST_IP:13701" \
  --model "$SERVED_MODEL_NAME" \
  --run-id r1 \
  --output-dir ./motivation1-results
```

`generate` 只生成文件，也可以在启动三个服务前执行。两个 `run` 命令需要依次执行：等 balanced 返回终端提示符，再运行 skewed。整个过程中，窗口 1–3 的服务始终保持运行。

客户端对每个请求发送 `X-data-parallel-rank`。代理将该头转发给 Decode，并在响应中回传 `X-Requested-DP-Rank` 和代理内部请求 ID。客户端会核对目标 rank，并保存代理请求 ID 以便对应服务日志；若没有运行修改后的代理，会直接报错。vLLM 的 [DP 路由文档](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/rl.html)说明了该请求头。

客户端等待所有请求收到第一个输出片段，再开始 Decode profiler。若有请求在采样窗口前或窗口内完成，记录会标为无效；可提高 `--output-tokens` 重新生成。`balanced_r1.json` 和 `skewed_r1.json` 保存每个请求的目标 rank、长度、首 token 时间、完成时间和采样窗口状态。重复运行时改用 `--run-id r2`；脚本会拒绝覆盖已有记录。若客户端与 Decode 在同一台机器，可传入 `--profile-dir /absolute/path/to/profile_attention_bubble`，结果中会列出本轮新生成的 profiler 目录。

## 看什么结果

Decode profiler 默认写入启动脚本工作目录下的 `profile_attention_bubble`。按 [Ascend 服务 profiling 指南](https://docs.vllm.ai/projects/ascend/en/main/developer_guide/performance_and_debug/service_profiling_guide.html)对各 `*ascend_pt` 目录运行 `torch_npu.profiler.profiler.analyse`，再打开每个 rank 的 `ASCEND_PROFILER_OUTPUT/trace_view.json`。

比较同一层、同一步中各 rank 的 Attention 结束及其后 EP 集合通信开始时间。先确认 profiler 确实显示 8 个 rank 的算子时间线，并从服务日志核对没有 KV preemption 或异常重算；再计算早到 rank 到最晚 rank 的到达差。集合通信调用总耗时包含等待和实际传输，不能整体标成“通信开销”。这一步需要在拿到真实 NPU trace 后核对事件名称与时钟对齐方式，不能仅凭客户端延迟推断 Attention 空泡。

`prepare_reference.py` 只适配所引用仓库当前 `main` 脚本的代码结构；仓库更新后若匹配位置变化，脚本会报出具体未匹配的位置，避免静默生成错误版本。

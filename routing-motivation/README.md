# RR / JSQ / KV-aware 动机实验

将整个 `routing-motivation` 文件夹上传到：

```
/workspace/ycp/vllm_serve_run/routing-motivation
```

在你平时启动 vLLM 的 Python 环境中运行。源码和启动脚本可以继续放在各自原来的目录。实验会读取当前 Python 环境找到的 vLLM / vLLM-Ascend，不要求它们与启动脚本在同一目录。

## 只改一个配置文件

打开 `config.json`，先改前三项：

```json
"model_path": "/实际的模型目录",
"host_ip": "服务器实际IP",
"served_model": "你使用的服务模型名"
```

`model_path` 是包含 `config.json` 和 tokenizer 文件的目录。`scripts_dir` 默认 `..`，即 `/workspace/ycp/vllm_serve_run`；无需设置 REPO 或一串 export。

默认读取该目录下三个已有启动脚本：

- `run_vllm_serve_8p8d_asyncSched_p.sh`
- `run_vllm_serve_8p8d_asyncSched_d_bl.sh`
- `run_proxy_server.sh`

若你的文件名不同，只改配置中的对应文件名。设备分配、模型并行和网络配置沿用这些脚本。默认端口为 8080 / 13700 / 13701；实验副本中的 proxy 主机和端口同步使用配置值，支持当前“一个 P 入口、一个内部 DP 的 D 入口”布局。模型最大上下文需至少为 10,240 tokens，才能容纳最长输入加输出。

## 一个窗口运行

服务器空闲后，在这个目录运行：

```bash
cd /workspace/ycp/vllm_serve_run/routing-motivation
bash run.sh validate check1 --policies rr
```

这一步自动生成数据、启动 P/D 和 proxy、跑 16 条短输出请求，并关闭本次启动的服务。它用于确认环境与请求链路。通过后运行正式实验：

```bash
bash run.sh run r1
```

不用另开窗口，也不用先启动 `automatic_benchmark.sh`。程序会依次完成 RR、JSQ、KV-aware，各重复三次，结束后直接打印结果。P 和 proxy 在本次实验中复用，每次切换策略或重复轮次都重新启动 Decoder、重新预热；总计九次 Decoder 启动，耗时包含模型加载。每轮策略顺序轮换。

需要 attention 时间线时，再运行独立采样：

```bash
bash run.sh profile p1
```

三种策略各采一次，自动调用 Decoder 的 `/start_profile` 和 `/stop_profile`，并在需要时调用 Torch-NPU 的 `analyse` 解析原始目录。采样默认从第 12 秒开始、持续 5 秒；可在配置中修改。采样采用 eager 执行以记录 step/layer 标记；正式性能实验保留原执行模式。两者分开报告，不能将 eager 采样的耗时直接当作正式性能耗时。

之后随时重新汇总：

```bash
bash run.sh report r1
bash run.sh report p1
```

## 三种分发具体是什么

| 名称 | 此脚本的选择规则 | 单独运行 |
| --- | --- | --- |
| RR | 按 rank 循环，不看负载 | `bash run.sh run rr1 --policies rr` |
| JSQ | 选当前 API client 未完成请求数最少的 rank；平局轮换 | `bash run.sh run jsq1 --policies jsq` |
| KV-aware | 选逻辑 KV blocks 总量最少的 rank，包含 running、waiting 和已发送但尚未反映在状态报告中的请求；平局轮换 | `bash run.sh run kv1 --policies kv` |

这里的 KV-aware 是一个明确的贪心基线，**不称作 BR / BalanceRoute 的复现**。如果第三条曲线暂时使用这个基线，论文图例应写 `KV-aware`。本实验不提供 `--policies br`。

你给的 `automatic_benchmark.sh --modes bl` 用于选择 baseline Decoder。它没有把入口分发改成 RR 或 JSQ。本工具继续使用同一个 BL Decoder，固定关闭 DyntraLB 本地置换和 recompute scheduler，只改变入口路由。

检查过的 vLLM `58d3918` 实现中，默认 DP 路由使用未完成请求数下界、running/waiting 快照，并在 KV 压力较高时增加 waiting 惩罚。因此默认实现不能直接标作严格 JSQ。这里的 JSQ 明确使用单 API server 的未完成请求计数，完成后沿用原生回调减计数。

路由通过本实验进程里的 Python 钩子切换，保留原生 ADD 发送、wave、完成和 abort 记账。**不用手工修改安装目录，也不用再叠加之前的 KV 路由补丁。** 已知旧路由开关会在实验 Decoder 中关闭。兼容检查针对已检查过的 DPLBAsyncMPClient 接口；版本不符会停止，并在 `runtime-check.json` 保存源码位置、版本和选择函数。

状态采集对三个策略都启用，默认每 100 ms 从每个 Decoder rank 拉取一次状态；KV-aware 复用同一份缓存，不为每个请求额外拉取全局状态。状态反映之前的时刻，路由日志保留其年龄。新请求按发送序号补计，rank 报告确认收到后去掉补计，避免突发请求都选中同一份旧快照里的低负载 rank。该实现是本实验的周期上报 KV-aware 基线，不与逐请求同步查询的旧版本混称。

## 已构建的请求集

`data/mixed96.jsonl` 和 `data/validate16.jsonl` 已附请求文本，`data/request-plan96.jsonl` 固定了顺序、长度、类别与随机种子。附带文本按项目中提供的 Qwen tokenizer 词表及 BPE merges 的 ASCII 子集计数构建，来源摘要在 `data/metadata.json`。首次在服务器执行时会使用**你的实际模型 AutoTokenizer，离线重新生成并核对长度**，覆盖这两个生成文件，不从网上下载数据。

| 类别 | 输入 token 数 | 输出 token 数 | 数量 |
| --- | ---: | ---: | ---: |
| SS | 2,048 | 512 | 24 |
| SL | 2,048 | 2,048 | 24 |
| LS | 8,192 | 512 | 24 |
| LL | 8,192 | 2,048 | 24 |

每条提示有不同的随机正文；输入实际长度与目标误差不超过 4 tokens，记录在 JSONL 中。使用 `/v1/completions`，不增加 chat template。三个策略使用同一份提示、顺序和到达计划；输出设置 `ignore_eos=true`、temperature=0。服务返回的实际输出 token 数不匹配时，本轮标为失败。

默认每秒到达 4 条，客户端最多 96 条未完成请求；`request_rate=0` 表示全部同时发出。固定输出长度让这组实验研究长度异质性、请求完成和批次变化，不衡量回答质量。小验证集 `validate16.jsonl` 每类 4 条，输出均为 32 tokens，只检查程序链路。

不手动指定请求的 DP rank，也不强制每个 rank 只能运行 4 条。实际 `max_num_seqs` 沿用你的 Decoder 脚本，并记录在状态数据中。若服务容量足以容纳全部 96 条，没有 WAITING 是一个可能的真实结果。

## 结果看哪些文件

每次实验保存在本目录 `results/你填写的名字/`。

| 文件 | 含义 |
| --- | --- |
| `summary.csv` | 每个策略、每轮的性能、负载不均衡和采样覆盖情况 |
| `policy-medians.csv` | 正式性能实验的重复轮次中位数 |
| `rr-r1/requests.jsonl` 等 | 每条请求的到达、发送、首内容块、结束、token 数、错误、各内容块时间 |
| `rr-r1/load-timeseries.csv` 等 | 各 rank 的 running/waiting 数、逻辑 blocks、物理空闲 blocks，随时间变化 |
| `rr-r1/observations/route-*.jsonl` 等 | 每次路由选择的 rank、评分、输入 blocks、状态年龄 |
| `rr-r1/decoder-effective-args.json` 等 | 实际生效的 Decoder 参数 |
| `rr-r1/profiler/` 等 | profiler 原始输出，采样实验才有 |
| `rr-r1/attention-kernels.csv` 等 | 各 rank **设备 attention kernel** 的数量、均值和分位耗时 |
| `rr-r1/trace-coverage.csv` 等 | trace 是否识别出 rank、设备 kernel、step/layer 关联 |
| `rr-r1/attention-completion-spread.csv` 等 | 可建立关联时，同 wave/step/layer 的 attention 结束时刻差；附时钟假设 |
| `rr-r1/communication-wait-fields.csv` 等 | profiler 若导出 `communication.json`，保留其中显式 wait 字段及原路径 |

主要指标：

- `output_tokens_s`：总生成 token 数 / 整批请求完成时间；越大越好。
- `ttft_p95_ms`：从计划到达到首个内容块的请求级 P95，包含客户端排队；另记录客户端延迟，便于识别压测端堵塞。
- `tpot_p95_ms`：每条请求 `(最后内容块时间−首内容块时间)/(输出token数−1)` 的 P95。MTP 一次返回多个 token 时，不能用相邻 SSE 块间隔直接当 ITL。
- `active_blocks_cv_mean`：每个采样时刻各 rank running 逻辑 blocks 的变异系数，再按时间加权平均；越大表示运行集合越不均衡。
- `owned_blocks_cv_mean`：包含等待与未确认新请求的总逻辑 blocks 的不均衡度；与 active 指标一起看，区分“整体持有负载平衡”和“当前执行批次平衡”。

`logical blocks` 是长度工作量指标，不等同于实际物理 KV 占用，尤其在混合 attention 模型上。物理空闲 blocks 另列。

每轮状态来自一次并发 RPC 拉取，各 rank 的具体 wave/step 都保留在文件中；它是近同时采样，不是一次跨 rank 原子快照。KV-aware 使用的是这些周期更新的状态。

## Attention 空泡怎样解释

脚本自动收集 attention 设备 kernel 和同 step/layer 的标记。均值不均衡代理 `1−mean(各rank平均attention耗时)/max(各rank平均attention耗时)` 描述 attention 耗时差异，**不能当作实际等待比例**。比较 kernel 统计时还要看各 rank 调用数和是否处于同样的工作阶段。

若 exporter 保留了 CPU/device correlation，分析器尝试关联同 wave/step/layer 的 kernel。`completion_spread_us` 是 attention 结束时刻之差，采用同主机 epoch 时间戳的对齐假设；跨设备时钟偏移尚未独立标定。它也不等于 EP 前的纯等待，因为 attention 后还可能有本地计算和重叠工作。

要给论文写出“attention 不均衡导致多少实际空泡”，还需要 trace 中的 EP 通信边界、设备时钟对齐及依赖关系。当前自动汇总将 `attention_bubble_wait_ms` 留为 unavailable，既不会拿 CPU `gloo:all_reduce` 冒充 EP，也不会把 `Free` 或整段通信耗时算成空泡。即使 trace 没有足够信息，吞吐、延迟、负载曲线和可识别的 attention kernel 统计仍会一次性输出；无需为这些指标再运行诊断脚本。

这组实验能检验“KV-aware 初始分发是否降低负载不均衡，以及 active 集合是否仍可能失衡”。它没有开启本地置换，因此不能单靠这组结果宣称置换已带来性能提升。

采样接口及离线解析参考 [vLLM-Ascend 官方指南](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/performance_and_debug/service_profiling_guide.html)。[Torch-NPU 官方说明](https://www.hiascend.com/document/detail/en/Pytorch/2610/apiref/customapi/docs/en/custom_APIs/torch_npu-profiler/torch_npu-profiler-ProfilerLevel.md)指出 Level1 会增加 HCCL 通信分析数据；当前上游 TorchNPUProfilerWrapper 使用 Level1。实际服务器版本与采集能力仍以运行输出为准。

## 运行边界

会生成本实验目录下的脚本副本、数据、日志和结果。原始脚本的全局杀进程与通用共享内存清理在副本中关闭；结束时只关闭本工具启动的进程组。端口已占用时会停止，不会抢占已有服务。NPU 分配沿用你的原启动脚本，仍应在这些设备空闲时运行。

本地已验证路由生命周期、未确认负载记账、流式客户端和汇总链路；本地没有 Ascend，尚未执行真实 NPU 实验。当前脚本与基线来源记录见 `sources.json`；服务器运行时另保存实际版本、配置和数据摘要。

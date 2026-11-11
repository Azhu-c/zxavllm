# 采集脚本与 profiling 目录

说明根据本任务当前的 Bash 脚本和仓库锁定版本的 vLLM / vLLM Ascend 实现核对。
服务器实际安装的 torch_npu / CANN 版本可能影响文件名后缀和可用文件。

## 谁负责什么

1. automatic_benchmark_profile.sh 启动 Proxy、Prefiller 和选定模式的 Decoder，并运行 benchmark。
2. run_decoder_with_profile.sh 接收现有 Decoder 启动脚本，给其中的 vllm serve 增加 profiler 参数。
3. 自动脚本冻结前 N 条请求，在 benchmark 启动前直接向 Decoder 的 /start_profile 发请求。
4. 各个后端 worker 记录全部轮次的 CPU / NPU 活动，不设秒数上限。
5. 全部请求运行结束后，自动脚本调用 /stop_profile，服务停止记录并写出 / 解析数据。
6. 分析采集数据.py 可用 torch_npu 的 analyse 接口补做解析；dyntra_profile_analyze.py 则读取已经生成的 CSV，生成可视化报告。

run_decoder_with_profile.sh 本身只配置采集能力，不决定何时开始 / 停止，
也不决定模型的并行拓扑。它把原有 vllm serve 命令增加以下参数：

    --profiler-config.profiler=torch
    --profiler-config.torch_profiler_dir=采集输出目录
    --profiler-config.torch_profiler_with_stack=false
    --profiler-config.ignore_frontend=true

ignore_frontend=true 是当前本地副本中的配置，用于关闭 API 前端进程的额外采集；
较早的脚本副本可能没有这一项。worker 内的 CPU 和 NPU 采集仍然启用。

## 目录示意

使用自动脚本时：

    profiling/
    └── run_YYYYMMDD_HHMMSS_主Bash进程PID/
        ├── workload/                       固定请求副本和清单
        ├── benchmark_results/              各轮详细结果及工作量核对
        ├── run_config.json
        ├── launch_scripts/
        └── dynam/                          其他模式有各自的 bl/、lb/ 目录
            ├── capture_events.jsonl
            ├── dp0_pp0_tp0_dcp0_epX_rankY_workerPID_时间_ascend_pt/
            │   ├── FRAMEWORK/
            │   ├── PROF_.../
            │   │   ├── host/
            │   │   └── device_.../
            │   ├── profiler_info*.json
            │   ├── profiler_metadata.json
            │   └── ASCEND_PROFILER_OUTPUT/
            │       ├── kernel_details.csv
            │       ├── op_statistic.csv
            │       ├── operator_details.csv
            │       ├── api_statistic.csv
            │       ├── step_trace_time.csv
            │       ├── trace_view.json
            │       ├── communication.json
            │       └── 其他版本相关的 CSV / DB
            ├── dp1_..._ascend_pt/
            ├── ...
            └── dp7_..._ascend_pt/

按当前 Decoder DP=8、TP=1 的部署，dp0–dp7 全部纳入分析。
并非每个文件在每个版本都会生成。直接调用 wrapper、没有传 PROFILE_DIR 时，
默认写入当前目录的 profiling/manual；不会自动创建含模式和 PID 的外层目录。

## 外层目录名

YYYYMMDD_HHMMSS 是自动脚本准备本次运行时的时间，
不是精确的 /start_profile 采集起点。bl / lb / dynam 是测试模式子目录。
末尾 PID 是自动脚本 Bash 进程的 $$，不是某张卡或 worker 的编号。

每个模式采集一次，覆盖该模式全部 benchmark 轮次。默认 1 轮，指定 10 轮时覆盖完整 10 轮。
benchmark 原始详细结果保存在该运行的 benchmark_results/ 中，各轮输出和平均值另存在 log/ 中。

## worker 目录名

vLLM 提供的标识通常包含以下字段：

| 字段 | 含义 |
|---|---|
| dp4 | 该 worker 在数据并行组中的位置 |
| pp0 | 流水线并行位置 |
| tp0 | 张量并行位置 |
| dcp0 | Decode Context Parallel 位置 |
| epX | Expert Parallel 位置 |
| rankY | worker 的分布式 rank 标识。在每个 DP 实例内只有一个 worker（常见 TP=1、PP=1）时，各 DP 目录都可能是 rank0；它不是整个 DP0–DP7 的统一设备序号 |
| workerPID | 写出这份数据的进程 PID |
| 时间 | torch_npu 创建该采集目录时的本地时间，通常包含毫秒 |
| ascend_pt | Ascend PyTorch Profiler 数据目录标识 |

dp4、Device_id=4、Stream 4 是三种不同编号，不能相互替代。
DP 编号本身不能判断 Prefiller / Decoder 角色，应根据启动服务和启停接口确定。
如果同一个 Decoder 服务本身配置 DP=8、TP=1、PP=1，那么它可以生成 dp0–dp7 共 8 个 worker 的目录，并全部带 rank0；应结合实际部署和 CSV 中的 Device_id 确认设备覆盖范围。

## 文件与子目录用途

| 文件或目录 | 含义 |
|---|---|
| FRAMEWORK/ | PyTorch / 框架侧采集的原始记录 |
| PROF_.../ | CANN 采集的原始数据；其中 host 与 device 子目录的确切布局依版本而异 |
| profiler_info*.json | 采集配置、环境及时间等元信息，具体字段依版本而异 |
| profiler_metadata.json | 框架 / 用户补充的元数据，存在性及字段依版本而异 |
| ASCEND_PROFILER_OUTPUT/ | 解析后的可读 / 可查询分析结果 |
| kernel_details.csv | 设备侧算子 / 任务明细，含 Type、Stream、开始时间、Duration 等 |
| op_statistic.csv | 按算子类型等维度汇总的次数、总 / 平均 / 最小 / 最大耗时 |
| operator_details.csv | 框架算子的 Host / Device 耗时及 Self / Total 字段；部分字段可能未采集 |
| api_statistic.csv | 解析器生成的 API 调用统计 |
| step_trace_time.csv | 解析器识别的 step 范围及计算、通信、Free 等拆分；step 不应直接当作请求数或输出 token 数 |
| trace_view.json | CPU、设备任务、通信和分析泳道的时间线，可用于 MindStudio Insight |
| communication*.json | 可用时生成的通信分析、通信矩阵等结果 |
| *.db | 某些版本 / 导出设置下生成的分析数据库 |

每个 worker 都有同名 CSV，是各自的数据分片，不是重复文件。
不要把它们复制到同一目录后互相覆盖。

## 分析范围与关键限制

当前 Decoder 过滤只影响 CSV 分析，不改变 /start_profile 控制的后端 worker 范围。
worker profiling 通常包含 worker 内的 CPU 侧算子和 NPU 任务，并不意味着所有独立调度进程都被采集。
DyntraLB 的全局 CPU 状态 All-gather 仍需调度进程的对应 trace 或单独计时。

出处：

- https://github.com/vllm-project/vllm/blob/0351e9aa1fdf1a51329d1906881528dfe61fc88e/vllm/distributed/utils.py
- https://github.com/Ydr-pku/vllm-ascend/blob/ba795ee1e12604ff9d8454be570f8357c73dca3a/vllm_ascend/worker/worker.py
- https://github.com/Ydr-pku/vllm-ascend/blob/ba795ee1e12604ff9d8454be570f8357c73dca3a/vllm_ascend/profiler/torch_npu_profiler.py
- https://github.com/Ascend/pytorch/blob/master/torch_npu/profiler/_profiler_path_creator.py

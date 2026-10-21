# DyntraLB：TPOT 修复与同实例动态 DP→CP

代码基于 [PR #12292](https://github.com/vllm-project/vllm-ascend/pull/12292) 的 head `2a982061275a8de5f9a05a99ae5adb9d6a1d0b50`，维护于 [Azhu-c/zxavllm 的 dyntralb-tpot-cp 分支](https://github.com/Azhu-c/zxavllm/tree/dyntralb-tpot-cp)。部署脚本由你维护。

目标配置按你确认的信息处理：**Qwen3.5-35B-A3B、Ascend 910、Decode DP8/TP1/EP、MTP=3**。你服务器的 vLLM 为 `0.26.1rc1.dev183`；这里修改的是 vllm-ascend，未修改 vLLM。本地没有 vLLM/torch-npu/NPU，尚不能确认与服务器环境的端到端兼容性。

## 源码安装

在现有服务的容器或虚拟环境中，将完整源码目录切换到本分支后执行：

```bash
python -m pip install -e . --no-deps --no-build-isolation
```

仓库随附固定版本的 CATLASS 头文件压缩包和 SHA256 文件。缺少 CATLASS 时，构建脚本会先校验并解压该包，不要求父源码目录包含 `.git`，也不需要从 GitCode 下载该依赖。请保留 `.gitmodules`、`catlass-41bf90da-headers.tar.gz` 及其 `.sha256` 文件；仅上传 `vllm_ascend/` 不能完成完整源码构建。CANN、编译器及其他构建依赖仍使用服务器环境。详细说明见 [CATLASS 安装修复](INSTALL_CATLASS_OFFLINE.md)。

## TPOT 问题与修改

整体吞吐提高，不代表被暂停请求的 token 间隔降低。原实现还存在可复现的调度问题：

| 问题 | 修改 |
| --- | --- |
| 用 block 数匹配计划，同长度请求不可区分；异步输出跨 block 后匹配失败 | 使用快照槽位身份，在本地解析为请求 ID；应用时不再按实时长度重新匹配 |
| 长请求可能持续暂停 | 默认暂停 4 个 scheduler step 后绕过 DyntraLB 准入限制，优先恢复；满 batch 时轮换可暂停请求 |
| 恢复后马上再次暂停 | 恢复后默认保护 4 步；保留原内存抢占语义 |
| 同一份计划可能重复应用 | 计划消费一次后清空 |
| 缩小与平均负载的距离，却增大全局最大负载 | 禁止这类 swap，例如 `[100]` 与 `[1, waiting=150]` 的错误替换 |
| 搜索最多 1000 次，暂停过多，经验系数写死 | 搜索上限 64；每 rank 每份计划默认最多暂停 1 个原运行请求；暂停收益门槛 2%；延迟模型可配置 |
| 每步协调且在激活窗口外也有额外 collective | 增加统一 `balance_interval`，窗口外不进行 DyntraLB collective，保留原 DP 生命周期同步 |
| 浮点配置可接受 NaN/Infinity | 检查有限性 |
| 有空闲卡时，调度器仍可能暂停可被 CP 加速的长请求 | CP 启用时，规划器保留可借卡的长请求，交给 worker 的动态 CP 路径 |

暂停期限限制的是 DyntraLB 的准入阻塞，不是硬实时 ITL SLA。KV 不足、异步输出未排空、LoRA、token budget 和原请求优先级仍可能推迟执行。FIFO 恢复最早暂停的请求；不会为恢复低优先级请求而暂停更高优先级请求。

## 同一实例内动态借卡 CP

本实现是默认关闭的**实验性双卡 CP 路径**，不是独立实例路由。一个长请求最多借用一个空闲 DP rank；DP8 下其余 rank 保持原来的请求所有权。

1. 在 target forward 的第一个 Full Attention 层，所有 DP rank 交换实际工作快照。只有本次没有本地请求、正在执行 dummy forward 的 rank 才能成为 helper。
2. 给长请求分配 owner/helper lease，仍可用时优先保留上一轮借用关系。多个长请求可以分配不同 helper，数量由 `max_pairs` 限制。
3. owner 保留完整权威 KV、请求状态和采样状态；helper 在独立、受预算限制的缓冲区保存历史 KV 前缀的副本。只复制完整历史 block，MTP 待验证/可能被拒绝的尾部不进入副本。
4. 稳定请求复用前缀，仅增量复制新增 block；request ID 改变或物理 block 表变化时失效重传。
5. owner 和 helper 计算各自 KV 范围的部分注意力，通过 **FP32 log-sum-exp** 合并；不是平均两个 attention output。只有 owner 将完整结果送入后续 gate/o_proj/MoE 和采样流程。
6. MTP 验证批次合并为一次 owner FIA 和一次 helper FIA；使用原生因果 mask，保留每个 query 的可见 KV 范围，避免读到后续 draft token；draft forward 不进入 CP 通信。`disable_padded_drafter_batch=false` 的行为不变。
7. helper 有新本地请求时立即撤销借用，释放副本；内存分配失败时所有 rank 一致回退到原注意力路径。owner 的权威 KV 不迁移、不释放。

Qwen3.5 的 **Full Attention** 层进入该路径，**Gated DeltaNet** 层沿用原逻辑。CP 不会复制 owner 的 MoE token 或额外采样，也不需要修改 P/D connector。

关键代码：

- `vllm_ascend/core/dyntra_cp_policy.py`：配置、拓扑校验和确定性借用策略。
- `vllm_ascend/core/dyntra_cp_runtime.py`：KV 副本、增量传输、P2P 协议、因果可见长度、LSE 合并及回退。
- `vllm_ascend/attention/context_parallel/dynamic_cp_npu.py`：昇腾 FIA 算子适配。
- `vllm_ascend/attention/attention_v1.py`：Full Attention 和 dummy helper 接入。
- `vllm_ascend/ascend_forward_context.py`、`worker/model_runner_v1.py`：每次 target forward 的真实请求身份。

## 配置接口与边界

你的启动脚本由你维护。新增配置位置为 `additional_config.scheduler_config.dyntra_lb_config.dynamic_cp_config`：

```json
{
  "enabled": true,
  "min_context_tokens": 32768,
  "cache_budget_mb": 1024,
  "max_pairs": 1
}
```

这是嵌套的 CP 配置，外层 `dyntra_lb_config.enabled` 也必须为 `true`。`cache_budget_mb` 是每个 helper 跨所有 Full Attention 层的副本预算，不含通信临时缓冲区；需要在服务器为它留出显存。原 KV pool 不缩小，因此 CP 不提供 KV 内存节省。

当前接入：同机内部 DP>1（目标 DP8）、TP1、PP1、model runner v1、**target model eager**、FP16/BF16 KV，允许 MTP 和异步调度。你给出的 drafter `enforce_eager=true` 与 target eager 是不同设置。代码只校验，不改写你的设置。不支持固定 DCP/PCP、图捕获、FlashComm/SP、跨 DP 的细粒度 TP、C8/FP8 KV 或 LoRA；不支持的组合会在启动时报错。MTP/异步与 Hybrid 层的真实组合尚待服务器验证。

`min_context_tokens` 使用实际序列 token 长度，不使用 KV block 阈值；不受 DyntraLB 的 `start_step` 窗口限制。没有空闲 rank 时不进行 CP，继续执行原路径。开启 `dyntra_lb_config.enable_diagnostics` 可在 rank 0 查看 `DYNTRA_CP leases=...`；各层 runtime 的 `stats` 记录借用步数、复制/复用 block 数和分配失败回退数。

初次复制、额外控制通信、额外的 FIA/P2P 协调都会增加开销；不能承诺短上下文获益。是否降低长尾 TPOT、合适的阈值和副本预算，需要服务器实测。当前每个请求 CP degree 为 2，并非把 7 张空闲卡同时分配给一个请求。

TPOT 调优字段仍位于外层 DyntraLB 配置：`balance_interval` 默认 1，可对比 4；`max_pause_steps=4`、`min_run_steps=4`、`max_pauses_per_rank=1`、`min_throughput_gain=0.02`。`max_pauses_per_rank=0` 可用于对比只调整准入的行为。原 `long_req_block_threshold` 仍按逻辑 KV block 计数，随 block size 变化；不要与 CP 的 token 阈值混用。`step_latency_fixed_us=20000`、`step_latency_per_block_us=19.2` 是原经验默认值，需要在实际 MTP/CP 配置上重新标定。

## 验证

本地真实多进程测试使用 Gloo/P2P，FIA 由 CPU 数值参考替代，覆盖 MTP=3 的 4-query 因果验证、前缀增量复用、request/block 复用、借用撤销、多个 owner/helper 和 OOM 回退。它不等于 Ascend HCCL/FIA 或完整模型验证。

```bash
python -m pytest -q tests/standalone/dyntra_lb
```

目标服务器上新增的真实 FIA 数值测试：

```bash
pytest -q tests/ut/attention/a2/test_dynamic_cp_numerics.py
```

并复测原 DyntraLB/recompute/config/platform UT。测试详情、CPU 微基准和服务器尚待验证项见 `DYNTRALB_VALIDATION.md`。保持同一请求序列、到达率、并发、MTP 和总卡数，比较关闭 DyntraLB、原实现、TPOT 修复、TPOT 修复+动态 CP。用 `vllm bench serve --save-result --save-detailed` 保存逐请求结果；`benchmarks/dyntra_lb/compare_serving_results.py` 可分长/短上下文计算 TTFT、TPOT、ITL 的均值和 p50/p95/p99。MTP 的 ITL 是接收 chunk 间隔，TPOT 按 `sum(ITLs)/(output_len-1)` 计算。

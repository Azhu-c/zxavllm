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

TPOT 可以简单理解为：一个请求已经开始回答后，平均每生成一个 token 要等多久。比如某个长请求本来每步都能继续生成，却为了让新请求进来被暂停了几步；新请求的 TTFT 可能变好、所有请求加起来的吞吐可能提高，但这个长请求两次输出之间的等待变长，TPOT 就会上升。下面把代码里发现的问题写成具体发生过程；它们是需要排查和限制的原因，最终收益仍要在昇腾服务器上测量。

| 原来会发生什么，以及为什么可能拖慢 TPOT | 现在怎样处理 |
| --- | --- |
| **暂停后迟迟轮不到恢复。** 为给新请求腾位置，长请求被暂停；接下来如果一直被 DyntraLB 的准入规则挡住，它就无法生成下一个 token，这段空等会计入 TPOT。 | 暂停满默认 4 个调度步后，绕过 DyntraLB 自己的准入限制，优先尝试恢复；满 batch 时可轮换符合条件的运行请求。 |
| **刚恢复，又被下一轮规划暂停。** 同一个请求反复停、跑、再停，即使总吞吐较好，它自身的输出间隔仍可能变长。 | 恢复后默认保护 4 步，期间不因负载均衡再次暂停它。 |
| **计划只记录请求占了多少 KV block，没有准确记住是哪一个请求。** 两个请求都占 128 块时可能选错；从规划到执行之间，异步输出让请求多占了一个 block 时，也可能找不到原计划的对象。错误暂停或计划落空会让调度结果不稳定。 | 规划时记住请求在快照中的位置，执行前解析成实际请求 ID；不再用执行时的 block 数猜请求身份。 |
| **上一轮计划可能被再次执行。** 一次负载调整做完后，如果相同计划在后续调度中再次生效，就可能重复暂停或错误放行请求。 | 每份计划只消费一次，执行后立即清空。 |
| **看起来更“平均”，实际最慢的卡更忙。** 例如原来最忙的卡有 100 块工作，某次交换却让另一张卡接下 150 块；一次 Decode 往往要等参与的卡协调，最慢卡变重可能拖长每步耗时。 | 拒绝让全局最大估计负载上升的交换。 |
| **为了找更好的分配，规划器尝试太多次，也可能暂停太多正在生成的请求。** 规划耗时和被暂停请求的等待都会增加；原来的延迟经验值也未按当前模型和 MTP 配置标定。 | 搜索上限从 1000 降为 64；每张卡每份计划默认最多暂停 1 个原运行请求；预计吞吐收益至少达到 2% 才允许这类暂停；延迟参数可调。 |
| **每个调度步都做一次跨卡负载协调，包括不需要平衡的时间。** 这些通信和规划占用 Decode 的步间时间，较慢的 rank 还会让其他 rank 等待。 | 用 `balance_interval` 控制协调频率；激活窗口外不做 DyntraLB 额外通信。 |
| **有空闲卡时，长请求仍可能先被暂停。** 这样长请求既要等恢复，也失去借空闲卡做 CP 的机会。 | 开启动态 CP 后，规划器优先保留可借卡的长请求，让它进入 CP 路径。 |
| **延迟参数填成 NaN 或 Infinity 时，判断大小和收益门槛会失去正常含义。** 这属于配置正确性问题，会使调度决策难以预期。 | 启动时拒绝这类非有限数值。 |

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

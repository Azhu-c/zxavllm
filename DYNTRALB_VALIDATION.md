# 本地验证记录

日期：2026-10-20。原始代码：PR #12292 head `2a982061275a8de5f9a05a99ae5adb9d6a1d0b50`。
目标：Qwen3.5-35B-A3B、Ascend 910、DP8/TP1/EP、MTP=3；修改只在 vllm-ascend 内，部署脚本不变。

## 已完成

- Python 3.12 / CPU PyTorch 2.10.0：**174 passed**，其中包含真实四进程 Gloo collective/P2P 测试。
- Ruff 0.14.0：所有修改的 Python 文件检查和格式检查通过。
- Git diff 空白检查通过。
- 100 组随机快照验证身份、容量、暂停预算不变量。
- 调度测试覆盖计划跨 block、一次性消费、暂停期限、恢复保护、最早暂停顺序、优先级、在途异步输出、collective cadence、激活窗口、长请求信号保留和 CP 候选保护。
- 动态 CP 测试覆盖两个 owner/helper pair、未参与借用的 rank、MTP 的 4-query 批量因果可见长度、前缀增量复制与复用、相同 block 表的新请求、helper 新本地请求导致撤销、副本内存分配失败回退、kernel 错误向所有 rank 回传、draft/profile 跳过及原始 K/V 参数保留。
- 指标工具测试验证 TPOT 包含暂停间隔，并排除单 token 响应的 TPOT。

运行命令：

```bash
python -m pytest -q tests/standalone/dyntra_lb
```

本机 Gloo 测试用回环接口 `lo0`。Linux 服务器按自己的网络接口设置运行即可；代码不设置网络环境变量。测试使用 CPU 数值参考代替 FIA，直接执行生产 runtime 的通信/缓存/合并逻辑。vLLM scheduler contract 测试替换运行时依赖、执行原方法体，不能代表完整引擎集成。

## CPU 规划开销微基准

随机种子 12292；8 个 rank，每 rank 128 running、128 waiting；前两个 rank 各有一个 4000-block 请求。每个策略运行 10 次。

| 版本 | 规划耗时中位数 | 最大耗时 | 平均暂停决策数 |
| --- | --- | --- | --- |
| 原 PR | 45.857 ms | 46.463 ms | 28 |
| 本修改 | 6.352 ms | 7.248 ms | 0 |

原始 JSON：`validation/dyntra_lb_planner_cpu.json`。

这是同一输入下不同策略的 CPU 开销对比。修改后的收益门槛使此输入产生更少决策与搜索；不能解释为相同算法的等价加速，也不能换算为服务 TTFT/TPOT/吞吐收益。该微基准没有执行模型、CP、NPU、KV transfer 或 MTP。

复现：

```bash
git show 2a982061275a8de5f9a05a99ae5adb9d6a1d0b50:vllm_ascend/patch/platform/patch_dyntra_lb_core.py > /tmp/dyntralb-original-core.py
python benchmarks/dyntra_lb/planner_benchmark.py \
  --baseline-source /tmp/dyntralb-original-core.py --iterations 10 --output /tmp/dyntralb-planner.json
```

## 服务器待验证

本地无 vLLM、torch-npu 或 Ascend NPU。以下尚未执行：

- 新增 `tests/ut/attention/a2/test_dynamic_cp_numerics.py`：真实 FIA 分片与完整因果注意力对照，覆盖 FP16/BF16、query_len=1/4、非连续物理 block 和 Qwen3.5 的 GQA head 形状。
- Ascend HCCL 的真实 P2P、多 DP 空闲/dummy 与 MoE 交错执行。
- 实际 Qwen3.5 的 Full Attention/GDN、MTP、异步调度、P/D KV transfer、取消、抢占与完成生命周期。
- 原仓库完整 DyntraLB/recompute/config/platform UT、全仓库 CI hooks。
- 端到端准确率以及 TTFT/TPOT/吞吐改善幅度。

服务器先运行新增的 FIA 数值测试，再运行已有 DyntraLB/config/platform UT。已有服务可用下面的 E2E 测试检查长请求和新增短请求能否完成；测试不会启动或修改服务配置：

```bash
pytest -q tests/e2e/dyntra_cp \
  --dyntra-cp-url http://YOUR_SERVICE \
  --dyntra-cp-model YOUR_MODEL_ALIAS \
  --dyntra-cp-prompt /path/to/long_prompt.txt
```

长 prompt 应超过 CP token 阈值，且服务应出现空闲 DP rank。用服务端 `DYNTRA_CP leases` 日志确认实际激活；HTTP 成功本身不能证明使用了 CP。可选 `--dyntra-cp-reference-url` 对比同权重、同 MTP、同采样配置但关闭动态 CP 的 greedy 输出。浮点路径导致输出不一致时需进一步检查数值和模型准确率，不能直接忽略。

实现仍是默认关闭的实验路径：target eager、model runner v1、每个长请求最多借用一个 helper。副本额外占用显存；初次复制、控制通信和额外 FIA/P2P 协调有成本。没有硬件结果前，不宣称动态 CP 必然降低 TPOT。

# DyntraLB 状态采集：沿用原 benchmark/profile 流程

## 这次参考了哪个脚本？

本目录以你指定的 `dyntralb_profile/run_decoder_with_profile.sh` 为基础修改，并附带对应的 `automatic_benchmark_profile.sh` 和 benchmark 辅助脚本。整个调用链是：

```text
automatic_benchmark_profile.sh
  → 本目录的 run_decoder_with_profile.sh
    → 你已有的 run_vllm_serve_8p8d_asyncSched_d_dynam.sh
      → vllm serve（启动 DyntraLB Decode）
```

在启动 `vllm serve` 时，包装脚本设置采集目录及 Python 搜索路径，调度器内的采集代码记录每个 rank 的状态、计划和动作。原有 profiler 开关保留。此工具与 bubble 实验无关。

**仍需在服务器所用的 `vllm_ascend/core/dyntra_lb_scheduler.py` 中加一个补丁。** 单独修改 bash 无法获取 Python 调度器内部状态。补丁不改变现有调度逻辑，也不会移除 all-gather；调度行为需要以实际服务器验证为准。

## 一、上传工具，不覆盖原脚本

将 `dyntralb-state-profile.tar.gz` 上传到服务器现有的 DyntraLB `scripts/` 目录。在该目录解压：

```bash
cd /实际路径/你的DyntraLB/scripts
tar -tzf dyntralb-state-profile.tar.gz | head
```

确认目录 `dyntralb-state-profile` 尚不存在后解压：

```bash
tar -xzf dyntralb-state-profile.tar.gz
export CAPTURE_HOME="$PWD/dyntralb-state-profile"
```

保留你现有的 `run_proxy_server.sh`、Prefill 脚本、`run_vllm_serve_8p8d_asyncSched_d_dynam.sh` 和 `benchmark_eta_history.py`；它们仍放在 `scripts/`。本工具不会覆盖这些文件。使用时工作目录保持为 `scripts/`，自动脚本依然按原来的相对路径查找它们。

工具目录包含：自动 benchmark 脚本、修改后的 `run_decoder_with_profile.sh`、benchmark 辅助文件、`capture.patch`、`dyntra_capture.py` 和离线分析 `analyze.py`。

## 二、确认服务实际使用的 Python 代码并打补丁

在**与 Decode 服务相同的 Python/Ascend 环境**下执行：

```bash
python3 -c 'import vllm_ascend.core.dyntra_lb_scheduler as m; print(m.__file__)'
```

例如，如果打印 `/workspace/dyntralb/vllm_ascend/core/dyntra_lb_scheduler.py`，则设置：

```bash
export REPO=/workspace/dyntralb
sha256sum "$REPO/vllm_ascend/core/dyntra_lb_scheduler.py"
```

实际路径不同就替换 `REPO`。确认 Decode 启动脚本没有切换环境或改变代码路径，否则这条命令看到的文件不一定是服务实际加载的文件。

目前补丁仅按本地现有代码包准备，适用的 `dyntra_lb_scheduler.py` SHA-256 是：

```text
98d6d5e33dd98477417d1cd293195cb4a7a6a50ea034dd5b3a4f6b420e60cf28
```

**哈希不一致时不要打补丁。** 请将真实文件路径、哈希和文件内容提供给我，以便按你的服务器版本适配。哈希一致后，确认没有运行中的服务正在使用该文件，推荐在实验用源码副本执行：

```bash
cd "$REPO"
patch --dry-run -p1 < "$CAPTURE_HOME/capture.patch"
```

检查成功后备份并应用：

```bash
if [ -e vllm_ascend/core/dyntra_lb_scheduler.py.before_capture ]; then
  echo '备份已存在，停止；请先检查原文件和备份。'
else
  cp vllm_ascend/core/dyntra_lb_scheduler.py vllm_ascend/core/dyntra_lb_scheduler.py.before_capture
  patch -p1 < "$CAPTURE_HOME/capture.patch"
fi
```

## 三、运行小规模采集

切回你原来的 `scripts/` 目录，使用原有数据集和 dynam 模式。下面的数据集路径必须替换为服务器上的真实路径：

```bash
cd /实际路径/你的DyntraLB/scripts
export CAPTURE_HOME="$PWD/dyntralb-state-profile"
PROFILE_ENABLE=0 STATE_CAPTURE_ENABLE=1 \
  bash "$CAPTURE_HOME/automatic_benchmark_profile.sh" \
  --modes dynam --rounds 1 --num-prompts 100 \
  --dataset-path /实际路径/数据集.jsonl
```

`PROFILE_ENABLE=0` 关闭设备 profiler，`STATE_CAPTURE_ENABLE=1` 开启状态采集。可以将 profiler 改为 1 同时使用原有 5 秒 profiler，但会增加开销；状态采集持续整个 Decoder 运行期，**不与 profiler 的开始/停止窗口绑定**。这些测量开销也可能影响原 benchmark 的性能结果。

自动脚本负责原有的 proxy、Prefill、Decode、benchmark 及收尾；沿用原脚本的地址、设备和模式配置。运行前检查端口与设备没有被其他服务占用。脚本默认跳过初始全局进程清理，但运行过程中会按原流程管理其启动的服务。若 Decode 脚本文件名不同，设置环境变量 `DECODER_SCRIPT_DYNAM` 指向它。

如果你想自己手动启动 Decode，也可在 `scripts/` 下执行：

```bash
PROFILE_ENABLE=0 STATE_CAPTURE_ENABLE=1 \
  bash "$CAPTURE_HOME/run_decoder_with_profile.sh" \
  ./run_vllm_serve_8p8d_asyncSched_d_dynam.sh
```

随后沿用原方式启动其余服务并发送请求。包装脚本依赖 Decode 脚本通过 `vllm serve` 调用导出的 bash 函数；若脚本直接使用 `command vllm`、绝对路径、其他程序或重新加载环境，可能绕过包装，需要适配。

## 四、查看采集文件并做离线检查

包装脚本启动 Decode 时会把路径写进 `log/decoder_dynam.log`，形如：

```text
调度状态采集目录: /.../scripts/state_dataset/run-XXXXXXXX
```

每次启动使用一个新目录。查看最近的路径并分析，下面的路径需替换为日志中的实际目录：

```bash
rg '调度状态采集目录' log/decoder_dynam.log
ls -lh /日志显示的目录
python3 "$CAPTURE_HOME/analyze.py" /日志显示的目录
```

如服务器没有 `rg`，可用 `grep '调度状态采集目录' log/decoder_dynam.log`。每个 `rank-*-pid-*.jsonl` 文件包含一个进程/rank 的逐次调度记录。离线分析按时间划分样本，比较使用状态和实际动作对下一步本地 block 负载的预测效果；不足 20 个训练转移和 5 个测试转移时会报告样本不足。

如果目录没有 JSONL 文件，请检查 Decode 日志、实际加载的代码位置、DyntraLB 是否启用，以及启动脚本是否绕过 `vllm` 包装。采集代码只覆盖上述已核对的 scheduler 版本，尚未在你的真实服务器验证。

## 五、恢复

如果以后不再采集：

```bash
cd "$REPO"
patch --dry-run -R -p1 < "$CAPTURE_HOME/capture.patch"
```

确认检查通过后，再执行：

```bash
patch -R -p1 < "$CAPTURE_HOME/capture.patch"
```

备份文件不会被删除。若源码在实验期间发生其他变化，先检查差异，不要强行覆盖。

## 数据的边界

- `state_pre` 在调度前采集：请求数量、估计 block 负载等；等待候选快照可能陈旧，不等同于实时 ready 队列。
- `plan_pre` 是调度前已存在的计划，`action_observed` 是实际调度结果。没有记录请求 ID 和请求文本。
- `scheduler_cpu_wall_ms` 是 CPU 调度函数耗时，不是模型执行时间；`reward` 为空，不能直接当作 bandit 奖励。
- 现有策略决定动作，因此仅能做观察性预测，不能从没有执行过的动作推断收益，更不能据此证明全局最优。

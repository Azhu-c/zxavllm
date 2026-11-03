# Attention 空泡动机实验

本实验使用你已有的 `/workspace/dyntralb` 源码和 `/workspace/ycp/vllm_serve_run` 启动脚本。你只需要在后者下面新建一个文件夹：

```text
/workspace/ycp/vllm_serve_run/bubble- motivation
```

文件夹名中有空格，下面的命令已经加好引号。实验脚本、生成的启动脚本、请求数据和结果都放在这个文件夹里。

**准备是在做什么？**程序会读取上一级目录的原始启动脚本，在实验文件夹里生成三份本实验用的启动脚本。它不会修改你的 DyntraLB 源码或原始启动脚本。你无需手动输入 `export`。

## 第一步：放入文件并填写配置

在服务器上创建文件夹，把本目录的以下文件复制进去：

```text
README.md
bubble.sh
bubble_config.sh
prepare_reference.py
motivation1.py
local_pairs.py
scheduler_probe.py
trace_metrics.py
```

进入文件夹，编辑配置文件：

```bash
cd "/workspace/ycp/vllm_serve_run/bubble- motivation"
nano bubble_config.sh
```

只需要修改前两行的值，保留双引号：

```bash
MODEL_PATH="服务器上模型文件所在的目录"
HOST_IP="服务器的实际IP"
```

例如模型目录和 IP 的具体内容，请使用你服务器上的真实值。第三行 `SERVED_MODEL_NAME` 默认为 `qwen3_30B`；如果你原来的服务使用其他模型名，也把它改成原来的名字。使用 nano 时，按 `Ctrl+O`、回车保存，再按 `Ctrl+X` 退出。

使用你平时运行 vLLM 的 Python/Ascend 环境。模型路径用于加载模型，服务器 IP 用于服务之间通信；这些配置只需填一次。

## 第二步：准备一次

仍在这个文件夹内运行：

```bash
bash bubble.sh prepare
```

看到“准备完成”即可。此步骤需要上一级 `/workspace/ycp/vllm_serve_run` 有以下三个文件：

```text
run_vllm_serve_8p8d_asyncSched_p.sh
run_vllm_serve_8p8d_asyncSched_d_original.sh
load_balance_proxy_server_example.py
```

如果提示找不到文件或脚本格式不匹配，请把报错发给我。生成文件已经存在时，准备程序会停止，不会自动覆盖；通常不需要再运行准备命令。

## 第三步：打开四个终端

前三个服务启动后会一直运行，**不要等待它们退出**。前三个终端和第四个终端都要进入同一个文件夹。如果平时需要激活 Conda 环境，请在各终端先激活。

**终端 1：启动 Prefill。** 等服务启动后保留窗口。

```bash
cd "/workspace/ycp/vllm_serve_run/bubble- motivation"
bash bubble.sh prefill
```

**终端 2：启动 Decode。** 等服务启动后保留窗口。

```bash
cd "/workspace/ycp/vllm_serve_run/bubble- motivation"
bash bubble.sh decode
```

**终端 3：启动代理。** 启动后保留窗口。

```bash
cd "/workspace/ycp/vllm_serve_run/bubble- motivation"
bash bubble.sh proxy
```

**终端 4：运行实验。** 前三个服务都启动后执行：

```bash
cd "/workspace/ycp/vllm_serve_run/bubble- motivation"
bash bubble.sh run r1
```

它会检查服务；第一次运行时自动生成请求；然后依次运行失衡和均衡两组的计时实验与 profiler 实验，共四组。每组都要等 64 条请求完成，运行可能较久。终端提示符回来且显示“四组已完成”后，再查看结果。重复实验时用新编号，例如 `bash bubble.sh run r2`。

## 第四步：查看结果

在终端 4 运行：

```bash
bash bubble.sh report r1
```

屏幕显示两组的整批完成时间和输出吞吐。详细记录在 `local-results/`，profiler 数据在 `profile_attention_bubble/`。如果某组报错，程序会停止并保存失败记录；请把终端报错和对应 JSON 发给我。再次完整运行时，请使用新编号。

## 实验条件与注意事项

- 每个 Decode rank 有 4 个长请求、4 个短请求，每次选择其中 4 个运行。均衡组是人工指定的参考组合，不代表 DyntraLB 自动调度的结果；等待队列里的请求也不一定已完成 KV 预取。
- 当前配置默认 P 用 8 张 NPU、D 用 8 张 NPU，P/D/客户端使用同一台主机和同一个 IP。实际设备编号、网络和部署要与原始启动脚本一致。服务器被大实验占用时，请等空闲后再启动本实验。
- 准备程序会在 Prefill **副本**中关闭参考脚本的共享内存自动清理，并让服务使用配置文件中的 IP；其他设备、源码路径和软件设置沿用你的原始脚本。如果原始脚本与参考版本不同，准备程序可能报错，需要按实际脚本适配。
- 脚本在本机通过了模拟检查，尚未在真实 NPU 服务器验证。`timing` 的整批完成时间包含两阶段提交和 P/D 处理。`profile` 的设备事件需要确认 rank、step 和时钟对齐后才能分析；可使用 `trace_metrics.py` 汇总已核实并导出的事件。

旧的 32 请求实验说明在 [README-legacy.md](README-legacy.md)。

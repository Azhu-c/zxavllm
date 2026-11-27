# BalanceRoute 官方合成负载 → DyntraLB JSONL

本次交付采用用户选择的**作者官方合成负载**：seed 100，共 8000 条。数据文件为 `jsonl/balanceroute_official_seed100_8000_qwen3_chat.jsonl`。

## 数据来源与保留内容

原表中的「8000 requests，平均 prompt 3197，平均 output 1185」来自 [BalanceRoute 论文 §6.1](https://arxiv.org/html/2605.06113v3#S6.SS1) 的 Proprietary Data。截至 2026-11-27 核查，[作者公开仓库](https://github.com/b010001y/BalanceRoute) 的初始发布提供合成负载、核心算法和 CPU 模拟器；本次未找到这份私有生产 trace 的公开下载。

本文件使用作者仓库内已经生成并冻结的 [100.npz](https://github.com/b010001y/BalanceRoute/blob/fdd2a6971ca6222494b1335cc67aaae8e4e28b08/src/balanceroute/reference/r16/inputs/100.npz)。保留其全部 8000 对输入/输出长度、对应关系及原始顺序，没有重新采样、调整均值或截断。作者的合成分布说明见 [复现协议](https://github.com/b010001y/BalanceRoute/blob/fdd2a6971ca6222494b1335cc67aaae8e4e28b08/docs/REPRODUCIBILITY.md)。

| 项目 | 数值 / 口径 |
| --- | --- |
| 样本数 | 8000 |
| 作者设定的平均输入 / 输出 | 3000 / 1000 tokens |
| seed 100 实际平均完整输入 | 2999.999875 tokens |
| seed 100 实际平均输出 | 1000.003625 tokens |
| 输入长度范围 | 128–65536 tokens |
| 输出长度范围 | 200–8192 tokens |
| 单条最大输入 + 输出 | 68321 tokens |
| 原始生产文本 | 作者未提供；JSONL 中的文本为本次新构造的合成文本 |
| 到达时间 | 作者未提供；由 benchmark 的 request rate 控制 |

这是合成负载，不能标注为论文的 Proprietary Data，也不能据此声称复现了该私有 trace 上的实验。

## JSONL 字段与输入长度

每行只有两个字段，符合 vLLM `custom` 数据集格式：

```json
{"prompt":"合成的用户消息文本","output_tokens":380}
```

实际文件中的 prompt 是完整合成文本。作者发布的是整数长度数组，本次用确定性的英文词序列补入文本，并在开头放入不同标识。全部 8000 条完整 chat 输入的首个 16-token 块均不同。

适配的 tokenizer 为 `Qwen/Qwen3-30B-A3B-Instruct-2507`，固定修订版 `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`，与项目现有 benchmark 脚本的模型目录名称一致。本次只下载了 tokenizer，没有下载模型权重。

**作者输入长度 `s[i]` 对应服务端完整的单条 user chat 输入。** 该 tokenizer 的单条 user 消息加 assistant generation prefix 占 8 tokens，因此 JSONL 里的纯文本长度为 `s[i] - 8`。全部样本均已验证：

```text
token_count(apply_chat_template([{"role": "user", "content": prompt}],
                               add_generation_prompt=True)) == s[i]
output_tokens == o[i]
```

纯文本平均为 2991.999875 tokens。下面的 chat benchmark 命令启用 `--skip-chat-template`，由服务端添加一次模板。该设置下部分 vLLM 版本的客户端输入长度统计只计纯文本；应以服务端 `usage.prompt_tokens` 核对完整输入长度。更换模型 tokenizer、自定义服务端模板、追加 system 消息或工具定义后，需要重新校验长度。

## 在现有 DyntraLB 服务上运行

将 JSONL 传到服务器后，用已有的 Prefill、Decode 和 proxy 服务运行下面的命令。将模型目录、服务名、地址和数据路径替换成当前部署值：

```bash
vllm bench serve \
  --backend openai-chat \
  --host 127.0.0.1 --port 8080 \
  --endpoint /v1/chat/completions \
  --model qwen3_30B \
  --tokenizer /path/to/Qwen3-30B-A3B-Instruct-2507 \
  --dataset-name custom \
  --dataset-path /path/to/balanceroute_official_seed100_8000_qwen3_chat.jsonl \
  --num-prompts 8000 \
  --custom-output-len -1 \
  --skip-chat-template \
  --disable-shuffle \
  --ignore-eos \
  --temperature 0 \
  --max-concurrency 128 \
  --request-rate 400 \
  --save-result --save-detailed
```

- `--custom-output-len -1`：读取每行的 `output_tokens`，避免统一长度覆盖作者分布。
- `--skip-chat-template`：客户端发送原始 user 文本，服务端只加一次 chat template。项目现有 `run_test_proxy_benchmark_profile.sh` 的命令没有这个标志；直接复用该脚本时应在其 `BENCHMARK_CMD` 中加入。
- `--ignore-eos`：尽可能按每条指定输出长度回放；若服务器提前停止或失败，应检查实际输出长度。项目现有命令也需要补入此标志。
- `--disable-shuffle`：保留官方 NPZ 中的原始顺序。
- 并发 128、请求速率 400 沿用项目现有脚本的默认值；它们是回放设置，作者没有随 NPZ 发布生产到达过程。

完整负载需要服务允许至少 68321 tokens 的输入加输出，并允许单条最多输出 8192 tokens。本文件保留了官方长请求；不要在某组对照实验中单独截断或跳过。建议先用 `--num-prompts 16` 检查服务，再用完整 8000 条运行。

## 来源校验与再生成

- 官方提交：`fdd2a6971ca6222494b1335cc67aaae8e4e28b08`。
- 官方 NPZ SHA-256：`383a8e89fde24998590518db4d44a872f8e36a7e9b5e617f62c49ef39629e484`。
- 官方有序长度数组 SHA-256：`e5eca3ccd0b7abbd03b007c3864d38bff04c241f950779c97e2aded0694b955e`，与作者 `expected.json` 的 seed 100 记录一致。
- 机器可读统计和 JSONL 校验值：`metadata/balanceroute_official_seed100.json`。
- 原始 NPZ、官方配置和许可文件：`raw/balanceroute_official/`。
- 本次转换脚本：`tools/convert_balanceroute_official.py`。在项目根目录运行：

```bash
dataset_bundle/.venv/bin/python dataset_bundle/tools/convert_balanceroute_official.py
```

在其他机器上，可先安装 `numpy`、`tokenizers==0.22.2` 和 `jinja2`，再用该机器的 Python 运行同一个脚本。压缩交付包按 `dataset_bundle/` 相对目录包含 JSONL、元数据、转换脚本、官方 NPZ、许可文件及固定版本 tokenizer；不包含原项目的其他大型数据集或本地虚拟环境。

转换脚本使用 NumPy、tokenizers 和 Jinja2，逐条校验完整 chat 输入 token 数、输出长度、顺序、字段和 UTF-8 编码。它只做离线转换；本次未运行服务器或 NPU 性能实验。

官方长度数据及随附文件来自 BalanceRoute，保留其 Apache-2.0 `LICENSE` 和 `NOTICE`。本次新增了提示文本、JSONL 转换与上述说明。

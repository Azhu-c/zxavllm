# BalanceRoute 官方合成负载（DyntraLB）

8000 条作者官方 seed 100 合成负载，已转换为 DyntraLB / vLLM `custom` 数据集的 JSONL 格式。保留全部输入、输出长度配对及原始顺序，逐条通过 Qwen3 tokenizer 校验；提示文本由转换脚本新构造。

- [下载完整压缩包（约 28 MB）](balanceroute_official_seed100_8000.tar.gz)
- [完整运行说明](dataset_bundle/balanceroute_official_README.md)
- [转换脚本](dataset_bundle/tools/convert_balanceroute_official.py)
- [统计与校验值](dataset_bundle/metadata/balanceroute_official_seed100.json)

数据来源为 [BalanceRoute 官方冻结长度文件](https://github.com/b010001y/BalanceRoute/blob/fdd2a6971ca6222494b1335cc67aaae8e4e28b08/src/balanceroute/reference/r16/inputs/100.npz)。这是作者公开的合成数据。原表中的平均 prompt 3197 / output 1185 对应论文的私有生产 trace；本包的平均长度约为 3000 / 1000。

| 项目 | 值 |
| --- | --- |
| 请求数 | 8000 |
| seed | 100 |
| 平均完整 chat 输入长度 | 2999.999875 tokens |
| 平均输出长度 | 1000.003625 tokens |
| 输入长度范围 | 128–65536 tokens |
| 输出长度范围 | 200–8192 tokens |
| 最大输入加输出 | 68321 tokens |
| 校验 tokenizer | Qwen/Qwen3-30B-A3B-Instruct-2507 |

## 解压与运行

在本目录解压，得到完整 JSONL、原始 NPZ、固定版本 tokenizer、元数据和脚本：

```bash
tar -xzf balanceroute_official_seed100_8000.tar.gz
```

JSONL 路径为：

```text
dataset_bundle/jsonl/balanceroute_official_seed100_8000_qwen3_chat.jsonl
```

每行包含 `prompt` 字符串和 `output_tokens` 整数。JSONL 内为原始 user 文本；服务端添加一次 Qwen3 chat template 后，完整输入长度等于官方长度数组。

使用已经运行的 DyntraLB proxy，替换地址、服务名和模型目录：

```bash
vllm bench serve \
  --backend openai-chat \
  --host 127.0.0.1 --port 8080 \
  --endpoint /v1/chat/completions \
  --model qwen3_30B \
  --tokenizer /path/to/Qwen3-30B-A3B-Instruct-2507 \
  --dataset-name custom \
  --dataset-path ./dataset_bundle/jsonl/balanceroute_official_seed100_8000_qwen3_chat.jsonl \
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

`--custom-output-len -1` 使用逐条输出长度，`--skip-chat-template` 避免重复套用模板，`--ignore-eos` 避免 EOS 提前终止改变输出分布，`--disable-shuffle` 保留顺序。完整回放要求服务支持至少 68321 tokens 的上下文，并允许单条输出 8192 tokens。并发和请求速率为项目回放设置；官方合成文件未提供生产到达时间。

本次验证包括全部 8000 条的字段、输入 token 数、输出长度、顺序和官方数组校验值，以及上游 `CustomDataset.sample` 方法的样本兼容检查；未运行 NPU 服务性能实验。

## 校验与许可

压缩包 SHA-256：

```text
a1c380657b8309455615ab35a46df66f56773cc19ae70eb1e57bd5ae8973e43c
```

JSONL SHA-256：

```text
e5000283b735517bceeef096630c01ccc35b8cd40a0b148a7db6bc4460df365c
```

约 126 MB 的原始 JSONL 保存在压缩包内。包内保留 BalanceRoute 的 Apache-2.0 `LICENSE` 和 `NOTICE`；本次新增提示文本、转换脚本和运行说明。

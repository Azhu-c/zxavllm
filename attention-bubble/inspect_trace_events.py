#!/usr/bin/env python3
"""检查现有 profiler 时间线中是否包含 attention 和通信事件。只读，不依赖第三方库。"""
import json
import re
from collections import Counter
from pathlib import Path

RUN = "r1"
CHUNK = 1024 * 1024
MAX_PENDING = 32 * CHUNK
COMM = re.compile(r"hccl|hcom|all.?to.?all|all.?gather|all.?reduce|reduce.?scatter|broadcast", re.I)
ATTN = re.compile(r"attention|flash.?attn|paged.?attn", re.I)


def events(path):
    """流式读取常见 Chrome trace 格式，避免一次载入整个大文件。"""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8-sig") as f:
        buf = ""
        while True:
            part = f.read(CHUNK)
            buf += part
            if not buf.strip() and part:
                continue
            if buf.lstrip().startswith("["):
                pos = buf.index("[") + 1
                break
            match = re.search(r'"traceEvents"\s*:\s*\[', buf)
            if match:
                pos = match.end()
                break
            if not part or len(buf) > MAX_PENDING:
                raise ValueError("无法识别时间线事件数组")
        buf = buf[pos:]
        eof = False
        while True:
            buf = buf.lstrip()
            if buf.startswith(","):
                buf = buf[1:].lstrip()
            if buf.startswith("]"):
                return
            if not buf and eof:
                raise ValueError("时间线事件数组未正常结束")
            try:
                value, end = decoder.raw_decode(buf)
            except json.JSONDecodeError as exc:
                if eof:
                    raise ValueError(f"无法解析时间线事件: {exc}") from exc
                if len(buf) > MAX_PENDING:
                    raise ValueError("单条事件过大，停止读取")
                part = f.read(CHUNK)
                eof = not part
                buf += part
                continue
            buf = buf[end:]
            if isinstance(value, dict):
                yield value


def inspect(condition, rank, folder):
    path = folder / "ASCEND_PROFILER_OUTPUT" / "trace_view.json"
    print(f"\n===== {condition} dp{rank} =====", flush=True)
    if not path.is_file():
        print("没有找到 trace_view.json")
        return
    counts = {"通信候选": Counter(), "Attention 候选": Counter()}
    examples = {}
    total = 0
    try:
        for event in events(path):
            total += 1
            name = str(event.get("name", ""))
            category = str(event.get("cat", ""))
            for label, pattern in (("通信候选", COMM), ("Attention 候选", ATTN)):
                if pattern.search(name + " " + category):
                    key = (name[:160], category[:80], str(event.get("ph", "")))
                    counts[label][key] += 1
                    if label not in examples:
                        examples[label] = {key: event.get(key) for key in ("name", "cat", "ph", "pid", "tid", "ts", "dur") if key in event}
    except (OSError, ValueError) as exc:
        print(f"读取失败: {exc}")
        return
    print(f"事件总数: {total}")
    for label, counter in counts.items():
        print(f"{label}: {sum(counter.values())} 条；前 8 种名称/类别/类型:")
        for (name, category, phase), count in counter.most_common(8):
            print(f"  {count} 条 | {name} | {category} | {phase}")
        if label in examples:
            print("  示例:", json.dumps(examples[label], ensure_ascii=False))
    print("提示：名称匹配是线索；没有匹配不代表没有通信。")


def main():
    for condition in ("skewed", "balanced"):
        record = Path("local-results") / f"{condition}_profile_{RUN}.json"
        try:
            data = json.loads(record.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"无法读取 {record}: {exc}")
            continue
        for rank in (0, 4):
            dirs = [Path(p) for p in data.get("new_profile_dirs", []) if Path(p).name.startswith(f"dp{rank}_")]
            if len(dirs) != 1:
                print(f"{condition} dp{rank}: 找到 {len(dirs)} 个目录，无法选择")
                continue
            inspect(condition, rank, dirs[0])
    print("\n这些结果不能直接计算跨 rank 等待，需进一步核对事件对应关系和时间基准。")


if __name__ == "__main__":
    main()

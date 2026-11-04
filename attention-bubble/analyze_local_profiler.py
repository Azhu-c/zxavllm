#!/usr/bin/env python3
"""在服务器上直接对比两组 Ascend profiler 数据。"""
import csv, json, math, re, statistics
from collections import Counter
from pathlib import Path

RUN = "r1"
conditions = ("skewed", "balanced")

def norm(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())

def get(row, *names):
    fields = {norm(k): v for k, v in row.items() if k is not None}
    return next((fields[norm(n)] for n in names if norm(n) in fields), None)

def num(value):
    try:
        value = float(str(value).replace(",", ""))
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None

def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f)

records, profiles = {}, {}
for condition in conditions:
    record = Path("local-results") / f"{condition}_profile_{RUN}.json"
    data = json.loads(record.read_text(encoding="utf-8"))
    if not all(data.get(k) for k in ("run_success", "valid_steady_window", "profile_started", "profile_stopped")):
        raise RuntimeError(f"{record} 的采样状态无效")
    records[condition] = data
    ranks = {}
    for folder in data.get("new_profile_dirs", []):
        path = Path(folder)
        match = re.match(r"dp(\d+)_", path.name)
        if not match or int(match.group(1)) in ranks or not path.is_dir():
            raise RuntimeError(f"rank 重复或目录不存在: {path}")
        ranks[int(match.group(1))] = path
    if set(ranks) != set(range(8)):
        raise RuntimeError(f"{condition} 的 rank 不完整: {sorted(ranks)}")
    profiles[condition] = ranks

for key in ("manifest_sha256", "model"):
    if not records["skewed"].get(key) or records["skewed"].get(key) != records["balanced"].get(key):
        raise RuntimeError(f"两组 {key} 不一致，不能比较")

from torch_npu.profiler.profiler import analyse
ops, steps, unnumbered = {}, {}, {}
for condition in conditions:
    for rank, path in sorted(profiles[condition].items()):
        out = path / "ASCEND_PROFILER_OUTPUT"
        stat = out / "op_statistic.csv"
        if not stat.is_file():
            print(f"解析 {condition} dp{rank} ...", flush=True)
            analyse(str(path))
        if not stat.is_file():
            raise RuntimeError(f"解析后没有找到 {stat}")
        found = {}
        for row in read_csv(stat):
            name = get(row, "OP Type", "Op Name", "Name", "Type") or ""
            core = get(row, "Core Type", "Accelerator Core") or "unknown"
            if not re.search(r"attention|flash.?attn|paged.?attn", name, re.I):
                continue
            count = num(get(row, "Count", "Calls"))
            total = num(get(row, "Total Time(us)", "Total Duration(us)"))
            avg = num(get(row, "Avg Time(us)", "Avg Duration(us)"))
            if count is None or count <= 0:
                continue
            if total is None and avg is not None:
                total = count * avg
            if total is None or total < 0:
                continue
            key = (name.strip(), core.strip())
            old_count, old_total = found.get(key, (0, 0.0))
            found[key] = (old_count + count, old_total + total)
        ops[condition, rank] = found
        values = {"Computing": [], "Communication(Not Overlapped)": [], "Free": []}
        blank_rows = []
        step_file = out / "step_trace_time.csv"
        if step_file.is_file():
            for row in read_csv(step_file):
                step = str(get(row, "Step", "Step ID", "Step Id") or "").strip()
                if step.lower() in ("", "n/a", "none", "null"):
                    names = ("Computing", "Communication", "Overlapped", "Communication(Not Overlapped)", "Free")
                    totals = {name: num(get(row, name, name + "(us)")) for name in names}
                    totals = {name: value / 1000 for name, value in totals.items() if value is not None and value >= 0}
                    if totals:
                        blank_rows.append((str(get(row, "Device_id", "Device ID") or "?"), totals))
                    continue
                if not re.fullmatch(r"(?:ProfilerStep#)?\d+", step):
                    continue
                for name in values:
                    value = num(get(row, name, name + "(us)"))
                    if value is not None and value >= 0:
                        values[name].append(value)
        steps[condition, rank] = {k: statistics.median(v) / 1000 for k, v in values.items() if v}
        unnumbered[condition, rank] = blank_rows

common = set.intersection(*(set(v) for v in ops.values()))
common = sorted(common, key=lambda k: sum(v[k][0] for v in ops.values()), reverse=True)
print("\n===== PROFILER 对比结果 =====")
if not common:
    print("没有找到两组所有 rank 共有的 Attention 候选算子。候选名称：")
    for condition in conditions:
        for rank in (0, 4):
            print(condition, f"dp{rank}", sorted(ops[condition, rank]))
for key in common[:3]:
    print(f"\nAttention 候选算子: {key[0]} | core: {key[1]}")
    print("rank  skewed_平均us(次数)  balanced_平均us(次数)")
    for rank in range(8):
        a_n, a_t = ops["skewed", rank][key]
        b_n, b_t = ops["balanced", rank][key]
        print(f"dp{rank}  {a_t/a_n:.3f}({int(a_n)})  {b_t/b_n:.3f}({int(b_n)})")
print("\n逐 rank step 统计中位数（毫秒，仅使用明确编号的 step）")
for condition in conditions:
    for rank in range(8):
        values = steps[condition, rank]
        text = ", ".join(f"{k}={v:.3f}" for k, v in values.items()) or "无明确编号的 step 统计"
        print(f"{condition} dp{rank}: {text}")
print("\nStep 无编号的原始统计行（毫秒；统计范围待确认，不能当作单步时间）")
for condition in conditions:
    for rank in range(8):
        blank_rows = unnumbered[condition, rank]
        if not blank_rows:
            print(f"{condition} dp{rank}: 无可显示的未编号统计行")
        for index, (device, values) in enumerate(blank_rows, 1):
            text = ", ".join(f"{k}={v:.3f}" for k, v in values.items())
            print(f"{condition} dp{rank} device={device} 行{index}/{len(blank_rows)}: {text}")

print("\n通信候选检查（仅查看 dp0 和 dp4 的 kernel_details.csv）")
communication_pattern = re.compile(r"hccl|hcom|all.?to.?all|all.?gather|all.?reduce|reduce.?scatter", re.I)
for condition in conditions:
    for rank in (0, 4):
        path = profiles[condition][rank] / "ASCEND_PROFILER_OUTPUT" / "kernel_details.csv"
        found = Counter()
        if path.is_file():
            for row in read_csv(path):
                name = (row.get("Name") or "").strip()
                kind = (row.get("Type") or "").strip()
                core = (row.get("Accelerator Core") or "").strip()
                if communication_pattern.search(" ".join((name, kind, core))):
                    found[(name, kind)] += 1
        if found:
            print(f"{condition} dp{rank}: {sum(found.values())} 条候选记录，前 5 类：")
            for (name, kind), count in found.most_common(5):
                print(f"  {name} | {kind} | {count} 条")
        else:
            print(f"{condition} dp{rank}: 未找到通信候选记录（不代表没有通信）")
print("\n注意：Free 是设备空闲，不等于跨 rank 等待；需核对同一步同一层的设备时间线。")

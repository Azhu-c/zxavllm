#!/usr/bin/env python3
"""Read Ascend profiler CSVs and create an offline HTML report; Python 3.8+."""
import argparse
import csv
import heapq
import html
import math
import re
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

TOP_N = 15
N_BINS = 80
HIST_EDGES = [50, 100, 200, 500, 1000, 5000, 10000, 50000, 100000]


def esc(value):
    return html.escape(str(value), quote=True)


def rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f)


def num(value):
    try:
        x = float(str(value).strip())
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def decimal(value):
    try:
        x = Decimal(str(value).strip())
        return x if x.is_finite() else None
    except (InvalidOperation, ValueError):
        return None


def fnum(value, digits=3):
    return "—" if value is None else f"{value:,.{digits}f}"


def table(headers, data):
    h = "".join("<th>" + esc(s) + "</th>" for s in headers)
    body = "".join("<tr>" + "".join("<td>" + esc(v) + "</td>" for v in row) + "</tr>" for row in data)
    return '<div class="tablewrap"><table><thead><tr>' + h + "</tr></thead><tbody>" + body + "</tbody></table></div>"


def stat_type(name):
    low = name.lower()
    if "attention" in low or "flashattn" in low:
        return "Attention候选"
    if "moedistributedispatch" in low:
        return "MoE Dispatch"
    if "moedistributecombine" in low:
        return "MoE Combine"
    if "groupedmatmul" in low:
        return "GroupedMatmul"
    return "其他"


def capture_group(root, stats_file):
    """Infer only clear group layouts; otherwise keep directories separate."""
    for parent in stats_file.parents:
        if parent.name.endswith("_ascend_pt"):
            return parent.parent
        if parent == root:
            break
    parent = stats_file.parent
    if parent.name == "ASCEND_PROFILER_OUTPUT":
        parent = parent.parent
    if re.fullmatch(r"(?i)(?:rank|device|npu)[_-]?\d+", parent.name):
        return parent.parent
    return parent


def info_note(message):
    return '<div class="note">' + esc(message) + "</div>"


def bars(items, unit="ms"):
    if not items:
        return '<p class="muted">没有可显示的记录。</p>'
    maximum = max(value for _, value in items)
    content = []
    for label, value in items:
        width = max(0.0, min(100.0, value * 100 / maximum)) if maximum > 0 else 0
        content.append(
            '<div class="barrow"><div class="barlabel">' + esc(label) +
            '</div><div class="bartrack"><div class="bar" style="width:' + f"{width:.4f}" +
            '%"></div></div><div class="barvalue">' + esc(fnum(value, 2)) + " " + esc(unit) + "</div></div>"
        )
    return "".join(content)


def bin_chart(bins, span_s, cat):
    values = [b[cat]["sum"] / b[cat]["count"] if b[cat]["count"] else 0 for b in bins]
    vmax = max(values, default=0)
    if not vmax:
        return '<p class="muted">明细中没有匹配的事件。</p>'
    rects = []
    for i, (value, b) in enumerate(zip(values, bins)):
        count = b[cat]["count"]
        title = (f"相对区间 {i * span_s / N_BINS:.6f}–{(i+1)*span_s/N_BINS:.6f} s；"
                 f"事件数 {count}；平均 duration {value:.3f} μs；最大 {b[cat]['max']:.3f} μs")
        height = max(1.0, 140 * value / vmax) if count else 0
        rects.append(f'<rect x="{i*10+1}" y="{155-height:.2f}" width="8" height="{height:.2f}" fill="#4674d5"><title>{esc(title)}</title></rect>')
    return ('<div class="plotwrap"><svg viewBox="0 0 810 190" role="img" aria-label="'
            + esc(cat) + ' 分时间段的平均任务耗时">'
            + '<line x1="0" y1="155" x2="800" y2="155" stroke="#8a98ac"/>'
            + "".join(rects)
            + f'<text x="0" y="177" font-size="12">0 s</text><text x="645" y="177" font-size="12">相对起点 {span_s:.3f} s</text>'
            + f'<text x="0" y="12" font-size="12">平均耗时，最高 {vmax:.1f} μs</text></svg></div>')


def read_stats(path):
    groups = defaultdict(list)
    for row in rows(path):
        name = (row.get("OP Type") or "").strip()
        total = num(row.get("Total Time(us)"))
        count = num(row.get("Count"))
        if not name or total is None or count is None:
            continue
        groups[(row.get("Device_id") or "unknown").strip()].append({
            "name": name, "core": (row.get("Core Type") or "").strip(), "count": int(count),
            "total": total, "avg": num(row.get("Avg Time(us)")),
            "min": num(row.get("Min Time(us)")), "max": num(row.get("Max Time(us)")),
            "source_ratio": num(row.get("Ratio(%)"))})
    return groups


def read_kernel(path, device):
    total = 0
    invalid = 0
    types = Counter()
    valid_types = Counter()
    duration_by_type = defaultdict(float)
    max_by_type = {}
    streams = Counter()
    hist = {cat: [0] * (len(HIST_EDGES) + 1) for cat in ("Attention候选", "MoE Dispatch")}
    first = None
    last = None
    top = []
    for row in rows(path):
        rid = (row.get("Device_id") or "").strip()
        if rid and device != "unknown" and rid != device:
            continue
        total += 1
        typ = (row.get("Type") or row.get("Name") or "unknown").strip()
        types[typ] += 1
        streams[(row.get("Stream ID") or "unknown").strip()] += 1
        dur = num(row.get("Duration(us)"))
        ts = decimal(row.get("Start Time(us)"))
        if dur is None or dur < 0:
            invalid += 1
            continue
        duration_by_type[typ] += dur
        valid_types[typ] += 1
        max_by_type[typ] = max(dur, max_by_type.get(typ, dur))
        cat = stat_type(typ)
        if cat in hist:
            index = next((i for i, edge in enumerate(HIST_EDGES) if dur < edge), len(HIST_EDGES))
            hist[cat][index] += 1
        if ts is not None:
            end = ts + Decimal(str(dur))
            first = ts if first is None or ts < first else first
            last = end if last is None or end > last else last
        item = (dur, total, typ, (row.get("Stream ID") or "").strip(),
                (row.get("Start Time(us)") or "").strip(),
                (row.get("Wait Time(us)") or "").strip())
        if len(top) < TOP_N:
            heapq.heappush(top, item)
        elif dur > top[0][0]:
            heapq.heapreplace(top, item)
    bins = [{cat: {"sum": 0.0, "count": 0, "max": 0.0}
             for cat in ("Attention候选", "MoE Dispatch")} for _ in range(N_BINS)]
    span = (last - first) if first is not None and last is not None else Decimal(0)
    if span > 0:
        for row in rows(path):
            rid = (row.get("Device_id") or "").strip()
            if rid and device != "unknown" and rid != device:
                continue
            typ = (row.get("Type") or row.get("Name") or "").strip()
            cat = stat_type(typ)
            if cat not in ("Attention候选", "MoE Dispatch"):
                continue
            ts = decimal(row.get("Start Time(us)"))
            dur = num(row.get("Duration(us)"))
            if ts is None or dur is None or dur < 0:
                continue
            i = min(N_BINS - 1, max(0, int(((ts - first) * N_BINS) / span)))
            b = bins[i][cat]
            b["sum"] += dur
            b["count"] += 1
            b["max"] = max(b["max"], dur)
    return {"total": total, "invalid": invalid, "types": types, "valid_types": valid_types,
            "duration_by_type": duration_by_type, "max_by_type": max_by_type,
            "streams": streams, "first": first, "last": last,
            "span_s": float(span / Decimal(1000000)), "bins": bins,
            "top": sorted(top, reverse=True), "hist": hist}


def read_operator(path):
    names = Counter()
    valid_counts = Counter()
    self_time = defaultdict(float)
    device_valid_counts = Counter()
    device_self_time = defaultdict(float)
    n = 0
    for row in rows(path):
        n += 1
        name = (row.get("Name") or "unknown").strip()
        names[name] += 1
        v = num(row.get("Host Self Duration(us)"))
        if v is not None and v >= 0:
            self_time[name] += v
            valid_counts[name] += 1
        device_v = num(row.get("Device Self Duration(us)"))
        if device_v is not None and device_v >= 0:
            device_self_time[name] += device_v
            device_valid_counts[name] += 1
    return {"rows": n, "names": names, "self_time": self_time, "valid_counts": valid_counts,
            "device_self_time": device_self_time, "device_valid_counts": device_valid_counts}


def comparison_matrix(groups, group_dir, kind, records, root):
    """Compare each source separately; never add durations across devices."""
    mappings = []
    all_keys = set()
    for device, path, stats, kernel, host in groups:
        values = {}
        if kind == "op_statistic":
            for item in stats:
                key = (item["name"], item["core"])
                v = values.setdefault(key, {"count": 0, "valid": 0, "total": 0.0, "max": None})
                v["count"] += item["count"]
                v["valid"] += item["count"]
                v["total"] += item["total"]
                if item["max"] is not None:
                    v["max"] = max(v["max"], item["max"]) if v["max"] is not None else item["max"]
            source = path
        elif kind == "kernel_details" and kernel is not None:
            for typ, count in kernel["types"].items():
                key = (typ, "")
                values[key] = {"count": count, "valid": kernel["valid_types"][typ],
                               "total": kernel["duration_by_type"].get(typ, 0.0),
                               "max": kernel["max_by_type"].get(typ)}
            source = path.parent / "kernel_details.csv"
        elif kind in ("host_self", "device_self") and host is not None:
            counts_key = "valid_counts" if kind == "host_self" else "device_valid_counts"
            time_key = "self_time" if kind == "host_self" else "device_self_time"
            for name, count in host["names"].items():
                key = (name, "")
                values[key] = {"count": count, "valid": host[counts_key][name],
                               "total": host[time_key].get(name, 0.0), "max": None}
            source = path.parent / "operator_details.csv"
        else:
            source = path.parent / ("kernel_details.csv" if kind == "kernel_details" else "operator_details.csv")
        all_keys.update(values)
        for (name, core), v in values.items():
            avg = v["total"] / v["valid"] if v["valid"] else None
            records.append([group_dir, kind, str(source), device, name, core, v["count"],
                            v["valid"], v["total"] if v["valid"] else "", avg, v["max"]])
        available = kind == "op_statistic" or (kind == "kernel_details" and kernel is not None) or (
            kind in ("host_self", "device_self") and host is not None)
        mappings.append((device, path, source, values, available))
    if not all_keys:
        return '<p class="muted">没有可对比的文件或字段。</p>'
    # Rank by the largest per-device total, without adding time across devices.
    keys = sorted(all_keys, key=lambda key: (-max(
        (v.get(key, {}).get("total", 0.0) for _, _, _, v, _ in mappings), default=0), key))
    headers = ["名称 / Core"] + [
        "Device " + device + " · " + str(path.parent.relative_to(Path(root)))
        for device, path, _, _, _ in mappings
    ]
    displayed = []
    for key in keys[:30]:
        row = [key[0] + (" / " + key[1] if key[1] else "")]
        for _, _, source, values, available in mappings:
            v = values.get(key)
            if v is None:
                row.append("— 无文件" if not source.is_file() else ("— 无法归属" if not available else "— 无该项"))
            elif not v["valid"]:
                row.append(f"次数 {v['count']}；无有效耗时")
            else:
                avg = v["total"] / v["valid"]
                row.append(f"平均 {fnum(avg, 3)} μs / 次数 {v['count']} / 累计 {fnum(v['total']/1000, 3)} ms")
        displayed.append(row)
    return table(headers, displayed)


def aligned_attention(groups):
    """Bucket attention starts on a common observed time axis; no step matching."""
    if len({device for device, _, _, _, _ in groups}) != len(groups):
        return None
    available = [(device, path, k) for device, path, _, k, _ in groups
                 if k and k["first"] is not None and k["last"] is not None]
    if len(available) < 2:
        return None
    start = max(k["first"] for _, _, k in available)
    end = min(k["last"] for _, _, k in available)
    span = end - start
    if span <= 0:
        return None
    summaries = []
    for device, path, _ in available:
        bins = [{"count": 0, "sum": 0.0, "max": 0.0} for _ in range(N_BINS)]
        for row in rows(path.parent / "kernel_details.csv"):
            rid = (row.get("Device_id") or "").strip()
            if rid and device != "unknown" and rid != device:
                continue
            if stat_type((row.get("Type") or row.get("Name") or "").strip()) != "Attention候选":
                continue
            ts, dur = decimal(row.get("Start Time(us)")), num(row.get("Duration(us)"))
            if ts is None or dur is None or dur < 0 or not start <= ts < end:
                continue
            index = min(N_BINS - 1, int((ts - start) * N_BINS / span))
            bucket = bins[index]
            bucket["count"] += 1
            bucket["sum"] += dur
            bucket["max"] = max(bucket["max"], dur)
        summaries.append((device, path, bins))
    return start, span, summaries


def attention_heatmap(span, summaries):
    avgs = [b["sum"] / b["count"] for _, _, bins in summaries for b in bins if b["count"]]
    vmax = max(avgs, default=0)
    if not vmax:
        return '<p class="muted">共同时间范围内未发现 Attention 候选任务。</p>'
    width = 800
    height = 40 + 28 * len(summaries)
    rects = []
    for j, (device, path, bins) in enumerate(summaries):
        y = 14 + j * 28
        rects.append(f'<text x="0" y="{y+15}" font-size="12">Device {esc(device)}</text>')
        for i, b in enumerate(bins):
            if not b["count"]:
                fill = "#f0f2f5"
                avg = None
            else:
                avg = b["sum"] / b["count"]
                strength = min(1.0, max(0.10, avg / vmax))
                fill = f"rgba(40,98,195,{strength:.3f})"
            t0 = float(span * i / N_BINS / Decimal(1000000))
            t1 = float(span * (i + 1) / N_BINS / Decimal(1000000))
            title = (f"Device {device}；共同起点后 {t0:.6f}–{t1:.6f} s；"
                     f"次数 {b['count']}；平均 {fnum(avg, 3)} μs；来源 {path.parent}")
            rects.append(f'<rect x="{100+i*8.5:.1f}" y="{y}" width="7.5" height="20" fill="{fill}"><title>{esc(title)}</title></rect>')
    span_s = float(span / Decimal(1000000))
    rects.append(f'<text x="100" y="{height-3}" font-size="12">0 s</text>')
    rects.append(f'<text x="640" y="{height-3}" font-size="12">{span_s:.3f} s</text>')
    return '<div class="plotwrap"><svg viewBox="0 0 '+str(width)+" "+str(height)+'" role="img" aria-label="各设备 Attention 时间分段">'+"".join(rects)+"</svg></div>"


def group_html(root, path, device, stats, sample, record, allow_host=True):
    title = f"{root} / {path.parent} / Device {device}"
    result = ['<section class="card"><h2>' + esc(title) + "</h2>"]
    stats = sorted(stats, key=lambda x: x["total"], reverse=True)
    total = sum(x["total"] for x in stats)
    result.append('<p>统计表 ' + str(len(stats)) + ' 类；记录累计耗时之和 <strong>'
                  + fnum(total / 1000000, 6) + ' s</strong>。百分比的分母是该统计表各类累计耗时之和，不是端到端时长。</p>')
    result.append(info_note("不同 Stream 的任务可能重叠。各任务累计耗时可超过采集窗口，不能将此处的百分比当作设备利用率或空泡率。"))
    result.append('<h3>算子累计耗时 Top 15</h3>')
    result.append(bars([(x["name"], x["total"] / 1000) for x in stats[:TOP_N]]))
    data = []
    for x in stats:
        pct = x["total"] / total * 100 if total else 0
        ratio = x["max"] / x["avg"] if x["max"] is not None and x["avg"] and x["avg"] > 0 else None
        data.append([x["name"], x["core"], str(x["count"]), fnum(x["total"] / 1000, 3),
                     fnum(x["avg"], 3), fnum(x["min"], 3), fnum(x["max"], 3), fnum(pct, 3),
                     fnum(ratio, 1)])
        record.append([str(root), str(path), device, x["name"], x["core"], x["count"],
                       x["total"], x["avg"], x["min"], x["max"], pct])
    result.append('<h3>完整算子统计</h3>')
    result.append(table(["OP Type", "Core", "次数", "累计 ms", "平均 μs", "最小 μs",
                         "最大 μs", "占表内累计 %", "最大/平均"], data))
    result.append(info_note("最大值明显大于平均值只提示有长尾或异常候选；它不等于等待时间、通信开销或空泡，需要用明细和时间线验证。"))
    kp = path.parent / "kernel_details.csv"
    k = None
    if kp.is_file():
        k = read_kernel(kp, device)
        result.append('<h3>Kernel 明细：事件级诊断</h3>')
        result.append("<p>读取 " + str(k["total"]) + " 行；无效 Duration " + str(k["invalid"])
                      + " 行；涉及 " + str(len(k["streams"])) + " 条 Stream。时间范围以明细首末记录为准，不代表完整采集窗口。</p>")
        if sample:
            result.append(info_note("当前通过 --sample 明确标记：kernel_details.csv 是截取样本，以下只描述截取内容，不能代表全窗口。"))
        compare = []
        for op in stats:
            if stat_type(op["name"]) != "其他":
                matching = k["types"].get(op["name"], 0)
                compare.append([op["name"], op["count"], matching])
        if compare:
            result.append(table(["算子类别", "统计表次数", "明细中同名 Type 次数"], compare))
            result.append(info_note("次数不一致可能来自截取、过滤或统计口径差异，不自动判断为采集错误。"))
        for cat in ("Attention候选", "MoE Dispatch"):
            result.append("<h4>" + esc(cat) + "：按时间分段的单次任务平均耗时</h4>")
            result.append(bin_chart(k["bins"], k["span_s"], cat))
        result.append(info_note("横轴按明细中的最早起点归零；柱子是该段内匹配任务的平均 Duration，空白不等于 NPU 空闲。等待时间字段不是全卡空泡时间。"))
        result.append("<h4>单次任务耗时分布</h4>")
        labels = ["<50", "50–100", "100–200", "200–500", "500–1,000",
                  "1,000–5,000", "5,000–10,000", "10,000–50,000",
                  "50,000–100,000", "≥100,000"]
        result.append(table(["Duration 区间 μs", "Attention候选次数", "MoE Dispatch 次数"],
                            [[label, k["hist"]["Attention候选"][i], k["hist"]["MoE Dispatch"][i]]
                             for i, label in enumerate(labels)]))
        topdata = [[t, fnum(d, 3), stream, start, wait] for d, _, t, stream, start, wait in k["top"]]
        result.append("<h4>最长的 15 个明细任务</h4>")
        result.append(table(["Type", "Duration μs", "Stream", "Start Time μs", "Wait Time μs（原值）"], topdata))
    else:
        result.append(info_note("同目录没有 kernel_details.csv，跳过事件级诊断。"))
    op = path.parent / "operator_details.csv"
    op_data = None
    if op.is_file() and allow_host:
        op_data = read_operator(op)
        names, times = op_data["names"], op_data["self_time"]
        result.append('<h3>Host 算子明细</h3><p>读取 ' + str(op_data["rows"]) + ' 行。按 Host Self Duration 汇总前 10 类；这不是 NPU kernel 耗时。</p>')
        if sample:
            result.append(info_note("operator_details.csv 标为截取样本，下面不能用于评价全窗口。"))
        result.append(table(["Name", "出现次数", "Host Self 累计 μs"],
                            [[name, names[name], fnum(value, 3)] for name, value in sorted(times.items(), key=lambda v: v[1], reverse=True)[:10]]))
    elif op.is_file():
        result.append(info_note("同一份 op_statistic.csv 包含多个 Device，operator_details.csv 无 Device_id：无法可靠归属，跳过 Host 对比。"))
    result.append("</section>")
    return "".join(result), k, op_data


def main():
    ap = argparse.ArgumentParser(description="生成 Ascend profiler 本地可视化报告，不需要安装第三方库。")
    ap.add_argument("--input", action="append", required=True, help="输入目录（可重复指定 BL 和 DYNAM 的目录）")
    ap.add_argument("--output", default="dyntra_report", help="输出目录，默认 ./dyntra_report")
    ap.add_argument("--sample", action="store_true", help="明确标记 kernel_details/operator_details 为截取样本")
    args = ap.parse_args()
    roots = [Path(p).expanduser().resolve() for p in args.input]
    out = Path(args.output).expanduser().resolve()
    paths = []
    seen = set()
    for root in roots:
        if not root.is_dir():
            ap.error(f"输入目录不存在: {root}")
        for name in ("kernel_details.csv", "operator_details.csv"):
            for detail in root.rglob(name):
                if not (detail.parent / "op_statistic.csv").is_file():
                    print("⚠️ 跳过明细（同目录缺少 op_statistic.csv）:", detail, flush=True)
        for p in sorted(root.rglob("op_statistic.csv")):
            if p.resolve() not in seen:
                seen.add(p.resolve())
                paths.append((root, p))
    if not paths:
        raise SystemExit("没有找到 op_statistic.csv；请检查 --input 目录。")
    out.mkdir(parents=True, exist_ok=True)
    sections = []
    records = []
    by_group = defaultdict(list)
    for root, path in paths:
        groups = read_stats(path)
        if not groups:
            print("跳过：无法读取 OP Type / Total Time(us) / Count:", path)
        for device, stats in groups.items():
            print("处理:", path, "Device", device, flush=True)
            section, k, op_data = group_html(root, path, device, stats, args.sample, records,
                                             allow_host=len(groups) == 1)
            by_group[(str(root), str(capture_group(root, path)))].append((device, path, stats, k, op_data))
            sections.append(section)
    if not sections:
        raise SystemExit("没有可用算子统计；请检查 CSV 列名与数值。")
    with (out / "operator_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["input_root", "source_file", "device_id", "op_type", "core_type",
                         "count", "total_us", "avg_us", "min_us", "max_us", "percent_of_op_total"])
        writer.writerows(records)
    page = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>DyntraLB Profiling 分析</title><style>
*{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:#19283b;font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
header{background:#132846;color:white;padding:32px max(24px,calc((100% - 1180px)/2))}
header h1{margin:0;font-size:28px}header p{margin:8px 0 0;color:#dbe8fa}
main{max-width:1180px;margin:24px auto;padding:0 20px}.card{background:white;border:1px solid #dce4ef;border-radius:12px;padding:24px;margin:18px 0;overflow:hidden}
h2{font-size:17px;overflow-wrap:anywhere;margin:0 0 16px}h3{margin:28px 0 10px;font-size:18px}h4{margin:18px 0 4px}
.note{background:#edf4ff;border-left:4px solid #527fd3;padding:10px 14px;margin:14px 0}.muted{color:#64748b}
.barrow{display:grid;grid-template-columns:minmax(140px,290px) 1fr 110px;gap:12px;align-items:center;margin:9px 0}
.barlabel{overflow-wrap:anywhere;font-size:13px}.bartrack{height:18px;background:#edf1f7;border-radius:3px}.bar{height:100%;background:#4474d2;border-radius:3px}
.barvalue{text-align:right;font-variant-numeric:tabular-nums}.tablewrap{overflow-x:auto}table{border-collapse:collapse;min-width:720px;width:100%;font-size:13px}
th,td{padding:8px 9px;text-align:left;border-bottom:1px solid #e5eaf2;white-space:nowrap}th{background:#f3f6fa}td:first-child{white-space:normal}
.plotwrap{overflow-x:auto}.plotwrap svg{width:100%;min-width:620px;max-height:200px}footer{color:#617089;font-size:13px}
@media(max-width:700px){.barrow{grid-template-columns:125px 1fr 75px;gap:6px}.card{padding:16px}.barvalue{font-size:12px}}
</style></head><body><header><h1>DyntraLB Profiling 分析</h1><p>基于 op_statistic.csv 和可用明细的单卡诊断</p></header><main>
<section class="card"><h2>先看这里</h2><p>每个数据目录和 Device 单独展示。Attention、MoE 按名称匹配，具体归属需结合实现确认。
比较多个目录时请自行确认 BL / DYNAM、采样窗口、rank 和请求负载可比。</p>
<div class="note">本报告不能单独证明跨卡 Attention 空泡，也不测量调度进程中的 CPU 全局状态 All-gather。
文件是否截取需要用户确认；未使用 --sample 也不代表文件一定完整。</div></section>"""
    aligned_records = []
    comparison_records = []
    for (root, group_dir), groups in by_group.items():
        if len(groups) < 2:
            continue
        page += '<section class="card"><h2>跨 Device 概览：' + esc(group_dir) + '</h2>'
        page += info_note("依据目录结构推测同组；请核对属于同一次采集、时间窗口和实验模式。Device 编号不一定等于 DP rank。平均单次耗时差异不等于卡间同步等待或空泡。")
        devices = [device for device, _, _, _, _ in groups]
        if len(set(devices)) != len(devices):
            page += info_note("同组出现重复 Device_id：可能包含多次采集，需检查来源，不能将它们当作不同卡。")
        summary = []
        for device, path, stats, _, _ in groups:
            def metric(cat):
                matches = [s for s in stats if stat_type(s["name"]) == cat]
                count = sum(s["count"] for s in matches)
                avg = sum(s["total"] for s in matches) / count if count else None
                maximum = max((s["max"] for s in matches if s["max"] is not None), default=None)
                return count, avg, maximum
            ac, aa, am = metric("Attention候选")
            dc, da, dm = metric("MoE Dispatch")
            summary.append([device, str(path.parent), ac, fnum(aa, 3), fnum(am, 3),
                            dc, fnum(da, 3), fnum(dm, 3)])
        page += table(["Device_id", "源目录", "Attention 次数", "Attention 平均 μs",
                       "Attention 最大 μs", "Dispatch 次数", "Dispatch 平均 μs",
                       "Dispatch 最大 μs"], summary)
        page += "<h3>op_statistic.csv：按算子和 Core 类型横向对比</h3>"
        page += info_note("每格显示单卡的平均、调用次数、累计耗时；每个算子及 Core 类型分别对齐。展示单卡累计耗时较高的前 30 项，完整记录见 rank_comparison.csv。")
        page += comparison_matrix(groups, group_dir, "op_statistic", comparison_records, root)
        page += "<h3>kernel_details.csv：按任务 Type 横向对比</h3>"
        page += info_note("由每张卡的明细计算；仅根据有效 Duration 计算平均。明细若被截取或过滤，则只代表提供的部分，不等于同步等待或设备空泡。")
        page += comparison_matrix(groups, group_dir, "kernel_details", comparison_records, root)
        page += "<h3>operator_details.csv：Host Self 横向对比</h3>"
        page += info_note("按 API Name 比较 CPU 侧 Host Self Duration，按单卡累计值选择前 30 项。线程可能并行；不将各卡或线程累计时间当作端到端耗时。")
        page += comparison_matrix(groups, group_dir, "host_self", comparison_records, root)
        page += "<h3>operator_details.csv：Device Self 横向对比</h3>"
        page += info_note("这是框架提供的算子 Device Self 字段，与 kernel_details 属于不同口径。0 可能来自采集或关联情况，不能直接解释为没有设备计算。")
        page += comparison_matrix(groups, group_dir, "device_self", comparison_records, root)
        for cat in ("Attention候选", "MoE Dispatch"):
            values = []
            for device, path, stats, _, _ in groups:
                matches = [s for s in stats if stat_type(s["name"]) == cat]
                count = sum(s["count"] for s in matches)
                if count:
                    label = "Device " + device + " · " + str(path.parent.relative_to(Path(root)))
                    values.append((label, sum(s["total"] for s in matches) / count))
            if values:
                page += "<h3>" + esc(cat) + "：各 Device 平均单次耗时</h3>" + bars(values, "μs")
        aligned = aligned_attention(groups)
        if aligned:
            start, span, series = aligned
            page += "<h3>共同时间段：各 Device 的 Attention 候选任务</h3>"
            page += info_note("按各文件中首末记录的时间交集，以任务开始时间分桶；鼠标悬停显示次数和平均耗时。灰色只代表该桶没有匹配记录，不代表设备空闲。请核对时间戳是否同一时钟、同一采集；相同时段也未必是同一层或同一轮，不能直接据此计算空泡。")
            page += attention_heatmap(span, series)
            for device, path, bins in series:
                for i, b in enumerate(bins):
                    avg = b["sum"] / b["count"] if b["count"] else ""
                    aligned_records.append([group_dir, device, str(path.parent),
                        str(start), str(span * i / N_BINS / Decimal(1000000)),
                        str(span * (i+1) / N_BINS / Decimal(1000000)),
                        b["count"], avg, b["max"] if b["count"] else ""])
        else:
            page += info_note("未绘制跨卡共同时间图：需要至少两张不同 Device 的 kernel_details，且记录时间范围有交集。")
        page += "</section>"
    page += "".join(sections) + '<footer>按表内字段生成；时间单位已标注。数据未上传网络。</footer></main></body></html>'
    with (out / "attention_aligned_bins.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["capture_group", "device_id", "source_directory", "common_start_us",
                         "bin_start_relative_s", "bin_end_relative_s", "attention_count",
                         "avg_duration_us", "max_duration_us"])
        writer.writerows(aligned_records)
    with (out / "rank_comparison.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["capture_group", "source_kind", "source_file", "device_id", "name", "core_type",
                         "record_count", "valid_duration_count", "total_us", "avg_us", "max_us"])
        writer.writerows(comparison_records)
    (out / "report.html").write_text(page, encoding="utf-8")
    print("报告:", out / "report.html")
    print("汇总:", out / "operator_summary.csv")
    print("跨卡明细:", out / "rank_comparison.csv",
          "(没有多个设备的组时只有表头)" if not comparison_records else "")
    if aligned_records:
        print("按时间分段:", out / "attention_aligned_bins.csv")
    else:
        print("没有可用的多卡共同时间段；attention_aligned_bins.csv 只有表头。")


if __name__ == "__main__":
    main()

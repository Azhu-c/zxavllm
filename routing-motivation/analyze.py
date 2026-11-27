"""Summarize all collected evidence. Missing waiting evidence stays unavailable."""
from __future__ import annotations
from bisect import bisect_right
from collections import defaultdict
import csv
import gzip
import json
import math
from pathlib import Path
import re
import statistics


def rows(path):
    if not path.exists():
        return []
    result = []
    for line in path.read_text().splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # A concurrently written final line can be incomplete.
    return result


def percentile(values, q):
    if not values:
        return None
    data = sorted(values)
    p = (len(data) - 1) * q
    low = int(p)
    return data[low] + (data[min(low + 1, len(data) - 1)] - data[low]) * (p - low)


def cv(values):
    mean = statistics.mean(values)
    return statistics.pstdev(values) / mean if mean else None


def csv_file(path, data):
    if not data:
        path.write_text("", encoding="utf-8")
        return
    columns = list(dict.fromkeys(k for row in data for k in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(data)


def performance(run):
    requests = rows(run / "requests.jsonl")
    meta = json.loads((run / "measurement.json").read_text())
    ttft, tpot, gaps, queue = [], [], [], []
    for r in requests:
        if r["error"] or not r["content_chunk_s"]:
            continue
        content = r["content_chunk_s"]
        ttft.append((content[0] - r["scheduled_s"]) * 1000)
        queue.append((r["dispatch_s"] - r["scheduled_s"]) * 1000)
        if r["output_tokens"] > 1:
            tpot.append((content[-1] - content[0]) * 1000 / (r["output_tokens"] - 1))
        gaps.extend((b - a) * 1000 for a, b in zip(content, content[1:]))
    valid = (meta["errors"] == 0 and len(requests) == meta["count"] and
             not meta["profile_errors"])
    return {"run": run.name, "policy": run.name.split("-r")[0],
            "measurement": meta["measurement"], "valid": valid,
            "requests": len(requests), "errors": meta["errors"], "seconds": meta["seconds"],
            "output_tokens": sum(r["output_tokens"] for r in requests if not r["error"]),
            "output_tokens_s": (sum(r["output_tokens"] for r in requests) / meta["seconds"]
                                if valid else None),
            "ttft_p50_ms": percentile(ttft, .5), "ttft_p95_ms": percentile(ttft, .95),
            "tpot_p50_ms": percentile(tpot, .5), "tpot_p95_ms": percentile(tpot, .95),
            "chunk_gap_p95_ms": percentile(gaps, .95),
            "client_queue_p95_ms": percentile(queue, .95)}, meta


def load_metrics(run, meta):
    observations = []
    for file in (run / "observations").glob("state-*.jsonl"):
        observations.extend(row for row in rows(file)
                            if meta["window_start_wall_ns"] <= row["wall_ns"] <= meta["window_end_wall_ns"])
    observations.sort(key=lambda r: r["wall_ns"])
    metrics = []
    series = []
    for row in observations:
        active = [r["running_blocks"] for r in row["ranks"]]
        owned = row["owned_blocks_with_pending"]
        counts = [r["running_count"] + r["waiting_count"] for r in row["ranks"]]
        point = {"seconds": (row["wall_ns"] - meta["window_start_wall_ns"]) / 1e9,
                 "active_blocks_cv": cv(active), "owned_blocks_cv": cv(owned),
                 "owned_count_cv": cv(counts)}
        metrics.append(point)
        for rank in row["ranks"]:
            series.append({"seconds": point["seconds"], **rank,
                           "owned_blocks_with_pending": owned[rank["rank"]]})
    csv_file(run / "load-timeseries.csv", series)
    # Time weights account for slightly irregular RPC/poll intervals.
    weights = [max(0, min((metrics[i + 1]["seconds"] if i + 1 < len(metrics)
                          else meta["seconds"]) - point["seconds"], 1.0))
               for i, point in enumerate(metrics)]
    result = {"state_samples": len(metrics)}
    for key in ("active_blocks_cv", "owned_blocks_cv", "owned_count_cv"):
        pairs = [(r[key], w) for r, w in zip(metrics, weights) if r[key] is not None]
        result[key + "_mean"] = (sum(v * w for v, w in pairs) / sum(w for _, w in pairs)
                                 if pairs and sum(w for _, w in pairs) else None)
        result[key + "_p95"] = percentile([v for v, _ in pairs], .95)
    return result


def correlation_ids(event):
    args = event.get("args") or {}
    return {(re.sub(r"[^a-z]", "", key.lower()), str(value))
            for key, value in args.items()
            if re.sub(r"[^a-z]", "", key.lower()) in
            ("externalid", "correlation", "correlationid")}


def attention_trace(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        document = json.load(handle)
    events = document.get("traceEvents", []) if isinstance(document, dict) else document
    markers = []
    for event in events:
        if str(event.get("name", "")).startswith("MOT_ATTN|") and event.get("ph") == "X":
            try:
                markers.append((event, json.loads(event["name"].split("|", 1)[1])))
            except (ValueError, KeyError):
                pass
    ranks = {marker["rank"] for _, marker in markers}
    if len(ranks) == 1:
        rank = next(iter(ranks))
    else:
        rank = None
        # Torch-NPU writes the explicit distributed rank in profiler_info*.json.
        # The experiment checks TP=1, so global distributed rank equals DP rank.
        for parent in (path.parent, path.parent.parent):
            candidates = set()
            for info in parent.glob("profiler_info*.json"):
                try:
                    value = json.loads(info.read_text()).get("rank_id")
                    if value is not None:
                        candidates.add(int(value))
                except (OSError, ValueError, TypeError):
                    pass
            if len(candidates) == 1:
                rank = next(iter(candidates))
                break
        if rank is None:
            match = re.search(r"(?:^|[/_.-])(?:dp(?:_rank)?|rank)[_-]?(\d+)(?:[/_.-]|$)", str(path))
            rank = int(match.group(1)) if match else None
    process_names = {e["pid"]: str((e.get("args") or {}).get("name", ""))
                     for e in events if e.get("ph") == "M" and e.get("name") == "process_name"}
    ranges = defaultdict(list)
    for event, marker in markers:
        ranges[(event.get("pid"), event.get("tid"))].append((float(event["ts"]),
            float(event["ts"]) + float(event.get("dur", 0)), marker))
    starts = {}
    for thread, spans in ranges.items():
        spans.sort(key=lambda x: x[0])
        starts[thread] = [s[0] for s in spans]
    def owner(event):
        thread = (event.get("pid"), event.get("tid"))
        if thread not in ranges:
            return None
        ts = float(event.get("ts", 0))
        i = bisect_right(starts[thread], ts) - 1
        if i >= 0 and ts + float(event.get("dur", 0)) <= ranges[thread][i][1] + .01:
            return ranges[thread][i][2]
        return None
    correlations = {}
    for event in events:
        if event.get("ph") == "X":
            marker = owner(event)
            if marker:
                for key in correlation_ids(event):
                    correlations[key] = marker
    kernels, matched = [], []
    base_ns = document.get("baseTimeNanoseconds", 0) if isinstance(document, dict) else 0
    for event in events:
        if event.get("ph") != "X":
            continue
        name = str(event.get("name", ""))
        simple = name.lower().replace("_", "")
        category = str(event.get("cat", "")).lower()
        hardware = process_names.get(event.get("pid"), "").lower()
        # Exact device kernel names avoid counting framework/enqueue/tiling aliases.
        is_attention = simple in ("fusedinferattentionscore", "pagedattention", "flashattention")
        is_device = ("cpu_op" not in category and "python" not in category and
                     "enqueue" not in category and "dequeue" not in category and
                     (is_attention or any(s in hardware for s in ("hardware", "device", "npu"))))
        if not is_attention or not is_device:
            continue
        dur = float(event.get("dur", 0))
        if dur <= 0:
            continue
        end = float(event["ts"]) + dur
        if end < 1e14 and base_ns:
            end += float(base_ns) / 1000
        marker = next((correlations[k] for k in correlation_ids(event) if k in correlations), None)
        kernels.append({"rank": rank, "duration_us": dur, "name": name})
        if marker is not None:
            matched.append({**marker, "end_us": end})
    return rank, kernels, matched


def profiler_metrics(run):
    root = run / "profiler"
    # Ascend's trace_view and the outer torch trace can contain the same kernels.
    # Select a single export family instead of adding both.
    paths = sorted(root.rglob("trace_view.json"))
    if not paths:
        paths = sorted(root.rglob("*.pt.trace.json")) + sorted(root.rglob("*.pt.trace.json.gz"))
    by_rank, grouped = defaultdict(list), defaultdict(dict)
    files = []
    for path in paths:
        try:
            rank, kernels, matched = attention_trace(path)
            files.append({"path": str(path.relative_to(run)), "rank": rank,
                          "device_attention_kernels": len(kernels), "correlated_kernels": len(matched)})
            if rank is not None:
                by_rank[rank].extend(k["duration_us"] for k in kernels)
            for marker in matched:
                if marker.get("wave") is None or marker.get("step") is None or marker["layer"] == "unknown":
                    continue
                key = (marker["wave"], marker["step"], marker["layer"], marker.get("occurrence", 0))
                grouped[key][marker["rank"]] = max(grouped[key].get(marker["rank"], 0), marker["end_us"])
        except Exception as error:
            files.append({"path": str(path.relative_to(run)), "error": repr(error)})
    rank_rows = [{"rank": rank, "calls": len(values), "mean_us": statistics.mean(values),
                  "p50_us": percentile(values, .5), "p95_us": percentile(values, .95)}
                 for rank, values in sorted(by_rank.items()) if values]
    csv_file(run / "attention-kernels.csv", rank_rows)
    csv_file(run / "trace-coverage.csv", files)
    args_file = run / "decoder-effective-args.json"
    expected = 8
    if args_file.exists():
        args = json.loads(args_file.read_text())
        if "--data-parallel-size" in args:
            expected = int(args[args.index("--data-parallel-size") + 1])
    complete = len(rank_rows) == expected
    # This proxy describes attention *duration imbalance*. It is not measured idle time.
    means = [row["mean_us"] for row in rank_rows]
    duration_imbalance = (1 - statistics.mean(means) / max(means)) if complete and max(means) > 0 else None
    completion_rows = []
    for key, ranks in grouped.items():
        if set(ranks) != set(range(expected)) or min(ranks.values()) < 1e14:
            continue
        # Same-host epoch timestamps are a clock-alignment assumption, recorded here.
        latest = max(ranks.values())
        completion_rows.append({"wave": key[0], "step": key[1], "layer": key[2],
                                "occurrence": key[3], "completion_spread_us": latest - min(ranks.values()),
                                "mean_completion_gap_us": statistics.mean(latest - t for t in ranks.values()),
                                "clock_basis": "same-host epoch timestamps; cross-device offset not independently calibrated"})
    csv_file(run / "attention-completion-spread.csv", completion_rows)
    # Retain explicit HCCL wait fields without conflating groups/steps or double counting totals.
    comm_rows = []
    def walk(value, path, filename):
        if isinstance(value, dict):
            for k, v in value.items():
                if k.lower().replace(" ", "") in ("waittime(ms)", "wait_time(ms)") and isinstance(v, (int, float)):
                    comm_rows.append({"file": filename, "path": "/".join(path), "field": k, "wait_ms": v})
                walk(v, path + [str(k)], filename)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                walk(v, path + [str(i)], filename)
    for path in root.rglob("communication.json"):
        try:
            walk(json.loads(path.read_text()), [], str(path.relative_to(run)))
        except (OSError, ValueError):
            pass
    csv_file(run / "communication-wait-fields.csv", comm_rows)
    result = {"profile_trace_files": len(paths), "attention_ranks_found": len(rank_rows),
              "attention_duration_imbalance_proxy": duration_imbalance,
              "correlated_attention_groups": len(completion_rows),
              "attention_completion_spread_p50_us": percentile([r["completion_spread_us"] for r in completion_rows], .5),
              "attention_completion_spread_p95_us": percentile([r["completion_spread_us"] for r in completion_rows], .95),
              "hccl_wait_fields_found": len(comm_rows),
              "attention_bubble_wait_ms": None,
              "bubble_status": "unavailable: completion skew and collective wait alone do not isolate attention-induced idle time"}
    (run / "profile-summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def report(session):
    summaries = []
    for run in sorted(session.glob("*-r*")):
        if not (run / "measurement.json").exists():
            continue
        summary, meta = performance(run)
        summary.update(load_metrics(run, meta))
        if meta["measurement"] == "profile":
            summary.update(profiler_metrics(run))
        summaries.append(summary)
    csv_file(session / "summary.csv", summaries)
    (session / "report.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    def fmt(value):
        return "N/A" if value is None else f"{value:.3f}"
    if any(s["measurement"] == "profile" for s in summaries):
        print("\nPROFILE / EAGER diagnostic results; keep separate from performance runs.")
    print("\nrun           valid   completion(s)  output(tok/s)  TTFT-p95(ms)  TPOT-p95(ms)  active-KV-CV")
    for s in summaries:
        print(f"{s['run']:13} {str(s['valid']):7} {s['seconds']:14.3f} {fmt(s['output_tokens_s']):>14} "
              f"{fmt(s['ttft_p95_ms']):>13} {fmt(s['tpot_p95_ms']):>13} {fmt(s['active_blocks_cv_mean']):>13}")
    print("\nPerformance repetitions (median; excludes profiler and validation runs):")
    aggregates = []
    for policy in ("rr", "jsq", "kv"):
        eligible = [s for s in summaries if s["policy"] == policy and s["measurement"] == "performance"
                    and s["valid"] and s["requests"] == 96]
        if eligible:
            aggregate = {"policy": policy, "n": len(eligible)}
            for k in ("seconds", "output_tokens_s", "ttft_p95_ms", "tpot_p95_ms", "active_blocks_cv_mean"):
                vals = [s[k] for s in eligible if s[k] is not None]
                aggregate[k] = statistics.median(vals) if vals else None
            aggregates.append(aggregate)
            print(f"{policy.upper()}: n={len(eligible)}, completion={fmt(aggregate['seconds'])}s, "
                  f"output={fmt(aggregate['output_tokens_s'])} tok/s, "
                  f"active KV CV={fmt(aggregate['active_blocks_cv_mean'])}")
    csv_file(session / "policy-medians.csv", aggregates)
    for s in summaries:
        if s["measurement"] == "profile":
            print(f"{s['run']}: attention ranks={s['attention_ranks_found']}, "
                  f"duration imbalance proxy={fmt(s['attention_duration_imbalance_proxy'])}; "
                  f"matched layer/step groups={s['correlated_attention_groups']}; "
                  "attention-induced bubble wait=N/A")
    print("SSE chunk gaps are retained as chunk gaps; with MTP they are not per-token ITL.")
    print(f"Saved: {session / 'summary.csv'}")
    return summaries

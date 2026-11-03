#!/usr/bin/env python3
"""Summarize verified rank/step/layer events exported from an NPU trace.

This utility does not identify profiler events or synchronize device clocks.
The user must first verify those operations and export the CSV described in the
README. Times are microseconds on a common clock. No values are fabricated.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

FIELDS = ("step", "layer", "rank", "attention_start_us", "attention_end_us",
          "collective_ready_us", "step_start_us", "step_end_us")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--ranks", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.ranks <= 1:
        raise ValueError("ranks must be > 1")
    groups = defaultdict(dict)
    step_bounds = defaultdict(dict)
    with args.csv.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not set(FIELDS).issubset(reader.fieldnames):
            raise ValueError("CSV requires columns: " + ", ".join(FIELDS))
        for line, row in enumerate(reader, 2):
            try:
                step, layer, rank = (int(row[x]) for x in ("step", "layer", "rank"))
                a, b, ready, start, end = (float(row[x]) for x in FIELDS[3:])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"line {line}: invalid numeric field") from exc
            if any(not math.isfinite(x) for x in (a, b, ready, start, end)):
                raise ValueError(f"line {line}: non-finite timestamp")
            if not 0 <= rank < args.ranks or not start <= a <= b <= ready <= end:
                raise ValueError(f"line {line}: invalid rank or event ordering")
            if rank in groups[step, layer]:
                raise ValueError(f"line {line}: duplicate step/layer/rank")
            old = step_bounds[step].get(rank)
            if old is not None and old != (start, end):
                raise ValueError(f"line {line}: inconsistent step boundaries")
            step_bounds[step][rank] = (start, end)
            groups[step, layer][rank] = {"attention_us": b - a, "ready_us": ready}
    if not groups:
        raise ValueError("CSV contains no events")
    expected = set(range(args.ranks))
    group_metrics = []
    for (step, layer), ranks in sorted(groups.items()):
        if set(ranks) != expected:
            raise ValueError(f"step {step}, layer {layer}: missing or extra ranks")
        latest = max(x["ready_us"] for x in ranks.values())
        earliest = min(x["ready_us"] for x in ranks.values())
        group_metrics.append({"step": step, "layer": layer,
                              "attention_us_by_rank": {str(r): ranks[r]["attention_us"] for r in sorted(ranks)},
                              "ready_skew_us": latest - earliest,
                              "early_arrival_gap_us_by_rank": {str(r): latest - ranks[r]["ready_us"] for r in sorted(ranks)}})
    spans = []
    for step, ranks in sorted(step_bounds.items()):
        if set(ranks) != expected:
            raise ValueError(f"step {step}: missing rank boundaries")
        # Trace-derived cross-rank span; it may include overlap or launch skew.
        spans.append({"step": step, "span_us": max(v[1] for v in ranks.values()) - min(v[0] for v in ranks.values())})
    result = {"source": str(args.csv.resolve()), "ranks": args.ranks,
              "groups": group_metrics, "step_spans": spans,
              "median_ready_skew_us": statistics.median(x["ready_skew_us"] for x in group_metrics),
              "median_step_span_us": statistics.median(x["span_us"] for x in spans),
              "note": "Early-arrival gaps are readiness skew, not measured communication or device idle time."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"{len(group_metrics)} step/layer groups, {len(spans)} steps; result: {args.output.resolve()}")
    print(f"median ready skew: {result['median_ready_skew_us']:.3f} us")
    print(f"median trace-derived step span: {result['median_step_span_us']:.3f} us")


if __name__ == "__main__":
    main()

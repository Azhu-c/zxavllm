# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare vllm bench serve JSON saved with --save-detailed.

TPOT is calculated per successful request as sum(ITLs)/(output_len-1).
It includes pauses between output tokens; whole-run throughput cannot expose
those pauses. Input lengths define the long-context cohort.
"""

import argparse
import json
import math
from pathlib import Path


def percentiles(values):
    if not values:
        return {"count": 0}
    values = sorted(values)

    def pct(fraction):
        pos = (len(values) - 1) * fraction
        lo = math.floor(pos)
        hi = math.ceil(pos)
        return values[lo] + (values[hi] - values[lo]) * (pos - lo)

    return {
        "count": len(values),
        "mean_ms": sum(values) / len(values) * 1000,
        **{f"p{p}_ms": pct(p / 100) * 1000 for p in (50, 95, 99)},
    }


def summarize(data, threshold):
    keys = ("ttfts", "input_lens", "output_lens", "itls")
    if any(key not in data for key in keys):
        raise ValueError("Need ttfts, input_lens, output_lens and itls: use --save-detailed")
    arrays = [data[key] for key in keys]
    if len({len(a) for a in arrays}) != 1:
        raise ValueError("Per-request arrays have different lengths")
    groups = {name: {"ttft": [], "tpot": [], "itl": []} for name in ("all", "short", "long")}
    for ttft, input_len, output_len, itls in zip(*arrays):
        if output_len < 1 or ttft <= 0:
            continue
        if not all(math.isfinite(x) for x in (ttft, input_len, output_len, *itls)):
            raise ValueError("Non-finite benchmark value")
        if any(itl < 0 for itl in itls):
            raise ValueError("Negative inter-token latency")
        for group in ("all", "long" if input_len >= threshold else "short"):
            groups[group]["ttft"].append(ttft)
            groups[group]["itl"].extend(itls)
            if output_len > 1:
                if not itls:
                    raise ValueError("Multi-token response has no ITLs; use streaming benchmark output")
                groups[group]["tpot"].append(sum(itls) / (output_len - 1))
    result = {
        name: {metric: percentiles(values) for metric, values in metrics.items()} for name, metrics in groups.items()
    }
    result["output_throughput"] = data.get("output_throughput")
    result["completed"] = data.get("completed")
    result["failed_or_missing_output"] = len(arrays[0]) - len(groups["all"]["ttft"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--long-input-tokens", type=int, default=32768)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    comparison = {str(path): summarize(json.loads(path.read_text()), args.long_input_tokens) for path in args.results}
    rendered = json.dumps(comparison, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()

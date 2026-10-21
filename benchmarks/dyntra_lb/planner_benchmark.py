# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU-only planning overhead, not an NPU serving/TPOT benchmark."""

import argparse
import ast
import importlib.util
import json
import random
import statistics
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, TypedDict


def load_planner(path):
    spec = importlib.util.spec_from_file_location("dyntra_lb_planner_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.balance_load


def load_baseline(path):
    tree = ast.parse(path.read_text())
    typed = [
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in ("_RequestItem", "_CardData", "_Modification")
    ]
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DyntraLBDPEngineCoreProc")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_balance_load")
    method.decorator_list = []
    ns = dict(Sequence=Sequence, Literal=Literal, TypedDict=TypedDict)
    exec(compile(ast.Module(body=[*typed, method], type_ignores=[]), str(path), "exec"), ns)
    return ns["_balance_load"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.iterations <= 0 or args.max_num_seqs <= 0:
        parser.error("iterations and max-num-seqs must be positive")
    rng = random.Random(12292)
    ranks = []
    for rank in range(8):
        blocks = [rng.randint(1, 80) for _ in range(args.max_num_seqs * 2)]
        if rank < 2:
            blocks[0] = 4000
        ranks.append((blocks, args.max_num_seqs))
    root = Path(__file__).resolve().parents[2]
    planners = {"modified": load_planner(root / "vllm_ascend/core/dyntra_lb_planner.py")}
    if args.baseline_source:
        planners["pr_12292"] = load_baseline(args.baseline_source)
    results = {
        "scope": "CPU planner only; no NPU performance claim",
        "ranks": 8,
        "max_num_seqs": args.max_num_seqs,
        "iterations": args.iterations,
        "results": {},
    }
    for label, planner in planners.items():
        times = []
        pauses = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            plan = planner(ranks, 8, max_num_seqs=args.max_num_seqs)
            times.append((time.perf_counter() - start) * 1000)
            pauses.append(sum(len(p["out_blk"]) for p in plan))
        results["results"][label] = {
            "median_ms": statistics.median(times),
            "max_ms": max(times),
            "mean_pause_decisions": statistics.mean(pauses),
        }
    rendered = json.dumps(results, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()

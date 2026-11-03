#!/usr/bin/env python3
"""Controlled same-rank running/waiting experiment for Attention bubbles."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import hashlib
import json
import re
import statistics
import time
from pathlib import Path

from motivation1 import make_prompt

RANKS = 8
PER_KIND = 4
ACTIVE_PER_RANK = 4


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def active(row, condition):
    if condition == "balanced":
        return row["index"] < 2
    if condition == "skewed":
        return row["kind"] == ("long" if row["rank"] < RANKS // 2 else "short")
    raise ValueError(condition)


def generate(args):
    if args.short_tokens >= args.long_tokens or args.short_tokens <= 0:
        raise ValueError("Require 0 < short-tokens < long-tokens")
    if args.long_tokens + args.output_tokens > args.max_model_len:
        raise ValueError("long prompt + output exceeds max-model-len")
    if args.output_tokens <= 0:
        raise ValueError("output-tokens must be positive")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=args.trust_remote_code)
    prompts = {}
    for kind, target in (("short", args.short_tokens), ("long", args.long_tokens)):
        # Identical prompts within each length class keep the global active prompt
        # multiset identical under both placements. Prefix caching must be disabled.
        text, n = make_prompt(tokenizer, target, kind, args.seed)
        prompts[kind] = (text, n)
    rows = []
    for rank in range(RANKS):
        for kind in ("long", "short"):
            for index in range(PER_KIND):
                prompt, n = prompts[kind]
                rows.append({"id": f"r{rank}-{kind[0]}{index}", "rank": rank,
                             "kind": kind, "index": index, "prompt": prompt,
                             "prompt_tokens": n, "prompt_sha256": digest(prompt),
                             "output_tokens": args.output_tokens})
    check_manifest(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "local_pairs.jsonl"
    if path.exists():
        raise FileExistsError(f"{path} already exists; choose another output directory")
    with path.open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"生成 {len(rows)} 个请求：{path.resolve()}")
    print("各 rank 固定 4 长 + 4 短；所有请求输出", args.output_tokens, "tokens")


def check_manifest(rows):
    if len(rows) != RANKS * PER_KIND * 2:
        raise ValueError("manifest must contain exactly 64 requests for eight ranks")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate request IDs")
    for rank in range(RANKS):
        local = [r for r in rows if r["rank"] == rank]
        if Counter((r["kind"], r["index"]) for r in local) != Counter(
                (kind, i) for kind in ("long", "short") for i in range(PER_KIND)):
            raise ValueError(f"rank {rank} must have four long and four short requests")
    for row in rows:
        if digest(row["prompt"]) != row["prompt_sha256"]:
            raise ValueError(f"prompt hash mismatch: {row['id']}")
        if int(row["prompt_tokens"]) <= 0 or int(row["output_tokens"]) <= 0:
            raise ValueError(f"invalid token count: {row['id']}")
    for kind in ("long", "short"):
        members = [r for r in rows if r["kind"] == kind]
        if len({(r["prompt_sha256"], r["prompt_tokens"], r["output_tokens"]) for r in members}) != 1:
            raise ValueError(f"{kind} requests must have identical prompts and lengths")
    if len({r["output_tokens"] for r in rows}) != 1:
        raise ValueError("all output lengths must match")
    if rows[0]["output_tokens"] <= 0:
        raise ValueError("output length must be positive")
    if max(r["prompt_tokens"] for r in rows if r["kind"] == "short") >= min(
            r["prompt_tokens"] for r in rows if r["kind"] == "long"):
        raise ValueError("short prompts must be shorter than long prompts")


def request_set_from_state(state, queue, expected):
    """Match UUID issued by the proxy to vLLM's decorated engine request ID."""
    found = {}
    for req in state.get(queue, []):
        matches = [row for row in expected if row["proxy_request_id"] in req["request_id"]]
        if len(matches) != 1:
            raise ValueError(f"unexpected or ambiguous {queue} request {req['request_id']}")
        req_id = matches[0]["id"]
        if req_id in found:
            raise ValueError(f"duplicate scheduler request for {req_id}")
        found[req_id] = req
    return found


def validate_snapshot(state, rank, selected, pending, start_wall_ns, max_age):
    if state.get("rank") != rank or state.get("max_num_seqs") != ACTIVE_PER_RANK:
        raise ValueError(f"rank {rank}: wrong rank or max_num_seqs, expected 4")
    stamp = state.get("wall_time_ns", 0)
    age = (time.time_ns() - stamp) / 1e9
    if stamp < start_wall_ns or age < -2 or age > max_age:
        raise ValueError(f"rank {rank}: stale snapshot (age={age:.2f}s)")
    running = request_set_from_state(state, "running", selected + pending)
    waiting = request_set_from_state(state, "waiting", selected + pending)
    expected_running = {r["id"] for r in selected}
    expected_waiting = {r["id"] for r in pending}
    if set(running) != expected_running or set(waiting) != expected_waiting:
        raise ValueError(f"rank {rank}: expected four selected running and four pending waiting")
    by_id = {r["id"]: r for r in selected + pending}
    for rid, item in list(running.items()) + list(waiting.items()):
        if item["prompt_tokens"] != by_id[rid]["prompt_tokens"]:
            raise ValueError(f"rank {rank}: prompt token count mismatch for {rid}")
        if item.get("preemptions", 0):
            raise ValueError(f"rank {rank}: preempted request {rid}")
    scheduled = state.get("scheduled_tokens", {})
    actual = {r["request_id"] for r in running.values()}
    if set(scheduled) != actual or any(n != 1 for n in scheduled.values()):
        raise ValueError(f"rank {rank}: snapshot is not a four-request, one-token decode step")
    return {"rank": rank, "snapshot": state}


async def run(args):
    if args.sample_seconds <= 0 or args.timeout <= 0 or args.snapshot_max_age <= 0:
        raise ValueError("sample duration, timeout and snapshot max age must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        raise ValueError("run-id may contain only letters, digits, '-' and '_'")
    if not args.state_dir.is_absolute():
        raise ValueError("state-dir must be the same absolute path used by the decoder")
    manifest_bytes = args.manifest.read_bytes()
    rows = [json.loads(line) for line in manifest_bytes.decode("utf-8").splitlines() if line.strip()]
    check_manifest(rows)
    selected = [dict(r) for r in rows if active(r, args.condition)]
    pending = [dict(r) for r in rows if not active(r, args.condition)]
    assert len(selected) == len(pending) == 32
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"{args.condition}_{args.mode}_{args.run_id}.json"
    if path.exists():
        raise FileExistsError(path)
    import httpx
    clock = time.perf_counter()
    wall_ns = time.time_ns()
    summary = {"condition": args.condition, "mode": args.mode, "run_id": args.run_id,
               "manifest": str(args.manifest.resolve()),
               "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
               "model": args.model, "sample_seconds": args.sample_seconds,
               "start_wall_ns": wall_ns,
               "run_success": False, "valid_steady_window": False,
               "profile_started": False, "profile_stopped": False,
               "requests": [], "snapshots": []}
    previous_profiles = set()
    if args.mode == "profile" and args.profile_dir is not None and args.profile_dir.is_dir():
        previous_profiles = {str(p.resolve()) for p in args.profile_dir.glob("*ascend_pt")}
    limits = httpx.Limits(max_connections=96, max_keepalive_connections=96)
    auth = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    tasks = []
    task_rows = {}
    profiler_on = False
    async with httpx.AsyncClient(timeout=None, limits=limits, trust_env=False) as client:
        async def one(row):
            info = {"id": row["id"], "rank": row["rank"], "kind": row["kind"],
                    "group": "selected" if active(row, args.condition) else "pending",
                    "submitted_s": time.perf_counter() - clock, "first_token_s": None,
                    "completion_tokens": None, "finish_reason": None, "done_marker": False}
            summary["requests"].append(info)
            payload = {"model": args.model, "prompt": row["prompt"],
                       "max_tokens": row["output_tokens"], "min_tokens": row["output_tokens"],
                       "ignore_eos": True, "skip_special_tokens": False,
                       "temperature": 0.0, "stream": True,
                       "stream_options": {"include_usage": True}}
            headers = {**auth, "X-data-parallel-rank": str(row["rank"]),
                       "X-Experiment-Request-Id": row["id"]}
            try:
                async with client.stream("POST", args.proxy_url.rstrip("/") + "/v1/completions",
                                         json=payload, headers=headers) as response:
                    if response.status_code != 200:
                        body = (await response.aread())[:1000].decode(errors="replace")
                        raise RuntimeError(f"HTTP {response.status_code}: {body}")
                    if response.headers.get("X-Requested-DP-Rank") != str(row["rank"]):
                        raise RuntimeError("ranked proxy missing or rank acknowledgement mismatch")
                    proxy_id = response.headers.get("X-Proxy-Request-Id")
                    if not proxy_id or len(proxy_id) < 16:
                        raise RuntimeError("proxy did not return a usable request ID")
                    row["proxy_request_id"] = proxy_id
                    info["proxy_request_id"] = proxy_id
                    info["proxy_response_s"] = time.perf_counter() - clock
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            info["done_marker"] = True
                            break
                        try:
                            event = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        usage = event.get("usage") or {}
                        if usage.get("completion_tokens") is not None:
                            info["completion_tokens"] = usage["completion_tokens"]
                        for choice in event.get("choices") or []:
                            content = choice.get("text") or (choice.get("delta") or {}).get("content")
                            if content and info["first_token_s"] is None:
                                info["first_token_s"] = time.perf_counter() - clock
                                row["first_seen"] = True
                            if choice.get("finish_reason") is not None:
                                info["finish_reason"] = choice["finish_reason"]
                info["finish_s"] = time.perf_counter() - clock
                if not info["done_marker"]:
                    raise RuntimeError("stream ended without [DONE]")
                if info["first_token_s"] is None:
                    raise RuntimeError("no output content received")
                if info["completion_tokens"] != row["output_tokens"]:
                    raise RuntimeError(f"missing or unexpected output token count: {info['completion_tokens']}")
                return info
            except Exception as exc:
                info["error"] = str(exc)
                info["finish_s"] = time.perf_counter() - clock
                raise

        def launch(items):
            for row in items:
                task = asyncio.create_task(one(row))
                tasks.append(task)
                task_rows[task] = row

        def reject_finished(stage):
            for task in tasks:
                if task.done():
                    if task.cancelled():
                        raise RuntimeError(f"request cancelled during {stage}")
                    error = task.exception()
                    if error:
                        raise RuntimeError(f"request {task_rows[task]['id']} failed during {stage}: {error}")
                    raise RuntimeError(f"request {task_rows[task]['id']} completed during {stage}; increase output length")

        async def wait_until(predicate, label):
            deadline = time.monotonic() + args.timeout
            while not predicate():
                reject_finished(label)
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timeout waiting for {label}")
                await asyncio.sleep(0.1)
            reject_finished(label)

        def read_states():
            states = []
            for rank in range(RANKS):
                p = args.state_dir / f"rank-{rank}.json"
                with p.open(encoding="utf-8") as f:
                    state = json.load(f)
                rank_selected = [r for r in selected if r["rank"] == rank]
                rank_pending = [r for r in pending if r["rank"] == rank]
                states.append(validate_snapshot(state, rank, rank_selected, rank_pending,
                                                wall_ns, args.snapshot_max_age))
            return states

        try:
            print("提交第一批 32 个选中请求，等待它们开始生成……", flush=True)
            launch(selected)
            await wait_until(lambda: all(r.get("first_seen") for r in selected), "selected first output")
            summary["selected_ready_s"] = time.perf_counter() - clock
            print("第一批已开始生成；提交每个 rank 的 4 个等待候选……", flush=True)
            launch(pending)
            summary["all_submitted_s"] = time.perf_counter() - clock
            await wait_until(lambda: all("proxy_request_id" in r for r in selected + pending),
                             "all proxy responses")
            last_error = None
            deadline = time.monotonic() + args.timeout
            while True:
                reject_finished("scheduler verification")
                try:
                    states = read_states()
                    break
                except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                    last_error = str(exc)
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"scheduler state did not match: {last_error}")
                await asyncio.sleep(0.2)
            summary["ready_s"] = time.perf_counter() - clock
            summary["snapshots"].append({"phase": "ready", "elapsed_s": summary["ready_s"], "ranks": states})
            print("已核对每个 rank：4 个指定运行请求、4 个等待请求。开始稳定窗口。", flush=True)
            profile_base = args.decoder_url.rstrip("/")
            if args.mode == "profile":
                resp = await client.post(profile_base + "/start_profile", headers=auth, timeout=30)
                resp.raise_for_status()
                profiler_on = True
                summary["profile_started"] = True
            summary["window_start_s"] = time.perf_counter() - clock
            deadline = time.monotonic() + args.sample_seconds
            while time.monotonic() < deadline:
                reject_finished("steady window")
                await asyncio.sleep(min(0.5, max(0.0, deadline - time.monotonic())))
                states = read_states()
                summary["snapshots"].append({"phase": "window", "elapsed_s": time.perf_counter() - clock,
                                             "ranks": states})
            reject_finished("steady window")
            summary["window_end_s"] = time.perf_counter() - clock
            if profiler_on:
                resp = await client.post(profile_base + "/stop_profile", headers=auth, timeout=120)
                resp.raise_for_status()
                profiler_on = False
                summary["profile_stopped"] = True
            summary["valid_steady_window"] = True
            print("采样完成；继续等待全部 64 个请求完成……", flush=True)
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=args.finish_timeout)
            summary["completed_s"] = max(item["finish_s"] for item in summary["requests"])
            summary["total_output_tokens"] = sum(item["completion_tokens"] for item in summary["requests"])
            summary["request_batch_tokens_per_second"] = summary["total_output_tokens"] / summary["completed_s"]
            summary["run_success"] = True
        except Exception as exc:
            summary["error"] = str(exc)
            raise
        finally:
            if profiler_on:
                try:
                    resp = await client.post(args.decoder_url.rstrip("/") + "/stop_profile", headers=auth, timeout=120)
                    summary["profile_stopped"] = resp.status_code == 200
                except Exception as exc:
                    summary["stop_profile_error"] = str(exc)
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            summary["elapsed_s"] = time.perf_counter() - clock
            if args.mode == "profile" and args.profile_dir is not None:
                current_profiles = ({str(p.resolve()) for p in args.profile_dir.glob("*ascend_pt")}
                                    if args.profile_dir.is_dir() else set())
                summary["profile_dir"] = str(args.profile_dir.resolve())
                summary["new_profile_dirs"] = sorted(current_profiles - previous_profiles)
            with path.open("x", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"实验记录：{path.resolve()}", flush=True)


def report(args):
    entries = []
    manifest_hashes = set()
    configurations = set()
    for path in args.records:
        with path.open(encoding="utf-8") as f:
            row = json.load(f)
        if not row.get("run_success") or not row.get("valid_steady_window"):
            raise ValueError(f"invalid run: {path}: {row.get('error', 'missing success flag')}")
        if row.get("mode") != "timing":
            raise ValueError(f"{path}: completion time comparison requires timing mode")
        if not row.get("manifest_sha256") or not row.get("model"):
            raise ValueError(f"{path}: missing recorded manifest hash or model")
        manifest_hashes.add(row["manifest_sha256"])
        configurations.add((row["model"], row["sample_seconds"]))
        entries.append(row)
    if len(manifest_hashes) != 1:
        raise ValueError("all records must use identical manifest contents")
    if len(configurations) != 1:
        raise ValueError("all records must use the same model and sample duration")
    print("condition,run_id,seconds,output_tokens,tokens_per_second")
    for row in entries:
        print(f"{row['condition']},{row['run_id']},{row['completed_s']:.4f},"
              f"{row['total_output_tokens']},{row['request_batch_tokens_per_second']:.3f}")
        ready = next((s for s in row.get("snapshots", []) if s.get("phase") == "ready"), None)
        if ready is not None:
            per_rank = []
            for item in ready["ranks"]:
                state = item["snapshot"]
                lengths = sum(req["prompt_tokens"] for req in state["running"])
                progress = [req["output_tokens"] for req in state["running"]]
                per_rank.append(f"r{item['rank']}={lengths}({min(progress)}-{max(progress)})")
            print("  running prompt tokens (output progress range): " + ", ".join(per_rank))
    for condition in ("balanced", "skewed"):
        values = [r["completed_s"] for r in entries if r["condition"] == condition]
        if values:
            print(f"{condition}: n={len(values)}, median completion={statistics.median(values):.4f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--tokenizer", required=True)
    gen.add_argument("--output-dir", type=Path, required=True)
    gen.add_argument("--short-tokens", type=int, default=2048)
    gen.add_argument("--long-tokens", type=int, default=8192)
    gen.add_argument("--output-tokens", type=int, default=4096)
    gen.add_argument("--max-model-len", type=int, default=36848)
    gen.add_argument("--seed", type=int, default=2026)
    gen.add_argument("--trust-remote-code", action="store_true")
    run_p = sub.add_parser("run")
    run_p.add_argument("--manifest", type=Path, required=True)
    run_p.add_argument("--condition", choices=("balanced", "skewed"), required=True)
    run_p.add_argument("--mode", choices=("profile", "timing"), required=True)
    run_p.add_argument("--proxy-url", required=True)
    run_p.add_argument("--decoder-url", required=True)
    run_p.add_argument("--model", required=True)
    run_p.add_argument("--state-dir", type=Path, required=True)
    run_p.add_argument("--profile-dir", type=Path,
                       help="Optional local directory containing the decoder's *ascend_pt outputs")
    run_p.add_argument("--output-dir", type=Path, required=True)
    run_p.add_argument("--run-id", required=True)
    run_p.add_argument("--api-key", default="")
    run_p.add_argument("--sample-seconds", type=float, default=3)
    run_p.add_argument("--timeout", type=float, default=900)
    run_p.add_argument("--finish-timeout", type=float, default=7200)
    run_p.add_argument("--snapshot-max-age", type=float, default=10)
    rep = sub.add_parser("report")
    rep.add_argument("records", nargs="+", type=Path)
    args = parser.parse_args()
    if args.action == "generate":
        generate(args)
    elif args.action == "run":
        asyncio.run(run(args))
    else:
        report(args)


if __name__ == "__main__":
    main()

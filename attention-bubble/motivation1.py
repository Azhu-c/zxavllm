#!/usr/bin/env python3
"""Paired long/short-context workload for the Attention bubble experiment.

Generate once, then run the same requests with balanced or skewed DP placement.
The target is the rank-aware copy of DyntraLB's P/D proxy made by
prepare_reference.py. No NPU dependency is needed on the client machine.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any


WORDS = (
    "system model request token cache decode attention expert layer batch "
    "state memory compute network result method trace sample queue input "
    "output rank time data value block process update local global service"
).split()


def token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=True))


def make_prompt(tokenizer: Any, target: int, request_id: str, seed: int) -> tuple[str, int]:
    rng = random.Random(seed)
    prefix = f"Record {request_id}. Continue the text using clear, simple words.\n"
    unit = " ".join(rng.choice(WORDS) for _ in range(64)) + ".\n"
    if token_count(tokenizer, prefix) >= target:
        raise ValueError("target prompt length is too small")

    hi = 1
    while token_count(tokenizer, prefix + unit * hi) < target:
        hi *= 2
    lo = 0
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if token_count(tokenizer, prefix + unit * mid) <= target:
            lo = mid
        else:
            hi = mid - 1
    base = prefix + unit * lo

    hi = 1
    while token_count(tokenizer, base + " data" * hi) < target:
        hi *= 2
    lo = 0
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if token_count(tokenizer, base + " data" * mid) <= target:
            lo = mid
        else:
            hi = mid - 1
    prompt = base + " data" * lo
    actual = token_count(tokenizer, prompt)
    if abs(actual - target) > max(4, target // 100):
        raise RuntimeError(f"{request_id}: target={target}, actual={actual}")
    return prompt, actual


def placement(ranks: int, per_rank: int) -> list[dict[str, Any]]:
    if ranks < 2 or ranks % 2 or per_rank < 2 or per_rank % 2:
        raise ValueError("ranks and per-rank must be positive even numbers")
    half = ranks // 2
    per_kind = ranks * per_rank // 2
    rows: list[dict[str, Any]] = []
    for kind in ("long", "short"):
        for i in range(per_kind):
            balanced = i // (per_rank // 2)
            skewed = i // per_rank + (0 if kind == "long" else half)
            rows.append({
                "id": f"{kind[0].upper()}{i:03d}",
                "kind": kind,
                "rank_balanced": balanced,
                "rank_skewed": skewed,
            })
    for condition in ("balanced", "skewed"):
        counts = Counter(row[f"rank_{condition}"] for row in rows)
        if counts != Counter({rank: per_rank for rank in range(ranks)}):
            raise AssertionError(f"bad {condition} placement: {counts}")
    return rows


def generate(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    if args.short_tokens >= args.long_tokens:
        raise ValueError("short-tokens must be smaller than long-tokens")
    if args.long_tokens + args.output_tokens > args.max_model_len:
        raise ValueError("long prompt plus output exceeds max-model-len")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code
    )
    rows = placement(args.ranks, args.per_rank)
    for index, row in enumerate(rows):
        target = args.long_tokens if row["kind"] == "long" else args.short_tokens
        row["prompt"], row["prompt_tokens"] = make_prompt(
            tokenizer, target, row["id"], args.seed + index
        )
        row["output_tokens"] = args.output_tokens
        row["prompt_sha256"] = hashlib.sha256(row["prompt"].encode()).hexdigest()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "placement.jsonl"
    custom = args.output_dir / "mixed_prompts.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    custom.write_text(
        "".join(
            json.dumps({"prompt": row["prompt"], "output_tokens": row["output_tokens"]},
                       ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    summary = {
        "ranks": args.ranks,
        "per_rank": args.per_rank,
        "short_target_tokens": args.short_tokens,
        "long_target_tokens": args.long_tokens,
        "output_tokens": args.output_tokens,
        "max_model_len": args.max_model_len,
        "seed": args.seed,
        "tokenizer": args.tokenizer,
        "actual_prompt_tokens": {
            kind: sorted({row["prompt_tokens"] for row in rows if row["kind"] == kind})
            for kind in ("short", "long")
        },
    }
    (args.output_dir / "workload.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"manifest": str(manifest), "custom_dataset": str(custom),
                      **summary}, ensure_ascii=False, indent=2))


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("manifest is empty or has duplicate request IDs")
    for row in rows:
        if row["kind"] not in ("short", "long") or row["output_tokens"] <= 0:
            raise ValueError(f"invalid row: {row.get('id')}")
        if hashlib.sha256(row["prompt"].encode()).hexdigest() != row["prompt_sha256"]:
            raise ValueError(f"prompt hash mismatch: {row['id']}")
    ranks = sorted({row["rank_balanced"] for row in rows})
    if ranks != list(range(len(ranks))) or len(ranks) < 2 or len(ranks) % 2:
        raise ValueError("manifest must have a contiguous, even number of DP ranks")
    per_rank = len(rows) // len(ranks)
    if per_rank % 2 or len(rows) != per_rank * len(ranks):
        raise ValueError("manifest has an invalid per-rank request count")
    if len({row["output_tokens"] for row in rows}) != 1:
        raise ValueError("all requests must have the same output length")
    for rank in ranks:
        balanced = [row for row in rows if row["rank_balanced"] == rank]
        skewed = [row for row in rows if row["rank_skewed"] == rank]
        if len(balanced) != per_rank or len(skewed) != per_rank:
            raise ValueError(f"rank {rank} has unequal request counts")
        if Counter(row["kind"] for row in balanced) != {
            "long": per_rank // 2, "short": per_rank // 2
        }:
            raise ValueError(f"rank {rank} is not balanced")
        expected_kind = "long" if rank < len(ranks) // 2 else "short"
        if any(row["kind"] != expected_kind for row in skewed):
            raise ValueError(f"rank {rank} is not skewed as specified")
    return rows


async def run(args: argparse.Namespace) -> None:
    if args.profile_seconds <= 0 or args.settle_seconds < 0:
        raise ValueError("profile-seconds must be positive and settle-seconds nonnegative")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        raise ValueError("run-id may contain only letters, digits, '-' and '_'")
    rows = load_manifest(args.manifest)
    rank_key = f"rank_{args.condition}"
    expected = Counter(row[rank_key] for row in rows)
    print(f"{args.condition}: {len(rows)} requests; rank counts {dict(sorted(expected.items()))}")
    if args.dry_run:
        return
    if len(set(expected.values())) != 1:
        raise ValueError("unequal request counts across ranks")

    import httpx

    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / f"{args.condition}_{args.run_id}.json"
    if target.exists():
        raise FileExistsError(f"run record already exists: {target}")
    previous_profile_dirs = (
        {str(path.resolve()) for path in args.profile_dir.glob("*ascend_pt")}
        if args.profile_dir is not None and args.profile_dir.is_dir() else set()
    )
    summary: dict[str, Any] = {
        "condition": args.condition,
        "run_id": args.run_id,
        "manifest": str(args.manifest.resolve()),
        "requested_rank_counts": dict(sorted(expected.items())),
        "profile_seconds": args.profile_seconds,
        "profile_started": False,
        "profile_stopped": False,
        "valid_steady_window": False,
        "run_success": False,
        "results": [],
    }
    first_seen: set[str] = set()
    finished_seen: set[str] = set()
    first_barrier = asyncio.Event()
    launch = time.perf_counter()
    auth = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    limits = httpx.Limits(max_connections=max(64, len(rows) + 8),
                          max_keepalive_connections=max(64, len(rows) + 8))

    async with httpx.AsyncClient(timeout=None, limits=limits, trust_env=False) as client:
        async def request_one(row: dict[str, Any]) -> dict[str, Any]:
            rank = row[rank_key]
            result: dict[str, Any] = {
                "id": row["id"], "kind": row["kind"], "rank": rank,
                "prompt_tokens": row["prompt_tokens"],
                "requested_output_tokens": row["output_tokens"],
                "first_token_s": None, "finish_s": None,
                "completion_tokens": None, "finish_reason": None,
            }
            payload = {
                "model": args.model,
                "prompt": row["prompt"],
                "max_tokens": row["output_tokens"],
                "min_tokens": row["output_tokens"],
                "ignore_eos": True,
                "skip_special_tokens": False,
                "temperature": 0.0,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            headers = {**auth, "X-data-parallel-rank": str(rank),
                       "X-Experiment-Request-Id": row["id"]}
            try:
                async with client.stream("POST", args.proxy_url.rstrip("/") + "/v1/completions",
                                         json=payload, headers=headers) as response:
                    if response.status_code != 200:
                        body = (await response.aread())[:1000].decode(errors="replace")
                        raise RuntimeError(f"HTTP {response.status_code}: {body}")
                    ack = response.headers.get("X-Requested-DP-Rank")
                    if ack != str(rank):
                        raise RuntimeError(
                            f"proxy did not acknowledge rank {rank} (got {ack!r}); "
                            "run prepare_reference.py and use the ranked proxy"
                        )
                    result["proxy_request_id"] = response.headers.get("X-Proxy-Request-Id")
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            event = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        usage = event.get("usage") or {}
                        if usage.get("completion_tokens") is not None:
                            result["completion_tokens"] = usage["completion_tokens"]
                        for choice in event.get("choices") or []:
                            content = choice.get("text") or (choice.get("delta") or {}).get("content")
                            if content and result["first_token_s"] is None:
                                result["first_token_s"] = time.perf_counter() - launch
                                first_seen.add(row["id"])
                                if len(first_seen) == len(rows):
                                    first_barrier.set()
                            if choice.get("finish_reason") is not None:
                                result["finish_reason"] = choice["finish_reason"]
                                finished_seen.add(row["id"])
                result["finish_s"] = time.perf_counter() - launch
                if result["first_token_s"] is None:
                    raise RuntimeError("no generated token observed")
                if result["completion_tokens"] is not None and result["completion_tokens"] != row["output_tokens"]:
                    raise RuntimeError("output length differs from requested output_tokens")
                return result
            except Exception as exc:
                result["error"] = str(exc)
                result["finish_s"] = time.perf_counter() - launch
                return result

        tasks = [asyncio.create_task(request_one(row)) for row in rows]
        barrier_task = asyncio.create_task(first_barrier.wait())
        profiler_on = False
        try:
            pending = set(tasks)
            deadline = time.perf_counter() + args.first_token_timeout
            while not barrier_task.done():
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError("not all requests reached first decode token")
                done, _ = await asyncio.wait(pending | {barrier_task}, timeout=remaining,
                                             return_when=asyncio.FIRST_COMPLETED)
                if not done:
                    raise TimeoutError("not all requests reached first decode token")
                if first_barrier.is_set():
                    break
                finished = [task.result() for task in done if task is not barrier_task]
                raise RuntimeError(f"request ended before all first tokens: {finished[:2]}")
            await asyncio.sleep(args.settle_seconds)
            if finished_seen or any(task.done() for task in tasks):
                raise RuntimeError("a request finished before the profiling window")
            profile_url = args.decoder_url.rstrip("/")
            response = await client.post(profile_url + "/start_profile", headers=auth,
                                         timeout=30.0)
            response.raise_for_status()
            profiler_on = True
            summary["profile_started"] = True
            summary["profile_start_s"] = time.perf_counter() - launch
            await asyncio.sleep(args.profile_seconds)
            response = await client.post(profile_url + "/stop_profile", headers=auth,
                                         timeout=120.0)
            response.raise_for_status()
            profiler_on = False
            summary["profile_stopped"] = True
            summary["profile_stop_s"] = time.perf_counter() - launch
            if finished_seen or any(task.done() for task in tasks):
                raise RuntimeError("a request finished during the profiling window")
            summary["valid_steady_window"] = True
            summary["results"] = await asyncio.wait_for(
                asyncio.gather(*tasks), timeout=args.finish_timeout
            )
            if any("error" in result for result in summary["results"]):
                raise RuntimeError("one or more requests failed; see results")
            summary["run_success"] = True
        except Exception as exc:
            summary["error"] = str(exc)
            for task in tasks:
                task.cancel()
            summary["results"] = [task.result() for task in tasks
                                  if task.done() and not task.cancelled()
                                  and task.exception() is None]
            raise
        finally:
            if profiler_on:
                try:
                    response = await client.post(args.decoder_url.rstrip("/") + "/stop_profile",
                                                 headers=auth, timeout=120.0)
                    summary["profile_stopped"] = response.status_code == 200
                except Exception as exc:
                    summary["stop_profile_error"] = str(exc)
            barrier_task.cancel()
            await asyncio.gather(*tasks, barrier_task, return_exceptions=True)
            summary["elapsed_s"] = time.perf_counter() - launch
            if args.profile_dir is not None and args.profile_dir.is_dir():
                current = {str(path.resolve()) for path in args.profile_dir.glob("*ascend_pt")}
                summary["new_profile_dirs"] = sorted(current - previous_profile_dirs)
            target.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                              encoding="utf-8")
            print(f"run record: {target}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="Create one paired mixed-request manifest")
    gen.add_argument("--tokenizer", required=True, help="Local model/tokenizer path")
    gen.add_argument("--output-dir", type=Path, required=True)
    gen.add_argument("--ranks", type=int, default=8)
    gen.add_argument("--per-rank", type=int, default=4)
    gen.add_argument("--short-tokens", type=int, default=2048)
    gen.add_argument("--long-tokens", type=int, default=8192)
    gen.add_argument("--output-tokens", type=int, default=1024)
    gen.add_argument("--max-model-len", type=int, default=36848)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--trust-remote-code", action="store_true")

    run_p = sub.add_parser("run", help="Run one placement and profile Decode")
    run_p.add_argument("--manifest", type=Path, required=True)
    run_p.add_argument("--condition", choices=("balanced", "skewed"), required=True)
    run_p.add_argument("--proxy-url", required=True, help="Rank-aware P/D proxy URL")
    run_p.add_argument("--decoder-url", required=True, help="Decode server URL for profiler API")
    run_p.add_argument("--model", required=True, help="Served model name")
    run_p.add_argument("--output-dir", type=Path, required=True)
    run_p.add_argument("--run-id", default="r1", help="Unique ID, e.g. r1 or r2")
    run_p.add_argument("--profile-dir", type=Path,
                       help="Optional local path to the decoder profiler output directory")
    run_p.add_argument("--api-key", default="")
    run_p.add_argument("--first-token-timeout", type=float, default=600.0)
    run_p.add_argument("--finish-timeout", type=float, default=1800.0)
    run_p.add_argument("--settle-seconds", type=float, default=1.0)
    run_p.add_argument("--profile-seconds", type=float, default=3.0)
    run_p.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "generate":
        generate(args)
    else:
        asyncio.run(run(args))


if __name__ == "__main__":
    main()

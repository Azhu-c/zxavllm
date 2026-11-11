#!/usr/bin/env python3
"""Freeze custom JSONL requests and verify that every replay did the same work."""

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def is_integer(value, minimum=0):
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def load_manifest(path, count=None):
    path = Path(path).resolve()
    manifest = read_json(path)
    snapshot = path.parent / "requests.jsonl"
    if sha256_file(snapshot) != manifest["requests_sha256"]:
        raise ValueError("固定请求副本已被修改；停止运行，避免模式之间工作量不同。")
    if count is not None and manifest["num_requests"] != count:
        raise ValueError("请求数与固定清单不一致。")
    return manifest


def prepare(args):
    if args.count < 1:
        raise ValueError("请求数必须是正整数。")
    source = Path(args.source).resolve()
    directory = Path(args.output_dir).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("固定请求输出目录已存在且非空；请使用新的运行目录。")
    selected = []
    records = []
    with source.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except ValueError as exc:
                raise ValueError("数据集第 %d 行不是有效 JSON。" % line_number) from exc
            if not isinstance(item, dict) or not isinstance(item.get("prompt"), str):
                raise ValueError("第 %d 行必须包含字符串 prompt。" % line_number)
            if not item["prompt"].strip():
                raise ValueError("第 %d 行 prompt 为空。" % line_number)
            if not is_integer(item.get("output_tokens"), 1):
                raise ValueError("第 %d 行必须包含正整数 output_tokens，以固定输出长度。" % line_number)
            # Only these two fields are consumed by vLLM's custom dataset.
            request = {"prompt": item["prompt"], "output_tokens": item["output_tokens"]}
            selected.append(json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n")
            records.append({
                "request_index": len(records),
                "source_line": line_number,
                "prompt_sha256": hashlib.sha256(item["prompt"].encode("utf-8")).hexdigest(),
                "prompt_characters": len(item["prompt"]),
                "output_tokens": item["output_tokens"],
            })
            if len(records) == args.count:
                break
    if len(records) != args.count:
        raise ValueError("数据集仅有 %d 条请求，需要 %d 条；不会复制请求补足。" % (len(records), args.count))
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = directory / "requests.jsonl"
    snapshot.write_text("".join(selected), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "source": str(source),
        "selection": "first N nonempty JSONL rows, original order",
        "requests_sha256": sha256_file(snapshot),
        "num_requests": len(records),
        "expected_total_output_tokens_per_round": sum(r["output_tokens"] for r in records),
        "ignore_eos": True,
        "requests": records,
    }
    save_json(directory / "workload.json", manifest)
    print("固定 %d 条请求；每轮目标输出 %d tokens；SHA256=%s" % (
        len(records), manifest["expected_total_output_tokens_per_round"], manifest["requests_sha256"]))


def verify(args):
    manifest_path = Path(args.manifest).resolve()
    manifest = load_manifest(manifest_path)
    result = read_json(args.result)
    expected = [r["output_tokens"] for r in manifest["requests"]]
    outputs = result.get("output_lens")
    inputs = result.get("input_lens")
    issues = []
    mismatches = []
    if result.get("completed") != len(expected):
        issues.append("成功请求数不等于固定请求数")
    if result.get("failed", 0) != 0:
        issues.append("存在失败请求")
    if any(result.get("errors") or []):
        issues.append("逐请求结果中存在错误")
    if not isinstance(outputs, list) or len(outputs) != len(expected):
        issues.append("逐请求 output_lens 缺失或数量不匹配")
    else:
        mismatches = [i for i, (actual, target) in enumerate(zip(outputs, expected))
                      if not is_integer(actual) or actual != target]
        if mismatches:
            issues.append("%d 条请求的实际输出长度不符合目标" % len(mismatches))
    expected_total = sum(expected)
    if result.get("total_output_tokens") != expected_total:
        issues.append("实际输出 token 总数不符合固定清单")
    if not isinstance(inputs, list) or len(inputs) != len(expected) or not all(is_integer(n, 1) for n in inputs):
        issues.append("逐请求 input_lens 缺失或无效")
    else:
        reference = manifest_path.parent / "reference_input_lens.json"
        if reference.exists():
            if read_json(reference)["input_lens"] != inputs:
                issues.append("逐请求输入 token 长度与首个有效运行不一致")
        elif not issues:
            save_json(reference, {"mode": args.mode, "round": args.round, "input_lens": inputs})
    summary = Path(args.summary_dir)
    summary.mkdir(parents=True, exist_ok=True)
    validation = {
        "mode": args.mode, "round": args.round, "valid": not issues,
        "requests_sha256": manifest["requests_sha256"],
        "expected_requests": len(expected), "completed": result.get("completed"),
        "failed": result.get("failed"), "expected_output_tokens": expected_total,
        "actual_output_tokens": result.get("total_output_tokens"),
        "mismatched_request_indices": mismatches,
        "issues": issues, "benchmark_result": str(Path(args.result).resolve()),
    }
    save_json(summary / ("%s_round%d.validation.json" % (args.mode, args.round)), validation)
    csv_path = summary / "workload_comparison.csv"
    columns = ["mode", "round", "valid", "requests_sha256", "expected_requests", "completed", "failed",
               "expected_output_tokens", "actual_output_tokens", "duration_seconds", "output_tokens_per_second"]
    row = {key: validation.get(key) for key in columns}
    row.update(duration_seconds=result.get("duration"), output_tokens_per_second=result.get("output_throughput"))
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    if issues:
        raise ValueError("工作量核对失败：" + "；".join(issues))
    print("工作量核对通过：%s 第 %d 轮，%d 条请求，实际输出 %d tokens。" % (
        args.mode, args.round, len(expected), expected_total))


def event(args):
    directory = Path(args.directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = read_json(args.manifest)
    data = {
        "event": args.event, "mode": args.mode,
        "unix_time_ns": time.time_ns(),
        "utc": datetime.now(timezone.utc).isoformat(),
        "requests_sha256": manifest["requests_sha256"],
        "workload_manifest": str(Path(args.manifest).resolve()),
    }
    with (directory / "capture_events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(data, ensure_ascii=False) + "\n")


def configure(args):
    manifest = load_manifest(args.manifest)
    save_json(Path(args.directory) / "run_config.json", {
        "modes": args.modes.split(","), "rounds_per_mode": args.rounds,
        "requests_per_round": manifest["num_requests"],
        "expected_requests_per_mode": manifest["num_requests"] * args.rounds,
        "expected_output_tokens_per_mode": manifest["expected_total_output_tokens_per_round"] * args.rounds,
        "requests_sha256": manifest["requests_sha256"], "benchmark_seed": args.seed,
        "max_concurrency": args.concurrency, "request_rate": args.rate,
        "ignore_eos": True, "disable_shuffle": True, "warmup_requests": 0,
        "ready_check_generation_requests": 0, "profile_enabled": args.profile == "1",
        "profile_scope": "all benchmark rounds, no time cutoff",
        "decoder_profile_url": args.decoder_url,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("prepare")
    freeze.add_argument("--source", required=True)
    freeze.add_argument("--count", type=int, required=True)
    freeze.add_argument("--output-dir", required=True)
    check = commands.add_parser("check")
    check.add_argument("--manifest", required=True)
    check.add_argument("--count", type=int, required=True)
    check.add_argument("--dataset")
    validate = commands.add_parser("verify")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--result", required=True)
    validate.add_argument("--mode", required=True)
    validate.add_argument("--round", type=int, required=True)
    validate.add_argument("--summary-dir", required=True)
    stamp = commands.add_parser("event")
    stamp.add_argument("--directory", required=True)
    stamp.add_argument("--event", required=True)
    stamp.add_argument("--mode", required=True)
    stamp.add_argument("--manifest", required=True)
    config = commands.add_parser("configure")
    config.add_argument("--directory", required=True)
    config.add_argument("--manifest", required=True)
    config.add_argument("--modes", required=True)
    config.add_argument("--rounds", type=int, required=True)
    config.add_argument("--seed", type=int, required=True)
    config.add_argument("--concurrency", type=int, required=True)
    config.add_argument("--rate", required=True)
    config.add_argument("--profile", required=True)
    config.add_argument("--decoder-url", required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            prepare(args)
        elif args.command == "check":
            manifest = load_manifest(args.manifest, args.count)
            if args.dataset and sha256_file(args.dataset) != manifest["requests_sha256"]:
                raise ValueError("压测读取的数据集与固定请求清单不一致。")
        elif args.command == "verify":
            verify(args)
        elif args.command == "configure":
            configure(args)
        else:
            event(args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print("错误：%s" % exc, file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())

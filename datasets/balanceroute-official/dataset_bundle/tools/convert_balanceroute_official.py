"""Convert official, frozen BalanceRoute lengths to a Qwen3 chat workload.

Lengths are unmodified author data. Prompt text is newly synthesized here.
Run from the project root with dataset_bundle/.venv/bin/python.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
from jinja2 import Environment, StrictUndefined
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "fdd2a6971ca6222494b1335cc67aaae8e4e28b08"
NPZ_SHA256 = "383a8e89fde24998590518db4d44a872f8e36a7e9b5e617f62c49ef39629e484"
LENGTHS_SHA256 = "e5eca3ccd0b7abbd03b007c3864d38bff04c241f950779c97e2aded0694b955e"
WORDS = ("red blue green sky sun moon tree leaf road path lake river stone sand "
         "book page pen desk chair room door wall floor light day night time "
         "data model token cache queue input output value state step load").split()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_digest(prompt, output):
    return hashlib.sha256(np.stack([prompt, output]).astype("<i8").tobytes()).hexdigest()


def summary(values):
    values = np.asarray(values, dtype=np.int64)
    return {"min": int(values.min()), "max": int(values.max()),
            "mean": float(values.mean()), "sum": int(values.sum()),
            "p50": float(np.quantile(values, 0.5)),
            "p95": float(np.quantile(values, 0.95)),
            "p99": float(np.quantile(values, 0.99))}


def load_tokenizer(directory):
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    config = json.loads((directory / "tokenizer_config.json").read_text(encoding="utf-8"))
    environment = Environment(undefined=StrictUndefined)
    template = environment.from_string(config["chat_template"])

    def render(text):
        return template.render(messages=[{"role": "user", "content": text}],
                               tools=None, add_generation_prompt=True)

    def count(text):
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    overhead = count(render("hello")) - count("hello")
    if overhead != 8:
        raise ValueError(f"Unexpected chat template overhead: {overhead}; expected 8")
    return tokenizer, render, count, overhead


def make_prompt(index, target, count, words):
    # Deterministic distinct prefixes avoid sharing the first KV-cache block.
    identifier = hashlib.sha256(f"balanceroute-seed100-row{index}".encode("ascii")).hexdigest()[:32]
    prefix = identifier + ". Continue writing in English using these reference words:"
    rng = random.Random(202611270000 + index)
    budget = target - count(prefix)
    if budget < 0:
        raise ValueError(f"Target too short at row {index}")
    body = [" " + rng.choice(words) for _ in range(budget)]
    prompt = prefix + "".join(body)
    actual = count(prompt)
    # Retokenization, rather than a word-count estimate, determines the length.
    while actual > target and body:
        del body[-max(1, actual - target):]
        prompt = prefix + "".join(body)
        actual = count(prompt)
    if actual < target:
        prompt += " x" * (target - actual)
    if count(prompt) != target:
        raise ValueError(f"Cannot fit exact prompt length at row {index}")
    return prompt


def convert(source, tokenizer_dir, destination, metadata_path):
    if sha256(source) != NPZ_SHA256:
        raise ValueError("Official seed 100 NPZ checksum mismatch")
    with np.load(source, allow_pickle=False) as data:
        inputs, outputs = data["s"].copy(), data["o"].copy()
    if inputs.shape != (8000,) or outputs.shape != (8000,):
        raise ValueError("Expected exactly 8000 requests")
    if input_digest(inputs, outputs) != LENGTHS_SHA256:
        raise ValueError("Official ordered length arrays differ from reference")
    tokenizer, render, count, overhead = load_tokenizer(tokenizer_dir)
    words = [word for word in WORDS if count(" " + word) == 1]
    if len(words) < 16:
        raise ValueError("Tokenizer has too few compatible synthetic words")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    first_blocks = set()
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for index, (input_length, output_length) in enumerate(zip(inputs, outputs), 1):
                prompt = make_prompt(index, int(input_length) - overhead, count, words)
                wrapped = tokenizer.encode(render(prompt), add_special_tokens=False).ids
                if len(wrapped) != input_length:
                    raise ValueError(f"Chat length mismatch at row {index}")
                first_block = tuple(wrapped[:16])
                if first_block in first_blocks:
                    raise ValueError(f"Duplicate first 16-token block at row {index}")
                first_blocks.add(first_block)
                handle.write(json.dumps({"prompt": prompt, "output_tokens": int(output_length)},
                                        ensure_ascii=False, separators=(",", ":")) + "\n")
                if index % 1000 == 0:
                    print(f"Converted and token-checked {index}/8000", flush=True)
        # Re-read every saved line and verify ordered outputs/schema independently.
        with temporary.open(encoding="utf-8") as handle:
            count_rows = 0
            for count_rows, line in enumerate(handle, 1):
                row = json.loads(line)
                if set(row) != {"prompt", "output_tokens"} or not isinstance(row["prompt"], str):
                    raise ValueError(f"Invalid schema at line {count_rows}")
                if type(row["output_tokens"]) is not int or row["output_tokens"] != int(outputs[count_rows-1]):
                    raise ValueError(f"Output length/order mismatch at line {count_rows}")
                if "<|im_start|>" in row["prompt"] or "\ufffd" in row["prompt"]:
                    raise ValueError(f"Unexpected special/replacement token at line {count_rows}")
        if count_rows != len(inputs):
            raise ValueError("Saved line count mismatch")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    tokenizer_source = json.loads((tokenizer_dir / "source.json").read_text(encoding="utf-8"))
    metadata = {
        "name": "BalanceRoute official synthetic seed 100; Qwen3 chat adaptation",
        "created_date": "2026-11-27",
        "records": len(inputs), "seed": 100,
        "source_type": "official frozen synthetic integer lengths; not proprietary production data",
        "source_commit": COMMIT,
        "source_url": f"https://github.com/b010001y/BalanceRoute/blob/{COMMIT}/src/balanceroute/reference/r16/inputs/100.npz",
        "source_npz_sha256": sha256(source),
        "official_ordered_lengths_sha256": input_digest(inputs, outputs),
        "paper_url": "https://arxiv.org/html/2605.06113v3#S6.SS1",
        "prompt_text": "new deterministic synthetic English words; authors publish lengths only",
        "length_policy": "preserve all original s/o pairs and order; s counts the complete single-user Qwen3 chat prompt",
        "raw_content_tokens": summary(inputs - overhead),
        "chat_template_overhead_tokens": overhead,
        "input_tokens_including_chat_template": summary(inputs),
        "output_tokens": summary(outputs),
        "max_input_plus_output_tokens": int((inputs + outputs).max()),
        "arrival_policy": "no timestamps supplied by authors; request-rate controls replay",
        "tokenizer": {**tokenizer_source, "tokenizer_json_sha256": sha256(tokenizer_dir / "tokenizer.json"),
                      "tokenizer_config_sha256": sha256(tokenizer_dir / "tokenizer_config.json")},
        "required_cli": {"backend": "openai-chat", "dataset_name": "custom", "custom_output_len": -1,
                         "skip_chat_template": True, "disable_shuffle": True, "ignore_eos": True},
        "validation": {"all_8000_schema_and_output_order_checked": True,
                       "all_8000_chat_input_token_counts_checked": True,
                       "distinct_first_16_token_blocks": len(first_blocks),
                       "npu_serving_executed": False},
        "jsonl_file": destination.name, "jsonl_bytes": destination.stat().st_size,
        "jsonl_sha256": sha256(destination),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_temporary = metadata_path.with_name(metadata_path.name + ".part")
    metadata_temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(metadata_temporary, metadata_path)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "raw/balanceroute_official/100.npz")
    parser.add_argument("--tokenizer-dir", type=Path, default=ROOT / "raw/balanceroute_official/tokenizer")
    parser.add_argument("--output", type=Path, default=ROOT / "jsonl/balanceroute_official_seed100_8000_qwen3_chat.jsonl")
    parser.add_argument("--metadata", type=Path, default=ROOT / "metadata/balanceroute_official_seed100.json")
    args = parser.parse_args()
    convert(args.source, args.tokenizer_dir, args.output, args.metadata)

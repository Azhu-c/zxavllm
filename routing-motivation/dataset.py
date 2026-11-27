"""Deterministic synthetic length-controlled workloads; generated offline."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import random

WORDS = ("system model request token cache attention layer memory compute network "
         "result method trace sample queue input output time data value block process "
         "update local global service river tree stone morning window book garden").split()


def specifications():
    rows = []
    for input_name, input_tokens in (("S", 2048), ("L", 8192)):
        for output_name, output_tokens in (("S", 512), ("L", 2048)):
            for i in range(24):
                rows.append({"id": f"{input_name}{output_name}-{i:02d}",
                             "class": f"{input_name}{output_name}",
                             "target_input_tokens": input_tokens,
                             "output_tokens": output_tokens})
    random.Random(20261126).shuffle(rows)
    for i, row in enumerate(rows):
        row["arrival_index"] = i
        row["seed"] = 20261126 + i
    return rows


def prompt_for(tokenizer, row):
    rng = random.Random(row["seed"])
    # A different body per request prevents a shared repeated prefix across requests.
    prefix = f"Record {row['id']}. Continue this record in simple English.\n"
    unit = " ".join(rng.choice(WORDS) for _ in range(128)) + ".\n"
    target = row["target_input_tokens"]
    count = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    hi = 1
    while count(prefix + unit * hi) < target:
        hi *= 2
    lo = 0
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(prefix + unit * mid) <= target:
            lo = mid
        else:
            hi = mid - 1
    base = prefix + unit * lo
    lo, hi = 0, target
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(base + " data" * mid) <= target:
            lo = mid
        else:
            hi = mid - 1
    text = base + " data" * lo
    actual = count(text)
    if not 0 <= target - actual <= 4:
        raise RuntimeError(f"Prompt length mismatch: {row['id']}: {actual}/{target}")
    return {**row, "prompt": text, "input_tokens": actual}


def generate(root, model_path):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True,
                                              trust_remote_code=True)
    rows = [prompt_for(tokenizer, row) for row in specifications()]
    directory = Path(root) / "data"
    directory.mkdir(exist_ok=True)
    manifest = directory / "mixed96.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    validation = []
    for kind in ("SS", "SL", "LS", "LL"):
        validation.extend({**r, "output_tokens": 32} for r in rows if r["class"] == kind
                          and int(r["id"].split("-")[1]) < 4)
    (directory / "validate16.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in validation), encoding="utf-8")
    metadata = {"seed": 20261126, "count": len(rows), "classes": {k: 24 for k in
                ("SS", "SL", "LS", "LL")}, "model_path": model_path,
                "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "meaning": "synthetic length-controlled observation workload; no accuracy labels",
                "input_length": "completion prompt, add_special_tokens=False; server usage also retained"}
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata

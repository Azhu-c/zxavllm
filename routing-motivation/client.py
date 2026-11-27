"""Streaming completions client using only the Python standard library."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import threading
import time
import urllib.request


def http(url, body=None, timeout=30):
    headers = {"Content-Type": "application/json"}
    if os.environ.get("OPENAI_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["OPENAI_API_KEY"]
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers=headers)
    # Internal serving endpoints must not pass through the workstation HTTP proxy.
    return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=timeout)


def one(row, url, model, origin, scheduled, timeout):
    time.sleep(max(0, origin + scheduled - time.perf_counter()))
    start = time.perf_counter()
    result = {"id": row["id"], "class": row["class"],
              "input_tokens": row["input_tokens"], "requested_output_tokens": row["output_tokens"],
              "scheduled_s": scheduled, "dispatch_s": start - origin,
              "dispatch_wall_ns": time.time_ns(), "content_chunk_s": [], "error": None}
    payload = {"model": model, "prompt": row["prompt"], "max_tokens": row["output_tokens"],
               "temperature": 0.0, "ignore_eos": True, "skip_special_tokens": False,
               "stream": True, "stream_options": {"include_usage": True}}
    texts = []
    usage = {}
    try:
        with http(url.rstrip("/") + "/v1/completions", payload, timeout) as response:
            result["response_request_id"] = response.headers.get("X-Request-Id")
            for line in response:
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    result["received_done"] = True
                    break
                event = json.loads(data)
                if "error" in event:
                    raise RuntimeError(str(event["error"]))
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    content = choice.get("text") or (choice.get("delta") or {}).get("content")
                    if content:
                        texts.append(content)
                        result["content_chunk_s"].append(time.perf_counter() - origin)
                    if choice.get("finish_reason") is not None:
                        result["finish_reason"] = choice["finish_reason"]
        if not result["content_chunk_s"]:
            raise RuntimeError("No generated content received")
        if not result.get("received_done") or not result.get("finish_reason"):
            raise RuntimeError("Truncated stream: missing DONE or finish_reason")
    except Exception as error:
        result["error"] = repr(error)
    result["finish_s"] = time.perf_counter() - origin
    result["finish_wall_ns"] = time.time_ns()
    result["usage"] = usage
    result["text"] = "".join(texts)
    return result


def benchmark(rows, cfg, output, tokenizer, profile=False):
    output.mkdir(parents=True, exist_ok=True)
    origin = time.perf_counter() + 0.5
    start_wall_ns = time.time_ns() + 500_000_000
    profile_error = []
    finished = threading.Event()
    def profile_window():
        start = origin + cfg["profile_delay_seconds"]
        if finished.wait(max(0, start - time.perf_counter())):
            profile_error.append("Workload ended before profiler start")
            return
        base = f"http://{cfg['host_ip']}:{cfg['decoder_port']}"
        started = False
        try:
            with http(base + "/start_profile", {}, 120):
                started = True
            if finished.wait(cfg["profile_seconds"]):
                profile_error.append("Workload ended during profiler window")
        except Exception as error:
            profile_error.append(repr(error))
        finally:
            if started:
                try:
                    with http(base + "/stop_profile", {}, 180):
                        pass
                except Exception as error:
                    profile_error.append(repr(error))
    thread = None
    if profile:
        thread = threading.Thread(target=profile_window, daemon=True)
        thread.start()
    results = []
    url = f"http://{cfg['host_ip']}:{cfg['proxy_port']}"
    rate = float(cfg["request_rate"])
    with (output / "requests.jsonl").open("w", encoding="utf-8", buffering=1) as handle:
        with ThreadPoolExecutor(max_workers=cfg["max_concurrency"]) as pool:
            jobs = [pool.submit(one, row, url, cfg["served_model"], origin,
                                0 if rate == 0 else i / rate,
                                cfg["request_timeout_seconds"]) for i, row in enumerate(rows)]
            for future in as_completed(jobs):
                result = future.result()
                text = result.pop("text")
                tokens = result["usage"].get("completion_tokens")
                if tokens is None:
                    tokens = len(tokenizer.encode(text, add_special_tokens=False))
                    result["token_count_source"] = "retokenized_text"
                else:
                    result["token_count_source"] = "server_usage"
                result["output_tokens"] = tokens
                if tokens != result["requested_output_tokens"]:
                    result["error"] = result["error"] or f"Output length mismatch: {tokens}"
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                results.append(result)
    finished.set()
    if thread:
        thread.join(timeout=310)
        if thread.is_alive():
            profile_error.append("Profiler control did not finish")
    meta = {"measurement": "profile" if profile else "performance",
            "window_start_wall_ns": start_wall_ns,
            "window_end_wall_ns": max(r["finish_wall_ns"] for r in results),
            "seconds": max(r["finish_s"] for r in results),
            "count": len(rows), "errors": sum(r["error"] is not None for r in results),
            "profile_errors": profile_error}
    (output / "measurement.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta

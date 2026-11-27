#!/usr/bin/env python3
"""One-terminal RR / JSQ / KV-aware motivation experiment."""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def config():
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    cfg["scripts_dir"] = str((ROOT / cfg["scripts_dir"]).resolve())
    cfg["model_path"] = str(Path(cfg["model_path"]).expanduser().resolve())
    return cfg


def transform_script(source, role, cfg):
    # Apply to experiment copies only. Preserve model/resource/scheduler options.
    source = re.sub(r"(?m)^\s*cleanup_stale_vllm_shm\s*(?:\|\|\s*exit\s+1)?\s*$",
                    "# Experiment copy: shared-memory cleanup is disabled.", source)
    source = re.sub(r"(?m)^\s*(?:sudo\s+)?(?:pkill|killall)\b[^\n]*$",
                    "# Experiment copy: global process cleanup is disabled.", source)
    if re.search(r"\b(?:pkill|killall)\b|xargs[^\n]*\bkill\b", "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith("#"))):
        raise RuntimeError("Unrecognized process cleanup in startup script; refusing to execute it")
    # Honor the small experiment config even when old scripts hard-code these variables.
    values = {"MODEL": cfg["model_path"], "MODEL_PATH": cfg["model_path"],
              "HOST_IP": cfg["host_ip"], "SERVED_MODEL": cfg["served_model"],
              "SERVED_MODEL_NAME": cfg["served_model"]}
    for name, value in values.items():
        source = re.sub(rf"(?m)^(export\s+)?{name}=[^\n]*$",
                        lambda m: (m.group(1) or "") + name + "=" + shlex.quote(value), source)
    # Prevent startup-script logs from sharing the user's existing experiment log path.
    source = re.sub(r"(?m)^(export\s+)?LOG_DIR=[^\n]*$",
                    'LOG_DIR="${MOTIVATION_LOG_DIR:?}"', source)
    source = source.replace("61.28.30.29", cfg["host_ip"])
    if role == "proxy":
        # The inspected proxy points to one P endpoint and one internal-DP D endpoint.
        for flag, value in (("host", cfg["host_ip"]), ("prefiller-hosts", cfg["host_ip"]),
                            ("decoder-hosts", cfg["host_ip"]),
                            ("port", cfg.get("proxy_port", 8080)),
                            ("prefiller-ports?", cfg.get("prefiller_port", 13700)),
                            ("decoder-ports?", cfg.get("decoder_port", 13701))):
            source = re.sub(rf"(--{flag}\s+)(?:\"[^\"]*\"|'[^']*'|[^\s\\]+)",
                            lambda m: m.group(1) + shlex.quote(str(value)), source)
    if role != "proxy":
        command = "python3 " + shlex.quote(str(ROOT / "serve_entry.py")) + " " + role + " serve"
        source, count = re.subn(r"(?m)^(\s*(?:exec\s+)?)vllm\s+serve\b",
                                lambda match: match.group(1) + command, source)
        if count != 1:
            raise RuntimeError(f"{role}: expected exactly one 'vllm serve', found {count}")
    return source


def check_runtime(cfg):
    executable = shutil.which("vllm")
    if not executable:
        raise RuntimeError("Run in the same Python environment that you normally use to start vLLM")
    spec = importlib.util.find_spec("vllm")
    ascend = importlib.util.find_spec("vllm_ascend")
    if not spec or not spec.origin or not ascend or not ascend.origin:
        raise RuntimeError("vllm and vllm_ascend must be importable in the current environment")
    vllm_root = Path(spec.origin).resolve().parent.parent
    ascend_root = Path(ascend.origin).resolve().parent.parent
    client = vllm_root / "vllm/v1/engine/core_client.py"
    tree = ast.parse(client.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DPLBAsyncMPClient")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "get_core_engine_for_request")
    text = ast.get_source_segment(client.read_text(), method)
    compatible = all(token in text for token in ("data_parallel_rank", "reqs_in_flight", "engine_inflight"))
    try:
        version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        version = "source checkout"
    report = {"vllm_executable": executable, "vllm_version": version,
              "vllm_root": str(vllm_root), "ascend_root": str(ascend_root),
              "core_client_sha256": hashlib.sha256(client.read_bytes()).hexdigest(),
              "native_selector": text, "routing_hook_compatible": compatible}
    dump(ROOT / "runtime-check.json", report)
    if not compatible:
        raise RuntimeError("This checkout differs from the inspected DP router. Details: runtime-check.json")
    return report


def prepare(cfg):
    if not (Path(cfg["model_path"]) / "config.json").is_file():
        raise RuntimeError("Fill model_path in config.json with the model directory containing config.json")
    environment = check_runtime(cfg)
    target = ROOT / "prepared"
    target.mkdir(exist_ok=True)
    provenance = {}
    for role, key in (("prefiller", "prefiller_script"), ("decoder", "decoder_script"), ("proxy", "proxy_script")):
        source_path = Path(cfg["scripts_dir"]) / cfg[key]
        source = source_path.read_text(encoding="utf-8")
        result = transform_script(source, role, cfg)
        dest = target / (role + ".sh")
        dest.write_text(result, encoding="utf-8")
        subprocess.run(["bash", "-n", str(dest)], check=True)
        provenance[role] = {"source": str(source_path),
                            "sha256": hashlib.sha256(source.encode()).hexdigest()}
    dump(target / "environment.json", environment)
    dump(target / "sources.json", provenance)
    from dataset import generate
    metadata = generate(ROOT, cfg["model_path"])
    print(f"已准备：{metadata['count']} 条正式请求，16 条连通性验证请求。", flush=True)
    print("原启动脚本和 vLLM 源码保持原样；实验副本在 prepared/。", flush=True)
    return environment


def port_open(host, port):
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


class Services:
    def __init__(self, cfg):
        self.cfg, self.children = cfg, []

    def start(self, role, run, policy, profile=False):
        log = run / (role + ".log")
        env = os.environ.copy()
        environment = json.loads((ROOT / "prepared/environment.json").read_text())
        env.update({"MODEL": self.cfg["model_path"], "MODEL_PATH": self.cfg["model_path"],
                    "HOST_IP": self.cfg["host_ip"], "SERVED_MODEL": self.cfg["served_model"],
                    "VLLM_ROOT": environment["vllm_root"], "ASCEND_ROOT": environment["ascend_root"],
                    "MOTIVATION_RUN": str(run), "MOTIVATION_DATA": str(run / "observations"),
                    "MOTIVATION_POLICY": policy, "MOTIVATION_ROLE": role,
                    "MOTIVATION_REPORT_MS": str(self.cfg["report_interval_ms"]),
                    "MOTIVATION_PROFILE": "1" if profile else "0",
                    "MOTIVATION_PROFILE_DIR": str(run / "profiler"),
                    "MOTIVATION_LOG_DIR": str(run / "service-logs"),
                    "LOG_DIR": str(run / "service-logs"),
                    "RECOMPUTE_SCHEDULER_ENABLE": "false"})
        (run / "service-logs").mkdir(exist_ok=True)
        with log.open("wb") as handle:
            process = subprocess.Popen(["bash", str(ROOT / "prepared" / (role + ".sh"))],
                                       cwd=self.cfg["scripts_dir"], env=env,
                                       stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        self.children.append(process)
        port = self.cfg[{"proxy": "proxy_port", "prefiller": "prefiller_port", "decoder": "decoder_port"}[role]]
        deadline = time.monotonic() + self.cfg["startup_timeout_seconds"]
        print(f"正在启动 {role}，日志：{log}", flush=True)
        from client import http
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"{role} exited: {log.read_text(errors='replace')[-3000:]}")
            if port_open(self.cfg["host_ip"], port):
                if role == "proxy":
                    return process
                try:
                    with http(f"http://{self.cfg['host_ip']}:{port}/health", timeout=2):
                        return process
                except Exception:
                    pass
            time.sleep(1)
        raise RuntimeError(f"{role} startup timed out; see {log}")

    def stop(self, process):
        if process not in self.children:
            return
        # Address the process group created by this task, including piped shell children.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            self.children.remove(process)
            process.wait()
            return
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            pass
        # The shell may exit before its NPU children; close any remaining group members.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        self.children.remove(process)

    def close(self):
        for process in reversed(list(self.children)):
            self.stop(process)


def wait_drained(directory, timeout=90):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        files = list(directory.glob("state-*.jsonl"))
        for file in files:
            lines = file.read_text().splitlines()
            if not lines:
                continue
            try:
                row = json.loads(lines[-1])
            except json.JSONDecodeError:
                continue
            if time.time_ns() - row["wall_ns"] < 5_000_000_000 and not any(row["owned_blocks_with_pending"]):
                return
        time.sleep(0.2)
    raise RuntimeError("Decoder did not drain, or observation reports stopped")


def export_profiler(session):
    directories = sorted(session.glob("*/profiler/**/*ascend_pt"))
    pending = [path for path in directories if not (path / "ASCEND_PROFILER_OUTPUT/trace_view.json").exists()]
    if not pending:
        return
    # Kept in a separate Python process: do not initialize torch/NPU in the
    # process that will later launch the next Decoder via multiprocessing.
    code = ("import sys; from torch_npu.profiler.profiler import analyse; "
            "[analyse(path) for path in sys.argv[1:]]")
    log = session / "profiler-export.log"
    print(f"正在解析 {len(pending)} 个 profiler 目录，日志：{log}", flush=True)
    with log.open("w") as handle:
        result = subprocess.run([sys.executable, "-c", code, *map(str, pending)],
                                stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        print(f"Profiler 导出未完成，原始文件保留；查看 {log}。", flush=True)


def execute(args, cfg):
    from transformers import AutoTokenizer
    from client import benchmark
    environment = prepare(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], local_files_only=True, trust_remote_code=True)
    profile = args.command == "profile"
    tag = args.tag or time.strftime("%Y%m%d-%H%M%S")
    session = ROOT / "results" / tag
    session.mkdir(parents=True, exist_ok=False)
    dump(session / "config.json", cfg)
    shutil.copy2(ROOT / "runtime-check.json", session)
    shutil.copy2(ROOT / "data/metadata.json", session / "dataset-metadata.json")
    dataset_file = "validate16.jsonl" if args.command == "validate" else "mixed96.jsonl"
    shutil.copy2(ROOT / "data" / dataset_file, session / "requests-input.jsonl")
    for port in (cfg["proxy_port"], cfg["prefiller_port"], cfg["decoder_port"]):
        if port_open(cfg["host_ip"], port):
            raise RuntimeError(f"Port {port} is already in use. Existing services were left running.")
    services = Services(cfg)
    shared = session / "shared"
    shared.mkdir()
    try:
        services.start("prefiller", shared, "rr")
        services.start("proxy", shared, "rr")
        filename = "validate16.jsonl" if args.command == "validate" else "mixed96.jsonl"
        rows = [json.loads(line) for line in (ROOT / "data" / filename).read_text().splitlines()]
        policies = args.policies.split(",")
        if any(p not in ("rr", "jsq", "kv") for p in policies) or len(set(policies)) != len(policies):
            raise ValueError("--policies accepts rr,jsq,kv")
        rounds = 1 if profile or args.command == "validate" else cfg["repeats"]
        for repeat in range(rounds):
            # Rotate execution order to reduce a fixed first/last-policy bias.
            order = policies[repeat % len(policies):] + policies[:repeat % len(policies)]
            for policy in order:
                run = session / f"{policy}-r{repeat + 1}"
                run.mkdir()
                process = services.start("decoder", run, policy, profile)
                try:
                    warm = [{**rows[i], "id": "warm-" + str(i), "output_tokens": 16}
                            for i in range(min(8, len(rows)))]
                    warm_cfg = {**cfg, "request_rate": 0}
                    meta = benchmark(warm, warm_cfg, run / "warmup", tokenizer)
                    if meta["errors"]:
                        raise RuntimeError(f"Warmup failed: {run / 'warmup/requests.jsonl'}")
                    wait_drained(run / "observations")
                    print(f"开始 {policy.upper()}，第 {repeat + 1} 轮，{len(rows)} 条请求。", flush=True)
                    meta = benchmark(rows, cfg, run, tokenizer, profile)
                    wait_drained(run / "observations")
                    if meta["errors"] or meta["profile_errors"]:
                        raise RuntimeError(f"Incomplete measurement: {run / 'measurement.json'}")
                    print(f"完成 {policy.upper()}：{meta['seconds']:.2f}s", flush=True)
                finally:
                    services.stop(process)
                time.sleep(3)
    finally:
        services.close()
    if profile:
        export_profiler(session)
    from analyze import report
    report(session)
    print(f"结果目录：{session}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check", "validate", "run", "profile", "report"))
    parser.add_argument("tag", nargs="?", help="Run name / existing result directory name")
    parser.add_argument("--policies", default="rr,jsq,kv")
    args = parser.parse_args()
    cfg = config()
    if args.command == "check":
        result = check_runtime(cfg)
        print(json.dumps({k: v for k, v in result.items() if k != "native_selector"}, indent=2))
    elif args.command == "prepare":
        prepare(cfg)
    elif args.command == "report":
        from analyze import report
        base = ROOT / "results"
        session = base / args.tag if args.tag else max(base.iterdir(), key=lambda p: p.stat().st_mtime)
        export_profiler(session)
        report(session)
    else:
        execute(args, cfg)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("已停止本次实验。", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(f"实验停止：{error}", file=sys.stderr)
        sys.exit(1)

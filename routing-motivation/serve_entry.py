"""Invoked by prepared copies of the user's existing P/D startup scripts."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys


def replace_option(args, flag, value=None):
    result = []
    i = 0
    while i < len(args):
        if args[i] == flag:
            i += 2 if value is not None else 1
        elif args[i].startswith(flag + "="):
            i += 1
        else:
            result.append(args[i])
            i += 1
    result.append(flag)
    if value is not None:
        result.append(str(value))
    return result


def main():
    root = Path(__file__).resolve().parent
    role, *args = sys.argv[1:]
    if not args or args[0] != "serve":
        raise RuntimeError("Only vllm serve is supported")
    cfg = json.loads((root / "config.json").read_text())
    cfg["model_path"] = str(Path(cfg["model_path"]).expanduser().resolve())
    args[1] = cfg["model_path"]
    args = replace_option(args, "--host", cfg["host_ip"])
    args = replace_option(args, "--port", cfg["decoder_port" if role == "decoder" else "prefiller_port"])
    args = replace_option(args, "--served-model-name", cfg["served_model"])
    args = [a for a in args if a != "--enable-prefix-caching" and not a.startswith("--enable-prefix-caching=")]
    args = replace_option(args, "--no-enable-prefix-caching")
    if role == "decoder":
        if "--tensor-parallel-size" in args and args[args.index("--tensor-parallel-size") + 1] != "1":
            raise RuntimeError("The current observation/rank mapping supports Decoder TP=1")
        # Explicitly prevent old KV extensions and this experiment from combining.
        for name in ("VLLM_ASCEND_KV_BLOCKS_ROUTING", "VLLM_ASCEND_KV_BLOCKS_LOG",
                     "VLLM_ASCEND_KV_BLOCKS_STEP_LOG", "VLLM_KV_BLOCKS_ROUTING",
                     "VLLM_DP_KV_BLOCKS_ROUTING", "VLLM_EXPERIMENTAL_DP_KV_ROUTING"):
            os.environ[name] = "0"
        args = replace_option(args, "--api-server-count", "1")
        if "--max-model-len" in args and int(args[args.index("--max-model-len") + 1]) < 10240:
            raise RuntimeError("The 96-request workload needs max-model-len >= 10240")
        if any(a.startswith(("--data-parallel-external-lb", "--data-parallel-hybrid-lb")) for a in args):
            raise RuntimeError("This experiment requires internal DP load balancing")
        index = next((i for i, a in enumerate(args) if a == "--additional-config"), None)
        if index is None:
            raise RuntimeError("BL script has no --additional-config; inspect prepared script")
        additional = json.loads(args[index + 1])
        scheduler = additional.setdefault("scheduler_config", {})
        scheduler["recompute_scheduler_enable"] = False
        scheduler.setdefault("dyntra_lb_config", {})["enabled"] = False
        args[index + 1] = json.dumps(additional, separators=(",", ":"))
        if os.environ.get("MOTIVATION_PROFILE") == "1":
            if any(a.startswith("--profiler-config") for a in args):
                raise RuntimeError("Remove pre-existing profiler options from the BL startup script")
            # Separate diagnostic run: eager mode gives attention CPU ranges per step.
            args = replace_option(args, "--enforce-eager")
            if "--compilation-config" in args:
                i = args.index("--compilation-config") + 1
                compilation = json.loads(args[i])
                compilation["cudagraph_mode"] = "NONE"
                args[i] = json.dumps(compilation)
            args += ["--profiler-config", json.dumps({"profiler": "torch",
                      "torch_profiler_dir": os.environ["MOTIVATION_PROFILE_DIR"],
                      "torch_profiler_with_stack": False, "ignore_frontend": True})]
    directory = Path(os.environ["MOTIVATION_RUN"])
    (directory / f"{role}-effective-args.json").write_text(json.dumps(args, indent=2), encoding="utf-8")
    paths = os.environ.get("PYTHONPATH", "")
    # sitecustomize is loaded by a fresh Python, after startup-script environment setup.
    os.environ["PYTHONPATH"] = str(root) + (os.pathsep + paths if paths else "")
    os.environ["MOTIVATION_ROLE"] = role
    real = json.loads((root / "prepared" / "environment.json").read_text())["vllm_executable"]
    os.execv(real, [real, *args])


if __name__ == "__main__":
    main()

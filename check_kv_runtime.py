"""Read-only diagnostics for an existing Linux Decode API-server process.

Reads /proc and source files. The optional package lookup runs a fresh Python
interpreter without importing vllm, vllm_ascend, torch, or any patch module.
It does not attach to or modify the running server.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

DEFAULTS = {
    "VLLM_ASCEND_KV_BLOCKS_ROUTING": "0",
    "VLLM_ASCEND_KV_BLOCKS_LOG": "1",
    "VLLM_ASCEND_KV_BLOCKS_STEP_LOG": "0",
    "VLLM_ASCEND_KV_BLOCKS_TIMEOUT_S": "30",
    "VLLM_LOGGING_LEVEL": "INFO",
    "VLLM_CONFIGURE_LOGGING": "1",
    "VLLM_LOGGING_CONFIG_PATH": "",
    "VLLM_LOGGING_STREAM": "ext://sys.stdout",
}

LOOKUP = r'''
import importlib.util, json, sys
from pathlib import Path
if len(sys.argv) > 1 and sys.argv[1]:
    sys.path.insert(0, sys.argv[1])
result = {}
for name in ("vllm", "vllm_ascend"):
    try:
        spec = importlib.util.find_spec(name)
        result[name] = spec.origin if spec else None
    except Exception as error:
        result[name] = {"error": str(error)}
origin = result.get("vllm_ascend")
if isinstance(origin, str):
    package = Path(origin).resolve().parent
    result["ascend_root"] = str(package.parent)
    checks = {}
    for relative, marker in (
        ("platform.py", "install_for_config(vllm_config)"),
        ("patch/platform/__init__.py", "import vllm_ascend.patch.platform.patch_kv_blocks_routing"),
        ("core/kv_blocks_routing.py", "VLLM_ASCEND_KV_BLOCKS_ROUTING"),
        ("patch/platform/patch_kv_blocks_routing.py", "AscendKVBlocksDPLBAsyncMPClient"),
    ):
        path = package / relative
        checks[relative] = {"exists": path.is_file(), "marker_present": False}
        if path.is_file():
            checks[relative]["marker_present"] = marker in path.read_text(errors="replace")
    result["ascend_source_checks"] = checks
print(json.dumps(result, ensure_ascii=False))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", required=True, type=int, help="current Decode APIServer PID")
    args = parser.parse_args()
    proc = Path("/proc") / str(args.pid)
    if not proc.is_dir():
        parser.error("This PID is not running here. Use the current Decode APIServer PID in the same container.")
    target_root, own_root = (proc / "root").stat(), Path("/").stat()
    if (target_root.st_dev, target_root.st_ino) != (own_root.st_dev, own_root.st_ino):
        parser.error("Run this check inside the same container as the API server so package paths are meaningful.")
    environment = {}
    for item in (proc / "environ").read_bytes().split(b"\0"):
        key, separator, value = item.partition(b"=")
        if separator:
            environment[os.fsdecode(key)] = os.fsdecode(value)
    executable = os.readlink(proc / "exe")
    cwd = os.readlink(proc / "cwd")
    command = [os.fsdecode(item) for item in (proc / "cmdline").read_bytes().split(b"\0") if item]
    interpreter = executable
    if command:
        candidate = command[0] if Path(command[0]).is_absolute() else shutil.which(command[0], path=environment.get("PATH"))
        if candidate and Path(candidate).is_file() and Path(candidate).samefile(executable):
            # Keep a virtualenv's original interpreter path rather than
            # replacing it with the physical /proc/PID/exe symlink target.
            interpreter = candidate
    script_directory = ""
    # Preserve the usual script directory used by a Python console entry point.
    for item in command[:2]:
        path = Path(item)
        if path.is_absolute() and path.is_file() and path.name in ("vllm", "api_server.py"):
            script_directory = str(path.parent)
            break
    print(f"API PID: {args.pid}")
    print(f"Python executable: {executable}")
    print(f"Python command used for lookup: {interpreter}")
    print(f"Working directory: {cwd}")
    print("\nSettings recorded when this process started:")
    for name, default in DEFAULTS.items():
        print(f"  {name}={environment[name]}" if name in environment
              else f"  {name}=<unset; default {default!r}>")
    if any(name.startswith("VLLM_DP_KV_BLOCKS_") for name in environment):
        print("  NOTE: old VLLM_DP_KV_BLOCKS_* settings exist; they do not activate the Ascend-only extension.")
    print("\nPackage lookup using that executable, environment and working directory (no package imports):")
    result = subprocess.run([interpreter, "-B", "-c", LOOKUP, script_directory], env=environment,
                            cwd=cwd, text=True, capture_output=True, timeout=30)
    if result.returncode:
        print("  Lookup failed:", result.stderr[-2000:])
        return 1
    lookup = json.loads(result.stdout.strip())
    print(json.dumps(lookup, indent=2, ensure_ascii=False))
    print("\nInterpretation:")
    if environment.get("VLLM_ASCEND_KV_BLOCKS_ROUTING", "0") != "1":
        print("  The process-start environment does not enable KV-block routing.")
    if environment.get("VLLM_LOGGING_LEVEL", "INFO").upper() in ("WARNING", "WARN", "ERROR", "CRITICAL", "FATAL"):
        print("  The vLLM log-level setting filters out INFO-level route and step arrays.")
    if environment.get("VLLM_LOGGING_CONFIG_PATH") or environment.get("VLLM_CONFIGURE_LOGGING", "1") != "1":
        print("  Custom/disabled vLLM logging is configured; handler filters must also be checked.")
    if any(not check["marker_present"] for check in lookup.get("ascend_source_checks", {}).values()):
        print("  At least one Ascend source hook/module is missing from this package location.")
    print("  Compare ascend_root with the --ascend-root used at installation.")
    print("  /proc/environ records startup settings; later in-process changes may differ.")
    print("  Package lookup checks disk resolution in a fresh interpreter, not the running API object's class.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"CHECK ERROR: {error}", file=sys.stderr)
        sys.exit(1)

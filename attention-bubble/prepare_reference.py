#!/usr/bin/env python3
"""Make experiment copies of the DyntraLB startup scripts."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def replace_once(text: str, before: str, after: str, label: str) -> str:
    count = text.count(before)
    if count != 1:
        raise RuntimeError(f"{label}: expected one matching location, found {count}")
    return text.replace(before, after, 1)


def patch_proxy(source: str) -> str:
    signature = (
        "async def stream_service_response_with_retry(\n"
        "    client: httpx.AsyncClient,\n"
        "    endpoint: str,\n"
        "    req_data: dict,\n"
        "    request_id: str,\n"
    )
    source = replace_once(
        source, signature, signature + "    data_parallel_rank: int | None = None,\n",
        "decoder streaming signature",
    )
    function_start = source.index("async def stream_service_response_with_retry(")
    function_end = source.index("async def _handle_select_instance(", function_start)
    function = source[function_start:function_end]
    header = '    headers = {"Authorization": f"Bearer {os.environ.get(\'OPENAI_API_KEY\')}", "X-Request-Id": request_id}\n'
    function = replace_once(
        function, header,
        header + '    if data_parallel_rank is not None:\n'
                 '        headers["X-data-parallel-rank"] = str(data_parallel_rank)\n',
        "decoder request header",
    )
    source = source[:function_start] + function + source[function_end:]

    handler_start = source.index("async def _handle_completions(")
    handler_end = source.index("async def _handle_adjust_instances(", handler_start)
    handler = source[handler_start:handler_end]
    handler = replace_once(
        handler,
        "        req_data = await request.json()\n",
        "        req_data = await request.json()\n"
        '        rank_header = request.headers.get("X-data-parallel-rank")\n'
        '        data_parallel_rank = int(rank_header) if rank_header is not None else None\n',
        "proxy incoming rank header",
    )
    handler = replace_once(
        handler,
        "                        request_id=instance_info.request_id,\n",
        "                        request_id=instance_info.request_id,\n"
        "                        data_parallel_rank=data_parallel_rank,\n",
        "decoder streaming call",
    )
    handler = replace_once(
        handler,
        "        return StreamingResponse(generate_stream(), media_type=media_type)\n",
        '        response_headers = {"X-Proxy-Request-Id": instance_info.request_id}\n'
        '        if data_parallel_rank is not None:\n'
        '            response_headers["X-Requested-DP-Rank"] = str(data_parallel_rank)\n'
        '        return StreamingResponse(generate_stream(), media_type=media_type,\n'
        '                                 headers=response_headers)\n',
        "proxy rank acknowledgement",
    )
    return source[:handler_start] + handler + source[handler_end:]


def patch_decoder(source: str) -> str:
    source = replace_once(
        source,
        'exec vllm serve "$MODEL_PATH" \\\n',
        'if [[ -z "${PROFILER_CONFIG:-}" ]]; then\n'
        '  PROFILER_CONFIG=\'{"profiler":"torch","torch_profiler_dir":"./profile_attention_bubble","torch_profiler_with_stack":false}\'\n'
        'fi\n\n'
        'exec vllm serve "$MODEL_PATH" \\\n',
        "decoder profiler config",
    )
    source = replace_once(
        source,
        '  --additional-config \\\n',
        '  --profiler-config "$PROFILER_CONFIG" \\\n'
        '  --additional-config \\\n',
        "decoder profiler flag",
    )
    return source


def patch_prefill(source: str) -> str:
    # Apply these changes only to a new experiment copy of the known script.
    # The original cleanup also deletes generic Python shared-memory files.
    source = replace_once(
        source, "cleanup_stale_vllm_shm || exit 1\n",
        '# Shared-memory cleanup intentionally disabled in the experiment copy.\n',
        "prefill shared-memory cleanup call",
    )
    source, count = re.subn(r'^export HCCL_IF_IP=[^\n]+$',
                            'export HCCL_IF_IP="$HOST_IP"', source, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"prefill HCCL_IF_IP: expected one matching location, found {count}")
    source, count = re.subn(r'^(\s*--host\s+)[^\s\\]+',
                            r'\1"$HOST_IP"', source, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"prefill --host: expected one matching location, found {count}")
    return source


def patch_local_decoder(source: str) -> str:
    source = patch_decoder(source)
    source = replace_once(
        source,
        'exec vllm serve "$MODEL_PATH" \\\n',
        ': "${EXPERIMENT_PYTHON_DIR:?Set EXPERIMENT_PYTHON_DIR to the directory containing scheduler_probe.py}"\n'
        ': "${BUBBLE_STATE_DIR:?Set BUBBLE_STATE_DIR to an absolute directory for scheduler snapshots}"\n'
        '[[ "$BUBBLE_STATE_DIR" = /* ]] || { echo "BUBBLE_STATE_DIR must be absolute" >&2; exit 1; }\n'
        '[[ -f "$EXPERIMENT_PYTHON_DIR/scheduler_probe.py" ]] || { echo "scheduler_probe.py not found" >&2; exit 1; }\n'
        'mkdir -p "$BUBBLE_STATE_DIR"\n'
        'export PYTHONPATH="$EXPERIMENT_PYTHON_DIR${PYTHONPATH:+:$PYTHONPATH}"\n'
        'export BUBBLE_STATE_DIR\n'
        'exec vllm serve "$MODEL_PATH" \\\n',
        "local experiment environment",
    )
    source = replace_once(
        source,
        '  --max-num-seqs 64 \\\n',
        '  --max-num-seqs 4 \\\n'
        '  --async-scheduling \\\n'
        '  --scheduler-cls scheduler_probe.ProbeAsyncScheduler \\\n',
        "local experiment scheduler",
    )
    return source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--repo", type=Path, help="Legacy layout: original scripts in REPO/scripts")
    source_group.add_argument("--scripts-dir", type=Path, help="Directory containing the original scripts")
    parser.add_argument("--output-dir", type=Path, help="Separate experiment directory (required with --scripts-dir)")
    parser.add_argument("--force", action="store_true", help="Replace earlier generated copies")
    args = parser.parse_args()
    if args.repo:
        if args.output_dir is not None:
            parser.error("--output-dir is only supported with --scripts-dir")
        scripts = output_dir = args.repo / "scripts"
        jobs = (("load_balance_proxy_server_example.py", "load_balance_proxy_server_ranked.py", patch_proxy),
                ("run_vllm_serve_8p8d_asyncSched_d_original.sh", "run_vllm_serve_8p8d_asyncSched_d_profile.sh", patch_decoder),
                ("run_vllm_serve_8p8d_asyncSched_d_original.sh", "run_vllm_serve_8p8d_asyncSched_d_local_pairs.sh", patch_local_decoder))
    else:
        if args.output_dir is None:
            parser.error("--output-dir is required with --scripts-dir")
        scripts, output_dir = args.scripts_dir, args.output_dir
        if scripts.resolve() == output_dir.resolve():
            parser.error("--output-dir must be different from --scripts-dir")
        jobs = (("run_vllm_serve_8p8d_asyncSched_p.sh", "run_vllm_serve_8p8d_asyncSched_p_bubble.sh", patch_prefill),
                ("load_balance_proxy_server_example.py", "load_balance_proxy_server_ranked.py", patch_proxy),
                ("run_vllm_serve_8p8d_asyncSched_d_original.sh", "run_vllm_serve_8p8d_asyncSched_d_local_pairs.sh", patch_local_decoder))
    if not output_dir.is_dir():
        raise NotADirectoryError(output_dir)
    prepared = []
    for input_name, output_name, transform in jobs:
        source_path = scripts / input_name
        output_path = output_dir / output_name
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        if output_path.is_symlink():
            raise RuntimeError(f"Refusing to overwrite symlink: {output_path}")
        if output_path.exists() and not args.force:
            raise FileExistsError(f"{output_path}; pass --force to replace it")
        prepared.append((source_path, output_path, transform(source_path.read_text(encoding="utf-8"))))
    for source_path, output_path, content in prepared:
        output_path.write_text(content, encoding="utf-8")
        output_path.chmod(source_path.stat().st_mode & 0o777)
        print(output_path)


if __name__ == "__main__":
    main()

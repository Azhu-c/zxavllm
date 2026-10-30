#!/usr/bin/env python3
"""Make rank-aware profiling copies of the referenced DyntraLB scripts.

Run this against a local clone of https://github.com/Ydr-pku/DyntraLB.
The original scripts are preserved. The copies are placed beside them.
"""

from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="Local DyntraLB clone")
    parser.add_argument("--force", action="store_true", help="Replace earlier generated copies")
    args = parser.parse_args()
    scripts = args.repo / "scripts"
    jobs = (
        ("load_balance_proxy_server_example.py",
         "load_balance_proxy_server_ranked.py", patch_proxy),
        ("run_vllm_serve_8p8d_asyncSched_d_original.sh",
         "run_vllm_serve_8p8d_asyncSched_d_profile.sh", patch_decoder),
    )
    prepared = []
    for input_name, output_name, transform in jobs:
        source_path = scripts / input_name
        output_path = scripts / output_name
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        if output_path.exists() and not args.force:
            raise FileExistsError(f"{output_path}; pass --force to replace it")
        prepared.append((source_path, output_path, transform(source_path.read_text(encoding="utf-8"))))
    for source_path, output_path, content in prepared:
        output_path.write_text(content, encoding="utf-8")
        output_path.chmod(source_path.stat().st_mode & 0o777)
        print(output_path)


if __name__ == "__main__":
    main()

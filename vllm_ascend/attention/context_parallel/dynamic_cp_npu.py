# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Ascend FIA adapter for the experimental dynamic-DP CP runtime."""

import torch_npu
from vllm.distributed import get_dp_group
from vllm.forward_context import get_forward_context
from vllm.logger import logger


def partial_attention(query, key_cache, value_cache, table, visible_tokens, impl, causal=False, causal_mask=None):
    if causal and causal_mask is None:
        raise RuntimeError("Dynamic CP MTP requires the target attention metadata's causal mask")
    return torch_npu.npu_fused_infer_attention_score(
        query=query,
        key=key_cache.view(key_cache.shape[0], key_cache.shape[1], -1),
        value=value_cache.view(value_cache.shape[0], value_cache.shape[1], -1),
        block_table=table,
        block_size=key_cache.shape[1],
        input_layout="TND",
        actual_seq_lengths=[query.shape[0]],
        actual_seq_lengths_kv=[visible_tokens],
        num_heads=impl.num_heads,
        num_key_value_heads=impl.num_kv_heads,
        scale=impl.scale,
        sparse_mode=3 if causal else 0,
        atten_mask=causal_mask if causal else None,
        softmax_lse_flag=True,
    )


def dynamic_cp_forward(runtime, impl, query, metadata, output, key=None, value=None, kv_cache=None):
    context = get_forward_context()
    first_layer = getattr(context, "dyntra_cp_frame", None) is None
    previous = runtime.previous
    group = get_dp_group()
    result = runtime.forward(
        impl,
        query.view(-1, impl.num_heads, impl.head_size),
        metadata,
        output,
        context,
        group,
        partial_attention,
        (key, value, kv_cache),
    )
    if first_layer and runtime.enable_diagnostics and group.rank_in_group == 0 and runtime.previous != previous:
        logger.info(
            "DYNTRA_CP leases=%s",
            [
                dict(
                    owner=p.owner,
                    helper=p.helper,
                    request_id=p.request.request_id,
                    prefix_blocks=len(p.request.prefix_blocks),
                    query_len=p.request.query_len,
                )
                for p in runtime.previous
            ],
        )
    return result

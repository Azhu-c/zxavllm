# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Server-side FIA/LSE checks for Qwen3.5-shaped GQA and MTP verification."""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.attention.context_parallel.dynamic_cp_npu import partial_attention
from vllm_ascend.core.dyntra_cp_runtime import merge_attention_partials


@pytest.mark.parametrize("query_len", [1, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dynamic_cp_fia_matches_unsplit_causal_attention(query_len, dtype):
    torch.manual_seed(12292)
    device = torch.device("npu")
    block_size, head_size, num_heads, num_kv_heads = 128, 256, 16, 2
    num_blocks, seq_len, prefix_blocks = 34, 4097, 12
    impl = SimpleNamespace(num_heads=num_heads, num_kv_heads=num_kv_heads, head_size=head_size, scale=head_size**-0.5)
    key = torch.randn(num_blocks, block_size, num_kv_heads, head_size, device=device, dtype=dtype)
    value = torch.randn_like(key)
    query = torch.randn(query_len, num_heads, head_size, device=device, dtype=dtype)
    # Non-contiguous physical blocks exercise compact replica addressing.
    table = torch.randperm(num_blocks, device=device, dtype=torch.int64).to(torch.int32).unsqueeze(0)
    compact_key = key[table[0, :prefix_blocks].long()].contiguous()
    compact_value = value[table[0, :prefix_blocks].long()].contiguous()
    helper_table = torch.arange(prefix_blocks, device=device, dtype=torch.int32).unsqueeze(0)
    prefix_tokens = prefix_blocks * block_size
    expected = []
    for i in range(query_len):
        visible = seq_len - query_len + i + 1
        full_out, _ = partial_attention(query[i : i + 1], key, value, table, visible, impl)
        expected.append(full_out.reshape(1, num_heads, head_size))
    causal_mask = torch.triu(torch.ones(2048, 2048, dtype=torch.bool, device=device), diagonal=1)
    owner_out, owner_lse = partial_attention(
        query,
        key,
        value,
        table[:, prefix_blocks:].contiguous(),
        seq_len - prefix_tokens,
        impl,
        causal=query_len > 1,
        causal_mask=causal_mask,
    )
    helper_out, helper_lse = partial_attention(query, compact_key, compact_value, helper_table, prefix_tokens, impl)
    merged = merge_attention_partials(
        owner_out.reshape(query_len, num_heads, head_size),
        owner_lse.reshape(query_len, num_heads, 1),
        helper_out.reshape(query_len, num_heads, head_size),
        helper_lse.reshape(query_len, num_heads, 1),
    )
    torch.testing.assert_close(merged.to(dtype), torch.cat(expected), rtol=2e-2, atol=2e-2)

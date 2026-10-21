# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from source_modules import load_source

POLICY = load_source("dyntra_cp_policy")
RUNTIME = load_source("dyntra_cp_runtime")


def candidate(name, length=40, query_len=4, blocks=(0, 1, 2, 3)):
    return POLICY.CPCandidate(name, 0, length, query_len, blocks, 4)


def test_sticky_lease_is_revoked_when_helper_has_local_work():
    frames = [POLICY.CPWorkload(0, False, (candidate("a"),)), POLICY.CPWorkload(1, True), POLICY.CPWorkload(2, True)]
    previous = POLICY.choose_cp_leases(frames, 1)
    assert previous[0].helper == 1
    frames[1] = POLICY.CPWorkload(1, False)
    current = POLICY.choose_cp_leases(frames, 1, previous)
    assert current[0].helper == 2
    frames[2] = POLICY.CPWorkload(2, False)
    assert not POLICY.choose_cp_leases(frames, 1, current)


def test_replica_identity_and_recycled_blocks():
    key = (0, "request-a")
    assert POLICY.reusable_prefix_blocks(key, (5, 7), key, (5, 7, 9)) == 2
    assert POLICY.reusable_prefix_blocks(key, (5, 7), key, (5, 8)) == 0
    assert POLICY.reusable_prefix_blocks(key, (5, 7), (0, "request-b"), (5, 7)) == 0


def test_lse_merge_is_not_an_unweighted_average():
    merged = RUNTIME.merge_attention_partials(
        torch.tensor([[[1.0]]]), torch.tensor([[[0.0]]]), torch.tensor([[[9.0]]]), torch.tensor([[[2.0]]])
    )
    expected = (1 + 9 * torch.exp(torch.tensor(2.0))) / (1 + torch.exp(torch.tensor(2.0)))
    torch.testing.assert_close(merged.squeeze(), expected)
    assert not torch.allclose(merged.squeeze(), torch.tensor(5.0))


def test_lse_merge_handles_extreme_and_empty_partitions():
    a = torch.tensor([[[10000.0]], [[-float("inf")]]])
    out = RUNTIME.merge_attention_partials(torch.ones(2, 1, 2), a, 9 * torch.ones(2, 1, 2), a - 1000)
    torch.testing.assert_close(out[0], torch.ones(1, 2))
    torch.testing.assert_close(out[1], torch.zeros(1, 2))


@dataclass
class Metadata:
    actual_seq_lengths_q: list[int]
    seq_lens_list: list[int]
    block_tables: torch.Tensor
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    num_actual_tokens: int
    num_decode_tokens: int
    num_decodes: int
    num_prefills: int = 0


def reference_partial(query, key, value, table, visible, impl, causal=False, causal_mask=None):
    block_size = key.shape[1]
    blocks = table[0, : (visible + block_size - 1) // block_size].long()
    keys = key[blocks].reshape(-1, impl.num_kv_heads, impl.head_size)[:visible]
    values = value[blocks].reshape_as(key[blocks]).reshape(-1, impl.num_kv_heads, impl.head_size)[:visible]
    keys = keys.repeat_interleave(impl.num_heads // impl.num_kv_heads, dim=1)
    values = values.repeat_interleave(impl.num_heads // impl.num_kv_heads, dim=1)
    scores = torch.einsum("qhd,khd->qhk", query, keys) * impl.scale
    if causal:
        query_positions = visible - query.shape[0] + torch.arange(query.shape[0])
        mask = torch.arange(visible).unsqueeze(0) > query_positions.unsqueeze(1)
        scores.masked_fill_(mask.unsqueeze(1), -float("inf"))
    lse = torch.logsumexp(scores, dim=-1, keepdim=True)
    out = torch.einsum("qhk,khd->qhd", scores.softmax(dim=-1), values)
    return out, lse


class ReferenceAttention:
    num_heads = 4
    num_kv_heads = 2
    head_size = 4
    scale = 0.5

    def __init__(self, seed):
        generator = torch.Generator().manual_seed(seed)
        self.key_cache = torch.randn((40, 4, 2, 4), generator=generator)
        self.value_cache = torch.randn((40, 4, 2, 4), generator=generator)

    def forward_impl(self, query, key, value, cache, metadata, output):
        start = 0
        for index, end in enumerate(metadata.actual_seq_lengths_q):
            q_len = end - start
            for j in range(q_len):
                out, _ = reference_partial(
                    query[start + j : start + j + 1],
                    self.key_cache,
                    self.value_cache,
                    metadata.block_tables[index : index + 1],
                    metadata.seq_lens_list[index] - q_len + j + 1,
                    self,
                )
                output[start + j : start + j + 1].copy_(out)
            start = end
        return output


def build_batch(rank, step):
    if rank == 0:
        requests = [("long-a" if step < 3 else "replacement", 40 + 4 * step, 4), ("short-a", 9, 1)]
    elif rank == 1:
        requests = [("long-b", 48 + 4 * step, 4)]
    elif rank == 2 and step >= 2:
        requests = [("new-local-2", 9, 1)]
    elif rank == 3 and step == 4:
        requests = [("new-local-3", 9, 1)]
    else:
        requests = []
    ids = tuple(item[0] for item in requests)
    generator = torch.Generator().manual_seed(100 + rank + step)
    if not requests:
        return ids, torch.zeros(1, 4, 4), None
    lengths, ends, tables = [], [], []
    total = 0
    for index, (_, length, q_len) in enumerate(requests):
        total += q_len
        lengths.append(length)
        ends.append(total)
        tables.append(list(range(index * 20, (index + 1) * 20)))
    lens = torch.tensor(lengths, dtype=torch.int32)
    metadata = Metadata(ends, lengths, torch.tensor(tables, dtype=torch.int32), lens, lens, total, total, len(ids))
    return ids, torch.randn(total, 4, 4, generator=generator), metadata


def distributed_worker(rank, world_size, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", rank=rank, world_size=world_size, init_method=f"file://{rendezvous}", timeout=timedelta(seconds=30)
    )
    try:
        group = SimpleNamespace(
            world_size=world_size,
            rank_in_group=rank,
            ranks=list(range(world_size)),
            cpu_group=dist.group.WORLD,
            device_group=dist.group.WORLD,
        )
        config = POLICY.DynamicCPConfig(
            {"enabled": True, "min_context_tokens": 24, "cache_budget_mb": 1, "max_pairs": 2}
        )
        runtime = RUNTIME.DynamicDPContextParallel(config, 1, 2, 4)
        impl = ReferenceAttention(rank)
        first_copied = 0
        for step in range(8):
            ids, query, metadata = build_batch(rank, step)
            if rank == 0 and step == 3:
                # Same physical block table, different request and KV values.
                impl = ReferenceAttention(900)
            if step == 5 and rank == 3:

                def fail_allocation(*args):
                    raise RuntimeError("simulated out of memory")

                runtime._allocate_prefix = fail_allocation
            if step == 6 and rank == 3:
                del runtime._allocate_prefix
            expected = torch.empty_like(query)
            if metadata is None:
                expected.zero_()
            else:
                impl.forward_impl(query, None, None, None, metadata, expected)
            context = SimpleNamespace(dyntra_cp_request_ids=ids, in_profile_run=False, is_draft_model=False)
            if step == 7:

                def partial_with_failure(*args, **kwargs):
                    if rank == 3:
                        raise RuntimeError("simulated FIA kernel failure")
                    return reference_partial(*args, **kwargs)

                with pytest.raises(RuntimeError, match="simulated FIA kernel failure"):
                    runtime.forward(
                        impl, query, metadata, torch.empty_like(query), context, group, partial_with_failure
                    )
                continue
            result = runtime.forward(impl, query, metadata, torch.empty_like(query), context, group, reference_partial)
            torch.testing.assert_close(result, expected, atol=2e-6, rtol=2e-6)
            if step == 1 and rank in (2, 3):
                assert runtime.stats["reused_blocks"] > 0
                assert runtime.stats["copied_blocks"] > first_copied
            if step == 0:
                first_copied = runtime.stats["copied_blocks"]
            if step == 2 and rank == 2:
                assert runtime.resident is None
            if step == 4:
                assert not context.dyntra_cp_frame.leases
                assert runtime.resident is None
            if step == 5:
                assert runtime.stats["allocation_fallbacks"] == 1
        assert runtime.stats["leased_steps"] > 0
    finally:
        dist.destroy_process_group()


def test_real_four_rank_transport_matches_causal_mtp_reference(tmp_path):
    # Real Gloo collectives and P2P; only the FIA kernel is a CPU reference.
    mp.spawn(distributed_worker, args=(4, str(tmp_path / "rendezvous")), nprocs=4, join=True)


def supported_config(model_type="qwen3_5_moe_text", **changes):
    parallel = SimpleNamespace(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
        data_parallel_size=8,
        nnodes_within_dp=1,
    )
    config = SimpleNamespace(
        parallel_config=parallel,
        model_config=SimpleNamespace(enforce_eager=True, hf_text_config=SimpleNamespace(model_type=model_type)),
        use_v2_model_runner=False,
        speculative_config=SimpleNamespace(method="mtp"),
        cache_config=SimpleNamespace(cache_dtype="auto"),
        lora_config=None,
    )
    for key, value in changes.items():
        setattr(config, key, value)
    return config


def test_target_qwen35_dp8_mtp_configuration_is_supported():
    POLICY.validate_dynamic_cp(POLICY.DynamicCPConfig({"enabled": True}), supported_config())


@pytest.mark.parametrize(
    "key,value", [("enabled", 1), ("min_context_tokens", 0), ("cache_budget_mb", True), ("max_pairs", -1)]
)
def test_invalid_dynamic_cp_config(key, value):
    with pytest.raises(ValueError):
        POLICY.DynamicCPConfig({key: value})


def test_graph_execution_fails_at_startup():
    config = supported_config()
    config.model_config.enforce_eager = False
    with pytest.raises(ValueError, match="eager"):
        POLICY.validate_dynamic_cp(POLICY.DynamicCPConfig({"enabled": True}), config)


@pytest.mark.parametrize("attribute", ["is_draft_model", "in_profile_run"])
def test_draft_and_profile_keep_native_inputs_and_skip_cp_collectives(attribute):
    runtime = RUNTIME.DynamicDPContextParallel(POLICY.DynamicCPConfig({"enabled": True}), 1, 2, 4)
    key, value, cache = object(), object(), object()
    output = torch.empty(1, 4, 4)
    seen = []

    class Attention:
        def forward_impl(self, q, k, v, kv, meta, out):
            seen.append((k, v, kv))
            return out.fill_(3)

    context = SimpleNamespace(**{attribute: True})
    result = runtime.forward(Attention(), output, object(), output, context, None, None, (key, value, cache))
    assert seen == [(key, value, cache)]
    torch.testing.assert_close(result, torch.full_like(result, 3))


def test_full_forward_context_reuses_one_plan_across_layers():
    runtime = RUNTIME.DynamicDPContextParallel(POLICY.DynamicCPConfig({"enabled": True}), 2, 2, 4)
    frame = RUNTIME.CPFrame(())
    context = SimpleNamespace(dyntra_cp_frame=frame)
    assert runtime._frame(None, None, None, context, None) is frame

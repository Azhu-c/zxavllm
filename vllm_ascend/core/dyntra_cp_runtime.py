# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Experimental eager dynamic-DP CP transport, independent of the NPU kernel.

Only immutable historical prefix blocks are replicated. The owner retains the
complete authoritative KV and request state. Idle helpers never sample or run
another copy of the owner's MoE tokens. All DP ranks follow the same protocol.
"""

from dataclasses import dataclass, replace

import torch
import torch.distributed as dist

from vllm_ascend.core.dyntra_cp_policy import (
    CPCandidate,
    CPLease,
    CPWorkload,
    DynamicCPConfig,
    choose_cp_leases,
    reusable_prefix_blocks,
)


@dataclass
class CPFrame:
    leases: tuple[CPLease, ...]


@dataclass
class ResidentPrefix:
    key: tuple[int, str]
    source_blocks: tuple[int, ...]
    kv: torch.Tensor


def merge_attention_partials(local_out, local_lse, remote_out, remote_lse):
    """FP32 log-sum-exp merge; ordinary averaging of partial outputs is wrong."""
    local_lse, remote_lse = local_lse.float(), remote_lse.float()
    combined = torch.logaddexp(local_lse, remote_lse)
    valid = torch.isfinite(combined)
    local_weight = torch.where(valid, (local_lse - combined).exp(), 0.0)
    remote_weight = torch.where(valid, (remote_lse - combined).exp(), 0.0)
    return local_out.float() * local_weight + remote_out.float() * remote_weight


class DynamicDPContextParallel:
    def __init__(
        self,
        config: DynamicCPConfig,
        num_layers: int,
        num_kv_heads: int,
        head_size: int,
        enable_diagnostics: bool = False,
    ):
        self.config = config
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.enable_diagnostics = enable_diagnostics
        self.previous: tuple[CPLease, ...] = ()
        self.resident: ResidentPrefix | None = None
        self.stats = dict(leased_steps=0, copied_blocks=0, reused_blocks=0, allocation_fallbacks=0)

    def _capacity(self, block_size: int, element_size: int) -> int:
        bytes_per_block_all_layers = (
            2 * block_size * self.num_kv_heads * self.head_size * element_size * self.num_layers
        )
        return self.config.cache_budget_mb * 1024 * 1024 // bytes_per_block_all_layers

    def _allocate_prefix(self, query, capacity, block_size):
        return query.new_empty((2, capacity, block_size, self.num_kv_heads, self.head_size))

    @staticmethod
    def _states(values, group):
        local = torch.tensor(values, dtype=torch.int64, device="cpu")
        gathered = [torch.empty_like(local) for _ in range(group.world_size)]
        dist.all_gather(gathered, local, group=group.cpu_group)
        return [item.tolist() for item in gathered]

    @staticmethod
    def _fail_collectively(phase, error, group):
        errors = [None] * group.world_size
        dist.all_gather_object(errors, error, group=group.cpu_group)
        raise RuntimeError(f"Dynamic CP {phase} failed; DP rank errors={errors}")

    @staticmethod
    def _transfer(tensors, peer, sending, group):
        op = dist.isend if sending else dist.irecv
        requests = dist.batch_isend_irecv(
            [dist.P2POp(op, tensor, group.ranks[peer], group=group.device_group) for tensor in tensors]
        )
        for request in requests:
            request.wait()

    def _frame(self, impl, query, metadata, context, group):
        existing = getattr(context, "dyntra_cp_frame", None)
        if existing is not None:
            return existing
        req_ids = getattr(context, "dyntra_cp_request_ids", ())
        candidates = []
        error = None
        try:
            if metadata is not None and req_ids and metadata.num_prefills == 0:
                if impl.key_cache is None or impl.key_cache.dtype != query.dtype:
                    raise ValueError("Dynamic CP requires initialized FP16/BF16 KV matching the query dtype")
                block_size = impl.key_cache.shape[1]
                capacity = self._capacity(block_size, query.element_size())
                start = 0
                for index, req_id in enumerate(req_ids):
                    end = metadata.actual_seq_lengths_q[index]
                    query_len = end - start
                    seq_len = metadata.seq_lens_list[index]
                    start = end
                    if not 1 <= query_len <= 16 or seq_len < self.config.min_context_tokens:
                        continue
                    # Prefix blocks end before the earliest verified query.
                    # Speculative/rejected tail slots can never enter a replica.
                    prefix_count = min((seq_len - query_len) // block_size // 2, capacity)
                    if prefix_count:
                        blocks = tuple(metadata.block_tables[index, :prefix_count].to("cpu").tolist())
                        candidates.append(CPCandidate(req_id, index, seq_len, query_len, blocks, block_size))
        except (RuntimeError, ValueError, IndexError) as exc:
            error = str(exc)
        local = CPWorkload(group.rank_in_group, metadata is None and not req_ids, tuple(candidates), error)
        flags = self._states([int(bool(candidates)), int(local.idle), int(error is not None)], group)
        if not any(f[2] for f in flags) and not (any(f[0] for f in flags) and any(f[1] for f in flags)):
            leases = ()
        else:
            gathered = [None] * group.world_size
            dist.all_gather_object(gathered, local, group=group.cpu_group)
            errors = [w.error for w in gathered if w.error]
            if errors:
                raise RuntimeError(f"Dynamic CP metadata validation failed on DP ranks: {errors}")
            leases = choose_cp_leases(gathered, self.config.max_pairs, self.previous)
        self.previous = leases
        context.dyntra_cp_frame = CPFrame(leases)
        if leases:
            # PyTorch requires all members to initialize a process group before
            # its first subset P2P operation. Do this in the same order on all
            # ranks, before any pair starts sending.
            bootstrap = query.new_zeros(1)
            dist.all_reduce(bootstrap, group=group.device_group)
        return context.dyntra_cp_frame

    @staticmethod
    def _native(impl, query, metadata, output, key=None, value=None, cache=None):
        if metadata is None:
            return output.zero_()
        return impl.forward_impl(query, key, value, cache, metadata, output)

    @classmethod
    def _other_requests(cls, impl, query, metadata, output, excluded):
        starts = [0, *metadata.actual_seq_lengths_q]
        indices = [i for i in range(len(metadata.seq_lens_list)) if i != excluded]
        if not indices:
            return
        slices = [(starts[i], starts[i + 1]) for i in indices]
        selected_query = torch.cat([query[a:b] for a, b in slices])
        selected_output = output.new_empty((selected_query.shape[0], *output.shape[1:]))
        cumulative = []
        total = 0
        for a, b in slices:
            total += b - a
            cumulative.append(total)
        selected_metadata = replace(
            metadata,
            actual_seq_lengths_q=cumulative,
            seq_lens_list=[metadata.seq_lens_list[i] for i in indices],
            seq_lens=metadata.seq_lens[indices],
            seq_lens_cpu=metadata.seq_lens_cpu[indices],
            block_tables=metadata.block_tables[indices],
            num_actual_tokens=total,
            num_decode_tokens=total,
            num_decodes=len(indices),
        )
        cls._native(impl, selected_query, selected_metadata, selected_output)
        offset = 0
        for a, b in slices:
            output[a:b].copy_(selected_output[offset : offset + b - a])
            offset += b - a

    def forward(
        self, impl, query, metadata, output, context, group, partial_attention, native_inputs=(None, None, None)
    ):
        if getattr(context, "in_profile_run", False) or getattr(context, "is_draft_model", False):
            return self._native(impl, query, metadata, output, *native_inputs)
        frame = self._frame(impl, query, metadata, context, group)
        rank = group.rank_in_group
        lease = next((p for p in frame.leases if rank in (p.owner, p.helper)), None)
        if not any(rank == p.helper for p in frame.leases):
            self.resident = None
        if not frame.leases:
            return self._native(impl, query, metadata, output, *native_inputs)
        self.stats["leased_steps"] += int(lease is not None)
        # All ranks participate even when locally idle or not in a lease.
        # Each rank belongs to at most one pair in a frame.
        helper_query = None
        ready = 1
        cached = 0
        try:
            if lease is not None and rank == lease.helper:
                block_size = lease.request.block_size
                capacity = self._capacity(block_size, query.element_size())
                if capacity < len(lease.request.prefix_blocks):
                    raise RuntimeError("Dynamic CP helper capacity is smaller than the advertised prefix")
                key = lease.owner, lease.request.request_id
                blocks = lease.request.prefix_blocks
                if self.resident is None:
                    kv = self._allocate_prefix(query, capacity, block_size)
                    self.resident = ResidentPrefix(key, (), kv)
                cached = reusable_prefix_blocks(self.resident.key, self.resident.source_blocks, key, blocks)
                self.resident.key = key
                self.resident.source_blocks = blocks[:cached]
                helper_query = query.new_empty((lease.request.query_len, impl.num_heads, impl.head_size))
        except RuntimeError:
            ready = 0
        states = self._states([ready, cached], group)
        if not all(s[0] for s in states):
            self.resident = None
            self.stats["allocation_fallbacks"] += 1
            return self._native(impl, query, metadata, output, *native_inputs)

        payload = None
        selected_query = None
        remote_result = None
        try:
            if lease is not None:
                prefix_count = len(lease.request.prefix_blocks)
                cached = states[lease.helper][1]
                if rank == lease.owner:
                    index = lease.request.request_index
                    start = 0 if index == 0 else metadata.actual_seq_lengths_q[index - 1]
                    selected_query = query[start : start + lease.request.query_len].contiguous()
                    if cached < prefix_count:
                        block_ids = metadata.block_tables[index, cached:prefix_count].long()
                        payload = torch.stack((impl.key_cache[block_ids], impl.value_cache[block_ids]))
                    remote_result = torch.empty(
                        (lease.request.query_len, impl.num_heads, impl.head_size + 1),
                        dtype=torch.float32,
                        device=query.device,
                    )
                elif cached < prefix_count:
                    payload = self.resident.kv[:, cached:prefix_count]
                    # The receive buffer must be contiguous. The two KV planes
                    # in a capacity-sized resident allocation are not adjacent.
                    payload = torch.empty_like(payload, memory_format=torch.contiguous_format)
        except RuntimeError:
            ready = 0
        if not all(s[0] for s in self._states([ready], group)):
            payload = None
            self.resident = None
            self.stats["allocation_fallbacks"] += 1
            return self._native(impl, query, metadata, output, *native_inputs)

        # Every pair posts transfers in a globally deterministic order.
        for pair in frame.leases:
            if rank not in (pair.owner, pair.helper):
                continue
            tensors = ([] if payload is None else [payload]) + [selected_query if rank == pair.owner else helper_query]
            self._transfer(tensors, pair.helper if rank == pair.owner else pair.owner, rank == pair.owner, group)
            if rank == pair.helper:
                if payload is not None:
                    self.resident.kv[:, cached:prefix_count].copy_(payload)
                    self.stats["copied_blocks"] += prefix_count - cached
                self.stats["reused_blocks"] += cached
                self.resident.source_blocks = pair.request.prefix_blocks
        payload = None

        local_out = local_lse = packed = None
        compute_ok = 1
        compute_error = None
        try:
            if lease is None:
                self._native(impl, query, metadata, output, *native_inputs)
            elif rank == lease.owner:
                self._other_requests(impl, query, metadata, output, lease.request.request_index)
                block_size = impl.key_cache.shape[1]
                prefix_tokens = len(lease.request.prefix_blocks) * block_size
                table = metadata.block_tables[
                    lease.request.request_index : lease.request.request_index + 1, len(lease.request.prefix_blocks) :
                ].contiguous()
                # Removing a historical prefix shifts both query positions
                # and key positions equally. Native bottom-right causal FIA
                # therefore preserves all MTP query visibility in one call.
                out, lse = partial_attention(
                    selected_query,
                    impl.key_cache,
                    impl.value_cache,
                    table,
                    lease.request.seq_len - prefix_tokens,
                    impl,
                    causal=lease.request.query_len > 1,
                    causal_mask=getattr(metadata, "attn_mask", None),
                )
                local_out = out.reshape(lease.request.query_len, impl.num_heads, impl.head_size)
                local_lse = lse.reshape(lease.request.query_len, impl.num_heads, 1)
            else:
                prefix_count = len(lease.request.prefix_blocks)
                prefix_tokens = prefix_count * self.resident.kv.shape[2]
                table = torch.arange(prefix_count, dtype=torch.int32, device=query.device).unsqueeze(0)
                # Every delegated key is older than the earliest MTP query,
                # so the whole verification batch can attend it without a mask.
                out, lse = partial_attention(
                    helper_query,
                    self.resident.kv[0],
                    self.resident.kv[1],
                    table,
                    prefix_tokens,
                    impl,
                )
                packed = torch.cat(
                    (
                        out.reshape(lease.request.query_len, impl.num_heads, impl.head_size).float(),
                        lse.reshape(lease.request.query_len, impl.num_heads, 1).float(),
                    ),
                    dim=-1,
                ).contiguous()
                output.zero_()
        except RuntimeError as exc:
            compute_ok = 0
            compute_error = str(exc)
        if not all(s[0] for s in self._states([compute_ok], group)):
            self._fail_collectively("attention", compute_error, group)
        merge_ok = 1
        merge_error = None
        for pair in frame.leases:
            if rank == pair.helper:
                self._transfer([packed], pair.owner, True, group)
            elif rank == pair.owner:
                self._transfer([remote_result], pair.helper, False, group)
                try:
                    merged = merge_attention_partials(
                        local_out, local_lse, remote_result[..., :-1], remote_result[..., -1:]
                    )
                    index = pair.request.request_index
                    start = 0 if index == 0 else metadata.actual_seq_lengths_q[index - 1]
                    target = output[start : start + pair.request.query_len]
                    target.copy_(merged.reshape_as(target).to(output.dtype))
                except RuntimeError as exc:
                    merge_ok = 0
                    merge_error = str(exc)
        # Prevent an uninvolved rank from entering MoE while a pair is still
        # issuing attention P2P operations on an overlapping communicator.
        if not all(s[0] for s in self._states([merge_ok], group)):
            self._fail_collectively("output merge", merge_error, group)
        return output

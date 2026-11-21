# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Deterministic, request-local leases for borrowing idle DP attention workers."""

from dataclasses import dataclass


class DynamicCPConfig:
    def __init__(self, config: dict | None = None):
        if config is None:
            config = {}
        if not isinstance(config, dict):
            raise ValueError("dynamic_cp_config must be a dict")
        defaults = dict(enabled=False, min_context_tokens=32768, cache_budget_mb=1024, max_pairs=1)
        if set(config) - defaults.keys():
            raise ValueError(f"Unknown dynamic_cp_config keys: {sorted(set(config) - defaults.keys())}")
        for key, default in defaults.items():
            setattr(self, key, config.get(key, default))
        if type(self.enabled) is not bool:
            raise ValueError("dynamic_cp_config.enabled must be a bool")
        for key in ("min_context_tokens", "cache_budget_mb", "max_pairs"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"dynamic_cp_config.{key} must be a positive integer")


def validate_dynamic_cp(config: DynamicCPConfig, vllm_config) -> None:
    if not config.enabled:
        return
    parallel = vllm_config.parallel_config
    model = vllm_config.model_config
    if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
        raise ValueError("Dynamic DP CP currently requires TP=1 and PP=1")
    if parallel.decode_context_parallel_size != 1 or parallel.prefill_context_parallel_size != 1:
        raise ValueError("Dynamic DP CP cannot be combined with fixed DCP/PCP")
    if parallel.data_parallel_size <= 1 or parallel.nnodes_within_dp != 1:
        raise ValueError("Dynamic DP CP requires single-node internal DP > 1")
    if getattr(parallel, "data_parallel_external_lb", False) or getattr(parallel, "data_parallel_hybrid_lb", False):
        raise ValueError("Dynamic DP CP requires internal DP lifecycle synchronization")
    if vllm_config.use_v2_model_runner:
        raise ValueError("Dynamic DP CP currently requires model runner v1")
    if not model.enforce_eager:
        raise ValueError("Dynamic DP CP currently requires eager execution for the target model")
    speculative = vllm_config.speculative_config
    if speculative is not None and speculative.method != "mtp":
        raise ValueError("Dynamic DP CP currently supports no speculation or MTP only")
    if getattr(model.hf_text_config, "model_type", None) not in ("qwen3_moe", "qwen3_5_moe", "qwen3_5_moe_text"):
        raise ValueError("Dynamic DP CP currently supports Qwen3/Qwen3.5 MoE Full Attention only")
    if vllm_config.cache_config.cache_dtype not in ("auto", "float16", "bfloat16"):
        raise ValueError("Dynamic DP CP requires unquantized FP16/BF16 KV cache")
    if vllm_config.lora_config is not None:
        raise ValueError("Dynamic DP CP does not yet support LoRA")


@dataclass(frozen=True)
class CPCandidate:
    request_id: str
    request_index: int
    seq_len: int
    query_len: int
    prefix_blocks: tuple[int, ...]
    block_size: int = 128


@dataclass(frozen=True)
class CPWorkload:
    rank: int
    idle: bool
    candidates: tuple[CPCandidate, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class CPLease:
    owner: int
    helper: int
    request: CPCandidate


def choose_cp_leases(
    workloads: list[CPWorkload], max_pairs: int, previous: tuple[CPLease, ...] = ()
) -> tuple[CPLease, ...]:
    """Never suspend local work to create a helper; retain live leases first."""
    idle = {w.rank for w in workloads if w.idle}
    candidates = {(w.rank, c.request_id): c for w in workloads for c in w.candidates if c.prefix_blocks}
    leases = []
    owners = set()
    for old in previous:
        key = old.owner, old.request.request_id
        if old.helper in idle and key in candidates and old.owner not in owners:
            leases.append(CPLease(old.owner, old.helper, candidates[key]))
            idle.remove(old.helper)
            owners.add(old.owner)
            if len(leases) == max_pairs:
                return tuple(leases)
    for (owner, _), candidate in sorted(candidates.items(), key=lambda item: (-item[1].seq_len, item[0])):
        if not idle or len(leases) >= max_pairs:
            break
        if owner in owners:
            continue
        helper = min(idle)
        idle.remove(helper)
        owners.add(owner)
        leases.append(CPLease(owner, helper, candidate))
    return tuple(sorted(leases, key=lambda lease: (lease.owner, lease.helper)))


def reusable_prefix_blocks(old_key, old_blocks: tuple[int, ...], new_key, new_blocks: tuple[int, ...]) -> int:
    """A recycled physical block or a different request invalidates the replica."""
    if old_key != new_key:
        return 0
    count = min(len(old_blocks), len(new_blocks))
    return count if old_blocks[:count] == new_blocks[:count] else 0

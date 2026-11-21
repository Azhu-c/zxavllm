# SPDX-License-Identifier: Apache-2.0
"""Fresh, serialized Decode DP routing using DyntraLB's logical block unit."""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
from itertools import chain
import math
import os
import time
from typing import Any


def enabled() -> bool:
    return os.getenv("VLLM_ASCEND_KV_BLOCKS_ROUTING", "0") == "1"


def route_log_enabled() -> bool:
    # Native routing remains untouched by default. Explicit LOG=1 enables
    # diagnostic observation even when KV-aware routing is disabled.
    return os.getenv("VLLM_ASCEND_KV_BLOCKS_LOG", "1" if enabled() else "0") == "1"


def step_log_enabled() -> bool:
    return os.getenv("VLLM_ASCEND_KV_BLOCKS_STEP_LOG", "0") == "1"


def extension_active() -> bool:
    return enabled() or route_log_enabled() or step_log_enabled()


def timeout_seconds() -> float:
    value = float(os.getenv("VLLM_ASCEND_KV_BLOCKS_TIMEOUT_S", "30"))
    if not math.isfinite(value) or value <= 0:
        raise ValueError("VLLM_ASCEND_KV_BLOCKS_TIMEOUT_S must be positive")
    return value


@dataclass(frozen=True)
class KVHeadroom:
    reserve_fraction: float
    growth_tokens: int
    report_interval_ms: int = 100


def headroom_config(config: Any) -> KVHeadroom | None:
    scheduler = (config.additional_config or {}).get("scheduler_config") or {}
    values = scheduler.get("kv_blocks_headroom")
    if values is None:
        return None
    keys = {"reserve_fraction", "growth_tokens", "report_interval_ms"}
    if not isinstance(values, dict) or set(values) - keys:
        raise ValueError(f"kv_blocks_headroom accepts {sorted(keys)}")
    fraction = values.get("reserve_fraction", 0.05)
    growth = values.get("growth_tokens", 16)
    if type(fraction) not in (int, float) or not 0 <= fraction < 1:
        raise ValueError("kv_blocks_headroom.reserve_fraction must be in [0, 1)")
    if type(growth) is not int or growth < 0:
        raise ValueError("kv_blocks_headroom.growth_tokens must be a nonnegative integer")
    interval = values.get("report_interval_ms", 100)
    if type(interval) is not int or interval <= 0:
        raise ValueError("kv_blocks_headroom.report_interval_ms must be a positive integer")
    return KVHeadroom(fraction, growth, interval)


def estimate_kv_blocks(scheduler: Any, request_id: str, tokens: int,
                       computed: int, growth_tokens: int) -> int:
    manager = scheduler.kv_cache_manager
    main_tokens = min(tokens + growth_tokens, manager.max_model_len)
    with_lookahead = min(main_tokens + scheduler.num_lookahead_tokens, manager.max_model_len)
    return manager.coordinator.get_num_blocks_to_allocate(
        request_id=request_id,
        num_tokens=with_lookahead,
        new_computed_blocks=manager.empty_kv_cache_blocks.blocks,
        num_encoder_tokens=0,
        total_computed_tokens=computed,
        num_local_computed_tokens=computed,
        num_tokens_main_model=main_tokens,
        apply_admission_cap=True,
    )


def estimate_request_kv_blocks(scheduler: Any, request: Any, growth_tokens: int) -> int:
    return estimate_kv_blocks(
        scheduler, request.request_id, request.num_tokens, request.num_computed_tokens, growth_tokens
    )


def kv_pool_floor(scheduler: Any, policy: KVHeadroom) -> tuple[int, int]:
    manager = scheduler.kv_cache_manager
    pool = manager.block_pool
    floor = math.ceil((pool.num_gpu_blocks - 1) * policy.reserve_fraction)
    return pool.get_num_free_blocks(), max(floor, manager.watermark_blocks)


def kv_pool_headroom(scheduler: Any, policy: KVHeadroom) -> tuple[int, int]:
    free, floor = kv_pool_floor(scheduler, policy)
    running_growth = sum(
        estimate_request_kv_blocks(scheduler, request, policy.growth_tokens)
        for request in scheduler.running
    )
    return free, floor + running_growth


def local_admission_reserve(scheduler: Any, headroom_reserve: int) -> int:
    # Let one request use the entire pool on an idle rank so a prompt that
    # legitimately needs more than the watermarked budget can still progress.
    if not scheduler.running and not scheduler._inflight_prefills:
        return 0
    return headroom_reserve + scheduler._inflight_prefill_reserved_blocks()


def validate_config(config: Any, client_count: int) -> None:
    parallel = config.parallel_config
    if client_count != 1:
        raise ValueError("KV-block routing requires --api-server-count 1")
    if (
        parallel.data_parallel_size <= 1
        or parallel.data_parallel_external_lb
        or parallel.data_parallel_hybrid_lb
    ):
        raise ValueError("KV-block routing requires internal DP over all Decode ranks")
    transfer = config.kv_transfer_config
    if transfer is None or transfer.kv_role != "kv_consumer":
        raise ValueError("Enable KV-block routing only on the P/D Decode kv_consumer")
    headroom_config(config)
    timeout_seconds()


def snapshot(scheduler: Any, rank: int) -> dict[str, Any]:
    # Exactly get_dyntra_lb_request_block_num(), including the choice of
    # cache_config.block_size rather than scheduler.block_size for hybrid models.
    block_size = scheduler.cache_config.block_size
    if type(block_size) is not int or block_size <= 0:
        raise RuntimeError(f"Invalid DyntraLB block size: {block_size}")

    def blocks(request: Any) -> int:
        return (len(request.all_token_ids) + block_size - 1) // block_size

    return {
        "version": 1,
        "rank": rank,
        "block_size": block_size,
        "running_blocks": [blocks(request) for request in scheduler.running],
        # Include remote-KV/grammar waits and preempted requests: they still
        # belong to this rank. No max_slots clipping or admissibility filter.
        "waiting_blocks": [
            blocks(request)
            for request in chain(scheduler.waiting, scheduler.skipped_waiting)
        ],
    }


def validate_states(states: list[Any], ranks: list[int]) -> list[int]:
    if not ranks or len(states) != len(ranks) or len(set(ranks)) != len(ranks):
        raise RuntimeError("Incomplete KV-block global snapshot")
    totals: list[int] = []
    block_sizes: set[int] = set()
    for expected_rank, state in zip(ranks, states):
        if (
            not isinstance(state, dict)
            or state.get("version") != 1
            or state.get("rank") != expected_rank
        ):
            raise RuntimeError(f"Invalid KV-block snapshot for rank {expected_rank}")
        block_size = state.get("block_size")
        if type(block_size) is not int or block_size <= 0:
            raise RuntimeError("Invalid KV-block snapshot block size")
        block_sizes.add(block_size)
        total = 0
        for key in ("running_blocks", "waiting_blocks"):
            values = state.get(key)
            if not isinstance(values, list) or any(
                type(value) is not int or value < 0 for value in values
            ):
                raise RuntimeError(f"Invalid KV-block snapshot {key}")
            total += sum(values)
        totals.append(total)
    if len(block_sizes) != 1:
        raise RuntimeError("Decode ranks have different KV-block load units")
    return totals


def choose_rank(totals: list[int], start: int, preferred: int | None,
                pressure: list[int] | None = None) -> int:
    if preferred is not None:
        if type(preferred) is not int or not 0 <= preferred < len(totals):
            raise ValueError(f"Invalid data_parallel_rank: {preferred}")
        return preferred
    ranks = ((start + offset) % len(totals) for offset in range(len(totals)))
    if pressure is None:
        return min(ranks, key=totals.__getitem__)
    return min(ranks, key=lambda rank: (pressure[rank], totals[rank]))


class LocalKVReport:
    """A Decode rank's latest local load and KV pressure."""

    def __init__(self, policy: KVHeadroom):
        self.policy = policy
        self.last_sample = -math.inf
        self.state: dict[str, int] | None = None
        self.waiter: Future | None = None

    def observe(self, scheduler: Any, received: int) -> None:
        now = time.monotonic()
        free, floor = kv_pool_floor(scheduler, self.policy)
        queues = (scheduler.running, scheduler.waiting, scheduler.skipped_waiting)
        pressure = int(free <= floor or (any(queues) and getattr(scheduler, "_lb_kv_waiting_for_blocks", False)))
        became_empty = self.state is not None and self.state["requests"] > 0 and not any(queues)
        if (
            self.state is not None
            and pressure == self.state["pressure"]
            and not became_empty
            and now - self.last_sample < self.policy.report_interval_ms / 1000
        ):
            return
        self.last_sample = now
        block_size = scheduler.cache_config.block_size
        data = {
            "blocks": sum((req.num_tokens + block_size - 1) // block_size for req in chain(*queues)),
            "requests": sum(map(len, queues)),
            "pressure": pressure,
            "received": received,
            "block_size": block_size,
        }
        if self.state is not None and all(self.state[key] == value for key, value in data.items()):
            return
        data["revision"] = 1 if self.state is None else self.state["revision"] + 1
        self.state = data
        if self.waiter is not None:
            waiter, self.waiter = self.waiter, None
            waiter.set_result(data)

    def listen(self, revision: int) -> dict[str, int] | Future:
        if self.state is None:
            raise RuntimeError("Local KV report must be sampled before subscribing")
        if self.state["revision"] != revision:
            return self.state
        if self.waiter is not None:
            raise RuntimeError("Local KV report already has a pending subscriber")
        self.waiter = Future()
        return self.waiter

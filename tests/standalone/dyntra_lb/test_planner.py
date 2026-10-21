# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random

import pytest
from source_modules import DyntraLBConfig, DyntraLBPausePolicy, balance_load


def test_swap_must_not_increase_global_step_latency():
    # The old distance-to-average heuristic replaces 1 with 150, increasing
    # the global maximum from 100 to 150 while the bubble gets smaller.
    plan = balance_load([([100], 1), ([1, 150], 1)], 2, threshold=1, min_throughput_gain=0)
    assert not any(p["out_blk"] or p["in_blk"] for p in plan)


def test_legacy_absolute_and_relative_thresholds():
    ranks = [([3], 1), ([1, 2], 1)]
    assert balance_load(ranks, 2, threshold=1, min_throughput_gain=0)[1]["in_blk"] == [2]
    assert not balance_load(ranks, 2, threshold=0.5, min_throughput_gain=0)[1]["in_blk"]


def test_pause_requires_minimum_throughput_gain():
    ranks = [([4000, 1000], 2), ([1000], 1)]
    assert balance_load(ranks, 2, threshold=1)[0]["out_blk"] == [4000]
    assert balance_load(ranks, 2, threshold=1, min_throughput_gain=0.99)[0]["out_blk"] == []


def test_latency_coefficients_are_configurable():
    ranks = [([4000, 1000], 2), ([1000], 1)]
    assert not balance_load(ranks, 2, step_latency_fixed_us=10_000_000)[0]["out_blk"]


def test_fixed_overhead_dominates_small_batches():
    assert not balance_load([([60, 40], 2), ([60], 1)], 2)[0]["out_blk"]


def test_indexed_plan_identifies_duplicate_block_counts():
    plan = balance_load([([5000, 5000, 1], 2), ([1], 1)], 2, max_num_seqs=2, with_request_indices=True)
    assert len(plan[0]["out_slots"]) == 1
    assert plan[0]["out_slots"][0] in (0, 1)
    assert plan[0]["in_slots"] == [2]


@pytest.mark.parametrize("seed", range(100))
def test_plan_preserves_local_identity_and_capacity(seed):
    rng = random.Random(seed)
    cap = 4
    ranks = []
    for _ in range(4):
        run = rng.randrange(cap + 1)
        wait = rng.randrange(cap + 1)
        ranks.append(([rng.choice([1, 2, 100, 4000]) for _ in range(run + wait)], run))
    plans = balance_load(ranks, 4, max_num_seqs=cap, with_request_indices=True)
    for (blocks, split), plan in zip(ranks, plans):
        out_slots, in_slots = plan["out_slots"], plan["in_slots"]
        assert len(set(out_slots)) == len(out_slots)
        assert len(set(in_slots)) == len(in_slots)
        assert all(0 <= i < split for i in out_slots)
        assert all(split <= i < len(blocks) for i in in_slots)
        assert split - len(out_slots) + len(in_slots) <= cap
        assert plan["out_blk"] == [blocks[i] for i in out_slots]
        assert plan["in_blk"] == [blocks[i] for i in in_slots]


def test_pause_budget_and_resume_protection():
    policy = DyntraLBPausePolicy(max_pause_steps=4, min_run_steps=4)
    policy.pause("long", 10)
    policy.pause("long", 11)  # Repeated plans cannot reset the deadline.
    assert not policy.overdue(13)
    assert policy.overdue(14) == ["long"]
    policy.resume("long", 14)
    assert not policy.can_pause("long", 17)
    assert policy.can_pause("long", 18)
    assert not policy.overdue(100)
    policy.discard("long")
    assert not policy.paused_at and not policy.protected_until


def test_default_latency_guards():
    config = DyntraLBConfig()
    assert (config.max_pause_steps, config.min_run_steps, config.balance_interval) == (4, 4, 1)
    assert config.min_throughput_gain == 0.02


@pytest.mark.parametrize(
    "key",
    [
        "bubble_threshold",
        "min_throughput_gain",
        "step_latency_fixed_us",
        "step_latency_per_block_us",
    ],
)
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), True, "1"])
def test_reject_nonfinite_or_nonnumeric_costs(key, bad):
    with pytest.raises(ValueError):
        DyntraLBConfig({key: bad})


@pytest.mark.parametrize("key", ["balance_interval", "max_pause_steps", "min_run_steps"])
@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_reject_invalid_step_counts(key, bad):
    with pytest.raises(ValueError):
        DyntraLBConfig({key: bad})


@pytest.mark.parametrize("bad", [-0.01, 1, 2])
def test_reject_invalid_gain(bad):
    with pytest.raises(ValueError):
        DyntraLBConfig({"min_throughput_gain": bad})


def test_admission_only_mode_never_pauses_original_running_requests():
    plans = balance_load(
        [([5000, 5000, 1], 2), ([1], 1)], 2, max_num_seqs=2, max_pauses_per_rank=0, with_request_indices=True
    )
    assert all(not p["out_slots"] for p in plans)


def test_pause_budget_is_per_rank_per_plan():
    plans = balance_load(
        [([5000, 5000, 5000, 1], 3), ([1], 1)], 2, max_num_seqs=3, with_request_indices=True, max_pauses_per_rank=1
    )
    assert all(len(p["out_slots"]) <= 1 for p in plans)


def test_idle_dp_cp_candidate_is_kept_running_instead_of_paused():
    ranks = [([4000, 1000], 2), ([], 0), ([1000], 1)]
    baseline = balance_load(ranks, 3, max_num_seqs=2)
    protected = balance_load(ranks, 3, max_num_seqs=2, cp_min_blocks=700)
    assert 4000 in baseline[0]["out_blk"]
    assert 4000 not in protected[0]["out_blk"]


def test_cp_protection_requires_an_actually_unfilled_rank():
    ranks = [([4000, 1000], 2), ([1000], 1)]
    assert balance_load(ranks, 2, max_num_seqs=2, cp_min_blocks=700)[0]["out_blk"] == [4000]

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from runtime_contracts import RequestStatus, load_scheduler, request
from source_modules import DyntraLBConfig, DyntraLBFirstTokenPolicy, balance_load_original
from test_runtime_contracts import make_engine


def scheduler_for(profile="ttft", **options):
    scheduler_class, _ = load_scheduler()
    return scheduler_class(
        vllm_config=SimpleNamespace(
            additional_config={"scheduler_config": {"dyntra_lb_config": {"policy_profile": profile, **options}}}
        )
    )


def first_token_request(req_id, output=0, status=RequestStatus.WAITING):
    req = request(req_id, status=status)
    req.num_output_tokens = output
    return req


def freeze(scheduler):
    scheduler.modifications = {"out_blk": [], "in_blk": [], "freeze": True}
    scheduler._apply_load_balance_modifications()


def test_frozen_first_token_request_gets_an_admission_attempt_after_four_deferrals():
    scheduler = scheduler_for()
    req = first_token_request("new")
    scheduler._lb_admission_candidates = [req]
    for step in range(4):
        scheduler.current_step = step
        freeze(scheduler)
        assert not scheduler._can_admit_waiting_request(req)
    scheduler.current_step = 4
    freeze(scheduler)
    assert scheduler._can_admit_waiting_request(req)
    # Opening the gate neither adds the request nor allocates any KV/slot.
    assert not scheduler.running
    assert req.status is RequestStatus.WAITING


def test_deferrals_are_counted_once_per_step_not_once_per_call():
    policy = DyntraLBFirstTokenPolicy(2, 1)
    for _ in range(10):
        assert not policy.can_admit("new", 0, False, False)
    assert not policy.can_admit("new", 1, False, False)
    assert policy.can_admit("new", 2, False, False)


def test_per_rank_budget_limits_bypass_without_resetting_other_requests():
    policy = DyntraLBFirstTokenPolicy(1, 1)
    assert not policy.can_admit("a", 0, False, False)
    assert not policy.can_admit("b", 0, False, False)
    assert policy.can_admit("a", 1, False, False)
    assert policy.can_admit("a", 1, False, False)
    assert not policy.can_admit("b", 1, False, False)
    assert policy.can_admit("b", 2, False, False)


def test_first_token_history_survives_selection_that_could_not_allocate_resources():
    policy = DyntraLBFirstTokenPolicy(1, 1)
    assert not policy.can_admit("a", 0, False, False)
    assert policy.can_admit("a", 1, True, False)
    assert policy.can_admit("a", 2, False, False)
    assert not policy.can_admit("a", 3, False, True)
    assert "a" not in policy.deferred


def test_cleanup_does_not_refund_bypass_budget_or_admit_a_reused_id():
    policy = DyntraLBFirstTokenPolicy(1, 1)
    assert not policy.can_admit("a", 0, False, False)
    assert not policy.can_admit("b", 0, False, False)
    assert policy.can_admit("a", 1, False, False)
    policy.discard("a")
    assert not policy.can_admit("a", 1, False, False)
    assert not policy.can_admit("b", 1, False, False)
    assert policy.can_admit("b", 2, False, False)


def test_generating_requests_never_bypass_the_original_gate():
    scheduler = scheduler_for()
    req = first_token_request("decode", output=1, status=RequestStatus.PREEMPTED)
    for step in range(10):
        scheduler.current_step = step
        freeze(scheduler)
        assert not scheduler._can_admit_waiting_request(req)


def test_ttft_protects_running_prefill_and_pending_first_token_but_not_decode():
    scheduler = scheduler_for()
    first = first_token_request("first", status=RequestStatus.RUNNING)
    decode = first_token_request("decode", output=1, status=RequestStatus.RUNNING)
    scheduler.running = [first, decode]
    # Duplicate costs must not cause the first-token request to be paused.
    scheduler.modifications = {"out_blk": [2], "in_blk": [], "freeze": True}
    scheduler._apply_load_balance_modifications()
    assert scheduler.running == [first]
    assert list(scheduler.waiting) == [decode]
    assert scheduler.modifications is None


def test_original_profile_keeps_first_token_pause_and_unbounded_lb_gate():
    scheduler = scheduler_for("original")
    req = first_token_request("first", status=RequestStatus.RUNNING)
    scheduler.running = [req]
    scheduler.modifications = {"out_blk": [2], "in_blk": [], "freeze": True}
    scheduler._apply_load_balance_modifications()
    assert not scheduler.running
    scheduler.current_step = 100
    freeze(scheduler)
    assert not scheduler._can_admit_waiting_request(req)


def test_request_cleanup_removes_first_token_history_for_reused_ids():
    scheduler = scheduler_for()
    req = first_token_request("reuse")
    freeze(scheduler)
    assert not scheduler._can_admit_waiting_request(req)
    assert "reuse" in scheduler._lb_first_token_policy.deferred
    scheduler._free_request(req)
    assert not scheduler._lb_first_token_policy.deferred


def test_priority_queue_order_is_preserved_and_running_requests_are_not_evicted():
    scheduler = scheduler_for()
    running = first_token_request("running", output=1, status=RequestStatus.RUNNING)
    waiting = first_token_request("waiting")
    scheduler.running = [running]
    scheduler.waiting.items = [waiting]
    scheduler._lb_first_token_policy.deferred["waiting"] = (4, 0)
    scheduler.current_step = 5
    freeze(scheduler)
    assert scheduler._can_admit_waiting_request(waiting)
    assert scheduler.running == [running]
    assert list(scheduler.waiting) == [waiting]


@pytest.mark.parametrize("profile", ["original", "ttft"])
def test_engine_uses_original_planner_and_block_plan_for_comparison(profile):
    engine, dist = make_engine({"policy_profile": profile, "mode": "static", "start_step": 0})
    running = first_token_request("run", output=1, status=RequestStatus.RUNNING)
    waiting = first_token_request("wait")
    engine.scheduler.running = [running]

    def gather(outputs, local, group):
        np.copyto(outputs[0], local)
        outputs[1][:] = [1, 0, 0, 0, 1, 0]

    dist.all_gather.side_effect = gather
    engine._do_lb_allgather([waiting])
    assert engine._balance_load is balance_load_original
    expected = balance_load_original([([2, 2, 0, 0], 1), ([1, 0, 0, 0], 1)], 2, max_num_seqs=2)
    assert engine.scheduler.modifications == expected[0]


@pytest.mark.parametrize("key", ["first_token_max_lb_wait_steps", "first_token_max_admissions_per_step"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "4"])
def test_invalid_first_token_limits_fail_at_startup(key, value):
    with pytest.raises(ValueError):
        DyntraLBConfig({key: value})


@pytest.mark.parametrize("value", [None, [], True, "unknown"])
def test_invalid_profile_fails_at_startup(value):
    with pytest.raises(ValueError):
        DyntraLBConfig({"policy_profile": value})


@pytest.mark.parametrize("profile", ["original", "ttft"])
def test_cp_cannot_accidentally_change_original_ttft_comparisons(profile):
    with pytest.raises(ValueError, match="require dynamic_cp_config.enabled=false"):
        DyntraLBConfig({"enabled": True, "policy_profile": profile, "dynamic_cp_config": {"enabled": True}})


def test_default_profile_preserves_previous_measured_tpot_revision():
    assert DyntraLBConfig().policy_profile == "tpot"


@pytest.mark.parametrize(
    "case", json.loads((Path(__file__).parent / "fixtures/original_planner_cases.json").read_text())["cases"]
)
def test_original_planner_reproduces_pinned_pr_outputs(case):
    assert balance_load_original(case["ranks"], len(case["ranks"]), **case["options"]) == case["expected"]

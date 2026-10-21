# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
from runtime_contracts import RequestStatus, SchedulingPolicy, load_engine, load_scheduler, request


def make_scheduler():
    scheduler_class, _ = load_scheduler()
    return scheduler_class(vllm_config=SimpleNamespace(additional_config={}))


def test_request_identity_survives_block_boundary_and_duplicate_costs():
    scheduler = make_scheduler()
    a, b = request("a"), request("b")
    scheduler.running = [a, b]
    scheduler.modifications = {
        "out_blk": [2],
        "in_blk": [],
        "freeze": True,
        "out_req_ids": ["a"],
        "in_req_ids": [],
    }
    a.all_token_ids.extend(range(8))  # Cost is now three blocks.
    scheduler._apply_load_balance_modifications()
    assert scheduler.running == [b]
    assert scheduler._lb_paused_req_ids == {"a"}
    assert scheduler.modifications is None
    scheduler._apply_load_balance_modifications()
    assert scheduler.running == [b]  # The old two-block plan cannot pause b.
    assert scheduler._can_admit_waiting_request(a)


def test_overdue_request_bypasses_freeze_and_rotates_a_full_batch():
    scheduler = make_scheduler()
    scheduler.max_num_running_reqs = 1
    old, new = request("old"), request("new")
    scheduler._lb_pause_request(old, 0)
    scheduler.running = [new]
    scheduler.current_step = 4
    scheduler.modifications = {"out_blk": [], "in_blk": [], "freeze": True, "out_req_ids": [], "in_req_ids": []}
    scheduler._apply_load_balance_modifications()
    assert not scheduler.running
    assert scheduler._can_admit_waiting_request(old)
    assert list(scheduler.skipped_waiting)[0] is old
    assert not scheduler.lb_freeze


def test_resumed_request_is_protected_and_cleanup_removes_state():
    scheduler = make_scheduler()
    req = request("long")
    scheduler._lb_pause_request(req, 0)
    scheduler.current_step = 4
    scheduler._update_after_schedule(SimpleNamespace(num_scheduled_tokens={"long": 1}))
    req.status = RequestStatus.RUNNING
    scheduler.running = [req]
    for step in range(5, 8):
        scheduler.current_step = step
        scheduler.modifications = {"out_blk": [2], "in_blk": [], "freeze": True, "out_req_ids": ["long"]}
        scheduler._apply_load_balance_modifications()
        assert scheduler.running == [req]
    scheduler._free_request(req)
    assert not scheduler._lb_pause_policy.protected_until


def test_overdue_inflight_output_does_not_evict_running_work():
    scheduler = make_scheduler()
    scheduler.max_num_running_reqs = 1
    paused, running = request("paused"), request("running")
    scheduler._lb_pause_request(paused, 0)
    paused.num_in_flight_tokens = 1
    scheduler.running = [running]
    scheduler.current_step = 4
    scheduler._apply_load_balance_modifications()
    assert scheduler.running == [running]


def make_engine(config):
    scheduler_class, block_num = load_scheduler()
    engine_class, dist = load_engine(block_num)
    engine = engine_class()
    vconfig = SimpleNamespace(
        additional_config={"scheduler_config": {"dyntra_lb_config": config}},
        scheduler_config=SimpleNamespace(max_num_seqs=2),
    )
    engine._init_data_parallel(vconfig)
    engine.scheduler = scheduler_class(vllm_config=vconfig)
    engine.current_wave = 0
    return engine, dist


def test_collective_cadence_and_activation_window():
    engine, dist = make_engine({"mode": "dynamic", "start_step": 8, "end_step": 20, "balance_interval": 4})
    for step in [1, 4, 7, 9, 20, 24]:
        engine.step_counter = step
        engine.run_balance_load()
    dist.all_reduce.assert_not_called()
    engine.step_counter = 8
    engine.run_balance_load()
    dist.all_reduce.assert_called_once()


def test_pending_long_request_is_not_lost_on_skipped_steps():
    engine, dist = make_engine({"mode": "dynamic", "start_step": 0, "balance_interval": 4})
    engine._lb_pending_long_req = True
    engine._lb_pending_long_req_blk = 800
    engine.step_counter = 1
    engine.run_balance_load()
    assert engine._lb_pending_long_req
    engine.scheduler.prepare_dyntra_lb_step = MagicMock(return_value=[])
    engine._do_lb_allgather = MagicMock(return_value=False)
    engine.step_counter = 4
    engine.run_balance_load()
    assert engine._lb_dynamic_enable
    engine._do_lb_allgather.assert_called_once()
    dist.all_reduce.assert_called_once()


def test_gather_resolves_snapshot_positions_to_local_request_ids():
    engine, dist = make_engine({"mode": "static", "start_step": 0})
    a, b = request("a", 40_000), request("b", 40_000)
    waiting = request("wait", 8, RequestStatus.WAITING)
    engine.scheduler.running = [a, b]
    remote = [1, 0, 0, 0, 1, 0]

    def gather(outputs, local, group):
        np.copyto(outputs[0], local)
        outputs[1][:] = remote

    dist.all_gather.side_effect = gather
    engine._do_lb_allgather([waiting])
    plan = engine.scheduler.modifications
    assert len(plan["out_req_ids"]) == 1
    assert plan["out_req_ids"][0] in ("a", "b")
    assert plan["in_req_ids"] == ["wait"]


def test_priority_deadline_does_not_evict_higher_priority_work():
    class RankedRequest(SimpleNamespace):
        def __lt__(self, other):
            return (self.priority, self.arrival_time) < (other.priority, other.arrival_time)

    scheduler = make_scheduler()
    scheduler.policy = SchedulingPolicy.PRIORITY
    scheduler.max_num_running_reqs = 1
    low = RankedRequest(**vars(request("low")), priority=10, arrival_time=0)
    high = RankedRequest(**vars(request("high")), priority=1, arrival_time=1)
    scheduler._lb_pause_request(low, 0)
    scheduler.running = [high]
    scheduler.current_step = 4
    scheduler._apply_load_balance_modifications()
    assert scheduler.running == [high]


def test_fcfs_deadline_resumes_oldest_pause_first():
    scheduler = make_scheduler()
    oldest, newer = request("oldest"), request("newer")
    scheduler._lb_pause_request(oldest, 0)
    scheduler.current_step = 1
    scheduler._lb_pause_request(newer, 0)
    scheduler.current_step = 5
    scheduler._apply_load_balance_modifications()
    assert list(scheduler.skipped_waiting) == [oldest, newer]


def test_admission_identity_survives_cost_growth():
    scheduler = make_scheduler()
    waiting = request("waiting", status=RequestStatus.WAITING)
    scheduler.modifications = {
        "out_blk": [],
        "in_blk": [2],
        "freeze": False,
        "out_req_ids": [],
        "in_req_ids": ["waiting"],
    }
    waiting.all_token_ids.extend(range(8))
    scheduler._apply_load_balance_modifications()
    assert scheduler._can_admit_waiting_request(waiting)

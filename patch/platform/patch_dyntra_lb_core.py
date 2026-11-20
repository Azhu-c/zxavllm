#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np
import torch
import torch.distributed as dist
import vllm.v1.engine.core as _engine_core_mod
import vllm.v1.request as _request_module
from vllm.logger import logger
from vllm.v1.engine.core import DPEngineCoreProc, EngineCoreProc
from vllm.v1.request import Request

import vllm_ascend.patch.platform.patch_balance_schedule as _balance_patch
from vllm_ascend.ascend_config import get_ascend_config, init_ascend_config
from vllm_ascend.core.dyntra_lb_config import DyntraLBConfig
from vllm_ascend.core.dyntra_lb_planner import balance_load
from vllm_ascend.core.dyntra_lb_scheduler import (
    diagnostics_enabled,
    get_dyntra_lb_request_block_num,
)
from vllm_ascend.core.kv_blocks_routing import (
    estimate_request_kv_blocks,
    headroom_config,
    kv_pool_headroom,
    local_admission_reserve,
)

from vllm_ascend.core.dyntra_lb_wire import decode_snapshot, encode_snapshot

RequestStatus = _request_module.RequestStatus
_ORIGINAL_HAS_GLOBAL_UNFINISHED_REQS = DPEngineCoreProc._has_global_unfinished_reqs


def _get_dyntra_lb_config(vllm_config) -> DyntraLBConfig:
    try:
        return get_ascend_config().scheduler_config.dyntra_lb_config
    except Exception:
        pass
    additional_config = getattr(vllm_config, "additional_config", None) or {}
    scheduler_config = additional_config.get("scheduler_config") or {}
    if not isinstance(scheduler_config, dict):
        return DyntraLBConfig()
    dyntra_lb_config = scheduler_config.get("dyntra_lb_config") or {}
    if not isinstance(dyntra_lb_config, dict):
        return DyntraLBConfig()
    return DyntraLBConfig(dyntra_lb_config)


def _dyntra_lb_enabled(vllm_config) -> bool:
    return _get_dyntra_lb_config(vllm_config).enabled


def _print_rank_0(message: str, dp_rank: int, enable_diagnostics: bool) -> None:
    if not enable_diagnostics or dp_rank != 0:
        return
    logger.info("[dyntra_lb] %s", message)


def _print_requests_by_rank(requests_by_rank, dp_rank: int, enable_diagnostics: bool) -> None:
    if not enable_diagnostics or dp_rank != 0:
        return

    logger.info("%s", "=" * 85)
    logger.info("[dyntra_lb] requests_by_rank (DP Workload Status):")
    for rank, (blocks, split_idx) in enumerate(requests_by_rank):
        running = blocks[:split_idx]
        waiting = [block for block in blocks[split_idx:] if block > 0]
        running_items = ", ".join(f"{block:3d}" for block in running)
        waiting_items = ", ".join(f"{block:3d}" for block in waiting)
        running_str = f"Run({len(running)}): [{running_items}]"
        waiting_str = f"Wait({len(waiting)}): [{waiting_items}]"
        logger.info(" DP%d | %-55s | %s", rank, running_str, waiting_str)
    logger.info("%s", "=" * 85)


def _print_modifications(modifications, dp_rank: int, enable_diagnostics: bool) -> None:
    if not enable_diagnostics or dp_rank != 0:
        return

    logger.info("%s", "=" * 60)
    logger.info("[dyntra_lb] modifications:")
    for rank, modification in enumerate(modifications):
        out_items = ", ".join(f"{block:3d}" for block in modification.get("out_blk", []))
        in_items = ", ".join(f"{block:3d}" for block in modification.get("in_blk", []))
        out_str = f"Out: [{out_items}]"
        in_str = f"In: [{in_items}]"
        freeze_str = f"Freeze: {bool(modification.get('freeze', False))}"
        logger.info(" DP%d | %-15s | %-15s | %s", rank, out_str, in_str, freeze_str)
    logger.info("%s", "=" * 60)


def _has_global_unfinished_reqs_with_diagnostics(self, local_unfinished: bool) -> bool:
    if diagnostics_enabled(self.vllm_config):
        logger.info("_has_global_unfinished_reqs | step_counter: %d", self.step_counter)
    return _ORIGINAL_HAS_GLOBAL_UNFINISHED_REQS(self, local_unfinished)


class DyntraLBDPEngineCoreProc(DPEngineCoreProc):
    _balance_load = staticmethod(balance_load)

    def _init_data_parallel(self, vllm_config):
        super()._init_data_parallel(vllm_config)

        ascend_config = init_ascend_config(vllm_config)
        dyntra_lb_config = ascend_config.scheduler_config.dyntra_lb_config
        self._lb_enable_diagnostics = dyntra_lb_config.enable_diagnostics
        self._lb_mode = dyntra_lb_config.mode
        self._lb_start_step = dyntra_lb_config.start_step
        self._lb_end_step = dyntra_lb_config.end_step
        self._lb_threshold = dyntra_lb_config.bubble_threshold
        self._lb_long_req_threshold = dyntra_lb_config.long_req_block_threshold
        self._lb_dynamic_max_step = dyntra_lb_config.dynamic_max_step
        self._lb_balance_interval = dyntra_lb_config.balance_interval
        self._lb_planner_options = {
            "min_throughput_gain": dyntra_lb_config.min_throughput_gain,
            "step_latency_fixed_us": dyntra_lb_config.step_latency_fixed_us,
            "step_latency_per_block_us": dyntra_lb_config.step_latency_per_block_us,
            "step_latency_per_seq_us": dyntra_lb_config.step_latency_per_seq_us,
            "pause_resume_cost_us": dyntra_lb_config.pause_resume_cost_us,
            "gain_horizon_steps": dyntra_lb_config.gain_horizon_steps,
            "min_balance_gain": dyntra_lb_config.min_balance_gain,
            "max_balance_throughput_loss": dyntra_lb_config.max_balance_throughput_loss,
            "with_request_indices": True,
            "max_pauses_per_rank": dyntra_lb_config.max_pauses_per_rank,
        }
        cp_config = dyntra_lb_config.dynamic_cp_config
        if cp_config.enabled:
            block_size = vllm_config.cache_config.block_size
            self._lb_planner_options.update(
                cp_min_blocks=(cp_config.min_context_tokens + block_size - 1) // block_size,
                cp_max_pairs=cp_config.max_pairs,
            )
        self._lb_dynamic_enable = False
        self._lb_dynamic_step = 0
        self._lb_pending_long_req = False
        self._lb_pending_long_req_blk = 0

        _print_rank_0("dyntra_lb_config.enabled = True", self.dp_rank, self._lb_enable_diagnostics)
        _print_rank_0(
            f"dyntra_lb_config.enable_diagnostics = {self._lb_enable_diagnostics}",
            self.dp_rank,
            self._lb_enable_diagnostics,
        )
        _print_rank_0(
            f"dyntra_lb_config.mode = {self._lb_mode}",
            self.dp_rank,
            self._lb_enable_diagnostics,
        )
        _print_rank_0(
            f"dyntra_lb_config.start_step = {self._lb_start_step}",
            self.dp_rank,
            self._lb_enable_diagnostics,
        )
        _print_rank_0(
            f"dyntra_lb_config.end_step = {self._lb_end_step}",
            self.dp_rank,
            self._lb_enable_diagnostics,
        )
        _print_rank_0(
            f"dyntra_lb_config.bubble_threshold = {self._lb_threshold}",
            self.dp_rank,
            self._lb_enable_diagnostics,
        )
        _print_rank_0(
            f"dyntra_lb_config.long_req_block_threshold = {self._lb_long_req_threshold}",
            self.dp_rank,
            self._lb_enable_diagnostics,
        )
        _print_rank_0(
            f"dyntra_lb_config.dynamic_max_step = {self._lb_dynamic_max_step}",
            self.dp_rank,
            self._lb_enable_diagnostics,
        )

        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        dp_size = dist.get_world_size(self.dp_group)
        max_slots = max_num_seqs * 2
        self._lb_dp_size_cached = dp_size
        self._lb_max_slots_cached = max_slots
        self._lb_max_num_seqs = max_num_seqs

        self._lb_headroom = headroom_config(vllm_config)
        self._lb_dynamic_flag_np: np.ndarray = np.zeros(2, dtype=np.int32)
        self._lb_dynamic_flag_t = torch.as_tensor(self._lb_dynamic_flag_np)

    def add_request(self, request: Request, request_wave: int = 0):
        if self._lb_mode == "dynamic":
            blk_num = get_dyntra_lb_request_block_num(self.scheduler, request)
            if blk_num > self._lb_long_req_threshold:
                self._lb_pending_long_req = True
                self._lb_pending_long_req_blk = max(self._lb_pending_long_req_blk, blk_num)
        super().add_request(request, request_wave)

    def run_balance_load(self):
        # Every rank has the same wave/step counter. Never gate a collective on
        # a rank-local request count, which can strand otherwise idle ranks.
        in_window = self.step_counter >= self._lb_start_step and (
            self._lb_end_step < 0 or self.step_counter < self._lb_end_step
        )
        if not in_window or self.step_counter % self._lb_balance_interval != 0:
            self.scheduler.modifications = None
            self.scheduler.lb_freeze = False
            self.scheduler._lb_kv_prefetch_enabled = False
            return
        static_lb_enable = self._lb_mode == "static"
        dynamic_lb_mode = self._lb_mode == "dynamic"
        dynamic_activated_this_step = False

        if dynamic_lb_mode and not self._lb_dynamic_enable:
            self._lb_dynamic_flag_np[0] = int(self._lb_pending_long_req)
            self._lb_dynamic_flag_np[1] = self._lb_pending_long_req_blk
            dist.all_reduce(self._lb_dynamic_flag_t, op=dist.ReduceOp.MAX, group=self.dp_group)
            if self._lb_dynamic_flag_np[0]:
                self._lb_dynamic_enable = True
                self._lb_dynamic_step = 0
                dynamic_activated_this_step = True
                _print_rank_0(
                    "Dynamic LB enabled by long request: "
                    f"blk_num={self._lb_dynamic_flag_np[1]}, "
                    f"threshold={self._lb_long_req_threshold}, "
                    f"current_wave={self.current_wave}, "
                    f"step_counter={self.step_counter}",
                    self.dp_rank,
                    self._lb_enable_diagnostics,
                )
            self._lb_pending_long_req = False
            self._lb_pending_long_req_blk = 0

        lb_active = (
            (static_lb_enable or self._lb_dynamic_enable)
            and self.step_counter >= self._lb_start_step
            and (self._lb_end_step < 0 or self.step_counter < self._lb_end_step)
        )
        self.scheduler._lb_kv_prefetch_enabled = lb_active
        if lb_active:
            admission_candidates = self.scheduler.prepare_dyntra_lb_step()
            has_new_long_req = self._do_lb_allgather(admission_candidates)
            if dynamic_lb_mode and self._lb_dynamic_enable:
                if dynamic_activated_this_step or has_new_long_req:
                    self._lb_dynamic_step = 0
                else:
                    self._lb_dynamic_step += self._lb_balance_interval
                    if self._lb_dynamic_step >= self._lb_dynamic_max_step:
                        self._lb_dynamic_enable = False
                        _print_rank_0(
                            "Dynamic LB disabled after "
                            f"{self._lb_dynamic_step} steps without long request: "
                            f"current_wave={self.current_wave}, "
                            f"step_counter={self.step_counter}",
                            self.dp_rank,
                            self._lb_enable_diagnostics,
                        )
        else:
            self.scheduler.modifications = None
            self.scheduler.lb_freeze = False

    def _has_global_unfinished_reqs(self, local_unfinished: bool) -> bool:
        result = _has_global_unfinished_reqs_with_diagnostics(self, local_unfinished)
        # DyntraLB collectives must follow the DP wave/idle synchronization.
        # The resulting plan is consumed by the next engine step.
        if result:
            self.run_balance_load()
        else:
            self.scheduler.modifications = None
            self.scheduler.lb_freeze = False
            self.scheduler._lb_kv_prefetch_enabled = False
            self._lb_dynamic_enable = False
            self._lb_dynamic_step = 0
            self._lb_pending_long_req = False
            self._lb_pending_long_req_blk = 0
            if self._lb_enable_diagnostics:
                logger.info(
                    "[dyntra_lb] run_busy_loop() | No engines running, "
                    "step_counter set to 0, current_wave incremented to %d.",
                    self.current_wave + 1,
                )
        return result

    def _do_lb_allgather(self, admission_candidates: list[Request]) -> bool:
        max_slots = self._lb_max_slots_cached
        dp_size = self._lb_dp_size_cached
        max_num_seqs = self._lb_max_num_seqs

        running_snapshot = list(self.scheduler.running)[:max_slots]
        waiting_snapshot = admission_candidates[: max_slots - len(running_snapshot)]
        running_blks = [get_dyntra_lb_request_block_num(self.scheduler, req) for req in running_snapshot]
        waiting_blks = [get_dyntra_lb_request_block_num(self.scheduler, req) for req in waiting_snapshot]

        run_cnt = len(running_blks)
        wait_cnt = len(waiting_blks)
        blocks = running_blks + waiting_blks
        if self._lb_headroom is not None:
            free, reserve = kv_pool_headroom(self.scheduler, self._lb_headroom)
            budget = max(0, free - local_admission_reserve(self.scheduler, reserve))
            costs = [estimate_request_kv_blocks(self.scheduler, req, 0) for req in waiting_snapshot]
            # Keep the existing compact varint exchange and its two collectives.
            # Shared headroom config determines how every rank reads the payload.
            blocks = [len(blocks), budget, *blocks, *costs]
        payload = encode_snapshot(blocks, run_cnt, self._lb_pending_long_req)
        self._lb_pending_long_req = False
        self._lb_pending_long_req_blk = 0

        local_size = np.array([len(payload)], dtype=np.int32)
        sizes = [np.zeros(1, dtype=np.int32) for _ in range(dp_size)]
        dist.all_gather([torch.as_tensor(size) for size in sizes], torch.as_tensor(local_size), group=self.dp_group)
        max_size = max(int(size[0]) for size in sizes)
        local_payload = np.zeros(max_size, dtype=np.uint8)
        local_payload[: len(payload)] = np.frombuffer(payload, dtype=np.uint8)
        gathered = [np.zeros(max_size, dtype=np.uint8) for _ in range(dp_size)]
        dist.all_gather(
            [torch.as_tensor(data) for data in gathered], torch.as_tensor(local_payload), group=self.dp_group
        )

        requests_by_rank = []
        has_new_long_req = False
        admission_by_rank = [] if self._lb_headroom is not None else None
        for rank_data, size in zip(gathered, sizes):
            blocks, split, pending_long = decode_snapshot(rank_data[: int(size[0])].tobytes())
            if admission_by_rank is not None:
                if len(blocks) < 2:
                    raise ValueError("Incomplete compact KV admission snapshot")
                count, budget = blocks[:2]
                if count < split or len(blocks) != 2 + count + (count - split):
                    raise ValueError("Invalid compact KV admission snapshot")
                admission_by_rank.append((budget, blocks[2 + count :]))
                blocks = blocks[2 : 2 + count]
            requests_by_rank.append((blocks, split))
            has_new_long_req = has_new_long_req or pending_long

        modifications = self._balance_load(
            requests_by_rank,
            dp_size,
            max_num_seqs=max_num_seqs,
            threshold=self._lb_threshold,
            admission_by_rank=admission_by_rank,
            **self._lb_planner_options,
        )
        _print_requests_by_rank(
            requests_by_rank,
            self.dp_rank,
            self._lb_enable_diagnostics,
        )
        _print_modifications(
            modifications,
            self.dp_rank,
            self._lb_enable_diagnostics,
        )
        local_plan = modifications[self.dp_rank]
        local_snapshot = running_snapshot[:run_cnt] + waiting_snapshot[:wait_cnt]
        self.scheduler.modifications = {
            **local_plan,
            "out_req_ids": [local_snapshot[i].request_id for i in local_plan["out_slots"]],
            "in_req_ids": [local_snapshot[i].request_id for i in local_plan["in_slots"]],
        }
        if not local_plan["freeze"]:
            self.scheduler._run_lb_kv_prefetch(set(self.scheduler.modifications["in_req_ids"]))
        return has_new_long_req


_PreviousRunEngineCore = EngineCoreProc.run_engine_core
_UpstreamRunEngineCore = _balance_patch._OriginalRunEngineCore
_OriginalDPEngineCoreProc = DPEngineCoreProc


def _dyntra_lb_run_engine_core(*args, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs):
    vllm_config = kwargs.get("vllm_config")
    dyntra_lb_config = _get_dyntra_lb_config(vllm_config)
    if not dyntra_lb_config.enabled:
        return _PreviousRunEngineCore(*args, dp_rank=dp_rank, local_dp_rank=local_dp_rank, **kwargs)

    _print_rank_0(
        "Enable DyntraLB DP load balancing.",
        dp_rank,
        dyntra_lb_config.enable_diagnostics,
    )
    _engine_core_mod.DPEngineCoreProc = DyntraLBDPEngineCoreProc
    try:
        return _UpstreamRunEngineCore(*args, dp_rank=dp_rank, local_dp_rank=local_dp_rank, **kwargs)
    finally:
        _engine_core_mod.DPEngineCoreProc = _OriginalDPEngineCoreProc


# Preserve the upstream staticmethod binding when replacing run_engine_core.
EngineCoreProc.run_engine_core = staticmethod(_dyntra_lb_run_engine_core)

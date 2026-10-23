# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Execute production policy methods without importing the NPU/vLLM runtime.

Only runtime dependencies are substituted. The class/method bodies are loaded
unchanged from the source AST. This is not a replacement for full scheduler UTs.
"""

import ast
import time
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
from source_modules import (
    DyntraLBConfig,
    DyntraLBFirstTokenPolicy,
    DyntraLBPausePolicy,
    balance_load,
    balance_load_original,
)

ROOT = Path(__file__).resolve().parents[3]


class RequestStatus(Enum):
    RUNNING = 1
    WAITING = 2
    PREEMPTED = 3


class SchedulingPolicy(Enum):
    FCFS = 1
    PRIORITY = 2


class Queue:
    def __init__(self):
        self.items = []

    def __iter__(self):
        return iter(self.items)

    def prepend_request(self, request):
        self.items.insert(0, request)

    def remove_requests(self, requests):
        self.items = [r for r in self.items if r not in requests]


class BaseScheduler:
    def __init__(self, **kwargs):
        self.running = []
        self.waiting = Queue()
        self.skipped_waiting = Queue()
        self.current_step = 0
        self.policy = SchedulingPolicy.FCFS
        self.cache_config = SimpleNamespace(block_size=8)
        self.log_stats = False
        self.max_num_running_reqs = 2

    def _update_after_schedule(self, output):
        pass

    def _free_request(self, request, delay_free_blocks):
        return None

    def _preempt_request(self, request, timestamp):
        pass


class BaseEngine:
    def _init_data_parallel(self, config):
        self.dp_group = object()
        self.dp_rank = 0


def load_scheduler():
    path = ROOT / "vllm_ascend/core/dyntra_lb_scheduler.py"
    nodes = ast.parse(path.read_text()).body
    names = {"get_dyntra_lb_block_size", "get_dyntra_lb_request_block_num", "DyntraLBPolicyMixin"}
    tree = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            *[n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names],
        ],
        type_ignores=[],
    )
    ns = dict(
        _SchedulerBase=BaseScheduler,
        RequestStatus=RequestStatus,
        SchedulingPolicy=SchedulingPolicy,
        DyntraLBConfig=DyntraLBConfig,
        DyntraLBPausePolicy=DyntraLBPausePolicy,
        DyntraLBFirstTokenPolicy=DyntraLBFirstTokenPolicy,
        time=time,
    )
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), ns)
    return ns["DyntraLBPolicyMixin"], ns["get_dyntra_lb_request_block_num"]


def load_engine(block_num):
    path = ROOT / "vllm_ascend/patch/platform/patch_dyntra_lb_core.py"
    nodes = ast.parse(path.read_text()).body
    tree = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == "DyntraLBDPEngineCoreProc"),
        ],
        type_ignores=[],
    )
    dist = SimpleNamespace(
        get_world_size=lambda _: 2,
        all_reduce=MagicMock(),
        all_gather=MagicMock(),
        ReduceOp=SimpleNamespace(MAX=object()),
    )
    ns = dict(
        DPEngineCoreProc=BaseEngine,
        balance_load=balance_load,
        balance_load_original=balance_load_original,
        np=np,
        torch=SimpleNamespace(as_tensor=lambda arr: arr),
        dist=dist,
        init_ascend_config=lambda config: SimpleNamespace(
            scheduler_config=SimpleNamespace(
                dyntra_lb_config=DyntraLBConfig(
                    config.additional_config.get("scheduler_config", {}).get("dyntra_lb_config")
                )
            )
        ),
        get_dyntra_lb_request_block_num=block_num,
        _print_rank_0=lambda *a: None,
        _print_requests_by_rank=lambda *a: None,
        _print_modifications=lambda *a: None,
    )
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), ns)
    return ns["DyntraLBDPEngineCoreProc"], dist


def request(req_id, tokens=16, status=RequestStatus.RUNNING):
    return SimpleNamespace(request_id=req_id, all_token_ids=list(range(tokens)), status=status)

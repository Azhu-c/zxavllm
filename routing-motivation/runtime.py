"""Isolated routing/observation hooks; no installed source files are changed.

Based on the inspected DPLBAsyncMPClient interface at vLLM 58d3918.
Native get_core_engine_for_request performs the actual send bookkeeping.
"""
from __future__ import annotations

import asyncio
from collections import deque
import contextvars
import functools
import importlib.abc
import importlib.machinery
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time

CONTEXT = contextvars.ContextVar("motivation_step", default=None)
ATTENTION_COUNTS = contextvars.ContextVar("motivation_attention_counts", default=None)
HANDLES = {}


def emit(kind, row):
    directory = Path(os.environ["MOTIVATION_DATA"])
    directory.mkdir(parents=True, exist_ok=True)
    key = str(directory / f"{kind}-{os.getpid()}.jsonl")
    if key not in HANDLES:
        HANDLES[key] = open(key, "a", encoding="utf-8", buffering=1)
    HANDLES[key].write(json.dumps({"wall_ns": time.time_ns(), **row},
                                 separators=(",", ":"), allow_nan=False) + "\n")


def request_blocks(request, block_size):
    ids = getattr(request, "all_token_ids", None)
    if ids is None:
        ids = request.prompt_token_ids
    return math.ceil(len(ids) / block_size)


def snapshot(core):
    scheduler = core.scheduler
    block_size = scheduler.cache_config.block_size
    running = {r.request_id: r for r in scheduler.running}
    waiting = {r.request_id: r for name in ("waiting", "skipped_waiting")
               for r in getattr(scheduler, name, ()) if r.request_id not in running}
    pool = getattr(scheduler.kv_cache_manager, "block_pool", None)
    return {
        "rank": int(core.dp_rank), "received": getattr(core, "_mot_received", 0),
        "wave": getattr(core, "current_wave", None),
        "step": getattr(core, "step_counter", None), "block_size": block_size,
        "running_count": len(running), "waiting_count": len(waiting),
        "running_blocks": sum(request_blocks(r, block_size) for r in running.values()),
        "waiting_blocks": sum(request_blocks(r, block_size) for r in waiting.values()),
        "physical_free_blocks": pool.get_num_free_blocks() if pool is not None else None,
        "max_num_seqs": scheduler.max_num_running_reqs,
    }


class RouterState:
    """Per-client ADD sequence minus acknowledged engine ADD sequence."""
    def __init__(self, size):
        self.pending = [deque() for _ in range(size)]
        self.sent = [0] * size
        self.reports = None
        self.updated = 0.0
        self.cursor = 0

    def update(self, reports):
        if [r["rank"] for r in reports] != list(range(len(self.sent))):
            raise RuntimeError("The router must own all Decoder DP ranks in rank order")
        if len({r["block_size"] for r in reports}) != 1:
            raise RuntimeError("DP ranks use different logical block sizes")
        for i, report in enumerate(reports):
            if report["received"] > self.sent[i]:
                raise RuntimeError("Untracked requests: use one API server and no other clients")
            while self.pending[i] and self.pending[i][0][0] <= report["received"]:
                self.pending[i].popleft()
        self.reports = reports
        self.updated = time.monotonic()

    def totals(self):
        return [r["running_blocks"] + r["waiting_blocks"] +
                sum(blocks for _, blocks in pending)
                for r, pending in zip(self.reports, self.pending)]

    def choose(self, policy, inflight):
        n = len(self.sent)
        scores = {"rr": [0] * n, "jsq": inflight,
                  "kv": self.totals()}[policy]
        selected = min(range(n), key=lambda i: (scores[i], (i - self.cursor) % n))
        self.cursor = (selected + 1) % n
        return selected, list(scores)

    def dispatch(self, rank, blocks):
        self.sent[rank] += 1
        self.pending[rank].append((self.sent[rank], blocks))


def patch_client(module):
    cls = module.DPLBAsyncMPClient
    original_init = cls.__init__
    original_add = cls.add_request_async
    original_select = cls.get_core_engine_for_request
    source = inspect.getsource(original_select)
    required = ("data_parallel_rank", "reqs_in_flight", "engine_inflight")
    if not all(token in source for token in required):
        raise RuntimeError("Unsupported DP selector: expected the inspected vLLM 58d3918 interface. "
                           "Run `bash run.sh check` and inspect runtime-check.json.")

    @functools.wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        parallel = self.vllm_config.parallel_config
        if (self.client_count != 1 or parallel.data_parallel_external_lb or
                parallel.data_parallel_hybrid_lb or
                list(self.engine_ranks_managed) != list(range(len(self.core_engines)))):
            raise RuntimeError("Use one API server and internal DP covering all Decoder ranks")
        self._mot_state = RouterState(len(self.core_engines))
        self._mot_lock = asyncio.Lock()
        self._mot_poll = None
        self._mot_error = None
        emit("runtime", {"event": "client_installed", "policy": os.environ["MOTIVATION_POLICY"],
                         "ranks": list(self.engine_ranks_managed)})

    async def poll(self):
        interval = float(os.environ.get("MOTIVATION_REPORT_MS", "100")) / 1000
        while not self.resources.engine_dead:
            try:
                reports = await asyncio.gather(*[
                    self._call_utility_async("motivation_snapshot", engine=engine)
                    for engine in self.core_engines])
                self._mot_state.update(reports)
                emit("state", {"ranks": reports,
                               "owned_blocks_with_pending": self._mot_state.totals()})
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self._mot_error = repr(error)
                emit("runtime", {"event": "poll_error", "error": repr(error)})
                return
            await asyncio.sleep(interval)

    @functools.wraps(original_add)
    async def add(self, request):
        if self._mot_poll is None:
            self._mot_poll = asyncio.create_task(poll(self))
        deadline = time.monotonic() + 30
        while self._mot_state.reports is None:
            if self._mot_error or time.monotonic() > deadline:
                raise RuntimeError("Decoder state reports unavailable: " + str(self._mot_error))
            await asyncio.sleep(0.01)
        async with self._mot_lock:
            if self._mot_error or time.monotonic() - self._mot_state.updated > 5:
                raise RuntimeError("Decoder state reports stopped: " + str(self._mot_error))
            # Original DP add performs ADD, wave notification and output-task startup.
            return await original_add(self, request)

    @functools.wraps(original_select)
    def select(self, request):
        if request.data_parallel_rank is not None or request.pooling_params is not None:
            raise RuntimeError("This experiment requires unpinned text generation requests")
        state = self._mot_state
        inflight = [self.engine_inflight[engine] for engine in self.core_engines]
        rank, scores = state.choose(os.environ["MOTIVATION_POLICY"], inflight)
        # Native pinned path updates lifecycle/abort counters once, then restore request.
        request.data_parallel_rank = rank
        try:
            chosen = original_select(self, request)
        finally:
            request.data_parallel_rank = None
        if chosen != self.core_engines[rank]:
            raise RuntimeError("Native client did not honor selected DP rank")
        blocks = request_blocks(request, state.reports[rank]["block_size"])
        state.dispatch(rank, blocks)
        emit("route", {"request_id": request.request_id, "rank": rank,
                       "policy": os.environ["MOTIVATION_POLICY"], "scores": scores,
                       "incoming_blocks": blocks,
                       "state_age_ms": (time.monotonic() - state.updated) * 1000})
        return chosen

    cls.__init__ = init
    cls.add_request_async = add
    cls.get_core_engine_for_request = select


def patch_core(module):
    cls = module.EngineCore
    original_add = cls.add_request
    original_init = cls.__init__

    @functools.wraps(original_add)
    def add(self, *args, **kwargs):
        result = original_add(self, *args, **kwargs)
        self._mot_received = getattr(self, "_mot_received", 0) + 1
        return result

    @functools.wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if os.environ.get("MOTIVATION_PROFILE") != "1":
            return
        schedule = self.scheduler.schedule

        def observed_schedule(*a, **kw):
            output = schedule(*a, **kw)
            marker = {"rank": int(self.dp_rank),
                      "wave": getattr(self, "current_wave", None),
                      "step": getattr(self, "step_counter", None)}
            # The inspected Ascend runner already passes extra scheduler-output
            # attributes across processes. The worker records whether this survived.
            output.motivation_marker = marker
            emit("steps", {**marker, "scheduled_count": len(output.num_scheduled_tokens),
                           "scheduled_tokens": sum(output.num_scheduled_tokens.values())})
            return output
        self.scheduler.schedule = observed_schedule

    cls.add_request = add
    cls.__init__ = init
    cls.motivation_snapshot = snapshot


def patch_runner(module):
    cls = module.NPUModelRunner
    original = cls.execute_model

    @functools.wraps(original)
    def execute(self, scheduler_output, *args, **kwargs):
        import torch
        marker = getattr(scheduler_output, "motivation_marker", None)
        if marker is None:
            emit("runtime", {"event": "worker_marker_missing", "rank": int(self.dp_rank)})
            return original(self, scheduler_output, *args, **kwargs)
        token = CONTEXT.set(marker)
        count_token = ATTENTION_COUNTS.set({})
        name = "MOT_STEP|" + json.dumps(marker, separators=(",", ":"))
        try:
            with torch.profiler.record_function(name):
                return original(self, scheduler_output, *args, **kwargs)
        finally:
            ATTENTION_COUNTS.reset(count_token)
            CONTEXT.reset(token)
    cls.execute_model = execute


def patch_attention(module):
    cls = module.AscendAttentionBackendImpl
    original = cls.forward

    @functools.wraps(original)
    def forward(self, layer, *args, **kwargs):
        marker = CONTEXT.get()
        if marker is None:
            return original(self, layer, *args, **kwargs)
        import torch
        name = str(getattr(layer, "layer_name", "unknown"))
        counts = ATTENTION_COUNTS.get()
        occurrence = counts.get(name, 0)
        counts[name] = occurrence + 1
        values = {**marker, "layer": name, "occurrence": occurrence}
        with torch.profiler.record_function("MOT_ATTN|" + json.dumps(values, separators=(",", ":"))):
            return original(self, layer, *args, **kwargs)
    cls.forward = forward


HOOKS = {"vllm.v1.engine.core": patch_core,
         "vllm.v1.engine.core_client": patch_client}
if os.environ.get("MOTIVATION_PROFILE") == "1":
    HOOKS.update({"vllm_ascend.worker.model_runner_v1": patch_runner,
                  "vllm_ascend.attention.attention_v1": patch_attention})


class Loader(importlib.abc.Loader):
    def __init__(self, wrapped, name):
        self.wrapped, self.name = wrapped, name

    def create_module(self, spec):
        return self.wrapped.create_module(spec) if hasattr(self.wrapped, "create_module") else None

    def exec_module(self, module):
        self.wrapped.exec_module(module)
        HOOKS[self.name](module)

    def __getattr__(self, name):
        return getattr(self.wrapped, name)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in HOOKS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = Loader(spec.loader, fullname)
        return spec


def install_import_hooks():
    sys.meta_path.insert(0, Finder())

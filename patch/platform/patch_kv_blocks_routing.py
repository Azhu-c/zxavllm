# SPDX-License-Identifier: Apache-2.0
"""Ascend-owned runtime extension; never writes to the vLLM installation."""
from __future__ import annotations

import asyncio
from collections import deque
import contextlib
import functools
import hashlib
import importlib
import inspect
from pathlib import Path
import threading

from vllm_ascend.core.kv_blocks_routing import (
    LocalKVReport,
    choose_rank,
    enabled,
    extension_active,
    headroom_config,
    route_log_enabled,
    snapshot,
    step_log_enabled,
    timeout_seconds,
    validate_config,
    validate_states,
)

_LOCK = threading.RLock()
_SOURCE_HASHES = {
    "vllm.v1.engine.core": "f9625f331b65be39b14e65e0c0a360d6515d9147777bf7b9cdc2d8bc5eafa622",
    "vllm.v1.engine.core_client": "afb2b627cf9ef861f05b411494156cc6ad5076ea770c6f2a4c8a941ca82103ec",
}
_EXPECTED_CLIENT_METHODS = {'DPLBAsyncMPClient.__init__': '692568c140acbdb0e02f0c5df7034e3e917cd19bbcbbc70bd05392ad10ca17f0', 'DPLBAsyncMPClient.get_core_engine_for_request': 'f9f091f3c1ab38c4de60ff58e216379bd6587570ac9bc3862c74ff9134f06255', 'DPLBAsyncMPClient.process_engine_outputs': '332aeb5b222ce1b73780f1858a210fc1c70ef0e10237010b28280647ffa9b5f8', 'DPLBAsyncMPClient.abort_requests_async': '2e9995eeef2cc91a9243876dbd557390a4bce110b9925aa9087898be5554bffe', 'DPAsyncMPClient.add_request_async': 'e937266fd11defc5e6980b72acf05e33e8531f0b51e0d1d8c0f66016b4c24269', 'AsyncMPClient._call_utility_async': '72f6f12f61197b0f3f3c1c3c5aa80d2b0bad3a21e26dac4102b41e8c0cde7ce9'}


def _verify_source(module) -> None:
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != _SOURCE_HASHES[module.__name__]:
        raise RuntimeError(
            f"Unsupported {module.__name__}: expected original vLLM 58d3918. "
            "Restore the previous vLLM-source patch before using the Ascend extension."
        )


class KVBlocksRoutingClientMixin:
    """Reuse upstream send, wave, lifecycle and abort handling through super()."""

    def __init__(self, vllm_config, executor_class, log_stats, client_addresses=None,
                 client_count=1, client_index=0):
        if enabled() or route_log_enabled():
            validate_config(vllm_config, client_count)
        self._kv_headroom = headroom_config(vllm_config) if enabled() else None
        self._kv_report_tasks = []
        self._kv_reports = {}
        self._kv_pending_load = {}
        self._kv_pending_blocks = {}
        self._kv_sent = {}
        self._kv_chosen_index = None
        self._kv_blocks_lock = asyncio.Lock()
        self._kv_blocks_next_engine = None
        self._kv_blocks_snapshot = None
        self._kv_blocks_sequence = 0
        self._kv_blocks_pending_requests = set()
        self._kv_blocks_aborted_requests = set()
        super().__init__(vllm_config, executor_class, log_stats,
                         client_addresses, client_count, client_index)

    async def add_request_async(self, request):
        if not enabled() and not route_log_enabled():
            return await super().add_request_async(request)
        request_id = request.request_id
        self._kv_blocks_pending_requests.add(request_id)
        try:
            await self._add_request_with_kv_blocks_async(request)
        finally:
            self._kv_blocks_pending_requests.discard(request_id)
            self._kv_blocks_aborted_requests.discard(request_id)

    async def _add_request_with_kv_blocks_async(self, request):
        async with self._kv_blocks_lock:
            if request.request_id in self._kv_blocks_aborted_requests:
                return
            ranks = list(self.engine_ranks_managed)
            if ranks != list(range(self.vllm_config.parallel_config.data_parallel_size)):
                raise RuntimeError("KV-block routing must cover every Decode rank")
            routing = enabled()
            pressure = None
            if self._kv_headroom is not None:
                if not self._kv_report_tasks:
                    self._kv_pending_load = {rank: deque() for rank in ranks}
                    self._kv_pending_blocks = dict.fromkeys(ranks, 0)
                    self._kv_sent = dict.fromkeys(ranks, 0)
                    self._kv_report_tasks = [
                        asyncio.create_task(self._watch_local_report(rank, self.core_engines[rank]))
                        for rank in ranks
                    ]
                states = []
                totals, pressure = self._cached_reports(ranks)
                policy = "kv_local" if totals is not None else "native_cold"
            else:
                policy = "kv_aware" if routing else "native"
                try:
                    states, totals = await self._pull_kv_blocks_states()
                except Exception as error:
                    if routing:
                        raise
                    from vllm.logger import logger

                    logger.warning("[kv_blocks_observer] Native routing snapshot unavailable: %s", error)
                    states, totals = [], None
            if request.request_id in self._kv_blocks_aborted_requests:
                return
            preferred = request.data_parallel_rank
            if preferred is None:
                from vllm.v1.pool.late_interaction import get_late_interaction_engine_index

                preferred = get_late_interaction_engine_index(request.pooling_params, len(ranks))
            use_kv_selection = routing and totals is not None
            selected = (
                choose_rank(totals, self.eng_start_index, preferred, pressure)
                if use_kv_selection else None
            )
            self._kv_blocks_sequence += 1
            self._kv_blocks_next_engine = selected
            self._kv_blocks_snapshot = (states, totals, ranks, preferred is not None, policy, pressure)
            try:
                await super().add_request_async(request)
                if use_kv_selection:
                    self.eng_start_index = (selected + 1) % len(ranks)
                if self._kv_headroom is not None:
                    self._record_dispatched_blocks(request, self._kv_chosen_index)
            finally:
                self._kv_blocks_next_engine = None
                self._kv_blocks_snapshot = None
                self._kv_chosen_index = None

    async def _watch_local_report(self, rank, engine):
        from vllm.logger import logger

        revision = 0
        while not self.resources.engine_dead:
            try:
                report = await self._call_utility_async(
                    "subscribe_ascend_kv_local_report", revision, engine=engine,
                )
                keys = ("revision", "received", "block_size", "blocks", "pressure")
                if (
                    not isinstance(report, dict)
                    or any(type(report.get(key)) is not int or report[key] < 0 for key in keys)
                    or report["pressure"] not in (0, 1)
                    or report["block_size"] == 0
                    or report["revision"] <= revision
                ):
                    raise RuntimeError(f"Invalid local report from DP{rank}")
                revision = report["revision"]
                pending = self._kv_pending_load[rank]
                while pending and pending[0][0] <= report["received"]:
                    _, blocks = pending.popleft()
                    self._kv_pending_blocks[rank] -= blocks
                self._kv_reports[rank] = report
            except Exception as error:
                self._kv_reports.pop(rank, None)
                revision = 0
                logger.warning("[kv_blocks_local] report unavailable DP%d: %s", rank, error)
                await asyncio.sleep(1)

    def _cached_reports(self, ranks):
        if len(self._kv_reports) != len(ranks):
            return None, None
        reports = [self._kv_reports[rank] for rank in ranks]
        if len({report["block_size"] for report in reports}) != 1:
            return None, None
        # Estimate load already sent through ADD but absent from the last report.
        totals = [
            report["blocks"] + self._kv_pending_blocks[rank]
            for rank, report in zip(ranks, reports)
        ]
        return totals, [report["pressure"] for report in reports]

    def _record_dispatched_blocks(self, request, rank):
        report = self._kv_reports.get(rank)
        block_size = self.vllm_config.cache_config.block_size
        ids = request.prompt_token_ids
        tokens = len(ids) if ids is not None else int(request.prompt_embeds.shape[0])
        sequence = self._kv_sent[rank] + 1
        self._kv_sent[rank] = sequence
        if report is None or sequence > report["received"]:
            blocks = (tokens + block_size - 1) // block_size
            self._kv_pending_load[rank].append((sequence, blocks))
            self._kv_pending_blocks[rank] += blocks

    def shutdown(self, timeout=None):
        for task in self._kv_report_tasks:
            loop = task.get_loop()
            if not loop.is_closed():
                loop.call_soon_threadsafe(task.cancel)
        return super().shutdown(timeout=timeout)

    async def _pull_kv_blocks_states(self):
        tasks = [
            asyncio.create_task(self._call_utility_async(
                "get_ascend_kv_blocks_routing_state", engine=engine
            ))
            for engine in self.core_engines
        ]
        pull = asyncio.gather(*tasks)
        try:
            states = await asyncio.wait_for(pull, timeout=timeout_seconds())
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await pull
            await asyncio.gather(*tasks, return_exceptions=True)
        return states, validate_states(states, list(self.engine_ranks_managed))

    def get_core_engine_for_request(self, request):
        if self._kv_blocks_snapshot is None:
            if enabled():
                raise RuntimeError("KV-block routing requires a fresh global pull")
            return super().get_core_engine_for_request(request)
        states, totals, ranks, pinned, policy, pressure = self._kv_blocks_snapshot
        if self._kv_blocks_next_engine is not None:
            # Let the original selector perform its own lifecycle bookkeeping.
            original_rank = request.data_parallel_rank
            request.data_parallel_rank = self._kv_blocks_next_engine
            try:
                chosen = super().get_core_engine_for_request(request)
            finally:
                request.data_parallel_rank = original_rank
        else:
            # Native policy, tie-breaking and count updates are all performed
            # by the original selector exactly once. KV totals are only logged.
            chosen = super().get_core_engine_for_request(request)
        selected = self.core_engines.index(chosen)
        self._kv_chosen_index = selected
        if route_log_enabled():
            from vllm.logger import logger

            logger.info(
                "[kv_blocks_route] seq=%d totals=%s selected=DP%d pinned=%s policy=%s",
                self._kv_blocks_sequence, totals if totals is not None else "unavailable",
                ranks[selected], pinned, policy,
            )
            for state in states:
                _log_state(logger, "kv_blocks_route", state, f"seq={self._kv_blocks_sequence}")
            if pressure is not None:
                logger.info(
                    "[kv_blocks_headroom] seq=%d selected=DP%d pressure=%s all_tight=%s",
                    self._kv_blocks_sequence, ranks[selected], pressure, all(pressure),
                )
        return chosen

    async def abort_requests_async(self, request_ids):
        if enabled() or route_log_enabled():
            self._kv_blocks_aborted_requests.update(
                request_id for request_id in request_ids
                if request_id in self._kv_blocks_pending_requests
            )
        return await super().abort_requests_async(request_ids)

    async def prepare_elastic_ep(self, new_data_parallel_size):
        if enabled() or route_log_enabled():
            raise RuntimeError("Disable KV-block routing before changing DP topology")
        return await super().prepare_elastic_ep(new_data_parallel_size)


def _log_state(logger, tag, state, label):
    running, waiting = state["running_blocks"], state["waiting_blocks"]
    logger.info(
        "[%s] %s DP%d | Run(%d): %s | Wait(%d): %s | total=%d block_size=%d",
        tag, label, state["rank"], len(running), running, len(waiting), waiting,
        sum(running) + sum(waiting), state["block_size"],
    )


def install_client_hooks():
    # Called from NPUPlatform.check_and_update_config, after normal imports
    # and before EngineCoreClient.make_async_mp_client creates the client.
    with _LOCK:
        module = importlib.import_module("vllm.v1.engine.core_client")
        if getattr(module, "_ascend_kv_blocks_routing_installed", False):
            return
        _verify_source(module)
        for dotted_name, expected in _EXPECTED_CLIENT_METHODS.items():
            class_name, method_name = dotted_name.split(".")
            method = getattr(getattr(module, class_name), method_name)
            actual = hashlib.sha256(inspect.getsource(method).encode()).hexdigest()
            if actual != expected:
                raise RuntimeError(f"Another extension changed {dotted_name}; refusing to combine routers")
        original_class = module.DPLBAsyncMPClient
        if original_class.__module__ != module.__name__:
            raise RuntimeError("Another extension already replaced the DP client class")
        original_factory = module.EngineCoreClient.make_async_mp_client
        client_class = type(
            "AscendKVBlocksDPLBAsyncMPClient", (KVBlocksRoutingClientMixin, original_class),
            {"__module__": __name__},
        )

        @functools.wraps(original_factory)
        def factory(vllm_config, executor_class, log_stats, client_addresses=None,
                    client_count=1, client_index=0):
            if enabled() or route_log_enabled():
                validate_config(vllm_config, client_count)
            return original_factory(vllm_config, executor_class, log_stats,
                                    client_addresses, client_count, client_index)

        globals()[client_class.__name__] = client_class
        module.DPLBAsyncMPClient = client_class
        module.EngineCoreClient.make_async_mp_client = staticmethod(factory)
        module._ascend_kv_blocks_routing_installed = True


def install_engine_hooks():
    # Global Ascend patches are loaded in engine-core subprocesses too.
    # Do not import core_client here: it may still be importing core.
    with _LOCK:
        module = importlib.import_module("vllm.v1.engine.core")
        if getattr(module, "_ascend_kv_blocks_routing_installed", False):
            return
        _verify_source(module)
        if hasattr(module.EngineCore, "get_ascend_kv_blocks_routing_state"):
            raise RuntimeError("An incompatible Ascend KV-block RPC already exists")
        original_step = module.EngineCoreProc._process_engine_step

        original_add = module.EngineCore.add_request

        def get_state(self):
            return snapshot(self.scheduler, self.dp_rank)

        def subscribe(self, revision):
            report = getattr(self, "_ascend_kv_local_report", None)
            if report is None:
                policy = headroom_config(self.vllm_config)
                if policy is None:
                    raise RuntimeError("Local KV reports require kv_blocks_headroom on Decode")
                report = self._ascend_kv_local_report = LocalKVReport(policy)
            report.observe(self.scheduler, getattr(self, "_ascend_kv_received", 0))
            return report.listen(revision)

        @functools.wraps(original_add)
        def add(self, request, request_wave=0):
            result = original_add(self, request, request_wave)
            received = getattr(self, "_ascend_kv_received", None)
            if received is None:
                received = 0 if enabled() and headroom_config(self.vllm_config) is not None else -1
            self._ascend_kv_received = received + 1 if received >= 0 else -1
            return result

        @functools.wraps(original_step)
        def step(self):
            if step_log_enabled():
                from vllm.logger import logger

                label = f"wave={self.current_wave} step={self.step_counter}"
                _log_state(logger, "kv_blocks_step", get_state(self), label)
            result = original_step(self)
            report = getattr(self, "_ascend_kv_local_report", None)
            if report is not None:
                report.observe(self.scheduler, getattr(self, "_ascend_kv_received", 0))
            return result

        module.EngineCore.get_ascend_kv_blocks_routing_state = get_state
        module.EngineCore.subscribe_ascend_kv_local_report = subscribe
        module.EngineCore.add_request = add
        module.EngineCoreProc._process_engine_step = step
        module._ascend_kv_blocks_routing_installed = True


def install_for_config(vllm_config):
    if not extension_active():
        return
    validate_config(vllm_config, 1)
    install_engine_hooks()
    if enabled() or route_log_enabled():
        install_client_hooks()


# This module is imported by the existing Ascend global-patch mechanism.
# The separate config hook handles API-client setup without circular imports.
if extension_active():
    install_engine_hooks()

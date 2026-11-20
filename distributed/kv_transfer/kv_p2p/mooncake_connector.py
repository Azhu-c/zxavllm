# SPDX-License-Identifier: Apache-2.0
import contextlib
import copy
import hashlib
import logging
import math
import os
import queue
import random
import struct
import threading
import time
from collections import OrderedDict, defaultdict, deque
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict

import msgspec
import numpy as np
import numpy.typing as npt
import torch
import torch_npu
import zmq
from mooncake.engine import TransferEngine  # type: ignore
from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed import get_pcp_group
from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.distributed.utils import get_pp_indices
from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.utils.network_utils import get_ip, make_zmq_path, make_zmq_socket
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.request import RequestStatus

from vllm_ascend import envs as ascend_envs
from vllm_ascend.ascend_config import get_ascend_config, init_ascend_config
from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec, AscendSlidingWindowMLASpec
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import global_te
from vllm_ascend.distributed.kv_transfer.utils.utils import (
    RegisterRegions,
    collect_storage_merged_register_regions,
    get_transfer_timeout_value,
    validate_register_region_count,
)
from vllm_ascend.distributed.utils import (
    get_decode_context_model_parallel_rank,
    get_decode_context_model_parallel_world_size,
)
from vllm_ascend.utils import enable_custom_op, enable_sfa_dcp_replicated_indexer, vllm_version_is

# isort: off
if TYPE_CHECKING:
    from vllm.v1.attention.backend import AttentionMetadata  # type: ignore
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request
# isort: on

GET_META_MSG = b"get_meta_msg"
DONE_RECVING_MSG = b"done_recving_msg"


# A busy peer can otherwise keep a global executor worker forever when the
# number of peers is larger than max_workers. Yield after a small FIFO batch so
# other peers already waiting in the global executor queue can make progress.
MAX_REQUESTS_PER_PEER_HANDLER = 5


class RemotePortInfo(TypedDict):
    num: int
    host: str


class MooncakeAgentMetadata(msgspec.Struct, omit_defaults=True, dict=True):
    engine_id: str
    te_rpc_port: int
    kv_group2layeridx: dict[int, tuple[dict[str, Any], list[int]]]
    block_size: int
    kv_caches_base_addr: list[list[int]]
    block_size_scale: list[list[int]]
    num_blocks: int
    block_lens: list[list[int]]
    block_strides: list[list[int]]
    local_ip: str = ""
    handshake_port: int = 0


@dataclass
class ReqMeta:
    local_block_ids: BlockIds
    num_external_tokens: int
    num_computed_tokens: int
    remote_block_ids: BlockIds

    remote_host: str
    remote_port: int
    remote_engine_id: str
    remote_request_id: str
    remote_pcp_size: int
    remote_dcp_size: int
    remote_ptp_size: int | None
    remote_multi_nodes_meta_mapping: dict[str, dict[str, Any]]
    num_prompt_blocks: int
    remote_block_size: int
    local_full_block_ids: BlockIds = tuple()
    do_virtual: bool = False


@dataclass(frozen=True)
class GroupPull:
    group_id: int
    remote_tp_offset: int
    num_group_pulls: int
    prefill_pp_rank: int = 0
    is_group_transfer_end: bool = False


@dataclass(frozen=True)
class GroupTransferInfo:
    tokens_per_block: int
    blocks_per_window: int
    is_state_group: bool


@dataclass
class SizedDict(OrderedDict):
    def __init__(self, max_size=16000, *args, **kwargs):
        self.max_size = max_size
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if len(self) > self.max_size:
            self.popitem(last=False)

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            value: dict[int, list[int]] = {}
            self[key] = value
            return value


class KVCacheTaskTracker:
    def __init__(self):
        super().__init__()

        self.done_task_lock = threading.Lock()
        self.finished_requests: set[str] = set()
        # Only used in prefill node. Tracks requests whose kv blocks freeing is
        # intentionally delayed. Each entry is a tuple of (request_id,
        # timestamp). If a request remains in this queue for too long, it will
        # be force-freed.
        self.delayed_free_requests: OrderedDict[str, float] = OrderedDict()
        self.reqs_to_process: set[str] = set()

    def add_req_to_process(self, request_id: str):
        self.reqs_to_process.add(request_id)

    def add_not_transfer_request(self, request_id: str):
        with self.done_task_lock:
            self.finished_requests.add(request_id)
            self.reqs_to_process.discard(request_id)

    def update_done_task_count(self, request_id: str):
        with self.done_task_lock:
            if request_id in self.reqs_to_process:
                self.finished_requests.add(request_id)
                self.reqs_to_process.discard(request_id)
                self.delayed_free_requests.pop(request_id, None)
            else:
                logger.warning(
                    "MooncakeConnector finish req not in reqs to process. "
                    "request_id=%s. "
                    "Possible cause: Request was already completed or not properly tracked. "
                    "Check: Verify request lifecycle and tracking logic.",
                    request_id,
                )

    def get_and_clear_finished_requests(self) -> set[str]:
        """
        Get and clear the requests that have been completed.
        Returns:
            A set of request IDs that have been completed.
        """
        with self.done_task_lock:
            finished_requests = self.finished_requests.copy()
            expired_requests = self._retrieve_expired_requests()
            finished_requests.update(expired_requests)
            self.finished_requests.clear()
        return finished_requests

    def add_delayed_request(self, request_id: str, delay_start_time: float):
        """Add a delayed free request."""
        with self.done_task_lock:
            if request_id in self.reqs_to_process:
                self.delayed_free_requests[request_id] = delay_start_time

    def _retrieve_expired_requests(self):
        """Retrieve all expired delayed requests."""
        expired_requests: set[str] = set()
        # Free delayed requests if they exceed the timeout
        current_time = time.time()
        while self.delayed_free_requests:
            request_id = next(iter(self.delayed_free_requests))
            delay_start_time = self.delayed_free_requests[request_id]
            if current_time - delay_start_time > envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT:
                self.delayed_free_requests.popitem(last=False)
                self.reqs_to_process.discard(request_id)
                expired_requests.add(request_id)
                logger.error(
                    "Force freed expired request: %s. "
                    "Reason: Request exceeded timeout threshold (%s seconds). "
                    "Action: Resources have been forcibly released to prevent memory leak.",
                    request_id,
                    envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT,
                )
            else:
                break
        return expired_requests


class KVCacheSendingThread(threading.Thread):
    def __init__(
        self,
        vllm_config: VllmConfig,
        tp_rank: int,
        prefill_tp_size: int,
        local_engine_id: str,
        side_channel_host: str,
        side_channel_port: int,
        metadata: MooncakeAgentMetadata,
        ready_event: threading.Event,
        kv_caches: dict[str, Any],
        pcp_rank: int,
    ):
        super().__init__(daemon=True, name="KVCacheSendingThread")
        self.tp_rank = tp_rank
        self.prefill_tp_size = prefill_tp_size
        self.pp_rank = get_pp_group().rank_in_group
        self.pcp_size = get_pcp_group().world_size
        self.pp_size = vllm_config.parallel_config.pipeline_parallel_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.local_engine_id = local_engine_id
        self.side_channel_host = side_channel_host
        self.side_channel_port = side_channel_port
        self.metadata = metadata
        self.ready_event = ready_event
        self.kv_caches = kv_caches
        self.pcp_rank = pcp_rank
        self.port_send_num: dict[str, int] = {}

        self.task_tracker = KVCacheTaskTracker()

    def get_and_clear_finished_requests(self) -> set[str]:
        """
        Get and clear the requests that have been completed.
        Returns:
            A set of request IDs that have been completed.
        """
        return self.task_tracker.get_and_clear_finished_requests()

    def add_not_transfer_request(self, request_id: str):
        self.task_tracker.add_not_transfer_request(request_id)

    def add_delayed_request(self, request_id: str, delay_start_time: float):
        return self.task_tracker.add_delayed_request(request_id, delay_start_time)

    def run(self):
        """Run the thread to handle KV cache transfer requests."""
        try:
            # Listen for new requests for metadata. NOTE(rob): we need each rank
            # to have a unique port. This hack to keeps us moving. We will
            # switch when moving to etcd or where we have a single ZMQ socket in
            # the scheduler.
            device_index = (self.pp_rank * self.pcp_size + self.pcp_rank) * self.tp_size + self.tp_rank
            handshake_port = self.side_channel_port + device_index
            path = make_zmq_path("tcp", self.side_channel_host, handshake_port)
            logger.info(
                "KVCacheSendingThread started listening on path: %s. Thread: tp_rank=%d, pp_rank=%d, pcp_rank=%d",
                path,
                self.tp_rank,
                self.pp_rank,
                self.pcp_rank,
            )
            with zmq_ctx(zmq.ROUTER, path) as sock:  # type: ignore
                self.ready_event.set()
                self.run_busy_loop(sock)
        except Exception as e:
            logger.exception(
                "Mooncake KVCacheSendingThread encountered exception. "
                "Thread: tp_rank=%d, pp_rank=%d, listening_path=%s. "
                "Error: %s",
                self.tp_rank,
                self.pp_rank,
                path,
                e,
            )

    def run_busy_loop(self, sock: zmq.Socket):  # type: ignore
        encoder = msgspec.msgpack.Encoder()
        encoded_data = encoder.encode(self.metadata)
        size_in_bytes = len(encoded_data)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Size of encoded MooncakeAgentMetadata: %s bytes", str(size_in_bytes))

        decoder = msgspec.msgpack.Decoder(type=tuple)
        while True:
            try:
                frames = sock.recv_multipart()
                if len(frames) < 2:
                    logger.error(
                        "Invalid message format in KVCacheSendingThread. "
                        "Expected: at least 2 frames (identity + payload). "
                        "Actual: %d frames. "
                        "Frames: %s. "
                        "Check: Verify message sender implementation.",
                        len(frames),
                        frames,
                    )
                    continue

                identity = frames[0]
                payload = [f for f in frames[1:] if f != b""]
                if len(payload) != 1:
                    logger.error(
                        "Invalid message format in KVCacheSendingThread. "
                        "Expected: exactly 1 payload frame. "
                        "Actual: %d payload frames. "
                        "Frames: %s. "
                        "Check: Verify message sender removes empty frames correctly.",
                        len(payload),
                        frames,
                    )
                    continue

                msg = decoder.decode(payload[0])
                if msg[0] == GET_META_MSG:
                    sock.send_multipart((identity, b"", encoded_data))
                elif msg[0] == DONE_RECVING_MSG:
                    logger.debug("Got DONE_RECVING_MSG for request %s", msg[1])
                    request_id = msg[1]
                    remote_port_send_num = msg[2]
                    if remote_port_send_num:
                        if request_id not in self.port_send_num:
                            self.port_send_num[request_id] = 0
                        self.port_send_num[request_id] += 1
                        device_index = (self.pp_rank * self.pcp_size + self.pcp_rank) * self.tp_size + self.tp_rank
                        handshake_port = self.side_channel_port + device_index
                        if self.port_send_num[request_id] >= remote_port_send_num[handshake_port]["num"]:
                            self.task_tracker.update_done_task_count(request_id)
                            del self.port_send_num[request_id]
                    else:
                        self.task_tracker.update_done_task_count(request_id)
                    # Acknowledge the request completion.
                    while True:
                        try:
                            # Send ACK to the sender.
                            sock.send_multipart((identity, b"", b"ACK"), flags=zmq.NOBLOCK)  # type: ignore
                            break
                        except zmq.Again:  # type: ignore
                            # If the socket is not ready, retry sending.
                            logger.debug("Socket not ready, retrying to send ACK for request %s", msg[1])
                            time.sleep(0.01)
                else:
                    logger.error(
                        "Connection listener received unexpected message type. "
                        "Expected: GET_META_MSG or DONE_RECVING_MSG. "
                        "Actual: %s. "
                        "Full message: %s. "
                        "Check: Verify message protocol implementation.",
                        msg[0] if msg else "empty",
                        msg,
                    )
            except Exception as e:
                logger.error(
                    "Connection listener encountered exception during message processing. "
                    "Exception type: %s. "
                    "Error: %s. "
                    "Context: Processing frames from socket. "
                    "Check: Review message handling logic and socket state.",
                    type(e).__name__,
                    e,
                )


class KVCacheRecvingThread(threading.Thread):
    def __init__(
        self,
        tp_rank: int,
        tp_size: int,
        _prefill_pp_size: int,
        engine: TransferEngine,
        local_engine_id: str,
        local_handshake_port: int,
        side_channel_port: int,
        local_kv_caches_base_addr: list[list[int]],
        block_len_per_addr: list[list[int]],
        block_stride_per_addr: list[list[int]],
        is_hma_required=False,
        ready_event: threading.Event | None = None,
        vllm_config: VllmConfig | None = None,
        kv_caches: dict[str, Any] | None = None,
        prefill_pp_layer_partition: str | None = None,
        kv_group2layeridx: dict[int, tuple[dict[str, Any], list[int]]] | None = None,
        block_size_scale: list[list[int]] | None = None,
    ):
        super().__init__(daemon=True, name="KVCacheRecvingThread")
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self._prefill_pp_size = _prefill_pp_size
        self.local_engine_id = local_engine_id
        self.local_handshake_port = local_handshake_port
        self.side_channel_port = side_channel_port
        self.engine = engine
        if ready_event is None:
            ready_event = threading.Event()
        self.ready_event = ready_event

        if kv_caches is None:
            kv_caches = {}
        self.kv_caches = kv_caches
        self.kv_caches_base_addr: dict[str, dict[int, list[list[int]]]] = SizedDict()
        self.kv_caches_base_addr[local_engine_id][local_handshake_port] = local_kv_caches_base_addr
        self.block_len_per_addr = block_len_per_addr
        self.block_stride_per_addr = block_stride_per_addr
        if kv_group2layeridx is None:
            kv_group2layeridx = {}
        self.kv_group2layeridx = kv_group2layeridx
        self.group_compress_ratios: dict[int, int] = {}
        for group_id, (group_spec, _) in self.kv_group2layeridx.items():
            compress_ratio = 1
            kv_cache_spec = group_spec.get("kv_cache_spec")
            if isinstance(kv_cache_spec, dict):
                for spec in kv_cache_spec.values():
                    if isinstance(spec, dict) and isinstance(spec.get("compress_ratio"), int):
                        compress_ratio = max(1, spec["compress_ratio"])
                        break
            self.group_compress_ratios[group_id] = compress_ratio
        self.remote_te_port: dict[str, dict[int, int]] = SizedDict()
        self.remote_block_size_scale: dict[str, dict[int, list[list[int]]]] = SizedDict()
        self.remote_block_stride_per_addr: dict[str, dict[int, list[list[int]]]] = SizedDict()
        self.remote_kv_group2layeridx: dict[str, dict[int, dict[int, tuple[dict[str, Any], list[int]]]]] = SizedDict()
        self.remote_metadata_lock = threading.Lock()

        self.request_queue: queue.Queue[Any] = queue.Queue()
        first_kv_cache = next(iter(self.kv_caches.values()))
        # NPU device selection is thread-local. Executor workers do not inherit
        # the device selected by the model worker thread and would otherwise
        # use device 0 on their first NPU operation.
        kv_cache_device = first_kv_cache[0].device
        self.executor = ThreadPoolExecutor(
            max_workers=32,
            initializer=torch.npu.set_device,
            initargs=(kv_cache_device,),
        )
        self.peer_request_queues: defaultdict[tuple[str, int], deque[dict[str, Any]]] = defaultdict(deque)
        self.active_peer_request_handlers: set[tuple[str, int]] = set()
        self.peer_request_queues_lock = threading.Lock()
        self.request_task_counts: defaultdict[str, int] = defaultdict(int)
        self.finished_request_markers: set[str] = set()
        self.request_task_counts_lock = threading.Lock()

        self.task_tracker = KVCacheTaskTracker()

        self.encoder = msgspec.msgpack.Encoder()
        self.decoder = msgspec.msgpack.Decoder(MooncakeAgentMetadata)
        self.remote_sockets_lock = threading.Lock()
        self.remote_sockets: dict[  # type: ignore
            str, deque[zmq.Socket]
        ] = defaultdict(  # type: ignore
            deque
        )
        self.timeout = 1.0  # seconds

        assert vllm_config is not None
        self.vllm_config: VllmConfig = vllm_config
        self.model_config = self.vllm_config.model_config
        self.num_speculative_tokens = (
            self.vllm_config.speculative_config.num_speculative_tokens
            if self.vllm_config.speculative_config is not None
            else 0
        )
        self.use_mla = self.model_config.is_deepseek_mla
        self.enable_sfa_dcp_replicated_indexer = enable_sfa_dcp_replicated_indexer(self.vllm_config)
        self.is_hma_required = is_hma_required
        self.block_size = self.vllm_config.cache_config.block_size
        try:
            hf_text_config = self.model_config.hf_text_config
            if hf_text_config is None:
                raise AttributeError
        except AttributeError:
            hf_text_config = self.model_config.hf_config
        self.num_layers = hf_text_config.num_hidden_layers
        total_num_layers = self.vllm_config.model_config.get_total_num_hidden_layers()
        self.index_cache_plane_base = total_num_layers if isinstance(total_num_layers, int) else self.num_layers
        if block_size_scale is None:
            block_size_scale = []
        self.block_size_scale = block_size_scale
        self.pp_layer_indices = {
            rank: get_prefill_pp_indices(self.num_layers, rank, self._prefill_pp_size, prefill_pp_layer_partition)
            for rank in range(self._prefill_pp_size)
        }
        self.proc_not_transfer_request: dict[str, bool] = {}
        self.proc_not_transfer_request_lock = threading.Lock()
        self.failed_recv_requests: set[str] = set()
        self.invalid_block_ids: set[int] = set()
        self.failed_recv_requests_lock = threading.Lock()

        self.num_draft_layers = 0
        if self.vllm_config.speculative_config is not None:
            if self.vllm_config.speculative_config.method == "mtp":
                # all MTP layer use the same kv cache layer, so only need to transfer once
                self.num_draft_layers = 1
            elif (
                hasattr(self.vllm_config.speculative_config.draft_model_config, "hf_config")
                and getattr(self.vllm_config.speculative_config.draft_model_config.hf_config, "num_hidden_layers", None)
                is not None
            ):
                self.num_draft_layers = (
                    self.vllm_config.speculative_config.draft_model_config.hf_config.num_hidden_layers
                )

    def add_request(
        self,
        request_id: str,
        remote_request_id: str,
        local_block_ids: BlockIds,
        remote_block_ids: BlockIds,
        group_pulls: list[GroupPull],
        remote_engine_id: str,
        remote_host: str,
        remote_handshake_port: int,
        remote_block_size=None,
        remote_port_send_num: dict[int, RemotePortInfo] | None = None,
        num_computed_tokens: int = 0,
        all_task_done: bool = False,
        local_block_ids_replicate_k: BlockIds | None = None,
        remote_block_ids_replicate_k: BlockIds | None = None,
    ):
        """Add a new request to the queue for processing."""
        if remote_port_send_num is None:
            remote_port_send_num = {}
        trans_info = {
            "request_id": request_id,
            "local_block_ids": local_block_ids,
            "remote_block_ids": remote_block_ids,
            "local_block_ids_replicate_k": local_block_ids_replicate_k or tuple(),
            "remote_block_ids_replicate_k": remote_block_ids_replicate_k or tuple(),
            "group_pulls": group_pulls,
            "remote_engine_id": remote_engine_id,
            "remote_request_id": remote_request_id,
            "remote_host": remote_host,
            "remote_handshake_port": remote_handshake_port,
            "num_computed_tokens": num_computed_tokens,
            "remote_port_send_num": remote_port_send_num,
            "all_task_done": all_task_done,
            "remote_block_size": remote_block_size,
        }
        logger.debug("Adding request %s to the queue.Trans info:%s", request_id, trans_info)
        self.request_queue.put(trans_info)

    def get_and_clear_finished_requests(self) -> set[str]:
        """
        Get and clear the requests that have been completed.
        Returns:
            A set of request IDs that have been completed.
        """
        return self.task_tracker.get_and_clear_finished_requests()

    def get_and_clear_invalid_block_ids(self) -> set[int]:
        """Get and clear block ids that failed to load."""
        with self.failed_recv_requests_lock:
            invalid_block_ids = self.invalid_block_ids
            self.invalid_block_ids = set()
        return invalid_block_ids

    def _is_failed_recv_request(self, request_id: str) -> bool:
        with self.failed_recv_requests_lock:
            return request_id in self.failed_recv_requests

    def _mark_failed_recv_request(self, request_id: str, local_block_ids: BlockIds) -> None:
        with self.failed_recv_requests_lock:
            self.failed_recv_requests.add(request_id)
            self.invalid_block_ids.update(local_block_ids[0])

    def _clear_failed_recv_request(self, request_id: str) -> None:
        with self.failed_recv_requests_lock:
            self.failed_recv_requests.discard(request_id)

    def run(self):
        """Run the thread to handle KV cache transfer requests."""
        self.ready_event.set()
        while True:
            try:
                request_data = self.request_queue.get()
                if request_data is None:
                    logger.warning("Received a None request. ")
                    self.request_queue.task_done()
                    continue
                self._submit_request(request_data)
            except Exception as e:
                logger.error("Error in KVCacheTransferThread. error=%s. ", e)

    def _submit_request(self, request_data: dict[str, Any]) -> None:
        peer_key = (request_data["remote_host"], request_data["remote_handshake_port"])
        self._mark_request_task_submitted(request_data)
        should_start_worker = False
        with self.peer_request_queues_lock:
            self.peer_request_queues[peer_key].append(request_data)
            if peer_key not in self.active_peer_request_handlers:
                self.active_peer_request_handlers.add(peer_key)
                should_start_worker = True

        if should_start_worker:
            self.executor.submit(self._handle_peer_requests, peer_key)

    def _handle_peer_requests(self, peer_key: tuple[str, int]) -> None:
        requests_handled = 0
        while requests_handled < MAX_REQUESTS_PER_PEER_HANDLER:
            with self.peer_request_queues_lock:
                peer_queue = self.peer_request_queues.get(peer_key)
                if not peer_queue:
                    self.peer_request_queues.pop(peer_key, None)
                    self.active_peer_request_handlers.discard(peer_key)
                    return
                req_meta = peer_queue.popleft()

            requests_handled += 1
            try:
                self._handle_request(req_meta)
            except Exception:
                logger.exception(
                    "Error handling KV cache transfer request for peer %s:%d.",
                    peer_key[0],
                    peer_key[1],
                )

        should_resubmit = False
        with self.peer_request_queues_lock:
            peer_queue = self.peer_request_queues.get(peer_key)
            if peer_queue:
                should_resubmit = True
            else:
                self.peer_request_queues.pop(peer_key, None)
                self.active_peer_request_handlers.discard(peer_key)

        if should_resubmit:
            self.executor.submit(self._handle_peer_requests, peer_key)

    def _mark_request_task_submitted(self, req_meta: dict[str, Any]) -> None:
        request_id = req_meta["request_id"]
        with self.request_task_counts_lock:
            self.request_task_counts[request_id] += 1
            if req_meta["all_task_done"]:
                self.finished_request_markers.add(request_id)

    def _mark_request_task_done(self, request_id: str, all_task_done: bool) -> bool:
        with self.request_task_counts_lock:
            pending_count = self.request_task_counts.get(request_id)
            if pending_count is None:
                return all_task_done

            pending_count -= 1
            if pending_count > 0:
                self.request_task_counts[request_id] = pending_count
                return False

            self.request_task_counts.pop(request_id, None)
            has_finished_marker = request_id in self.finished_request_markers
            self.finished_request_markers.discard(request_id)
            return has_finished_marker

    def _handle_request(self, req_meta: dict[str, Any]):
        request_id = req_meta["request_id"]
        remote_request_id = req_meta["remote_request_id"]
        remote_host = req_meta["remote_host"]
        remote_handshake_port = req_meta["remote_handshake_port"]
        remote_port_send_num = req_meta["remote_port_send_num"]
        all_task_done = req_meta["all_task_done"]
        transfer_failed = self._is_failed_recv_request(request_id)

        try:
            if transfer_failed:
                self._mark_failed_recv_request(request_id, req_meta["local_block_ids"])
                logger.warning("Skipping KV cache transfer for request. remote_request_id=%s. ", remote_request_id)
            else:
                try:
                    logger.debug("Starting to transfer KV cache for request %s.", remote_request_id)
                    self._transfer_kv_cache_all_groups(req_meta)
                    logger.debug("Finished transferring KV cache for request %s.", remote_request_id)
                except Exception as e:
                    transfer_failed = True
                    self._mark_failed_recv_request(request_id, req_meta["local_block_ids"])
                    logger.exception("Failed to transfer KV cache for request %s: %s", remote_request_id, e)
        finally:
            if self._mark_request_task_done(request_id, all_task_done):
                self.task_tracker.update_done_task_count(request_id)
                with self.proc_not_transfer_request_lock:
                    self.proc_not_transfer_request.pop(remote_request_id, None)
                self._clear_failed_recv_request(request_id)
            self.request_queue.task_done()
            self._send_done_signal_to_free_remote_port(remote_request_id, remote_host, remote_port_send_num)
            # Always send the done signal to the remote host to ensure proper
            # resource cleanup. Failing to do so may cause a memory leak on the
            # remote host.
            self._send_done_recv_signal(remote_request_id, remote_host, remote_handshake_port, remote_port_send_num)

    def _send_done_signal_to_free_remote_port(
        self, request_id: str, remote_host: str, remote_port_send_num: dict[int, RemotePortInfo]
    ):
        if self.side_channel_port != self.local_handshake_port or not remote_port_send_num:
            return
        with self.proc_not_transfer_request_lock:
            if request_id not in self.proc_not_transfer_request:
                self.proc_not_transfer_request[request_id] = True
            should_send = self.proc_not_transfer_request[request_id]
            if should_send:
                self.proc_not_transfer_request[request_id] = False
        if should_send:
            for remote_port in remote_port_send_num:
                if remote_port_send_num[remote_port]["num"] == 0:
                    remote_host_ = remote_port_send_num[remote_port]["host"]
                    self._send_done_recv_signal(request_id, remote_host_, remote_port, remote_port_send_num)

    def _transfer_kv_cache_all_groups(self, req_meta: dict[str, Any]):
        """Handle a KV cache transfer request."""
        remote_request_id = req_meta["remote_request_id"]
        local_block_ids: BlockIds = req_meta["local_block_ids"]
        remote_block_ids: BlockIds = req_meta["remote_block_ids"]
        local_block_ids_replicate_k: BlockIds = req_meta.get("local_block_ids_replicate_k", tuple())
        remote_block_ids_replicate_k: BlockIds = req_meta.get("remote_block_ids_replicate_k", tuple())
        has_replicate_k_blocks = any(local_block_ids_replicate_k) and any(remote_block_ids_replicate_k)
        group_pulls: list[GroupPull] = req_meta["group_pulls"]
        remote_engine_id = req_meta["remote_engine_id"]
        remote_host = req_meta["remote_host"]
        remote_handshake_port = req_meta["remote_handshake_port"]
        # Full prefix cache hit: do not need to read remote blocks, just notify
        # P worker that we have the blocks we need.
        num_local_blocks = sum(len(group_block_ids) for group_block_ids in local_block_ids)
        if num_local_blocks == 0 and not has_replicate_k_blocks:
            return

        # Check if we have the remote metadata cached.
        with self.remote_metadata_lock:
            has_remote_metadata = (
                remote_engine_id in self.kv_caches_base_addr
                and remote_handshake_port in self.kv_caches_base_addr[remote_engine_id]
            )
        if not has_remote_metadata:
            self._get_remote_metadata(remote_host, remote_handshake_port)
        with self.remote_metadata_lock:
            remote_kv_caches_base_addrs = self.kv_caches_base_addr[remote_engine_id][remote_handshake_port]
            local_kv_caches_base_addrs = self.kv_caches_base_addr[self.local_engine_id][self.local_handshake_port]
            remote_transfer_port = self.remote_te_port[remote_engine_id][remote_handshake_port]
            remote_block_stride_per_addr = self.remote_block_stride_per_addr[remote_engine_id][remote_handshake_port]
            remote_kv_group2layeridx = self.remote_kv_group2layeridx.get(remote_engine_id, {}).get(
                remote_handshake_port,
                self.kv_group2layeridx,
            )
        remote_layer_name_to_idx = build_layer_name_to_metadata_idx(remote_kv_group2layeridx)
        session_id = f"{remote_host}:{remote_transfer_port}"

        req_start_time = time.perf_counter()
        src_list: list[int] = []
        dst_list: list[int] = []
        length_list: list[int] = []
        attention_group_reformat_block_ids: list[tuple[tuple[int, list[list[int]], int, list[int]], bool]] = []
        grouped_remote_k_block_ids: list[list[int]] = []
        grouped_local_k_block_ids: list[list[int]] = []
        if self.enable_sfa_dcp_replicated_indexer and has_replicate_k_blocks:
            grouped_remote_k_block_ids, grouped_local_k_block_ids = group_concurrent_contiguous(
                remote_block_ids_replicate_k[0],
                local_block_ids_replicate_k[0],
            )

        def pp_layer_indices(layer_indices: list[int], prefill_pp_rank: int) -> list[int]:
            first_layer_index, end_layer_index = self.pp_layer_indices[prefill_pp_rank]
            if self.vllm_config.speculative_config is not None and prefill_pp_rank == self._prefill_pp_size - 1:
                end_layer_index += self.num_draft_layers

            def in_partition(metadata_layer_idx: int) -> bool:
                transformer_layer = (
                    metadata_layer_idx - self.index_cache_plane_base
                    if metadata_layer_idx >= self.index_cache_plane_base
                    else metadata_layer_idx
                )
                return first_layer_index <= transformer_layer < end_layer_index

            return [layer_idx for layer_idx in layer_indices if in_partition(layer_idx)]

        use_transfer_group_block_ids = transfer_groups_need_independent_block_ids(
            self.kv_group2layeridx,
            self.block_size_scale,
        )

        def get_remote_layer_idx(
            local_layer_idx: int,
            group_spec: dict[str, Any],
            local_layer_indices: list[int],
        ) -> int:
            # Older peers and lightweight tests may not provide layer names in
            # their cache metadata. Identical layouts remain position-compatible.
            if not remote_layer_name_to_idx or not group_spec.get("layer_names"):
                return local_layer_idx
            return resolve_remote_layer_idx(
                local_layer_idx,
                group_spec,
                local_layer_indices,
                remote_layer_name_to_idx,
            )

        for group_pull in group_pulls:
            group_idx = group_pull.group_id
            group_spec, layer_indices = self.kv_group2layeridx[group_idx]
            kv_cache_group_id = group_spec.get("kv_cache_group_id", group_idx)
            raw_layer_indices = layer_indices
            layer_indices = pp_layer_indices(layer_indices, group_pull.prefill_pp_rank)

            if not layer_indices:
                continue
            tp_num_need_pulls = group_pull.num_group_pulls
            inner_offset = group_pull.remote_tp_offset
            is_mamba_group = group_spec["kv_cache_spec_type"] == "MambaSpec"
            block_id_idx = group_idx if use_transfer_group_block_ids else kv_cache_group_id
            local_group_block_ids = local_block_ids[block_id_idx]
            remote_group_block_ids = remote_block_ids[block_id_idx]
            has_group_blocks = bool(local_group_block_ids)
            if not has_group_blocks and (is_mamba_group or not has_replicate_k_blocks):
                continue
            if not is_mamba_group:
                grouped_remote_block_ids: list[list[int]] = []
                grouped_local_block_ids: list[list[int]] = []
                if has_group_blocks:
                    is_group_transfer_end = group_pull.is_group_transfer_end
                    # Block ids are already expanded to kernel granularity and truncated in
                    # _get_kv_split_metadata, so consume them directly here.
                    kernel_remote_block_ids = remote_group_block_ids
                    kernel_local_block_ids = local_group_block_ids

                    if tp_num_need_pulls == 1:
                        grouped_remote_block_ids, grouped_local_block_ids = group_concurrent_contiguous(
                            kernel_remote_block_ids, kernel_local_block_ids
                        )
                    else:
                        grouped_remote_block_ids = [[block_id] for block_id in kernel_remote_block_ids]
                        grouped_local_block_ids = [[block_id] for block_id in kernel_local_block_ids]
                    attention_group_reformat_block_ids.append(
                        (
                            (group_idx, grouped_local_block_ids, tp_num_need_pulls, layer_indices),
                            is_group_transfer_end,
                        )
                    )
            else:
                # When Prefix Caching is enabled on both P and D nodes, num_block should not be forced to match,
                # as the D-node requires dynamic allocation based on its specific cache hit rate.
                transfer_block_idx = len(remote_group_block_ids) - self.num_speculative_tokens - 1
                grouped_remote_block_ids = [[remote_group_block_ids[transfer_block_idx]]]
                grouped_local_block_ids = [[local_group_block_ids[0]]]

            if is_mamba_group:
                for layer_idx in layer_indices:
                    remote_layer_idx = get_remote_layer_idx(
                        layer_idx,
                        group_spec,
                        raw_layer_indices,
                    )
                    start_meta_idx = len(src_list)
                    self._append_mamba_transfer_meta(
                        src_list,
                        dst_list,
                        length_list,
                        group_spec=group_spec,
                        src_layer_base_addr=local_kv_caches_base_addrs[layer_idx],
                        dst_layer_base_addr=remote_kv_caches_base_addrs[remote_layer_idx],
                        block_len=self.block_len_per_addr[layer_idx],
                        block_stride=self.block_stride_per_addr[layer_idx],
                        remote_block_stride=remote_block_stride_per_addr[remote_layer_idx],
                        remote_block_id=grouped_remote_block_ids[0][0],
                        local_block_id=grouped_local_block_ids[0][0],
                        tp_num_need_pulls=tp_num_need_pulls,
                        remote_tp_offset=inner_offset,
                    )
                    if logger.isEnabledFor(logging.DEBUG):
                        for src, dst, length in zip(
                            src_list[start_meta_idx:], dst_list[start_meta_idx:], length_list[start_meta_idx:]
                        ):
                            logger.debug(
                                "Mooncake mamba transfer meta: request_id=%s group_idx=%s layer_idx=%s "
                                "local_block_id=%s remote_block_id=%s tp_num_need_pulls=%s "
                                "remote_tp_offset=%s  session_id=%s",
                                remote_request_id,
                                group_idx,
                                layer_idx,
                                grouped_local_block_ids[0][0],
                                grouped_remote_block_ids[0][0],
                                tp_num_need_pulls,
                                inner_offset,
                                session_id,
                            )
                continue

            for layer_idx in layer_indices:
                remote_layer_idx = get_remote_layer_idx(
                    layer_idx,
                    group_spec,
                    raw_layer_indices,
                )
                for cache_idx in range(len(local_kv_caches_base_addrs[layer_idx])):
                    src_layer_base_addr = local_kv_caches_base_addrs[layer_idx][cache_idx]
                    dst_layer_base_addr = remote_kv_caches_base_addrs[remote_layer_idx][cache_idx]
                    block_len = self.block_len_per_addr[layer_idx][cache_idx]
                    block_stride = self.block_stride_per_addr[layer_idx][cache_idx]
                    remote_block_stride = remote_block_stride_per_addr[remote_layer_idx][cache_idx]
                    inner_block_len = block_len // tp_num_need_pulls
                    if self.enable_sfa_dcp_replicated_indexer and self.block_size_scale[layer_idx][cache_idx] > 1:
                        if has_replicate_k_blocks:
                            transfer_remote_block_ids = grouped_remote_k_block_ids
                            transfer_local_block_ids = grouped_local_k_block_ids
                        else:
                            continue
                    else:
                        if not has_group_blocks:
                            continue
                        transfer_remote_block_ids, transfer_local_block_ids = split_if_not_byte_contiguous(
                            grouped_remote_block_ids,
                            grouped_local_block_ids,
                            src_block_stride=remote_block_stride,
                            dst_block_stride=block_stride,
                            block_len=inner_block_len,
                        )
                    for remote_block_id, local_block_id in zip(transfer_remote_block_ids, transfer_local_block_ids):
                        src = src_layer_base_addr + local_block_id[0] * block_stride + inner_offset * inner_block_len
                        dst = dst_layer_base_addr + remote_block_id[0] * remote_block_stride
                        length = inner_block_len * len(local_block_id)
                        src_list.append(src)
                        dst_list.append(dst)
                        length_list.append(length)
                    logger.debug(
                        "Mooncake kv transfer meta: request_id=%s group_idx=%s layer_idx=%s local_block_ids=%s "
                        "remote_block_ids=%s tp_num_need_pulls=%s remote_tp_offset=%s session_id=%s",
                        remote_request_id,
                        group_idx,
                        layer_idx,
                        grouped_local_block_ids,
                        grouped_remote_block_ids,
                        tp_num_need_pulls,
                        inner_offset,
                        session_id,
                    )
        if not src_list:
            return

        logger.debug(
            "Mooncake transfer request=%s session id=%s src=%s dst=%s length=%s",
            remote_request_id,
            session_id,
            src_list,
            dst_list,
            length_list,
        )
        ret = self.engine.batch_transfer_sync_read(session_id, src_list, dst_list, length_list)
        if ret < 0:
            logger.error(
                "Mooncake transfer failed for request. remote_request_id=%s, ret=%d. ",
                req_meta["remote_request_id"],
                ret,
            )
            raise RuntimeError(f"Mooncake transfer failed, ret: {ret}")

        req_end_time = time.perf_counter()
        req_transfer_elapsed = (req_end_time - req_start_time) * 1000
        logger.info(
            "KV cache transfer for request %s took %.2f ms. local_ip %s local_device_id %s remote_session_id %s",
            remote_request_id,
            req_transfer_elapsed,
            get_ip(),
            self.tp_rank,
            session_id,
        )

        ready_attention_group_reformat_block_ids = []
        for reformat_group, is_group_transfer_end in attention_group_reformat_block_ids:
            if is_group_transfer_end:
                ready_attention_group_reformat_block_ids.append(reformat_group)
        if not ready_attention_group_reformat_block_ids:
            return

        gqa_reformat_groups = [
            (group_idx, grouped_local_block_ids, num_group_pulls, layer_indices)
            for (
                group_idx,
                grouped_local_block_ids,
                num_group_pulls,
                layer_indices,
            ) in ready_attention_group_reformat_block_ids
            if num_group_pulls > 1
            and not MooncakeConnectorWorker._group_skip_kv_reformat(self.kv_group2layeridx[group_idx][0])
        ]

        if self.is_hma_required:
            for group_idx, grouped_local_block_ids, num_group_pulls, layer_indices in gqa_reformat_groups:
                num_reformat_blocks = sum(len(block_ids) for block_ids in grouped_local_block_ids)
                logger.debug(
                    "Reformat hybrid linear KV cache for GQA attention group. "
                    "group_idx=%s, num_group_pulls=%s, num_block_groups=%s, num_reformat_blocks=%s, "
                    "layer_indices=%s",
                    group_idx,
                    num_group_pulls,
                    len(grouped_local_block_ids),
                    num_reformat_blocks,
                    layer_indices,
                )
                group_kv_caches = self._get_group_kv_caches(group_idx, layer_indices)
                if not group_kv_caches:
                    continue
                self.reformat_kv_cache_hybrid_linear_torch(grouped_local_block_ids, num_group_pulls, group_kv_caches)
            return

        uniform_num_pulls = {num_group_pulls for _, _, num_group_pulls, _ in ready_attention_group_reformat_block_ids}
        if len(uniform_num_pulls) != 1:
            raise RuntimeError(
                f"Non-hybrid Mooncake KV reformat expects uniform group pulls, but got {uniform_num_pulls}."
            )

        num_group_pulls = next(iter(uniform_num_pulls))
        need_cat_cache = num_group_pulls > 1
        need_nz_cache = get_ascend_config().enable_kv_nz
        if not (need_cat_cache or need_nz_cache):
            return

        use_fused_op = ascend_envs.VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK
        for group_idx, reformat_block_ids, _, layer_indices in ready_attention_group_reformat_block_ids:
            group_spec = self.kv_group2layeridx[group_idx][0]
            if MooncakeConnectorWorker._group_skip_kv_reformat(group_spec):
                continue
            group_kv_caches = self._get_group_kv_caches(group_idx, layer_indices)
            if not group_kv_caches:
                continue
            if use_fused_op and enable_custom_op():
                if need_cat_cache:
                    self.reformat_kv_cache_with_fused_op(reformat_block_ids, num_group_pulls, group_kv_caches)
                if need_nz_cache:
                    self.reformat_kv_cache(reformat_block_ids, num_group_pulls, False, need_nz_cache, group_kv_caches)
            else:
                self.reformat_kv_cache(
                    reformat_block_ids,
                    num_group_pulls,
                    need_cat_cache,
                    need_nz_cache,
                    group_kv_caches,
                )

    @torch.no_grad()
    def reformat_kv_cache_hybrid_linear_torch(
        self, block_ids: list[list[int]], tp_num_need_pulls: int, group_kv_caches
    ):
        flat_block_ids = [item for sublist in block_ids for item in sublist]
        if not flat_block_ids or tp_num_need_pulls == 1:
            return
        device = list(self.kv_caches.values())[0][0].device
        block_ids_tensor = torch.tensor(flat_block_ids, dtype=torch.long, device=device)
        num_blocks = block_ids_tensor.numel()

        def _transpose_cache_by_block(cache: torch.Tensor):
            # The transferred cache is laid out as
            # [block, split, token, head_per_split, dim]. Restore it to
            # [block, token, split, head_per_split, dim] in the selected blocks.
            selected = cache.index_select(0, block_ids_tensor)
            block_size = cache.shape[1]
            transposed = (
                selected.reshape(num_blocks, tp_num_need_pulls, block_size, -1)
                .transpose(1, 2)
                .contiguous()
                .reshape_as(selected)
            )
            cache.index_copy_(0, block_ids_tensor, transposed)

        for _, (k_cache_layer, v_cache_layer) in group_kv_caches.items():
            _transpose_cache_by_block(k_cache_layer)
            _transpose_cache_by_block(v_cache_layer)

    def _append_mamba_transfer_meta(
        self,
        src_list: list[int],
        dst_list: list[int],
        length_list: list[int],
        group_spec: dict[str, Any],
        src_layer_base_addr: list[int],
        dst_layer_base_addr: list[int],
        block_len: list[int],
        block_stride: list[int],
        remote_block_stride: list[int],
        remote_block_id: int,
        local_block_id: int,
        tp_num_need_pulls: int,
        remote_tp_offset: int,
    ) -> None:
        remote_tp_size = self.tp_size * tp_num_need_pulls
        assert remote_tp_size >= self.tp_size, "Mamba prefill TP size must be >= decode TP size."
        assert remote_tp_size % self.tp_size == 0, "Mamba prefill TP size must be divisible by decode TP size."

        remote_conv_addr, remote_ssm_addr = dst_layer_base_addr[:2]
        local_conv_addr, local_ssm_addr = src_layer_base_addr[:2]
        local_conv_len, local_ssm_len = block_len[:2]
        local_conv_stride, local_ssm_stride = block_stride[:2]
        remote_conv_stride, remote_ssm_stride = remote_block_stride[:2]

        tp_ratio = tp_num_need_pulls
        remote_conv_len = local_conv_len // tp_ratio
        remote_ssm_len = local_ssm_len // tp_ratio

        if tp_ratio == 1:
            src_list.extend(
                [
                    local_conv_addr + local_block_id * local_conv_stride,
                    local_ssm_addr + local_block_id * local_ssm_stride,
                ]
            )
            dst_list.extend(
                [
                    remote_conv_addr + remote_block_id * remote_conv_stride,
                    remote_ssm_addr + remote_block_id * remote_ssm_stride,
                ]
            )
            length_list.extend([remote_conv_len, remote_ssm_len])
            return

        conv_shape = group_spec["shapes"][0]
        conv_dtype_size = group_spec["dtype_sizes"][0]

        linear_key_head_dim = self.vllm_config.model_config.hf_text_config.linear_key_head_dim
        linear_num_key_heads = self.vllm_config.model_config.hf_text_config.linear_num_key_heads
        linear_value_head_dim = self.vllm_config.model_config.hf_text_config.linear_value_head_dim
        linear_num_value_heads = self.vllm_config.model_config.hf_text_config.linear_num_value_heads
        remote_num_key_heads = linear_num_key_heads // remote_tp_size
        remote_num_value_heads = linear_num_value_heads // remote_tp_size
        remote_conv_width = (
            remote_num_key_heads * 2 * linear_key_head_dim + remote_num_value_heads * linear_value_head_dim
        )
        remote_conv_offsets = [
            0,
            remote_num_key_heads * linear_key_head_dim,
            remote_num_key_heads * 2 * linear_key_head_dim,
        ]
        remote_conv_sizes = [
            remote_num_key_heads * linear_key_head_dim,
            remote_num_key_heads * linear_key_head_dim,
            remote_num_value_heads * linear_value_head_dim,
        ]

        for i in range(conv_shape[0]):
            for remote_conv_offset, remote_conv_size in zip(remote_conv_offsets, remote_conv_sizes):
                remote_addr_offset = (i * remote_conv_width + remote_conv_offset) * conv_dtype_size
                local_addr_offset = (
                    (i * remote_conv_width + remote_conv_offset) * tp_ratio + remote_tp_offset * remote_conv_size
                ) * conv_dtype_size
                src_list.append(local_conv_addr + local_block_id * local_conv_stride + local_addr_offset)
                dst_list.append(remote_conv_addr + remote_block_id * remote_conv_stride + remote_addr_offset)
                length_list.append(remote_conv_size * conv_dtype_size)

        src_list.append(
            local_ssm_addr + local_block_id * local_ssm_stride + remote_tp_offset * local_ssm_len // tp_num_need_pulls
        )
        dst_list.append(remote_ssm_addr + remote_block_id * remote_ssm_stride)
        length_list.append(remote_ssm_len)

    def _get_group_kv_caches(self, group_idx: int, layer_indices: list[int] | None = None) -> dict[str, Any]:
        if layer_indices is None:
            _, layer_indices = self.kv_group2layeridx[group_idx]
        layer_index_set = set(layer_indices)
        model_type = self.vllm_config.model_config.hf_text_config.model_type
        num_attn_module = 2 if model_type in ("longcat_flash", "longcat_flash_ngram") else 1
        from vllm.v1.worker.utils import extract_layer_index

        def layer_in_group(layer_name: str) -> bool:
            if "mtp" in layer_name:
                return any(layer_idx >= self.num_layers for layer_idx in layer_index_set)
            return extract_layer_index(layer_name, num_attn_module) in layer_index_set

        return {
            layer_name: layer_cache for layer_name, layer_cache in self.kv_caches.items() if layer_in_group(layer_name)
        }

    @staticmethod
    def _get_kv_cache_dims_from_tensors(kv_caches: dict[str, Any]) -> tuple[int, int, int]:
        """Return (num_kv_heads, k_head_dim, v_head_dim) from registered KV cache tensors."""
        k_cache, v_cache = next(iter(kv_caches.values()))
        return int(k_cache.shape[-2]), int(k_cache.shape[-1]), int(v_cache.shape[-1])

    def reformat_kv_cache_with_fused_op(
        self,
        block_ids: list[list[int]],
        tp_num_need_pulls: int,
        kv_caches: dict[str, Any] | None = None,
    ):
        if kv_caches is None:
            kv_caches = self.kv_caches
        k_cache = list(kv_caches.values())[0][0]
        device = k_cache.device
        num_kv_head, head_dim, _ = self._get_kv_cache_dims_from_tensors(kv_caches)
        block_size = self.vllm_config.cache_config.block_size
        layers = len(kv_caches)
        flat_block_ids = [item for sublist in block_ids for item in sublist]
        block_ids_tensor = torch.tensor(flat_block_ids, dtype=torch.int64, device=device)

        k_caches = []
        v_caches = []
        for _, (k_cache_layer, v_cache_layer) in kv_caches.items():
            k_caches.append(k_cache_layer)
            v_caches.append(v_cache_layer)

        torch.ops._C_ascend.transpose_kv_cache_by_block(
            k_caches, v_caches, block_ids_tensor, block_size, num_kv_head, head_dim, tp_num_need_pulls, layers
        )

    def reformat_kv_cache(
        self,
        block_ids: list[list[int]],
        tp_num_need_pulls: int,
        need_cat_cache: bool = False,
        need_nz_cache: bool = False,
        kv_caches: dict[str, Any] | None = None,
    ):
        if kv_caches is None:
            kv_caches = self.kv_caches
        k_cache = list(kv_caches.values())[0][0]
        dtype = k_cache.dtype
        device = k_cache.device
        num_kv_heads, k_head_dim, v_head_dim = self._get_kv_cache_dims_from_tensors(kv_caches)

        flat_block_ids = [item for sublist in block_ids for item in sublist]
        block_ids_tensor = torch.tensor(flat_block_ids, dtype=torch.int32, device=device)
        num_blocks = len(flat_block_ids)
        num_tokens = num_blocks * self.block_size

        # Create device tensors for copy operations
        block_table = block_ids_tensor.view(1, -1)
        block_len_tensor = torch.tensor([num_tokens], dtype=torch.int32, device=device)
        seq_start_tensor = torch.tensor([0], dtype=torch.int32, device=device)

        k_buffer = torch.empty((num_tokens, num_kv_heads, k_head_dim), dtype=dtype, device=device)
        v_buffer = torch.empty((num_tokens, num_kv_heads, v_head_dim), dtype=dtype, device=device)

        # Create slot mapping for reshape operations
        block_offsets = torch.arange(0, self.block_size, dtype=torch.int32, device=device)
        slot_mapping = (
            block_offsets.reshape((1, self.block_size)) + block_ids_tensor.reshape((num_blocks, 1)) * self.block_size
        ).flatten()

        # FIXME: Right now, if we skip synchronization at this point, the system
        # will crash in GQA scenarios. However, we still haven't identified the
        # root cause.
        torch.npu.synchronize()

        # Process each layer in the KV cache
        for _, (k_cache_layer, v_cache_layer) in kv_caches.items():
            # Load cache data into buffers
            torch_npu.npu_gather_pa_kv_cache(
                k_cache_layer,
                v_cache_layer,
                block_table,
                block_len_tensor,
                seq_offset=seq_start_tensor,
                key=k_buffer,
                value=v_buffer,
            )
            if need_cat_cache:
                self._cat_kv_cache(
                    k_cache_layer,
                    v_cache_layer,
                    k_buffer,
                    v_buffer,
                    tp_num_need_pulls,
                    num_blocks,
                    num_tokens,
                    slot_mapping,
                    num_kv_heads,
                )
            if need_nz_cache:
                self._nz_kv_cache(
                    k_cache_layer,
                    v_cache_layer,
                    k_buffer,
                    v_buffer,
                    slot_mapping,
                    num_kv_heads,
                    k_head_dim,
                    v_head_dim,
                )
        # Clean up buffers
        del k_buffer, v_buffer

    def _cat_kv_cache(
        self,
        k_cache_layer,
        v_cache_layer,
        k_buffer,
        v_buffer,
        tp_num_need_pulls,
        num_blocks,
        num_tokens,
        slot_mapping,
        num_kv_heads: int,
    ):
        def _transpose_kv_cache_between_head(buffer: torch.Tensor) -> torch.Tensor:
            buffer = buffer.view(num_blocks, tp_num_need_pulls, self.block_size, -1)
            buffer.transpose_(1, 2)
            return buffer.contiguous().view(num_tokens, num_kv_heads, -1)

        # Transpose KV cache
        k_buffer = _transpose_kv_cache_between_head(k_buffer)
        v_buffer = _transpose_kv_cache_between_head(v_buffer)

        # Reshape and cache the processed buffers
        torch_npu.npu_scatter_pa_kv_cache(
            key=k_buffer,
            value=v_buffer,
            key_cache=k_cache_layer,
            value_cache=v_cache_layer,
            slot_mapping=slot_mapping,
            cache_mode="Norm",
        )

    def _nz_kv_cache(
        self,
        k_cache_layer,
        v_cache_layer,
        k_buffer,
        v_buffer,
        slot_mapping,
        num_kv_heads: int,
        k_head_dim: int,
        v_head_dim: int,
    ):
        nz_fmt_last_dim = 16
        k_cache_layer = k_cache_layer.view(
            -1, k_head_dim * num_kv_heads // nz_fmt_last_dim, self.block_size, nz_fmt_last_dim
        )
        v_cache_layer = v_cache_layer.view(
            -1, v_head_dim * num_kv_heads // nz_fmt_last_dim, self.block_size, nz_fmt_last_dim
        )
        torch_npu.npu_scatter_pa_kv_cache(k_buffer, v_buffer, k_cache_layer, v_cache_layer, slot_mapping)

    def _get_remote_metadata(self, remote_host: str, remote_handshake_port: int) -> None:
        """Get the metadata from the remote host."""
        sock: zmq.Socket | None = None  # type: ignore
        try:
            sock = self._get_remote_socket(remote_host, remote_handshake_port)
            ensure_zmq_send(sock, self.encoder.encode((GET_META_MSG, "")), f"{remote_host}:{remote_handshake_port}")
            metadata_bytes = ensure_zmq_recv(sock, f"{remote_host}:{remote_handshake_port}")
            agent_meta = self.decoder.decode(metadata_bytes)
            engine_id = agent_meta.engine_id
            assert engine_id != self.local_engine_id, (
                f"Conflict engine id {engine_id} with local engine id {self.local_engine_id}."
            )
            if agent_meta.kv_group2layeridx != self.kv_group2layeridx:
                logger.debug(
                    "Remote kv_group2layeridx differs from local. Inspect the remote and local metadata "
                    "to determine whether the difference is expected for the configured parallelism "
                    "and KV cache layouts. remote=%s, local=%s. ",
                    agent_meta.kv_group2layeridx,
                    self.kv_group2layeridx,
                )
            with self.remote_metadata_lock:
                self.remote_kv_group2layeridx[engine_id][remote_handshake_port] = agent_meta.kv_group2layeridx
                self.kv_caches_base_addr[engine_id][remote_handshake_port] = agent_meta.kv_caches_base_addr
                self.remote_te_port[engine_id][remote_handshake_port] = agent_meta.te_rpc_port
                self.remote_block_size_scale[engine_id][remote_handshake_port] = agent_meta.block_size_scale
                self.remote_block_stride_per_addr[engine_id][remote_handshake_port] = agent_meta.block_strides
        except Exception:
            if isinstance(sock, zmq.Socket):  # type: ignore
                sock.close()
                sock = None
            raise
        finally:
            if sock is not None:
                self._return_remote_socket(sock, remote_host, remote_handshake_port)
                logger.debug("Returned socket to pool for %s:%d", remote_host, remote_handshake_port)

    def _send_done_recv_signal(
        self,
        request_id: str,
        remote_host: str,
        remote_handshake_port: int,
        remote_port_send_num: dict[int, RemotePortInfo],
    ):
        logger.debug(
            "Sending done recving signal for request %s to %s:%d", request_id, remote_host, remote_handshake_port
        )
        sock: zmq.Socket | None = None  # type: ignore
        try:
            sock = self._get_remote_socket(remote_host, remote_handshake_port)
            data_bytes = self.encoder.encode((DONE_RECVING_MSG, request_id, remote_port_send_num))
            ensure_zmq_send(sock, data_bytes, f"{remote_host}:{remote_handshake_port}")
            resp = ensure_zmq_recv(sock, f"{remote_host}:{remote_handshake_port}")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Received response for request %s: %s", request_id, resp.decode("utf-8"))
            if resp != b"ACK":
                logger.error(
                    "Failed to receive ACK for request. request_id=%s, source=%s:%d. ",
                    request_id,
                    remote_host,
                    remote_handshake_port,
                )
                raise RuntimeError(f"Failed to receive ACK, resp: {resp.decode('utf-8')}")
        except RuntimeError as e:
            if isinstance(sock, zmq.Socket):  # type: ignore
                sock.close()
                sock = None
                logger.warning("Unexpected error occurred in socket. error=%s. ", e)
        finally:
            if sock is not None:
                self._return_remote_socket(sock, remote_host, remote_handshake_port)
                logger.debug("Returned socket to pool for %s:%d", remote_host, remote_handshake_port)

    def _get_remote_socket(self, remote_host: str, remote_handshake_port: int) -> zmq.Socket:  # type: ignore
        """Get a socket to the remote host."""
        remote_path = make_zmq_path("tcp", remote_host, remote_handshake_port)
        with self.remote_sockets_lock:
            if self.remote_sockets[remote_path]:
                return self.remote_sockets[remote_path].popleft()

            ctx = zmq.Context()  # type: ignore
            sock = make_zmq_socket(
                ctx=ctx,
                path=remote_path,
                socket_type=zmq.REQ,  # type: ignore
                bind=False,
            )
            sock.setsockopt(
                zmq.SNDTIMEO,  # type: ignore
                int(self.timeout * 1000),
            )
            sock.setsockopt(
                zmq.RCVTIMEO,  # type: ignore
                int(self.timeout * 1000),
            )
            return sock

    def _return_remote_socket(
        self,
        sock: zmq.Socket,  # type: ignore
        remote_host: str,
        remote_handshake_port: int,
    ) -> None:
        """Return the remote socket to the pool."""
        remote_path = make_zmq_path("tcp", remote_host, remote_handshake_port)
        with self.remote_sockets_lock:
            self.remote_sockets[remote_path].append(sock)


class MooncakeConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        self.requests: dict[str, ReqMeta] = {}
        self.requests_to_send: dict[str, float] = {}
        self.reqs_in_batch: set[str] = set()

    def add_new_req(
        self,
        request_id: str,
        local_block_ids: BlockIds,
        num_external_tokens: int,
        kv_transfer_params: dict[str, Any],
        local_full_block_ids: BlockIds | None = None,
    ):
        self.requests[request_id] = ReqMeta(
            local_block_ids=local_block_ids,
            num_external_tokens=num_external_tokens,
            num_computed_tokens=kv_transfer_params.get("num_computed_tokens", 0),
            remote_block_ids=kv_transfer_params["remote_block_ids"],
            remote_engine_id=kv_transfer_params["remote_engine_id"],
            remote_request_id=kv_transfer_params["remote_request_id"],
            remote_host=kv_transfer_params["remote_host"],
            remote_port=kv_transfer_params["remote_port"],
            remote_pcp_size=kv_transfer_params.get("remote_pcp_size", 1),
            remote_dcp_size=kv_transfer_params.get("remote_dcp_size", 1),
            remote_ptp_size=kv_transfer_params.get("remote_ptp_size"),
            remote_multi_nodes_meta_mapping=kv_transfer_params.get("remote_multi_nodes_meta_mapping", {}),
            num_prompt_blocks=kv_transfer_params.get("num_prompt_blocks", 0),
            remote_block_size=kv_transfer_params.get("remote_block_size", 0),
            local_full_block_ids=local_full_block_ids or tuple(),
            do_virtual=kv_transfer_params.get("do_virtual", False),
        )


class MooncakeConnector(KVConnectorBase_V1, SupportsHMA):
    # main2main compat: upstream KVConnectorBase_V1.__init__() now sets
    # _kv_transfer_config (required by requires_kv_delivery property in
    # Scheduler.__init__() post-0.26.0).
    # Remove the version gate once 0.26.0 support is dropped.
    if vllm_version_is("0.26.0"):

        def __init__(
            self, vllm_config: VllmConfig, role: KVConnectorRole, kv_cache_config: KVCacheConfig | None = None
        ):
            assert vllm_config.kv_transfer_config is not None
            self.engine_id = vllm_config.kv_transfer_config.engine_id
            self._connector_metadata = MooncakeConnectorMetadata()

            if role == KVConnectorRole.SCHEDULER:
                self.connector_scheduler: MooncakeConnectorScheduler | None = MooncakeConnectorScheduler(
                    vllm_config, str(self.engine_id), kv_cache_config
                )
                self.connector_worker: MooncakeConnectorWorker | None = None
            elif role == KVConnectorRole.WORKER:
                self.connector_scheduler = None
                self.connector_worker = MooncakeConnectorWorker(vllm_config, str(self.engine_id), kv_cache_config)
    else:

        def __init__(  # type: ignore[misc]
            self, vllm_config: VllmConfig, role: KVConnectorRole, kv_cache_config: KVCacheConfig | None = None
        ):
            assert vllm_config.kv_transfer_config is not None
            self._kv_transfer_config = vllm_config.kv_transfer_config
            self.engine_id = vllm_config.kv_transfer_config.engine_id
            self._connector_metadata = MooncakeConnectorMetadata()

            if role == KVConnectorRole.SCHEDULER:
                self.connector_scheduler: MooncakeConnectorScheduler | None = MooncakeConnectorScheduler(  # type: ignore[no-redef]
                    vllm_config, str(self.engine_id), kv_cache_config
                )
                self.connector_worker: MooncakeConnectorWorker | None = None  # type: ignore[no-redef]
            elif role == KVConnectorRole.WORKER:
                self.connector_scheduler = None
                self.connector_worker = MooncakeConnectorWorker(vllm_config, str(self.engine_id), kv_cache_config)

    ############################################################
    # Scheduler Side Methods
    ############################################################

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, (block_ids,))

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    ############################################################
    # Worker Side Methods
    ############################################################
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """Get the finished recving and sending requests."""
        assert self.connector_worker is not None
        return self.connector_worker.get_finished()

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Get the block ids whose KV load failed."""
        assert self.connector_worker is not None
        return self.connector_worker.get_block_ids_with_load_errors()

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, MooncakeConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """MooncakeConnector does not do layerwise saving."""
        pass

    def save_kv_layer(
        self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: "AttentionMetadata", **kwargs
    ) -> None:
        """MooncakeConnector does not save explicitly."""
        pass

    def wait_for_save(self):
        """MooncakeConnector does not save explicitly."""
        pass

    def get_handshake_metadata(self) -> KVConnectorHandshakeMetadata | None:
        """
        Get the KVConnector handshake metadata for this connector.
        This metadata is used for out-of-band connector handshake
        between P/D workers.

        Returns:
            KVConnectorHandshakeMetadata: the handshake metadata.
            None if no handshake metadata is available.
        """
        assert self.connector_worker is not None
        return self.connector_worker.xfer_handshake_metadata

    def set_xfer_handshake_metadata(
        self, metadata: Mapping[int | tuple[int, ...], KVConnectorHandshakeMetadata]
    ) -> None:
        """
        Set the KV connector handshake metadata for this connector.

        Args:
            metadata (dict): the handshake metadata to set.
        """
        assert self.connector_scheduler is not None
        self.connector_scheduler.set_xfer_handshake_metadata(metadata)

    def set_xfer_handshake_metadata_pp_aware(
        self, metadata: Mapping[int | tuple[int, ...], KVConnectorHandshakeMetadata]
    ) -> None:
        assert self.connector_scheduler is not None
        self.connector_scheduler.set_xfer_handshake_metadata_from_workers(metadata)


class MooncakeConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str, kv_cache_config: KVCacheConfig):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        init_ascend_config(vllm_config)
        self.ascend_config = get_ascend_config()
        self.block_size = vllm_config.cache_config.block_size
        self.engine_id = engine_id
        self.local_ip = get_ip()
        logger.info("Initializing Mooncake Scheduler %s", engine_id)

        self.side_channel_host = get_ip()
        self.pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
        self.dcp_size = vllm_config.parallel_config.decode_context_parallel_size
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.max_device_id = (
            vllm_config.parallel_config.tensor_parallel_size
            * vllm_config.parallel_config.data_parallel_size
            * self.pcp_size
            * vllm_config.parallel_config.pipeline_parallel_size
        )

        # Handshake base port
        self.side_channel_port = (
            vllm_config.kv_transfer_config.kv_port
            + vllm_config.parallel_config.data_parallel_rank
            * vllm_config.parallel_config.tensor_parallel_size
            * vllm_config.parallel_config.pipeline_parallel_size
            * self.pcp_size
        )
        # Requests that need to start recv.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        self._reqs_need_recv: dict[str, tuple[Request, BlockIds, BlockIds, int]] = {}
        self._reqs_need_send: dict[str, float] = {}
        self._reqs_in_batch: set[str] = set()

        # master-slave meta information for cross-nodes
        self.multi_nodes_meta_mapping: dict[str, dict[str, Any]] = {}
        self.kv_cache_groups = kv_cache_config.kv_cache_groups
        self.use_compress = self._model_uses_compress()
        self.group_transfer_info = [self._get_group_transfer_info(group) for group in kv_cache_config.kv_cache_groups]
        self.need_truncate = self.use_compress or any(info.is_state_group for info in self.group_transfer_info)

    def _model_uses_compress(self) -> bool:
        hf_config = getattr(self.vllm_config.model_config, "hf_config", None)
        compress_ratios = getattr(hf_config, "compress_ratios", None)
        return isinstance(compress_ratios, (list, tuple, dict))

    def _get_group_transfer_info(self, group: Any) -> GroupTransferInfo:
        specs = self._get_group_unique_specs(group)
        first_spec = specs[0] if specs else group.kv_cache_spec
        block_size = getattr(group.kv_cache_spec, "block_size", getattr(first_spec, "block_size", self.block_size))
        is_state_group = any(isinstance(spec, MambaSpec) for spec in specs)
        sliding_window = 0
        compress_ratio = 1
        for spec in specs:
            if isinstance(spec, SlidingWindowSpec):
                sliding_window = spec.sliding_window
            elif hasattr(spec, "compress_ratio"):
                compress_ratio = spec.compress_ratio

        return GroupTransferInfo(
            tokens_per_block=block_size * max(1, int(compress_ratio)),
            blocks_per_window=cdiv(sliding_window, block_size) + 1 if sliding_window else 0,
            is_state_group=is_state_group,
        )

    def _get_group_unique_specs(self, group: Any) -> list[Any]:
        if not isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
            return [group.kv_cache_spec]

        specs = []
        for layer_name in group.layer_names:
            layer_spec = group.kv_cache_spec.kv_cache_specs[layer_name]
            if layer_spec not in specs:
                specs.append(layer_spec)
        return specs

    def _get_transfer_block_ids(self, block_ids: BlockIds, prompt_len: int) -> BlockIds:
        """Return blocks that contain prompt KV, dropping MTP extra blocks.

        State groups such as Mamba are not context-block aligned with attention
        KV, so keep them unchanged and only clip attention-like groups here.
        SWA tail clipping is handled as a separate step after this.
        """
        if len(block_ids) == 0:
            return block_ids

        assert len(block_ids) == len(self.group_transfer_info), "Number of KV cache groups must match"

        transfer_block_ids = []
        cp_size = max(1, self.pcp_size * self.dcp_size)
        for blocks, group_info in zip(block_ids, self.group_transfer_info):
            if group_info.is_state_group:
                transfer_block_ids.append(blocks)
            else:
                # In context parallelism, each scheduler-visible block id is a
                # CP-grouped/virtual block shared by all CP ranks. It therefore
                # covers cp_size times the token span of one no-CP block.
                num_prompt_blocks = cdiv(prompt_len, group_info.tokens_per_block * cp_size)
                transfer_block_ids.append(blocks[:num_prompt_blocks])
        return tuple(transfer_block_ids)

    def _get_swa_transfer_block_ids(self, block_ids: BlockIds) -> BlockIds:
        """Clip SWA groups to their window tail and drop placeholder block 0."""
        if len(block_ids) == 0:
            return block_ids

        assert len(block_ids) == len(self.group_transfer_info), "Number of KV cache groups must match"

        transfer_block_ids = []
        for blocks, group_info in zip(block_ids, self.group_transfer_info):
            if group_info.is_state_group or group_info.blocks_per_window == 0:
                transfer_block_ids.append(blocks)
            else:
                window_blocks = blocks[-group_info.blocks_per_window :]
                transfer_block_ids.append([block_id for block_id in window_blocks if block_id != 0])
        return tuple(transfer_block_ids)

    def _state_prefill_token_count(self, num_prompt_tokens: int) -> int:
        """D-side only. Returns N-1 for Mamba models since the decoder
        always recomputes the last token and must start from h(N-1)."""
        if self.need_truncate and num_prompt_tokens > 1:
            return num_prompt_tokens - 1
        return num_prompt_tokens

    def _truncate_request_for_prefill(self, request: "Request") -> None:
        """P-side only: drop the last prompt token so the prefiller computes
        h(N-1) instead of h(N). The decoder recomputes the last token to
        derive h(N) correctly.

        Guarded by ``_p_side_truncated`` to avoid repeated truncation if the
        request is preempted and rescheduled."""
        params = request.kv_transfer_params
        if (
            params is not None
            # Guard against repeated truncation after preemption/reschedule.
            and not params.get("_p_side_truncated")
            and request.num_prompt_tokens > 1
        ):
            if request.prompt_token_ids is not None:
                request.prompt_token_ids.pop()
            elif request.prompt_embeds is not None:
                request.prompt_embeds = request.prompt_embeds[:-1]
            else:
                return

            request._all_token_ids.pop()
            request.num_prompt_tokens -= 1
            request.max_tokens = 1
            params["_p_side_truncated"] = True

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
        """
        For remote prefill, pull all prompt blocks from remote
        asynchronously relative to engine execution.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request
        Returns:
            * the number of tokens that can be loaded from the
              external KV cache beyond what is already computed.
            * true if the external KV cache tokens will be loaded
              asynchronously (between scheduler steps).
        """

        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector get_num_new_matched_tokens: num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens,
            params,
        )

        if params is not None and params.get("do_remote_prefill"):
            # Remote prefill: get all prompt blocks from remote.
            token_ids = request.prompt_token_ids or []
            actual = self._state_prefill_token_count(len(token_ids))
            params["num_computed_tokens"] = num_computed_tokens
            count = max(actual - num_computed_tokens, 0)
            if count > 0:
                return count, True

        if params is not None and params.get("do_remote_decode") and self.need_truncate:
            self._truncate_request_for_prefill(request)

        # No remote prefill for this request.
        return 0, False

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector update_state_after_alloc: num_external_tokens=%s, kv_transfer_params=%s",
            num_external_tokens,
            params,
        )

        if params is not None and (params.get("do_remote_prefill", False) or params.get("do_remote_decode", False)):
            self._reqs_in_batch.add(request.request_id)
        if params is not None and params.get("do_remote_prefill"):
            if params.get("remote_block_ids"):
                if all(p in params for p in ("remote_engine_id", "remote_host", "remote_port", "remote_request_id")):
                    local_block_ids = blocks.get_unhashed_block_ids_all_groups() if num_external_tokens > 0 else []
                    local_full_block_ids = blocks.get_block_ids() if num_external_tokens > 0 else tuple()
                    # Get unhashed blocks to pull from remote.
                    self._reqs_need_recv[request.request_id] = (
                        request,
                        local_block_ids,
                        local_full_block_ids,
                        num_external_tokens,
                    )
                else:
                    logger.warning("Got invalid KVTransferParams. params=%s. ", params)
            else:
                assert num_external_tokens == 0
            # Only trigger 1 KV transfer per request.
            params["do_remote_prefill"] = False

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = MooncakeConnectorMetadata()

        # Loop through scheduled reqs and convert to ReqMeta.
        for req_id, (req, block_ids, full_block_ids, num_external_tokens) in self._reqs_need_recv.items():
            assert req.kv_transfer_params is not None
            # For the case where there are no remote blocks to pull
            # (block_ids is empty), we don't need to schedule
            # an async read on the worker side.
            meta.add_new_req(
                request_id=req_id,
                local_block_ids=block_ids,
                local_full_block_ids=full_block_ids,
                num_external_tokens=num_external_tokens,
                kv_transfer_params=req.kv_transfer_params,
            )

        # Clear the list once workers start the transfers
        self._reqs_need_recv.clear()
        meta.requests_to_send = self._reqs_need_send
        self._reqs_need_send = {}
        meta.reqs_in_batch = self._reqs_in_batch
        self._reqs_in_batch = set()

        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: BlockIds,
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """

        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector request_finished, request_status=%s, kv_transfer_params=%s", request.status, params
        )

        if (
            params is None
            or not params.get("do_remote_decode")
            or request.status != RequestStatus.FINISHED_LENGTH_CAPPED
        ):
            return False, None

        num_prompt_blocks = math.ceil(len(request.prompt_token_ids) / self.block_size)
        computed_block_ids = self._get_transfer_block_ids(block_ids, len(request.prompt_token_ids))
        computed_block_ids = self._get_swa_transfer_block_ids(computed_block_ids)
        computed_block_lens = [len(block_id_list) for block_id_list in computed_block_ids]
        delay_free_blocks = sum(computed_block_lens) > 0
        if delay_free_blocks:
            logger.info("Delaying free of %d blocks for request %s", sum(computed_block_lens), request.request_id)
            self._reqs_need_send[request.request_id] = time.time()

        return delay_free_blocks, dict(
            do_remote_prefill=True,
            do_remote_decode=False,
            remote_block_ids=computed_block_ids,
            remote_engine_id=self.engine_id,
            remote_request_id=request.request_id,
            remote_host=self.side_channel_host,
            remote_port=self.side_channel_port,
            remote_pcp_size=self.pcp_size,
            remote_dcp_size=self.dcp_size,
            remote_ptp_size=self.tp_size,
            last_token_id=request.output_token_ids[-1],
            remote_multi_nodes_meta_mapping=self.multi_nodes_meta_mapping,
            num_prompt_blocks=num_prompt_blocks,
            remote_block_size=self.block_size,
        )

    def _port_offset_from_handshake_metadata(
        self,
        rank_metadata: KVConnectorHandshakeMetadata,
        metadata_key: int | tuple[int, ...],
    ) -> int:
        kv_port = self.vllm_config.kv_transfer_config.kv_port
        handshake_port = getattr(rank_metadata, "handshake_port", 0)
        if handshake_port > 0:
            return handshake_port - kv_port
        if isinstance(metadata_key, int):
            return metadata_key
        raise ValueError(f"Mooncake handshake metadata missing handshake_port for worker key {metadata_key}")

    def set_xfer_handshake_metadata_from_workers(
        self,
        metadata: Mapping[int | tuple[int, ...], KVConnectorHandshakeMetadata],
    ) -> None:
        """Build host mapping for one DP group that may span multiple nodes."""
        if not metadata:
            return

        updated_mapping: dict[str, dict[str, Any]] = {}
        kv_port = self.vllm_config.kv_transfer_config.kv_port
        for metadata_key, rank_metadata in metadata.items():
            port_offset = self._port_offset_from_handshake_metadata(rank_metadata, metadata_key)
            updated_mapping[str(port_offset)] = {
                "host": rank_metadata.local_ip,
                "engine_id": rank_metadata.engine_id,
                "handshake_port": kv_port + port_offset,
            }

        self.multi_nodes_meta_mapping.update(updated_mapping)
        logger.info(
            "MooncakeConnector set_xfer_handshake_metadata: worker_count=%d, updated=%s, multi_nodes_meta_mapping=%s",
            len(metadata),
            updated_mapping,
            self.multi_nodes_meta_mapping,
        )

    def set_xfer_handshake_metadata(
        self, metadata: Mapping[int | tuple[int, ...], KVConnectorHandshakeMetadata]
    ) -> None:
        """Legacy int-keyed entry point (port offset keys)."""
        self.set_xfer_handshake_metadata_from_workers(metadata)


class MooncakeConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str, kv_cache_config: KVCacheConfig):
        self._get_prefill_decode_size(vllm_config)
        os.environ["ASCEND_TRANSFER_TIMEOUT"] = str(get_transfer_timeout_value())
        if self._prefill_tp_size < self._decode_tp_size:
            raise ValueError(
                f"prefill_tp_size: {self._prefill_tp_size} must be greater than"
                f" or equal to the decode_tp_size: {self._decode_tp_size}"
            )

        # Metadata.
        self.vllm_config = vllm_config
        self.ascend_config = get_ascend_config()
        self.engine_id = engine_id
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.tp_group = get_tp_group()
        self.pp_rank = get_pp_group().rank_in_group
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank_local
        self.dp_size = vllm_config.parallel_config.data_parallel_size_local
        self.pp_size = vllm_config.parallel_config.pipeline_parallel_size
        self.kv_caches: dict[str, torch.Tensor] = {}
        self.side_channel_host = get_ip()
        self.pcp_size = get_pcp_group().world_size
        self.total_layers = vllm_config.model_config.get_total_num_hidden_layers()
        # Assert that pp_size and pcp_size cannot both be greater than 1
        assert not (self.pp_size > 1 and self.pcp_size > 1), "pp and pcp cannot open in same time"
        self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0
        self.dcp_size = get_decode_context_model_parallel_world_size()
        self.dcp_rank = get_decode_context_model_parallel_rank() if self.dcp_size > 1 else 0

        self.max_device_id = self.tp_size * self.dp_size * self.pcp_size * self.pp_size
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.num_key_value_heads = self.vllm_config.model_config.hf_text_config.num_key_value_heads

        # kv cache config
        self.kv_cache_config = kv_cache_config
        self.num_blocks: int = kv_cache_config.num_blocks
        self.kv_group2layeridx: dict[int, tuple[dict[str, Any], list[int]]] = {}
        _is_fullattn_group = []
        for g in self.kv_cache_config.kv_cache_groups:
            if isinstance(g.kv_cache_spec, UniformTypeKVCacheSpecs):
                is_fullattn_spec = []
                for layer_name in g.layer_names:
                    layer_spec = g.kv_cache_spec.kv_cache_specs[layer_name]
                    is_fullattn_spec.append(isinstance(layer_spec, FullAttentionSpec))
                _is_fullattn_group.append(all(is_fullattn_spec))
            else:
                _is_fullattn_group.append(isinstance(g.kv_cache_spec, FullAttentionSpec))
        self.use_hybrid = (
            not self.vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
            and not all(_is_fullattn_group)
            and len(self.kv_cache_config.kv_cache_groups) > 1
        )
        self._is_hma_required = not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager and not all(
            _is_fullattn_group
        )
        self._layer_specs = self._build_layer_specs_from_kv_cache_config(kv_cache_config)

        # Handshake base port
        self.side_channel_port = (
            vllm_config.kv_transfer_config.kv_port
            + vllm_config.parallel_config.data_parallel_rank
            * vllm_config.parallel_config.tensor_parallel_size
            * vllm_config.parallel_config.pipeline_parallel_size
            * self.pcp_size
        )
        device_index = (self.pp_rank * self.pcp_size + self.pcp_rank) * self.tp_size + self.tp_rank
        self.handshake_port = self.side_channel_port + device_index
        self.sockets: dict = {}
        device_name = str(torch.npu.current_device()) if self.pp_size > 1 else None
        self.engine = global_te.get_transfer_engine(
            self.side_channel_host,
            device_name=device_name,
        )
        self.te_rpc_port = self.engine.get_rpc_port()

        # Background thread for sending or receiving KV caches.
        self.kv_send_thread: KVCacheSendingThread | None = None
        self.kv_recv_thread: KVCacheRecvingThread | None = None

        # Handshake metadata of this worker
        self.xfer_handshake_metadata: MooncakeAgentMetadata | None = None

        # kv_transfer variables
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        if self.vllm_config.model_config.is_deepseek_mla:
            self.tp_num_need_pulls = 1
        else:
            num_d_block_heads = max(1, self.num_key_value_heads // self.tp_size)
            num_p_block_heads = max(1, self.num_key_value_heads // self._prefill_tp_size)
            self.tp_num_need_pulls = num_d_block_heads // num_p_block_heads
        self.local_remote_block_port_mapping: dict[str, list[list[int]] | None] = {}
        self.remote_port_send_num: dict[str, dict[int, RemotePortInfo]] = {}


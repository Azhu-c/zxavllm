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

"""Read-only scheduler snapshots for the local-pair motivation experiment.

Loaded by vLLM via --scheduler-cls in the decoder process. This subclasses the
same AsyncScheduler used by the reference's async configuration and never
changes its decisions or request queues. A snapshot is an observation after a
schedule() call; it is not a device timeline or proof that a waiting KV is ready.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from vllm.v1.core.sched.async_scheduler import AsyncScheduler


class ProbeAsyncScheduler(AsyncScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        state_dir = os.environ.get("BUBBLE_STATE_DIR")
        if not state_dir or not os.path.isabs(state_dir):
            raise RuntimeError("BUBBLE_STATE_DIR must be an absolute directory")
        self._bubble_state_dir = Path(state_dir)
        self._bubble_state_dir.mkdir(parents=True, exist_ok=True)
        self._bubble_rank = int(self.vllm_config.parallel_config.data_parallel_rank)
        self._bubble_interval = float(os.environ.get("BUBBLE_SNAPSHOT_INTERVAL", "0.25"))
        if self._bubble_interval <= 0:
            raise RuntimeError("BUBBLE_SNAPSHOT_INTERVAL must be positive")
        self._bubble_last = float("-inf")
        self._bubble_seq = 0

    @staticmethod
    def _bubble_request(req):
        status = getattr(req.status, "name", str(req.status))
        return {
            "request_id": req.request_id,
            "status": status,
            "prompt_tokens": int(req.num_prompt_tokens),
            "computed_tokens": int(req.num_computed_tokens),
            "output_tokens": int(req.num_output_tokens),
            "preemptions": int(getattr(req, "num_preemptions", 0)),
        }

    def schedule(self, *args, **kwargs):
        output = super().schedule(*args, **kwargs)
        now = time.monotonic()
        if now - self._bubble_last < self._bubble_interval:
            return output
        self._bubble_last = now
        self._bubble_seq += 1
        waiting = []
        for queue_name in ("waiting", "skipped_waiting"):
            for req in getattr(self, queue_name, ()):
                item = self._bubble_request(req)
                item["queue"] = queue_name
                waiting.append(item)
        state = {
            "rank": self._bubble_rank,
            "pid": os.getpid(),
            "seq": self._bubble_seq,
            "wall_time_ns": time.time_ns(),
            "max_num_seqs": self.max_num_running_reqs,
            "running": [self._bubble_request(req) for req in self.running],
            "waiting": waiting,
            "scheduled_tokens": dict(output.num_scheduled_tokens),
        }
        path = self._bubble_state_dir / f"rank-{self._bubble_rank}.json"
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(state, f, separators=(",", ":"))
        os.replace(tmp, path)
        return output

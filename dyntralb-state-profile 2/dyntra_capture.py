"""Optional instrumentation for the archived DyntraLB scheduler version.

One JSONL row per schedule call. No requests or prompt content are recorded.
"""
import json
import os
import time
import warnings
from pathlib import Path


def install(cls, block_count):
    original = cls.schedule
    if getattr(original, '_dataset_wrapped', False):
        return

    def schedule(self, *args, **kwargs):
        directory = os.environ.get('DYNTRA_DATA_DIR')
        if not directory:
            return original(self, *args, **kwargs)
        try:
            running = list(self.running)
            waiting = list(self.waiting) + list(self.skipped_waiting)
            candidates = list(getattr(self, '_lb_admission_candidates', []))
            all_requests = {r.request_id: r for r in running + waiting}
            blocks = {k: block_count(self, v) for k, v in all_requests.items()}
            running_ids = {r.request_id for r in running}
            waiting_ids = {r.request_id for r in waiting}
            candidate_ids = {r.request_id for r in candidates}
            status_counts = {}
            for req in waiting:
                status = getattr(req.status, 'name', str(req.status))
                status_counts[status] = status_counts.get(status, 0) + 1
            state = {
                'running_count': len(running), 'waiting_count': len(waiting),
                'candidate_count_snapshot': len(candidates),
                'candidate_snapshot_may_be_stale': True,
                'running_blocks': sum(blocks[r.request_id] for r in running),
                'waiting_blocks': sum(blocks[r.request_id] for r in waiting),
                'running_max_blocks': max((blocks[r.request_id] for r in running), default=0),
                'waiting_max_blocks': max((blocks[r.request_id] for r in waiting), default=0),
                'waiting_status_counts': status_counts,
                'max_running_requests': self.max_num_running_reqs,
                'plan_present': self.modifications is not None,
            }
            start_mono = time.monotonic_ns()
            start_wall = time.time_ns()
            plan = self.modifications
            planned = None if plan is None else {
                'freeze': bool(plan.get('freeze', False)),
                'in_count': len(plan.get('in_req_ids', plan.get('in_blk', []))),
                'out_count': len(plan.get('out_req_ids', plan.get('out_blk', []))),
                'count_source': 'request_ids' if 'in_req_ids' in plan else 'blocks',
            }
        except Exception as exc:
            warnings.warn('Dyntra capture pre-snapshot failed: ' + str(exc))
            return original(self, *args, **kwargs)
        output = original(self, *args, **kwargs)
        end_mono = time.monotonic_ns()
        try:
            scheduled = dict(output.num_scheduled_tokens)
            ids = set(scheduled)
            rank = int(self.vllm_config.parallel_config.data_parallel_rank)
            row = {
                'schema_version': 1, 'run_id': os.environ.get('DYNTRA_RUN_ID', 'unspecified'),
                'rank': rank, 'pid': os.getpid(), 'step': int(self.current_step),
                'wall_time_ns': start_wall, 'monotonic_start_ns': start_mono,
                'state_pre': state, 'plan_pre': planned,
                'action_observed': {
                    'scheduled_requests': len(ids),
                    'scheduled_tokens': sum(scheduled.values()),
                    'scheduled_from_running': len(ids & running_ids),
                    'scheduled_from_waiting': len(ids & waiting_ids),
                    'scheduled_from_candidate_snapshot': len(ids & candidate_ids),
                    'known_scheduled_blocks_pre': sum(blocks[k] for k in ids if k in blocks),
                    'scheduled_ids_missing_from_pre': len(ids - blocks.keys()),
                },
                'outcome': {'scheduler_cpu_wall_ms': (end_mono - start_mono) / 1e6,
                            'model_execution_ms': None, 'reward': None},
            }
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            with (path / f'rank-{rank}-pid-{os.getpid()}.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n')
        except Exception as exc:
            warnings.warn('Dyntra capture write failed: ' + str(exc))
        return output
    schedule._dataset_wrapped = True
    cls.schedule = schedule

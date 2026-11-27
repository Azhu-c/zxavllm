"""Engineering tests only; none of these numbers are research results."""
from __future__ import annotations
import asyncio
from collections import Counter
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import runtime
from experiment import transform_script
from client import benchmark
from analyze import attention_trace, performance, report
from dataset import specifications


class BaseIO:
    def __init__(self, config, *args):
        self.vllm_config = config
        self.client_index = 0
        self.core_engines = [b"e0", b"e1", b"e2"]
        self.engine_ranks_managed = [0, 1, 2]
        self.lb_engines = [[0, 0, 0] for _ in self.core_engines]
        self.resources = NS(engine_dead=False)
        self.current_wave = 1
        self.engines_running = True
        self.sent_requests = []

    def _ensure_stats_update_task(self):
        pass

    def _ensure_output_queue_task(self):
        pass

    def _send_input(self, kind, request, engine):
        async def send():
            self.sent_requests.append((request.request_id, engine))
        return send()

    async def _call_utility_async(self, method, engine):
        rank = self.core_engines.index(engine)
        return {"rank": rank, "received": 0, "block_size": 128,
                "running_blocks": 0, "waiting_blocks": 0}


def native_module():
    module = ModuleType("native_fixture")
    module.__dict__.update(BaseIO=BaseIO,
        get_late_interaction_engine_index=lambda *args: None,
        EngineCoreRequestType=NS(ADD="add"))
    code = (ROOT / "tests/native_client.txt").read_text()
    # inspect.getsource uses a file-backed module; the fixture suffix is intentional.
    filename = str(ROOT / "tests/native_client.txt")
    exec(compile(code, filename, "exec"), module.__dict__)
    return module


class RoutingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, MOTIVATION_DATA=self.tmp.name,
                              MOTIVATION_REPORT_MS="100", MOTIVATION_POLICY="kv")
        self.env.start()

    async def asyncTearDown(self):
        self.env.stop()
        # Runtime writer caches intentionally persist for the lifetime of a worker.
        for handle in runtime.HANDLES.values():
            handle.close()
        runtime.HANDLES.clear()
        self.tmp.cleanup()

    async def test_native_lifecycle_and_burst_routing(self):
        for policy, lengths, expected in (
            ("rr", [8192, 8192, 2048, 2048], [0, 1, 2, 0]),
            ("jsq", [8192, 8192, 2048, 2048], [0, 1, 2, 0]),
            ("kv", [8192, 8192, 2048, 2048], [0, 1, 2, 2]),
        ):
            with self.subTest(policy=policy):
                os.environ["MOTIVATION_POLICY"] = policy
                module = native_module()
                runtime.patch_client(module)
                config = NS(parallel_config=NS(data_parallel_external_lb=False,
                                                data_parallel_hybrid_lb=False))
                client = module.DPLBAsyncMPClient(config, None, False)
                try:
                    for i, length in enumerate(lengths):
                        request = NS(request_id=str(i), data_parallel_rank=None,
                                     pooling_params=None, prompt_token_ids=[0] * length)
                        await client.add_request_async(request)
                        self.assertIsNone(request.data_parallel_rank)
                    ranks = [client.core_engines.index(e) for _, e in client.sent_requests]
                    self.assertEqual(ranks, expected)
                    self.assertEqual(sum(client.engine_inflight.values()), 4)
                    await module.DPLBAsyncMPClient.process_engine_outputs(client, NS(finished_requests=["0", "2"]))
                    self.assertEqual(sum(client.engine_inflight.values()), 2)
                    self.assertEqual(set(client.reqs_in_flight), {"1", "3"})
                finally:
                    client.resources.engine_dead = True
                    client._mot_poll.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await client._mot_poll

    async def test_pending_ack_and_waiting_block_accounting(self):
        state = runtime.RouterState(2)
        reports = [{"rank": i, "received": 0, "block_size": 128,
                    "running_blocks": 0, "waiting_blocks": 0} for i in range(2)]
        state.update(reports)
        state.dispatch(0, 64)
        self.assertEqual(state.totals(), [64, 0])
        state.update([{**reports[0], "received": 1, "waiting_blocks": 64}, reports[1]])
        self.assertEqual(state.totals(), [64, 0])  # Acknowledgement removes the extra copy.
        state.update([{**reports[0], "received": 1}, reports[1]])
        self.assertEqual(state.totals(), [0, 0])
        with self.assertRaises(RuntimeError):
            state.update([{**reports[0], "received": 2}, reports[1]])

    async def test_core_snapshot_and_step_marker(self):
        running = NS(request_id="r", all_token_ids=list(range(129)))
        waiting = NS(request_id="w", all_token_ids=list(range(257)))
        output = NS(num_scheduled_tokens={"r": 1})
        class Core:
            def __init__(self):
                self.dp_rank, self.current_wave, self.step_counter = 0, 2, 4
                self.scheduler = NS(cache_config=NS(block_size=128), running=[running],
                                    waiting=[waiting], skipped_waiting=[waiting],
                                    kv_cache_manager=NS(block_pool=None), max_num_running_reqs=8,
                                    schedule=lambda: output)
            def add_request(self, request):
                return request
        module = NS(EngineCore=Core)
        with patch.dict(os.environ, MOTIVATION_PROFILE="1"):
            runtime.patch_core(module)
            core = Core()
            core.add_request(waiting)
            state = core.motivation_snapshot()
            self.assertEqual(state["running_blocks"], 2)
            self.assertEqual(state["waiting_blocks"], 3)
            self.assertEqual(state["received"], 1)
            self.assertIs(core.scheduler.schedule(), output)
            self.assertEqual(output.motivation_marker, {"rank": 0, "wave": 2, "step": 4})


class PipelineTests(unittest.TestCase):
    def test_script_copies_and_workload_contract(self):
        cfg = {"model_path": "/model with space", "host_ip": "10.1.1.1", "served_model": "m"}
        source = ('#!/bin/bash\nMODEL_PATH="old"\npgrep -af "vllm serve|EngineCore"\n'
                  'cleanup_stale_vllm_shm || exit 1\n'
                  'pkill -9 python || true\nLOG_DIR="./log"\n'
                  'exec vllm serve "$MODEL_PATH" --max-num-seqs 8\n')
        result = transform_script(source, "decoder", cfg)
        self.assertIn("--max-num-seqs 8", result)
        self.assertIn("serve_entry.py", result)
        self.assertNotIn("\npkill", result)
        self.assertIn("MOTIVATION_LOG_DIR", result)
        self.assertIn("'/model with space'", result)
        plan = specifications()
        self.assertEqual(len(plan), 96)
        self.assertEqual(Counter(r["class"] for r in plan), {"SS": 24, "SL": 24, "LS": 24, "LL": 24})
        self.assertEqual(plan, specifications())

    def test_stream_to_report_with_multi_token_chunks(self):
        row = {"id": "SS-00", "class": "SS", "prompt": "text", "input_tokens": 4, "output_tokens": 4}
        class Response(io.BytesIO):
            headers = {}
        def fake_http(url, body=None, timeout=0):
            self.assertTrue(body["ignore_eos"])
            self.assertNotIn("data_parallel_rank", body)
            messages = [
                {"choices": [{"text": "a b", "finish_reason": None}]},
                {"choices": [{"text": " c d", "finish_reason": "length"}]},
                {"choices": [], "usage": {"completion_tokens": 4, "prompt_tokens": 4}},
            ]
            return Response(("".join("data: " + json.dumps(m) + "\n\n" for m in messages)
                             + "data: [DONE]\n\n").encode())
        cfg = {"host_ip": "localhost", "proxy_port": 8080, "served_model": "m",
               "request_rate": 0, "max_concurrency": 1, "request_timeout_seconds": 2}
        with tempfile.TemporaryDirectory() as temp, patch("client.http", side_effect=fake_http):
            run = Path(temp) / "rr-r1"
            benchmark([row], cfg, run, None)
            summary, _ = performance(run)
            self.assertTrue(summary["valid"])
            self.assertEqual(summary["output_tokens"], 4)
            self.assertIsNotNone(summary["tpot_p50_ms"])
            with contextlib.redirect_stdout(io.StringIO()):
                summaries = report(Path(temp))
            self.assertEqual(len(summaries), 1)
            self.assertTrue((Path(temp) / "summary.csv").exists())

    def test_device_attention_correlation_excludes_cpu_aliases(self):
        marker = {"rank": 0, "wave": 1, "step": 3, "layer": "model.layers.0", "occurrence": 0}
        events = [
            {"ph": "X", "name": "MOT_ATTN|" + json.dumps(marker), "pid": 1, "tid": 1, "ts": 1, "dur": 20},
            {"ph": "X", "name": "aclnnFusedInferAttentionScoreV3", "cat": "cpu_op", "pid": 1,
             "tid": 1, "ts": 2, "dur": 2, "args": {"External id": 9}},
            {"ph": "X", "name": "FusedInferAttentionScore", "pid": 2, "tid": 1, "ts": 3,
             "dur": 5, "args": {"External id": 9}},
            {"ph": "X", "name": "FusedInferAttentionScore", "cat": "cpu_op", "pid": 1,
             "tid": 1, "ts": 2, "dur": 2},
        ]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "trace_view.json"
            path.write_text(json.dumps({"traceEvents": events}))
            rank, kernels, matched = attention_trace(path)
            self.assertEqual(rank, 0)
            self.assertEqual([k["duration_us"] for k in kernels], [5])
            self.assertEqual(matched[0]["step"], 3)

    def test_effective_server_options_keep_bl_and_profile_separate(self):
        import serve_entry
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / "prepared").mkdir()
            (directory / "prepared/environment.json").write_text(json.dumps({"vllm_executable": "/vllm"}))
            (directory / "config.json").write_text(json.dumps({"model_path": "/model", "host_ip": "10.0.0.1",
                  "served_model": "m", "decoder_port": 13701}))
            args = ["serve_entry.py", "decoder", "serve", "/old-model", "--max-num-seqs", "8",
                    "--additional-config", '{"scheduler_config":{"dyntra_lb_config":{"enabled":true}}}',
                    "--compilation-config", '{"cudagraph_mode":"FULL_DECODE_ONLY"}']
            with patch.object(serve_entry, "__file__", str(directory / "serve_entry.py")), \
                 patch.object(sys, "argv", args), patch("serve_entry.os.execv") as execute, \
                 patch.dict(os.environ, MOTIVATION_RUN=temp, MOTIVATION_PROFILE="1", MOTIVATION_PROFILE_DIR=temp):
                serve_entry.main()
                effective = execute.call_args.args[1]
                self.assertIn("--enforce-eager", effective)
                self.assertEqual(effective[effective.index("--max-num-seqs") + 1], "8")
                additional = json.loads(effective[effective.index("--additional-config") + 1])
                self.assertFalse(additional["scheduler_config"]["dyntra_lb_config"]["enabled"])
                compilation = json.loads(effective[effective.index("--compilation-config") + 1])
                self.assertEqual(compilation["cudagraph_mode"], "NONE")
                self.assertEqual(os.environ["VLLM_DP_KV_BLOCKS_ROUTING"], "0")


if __name__ == "__main__":
    unittest.main()

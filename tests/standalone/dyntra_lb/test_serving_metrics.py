# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[3] / "benchmarks/dyntra_lb/compare_serving_results.py"
SPEC = importlib.util.spec_from_file_location("dyntralb_serving_metrics", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_tpot_includes_long_pause_and_single_token_is_excluded():
    result = MODULE.summarize(
        {
            "ttfts": [0.1, 0.2, 0],
            "input_lens": [40_000, 10, 0],
            "output_lens": [3, 1, 0],
            "itls": [[0.1, 0.9], [], []],
            "output_throughput": 99,
            "completed": 2,
        },
        32_768,
    )
    assert result["long"]["tpot"]["mean_ms"] == 500
    assert result["short"]["tpot"]["count"] == 0
    assert result["failed_or_missing_output"] == 1
    assert result["output_throughput"] == 99


def test_rejects_multi_token_nonstreaming_metrics():
    with pytest.raises(ValueError, match="no ITLs"):
        MODULE.summarize({"ttfts": [0.1], "input_lens": [10], "output_lens": [2], "itls": [[]]}, 100)


def test_rejects_mismatched_per_request_arrays():
    with pytest.raises(ValueError, match="different lengths"):
        MODULE.summarize({"ttfts": [0.1], "input_lens": [], "output_lens": [2], "itls": [[0.1]]}, 100)

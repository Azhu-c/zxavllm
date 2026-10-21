# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Live same-instance CP smoke tests. Deployment stays in the operator's script.

The long prompt must exceed dynamic_cp_config.min_context_tokens. Check server
DYNTRA_CP diagnostics to confirm activation; an HTTP success alone cannot prove
that a helper was idle. The optional reference service should use identical
weights, MTP, tokenizer and sampling settings with dynamic CP disabled.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest


@pytest.fixture
def service(request):
    config = {
        name: request.config.getoption(f"--dyntra-cp-{name}") for name in ("url", "model", "prompt", "reference-url")
    }
    if not all(config[name] for name in ("url", "model", "prompt")):
        pytest.skip("Supply --dyntra-cp-url, --dyntra-cp-model and --dyntra-cp-prompt for live testing")
    config["long_prompt"] = Path(config["prompt"]).read_text()
    return config


def completion(url, model, prompt, limit=64):
    text = []
    body = dict(model=model, prompt=prompt, max_tokens=limit, temperature=0, seed=12292, stream=True)
    with (
        httpx.Client(timeout=600) as client,
        client.stream("POST", url.rstrip("/") + "/v1/completions", json=body) as result,
    ):
        assert result.status_code == 200, result.read().decode()
        done = False
        for line in result.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[len("data: ") :]
            if payload == "[DONE]":
                done = True
                break
            chunk = json.loads(payload)
            assert "error" not in chunk
            text.extend(choice.get("text", "") for choice in chunk.get("choices", []))
        assert done
    return "".join(text)


def test_long_request_and_optional_greedy_reference(service):
    result = completion(service["url"], service["model"], service["long_prompt"])
    assert result
    if service["reference-url"]:
        reference = completion(service["reference-url"], service["model"], service["long_prompt"])
        assert result == reference


def test_long_and_new_short_requests_complete_without_deadlock(service):
    with ThreadPoolExecutor(max_workers=8) as executor:
        long_request = executor.submit(completion, service["url"], service["model"], service["long_prompt"])
        short_requests = [
            executor.submit(completion, service["url"], service["model"], "Reply briefly: 1 + 1 =", 16)
            for _ in range(7)
        ]
        assert long_request.result(timeout=600)
        assert all(future.result(timeout=600) for future in short_requests)

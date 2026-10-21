# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


def pytest_addoption(parser):
    group = parser.getgroup("dyntra-cp")
    group.addoption("--dyntra-cp-url", default=None)
    group.addoption("--dyntra-cp-reference-url", default=None)
    group.addoption("--dyntra-cp-model", default=None)
    group.addoption("--dyntra-cp-prompt", default=None)

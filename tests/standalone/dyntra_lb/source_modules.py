# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Load independent production modules without plugin package initialization."""

import ast
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def load_source(name):
    module_name = f"_cpu_test_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = ROOT / f"vllm_ascend/core/{name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    # Inject only independent production dependencies, bypassing plugin package
    # initialization. Every class/function body is executed unchanged.
    tree = ast.parse(path.read_text())
    kept = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("vllm_ascend.core."):
            dependency = load_source(node.module.rsplit(".", 1)[-1])
            for alias in node.names:
                module.__dict__[alias.asname or alias.name] = getattr(dependency, alias.name)
        else:
            kept.append(node)
    tree.body = kept
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


DyntraLBConfig = load_source("dyntra_lb_config").DyntraLBConfig
balance_load = load_source("dyntra_lb_planner").balance_load
DyntraLBPausePolicy = load_source("dyntra_lb_policy").DyntraLBPausePolicy

balance_load_original = load_source("dyntra_lb_original").balance_load_original
DyntraLBFirstTokenPolicy = load_source("dyntra_lb_policy").DyntraLBFirstTokenPolicy

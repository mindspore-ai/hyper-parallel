# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""SAPP-PPB: Symbolic Automatic Parallel Planner - Pipeline Parallelism Balancing.

Offline ILP planner for automatically generating layer distribution and
recomputation policies across pipeline parallel stages.
"""
# pylint: disable=invalid-name,undefined-all-variable

from importlib import import_module as _import_module
import sys as _sys

_sys.modules.setdefault("sapp_ppb", _sys.modules[__name__])

__all__ = [
    "SappPipeline",
    "choose_interleave",
    "flatten",
    "SappSolver",
    "PipelineMemoryConstraint",
    "Layer",
    "generate_layers_list",
    "compute_memories",
    "initialize_layer_json",
    "build_arg_parser",
    "run",
    "main",
]

_EXPORTS = {
    "SappPipeline": ".sapp.sapp_pipeline",
    "choose_interleave": ".sapp.sapp_pipeline",
    "flatten": ".sapp.sapp_pipeline",
    "SappSolver": ".sapp.sapp_solver",
    "PipelineMemoryConstraint": ".sapp.sapp_solver",
    "Layer": ".utils.layer",
    "generate_layers_list": ".utils.layer",
    "compute_memories": ".utils.compute_memory",
    "initialize_layer_json": ".utils.config",
    "build_arg_parser": ".run_pipeline_balance",
    "run": ".run_pipeline_balance",
    "main": ".run_pipeline_balance",
}


def __getattr__(name):
    """Lazily import public SAPP-PPB interfaces."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = _import_module(_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value

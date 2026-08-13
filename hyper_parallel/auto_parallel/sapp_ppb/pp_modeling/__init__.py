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
"""Pipeline Parallelism modeling and load balancing for auto-parallel strategy search."""

from importlib import import_module as _import_module

from hyper_parallel.auto_parallel.sapp_ppb.pp_modeling.pp_structs import (
    PPStrategyResult,
    PPBOutput,
    RecomputeType,
)
from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.yaml_parser import (
    YamlOptimizationConfig,
    parse_yaml_for_optimization,
)

_LAZY_EXPORTS = {
    "PPOptimizer": "hyper_parallel.auto_parallel.sapp_ppb.pp_optimizer",
}


def __getattr__(name):  # pylint: disable=invalid-name
    """Lazily import PPOptimizer to avoid circular imports."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = _import_module(_LAZY_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = [
    "PPStrategyResult",
    "PPBOutput",
    "RecomputeType",
    "YamlOptimizationConfig",
    "parse_yaml_for_optimization",
    "PPOptimizer",  # pylint: disable=E0603
]

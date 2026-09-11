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
"""Public high-performance module interfaces.

Exports are resolved lazily so modules backed only by torch-npu do not import
unrelated optional Omni custom operators. Accessing an Omni-backed module still
loads and validates that dependency normally.
"""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hyper_parallel.components.modules.dsa_attention import DSAAttention, DeepseekV32DSAAttention
    from hyper_parallel.components.modules.gqa_attention import GatedGQAAttention, GQAAttention
    from hyper_parallel.components.modules.grouped_experts import GroupedExperts
    from hyper_parallel.components.modules.mhc import MhcPostModule, MhcPreModule
    from hyper_parallel.components.modules.mla_attention import MLAAttention
    from hyper_parallel.components.modules.rms_norm import OffsetRMSNorm, RMSNorm
    from hyper_parallel.components.modules.shared_expert import SharedExpert
    from hyper_parallel.components.modules.swiglu_mlp import SwiGLUMLP

_EXPORT_TO_MODULE = {
    "DeepseekV32DSAAttention": "dsa_attention",
    "DSAAttention": "dsa_attention",
    "GQAAttention": "gqa_attention",
    "GatedGQAAttention": "gqa_attention",
    "GroupedExperts": "grouped_experts",
    "MhcPostModule": "mhc",
    "MhcPreModule": "mhc",
    "MLAAttention": "mla_attention",
    "OffsetRMSNorm": "rms_norm",
    "RMSNorm": "rms_norm",
    "SharedExpert": "shared_expert",
    "SwiGLUMLP": "swiglu_mlp",
}

__all__ = list(_EXPORT_TO_MODULE)


def __getattr__(name: str) -> Any:  # pylint: disable=invalid-name
    """Resolve a public class by importing only its owning submodule."""
    submodule = _EXPORT_TO_MODULE.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f"hyper_parallel.components.modules.{submodule}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy public exports in ``dir()``."""
    return sorted(set(globals()) | set(__all__))

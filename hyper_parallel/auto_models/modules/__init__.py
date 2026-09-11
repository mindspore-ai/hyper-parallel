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
"""Public high-performance module interfaces."""

from importlib import import_module as _import_module  # pylint: disable=invalid-name


# Several high-performance modules depend on optional custom operators. Resolve
# only the requested public module so unrelated optional dependencies do not
# become an import-time requirement.
_LAZY_EXPORTS = {
    "DeepseekV32DSAAttention": ".dsa_attention",
    "DSAAttention": ".dsa_attention",
    "GQAAttention": ".gqa_attention",
    "GatedGQAAttention": ".gqa_attention",
    "GroupedExperts": ".grouped_experts",
    "MhcPostModule": ".mhc",
    "MhcPostProcessModule": ".mhc",
    "MhcPreModule": ".mhc",
    "MLAAttention": ".mla_attention",
    "OffsetRMSNorm": ".rms_norm",
    "RMSNorm": ".rms_norm",
    "SharedExpert": ".shared_expert",
    "SwiGLUMLP": ".swiglu_mlp",
}


def __getattr__(name):  # pylint: disable=invalid-name
    """Lazily import a public high-performance module."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = _import_module(_LAZY_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():  # pylint: disable=invalid-name
    """Include lazy public exports in ``dir()``."""
    return sorted(set(globals()) | set(_LAZY_EXPORTS))

__all__ = [
    "DeepseekV32DSAAttention",
    "DSAAttention",
    "GQAAttention",
    "GatedGQAAttention",
    "GroupedExperts",
    "MhcPostModule",
    "MhcPostProcessModule",
    "MhcPreModule",
    "MLAAttention",
    "OffsetRMSNorm",
    "RMSNorm",
    "SharedExpert",
    "SwiGLUMLP",
]

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
from typing import Any


_EXPORT_TO_MODULE = {
    "DeepseekV32DSAAttention": "dsa_attention",
    "DSAAttention": "dsa_attention",
    "GQAAttention": "gqa_attention",
    "GatedGQAAttention": "gqa_attention",
    "EngramModule": "engram",
    "GroupedExperts": "grouped_experts",
    "KimiDeltaAttention": "kimi_delta_attention",
    "KimiRMSNormGated": "kimi_delta_attention",
    "MhcPostModule": "mhc",
    "MhcPreModule": "mhc",
    "NgramHashMapping": "engram",
    "PipelinedMhcModule": "mhc",
    "MLAAttention": "mla_attention",
    "OffsetRMSNorm": "rms_norm",
    "RMSNorm": "rms_norm",
    "SharedExpert": "shared_expert",
    "SharedCompressedAttentionCPContext": "shared_compressed_dsa_attention",
    "SharedCompressedAttentionState": "shared_compressed_dsa_attention",
    "SharedCompressedPackedSequence": "shared_compressed_dsa_attention",
    "SharedCompressedAttentionTPContext": "shared_compressed_dsa_attention",
    "SharedCompressedDSAAttention": "shared_compressed_dsa_attention",
    "SharedCompressedDSAIndexer": "shared_compressed_dsa_attention",
    "SwiGLUMLP": "swiglu_mlp",
}


def __getattr__(name: str) -> Any:  # pylint: disable=invalid-name
    """Resolve a public class by importing only its owning submodule."""
    submodule = _EXPORT_TO_MODULE.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f"hyper_parallel.components.modules.{submodule}")
    value = getattr(module, name)
    globals()[name] = value
    return value

__all__ = list(_EXPORT_TO_MODULE)

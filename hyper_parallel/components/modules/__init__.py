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

__all__: list[str] = []

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hyper_parallel.components.modules.attention.multi_latent_attention import (
        MultiLatentAttention, MultiLatentAttentionModule,
    )
    from hyper_parallel.components.modules.attention.gated_multi_latent_attention import (
        GatedMultiLatentAttention, GatedMultiLatentAttentionModule,
    )
    from hyper_parallel.components.modules.mtp import (
        MTPLayer, MultiTokenPrediction, MTPOutput, build_mtp,
    )
    from hyper_parallel.components.modules.dsa_attention import DSAAttention, DeepseekV32DSAAttention
    from hyper_parallel.components.modules.attention.grouped_query_attention import (
        GroupedQueryAttention, SplitGroupedQueryAttention,
    )
    from hyper_parallel.components.modules.attention.gated_grouped_query_attention import (
        GatedGroupedQueryAttention,
    )
    from hyper_parallel.components.modules.grouped_experts import GroupedExperts
    from hyper_parallel.components.modules.linear import Linear
    from hyper_parallel.components.modules.mhc import MhcPostModule, MhcPreModule
    from hyper_parallel.components.modules.rms_norm import OffsetRMSNorm, RMSNorm
    from hyper_parallel.components.modules.shared_expert import SharedExpert
    from hyper_parallel.components.modules.swiglu_mlp import SwiGLUMLP

_EXPORT_TO_MODULE = {
    "TritonGDN": "gdn_triton",
    "build_mtp": "mtp",
    "MTPLayer": "mtp",
    "MultiTokenPrediction": "mtp",
    "MTPOutput": "mtp",
    "DeepseekV32DSAAttention": "dsa_attention",
    "DSAAttention": "dsa_attention",
    "GroupedQueryAttention": "attention.grouped_query_attention.grouped_query_attention_family",
    "SplitGroupedQueryAttention": "attention.grouped_query_attention.grouped_query_attention_family",
    "GatedGroupedQueryAttention": "attention.gated_grouped_query_attention.gated_grouped_query_attention_family",
    "EngramModule": "engram",
    "GroupedExperts": "grouped_experts",
    "Linear": "linear",
    "KimiDeltaAttention": "kimi_delta_attention",
    "KimiRMSNormGated": "kimi_delta_attention",
    "MhcPostModule": "mhc",
    "MhcPreModule": "mhc",
    "NgramHashMapping": "engram",
    "PipelinedMhcModule": "mhc",
    "MultiLatentAttention": "attention.multi_latent_attention",
    "MultiLatentAttentionModule": "attention.multi_latent_attention",
    "GatedMultiLatentAttention": "attention.gated_multi_latent_attention",
    "GatedMultiLatentAttentionModule": "attention.gated_multi_latent_attention",
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
_EXPORT_TO_MODULE.update({name: "module_family" for name in (
    "ModuleFamily", "SelectorSpec", "VariantRule", "FamilyRequest", "ResolvedFamily",
    "build_family", "register_module_family", "get_family_build_info", "collect_family_modules",
    "FamilyShardingContext",
)})
__all__.extend(_EXPORT_TO_MODULE)


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

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
    from hyper_parallel.components.modules.common.mtp import (
        MTPLayer, MultiTokenPrediction, MTPOutput, build_mtp,
    )
    from hyper_parallel.components.modules.attention.deepseek_sparse_attention import (
        CPDeepseekSparseAttentionModule, DeepseekSparseAttention, DeepseekSparseAttentionModule,
    )
    from hyper_parallel.components.modules.attention.grouped_query_attention import (
        GroupedQueryAttention,
    )
    from hyper_parallel.components.modules.attention.gated_grouped_query_attention import (
        GatedGroupedQueryAttention,
    )
    from hyper_parallel.components.modules.moe.grouped_experts.grouped_experts_family import GroupedExperts
    from hyper_parallel.components.modules.linear import Linear
    from hyper_parallel.components.modules.mhc import MhcPostModule, MhcPreModule
    from hyper_parallel.components.modules.common.rms_norm import OffsetRMSNorm, RMSNorm
    from hyper_parallel.components.modules.moe.shared_expert import SharedExpert
    from hyper_parallel.components.modules.swiglu_mlp_family import SwiGLUMLP

_EXPORT_TO_MODULE = {
    "CompressedSparseAttention2": "attention.compressed_sparse_attention2",
    "Gate": "moe.router",
    "GateModule": "moe.router",
    "MegaMhc": "mhc.mhc_family",
    "MegaMhcModule": "mhc.mega_mhc",
    "SpatialMergeProjector": "common.spatial_merge_projector",
    "SpatialMergeProjectorModule": "common.spatial_merge_projector",
    "RotaryEmbedding": "common.rotary_embedding",
    "RotaryEmbeddingModule": "common.rotary_embedding",
    "HostEngramModule": "engram.host_engram",
    "HostEngramTable": "engram.host_engram_table",
    "Engram": "engram",
    "MHCPre": "mhc",
    "MHCPost": "mhc",
    "PipelinedMhc": "mhc",
    "SwiGLUMLPModule": "swiglu_mlp_family",
    "SwiGLUMLP": "swiglu_mlp_family",
    "SharedExpertModule": "moe.shared_expert",
    "SharedExpert": "moe.shared_expert",
    "RMSNormModule": "common.rms_norm",
    "NativeRMSNormModule": "common.rms_norm",
    "NativeOffsetRMSNormModule": "common.rms_norm",
    "RMSNorm": "common.rms_norm",
    "OffsetRMSNormModule": "common.rms_norm",
    "OffsetRMSNorm": "common.rms_norm",
    "build_mtp": "common.mtp",
    "MTPLayer": "common.mtp",
    "MultiTokenPrediction": "common.mtp",
    "MTPOutput": "common.mtp",
    "DeepseekSparseAttention": "attention.deepseek_sparse_attention",
    "DeepseekSparseAttentionModule": "attention.deepseek_sparse_attention",
    "CPDeepseekSparseAttentionModule": "attention.deepseek_sparse_attention",
    "GroupedQueryAttention": "attention.grouped_query_attention.grouped_query_attention_family",
    "GatedGroupedQueryAttention": "attention.gated_grouped_query_attention.gated_grouped_query_attention_family",
    "EngramModule": "engram",
    "GroupedExperts": "moe.grouped_experts.grouped_experts_family",
    "Linear": "linear",
    "KimiDeltaAttention": "attention.kimi_delta_attention",
    "KimiDeltaAttentionModule": "attention.kimi_delta_attention",
    "GatedDeltaNet": "attention.gated_delta_net",
    "GatedDeltaNetModule": "attention.gated_delta_net",
    "KimiRMSNormGated": "attention.kimi_delta_attention.kimi_delta_attention",
    "MhcPostModule": "mhc",
    "MhcPreModule": "mhc",
    "NgramHashMapping": "engram",
    "PipelinedMhcModule": "mhc",
    "MultiLatentAttention": "attention.multi_latent_attention",
    "MultiLatentAttentionModule": "attention.multi_latent_attention",
    "GatedMultiLatentAttention": "attention.gated_multi_latent_attention",
    "GatedMultiLatentAttentionModule": "attention.gated_multi_latent_attention",
    "SharedCompressedAttentionState": "attention.compressed_sparse_attention2.compressed_sparse_attention2",
    "SharedCompressedPackedSequence": "attention.compressed_sparse_attention2.document_layout",
    "CSA2Indexer": "attention.compressed_sparse_attention2.indexer",
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

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
"""Default backend orders for the built-in operator catalog."""

from .families.attention import GQA_ATTENTION, MLA_ATTENTION, MLA_WITH_STATS, VISION_ATTENTION
from .families.causal_conv1d import CAUSAL_CONV1D
from .families.dsa_indexer import DSA_INDEXER, DSA_PACKED_INDEXER
from .families.dsa_indexer_kl import DSA_DENSE_KL, DSA_INDEXER_KL, DSA_PACKED_KL
from .families.dsa_sparse_attention import DSA_ABSORBED, DSA_ABSORBED_SINK, DSA_COMPRESSED_BANKS
from .families.gdn import GDN
from .families.grouped_matmul import GROUPED_MATMUL
from .families.grouped_swiglu import GROUPED_SWIGLU
from .families.kda import KDA
from .families.mhc_post import MHC_POST
from .families.moe_router import MOE_ROUTER
from .families.moe_token_reorder import MOE_TOKEN_REORDER, MOE_TOKEN_RESTORE
from .families.mome import MOME
from .families.rms_norm import RMS_NORM, RMS_NORM_FP32_AFFINE
from .families.rope import ROPE, ROPE_INTERLEAVED
from .families.sinkhorn import SINKHORN
from .families.swiglu import SWIGLU, SWIGLU_CLAMPED
from .registry import OpRegistry


def apply_auto_policy(registry: OpRegistry) -> None:
    """Set device-specific priorities independently of implementation registration."""
    for spec in (RMS_NORM, RMS_NORM_FP32_AFFINE, SWIGLU, SWIGLU_CLAMPED):
        registry.set_auto_order(spec, device="npu", candidates=("torch_npu",))
        registry.set_auto_order(spec, device="cuda", candidates=("native",))
        registry.set_auto_order(spec, device="cpu", candidates=("native",))
    registry.set_auto_order(MHC_POST, device="npu", candidates=("cann", "omni", "native"))
    registry.set_auto_order(MHC_POST, device="cuda", candidates=("native",))
    registry.set_auto_order(MHC_POST, device="cpu", candidates=("native",))
    registry.set_auto_order(GROUPED_MATMUL, device="npu", candidates=("torch_npu",))
    registry.set_auto_order(GROUPED_MATMUL, device="cuda", candidates=("torch_grouped_mm", "native"))
    registry.set_auto_order(GROUPED_MATMUL, device="cpu", candidates=("native",))
    registry.set_auto_order(GROUPED_SWIGLU, device="npu", candidates=("torch_npu", "native"))
    registry.set_auto_order(GROUPED_SWIGLU, device="cuda", candidates=("torch_grouped_mm", "native"))
    registry.set_auto_order(GROUPED_SWIGLU, device="cpu", candidates=("native",))
    for spec in (CAUSAL_CONV1D, GDN, KDA):
        registry.set_auto_order(spec, device="npu", candidates=("triton_ascend", "ascendc", "native"))
        registry.set_auto_order(spec, device="cuda", candidates=("native",))
        registry.set_auto_order(spec, device="cpu", candidates=("native",))
    for spec in (VISION_ATTENTION, GQA_ATTENTION, MLA_ATTENTION, MLA_WITH_STATS):
        registry.set_auto_order(spec, device="npu", candidates=("torch_npu", "native"))
        for device in ("cuda", "cpu"):
            registry.set_auto_order(spec, device=device, candidates=("sdpa", "native"))
    for spec, fused in (
        (MOME, "omni"),
        (ROPE, "torch_npu"), (ROPE_INTERLEAVED, "torch_npu"),
        (SINKHORN, "omni"), (MOE_ROUTER, "megagate"),
        (MOE_TOKEN_REORDER, "torch_npu"), (MOE_TOKEN_RESTORE, "torch_npu"),
        (DSA_COMPRESSED_BANKS, "cann"), (DSA_INDEXER, "cann"), (DSA_INDEXER_KL, "cann"),
        (DSA_PACKED_INDEXER, "cann"), (DSA_ABSORBED, "cann"), (DSA_ABSORBED_SINK, "omni"),
        (DSA_PACKED_KL, "cann"), (DSA_DENSE_KL, "cann"),
    ):
        registry.set_auto_order(spec, device="npu", candidates=(fused, "native"))
        registry.set_auto_order(spec, device="cuda", candidates=("native",))
        registry.set_auto_order(spec, device="cpu", candidates=("native",))

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

from .families.attention import VISION_ATTENTION
from .families.dsa_indexer import DSA_INDEXER
from .families.dsa_indexer_kl import DSA_INDEXER_KL
from .families.dsa_sparse_attention import DSA_COMPRESSED_BANKS
from .families.grouped_matmul import GROUPED_MATMUL
from .families.mhc_post import MHC_POST
from .families.moe_router import MOE_ROUTER
from .families.moe_token_reorder import MOE_TOKEN_REORDER, MOE_TOKEN_RESTORE
from .families.rms_norm import RMS_NORM, RMS_NORM_FP32_AFFINE
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
    registry.set_auto_order(VISION_ATTENTION, device="npu", candidates=("torch_npu", "native"))
    for device in ("cuda", "cpu"):
        registry.set_auto_order(VISION_ATTENTION, device=device, candidates=("sdpa", "native"))
    for spec, fused in (
        (SINKHORN, "omni"), (MOE_ROUTER, "megagate"),
        (MOE_TOKEN_REORDER, "torch_npu"), (MOE_TOKEN_RESTORE, "torch_npu"),
        (DSA_COMPRESSED_BANKS, "cann"), (DSA_INDEXER, "cann"), (DSA_INDEXER_KL, "cann"),
    ):
        registry.set_auto_order(spec, device="npu", candidates=(fused, "native"))
        registry.set_auto_order(spec, device="cuda", candidates=("native",))
        registry.set_auto_order(spec, device="cpu", candidates=("native",))

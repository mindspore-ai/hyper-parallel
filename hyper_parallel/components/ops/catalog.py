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
"""Assemble the shared catalog of built-in operator backends."""

from .auto_policy import apply_auto_policy
from .impl.attention import register_attention
from .impl.causal_conv1d import register_causal_conv1d
from .impl.dsa_indexer import register_dsa_indexer
from .impl.dsa_indexer_kl import register_dsa_indexer_kl
from .impl.deepseek_sparse_attention import register_deepseek_sparse_attention
from .impl.gdn import register_gdn
from .impl.gemm import register_gemm
from .impl.engram import register_engram
from .impl.grouped_matmul import register_grouped_matmul
from .impl.grouped_swiglu import register_grouped_swiglu
from .impl.kda import register_kda
from .impl.mhc_post import register_mhc_post
from .impl.mhc_pre import register_mhc_pre
from .impl.moe_router import register_moe_router
from .impl.moe_token_reorder import register_moe_token_reorder
from .impl.mome import register_mome
from .impl.rms_norm import register_rms_norm
from .impl.rope import register_rope
from .impl.quantization import register_quantization
from .impl.sinkhorn import register_sinkhorn
from .impl.swiglu import register_swiglu
from .registry import OpRegistry

builtin_registry = OpRegistry()
register_rms_norm(builtin_registry)
register_rope(builtin_registry)
register_swiglu(builtin_registry)
register_mhc_post(builtin_registry)
register_mhc_pre(builtin_registry)
register_engram(builtin_registry)
register_quantization(builtin_registry)
register_grouped_matmul(builtin_registry)
register_grouped_swiglu(builtin_registry)
register_attention(builtin_registry)
register_causal_conv1d(builtin_registry)
register_gdn(builtin_registry)
register_gemm(builtin_registry)
register_kda(builtin_registry)
register_moe_router(builtin_registry)
register_moe_token_reorder(builtin_registry)
register_mome(builtin_registry)
register_sinkhorn(builtin_registry)
register_dsa_indexer(builtin_registry)
register_deepseek_sparse_attention(builtin_registry)
register_dsa_indexer_kl(builtin_registry)
apply_auto_policy(builtin_registry)
builtin_registry.seal()

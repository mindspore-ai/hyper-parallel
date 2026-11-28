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
"""Public operator functions with process-wide backend selection."""

from . import catalog as _catalog
from .factory import ops
from ._dsa_layout import DsaQueryLayout
from .impl.aggregate_hidden import aggregate_hidden
from .impl.attention import (
    vision_attention,
    grouped_query_attention,
    multi_latent_attention,
    multi_latent_attention_with_stats,
)
from .impl.causal_conv1d import causal_conv1d
from .impl.deepseek_sparse_attention import (
    deepseek_sparse_attention_compressed_banks,
    deepseek_sparse_attention_absorbed,
    deepseek_sparse_attention_with_sink,
)
from .impl.dsa_indexer import dsa_indexer, dsa_indexer_packed
from .impl.dsa_indexer_kl import dsa_indexer_kl, dsa_indexer_kl_packed, dsa_indexer_kl_dense
from .impl.gdn import gdn
from .impl.grouped_matmul import grouped_matmul
from .impl.grouped_swiglu import grouped_swiglu
from .impl.kda import kda
from .impl.mhc_post import mhc_post
from .impl.moe_router import moe_router, router_scores
from .impl.moe_token_permute import moe_token_permute
from .impl.moe_token_unpermute import moe_token_unpermute
from .impl.rms_norm import rms_norm, rms_norm_fp32_affine
from .impl.rope import rope, rope_interleaved, rope_interleaved_single
from .impl.sinkhorn import sinkhorn
from .impl.swiglu import swiglu, swiglu_clamped

__all__ = [
    "DsaQueryLayout",
    "ops",
    "router_scores",
    "aggregate_hidden",
    "vision_attention",
    "grouped_query_attention",
    "multi_latent_attention",
    "multi_latent_attention_with_stats",
    "causal_conv1d",
    "deepseek_sparse_attention_compressed_banks",
    "deepseek_sparse_attention_absorbed",
    "deepseek_sparse_attention_with_sink",
    "dsa_indexer",
    "dsa_indexer_packed",
    "dsa_indexer_kl",
    "dsa_indexer_kl_packed",
    "dsa_indexer_kl_dense",
    "gdn",
    "grouped_matmul",
    "grouped_swiglu",
    "kda",
    "mhc_post",
    "moe_router",
    "moe_token_permute",
    "moe_token_unpermute",
    "rms_norm",
    "rms_norm_fp32_affine",
    "rope",
    "rope_interleaved",
    "rope_interleaved_single",
    "sinkhorn",
    "swiglu",
    "swiglu_clamped",
]

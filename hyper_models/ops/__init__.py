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
"""Public high-performance function interfaces."""

from hyper_models.ops.aggregate_hidden import aggregate_hidden
from hyper_models.ops.attention_rescale import attention_rescale
from hyper_models.ops.aux_loss import aux_loss_auto_scale, set_aux_loss_scale
from hyper_models.ops.dsa_indexer import dsa_indexer
from hyper_models.ops.dsa_kl_loss import dsa_kl_loss
from hyper_models.ops.dsa_sparse_attention import dsa_sparse_attention
from hyper_models.ops.dsa_sparse_attention_rescale import dsa_sparse_attention_rescale
from hyper_models.ops.grouped_matmul import grouped_matmul
from hyper_models.ops.mhc_post import mhc_post
from hyper_models.ops.mhc_pre import mhc_pre
from hyper_models.ops.moe_token_permute import moe_token_permute
from hyper_models.ops.moe_token_unpermute import moe_token_unpermute
from hyper_models.ops.npu_fusion_attention import npu_fusion_attention_forward
from hyper_models.ops.rms_norm import rms_norm
from hyper_models.ops.rotary_embedding import (
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_interleave,
)
from hyper_models.ops.sink_attention import sink_attention
from hyper_models.ops.sinkhorn import sinkhorn
from hyper_models.ops.swiglu import swiglu

__all__ = [
    "aggregate_hidden",
    "apply_rotary_pos_emb",
    "apply_rotary_pos_emb_interleave",
    "attention_rescale",
    "aux_loss_auto_scale",
    "dsa_indexer",
    "dsa_kl_loss",
    "dsa_sparse_attention",
    "dsa_sparse_attention_rescale",
    "grouped_matmul",
    "mhc_post",
    "mhc_pre",
    "moe_token_permute",
    "moe_token_unpermute",
    "npu_fusion_attention_forward",
    "rms_norm",
    "set_aux_loss_scale",
    "sink_attention",
    "sinkhorn",
    "swiglu",
]

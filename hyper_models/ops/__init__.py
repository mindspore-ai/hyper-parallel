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
from hyper_models.ops.dsa_indexer import dsa_indexer
from hyper_models.ops.dsa_kl_loss import dsa_kl_loss
from hyper_models.ops.npu_fusion_attention import npu_fusion_attention_forward
from hyper_models.ops.rms_norm import rms_norm
from hyper_models.ops.rotary_embedding import (
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_interleave,
)
from hyper_models.ops.sink_attention import sink_attention
from hyper_models.ops.dsa_sparse_attention import dsa_sparse_attention
from hyper_models.ops.dsa_sparse_attention_rescale import dsa_sparse_attention_rescale

__all__ = [
    "aggregate_hidden",
    "apply_rotary_pos_emb",
    "apply_rotary_pos_emb_interleave",
    "attention_rescale",
    "dsa_indexer",
    "dsa_kl_loss",
    "dsa_sparse_attention",
    "dsa_sparse_attention_rescale",
    "npu_fusion_attention_forward",
    "rms_norm",
    "sink_attention",
]

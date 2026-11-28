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
"""Rollout Q/K preparation for HP grouped QKV with compatible affine RMSNorms."""

from types import MethodType

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRMSNorm

from hyper_parallel.components.checkpoint.load_groups import local_target_tensor
from hyper_parallel.components.modules.attention.grouped_query_attention.grouped_query_attention import (
    GroupedQueryAttentionModule,
)


def bind_qk_norm_rope(attention: GroupedQueryAttentionModule) -> None:
    """Fuse the ordinary grouped projection after HP has sharded its parameters."""
    norm_forwards = (Qwen3RMSNorm.forward, Qwen3MoeRMSNorm.forward)
    # Subclasses and hooked norms must retain their own projection semantics.
    # pylint: disable=unidiomatic-typecheck,protected-access
    if (type(attention) is GroupedQueryAttentionModule and attention.fuse_qkv
            and attention.qk_head_dim == attention.v_head_dim == 128
            and not attention.rotary_interleaved
            and all(getattr(type(norm), "forward", None) in norm_forwards and "forward" not in vars(norm)
                    and type(norm).__call__ is torch.nn.Module.__call__
                    and not norm._forward_hooks and not norm._forward_pre_hooks
                    for norm in (attention.q_norm, attention.k_norm))
            and attention.q_norm.variance_epsilon == attention.k_norm.variance_epsilon):
        attention.project_qkv = MethodType(project_qkv_with_fused_rope, attention)


def project_qkv_with_fused_rope(self, hidden_states, positions):
    """Return HP's local BSND query, key, value and optional output state."""
    if (hidden_states.device.type == "cpu" or torch.is_grad_enabled() or (self.training and self.attention_dropout)
            or hidden_states.ndim != 3 or hidden_states.shape[0] != 1
            or hidden_states.dtype != torch.bfloat16
            or self.q_norm.weight.dtype != torch.bfloat16 or self.k_norm.weight.dtype != torch.bfloat16
            or positions is None
            or any(x.dtype != torch.bfloat16 or x.shape != (*hidden_states.shape[:2], 128)
                   for x in positions)):
        return GroupedQueryAttentionModule.project_qkv(self, hidden_states, positions)

    # Triton is needed only when an accelerator executes the fused layout.
    from rl.unified_execution.vllm.attention.qk_norm_rope import qk_norm_rope  # pylint: disable=C0415

    batch, rows, _ = hidden_states.shape
    packed = self.linear_qkv(hidden_states).flatten(0, 1)
    groups = self.num_key_value_heads
    cos, sin = positions
    query, key = qk_norm_rope(packed.contiguous(), local_target_tensor(self.q_norm.weight),
                              local_target_tensor(self.k_norm.weight), cos.reshape(rows, 128).contiguous(),
                              sin.reshape(rows, 128).contiguous(), groups, self.q_norm.variance_epsilon)
    value = packed.view(batch, rows, groups, self.qkv_group_width)[..., -self.v_head_dim:]
    return query.unsqueeze(0), key.unsqueeze(0), value, None

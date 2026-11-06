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
"""Restore native Qwen3-MoE rotary state after RL model materialization."""

import torch
from torch import nn
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRotaryEmbedding


def rebuild_qwen3_moe_rotary_state(model: nn.Module) -> None:
    """Recreate non-persistent RoPE buffers absent from pretrained checkpoints.

    Meta materialization allocates buffer storage without preserving constructor
    values. Rebuild from the native configuration instead of checkpoint tensors;
    both current and original frequencies must remain FP32 for dynamic RoPE.
    Call after model construction and before the first forward.
    """
    count = 0
    for module in model.modules():
        if not isinstance(module, Qwen3MoeRotaryEmbedding):
            continue
        device = module.inv_freq.device
        if device.type == "meta":
            raise ValueError("Qwen3-MoE RoPE recovery requires a materialized model")
        reference = Qwen3MoeRotaryEmbedding(config=module.config, device=device)
        for name, buffer in reference.named_buffers(recurse=False):
            module.register_buffer(name, buffer.to(dtype=torch.float32), persistent=False)
        module.attention_scaling = reference.attention_scaling
        module.max_seq_len_cached = reference.max_seq_len_cached
        module.original_max_seq_len = reference.original_max_seq_len
        count += 1
    if count == 0:
        raise ValueError("Qwen3-MoE model has no native rotary embedding to restore")

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
"""Parameter and buffer initialization for Kimi multimodal pretraining."""

# This model adapter implements the PyTorch Transformers training contract.
import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3RotaryEmbedding,
    DeepseekV3TopkRouter,
)

from hyper_parallel import DTensor, distribute_tensor


@torch.no_grad()
def initialize_kimi_k26_parameters(model: nn.Module) -> None:
    """Apply HF's normal/ones initialization through the actual DTensor layouts.

    HF initializes via ``parameter.data`` and indexes embedding padding rows
    globally. Here random operations retain the DTensor RNG offset tracking,
    and only the rank owning the padding row clears it. No full weight is
    allocated, including when EP stacks the expert projection matrices.

    Args:
        model: Materialized Kimi model with final FSDP/EP parameter layouts.
    """
    text_config = model.config.text_config
    for name, parameter in model.named_parameters():
        if name.endswith(".bias"):
            parameter.zero_()
        elif parameter.ndim == 1:
            parameter.fill_(1.0)
        else:
            std = 1.0 if name.endswith("pos_emb.weight") else text_config.initializer_range
            parameter.normal_(mean=0.0, std=std)
    embedding = model.get_input_embeddings()
    if embedding.padding_idx is not None:
        weight = embedding.weight
        rows = torch.arange(embedding.num_embeddings, device=weight.device)
        if isinstance(weight, DTensor):
            rows = distribute_tensor(rows, weight.device_mesh, weight.placements).to_local()
            weight = weight.to_local()
        weight[rows == embedding.padding_idx] = 0


def prepare_kimi_k26_training(model: nn.Module) -> None:
    """Initialize config-only buffers after model materialization.

    Call once after meta materialization and random parameter initialization,
    before any forward or checkpoint restore. This intentionally resets the
    router correction buffer; it must never run after restoring training state.
    Native attention, routing and expert projection modules remain unchanged.

    Args:
        model: Multimodal Kimi model after HyperParallel construction.
    """
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, DeepseekV3RotaryEmbedding):
                frequencies, scaling = module.rope_init_fn(module.config, module.inv_freq.device)
                module.inv_freq = frequencies
                module.original_inv_freq = frequencies
                module.attention_scaling = scaling
            elif isinstance(module, DeepseekV3TopkRouter):
                module.e_score_correction_bias.zero_()

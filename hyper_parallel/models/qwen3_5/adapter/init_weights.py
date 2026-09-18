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
"""Shard-aware weight initialization for Qwen3.5 models."""

import torch
from torch import nn
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet


@torch.no_grad()
def initialize_weights(model: nn.Module) -> None:
    """Initialize Qwen3.5 weights with shard-aware gated-delta state.

    Args:
        model: Materialized Qwen3.5 model whose parameters are already sharded.
    """
    for module in model.modules():
        if not isinstance(module, Qwen3_5GatedDeltaNet):
            continue
        nn.init.ones_(module.dt_bias)
        module.A_log.copy_(torch.empty_like(module.A_log).uniform_(0.01, 16.0).log_())
        module.dt_bias._is_hf_initialized = True  # pylint: disable=W0212
        module.A_log._is_hf_initialized = True  # pylint: disable=W0212
        module._is_hf_initialized = True  # pylint: disable=W0212

    model.initialize_weights()

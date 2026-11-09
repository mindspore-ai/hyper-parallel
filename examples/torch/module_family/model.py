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
"""Model-owned routing composes public Attention and Experts families."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from examples.torch.module_family.attention import CausalAttention
from examples.torch.module_family.config import DecoderConfig
from examples.torch.module_family.grouped_experts import GroupedExperts
from hyper_parallel.core.dtensor.placement_types import Replicate
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.registry import register_model_adapter


class FamilyDecoder(nn.Module):
    """A small causal block with model-owned routing and routed SwiGLU experts."""

    def __init__(self, config: DecoderConfig, *, mesh: Any = None, context: Any = None) -> None:
        """Compose ordinary family interfaces without parallel-specific branches.

        Args:
            config: Global algorithm dimensions.
            mesh: Existing job meshes borrowed for member selection, layout and binding.
            context: Internal constraints injected by the managed model builder.
        """
        super().__init__()
        self.config = config
        self.attention = CausalAttention(config=config, mesh=mesh, context=context)
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = GroupedExperts(config=config, mesh=mesh, context=context)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute the model without interpreting any parallel job flags.

        Args:
            hidden_states: Already selected local sequence tokens [B,S,H].
        """
        attended = hidden_states + self.attention(hidden_states)
        flat = attended.reshape(-1, self.config.hidden_size)
        probabilities = self.router(flat).softmax(dim=-1)
        weights, indices = probabilities.topk(self.config.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return attended + self.experts(flat, indices, weights).reshape_as(attended)


def _fsdp_units(model: nn.Module) -> tuple[str, ...]:
    del model
    return ("attention", "experts")


def _model_sharding(*, model: nn.Module, context: Any) -> dict[str, ModuleShardingSpec]:
    del model, context
    return {"router": ModuleShardingSpec(params={"weight": {"cp": Replicate()}},
                                        in_src={}, in_dst={}, is_boundary=False)}


MODEL_SPEC = ModelAdapterSpec("FamilyDecoder", "module_family_decoder", fsdp_wrap_modules=_fsdp_units,
                              sharding=_model_sharding)
register_model_adapter(MODEL_SPEC)

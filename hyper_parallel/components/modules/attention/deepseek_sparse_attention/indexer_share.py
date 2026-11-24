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
"""Invocation-local Indexer sharing contracts for DeepSeek-V3.2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class IndexSharePlan:
    """Validated global layer roles and the supervision policy."""

    leaders: tuple[int, ...]
    loss_mode: str

    @property
    def enabled(self) -> bool:
        """Whether any layer consumes another layer's selection."""
        return any(index != leader for index, leader in enumerate(self.leaders))

    def group_size(self, layer: int) -> int:
        """Return the actual group size, including a shorter final group."""
        return self.leaders.count(self.leaders[layer])


def resolve_index_share_plan(config: Any) -> IndexSharePlan:
    """Resolve explicit roles or MF-compatible zero-based frequency/offset.

    Args:
        config: Model config with num_hidden_layers and optional dsa_indexer_types,
            dsa_index_topk_freq, dsa_index_skip_topk_offset, and dsa_index_share_loss.
            Explicit roles take precedence over frequency and offset.
    """
    loss_mode = getattr(config, "dsa_index_share_loss", "served")
    if loss_mode not in ("leader", "served"):
        raise ValueError("dsa_index_share_loss must be 'leader' or 'served'")
    count = getattr(config, "num_hidden_layers", 0)
    if type(count) is not int or count < 0:
        raise ValueError("num_hidden_layers must be a nonnegative integer")
    roles = getattr(config, "dsa_indexer_types", None)
    frequency = getattr(config, "dsa_index_topk_freq", 1)
    offset = getattr(config, "dsa_index_skip_topk_offset", 0)
    if roles is not None:
        if not isinstance(roles, (tuple, list)) or len(roles) != count or not roles:
            raise ValueError("dsa_indexer_types must have num_hidden_layers entries")
        if any(role not in ("full", "shared") for role in roles) or roles[0] != "full":
            raise ValueError("dsa_indexer_types must start with 'full' and contain only 'full'/'shared'")
    else:
        if type(frequency) is not int or frequency < 1 or type(offset) is not int or offset < 0:
            raise ValueError("DSA index frequency must be a positive integer and offset a nonnegative integer")
        if frequency > 1 and count < 1:
            raise ValueError("Indexer sharing requires num_hidden_layers")
        roles = tuple("full" if max(index - offset, 0) % frequency == 0 else "shared"
                      for index in range(count))
    leader = 0
    leaders = []
    for index, role in enumerate(roles):
        if role == "full":
            leader = index
        leaders.append(leader)
    plan = IndexSharePlan(tuple(leaders), loss_mode)
    if plan.enabled and getattr(config, "dsa_index_share_topk_offload", False):
        raise ValueError("Indexer sharing does not yet support Top-K offload")
    return plan


@dataclass(frozen=True)
class IndexShareState:
    """One leader's selection and differentiable student projections.

    Projections retain their autograd edges; only the discrete selection is detached.
    The state belongs to one forward invocation, never an attention module.
    """

    index_states: tuple[torch.Tensor, ...]
    topk: torch.Tensor | None
    use_dense: bool
    index_topk: int


@dataclass
class IndexShareContext:
    """Explicit carrier shared by layers of one model invocation.

    Create a fresh instance for each forward. Do not reuse it for another batch.
    Non-reentrant checkpoint closures retain this argument for replay.
    """

    groups: dict[int, IndexShareState] = field(default_factory=dict)


def _start_index_share_forward(
    module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Create an invocation-local carrier at the decoder boundary."""
    del module
    if "index_share_context" in kwargs:
        raise ValueError("The model owns index_share_context; pass it only when calling individual attention layers")
    return args, {**kwargs, "index_share_context": IndexShareContext()}


def install_indexer_share(model: nn.Module) -> None:
    """Install the per-forward Indexer Share carrier on a V3.2 decoder.

    Args:
        model: HF causal LM or decoder model. All decoder attentions must use
            DeepseekSparseAttention; call before compilation or checkpoint wrapping.
    """
    decoder = getattr(model, "model", model)
    plan = resolve_index_share_plan(decoder.config)
    if not plan.enabled:
        return
    layers = getattr(decoder, "layers", ())
    if len(layers) != len(plan.leaders):
        raise ValueError("Indexer sharing requires the complete decoder layer stack")
    for layer in layers:
        attentions = [
            module for module in layer.modules()
            if hasattr(module, "index_share_plan")
        ]
        if any(module.index_share_plan != plan for module in attentions):
            raise ValueError("Decoder attention layers must use the same Indexer Share plan as the model config")
    if getattr(decoder, "_hp_index_share_hook", None) is None:
        decoder._hp_index_share_hook = decoder.register_forward_pre_hook(  # pylint: disable=protected-access
            _start_index_share_forward, with_kwargs=True,
        )

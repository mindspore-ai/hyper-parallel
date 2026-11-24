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
"""Bind shared-expert overlap without rewriting model forwards or routing interfaces."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

if TYPE_CHECKING:
    from hyper_parallel.distributed.expert_parallel.ep_context import EPContext


def bind_shared_expert_overlap(model: nn.Module, expert_fqn: str, context: EPContext) -> None:
    """Bind the containing MoE's shared family member before activation/FSDP wrapping.

    Args:
        model: Final module tree after replacement and layout.
        expert_fqn: Routed expert path in the parallel plan.
        context: Ordinary or MoonEP context controlling shared-stage scheduling.

    Native shared modules without the split-stage/cache protocol remain serial.
    The owner must feed the same activation to routed and shared branches.
    """
    owner = model.get_submodule(expert_fqn.rpartition(".")[0])
    shared = [child for child in owner.children()
              if getattr(child, "split_forward_supported", False) and callable(getattr(child, "bind_ep_context", None))]
    if len(shared) > 1:
        raise ValueError(f"MoE owner of {expert_fqn!r} has multiple shared-expert candidates")
    if shared:
        context.bind_shared_expert(owner, shared[0], model.get_submodule(expert_fqn))

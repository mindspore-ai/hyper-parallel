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
"""Public vocabulary projection and pre-shifted chunked cross-entropy family."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from hyper_parallel.components.losses.chunked_cross_entropy import ChunkedCausalLMOutput, chunked_cross_entropy
from hyper_parallel.components.modules.module_family import ModuleFamily, build_family, register_module_family
from hyper_parallel.models.replacement import module_replacement


class CausalLMHeadModule(nn.Linear):
    """Own the vocabulary weight while supporting logits and memory-bounded CE.

    Inheriting Linear retains HF initialization and embedding tying semantics.
    Source adoption allocates no replacement weight and changes no parameter key.
    """

    supports_chunked_cross_entropy = True

    def __init__(
        self, *, config: Any = None, module: nn.Module | None = None, module_fqn: str = "", context: Any = None
    ) -> None:
        """Construct a bias-free head or adopt an existing compatible projection."""
        del module_fqn, context
        if (config is None) == (module is None):
            raise ValueError("provide exactly one of config or module")
        if module is None:
            for name in ("hidden_size", "vocab_size"):
                value = getattr(config, name, None)
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError(f"{name} must be a positive integer")
            super().__init__(config.hidden_size, config.vocab_size, bias=False)
        else:
            if not isinstance(module, nn.Linear) or module.bias is not None:
                raise ValueError("CausalLMHead requires a bias-free Linear source")
            nn.Module.__init__(self)
            self.in_features, self.out_features = module.in_features, module.out_features
            self.register_parameter("weight", module.weight)
            self.register_parameter("bias", None)
            self.training = module.training

    def make_transforms(self) -> list:
        """Projection names and shapes are unchanged."""
        return []

    def forward(
        self,
        input: torch.Tensor,
        *,
        targets: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        chunk_size: int = 1024,
        ignore_index: int = -100,
    ) -> torch.Tensor | ChunkedCausalLMOutput:
        """Project to logits or compute summed CE from already aligned targets.

        Targets are pre-shifted by the data owner; this component never shifts a
        locally sharded sequence. Chunked mode never creates full-sequence logits.
        """
        if targets is None:
            if loss_mask is not None:
                raise ValueError("loss_mask requires aligned targets")
            return F.linear(input, self.weight)  # pylint: disable=not-callable
        if loss_mask is not None:
            if loss_mask.shape != targets.shape:
                raise ValueError("loss_mask and targets must have identical shapes")
            targets = targets.masked_fill(~loss_mask.bool(), ignore_index)
        loss = chunked_cross_entropy(input, targets, self.weight, chunk_size=chunk_size, ignore_index=ignore_index)
        return ChunkedCausalLMOutput(loss_sum=loss, valid_token_count=(targets != ignore_index).sum())


@module_replacement
def CausalLMHead(
    *,
    config: Any = None,
    module: nn.Module | None = None,  # pylint: disable=invalid-name
    module_fqn: str = "",
    context: Any = None,
    mesh: Any = None,
) -> nn.Module:
    """Construct the tracked vocabulary-projection and chunked-CE component."""
    return build_family(CausalLMHead, config=config, module=module, module_fqn=module_fqn, context=context, mesh=mesh)


register_module_family(
    ModuleFamily(
        name="causal_lm_head",
        baseline=CausalLMHead,
        variants={"local": "hyper_parallel.components.modules.common.causal_lm_head.CausalLMHeadModule"},
        default_variant="local",
    )
)

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
"""HF model orchestration composing the public terminal family for pre-shifted training loss."""
from __future__ import annotations

from typing import Any

import torch
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeForConditionalGeneration as HFMultimodalModel,
)

from hyper_parallel.components.losses.chunked_cross_entropy import ChunkedCausalLMOutput
from hyper_parallel.components.losses.router_load_balancing import token_choice_load_balancing_loss
from hyper_parallel.components.modules import CausalLMHead


def _forward_chunked(
    model: Any,
    targets: torch.Tensor,
    mask: torch.Tensor | None,
    chunk_size: int,
    ignore_index: int,
    kwargs: dict[str, Any],
) -> ChunkedCausalLMOutput:
    """Compose the actual backbone and terminal family; media fusion stays in the source architecture."""
    if kwargs.pop("labels", None) is not None or kwargs.get("past_key_values") is not None or kwargs.get("use_cache"):
        raise ValueError("chunked training requires pre-shifted targets without labels or KV cache")
    retained = kwargs.pop("logits_to_keep", 0)
    if isinstance(retained, torch.Tensor) or retained != 0 or kwargs.pop("return_dict", True) is False:
        raise ValueError("chunked training requires complete token coverage and return_dict=True")
    kwargs["use_cache"] = False
    config = getattr(model.config, "text_config", model.config)
    is_moe = hasattr(config, "num_experts")
    router_logits = kwargs.pop("output_router_logits", None)
    if is_moe:
        router_logits = config.output_router_logits if router_logits is None else router_logits
        kwargs["output_router_logits"] = router_logits
    outputs = model.model(return_dict=True, **kwargs)
    if not getattr(model.lm_head, "supports_chunked_cross_entropy", False):
        raise TypeError("chunked training requires the CausalLMHead family")
    result = model.lm_head(
        outputs.last_hidden_state, targets=targets, loss_mask=mask, chunk_size=chunk_size, ignore_index=ignore_index
    )
    aux_loss = None
    if is_moe and router_logits:
        aux_loss = token_choice_load_balancing_loss(
            outputs.router_logits, config.num_experts, config.num_experts_per_tok, kwargs.get("attention_mask")
        )
    return ChunkedCausalLMOutput(
        loss_sum=result.loss_sum,
        valid_token_count=result.valid_token_count,
        aux_loss=aux_loss,
        aux_loss_coef=float(getattr(config, "router_aux_loss_coef", 0.0)),
        router_logits=getattr(outputs, "router_logits", None),
        hidden_states=getattr(outputs, "hidden_states", None),
        attentions=getattr(outputs, "attentions", None),
    )


class Qwen3VLMoeForConditionalGeneration(HFMultimodalModel):
    """Retain source construction and ordinary outputs, with explicit chunked training composition."""

    supports_chunked_loss = True

    @classmethod

    def _can_set_experts_implementation(cls) -> bool:
        """Retain the native source capability when HF inspects this orchestration module."""
        return HFMultimodalModel._can_set_experts_implementation()

    def __init__(self, config: Any) -> None:
        """Install the terminal family while preserving source weight tying and initialization."""
        super().__init__(config)
        self.lm_head = CausalLMHead(module=self.lm_head)

    def forward(
        self,
        *args: Any,
        chunk_loss_targets: torch.Tensor | None = None,
        chunk_loss_mask: torch.Tensor | None = None,
        chunk_loss_chunk_size: int = 1024,
        chunk_loss_ignore_index: int = -100,
        **kwargs: Any,
    ) -> Any:
        """Dispatch normal source calls or compose backbone hidden states with pre-shifted loss targets."""
        if chunk_loss_targets is None:
            return super().forward(*args, **kwargs)
        if args:
            if len(args) != 1 or "input_ids" in kwargs:
                raise ValueError("chunked training accepts a single positional input_ids argument")
            kwargs["input_ids"] = args[0]
        return _forward_chunked(
            self, chunk_loss_targets, chunk_loss_mask, chunk_loss_chunk_size, chunk_loss_ignore_index, kwargs
        )

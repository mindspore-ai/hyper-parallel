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
"""HF-compatible Qwen3-MoE orchestration with a public terminal projection family."""
from __future__ import annotations

from typing import Any

import torch
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM as HFQwen3MoeForCausalLM

from hyper_parallel.components.losses.chunked_cross_entropy import ChunkedCausalLMOutput
from hyper_parallel.components.losses.router_load_balancing import token_choice_load_balancing_loss
from hyper_parallel.components.modules.common.causal_lm_head import CausalLMHead


class Qwen3MoeForCausalLM(HFQwen3MoeForCausalLM):
    """Keep HF construction and ordinary outputs; compose the public head for chunked training."""

    supports_chunked_loss = True

    @classmethod

    def _can_set_experts_implementation(cls) -> bool:
        """Keep the source Experts backend capability across model-owned orchestration."""
        return HFQwen3MoeForCausalLM._can_set_experts_implementation()

    def __init__(self, config: Any) -> None:
        """Retain HF parameters and tying while installing the terminal family."""
        super().__init__(config)
        self.lm_head = CausalLMHead(module=self.lm_head)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_router_logits: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        *,
        chunk_loss_targets: torch.Tensor | None = None,
        chunk_loss_mask: torch.Tensor | None = None,
        chunk_loss_chunk_size: int = 1024,
        chunk_loss_ignore_index: int = -100,
        **kwargs: Any,
    ) -> Any:
        """Preserve the HF call protocol and route pre-shifted targets to the public head."""
        if chunk_loss_targets is None:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_router_logits=output_router_logits,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )
        if labels is not None or past_key_values is not None or use_cache:
            raise ValueError("chunked training requires pre-shifted targets without labels or KV cache")
        if isinstance(logits_to_keep, torch.Tensor) or logits_to_keep != 0 or kwargs.get("return_dict") is False:
            raise ValueError("chunked training requires full token coverage and return_dict=True")
        output_router_logits = (
            bool(self.config.output_router_logits) if output_router_logits is None else output_router_logits
        )
        kwargs.pop("return_dict", None)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_router_logits=output_router_logits,
            return_dict=True,
            **kwargs,
        )
        if not getattr(self.lm_head, "supports_chunked_cross_entropy", False):
            raise TypeError("chunked training requires the CausalLMHead family")
        output = self.lm_head(
            outputs.last_hidden_state,
            targets=chunk_loss_targets,
            loss_mask=chunk_loss_mask,
            chunk_size=chunk_loss_chunk_size,
            ignore_index=chunk_loss_ignore_index,
        )
        aux_loss = None
        if output_router_logits:
            aux_loss = token_choice_load_balancing_loss(
                outputs.router_logits, self.num_experts, self.num_experts_per_tok, attention_mask
            )
        return ChunkedCausalLMOutput(
            loss_sum=output.loss_sum,
            valid_token_count=output.valid_token_count,
            aux_loss=aux_loss,
            aux_loss_coef=float(self.router_aux_loss_coef),
            router_logits=getattr(outputs, "router_logits", None),
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )

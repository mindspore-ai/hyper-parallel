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
"""Native K2.6 multimodal composition with a family-owned terminal projection/loss."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers.models.kimi_k25.modeling_kimi_k25 import (
    Kimi_K25ForConditionalGeneration as HFMultimodalModel,
    Kimi_K25Model as HFMultimodalCore,
    Kimi_K25PreTrainedModel,
    Kimi_K25VisionModel,
    Kimi_K25MultimodalProjection,
)
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3Model
from hyper_parallel.distributed.context_parallel.attention import build_cp_attention_mask
from hyper_parallel.models.kimi_k26 import KIMI_MODEL_TYPES

from hyper_parallel.components.losses.chunked_cross_entropy import ChunkedCausalLMOutput
from hyper_parallel.components.modules import CausalLMHead


class KimiLanguageModel(DeepseekV3Model):
    """Retain the HF decoder while owning its post-media sequence-partition call protocol."""

    cp_size = 1
    cp_rank = 0

    def bind_context_parallel(self, cp_size: int, cp_rank: int) -> None:
        """Bind decoder coordinates once without replacing any instance forward."""
        if (isinstance(cp_size, bool) or not isinstance(cp_size, int) or cp_size < 1
                or isinstance(cp_rank, bool) or not isinstance(cp_rank, int) or not 0 <= cp_rank < cp_size):
            raise ValueError("decoder CP coordinates require positive size and rank within that size")
        if self.cp_size != 1 and (self.cp_size, self.cp_rank) != (cp_size, cp_rank):
            raise ValueError("Kimi language model already has different CP coordinates")
        self.cp_size, self.cp_rank = cp_size, cp_rank

    def _run_language(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the HF-owned decoder after architecture input preparation."""
        return super().forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Partition fully fused media/text embeddings and preserve global positions and padding."""
        # HF forwards shared kwargs to both towers; visual geometry is consumed
        # exclusively by the complete visual family, never by decoder attention.
        kwargs.pop("vision_geometry", None)
        if self.cp_size == 1:
            return self._run_language(*args, **kwargs)
        if args:
            raise TypeError("Kimi CP language-model inputs must be supplied by keyword")
        hidden = kwargs.get("inputs_embeds")
        if hidden is None:
            raise RuntimeError("Kimi CP requires inputs_embeds after the complete media scatter")
        length = hidden.shape[1]
        if length % self.cp_size:
            raise ValueError("Kimi padded sequence length must be divisible by cp_size")
        if kwargs.get("past_key_values") is not None or kwargs.get("use_cache"):
            raise ValueError("Kimi CP training does not accept KV cache")
        local_length = length // self.cp_size
        start = self.cp_rank * local_length
        prepared = dict(kwargs)
        prepared["inputs_embeds"] = hidden[:, start:start + local_length]
        positions = kwargs.get("position_ids")
        if positions is None:
            positions = torch.arange(length, device=hidden.device).unsqueeze(0).expand(hidden.shape[0], -1)
        elif (positions.ndim != 2 or positions.shape[0] not in (1, hidden.shape[0])
              or positions.shape[1] != length):
            raise ValueError("Kimi CP position_ids must cover the full fused sequence as [1 or batch, sequence]")
        prepared["position_ids"] = positions[:, start:start + local_length]
        prepared["attention_mask"] = build_cp_attention_mask(
            kwargs.get("attention_mask"), q_len=length, kv_len=length,
            query_offset=0, device=hidden.device,
        )
        prepared.pop("input_ids", None)
        prepared["past_key_values"], prepared["use_cache"] = None, False
        return self._run_language(**prepared)


class KimiMultimodalModel(HFMultimodalCore):
    """Compose native vision, decoder and projection without installing forward callbacks."""

    def __init__(self, config: Any) -> None:
        """Construct each source boundary once in the source initialization order."""
        # Build each child once; the full HF parent would construct a second decoder.
        Kimi_K25PreTrainedModel.__init__(self, config)  # pylint: disable=non-parent-init-called
        if config.text_config.model_type != "deepseek_v3":
            raise ValueError("Kimi K2.6 native composition requires a DeepseekV3 text configuration")
        self.vision_tower = Kimi_K25VisionModel._from_config(config.vision_config)
        self.language_model = KimiLanguageModel._from_config(config.text_config)
        self.mm_projector = Kimi_K25MultimodalProjection(config)
        self.post_init()


def bind_context_parallel(model: nn.Module, mesh_context: Any) -> None:
    """Bind the architecture's decoder CP coordinates without rewriting its methods."""
    size, rank = mesh_context.cp_size, mesh_context.cp_rank
    if size == 1:
        return
    if getattr(getattr(model, "config", None), "model_type", None) not in KIMI_MODEL_TYPES:
        raise TypeError(f"Kimi CP requires config.model_type in {KIMI_MODEL_TYPES}")
    language = getattr(getattr(model, "model", None), "language_model", None)
    if not isinstance(language, KimiLanguageModel):
        raise TypeError("Kimi CP requires owned model.model.language_model; build the registered native model")
    language.bind_context_parallel(size, rank)


class Kimi_K25ForConditionalGeneration(HFMultimodalModel):
    """Keep native media fusion and compose its final hidden states with the public head."""

    supports_chunked_loss = True

    def __init__(self, config: Any) -> None:
        """Preserve source construction, weight tying and the terminal weight identity."""
        # Build each child once; the full HF parent would construct a second decoder.
        Kimi_K25PreTrainedModel.__init__(self, config)  # pylint: disable=non-parent-init-called
        self.model = KimiMultimodalModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()
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
        """Use normal HF outputs or evaluate pre-shifted targets before full logits exist."""
        coordinates = kwargs.pop("kimi_cp_coordinates", None)
        if coordinates is not None:
            self.model.language_model.bind_context_parallel(*coordinates)
        if self.model.language_model.cp_size > 1 and chunk_loss_targets is None:
            raise ValueError("Kimi multimodal CP requires globally shifted local chunk_loss_targets")
        if chunk_loss_targets is None:
            return super().forward(*args, **kwargs)
        if (
            kwargs.pop("labels", None) is not None
            or kwargs.get("past_key_values") is not None
            or kwargs.get("use_cache")
        ):
            raise ValueError("chunked training requires pre-shifted targets without labels or KV cache")
        retained = kwargs.pop("logits_to_keep", 0)
        if isinstance(retained, torch.Tensor) or retained != 0 or kwargs.pop("return_dict", True) is False:
            raise ValueError("chunked training requires complete token coverage and return_dict=True")
        if args:
            if len(args) != 1 or "input_ids" in kwargs:
                raise ValueError("chunked training accepts a single positional input_ids argument")
            kwargs["input_ids"] = args[0]
        kwargs["use_cache"] = False
        outputs = self.model(return_dict=True, **kwargs)
        if not getattr(self.lm_head, "supports_chunked_cross_entropy", False):
            raise TypeError("chunked training requires the CausalLMHead family")
        result = self.lm_head(
            outputs.last_hidden_state,
            targets=chunk_loss_targets,
            loss_mask=chunk_loss_mask,
            chunk_size=chunk_loss_chunk_size,
            ignore_index=chunk_loss_ignore_index,
        )
        return ChunkedCausalLMOutput(
            loss_sum=result.loss_sum,
            valid_token_count=result.valid_token_count,
            past_key_values=getattr(outputs, "past_key_values", None),
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )

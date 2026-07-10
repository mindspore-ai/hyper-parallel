# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Multi-Token-Prediction (MTP) head for Qwen3.5-MoE.

The Qwen3.5-MoE checkpoint ships an MTP head (``mtp.*``) for speculative
decoding / next-2-token training. The head is implemented from the checkpoint
layout and Qwen MTP design: a single decoder module that, at each position
``i``, fuses the main trunk's hidden state ``h_i`` with the embedding of token
``t_{i+1}`` and predicts ``t_{i+2}`` through the shared output head:

    h'_i = fc([ RMSNorm(h_i) ; RMSNorm(Emb(t_{i+1})) ])
    h''_i = TransformerLayer(h'_i)
    logit_{i+2} = lm_head(RMSNorm_final(h''_i))

The MTP path is validated structurally: it loads every ``mtp.*`` weight and
runs. An enabled MTP loss is outside the main-loss guarantees covered by this
module family.
The main-loss path is unaffected unless ``mtp_loss_weight > 0``.
"""
# pylint: disable=C0103  # Qwen class-name convention (Qwen3_5*)
import copy
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from hyper_parallel.models.qwen3_5_moe.model import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeDecoder,
    Qwen3_5RMSNorm,
)
from hyper_parallel.models.modules.rope import MultiModalRotaryEmbedding


class Qwen3_5MoeMTP(nn.Module):
    """One MTP module: pre-fusion norms + ``fc`` + one full-attention decoder
    layer + a final norm. ``embed_tokens`` / ``lm_head`` are shared with the
    main model (passed in at call time), so this module owns only the ``mtp.*``
    parameters.

    Submodule names match the checkpoint::

        mtp.pre_fc_norm_embedding / mtp.pre_fc_norm_hidden
        mtp.fc
        mtp.layers.0.*   (a Qwen3.5-MoE full-attention decoder layer)
        mtp.norm
    """

    def __init__(self, config: Qwen3_5MoeConfig,
                 rotary_emb: MultiModalRotaryEmbedding):
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(h, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(h, eps=config.rms_norm_eps)
        self.fc = nn.Linear(2 * h, h, bias=False)
        # The MTP decoder layer is always a full-attention (gated GQA + MoE)
        # block regardless of the trunk's per-layer dispatch, so build it with a
        # single-entry ``full_attention`` layer_types. It shares the trunk's
        # ``rotary_emb`` (the same instance every trunk layer uses, built from the
        # same text config) so the rope geometry can never silently diverge.
        layer_cfg = copy.copy(config)
        layer_cfg.layer_types = ["full_attention"]
        self.layers = nn.ModuleList([Qwen3_5MoeDecoder(layer_cfg, 0, rotary_emb)])
        self.norm = Qwen3_5RMSNorm(h, eps=config.rms_norm_eps)

    def forward(
        self,
        trunk_hidden: torch.Tensor,
        next_token_embeds: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the MTP final hidden states ``[B, S, H]`` (pre lm_head).

        Args:
            trunk_hidden: the main trunk's pre-final-norm hidden ``[B, S, H]``.
            next_token_embeds: ``Emb(t_{i+1})`` aligned to position ``i``
                (the caller left-shifts ``input_ids`` by one before embedding).
        """
        hn = self.pre_fc_norm_hidden(trunk_hidden)
        en = self.pre_fc_norm_embedding(next_token_embeds)
        fused = self.fc(torch.cat([hn, en], dim=-1).to(self.fc.weight.dtype))
        hidden_states = fused.to(torch.float32)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, position_ids=position_ids, attention_mask=attention_mask,
            )
        return self.norm(hidden_states)


def mtp_loss(mtp_logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Next-2-token cross-entropy: position ``i`` predicts ``t_{i+2}``.

    ``input_ids`` are the raw ids; the last two positions have no t+2 target and
    are masked with ``-100``.
    """
    targets = F.pad(input_ids, (0, 2), value=-100)[..., 2:].contiguous()
    logits_fp = mtp_logits.float()
    return F.cross_entropy(
        logits_fp.view(-1, logits_fp.size(-1)),
        targets.view(-1),
        ignore_index=-100,
        reduction="mean",
    )


__all__ = ["Qwen3_5MoeMTP", "mtp_loss"]

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
"""Teacher-forced LM/MTP training over the official Xing4 backbone."""
from functools import partial
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers.masking_utils import create_causal_mask
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4HyperConnection
from transformers.utils import ModelOutput

from hyper_parallel.components.functional.mhc_post import mhc_post
from hyper_parallel.components.losses.chunked_cross_entropy import chunked_cross_entropy
from hyper_parallel.components.modules.mtp import MTPLayer, build_mtp
from hyper_parallel.data.constants import IGNORE_INDEX
from hyper_parallel.models.materialization import register_rebuildable_buffer
from hyper_parallel.models.xing_4.adapter.moe import Xing4MoE, Xing4Router, adapt_mlp
from hyper_parallel.models.xing_4.transformer.configuration_xing4_0 import Xing4_0Config
from hyper_parallel.models.xing_4.transformer.modeling_xing4_0 import (
    Xing4_0DecoderLayer, Xing4_0ForCausalLM, Xing4_0HyperConnection,
    Xing4_0Model, Xing4_0MoE, Xing4_0PreTrainedModel, Xing4_0RMSNorm,
)


class Xing4HyperConnection(Xing4_0HyperConnection):
    """Retain official parameters with the FP32 training mHC contract."""

    def __init__(self, config: Any, source: Xing4_0HyperConnection) -> None:
        """Adopt the official projection and coefficients without renaming parameters."""
        nn.Module.__init__(self)
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.input_norm = source.input_norm
        self.input_norm.eps = config.mhc_layernorm_epsilon
        self.hc_fn, self.hc_base, self.hc_scale = source.hc_fn, source.hc_base, source.hc_scale

    @property
    def fn(self) -> nn.Parameter:
        """Expose the shared portable mHC projection without a second weight owner."""
        return self.hc_fn

    @property
    def base(self) -> nn.Parameter:
        """Expose the coefficient offsets to the shared portable implementation."""
        return self.hc_base

    @property
    def scale(self) -> nn.Parameter:
        """Expose the three coefficient scales to the shared implementation."""
        return self.hc_scale

    def forward(self, hidden_streams: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Reuse FP32 projection and stabilized Sinkhorn from Transformers."""
        return (*DeepseekV4HyperConnection.forward(self, hidden_streams), hidden_streams)



class Xing4DecoderLayer(Xing4_0DecoderLayer):
    """Keep HF submodules and expose its inline residual mixing for replacement."""

    def __init__(self, config: Any, layer_idx: int, source: nn.Module | None = None) -> None:
        """Retain HF attention, norms and feed-forward weights."""
        nn.Module.__init__(self)
        source = Xing4_0DecoderLayer(config, layer_idx) if source is None else source
        self.hidden_size = source.hidden_size
        self.self_attn = source.self_attn
        self.mlp = (Xing4MoE(config, source.mlp) if isinstance(source.mlp, Xing4_0MoE)
                    else adapt_mlp(source.mlp))
        self.input_layernorm = source.input_layernorm
        self.post_attention_layernorm = source.post_attention_layernorm
        self.attn_hc = Xing4HyperConnection(config, source.attn_hc)
        self.ffn_hc = Xing4HyperConnection(config, source.ffn_hc)
        # The training recipe applies its norm epsilon to both MLA latent projections.
        self.self_attn.q_a_layernorm.variance_epsilon = config.rms_norm_eps
        self.self_attn.kv_a_layernorm.variance_epsilon = config.rms_norm_eps
        if config.attention_softmax_scale is not None:
            self.self_attn.scaling = config.attention_softmax_scale

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None,
                position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, **kwargs: Any) -> torch.Tensor:
        """Call the official blocks through replaceable pre/post mHC operations."""
        post, mix, collapsed, residual = self.attn_hc(hidden_states)
        output = self.self_attn(hidden_states=self.input_layernorm(collapsed),
                                position_embeddings=position_embeddings, attention_mask=attention_mask, **kwargs)[0]
        hidden_states = self._post(output, residual, post, mix)
        post, mix, collapsed, residual = self.ffn_hc(hidden_states)
        output = self.mlp(self.post_attention_layernorm(collapsed))
        return self._post(output, residual, post, mix)

    @staticmethod
    def _post(output: torch.Tensor, residual: torch.Tensor,
              post: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
        """Convert destination-by-source mixing to the shared post layout."""
        streams = residual.shape[2]
        result = mhc_post(output, residual.flatten(2), post, mix.transpose(-1, -2).contiguous(), streams)
        return result.unflatten(-1, (streams, output.shape[-1]))


@torch.no_grad()
def initialize(owner: Any, module: nn.Module) -> None:
    """Initialize training state and repair the bundled HC initializer's field lookup."""
    if isinstance(module, Xing4_0HyperConnection):
        nn.init.normal_(module.hc_fn, mean=0.0, std=1e-4)
        module.hc_scale.fill_(owner.config.mhc_init_gating_factor)
        module.hc_base.zero_()
    elif isinstance(module, Xing4Router):
        nn.init.normal_(module.weight, mean=0.0, std=owner.config.initializer_range)
        module._token_counts.zero_()
        module.e_score_correction_bias.zero_()
    else:
        Xing4_0PreTrainedModel._init_weights(owner, module)


class Xing4Model(Xing4_0Model):
    """Use the unchanged HF backbone forward with training-aware decoder blocks."""

    config_class = Xing4_0Config
    _supports_attention_backend = False

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.layers = nn.ModuleList(
            Xing4DecoderLayer(config, index, source=layer) for index, layer in enumerate(self.layers))
        rotary = self.rotary_emb
        rope_init_fn = (rotary.compute_default_rope_parameters if rotary.rope_type == "default"
                        else ROPE_INIT_FUNCTIONS[rotary.rope_type])
        with torch.device("cpu"):
            frequencies, rotary.attention_scaling = rope_init_fn(config, device="cpu")
        for name in ("inv_freq", "original_inv_freq"):
            register_rebuildable_buffer(rotary, name, value=frequencies)
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        initialize(self, module)

    @staticmethod
    def shift_tokens(tokens: torch.Tensor, pad_value: int) -> torch.Tensor:
        """Shift a full training sequence; CP recipes replace its boundary exchange."""
        return F.pad(tokens[:, 1:], (0, 1), value=pad_value)

    @staticmethod
    def mean_loss(loss_sum: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
        """Average one sequence objective; CP recipes reduce its numerator and count."""
        return loss_sum / count.clamp_min(1)


class Xing4MTPBlock(Xing4DecoderLayer):
    """Adapt the shared MTP block's 3-D input to official Xing4 residual streams."""

    def __init__(self, config: Any, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        if not isinstance(self.mlp, Xing4MoE):
            raise ValueError("Xing4 MTP requires an MoE decoder at its native layer index")
        self.hc_mult = config.hc_mult

    def forward(self, hidden_states: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Expand and collapse streams, leaving fusion and norms to MTPLayer."""
        streams = hidden_states.unsqueeze(2).expand(-1, -1, self.hc_mult, -1).contiguous()
        return super().forward(streams, **kwargs).mean(dim=2)


class Xing4ForCausalLM(Xing4_0ForCausalLM):
    """Add teacher-forced LM/MTP losses to the official causal language model."""

    config_class = Xing4_0Config
    _no_split_modules = ["Xing4DecoderLayer", "MTPLayer"]
    _keys_to_ignore_on_load_unexpected = None
    _can_record_outputs = None
    _supports_flash_attn = False
    _supports_flex_attn = False
    _supports_attention_backend = False

    def __init__(self, config: Any) -> None:
        Xing4_0PreTrainedModel.__init__(self, config)
        if config.hidden_act != "silu" or config.q_lora_rank is None:
            raise ValueError("Xing4 training requires SiLU and low-rank query projection")
        if config.rope_parameters["rope_type"] not in ("default", "yarn") or not config.rope_interleave:
            raise ValueError("Xing4 training requires default or YaRN interleaved RoPE")
        if getattr(config, "enable_cmhc", False) or getattr(config, "enable_hc_head", False):
            raise ValueError("This Xing4 backbone uses the official mHC streams and mean collapse")
        if not getattr(config, "mtp_enable_hyper_connections", True):
            raise ValueError("Xing4 MTP uses the same mHC decoder as the backbone")
        if config.attention_dropout:
            raise ValueError("Xing4 training requires attention_dropout=0")
        compute_dtype = getattr(config, "compute_dtype", None)
        self.embedding_compute_dtype = (getattr(torch, compute_dtype, compute_dtype)
                                        if isinstance(compute_dtype, str) else compute_dtype)
        if self.embedding_compute_dtype not in (None, torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(f"Unsupported Xing4 compute_dtype: {compute_dtype!r}")
        self.model = Xing4Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mtp = build_mtp(config.num_nextn_predict_layers, layer_factory=lambda depth: MTPLayer(
            config.hidden_size, Xing4MTPBlock(config, config.num_hidden_layers + depth),
            norm_factory=partial(Xing4_0RMSNorm, eps=config.rms_norm_eps)))
        if self.mtp is not None:
            self.mtp.gradient_checkpointing = False
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        initialize(self, module)

    @classmethod
    def from_config(cls, config: Any, **kwargs: Any) -> "Xing4ForCausalLM":
        """Construct through the shared custom-model builder contract."""
        del kwargs
        return cls(config)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None,
                shift_labels: torch.Tensor | None = None, position_ids: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None, use_cache: bool = False) -> ModelOutput:
        """Return LM and weighted MTP losses for aligned causal training sequences."""
        del labels
        if attention_mask is not None or use_cache:
            raise ValueError("Xing4 trains causal sequences without attention masks or KV cache")
        if shift_labels is None or input_ids.ndim != 2 or min(input_ids.shape) < 1 \
                or shift_labels.shape != input_ids.shape:
            raise ValueError("Xing4 requires nonempty [batch, sequence] input_ids with matching shift_labels")
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        embeddings = self.model.embed_tokens(input_ids).to(dtype=self.embedding_compute_dtype)
        hidden = self.model(inputs_embeds=embeddings, position_ids=position_ids, use_cache=False).last_hidden_state
        targets = shift_labels.masked_fill(shift_labels < 0, IGNORE_INDEX)
        losses = {"foundation_loss": self._cross_entropy(hidden, targets)}
        if self.mtp is not None:
            position_embeddings = self.model.rotary_emb(hidden, position_ids)
            losses["foundation_loss/mtp"] = self._mtp_loss(
                hidden, input_ids, targets, position_embeddings, position_ids)
        return ModelOutput(loss=losses)

    def _mtp_loss(self, hidden: torch.Tensor, input_ids: torch.Tensor, targets: torch.Tensor,
                  position_embeddings: tuple[torch.Tensor, torch.Tensor],
                  position_ids: torch.Tensor) -> torch.Tensor:
        """Return ``mtp_loss_factor / D * sum_k CE_k``, where depth ``k`` predicts ``k`` tokens further.

        Inputs and targets shift left by one token per depth without wrapping.
        """
        total = torch.zeros((), device=hidden.device, dtype=torch.float32)
        for layer in self.mtp.layers:
            input_ids = self.model.shift_tokens(input_ids, 0)
            targets = self.model.shift_tokens(targets, IGNORE_INDEX)
            embeddings = self.model.embed_tokens(input_ids).to(dtype=self.embedding_compute_dtype)
            attention_mask = create_causal_mask(
                config=self.config, inputs_embeds=embeddings, attention_mask=None,
                past_key_values=None, position_ids=position_ids)
            hidden = layer(hidden, embeddings, position_embeddings=position_embeddings,
                           attention_mask=attention_mask)
            total = total + self._cross_entropy(hidden, targets)
        return total * (self.config.mtp_loss_factor / len(self.mtp.layers))

    def _cross_entropy(self, hidden: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Return the LM-head cross entropy averaged over the non-ignored targets.

        A head holding the whole vocabulary uses the chunked cross entropy, which never
        materializes the sequence's logits. A tensor-parallel head runs as a module, so its
        sharding plan gathers the logits or, under loss parallelism, keeps them
        vocabulary-sharded for the ``loss_function`` the Trainer installs.
        """
        count = targets.ne(IGNORE_INDEX).sum()
        weight = self.lm_head.weight
        if getattr(weight, "placements", None) is None and weight.shape[0] == self.config.vocab_size:
            chunk_size = getattr(self.config, "chunk_loss_size", None)
            if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
                raise ValueError("Full-vocabulary Xing4 cross entropy requires a positive integer chunk_loss_size")
            loss_sum = chunked_cross_entropy(hidden, targets, weight,
                                             chunk_size=chunk_size, ignore_index=IGNORE_INDEX)
        else:
            loss_sum = self.loss_function(self.lm_head(hidden), None, self.config.vocab_size, num_items_in_batch=1,
                                          ignore_index=IGNORE_INDEX, shift_labels=targets)
        return self.model.mean_loss(loss_sum, count)

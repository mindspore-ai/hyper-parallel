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
"""Model wrapper for GraphTrainer TP/SP demo.

Self-contained pure-PyTorch model that reuses AutoModel's **pipeline**
(``ShardingPlanner.plan`` + ``apply_sharding_plan``) while keeping the
module implementations local to avoid NPU-only functional dependencies.

Key design property: ``rotary_emb`` lives inside ``Attention.forward``
(not at the model level), so cos/sin are computed *after* the SP boundary
all-gather — they naturally have the full sequence length.

Submodule names (``q_proj``, ``k_proj``, ``v_proj``, ``o_proj``,
``gate_proj``, ``up_proj``, ``down_proj``, ``embed_tokens``, ``lm_head``)
match the ShardingPlanner conventions so ``ShardingPlanner.plan()``
derives correct TP placements without any plan_overrides.

Public API:
    - ``AutoModelAdapterForCausalLM`` -- CausalLM model
    - ``build_model`` -- factory: create model from config dict
    - ``shard_for_sp`` -- shard input tensor along sequence dim for SP
    - ``DataSampler`` -- data sampler that yields full-sequence batches
"""

import torch
from torch import nn
import torch.nn.functional as F
from transformers import LlamaConfig

from hyper_parallel.compile.examples.automodel_tp.mock_modules import (
    GroupQueryAttention,
    RMSNorm,
    SwiGLUMLP,
)


# ============================================================================
# Model assembly
# ============================================================================


class AutoModelAdapterDecoderLayer(nn.Module):
    """Decoder layer: attention + MLP with pre-norm residuals."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.self_attn = GroupQueryAttention(config)
        self.mlp = SwiGLUMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)

    def forward(self, hidden_states, position_ids=None):
        """Standard pre-norm decoder layer forward: attn + MLP with residuals."""
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_ids=position_ids)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class AutoModelAdapterModel(nn.Module):
    """Inner model: embed_tokens, layers, norm.

    No rotary_emb at this level -- it lives inside each
    ``GroupQueryAttention`` (SP-safe design).
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            AutoModelAdapterDecoderLayer(config)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size)

    def forward(self, input_ids, position_ids=None):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids=position_ids)
        return self.norm(hidden_states)


class AutoModelAdapterForCausalLM(nn.Module):
    """CausalLM model for GraphTrainer TP/SP demo.

    Reuses AutoModel's ``ShardingPlanner`` + ``apply_sharding_plan``
    pipeline. The model itself uses pure-PyTorch modules to ensure
    CPU/NPU compatibility.

    Submodule layout (matches ShardingPlanner conventions):
        model.embed_tokens.weight
        model.layers.{i}.self_attn.{q,k,v,o}_proj
        model.layers.{i}.mlp.{gate,up,down}_proj
        model.layers.{i}.{input,post_attention}_layernorm
        model.norm
        lm_head.weight
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.model = AutoModelAdapterModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, labels=None, position_ids=None):
        """Forward: embeddings -> decoder layers -> norm -> lm_head -> loss.

        Args:
            input_ids: token IDs of shape (batch, seq_len).
            labels: optional target token IDs for CE loss.
            position_ids: optional; if None, derived from seq_len.

        Returns:
            Dict with ``loss`` (or None) and ``logits``.
        """
        seq_len = input_ids.shape[1]
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        hidden_states = self.model(input_ids, position_ids=position_ids)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return {"loss": loss, "logits": logits}


# ============================================================================
# SP data utilities
# ============================================================================

def shard_for_sp(tensor, tp_size, tp_rank):
    """Shard input tensor along sequence dimension (dim 1) for SP.

    In SP mode, each TP rank receives a 1/tp_size slice of the sequence.
    The boundary forward's in_plan (Shard(1) -> Replicate()) will
    all-gather it back to full sequence inside the model.
    """
    seq_len = tensor.shape[1]
    chunk = seq_len // tp_size
    if seq_len % tp_size != 0:
        raise ValueError(
            f"Sequence length {seq_len} not divisible by tp_size {tp_size}"
        )
    return tensor[:, tp_rank * chunk : (tp_rank + 1) * chunk].contiguous()


def build_model(cfg: dict, device: torch.device) -> torch.nn.Module:
    """Factory: create a CausalLM from config dict."""
    model_cfg = LlamaConfig(
        vocab_size=cfg["vocab_size"],
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        max_position_embeddings=cfg["max_position_embeddings"],
        torch_dtype=cfg.get("torch_dtype", "float32"),
        tie_word_embeddings=False,
    )
    return AutoModelAdapterForCausalLM(model_cfg).to(device)


class DataSampler:
    """Yield dummy batches for training.

    In SP mode, yields full-sequence data; SP sharding is handled by
    the embedding boundary's reduce-scatter on hidden_states.
    """

    def __init__(
        self,
        vocab_size: int,
        batch_size: int,
        seq_len: int,
        max_steps: int,
        tp_size: int,
        tp_rank: int,
        sequence_parallel: bool,
        device: torch.device,
    ):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.max_steps = max_steps
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.sequence_parallel = sequence_parallel
        self.device = device

    def sample(self):
        """Return one (input_ids, labels) pair with full sequence length."""
        inp = torch.randint(
            0, self.vocab_size,
            (self.batch_size, self.seq_len), device=self.device
        )
        lbl = torch.randint(
            0, self.vocab_size,
            (self.batch_size, self.seq_len), device=self.device
        )
        return inp, lbl

    def __iter__(self):
        for _ in range(self.max_steps):
            yield self.sample()

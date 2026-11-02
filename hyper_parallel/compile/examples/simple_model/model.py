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
"""Toy causal LM for the config-driven graph-mode ``TextTrainer`` example.

The model follows the Trainer model contract and the activation-checkpoint
discovery contract:

- it is a ``torch.nn.Module`` with a ``config`` exposing ``to_dict()`` (the
  Trainer logs the model config);
- ``forward(**model_inputs, use_cache=False)`` returns an object exposing a
  ``loss`` tensor (the default ``ModelOutputLoss`` reads ``model_output.loss``);
- the ``gradient_checkpointing`` marker plus the repeated ``layers`` container
  make the transformer blocks discoverable by activation checkpointing (the
  same structural contract the eager path requires).

``build_model`` is the YAML ``model._target_`` factory. FSDP sharding and
activation recompute are left to the graph-mode ``FSDPPass`` /
``ActivationCheckpointPass`` (``compile.enabled`` + ``use_joint_graph``), so
the factory only materializes the model on the current device.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from hyper_parallel.trainer.runtime.device import get_device_type

__all__ = [
    "SimpleCausalLMConfig",
    "SimpleCausalLMOutput",
    "SimpleBlock",
    "SimpleCausalLM",
    "build_model",
]


class SimpleCausalLMConfig:
    """Minimal Transformers-style config exposing ``to_dict()``."""

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        hidden_dim: int,
        num_layers: int,
        max_seq_len: int,
    ) -> None:
        """Record the toy model dimensions."""
        self.vocab_size = vocab_size
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.model_type = "simple_graph_lm"
        self.architectures = ["SimpleCausalLM"]

    def to_dict(self) -> dict:
        """Return the JSON-serializable config mapping used by the Trainer."""
        return {
            "model_type": self.model_type,
            "architectures": list(self.architectures),
            "vocab_size": self.vocab_size,
            "dim": self.dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "max_seq_len": self.max_seq_len,
        }


class SimpleCausalLMOutput:
    """Model output carrying the (optional) causal-LM ``loss`` and ``logits``."""

    def __init__(self, loss: torch.Tensor | None, logits: torch.Tensor) -> None:
        """Store the loss and logits."""
        self.loss = loss
        self.logits = logits


class SimpleBlock(nn.Module):
    """One residual transformer-style MLP block (the recompute unit)."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        """Build the two residual sublayers."""
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Run norm -> MLP -> residual, then norm -> proj -> residual."""
        hidden = hidden + self.fc2(self.act(self.fc1(self.norm1(hidden))))
        return hidden + self.proj(self.norm2(hidden))


class SimpleCausalLM(nn.Module):
    """Embedding + residual blocks + projection causal LM.

    The ``gradient_checkpointing`` attribute plus the repeated ``layers``
    container is the structural contract activation checkpointing discovers;
    the graph pass wraps no modules, it only reads this structure to resolve
    the recompute scope.
    """

    def __init__(self, config: SimpleCausalLMConfig) -> None:
        """Build the embedding, the repeated blocks, the norm and the head."""
        super().__init__()
        self.config = config
        # The eager path consumes this flag; graph mode only reads its presence.
        self.gradient_checkpointing = False
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(
            SimpleBlock(config.dim, config.hidden_dim) for _ in range(config.num_layers)
        )
        self.norm = nn.LayerNorm(config.dim)
        self.linear = nn.Linear(config.dim, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> SimpleCausalLMOutput:
        """Run the blocks, project to logits, and build the shifted CE loss.

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)``.
            labels: Optional next-token targets of shape ``(batch, seq_len)``.
            use_cache: Accepted for the Trainer forward contract; unused.
            **kwargs: Ignored extra model inputs.

        Returns:
            Output exposing ``loss`` (``None`` without labels) and ``logits``.
        """
        del use_cache, kwargs
        hidden = self.embed(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        logits = self.linear(self.norm(hidden))

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )
        return SimpleCausalLMOutput(loss=loss, logits=logits)


def build_model(
    *,
    vocab_size: int,
    dim: int,
    hidden_dim: int,
    num_layers: int,
    max_seq_len: int,
    distributed_setup: Any = None,
    **kwargs: Any,
) -> SimpleCausalLM:
    """YAML ``model`` factory: build the toy model on the current device.

    Args:
        vocab_size: Vocabulary size.
        dim: Hidden dimension.
        hidden_dim: MLP expansion dimension.
        num_layers: Number of repeated recompute blocks.
        max_seq_len: Maximum sequence length recorded in the config.
        distributed_setup: Trainer distributed setup (unused; FSDP sharding is
            done by the graph-mode ``FSDPPass``).
        **kwargs: Ignored extra Trainer build kwargs (including the
            ``activation_checkpoint`` options projected by the Trainer: graph
            mode recomputes on the traced joint graph, not by wrapping the
            live module).

    Returns:
        The materialized ``SimpleCausalLM`` on the current device.
    """
    del distributed_setup, kwargs
    config = SimpleCausalLMConfig(
        vocab_size=vocab_size,
        dim=dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
    )
    model = SimpleCausalLM(config)
    return model.to(torch.device(get_device_type()))

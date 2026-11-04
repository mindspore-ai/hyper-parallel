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
"""Logical MLA/MLP matrix boundaries for framework Muon callbacks.

Head grouping follows townwish4git's JTMuonLayout (hf1s/hyper-parallel#303).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import cached_property
from typing import Any

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.core.optimizer.muon import NSInputTransform


@dataclass(frozen=True)
class Xing4MuonLayout:
    """Describe complete logical matrices for the Muon callbacks.

    Tensor-parallel fragments must contain complete matrix axes; independent
    expert batches may stay local. The framework scales each logical matrix.
    """

    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    num_attention_heads: int
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    n_shared_experts: int = 0

    def __post_init__(self) -> None:
        """Validate matrix dimensions before constructing the optimizer."""
        for item in fields(self):
            value = getattr(self, item.name)
            minimum = 0 if item.name == "n_shared_experts" else 1
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"Xing4 Muon layout requires integer {item.name} >= {minimum}")

    @classmethod
    def from_config(cls, config: Any) -> "Xing4MuonLayout":
        """Read logical-matrix dimensions from a model configuration.

        Args:
            config: Configuration containing MLA and expert dimensions.

        Returns:
            A validated matrix layout.
        """
        return cls(**{item.name: getattr(config, item.name, item.default) for item in fields(cls)})

    @staticmethod
    def _check_shape(name: str, tensor: torch.Tensor, shape: tuple[int, ...]) -> None:
        """Require a complete logical matrix, including its non-split axis."""
        if tuple(tensor.shape) != shape:
            raise ValueError(f"Unexpected Xing4 Muon shape for {name}: {tuple(tensor.shape)}; expected {shape}")

    @cached_property
    def _row_sections(self) -> dict[str, tuple[int, ...]]:
        """Map parameter suffixes to contiguous logical row sizes."""
        shared_rows = self.n_shared_experts * self.moe_intermediate_size
        return {
            "self_attn.linear_qkv.weight": (self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim),
            "mlp.linear_fc1.weight": (self.intermediate_size, self.intermediate_size),
            "shared_experts.linear_fc1.weight": (shared_rows, shared_rows),
        }

    def reshape_fn(self, name: str, tensor: torch.Tensor) -> list[torch.Tensor]:
        """Split fused rows or expert channels into storage-sharing views.

        Args:
            name: Full model parameter name.
            tensor: Complete logical gradient or local expert batch.

        Returns:
            Matrix views for independent Muon updates.
        """
        if name.endswith(".mlp.experts.gate_up_proj"):
            if tensor.shape[0] == 0:
                raise ValueError(f"Xing4 Muon requires a nonempty expert batch for {name}")
            self._check_shape(name, tensor, (tensor.shape[0], self.hidden_size, 2 * self.moe_intermediate_size))
            return list(tensor.split(self.moe_intermediate_size, dim=-1))
        rows = self._row_sections.get(".".join(name.split(".")[-3:]))
        if rows is None:
            return [tensor]
        self._check_shape(name, tensor, (sum(rows), self.hidden_size))
        return list(tensor.split(rows, dim=0))

    def ns_transform_fn(self, name: str, tensor: torch.Tensor) -> NSInputTransform | None:
        """Gather channels across heads and restore native rows after NS.

        Q splits no-RoPE/RoPE; KV splits key/value. Expert gate/up occupy the
        last axis and are handled by reshape_fn instead.

        Args:
            name: Full model parameter name.
            tensor: Complete MLA projection gradient.

        Returns:
            Matrix groups and their inverse mapping, or None for other weights.
        """
        if name.endswith(".self_attn.q_b_proj.weight"):
            first, second, columns = self.qk_nope_head_dim, self.qk_rope_head_dim, self.q_lora_rank
        elif name.endswith(".self_attn.kv_b_proj.weight"):
            first, second, columns = self.qk_nope_head_dim, self.v_head_dim, self.kv_lora_rank
        else:
            return None
        heads = self.num_attention_heads
        width = first + second
        self._check_shape(name, tensor, (heads * width, columns))
        grouped = tensor.view(heads, width, columns)
        pieces = [grouped[:, :first].contiguous().view(heads * first, columns),
                  grouped[:, first:].contiguous().view(heads * second, columns)]

        def restore(updates: list[torch.Tensor], output: torch.Tensor) -> None:
            """Scatter complete group updates without changing their per-matrix scale."""
            destination = output.view(heads, width, columns)
            destination[:, :first].copy_(updates[0].view(heads, first, columns))
            destination[:, first:].copy_(updates[1].view(heads, second, columns))

        return NSInputTransform(tensors=pieces, restore=restore)

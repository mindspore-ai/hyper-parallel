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
"""Logical MLA/MLP matrices for the framework Muon transform interfaces.

The reversible head grouping follows townwish4git's JTMuonLayout in
hf1s/hyper-parallel#303 (b743f77). Xing40's expert packing is transposed:
gate/up occupy the last axis. Scaling remains the framework's per-matrix
policy, as in MindFormers PyNative, without the later grouped-scale change.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields
from typing import Any

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.core.optimizer.muon import NSInputTransform

_MISSING_ATTR = object()


@dataclass(frozen=True)
class Xing40MuonLayout:
    """Describe complete logical matrices in Xing40's native parameter packing.

    Supply both bound methods to Muon. Contiguous fused sections share storage;
    periodic attention rows are gathered and restored through NSInputTransform.
    Dimensions check that tensor-parallel fragments are not silently mistaken
    for complete logical matrices. Independent expert batches may stay local.
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
    n_shared_experts: int = field(default=0, metadata={"minimum": 0})

    def __post_init__(self) -> None:
        """Validate matrix dimensions before constructing the optimizer."""
        for item in fields(self):
            value = getattr(self, item.name)
            minimum = item.metadata.get("minimum", 1)
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"Xing40 Muon layout requires integer {item.name} >= {minimum}")

    @classmethod
    def from_config(cls, config: Any) -> "Xing40MuonLayout":
        """Bind matrix dimensions from the model configuration.

        A field missing from ``config`` falls back to its own dataclass default, if any;
        a field present but holding a non-integer value (including ``None``) is passed
        through so ``__post_init__`` reports it.

        Args:
            config: Xing40 configuration containing MLA and feed-forward sizes.

        Returns:
            One layout shared by the reshape and reversible-transform callbacks.
        """
        values = {}
        for item in fields(cls):
            raw = getattr(config, item.name, _MISSING_ATTR)
            values[item.name] = item.default if raw is _MISSING_ATTR and item.default is not MISSING else raw
        return cls(**values)

    @staticmethod
    def _check_shape(name: str, tensor: torch.Tensor, shape: tuple[int, ...]) -> None:
        """Require a complete logical matrix, including its non-split axis."""
        if tuple(tensor.shape) != shape:
            raise ValueError(f"Unexpected Xing40 Muon shape for {name}: {tuple(tensor.shape)}; expected {shape}")

    def _row_sections(self) -> dict[str, tuple[int, ...]]:
        """Map a parameter's trailing name segments to its independent row sections.

        Returns:
            Row sizes summing to the complete parameter's row count, keyed by the
            parameter name's last three dot-separated segments.
        """
        shared_rows = self.n_shared_experts * self.moe_intermediate_size
        return {
            "self_attn.linear_qkv.weight": (self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim),
            "mlp.linear_fc1.weight": (self.intermediate_size, self.intermediate_size),
            "shared_experts.linear_fc1.weight": (shared_rows, shared_rows),
        }

    def reshape_fn(self, name: str, tensor: torch.Tensor) -> list[torch.Tensor]:
        """Expose independent fused sections as views of the native storage.

        The expert batch is checked separately: it splits its last axis, not rows, and
        keeps the leading expert axis whole. Every other recognized suffix splits full
        rows and is looked up in :meth:`_row_sections`; an unrecognized suffix keeps the
        tensor as one logical matrix.

        Args:
            name: Fully qualified model parameter name.
            tensor: Complete parameter-shaped Newton-Schulz input.

        Returns:
            Storage-sharing logical matrices, or the unchanged matrix.
        """
        if name.endswith(".mlp.experts.gate_up_proj"):
            if tensor.shape[0] == 0:
                raise ValueError(f"Xing40 Muon requires a nonempty expert batch for {name}")
            self._check_shape(name, tensor, (tensor.shape[0], self.hidden_size, 2 * self.moe_intermediate_size))
            return list(tensor.split(self.moe_intermediate_size, dim=-1))
        rows = self._row_sections().get(".".join(name.split(".")[-3:]))
        if rows is None:
            return [tensor]
        self._check_shape(name, tensor, (sum(rows), self.hidden_size))
        return list(tensor.split(rows, dim=0))

    def ns_transform_fn(self, name: str, tensor: torch.Tensor) -> NSInputTransform | None:
        """Group corresponding channels across heads and restore native rows.

        Args:
            name: Fully qualified model parameter name.
            tensor: Complete parameter-shaped Newton-Schulz input.

        Returns:
            The two reversible whole-head-group matrices, or None to use reshape_fn.
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

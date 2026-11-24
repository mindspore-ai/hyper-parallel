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
"""NoPE MLA with a required sigmoid output gate."""

from __future__ import annotations

from typing import Any

# The high-performance attention modules use native Torch.
import torch
from torch import nn

from hyper_parallel.components.modules.attention.multi_latent_attention.multi_latent_attention import (
    MultiLatentAttentionModule,
)


class GatedMultiLatentAttentionModule(MultiLatentAttentionModule):
    """Define NoPE attention and its mandatory output gate in one math module.

    The shared key features retain the source projection/checkpoint dimensions;
    NoPE disables their rotation. The family also retains the declared source result protocol.
    """

    source_extra_children = ("g_proj",)

    def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        """Consume the selected source's positional arguments inside the complete family."""
        if self.source_contract == "tensor":
            names = ("attention_mask", "position_ids", "past_key_values")
            if len(args) > len(names):
                raise TypeError("NoPE source attention accepts at most three arguments after hidden_states")
            for name, value in zip(names, args):
                if name in kwargs:
                    raise TypeError(f"Multiple values for attention argument {name}")
                kwargs[name] = value
            kwargs.pop("position_ids", None)
            return super().forward(hidden_states, **kwargs)
        return super().forward(hidden_states, *args, **kwargs)

    def __init__(self, *, module: nn.Module | None = None, config: Any = None,
                 fuse_latent_qkv: bool = True, source_contract: str = "tuple") -> None:
        """Construct mandatory-gate NoPE MLA with the ordinary MLA parameter layout.

        Args:
            module: Optional NoPE source with an enabled plain Linear output gate.
            config: Direct construction dimensions, exclusive with module; NoPE is intrinsic to this class.
        """
        if source_contract not in ("tuple", "tensor"):
            raise ValueError("Gated MLA source_contract must be tuple or tensor")
        self.source_contract = source_contract
        if module is None:
            super().__init__(config=config, fuse_latent_qkv=fuse_latent_qkv, key_expansion_layout="bnsd")
            self.use_nope = True
            self.g_proj = nn.Linear(
                config.hidden_size, self.num_heads * self.v_head_dim,
                bias=getattr(config, "attention_bias", False), dtype=getattr(config, "dtype", None),
            )
            return
        if not getattr(module, "use_nope", False):
            raise ValueError("GatedMultiLatentAttention requires a NoPE source module")
        if not isinstance(module, GatedMultiLatentAttentionModule) and not getattr(module, "use_output_gate", False):
            raise ValueError("GatedMultiLatentAttention requires an enabled output gate")
        if type(module.g_proj) is not nn.Linear:  # pylint: disable=unidiomatic-typecheck
            raise ValueError("Gated MLA requires a plain Linear output gate")
        super().__init__(module=module, config=config, fuse_latent_qkv=fuse_latent_qkv,
                         key_expansion_layout="bnsd")
        self.use_nope = True
        self.g_proj = module.g_proj

    @staticmethod

    def apply_nope(tensor: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
                   *, interleaved: bool = False) -> torch.Tensor:
        """Keep Q/shared-K features unchanged through the CP rotary callback.

        Args:
            tensor: Query or shared-key features in the runtime's BSND layout.
            position_embeddings: Ignored by NoPE attention.
            interleaved: Ignored because no positional rotation is applied.

        Returns:
            The original tensor, without a copy or a positional transform.
        """
        del position_embeddings, interleaved
        return tensor

    def apply_position(
        self, tensor: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        *, interleaved: bool | None = None,
    ) -> torch.Tensor:
        """Keep query/shared-key features unchanged because this algorithm uses NoPE.

        Args:
            tensor: Query or shared-key features, in BSND layout.
            position_embeddings: Ignored even when supplied by a generic model caller.
            interleaved: Ignored because no rotation or feature permutation is performed.

        Returns:
            The original tensor, preserving its identity and view layout.
        """
        del position_embeddings, interleaved
        return tensor

    @staticmethod

    def prepare_positions(position_embeddings: Any) -> None:
        """Discard frequencies and their layout because this algorithm is NoPE.

        Args:
            position_embeddings: Ignored positional input from the model caller.
        """
        del position_embeddings

    def project_gate(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute the mandatory token-local output gate through its managed Linear.

        Args:
            hidden_states: Original local input in BSH layout.
        """
        return self.g_proj(hidden_states).sigmoid()

    def reset_parameters(self) -> None:
        """Initialize direct-config projections and gate, preserving source initialization."""
        if self._config_initialization:
            super().reset_parameters()
            self.g_proj.reset_parameters()

    def project_output(self, attention_output: torch.Tensor, hidden_states: torch.Tensor,
                       output_state: torch.Tensor | None) -> torch.Tensor:
        """Apply the source gate before Wo, without an additional output Norm.

        Args:
            attention_output: Token-local attention output in BSND layout.
            hidden_states: Original token-local input for the gate projection.
            output_state: Optional gate already computed by CP scheduling; local execution computes it here.

        Returns:
            Gated and projected hidden states.
        """
        gate = self.project_gate(hidden_states) if output_state is None else output_state
        attention_output = attention_output.reshape(*hidden_states.shape[:2], -1)
        return self.o_proj(attention_output * gate)

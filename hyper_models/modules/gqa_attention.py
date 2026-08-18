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
"""Grouped-query attention module using the Hyper high-performance layout."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

# This package provides PyTorch-specific high-performance modules.
# pylint: disable=forbidden-backend-import
import torch  # pylint: disable=forbidden-backend-import
from torch import nn
from transformers.core_model_loading import WeightConverter

from hyper_models.components.checkpoint import InterleaveGateQKV, InterleaveQKV
from hyper_models.components.model_transform import module_replacement
from hyper_models.ops import (
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_interleave,
    npu_fusion_attention_forward,
)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of the last dimension as Transformers does."""
    first = x[..., : x.shape[-1] // 2]
    second = x[..., x.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)


@module_replacement
class GQAAttention(nn.Module):
    """Transformers-compatible GQA using a grouped ``linear_qkv`` layout.

    The source module may expose a fused ``qkv_proj`` or separate
    ``q_proj``/``k_proj``/``v_proj`` layers. Its checkpoint layout is converted
    to the per-KV-head grouping consumed by the original high-performance PR.
    Construction only creates the target structure; ``make_transforms``
    declares the checkpoint conversion.
    """

    def __init__(
        self,
        *,
        module: nn.Module,
        module_fqn: str = "",
        context: Mapping[str, Any] | None = None,
        attention_interface: Callable[..., tuple[torch.Tensor, torch.Tensor | None]] = (
            npu_fusion_attention_forward
        ),
    ) -> None:
        """Build the high-performance module from a Transformers attention module."""
        super().__init__()
        del module_fqn, context
        self.attention_interface = attention_interface
        config = module.config
        self.config = config
        self.layer_idx = getattr(module, "layer_idx", getattr(module, "layer_number", None))

        self.num_heads = getattr(module, "num_attention_heads", config.num_attention_heads)
        self.num_key_value_heads = getattr(
            module,
            "num_key_value_heads",
            config.num_key_value_heads,
        )
        if self.num_heads % self.num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.qk_head_dim = getattr(
            module,
            "head_dim",
            getattr(config, "head_dim", config.hidden_size // self.num_heads),
        )
        self.v_head_dim = getattr(module, "v_head_dim", None) or getattr(
            config,
            "v_head_dim",
            None,
        ) or self.qk_head_dim
        self.qkv_split_sizes = (
            self.num_key_value_groups * self.qk_head_dim,
            self.qk_head_dim,
            self.v_head_dim,
        )
        self.qkv_group_width = sum(self.qkv_split_sizes)
        self.scaling = getattr(module, "scaling", self.qk_head_dim**-0.5)
        self.attention_dropout = getattr(module, "attention_dropout", 0.0)
        if isinstance(self.attention_dropout, nn.Dropout):
            self.attention_dropout = self.attention_dropout.p
        self.is_causal = getattr(module, "is_causal", True)
        self.sliding_window = getattr(
            module, "sliding_window", getattr(config, "sliding_window", None)
        )

        self.rotary_interleaved = getattr(
            module,
            "rotary_interleaved",
            getattr(config, "rope_interleave", getattr(config, "rope_interleaved", False)),
        )
        if getattr(module, "param_sink_number", 0) > 0:
            raise ValueError(
                "GQAAttention only implements standard GQA; parameter-sink attention "
                "must use the sink_attention high-performance interface."
            )
        self.attn_groupnorm = bool(getattr(module, "attn_groupnorm", False))
        self.attn_elementwise_gate = bool(getattr(module, "attn_elementwise_gate", False))
        if self.attn_groupnorm or self.attn_elementwise_gate:
            raise ValueError(
                "GQAAttention only implements standard GQA; group normalization and "
                "elementwise attention gates require a dedicated replacement."
            )

        self._source_is_fused = hasattr(module, "qkv_proj")
        if self._source_is_fused:
            source_qkv = module.qkv_proj
            if not isinstance(source_qkv, nn.Linear):
                raise TypeError("qkv_proj must be torch.nn.Linear")
            source_projection_names = ("qkv_proj",)
            expected_projection_size = (
                self.num_heads * self.qk_head_dim
                + self.num_key_value_heads * self.qk_head_dim
                + self.num_key_value_heads * self.v_head_dim
            )
            if source_qkv.out_features != expected_projection_size:
                raise ValueError(
                    "qkv_proj output size is incompatible with the configured GQA dimensions"
                )
        else:
            required = ("q_proj", "k_proj", "v_proj")
            if any(not hasattr(module, name) for name in required):
                raise TypeError(
                    "GQAAttention requires qkv_proj or q_proj/k_proj/v_proj on the source module"
                )
            source_qkv = module.q_proj
            source_projection_names = required
            source_projections = tuple(getattr(module, name) for name in required)
            if any(
                not isinstance(projection, nn.Linear)
                for projection in source_projections
            ):
                raise TypeError("q_proj, k_proj, and v_proj must be torch.nn.Linear")
            expected_projection_sizes = (
                self.num_heads * self.qk_head_dim,
                self.num_key_value_heads * self.qk_head_dim,
                self.num_key_value_heads * self.v_head_dim,
            )
            actual_projection_sizes = tuple(
                projection.out_features for projection in source_projections
            )
            if actual_projection_sizes != expected_projection_sizes:
                raise ValueError(
                    "Q/K/V projection sizes are incompatible with the configured GQA dimensions"
                )
            biases = tuple(projection.bias for projection in source_projections)
            can_fuse = (
                len({projection.in_features for projection in source_projections}) == 1
                and len(
                    {projection.weight.requires_grad for projection in source_projections}
                )
                == 1
                and (
                    all(bias is None for bias in biases)
                    or all(bias is not None for bias in biases)
                )
                and (
                    biases[0] is None
                    or len({bias.requires_grad for bias in biases}) == 1
                )
            )
            if not can_fuse:
                raise ValueError(
                    "Q, K, and V projections cannot be represented by one fused projection"
                )
        self._source_projection_names = source_projection_names

        projection_size = self.num_key_value_heads * self.qkv_group_width
        self.linear_qkv = nn.Linear(
            source_qkv.in_features,
            projection_size,
            bias=source_qkv.bias is not None,
            device=source_qkv.weight.device,
            dtype=source_qkv.weight.dtype,
        )
        self.linear_qkv.weight.requires_grad_(source_qkv.weight.requires_grad)
        if self.linear_qkv.bias is not None:
            self.linear_qkv.bias.requires_grad_(source_qkv.bias.requires_grad)

        if not hasattr(module, "o_proj") or not isinstance(module.o_proj, nn.Linear):
            raise TypeError("GQAAttention requires o_proj on the source module")
        if module.o_proj.in_features != self.num_heads * self.v_head_dim:
            raise ValueError(
                "o_proj input size is incompatible with the configured GQA dimensions"
            )
        self.o_proj = module.o_proj
        self.q_norm = getattr(module, "q_norm", None)
        self.k_norm = getattr(module, "k_norm", None)

        self.train(module.training)

    def make_transforms(self) -> list[WeightConverter]:
        """Describe reversible source-checkpoint to high-performance conversion."""
        if self._source_is_fused:
            qkv_sources = "qkv_proj.weight"
            bias_sources: str | list[str] = "qkv_proj.bias"
        else:
            qkv_sources = [f"{name}.weight" for name in self._source_projection_names]
            bias_sources = [f"{name}.bias" for name in self._source_projection_names]

        transforms: list[WeightConverter] = [
            WeightConverter(
                source_patterns=qkv_sources,
                target_patterns="linear_qkv.weight",
                operations=[
                    InterleaveQKV(
                        self.num_key_value_heads,
                        self.num_key_value_groups,
                        self.qk_head_dim,
                        self.v_head_dim,
                        self._source_is_fused,
                    )
                ],
            )
        ]
        if self.linear_qkv.bias is not None:
            transforms.append(
                WeightConverter(
                    source_patterns=bias_sources,
                    target_patterns="linear_qkv.bias",
                    operations=[
                        InterleaveQKV(
                            self.num_key_value_heads,
                            self.num_key_value_groups,
                            self.qk_head_dim,
                            self.v_head_dim,
                            self._source_is_fused,
                        )
                    ],
                )
            )
        return transforms

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run GQA with the same external contract as Transformers attention."""
        input_shape = hidden_states.shape[:-1]
        qkv_states = self.linear_qkv(hidden_states).view(
            *input_shape,
            self.num_key_value_heads,
            self.qkv_group_width,
        )
        query_states, key_states, value_states = torch.split(
            qkv_states, self.qkv_split_sizes, dim=-1
        )
        query_states = query_states.reshape(
            *input_shape, self.num_heads, self.qk_head_dim
        )

        if self.q_norm is not None:
            query_states = self.q_norm(query_states)
        if self.k_norm is not None:
            key_states = self.k_norm(key_states)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            if self.rotary_interleaved:
                query_states, key_states = apply_rotary_pos_emb_interleave(
                    query_states, key_states, cos, sin, unsqueeze_dim=2
                )
            elif cos.shape[-1] < self.qk_head_dim:
                rotary_dim = cos.shape[-1]
                query_rot, query_pass = (
                    query_states[..., :rotary_dim],
                    query_states[..., rotary_dim:],
                )
                key_rot, key_pass = (
                    key_states[..., :rotary_dim],
                    key_states[..., rotary_dim:],
                )
                cos = cos.unsqueeze(2)
                sin = sin.unsqueeze(2)
                query_states = torch.cat(
                    (query_rot * cos + rotate_half(query_rot) * sin, query_pass),
                    dim=-1,
                )
                key_states = torch.cat(
                    (key_rot * cos + rotate_half(key_rot) * sin, key_pass),
                    dim=-1,
                )
            else:
                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, unsqueeze_dim=2
                )
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        attn_output, attn_weights = self.attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


@module_replacement
class GatedGQAAttention(nn.Module):
    """GQA with a sigmoid output gate derived from each query head.

    The source ``q_proj`` is expected to store ``[query, gate]`` rows for each
    query head. This is the gated-query layout used by Qwen3.5 full attention.
    """

    def __init__(
        self,
        *,
        module: nn.Module,
        module_fqn: str = "",
        context: Mapping[str, Any] | None = None,
        attention_interface: Callable[..., tuple[torch.Tensor, torch.Tensor | None]] = (
            npu_fusion_attention_forward
        ),
    ) -> None:
        """Build gated GQA from a source with separate Q/gate, K, and V projections."""
        super().__init__()
        del module_fqn, context
        if hasattr(module, "qkv_proj"):
            raise ValueError(
                "GatedGQAAttention requires separate q_proj/k_proj/v_proj projections"
            )
        required = ("q_proj", "k_proj", "v_proj", "q_norm", "k_norm", "o_proj")
        if any(not hasattr(module, name) for name in required):
            raise TypeError(
                "GatedGQAAttention requires q_proj, k_proj, v_proj, q_norm, "
                "k_norm, and o_proj on the source module"
            )

        self.attention_interface = attention_interface
        self.config = module.config
        self.layer_idx = getattr(module, "layer_idx", None)
        self.num_heads = getattr(
            module, "num_attention_heads", self.config.num_attention_heads
        )
        self.num_key_value_heads = getattr(
            module, "num_key_value_heads", self.config.num_key_value_heads
        )
        if self.num_heads % self.num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.qk_head_dim = getattr(
            module,
            "head_dim",
            getattr(
                self.config,
                "head_dim",
                self.config.hidden_size // self.num_heads,
            ),
        )
        self.v_head_dim = getattr(module, "v_head_dim", None) or getattr(
            self.config, "v_head_dim", None
        ) or self.qk_head_dim
        self.qkv_split_sizes = (
            self.num_key_value_groups * self.qk_head_dim,
            self.qk_head_dim,
            self.v_head_dim,
        )
        self.qkv_group_width = sum(self.qkv_split_sizes)
        self.gated_qkv_group_width = (
            self.qkv_group_width + self.num_key_value_groups * self.qk_head_dim
        )
        self.scaling = getattr(module, "scaling", self.qk_head_dim**-0.5)
        self.attention_dropout = getattr(module, "attention_dropout", 0.0)
        if isinstance(self.attention_dropout, nn.Dropout):
            self.attention_dropout = self.attention_dropout.p
        self.is_causal = getattr(module, "is_causal", True)
        self.sliding_window = getattr(
            module,
            "sliding_window",
            getattr(self.config, "sliding_window", None),
        )
        self.rotary_interleaved = getattr(
            module,
            "rotary_interleaved",
            getattr(
                self.config,
                "rope_interleave",
                getattr(self.config, "rope_interleaved", False),
            ),
        )

        source_projections = (module.q_proj, module.k_proj, module.v_proj)
        if any(
            not isinstance(projection, nn.Linear)
            for projection in source_projections
        ):
            raise TypeError("q_proj, k_proj, and v_proj must be torch.nn.Linear")
        expected_query_gate_size = 2 * self.num_heads * self.qk_head_dim
        expected_projection_sizes = (
            expected_query_gate_size,
            self.num_key_value_heads * self.qk_head_dim,
            self.num_key_value_heads * self.v_head_dim,
        )
        actual_projection_sizes = tuple(
            projection.out_features for projection in source_projections
        )
        if actual_projection_sizes != expected_projection_sizes:
            raise ValueError(
                "Q/gate, K, and V projection sizes are incompatible with the "
                "configured gated GQA dimensions"
            )
        biases = tuple(projection.bias for projection in source_projections)
        can_fuse = (
            len({projection.in_features for projection in source_projections}) == 1
            and len(
                {projection.weight.requires_grad for projection in source_projections}
            )
            == 1
            and (
                all(bias is None for bias in biases)
                or all(bias is not None for bias in biases)
            )
            and (
                biases[0] is None
                or len({bias.requires_grad for bias in biases}) == 1
            )
        )
        if not can_fuse:
            raise ValueError(
                "Q/gate, K, and V projections cannot be represented by one fused projection"
            )
        gate_size = self.num_heads * self.qk_head_dim
        attention_output_size = self.num_heads * self.v_head_dim
        if gate_size != attention_output_size:
            raise ValueError(
                "GatedGQAAttention requires the gate width to match the attention output width"
            )
        if not isinstance(module.o_proj, nn.Linear):
            raise TypeError("o_proj must be torch.nn.Linear")
        if module.o_proj.in_features != attention_output_size:
            raise ValueError(
                "o_proj input size is incompatible with the configured gated GQA dimensions"
            )

        self.linear_qkv = nn.Linear(
            module.q_proj.in_features,
            self.num_key_value_heads * self.gated_qkv_group_width,
            bias=module.q_proj.bias is not None,
            device=module.q_proj.weight.device,
            dtype=module.q_proj.weight.dtype,
        )
        self.linear_qkv.weight.requires_grad_(module.q_proj.weight.requires_grad)
        if self.linear_qkv.bias is not None:
            self.linear_qkv.bias.requires_grad_(module.q_proj.bias.requires_grad)
        self.q_norm = module.q_norm
        self.k_norm = module.k_norm
        self.o_proj = module.o_proj
        self.train(module.training)

    def make_transforms(self) -> list[WeightConverter]:
        """Describe reversible gated-query and grouped-QKV conversion."""
        source_weights = ["q_proj.weight", "k_proj.weight", "v_proj.weight"]
        operation = InterleaveGateQKV(
            self.num_key_value_heads,
            self.num_key_value_groups,
            self.qk_head_dim,
            self.v_head_dim,
        )
        transforms = [
            WeightConverter(
                source_patterns=source_weights,
                target_patterns="linear_qkv.weight",
                operations=[operation],
            )
        ]
        if self.linear_qkv.bias is not None:
            transforms.append(
                WeightConverter(
                    source_patterns=["q_proj.bias", "k_proj.bias", "v_proj.bias"],
                    target_patterns="linear_qkv.bias",
                    operations=[
                        InterleaveGateQKV(
                            self.num_key_value_heads,
                            self.num_key_value_groups,
                            self.qk_head_dim,
                            self.v_head_dim,
                        )
                    ],
                )
            )
        return transforms

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run gated GQA with the same external contract as Transformers attention."""
        input_shape = hidden_states.shape[:-1]
        qkv_states = self.linear_qkv(hidden_states).view(
            *input_shape,
            self.num_key_value_heads,
            self.gated_qkv_group_width,
        )
        query_states, key_states, value_states, gate = torch.split(
            qkv_states,
            (*self.qkv_split_sizes, self.num_key_value_groups * self.qk_head_dim),
            dim=-1,
        )
        query_states = query_states.reshape(
            *input_shape, self.num_heads, self.qk_head_dim
        )
        gate = gate.reshape(*input_shape, -1)

        if self.q_norm is not None:
            query_states = self.q_norm(query_states)
        if self.k_norm is not None:
            key_states = self.k_norm(key_states)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            if self.rotary_interleaved:
                query_states, key_states = apply_rotary_pos_emb_interleave(
                    query_states, key_states, cos, sin, unsqueeze_dim=2
                )
            elif cos.shape[-1] < self.qk_head_dim:
                rotary_dim = cos.shape[-1]
                query_rot, query_pass = (
                    query_states[..., :rotary_dim],
                    query_states[..., rotary_dim:],
                )
                key_rot, key_pass = (
                    key_states[..., :rotary_dim],
                    key_states[..., rotary_dim:],
                )
                cos = cos.unsqueeze(2)
                sin = sin.unsqueeze(2)
                query_states = torch.cat(
                    (query_rot * cos + rotate_half(query_rot) * sin, query_pass),
                    dim=-1,
                )
                key_states = torch.cat(
                    (key_rot * cos + rotate_half(key_rot) * sin, key_pass),
                    dim=-1,
                )
            else:
                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, unsqueeze_dim=2
                )
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        attn_output, attn_weights = self.attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

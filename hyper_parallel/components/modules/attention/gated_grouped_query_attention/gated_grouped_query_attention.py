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
"""Gated GQA: per-query-head gate weights and output math, without CP transport."""

from __future__ import annotations

import torch
from torch import nn
from hyper_parallel.components.checkpoint import InterleaveGateQKV
from hyper_parallel.components.checkpoint.weight_conversion import WeightConverter
from ..grouped_query_attention.grouped_query_attention import GroupedQueryAttentionModule, _projections_can_fuse


class GatedGroupedQueryAttentionModule(GroupedQueryAttentionModule):
    """GQA with a sigmoid output gate derived from each query head.

    The source ``q_proj`` is expected to store ``[query, gate]`` rows for each
    query head. This is the gated-query layout used by Qwen3.5 full attention.
    """

    def _initialize_dimensions(self, module: nn.Module) -> None:
        """Initialize gated grouped-QKV dimensions and attention options."""
        config = self.config
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        if self.num_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.qk_head_dim = getattr(
            module, "head_dim", getattr(config, "head_dim", config.hidden_size // self.num_heads)
        )
        self.v_head_dim = (
            getattr(module, "v_head_dim", None) or getattr(config, "v_head_dim", None) or self.qk_head_dim
        )
        self.qkv_split_sizes = (
            self.num_key_value_groups * self.qk_head_dim,
            self.qk_head_dim,
            self.v_head_dim,
        )
        self.qkv_group_width = sum(self.qkv_split_sizes)
        self.gated_qkv_group_width = self.qkv_group_width + self.num_key_value_groups * self.qk_head_dim
        self.scaling = getattr(module, "scaling", self.qk_head_dim**-0.5)
        self.attention_dropout = getattr(module, "attention_dropout", 0.0)
        if isinstance(self.attention_dropout, nn.Dropout):
            self.attention_dropout = self.attention_dropout.p
        self.is_causal = getattr(module, "is_causal", True)
        self.sliding_window = getattr(module, "sliding_window", getattr(config, "sliding_window", None))
        self.rotary_interleaved = getattr(
            module,
            "rotary_interleaved",
            getattr(config, "rope_interleave", getattr(config, "rope_interleaved", False)),
        )

    def _validate_source_projections(self, module: nn.Module) -> None:
        """Validate separate gated-query, key, value, and output projections."""
        projections = (module.q_proj, module.k_proj, module.v_proj)
        if any(not isinstance(projection, nn.Linear) for projection in projections):
            raise TypeError("q_proj, k_proj, and v_proj must be torch.nn.Linear")
        expected_sizes = (
            2 * self.num_heads * self.qk_head_dim,
            self.num_key_value_heads * self.qk_head_dim,
            self.num_key_value_heads * self.v_head_dim,
        )
        if tuple(projection.out_features for projection in projections) != expected_sizes:
            raise ValueError(
                "Q/gate, K, and V projection sizes are incompatible with the configured gated GQA dimensions"
            )
        if not _projections_can_fuse(projections):
            raise ValueError("Q/gate, K, and V projections cannot be represented by one fused projection")
        attention_output_size = self.num_heads * self.v_head_dim
        if self.num_heads * self.qk_head_dim != attention_output_size:
            raise ValueError("GatedGroupedQueryAttention requires the gate width to match the attention output width")
        if not isinstance(module.o_proj, nn.Linear):
            raise TypeError("o_proj must be torch.nn.Linear")
        if module.o_proj.in_features != attention_output_size:
            raise ValueError("o_proj input size is incompatible with the configured gated GQA dimensions")

    def _initialize_projection(self, module: nn.Module) -> None:
        """Create the fused gated-QKV projection and attach source output layers."""
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
        self.q_norm = getattr(module, "q_norm", None)
        self.k_norm = getattr(module, "k_norm", None)
        self.o_proj = module.o_proj

    def _initialize_attention_options(self, module: nn.Module) -> None:
        """Gated options are initialized with dimensions; the gate is mandatory."""
        del module

    def _source_qkv(self, module: nn.Module) -> tuple:
        """Validate the per-query-head source gate layout before fusion."""
        if any(not hasattr(module, name) for name in ("q_proj", "k_proj", "v_proj", "o_proj")):
            raise TypeError("Gated GQA requires separate Q/gate, K, V and output projections")
        self._validate_source_projections(module)
        self._source_is_fused = False
        return module.q_proj, ("q_proj", "k_proj", "v_proj")

    def _initialize_projections(self, module: nn.Module, source_qkv: nn.Linear) -> None:
        del source_qkv
        self._initialize_projection(module)

    def make_transforms(self) -> list[WeightConverter]:
        """Describe reversible gated-query and grouped-QKV conversion."""
        source_weights = ["q_proj.weight", "k_proj.weight", "v_proj.weight"]
        operation = InterleaveGateQKV(
            self.config.num_key_value_heads,
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
                            self.config.num_key_value_heads,
                            self.num_key_value_groups,
                            self.qk_head_dim,
                            self.v_head_dim,
                        )
                    ],
                )
            )
        return transforms

    def split_projection(self, hidden_states: torch.Tensor) -> tuple:
        """Keep the fused per-KV-head Q/K/V/gate checkpoint row ordering."""
        shape = hidden_states.shape[:-1]
        projected = self.linear_qkv(hidden_states).view(*shape, self.num_key_value_heads, self.gated_qkv_group_width)
        query, key, value, gate = projected.split(
            (*self.qkv_split_sizes, self.num_key_value_groups * self.qk_head_dim), dim=-1)
        return query.reshape(*shape, self.num_heads, self.qk_head_dim), key, value, gate.reshape(*shape, -1)

    def project_output(self, attention_output: torch.Tensor, output_state: torch.Tensor) -> torch.Tensor:
        """Apply the mandatory gate after CP restores the local token/head layout."""
        return self.o_proj(attention_output.flatten(-2).contiguous() * torch.sigmoid(output_state))

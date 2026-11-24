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

from typing import Any

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

    def _initialize_dimensions(self, module: nn.Module | None) -> None:
        """Extend ordinary GQA dimensions by one gate value per query channel."""
        super()._initialize_dimensions(module)
        if self.qk_head_dim != self.v_head_dim:
            raise ValueError("GatedGroupedQueryAttention requires the gate width to match the attention output width")
        self.gated_qkv_group_width = self.qkv_group_width + self.num_key_value_groups * self.qk_head_dim

    def _initialize_config_projections(self, config: Any, bias: bool, dtype: torch.dtype | None) -> None:
        """Allocate native [grouped Q, K, V, grouped gate] rows directly."""
        if self.fuse_qkv:
            self.linear_qkv = nn.Linear(
                config.hidden_size, self.num_key_value_heads * self.gated_qkv_group_width, bias=bias, dtype=dtype)
        else:
            self.q_proj = nn.Linear(config.hidden_size, 2 * self.num_heads * self.qk_head_dim, bias=bias, dtype=dtype)
            self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.qk_head_dim,
                                    bias=bias, dtype=dtype)
            self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.v_head_dim,
                                    bias=bias, dtype=dtype)

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
        if self.fuse_qkv and not _projections_can_fuse(projections):
            raise ValueError("Q/gate, K, and V projections cannot be represented by one fused projection")
        attention_output_size = self.num_heads * self.v_head_dim
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
        self.linear_qkv.training = module.q_proj.training
        self.linear_qkv.weight.requires_grad_(module.q_proj.weight.requires_grad)
        if self.linear_qkv.bias is not None:
            self.linear_qkv.bias.requires_grad_(module.q_proj.bias.requires_grad)
        self.q_norm = getattr(module, "q_norm", None)
        self.k_norm = getattr(module, "k_norm", None)
        self.o_proj = module.o_proj

    def _initialize_attention_options(self, module: Any) -> None:
        """Copy attention options; this representation always computes the query gate."""
        config = self.config
        self.scaling = getattr(module, "scaling", self.qk_head_dim**-0.5)
        self.attention_dropout = getattr(module, "attention_dropout", 0.0)
        if isinstance(self.attention_dropout, nn.Dropout):
            self.attention_dropout = self.attention_dropout.p
        self.is_causal = getattr(module, "is_causal", True)
        self.sliding_window = getattr(module, "sliding_window", None)
        if not isinstance(module, nn.Module):
            self.sliding_window = getattr(config, "sliding_window", None)
        self.rotary_interleaved = getattr(
            module, "rotary_interleaved",
            getattr(config, "rope_interleave", getattr(config, "rope_interleaved", False)),
        )

    def _source_qkv(self, module: nn.Module) -> tuple:
        """Validate the per-query-head source gate layout before fusion."""
        if any(not hasattr(module, name) for name in ("q_proj", "k_proj", "v_proj", "o_proj")):
            raise TypeError("Gated GQA requires separate Q/gate, K, V and output projections")
        self._validate_source_projections(module)
        self._source_is_fused = False
        return module.q_proj, ("q_proj", "k_proj", "v_proj")

    def _initialize_projections(self, module: nn.Module, source_qkv: nn.Linear) -> None:
        del source_qkv
        if self.fuse_qkv:
            self._initialize_projection(module)
        else:
            self._initialize_separate_projections(module)

    def make_transforms(self) -> list[WeightConverter]:
        """Describe reversible gated-query and grouped-QKV conversion."""
        if self._config_initialization or not self.fuse_qkv:
            return []
        suffixes = ("weight", "bias") if self.linear_qkv.bias is not None else ("weight",)
        transforms = []
        for suffix in suffixes:
            transforms.append(WeightConverter(
                source_patterns=[f"{name}.{suffix}" for name in ("q_proj", "k_proj", "v_proj")],
                target_patterns=f"linear_qkv.{suffix}",
                operations=[InterleaveGateQKV(
                    self.config.num_key_value_heads, self.num_key_value_groups, self.qk_head_dim, self.v_head_dim,
                )],
            ))
        return transforms

    def split_projection(self, hidden_states: torch.Tensor) -> tuple:
        """Keep the fused per-KV-head Q/K/V/gate checkpoint row ordering."""
        shape = hidden_states.shape[:-1]
        if not self.fuse_qkv:
            query, gate = self.q_proj(hidden_states).view(*shape, self.num_heads, 2 * self.qk_head_dim).chunk(2, dim=-1)
            key = self.k_proj(hidden_states).view(*shape, self.num_key_value_heads, self.qk_head_dim)
            value = self.v_proj(hidden_states).view(*shape, self.num_key_value_heads, self.v_head_dim)
            return query, key, value, gate.reshape(*shape, -1)
        projected = self.linear_qkv(hidden_states).view(*shape, self.num_key_value_heads, self.gated_qkv_group_width)
        query, key, value, gate = projected.split(
            (*self.qkv_split_sizes, self.num_key_value_groups * self.qk_head_dim), dim=-1)
        return query.reshape(*shape, self.num_heads, self.qk_head_dim), key, value, gate.reshape(*shape, -1)

    def project_query_state(self, hidden_states: torch.Tensor, positions: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep the per-query gate local while CP submits independent Q/K/V transfers."""
        shape = hidden_states.shape[:-1]
        query, gate = self.q_proj(hidden_states).view(*shape, self.num_heads, 2 * self.qk_head_dim).chunk(2, dim=-1)
        return self.normalize_position(query, self.q_norm, positions), gate.reshape(*shape, -1)

    def project_output(self, attention_output: torch.Tensor, output_state: torch.Tensor) -> torch.Tensor:
        """Apply the mandatory gate before the output projection."""
        return self.o_proj(attention_output.flatten(-2).contiguous() * torch.sigmoid(output_state))

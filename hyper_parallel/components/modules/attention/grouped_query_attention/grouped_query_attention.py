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
"""Single-device GQA math and source-checkpoint conversion, without parallel execution."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from hyper_parallel.components.checkpoint import InterleaveQKV
from hyper_parallel.components.checkpoint.weight_conversion import WeightConverter
from hyper_parallel.components.functional import apply_rotary_pos_emb_single, npu_fusion_attention_forward
from hyper_parallel.components.modules.rms_norm import RMSNorm


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of the last dimension as Transformers does."""
    first = x[..., : x.shape[-1] // 2]
    second = x[..., x.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)


def _projections_can_fuse(projections: tuple[nn.Linear, ...]) -> bool:
    """Return whether projections share the layout needed for fusion."""
    biases = tuple(projection.bias for projection in projections)
    return (
        len({projection.in_features for projection in projections}) == 1
        and len({projection.weight.requires_grad for projection in projections}) == 1
        and (all(bias is None for bias in biases) or all(bias is not None for bias in biases))
        and (biases[0] is None or len({bias.requires_grad for bias in biases}) == 1)
    )


def apply_gqa_position(tensor: torch.Tensor, positions: Any, interleaved: bool) -> torch.Tensor:
    """Rotate one BSND branch, retaining the source half-band/interleaved convention."""
    if positions is None:
        return tensor
    cos, sin = positions
    if cos.ndim == 2:
        cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
    rotary_dim = cos.shape[-1]
    if rotary_dim > tensor.shape[-1] or rotary_dim % 2:
        raise ValueError("GQA rotary dimension must be even and fit the head dimension")
    # Keep partial-width half-layout rotation on its established eager path.
    if not interleaved and rotary_dim < tensor.shape[-1]:
        rotary, tail = tensor.split((rotary_dim, tensor.shape[-1] - rotary_dim), dim=-1)
        return torch.cat((rotary * cos.unsqueeze(2) + rotate_half(rotary) * sin.unsqueeze(2), tail), dim=-1)
    if interleaved:
        cos, sin = (part[..., :rotary_dim // 2].repeat_interleave(2, dim=-1) for part in (cos, sin))
    cos, sin = (part.expand(tensor.shape[0], -1, -1).transpose(0, 1) for part in (cos, sin))
    result = apply_rotary_pos_emb_single(tensor.transpose(0, 1), cos, sin,
                                       rotary_interleaved=interleaved, use_fused_rotary_pos_emb=True)
    result = result.transpose(0, 1)
    if interleaved:
        rotary, tail = result.split((rotary_dim, result.shape[-1] - rotary_dim), dim=-1)
        rotary = rotary.unflatten(-1, (-1, 2)).transpose(-1, -2).flatten(-2)
        result = torch.cat((rotary, tail), dim=-1) if tail.shape[-1] else rotary
    return result


@dataclass(frozen=True)
class GQAInputs:
    """Attention operands in BNSD order and metadata for this invocation."""

    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    attention_mask: torch.Tensor | None
    options: dict[str, Any]
    output_state: Any


class GroupedQueryAttentionModule(nn.Module):
    """GQA with fused or independent Q/K/V Parameters fixed at construction.

    The source module may expose a fused ``qkv_proj`` or separate
    ``q_proj``/``k_proj``/``v_proj`` layers. Its checkpoint layout is converted
    to per-KV-head grouping when fuse_qkv=True; False retains independent
    projections and their individual parameter identities.
    Alternatively, ``config`` constructs the selected parameters directly.
    Source construction declares conversion through ``make_transforms``;
    native checkpoints already use this module's layout and need no conversion.
    """

    def _initialize_dimensions(self, module: nn.Module | None) -> None:
        """Initialize head dimensions and grouped-QKV layout metadata."""
        config = self.config
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        dimensions = (config.hidden_size, self.num_heads, self.num_key_value_heads)
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in dimensions):
            raise ValueError("GQA hidden size and head counts must be positive integers")
        if self.num_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.qk_head_dim = getattr(module, "head_dim", getattr(config, "head_dim", None))
        if self.qk_head_dim is None:
            if config.hidden_size % self.num_heads:
                raise ValueError("GQA requires an explicit head_dim when hidden_size is not divisible by query heads")
            self.qk_head_dim = config.hidden_size // self.num_heads
        self.v_head_dim = getattr(module, "v_head_dim", getattr(config, "v_head_dim", None))
        if self.v_head_dim is None:
            self.v_head_dim = self.qk_head_dim
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
               for value in (self.qk_head_dim, self.v_head_dim)):
            raise ValueError("GQA head dimensions must be positive integers")
        self.qkv_split_sizes = (
            self.num_key_value_groups * self.qk_head_dim,
            self.qk_head_dim,
            self.v_head_dim,
        )
        self.qkv_group_width = sum(self.qkv_split_sizes)

    def _initialize_attention_options(self, module: Any) -> None:
        """Copy supported attention options and reject unsupported variants."""
        config = self.config
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
        # Reject unsupported source mathematics rather than dropping its sink,
        # normalization or gate while converting the checkpoint.
        if getattr(module, "param_sink_number", 0) > 0:
            raise ValueError(
                "GroupedQueryAttention only implements standard GQA; parameter-sink attention "
                "must use the sink_attention high-performance interface."
            )
        if getattr(module, "attn_groupnorm", False) or getattr(module, "attn_elementwise_gate", False):
            raise ValueError(
                "GroupedQueryAttention only implements standard GQA; group normalization and "
                "elementwise attention gates require a dedicated replacement."
            )

    def _source_qkv(self, module: nn.Module) -> tuple[nn.Linear, tuple[str, ...]]:
        """Validate and return the source QKV projection layout."""
        self._source_is_fused = hasattr(module, "qkv_proj")
        if self._source_is_fused:
            if not self.fuse_qkv:
                raise ValueError("fuse_qkv=False requires separate Q/K/V source projections")
            source_qkv = module.qkv_proj
            if not isinstance(source_qkv, nn.Linear):
                raise TypeError("qkv_proj must be torch.nn.Linear")
            expected_size = (
                self.num_heads * self.qk_head_dim
                + self.num_key_value_heads * self.qk_head_dim
                + self.num_key_value_heads * self.v_head_dim
            )
            if source_qkv.out_features != expected_size:
                raise ValueError("qkv_proj output size is incompatible with the configured GQA dimensions")
            return source_qkv, ("qkv_proj",)
        required = ("q_proj", "k_proj", "v_proj")
        if any(not hasattr(module, name) for name in required):
            raise TypeError("GroupedQueryAttention requires qkv_proj or q_proj/k_proj/v_proj on the source module")
        projections = tuple(getattr(module, name) for name in required)
        if any(not isinstance(projection, nn.Linear) for projection in projections):
            raise TypeError("q_proj, k_proj, and v_proj must be torch.nn.Linear")
        expected_sizes = (
            self.num_heads * self.qk_head_dim,
            self.num_key_value_heads * self.qk_head_dim,
            self.num_key_value_heads * self.v_head_dim,
        )
        if tuple(projection.out_features for projection in projections) != expected_sizes:
            raise ValueError("Q/K/V projection sizes are incompatible with the configured GQA dimensions")
        if len({projection.in_features for projection in projections}) != 1:
            raise ValueError("Q/K/V projections must accept the same hidden width")
        if self.fuse_qkv and not _projections_can_fuse(projections):
            raise ValueError("Q, K, and V projections cannot be represented by one fused projection")
        return module.q_proj, required

    def _initialize_projections(self, module: nn.Module, source_qkv: nn.Linear) -> None:
        """Create fused rows or retain the independent source projection boundaries."""
        if not isinstance(getattr(module, "o_proj", None), nn.Linear):
            raise TypeError("GroupedQueryAttention requires o_proj on the source module")
        if module.o_proj.in_features != self.num_heads * self.v_head_dim:
            raise ValueError("o_proj input size is incompatible with the configured GQA dimensions")
        if not self.fuse_qkv:
            self._initialize_separate_projections(module)
            return
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
        self.o_proj = module.o_proj
        self.q_norm = getattr(module, "q_norm", None)
        self.k_norm = getattr(module, "k_norm", None)

    def _initialize_separate_projections(self, module: nn.Module) -> None:
        """Retain source Linear boundaries for FSDP and independent Q/K/V submission."""
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, name, getattr(module, name))
        for name in ("q_norm", "k_norm"):
            norm = getattr(module, name, None)
            # The former three-projection Qwen path used fused direct-scale RMSNorm.
            # Retain that kernel as a real child Module so FSDP hooks are not bypassed.
            if norm is not None and hasattr(norm, "variance_epsilon"):
                if set(dict(norm.named_parameters())) != {"weight"}:
                    raise ValueError("GQA direct-scale RMSNorm requires only a weight parameter")
                norm = RMSNorm(module=norm, use_fused_ops=True)
            setattr(self, name, norm)

    def __init__(self, *, config: Any = None, module: nn.Module | None = None,
                 fuse_qkv: bool = True) -> None:
        """Build fixed QKV parameters from configuration or a supported source.

        Args:
            config: Native dimensions and math options; exclusive with module.
            module: Source attention providing weights and checkpoint conversion metadata.
            fuse_qkv: Use grouped linear_qkv rows or independent Q/K/V children.
        """
        super().__init__()
        if not isinstance(fuse_qkv, bool):
            raise TypeError("GQA fuse_qkv must be boolean")
        self.fuse_qkv = fuse_qkv
        if (config is None) == (module is None):
            raise ValueError("GroupedQueryAttention requires exactly one of config or module")
        self._config_initialization = module is None or getattr(module, "_config_initialization", False)
        self.attention_interface = npu_fusion_attention_forward
        self.config = config if module is None else module.config
        source = config if module is None else module
        self.layer_idx = getattr(source, "layer_idx", getattr(source, "layer_number", None))
        self._initialize_dimensions(module)
        if module is None:
            self._initialize_config(config)
            return
        if isinstance(module, GroupedQueryAttentionModule):
            if self.fuse_qkv != module.fuse_qkv:
                raise ValueError("GQA promotion cannot change the fused/split parameter representation")
            if hasattr(self, "gated_qkv_group_width") != hasattr(module, "gated_qkv_group_width"):
                raise ValueError("GQA promotion cannot add or remove query gate parameters")
            # Promotion happens before sharding: retain weights, FQNs and conversion provenance.
            for name in ("_source_is_fused", "_source_projection_names"):
                setattr(self, name, getattr(module, name, None))
            # Preserve registration order and aliases as well as parameter names.
            for name, child in module.named_modules(remove_duplicate=False):
                if name and "." not in name:
                    self.add_module(name, child)
            for name in ("q_norm", "k_norm"):
                if not hasattr(self, name):
                    setattr(self, name, getattr(module, name, None))
            self._initialize_attention_options(module)
            self.train(module.training)
            return
        self._initialize_attention_options(module)
        source_qkv, source_projection_names = self._source_qkv(module)
        self._source_projection_names = source_projection_names
        self._initialize_projections(module, source_qkv)
        self.train(module.training)

    def _initialize_config(self, config: Any) -> None:
        """Create native projections and optional direct-scale Q/K RMSNorm."""
        self._initialize_attention_options(config)
        bias = getattr(config, "attention_bias", False)
        output_bias = getattr(config, "o_proj_bias", bias)
        qk_norm = getattr(config, "qk_norm", False)
        if any(not isinstance(value, bool) for value in (bias, output_bias, qk_norm)):
            raise ValueError("GQA attention_bias, o_proj_bias and qk_norm must be boolean")
        eps = getattr(config, "rms_norm_eps", 1e-6)
        if qk_norm and (not isinstance(eps, (int, float)) or isinstance(eps, bool) or not eps > 0):
            raise ValueError("GQA rms_norm_eps must be positive when qk_norm is enabled")
        dtype = getattr(config, "dtype", None)
        self._source_is_fused = False
        self._source_projection_names = ()
        self._initialize_config_projections(config, bias, dtype)
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, config.hidden_size, bias=output_bias, dtype=dtype)
        # Norm presence is an algorithm choice, not something inferred from model_type
        # or rms_norm_eps. Native gamma is direct scale; source Norms keep their own convention.
        self.q_norm = (
            RMSNorm(config=SimpleNamespace(hidden_size=self.qk_head_dim, eps=eps), use_fused_ops=True).to(dtype=dtype)
            if qk_norm else None
        )
        self.k_norm = (
            RMSNorm(config=SimpleNamespace(hidden_size=self.qk_head_dim, eps=eps), use_fused_ops=True).to(dtype=dtype)
            if qk_norm else None
        )

    def _initialize_config_projections(self, config: Any, bias: bool, dtype: torch.dtype | None) -> None:
        """Allocate native grouped rows or independent Q/K/V projections."""
        if self.fuse_qkv:
            self.linear_qkv = nn.Linear(
                config.hidden_size, self.num_key_value_heads * self.qkv_group_width, bias=bias, dtype=dtype)
        else:
            self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.qk_head_dim, bias=bias, dtype=dtype)
            self.k_proj = nn.Linear(
                config.hidden_size, self.num_key_value_heads * self.qk_head_dim, bias=bias, dtype=dtype)
            self.v_proj = nn.Linear(
                config.hidden_size, self.num_key_value_heads * self.v_head_dim, bias=bias, dtype=dtype)

    def reset_parameters(self) -> None:
        """Initialize native/meta parameters without resetting converted source weights."""
        if not self._config_initialization:
            return
        for name in ("linear_qkv", "q_proj", "k_proj", "v_proj", "o_proj"):
            projection = getattr(self, name, None)
            if projection is not None:
                projection.reset_parameters()
        for norm in (self.q_norm, self.k_norm):
            if norm is not None:
                nn.init.ones_(norm.weight)

    def make_transforms(self) -> list[WeightConverter]:
        """Describe reversible source-checkpoint to grouped-QKV conversion."""
        if self._config_initialization or not self.fuse_qkv:
            return []
        suffixes = ("weight", "bias") if self.linear_qkv.bias is not None else ("weight",)
        transforms = []
        for suffix in suffixes:
            sources = [f"{name}.{suffix}" for name in self._source_projection_names]
            transforms.append(WeightConverter(
                source_patterns=sources[0] if self._source_is_fused else sources,
                target_patterns=f"linear_qkv.{suffix}",
                operations=[InterleaveQKV(
                    self.config.num_key_value_heads, self.num_key_value_groups,
                    self.qk_head_dim, self.v_head_dim, self._source_is_fused,
                )],
            ))
        return transforms

    def split_projection(self, hidden_states: torch.Tensor) -> tuple:
        """Produce grouped Q/K/V and an empty output-stage state for ordinary GQA."""
        projected = self.linear_qkv(hidden_states).view(
            *hidden_states.shape[:-1], self.num_key_value_heads, self.qkv_group_width)
        query, key, value = projected.split(self.qkv_split_sizes, dim=-1)
        return query.reshape(*hidden_states.shape[:-1], self.num_heads, self.qk_head_dim), key, value, None

    def normalize_position(self, tensor: torch.Tensor, norm: nn.Module | None, positions: Any) -> torch.Tensor:
        """Apply per-head Norm and position math through ordinary child-module calls."""
        if norm is not None:
            tensor = norm(tensor)
        return apply_gqa_position(tensor, positions, self.rotary_interleaved)

    def project_query(self, hidden_states: torch.Tensor, positions: Any) -> torch.Tensor:
        """Compute a query from the independent projection representation.

        Args:
            hidden_states: Local input tokens in BSH layout.
            positions: Local cosine/sine tables, or None.

        Returns:
            Query in BSND after its optional Norm and positional transform.
        """
        query = self.q_proj(hidden_states).view(*hidden_states.shape[:-1], self.num_heads, self.qk_head_dim)
        return self.normalize_position(query, self.q_norm, positions)

    def project_key(self, hidden_states: torch.Tensor, positions: Any) -> torch.Tensor:
        """Compute a key from the independent projection representation.

        Args:
            hidden_states: Local input tokens in BSH layout.
            positions: Local cosine/sine tables, or None.

        Returns:
            Key in BSND after its optional Norm and positional transform.
        """
        key = self.k_proj(hidden_states).view(*hidden_states.shape[:-1], self.num_key_value_heads, self.qk_head_dim)
        return self.normalize_position(key, self.k_norm, positions)

    def project_value(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute a value from the independent projection representation.

        Args:
            hidden_states: Local input tokens in BSH layout.

        Returns:
            Value in BSND without a positional transform.
        """
        return self.v_proj(hidden_states).view(*hidden_states.shape[:-1], self.num_key_value_heads, self.v_head_dim)

    def project_qkv(self, hidden_states: torch.Tensor, positions: Any) -> tuple:
        """Compute local Q/K/V and the optional gate using the selected projections.

        Args:
            hidden_states: Local input tokens in BSH layout.
            positions: Local cosine/sine tables, or None.

        Returns:
            Ready BSND Q/K/V and algorithm-owned output state. Fused parameters
            use one GEMM; independent parameters keep their ordinary child calls.
        """
        if not self.fuse_qkv:
            query = self.project_query(hidden_states, positions)
            key = self.project_key(hidden_states, positions)
            value = self.project_value(hidden_states)
            return query, key, value, None
        query, key, value, output_state = self.split_projection(hidden_states)
        query = self.normalize_position(query, self.q_norm, positions)
        key = self.normalize_position(key, self.k_norm, positions)
        return query, key, value, output_state

    def project_output(self, attention_output: torch.Tensor, output_state: Any) -> torch.Tensor:
        """Project local tokens; ordinary GQA has no output gate."""
        del output_state
        return self.o_proj(attention_output.flatten(-2).contiguous())

    def prepare_positions(self, query_shape: tuple[int, int], positions: Any) -> Any:
        """Preserve the precomputed cosine/sine inputs for these tokens.

        Args:
            query_shape: Original batch and local sequence dimensions.
            positions: Token-aligned cosine/sine inputs, or None; these are not position IDs.
        """
        del query_shape
        return positions

    def prepare_attention_inputs(self, hidden_states: torch.Tensor, positions: Any,
                                 attention_mask: torch.Tensor | None = None, past_key_values: Any = None,
                                 actual_seq_len: torch.Tensor | Sequence[int] | None = None,
                                 **kwargs: Any) -> Any:
        """Prepare local operands using the shared projection and position math.

        This implementation returns ready GQAInputs. The shared forward passes
        the preparation result to exchange_attention before run_attention.

        Args:
            hidden_states: Original local input tokens in BSH layout.
            positions: Local cosine/sine tables, or None.
            attention_mask: Optional padding or visibility mask.
            past_key_values: Optional local inference cache.
            actual_seq_len: Cumulative document endpoints.
            **kwargs: Additional attention options.

        Returns:
            Ready local GQAInputs, consumed by exchange_attention.
        """
        query, key, value, output_state = self.project_qkv(hidden_states, positions)
        query, key, value = (tensor.transpose(1, 2) for tensor in (query, key, value))
        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx)
        if attention_mask is not None and attention_mask.ndim == 4:
            # HF cache masks may reserve more KV slots than the cache currently exposes.
            attention_mask = attention_mask[..., :key.shape[2]]
        return GQAInputs(query, key, value, attention_mask,
                         dict(kwargs, actual_seq_len=actual_seq_len), output_state)

    def exchange_attention(self, prepared: Any) -> GQAInputs:
        """Return ready FA inputs without changing local ownership.

        Args:
            prepared: Local GQAInputs returned by prepare_attention_inputs.

        Returns:
            Ready BNSD operands and attention metadata.
        """
        return prepared

    def run_attention(self, inputs: GQAInputs) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the common attention algorithm after operands are ready.

        Args:
            inputs: Ready BNSD operands, mask and FA options for this invocation.
        """
        options = dict(inputs.options)
        options.setdefault("sliding_window", self.sliding_window)
        return self.attention_interface(
            self, inputs.query, inputs.key, inputs.value, inputs.attention_mask,
            dropout=self.attention_dropout if self.training else 0.0, scaling=self.scaling, **options)

    def restore_attention(self, output: torch.Tensor, output_state: Any) -> tuple[torch.Tensor, Any]:
        """Preserve local query ownership and return algorithm-only output state.

        Args:
            output: Attention output in BSND layout.
            output_state: Optional gate tensor produced by the projection stage.

        Returns:
            Local-token attention output and the state consumed by project_output.
        """
        return output, output_state

    def forward(self, hidden_states: torch.Tensor, position_embeddings: Any = None,
                attention_mask: torch.Tensor | None = None, past_key_values: Any = None,
                actual_seq_len: torch.Tensor | Sequence[int] | None = None,
                **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run shared GQA math with class-defined operand exchange and restoration.

        Args:
            hidden_states: Local input tokens in BSH layout.
            position_embeddings: Optional precomputed cosine/sine tables.
            attention_mask: Optional padding or explicit visibility mask.
            past_key_values: Optional local inference cache.
            actual_seq_len: Optional cumulative document endpoints.
            **kwargs: Additional attention options.
        """
        query_shape = hidden_states.shape[:2]
        positions = self.prepare_positions(query_shape, position_embeddings)
        prepared = self.prepare_attention_inputs(
            hidden_states, positions, attention_mask, past_key_values, actual_seq_len, **kwargs)
        inputs = self.exchange_attention(prepared)
        del prepared
        output, weights = self.run_attention(inputs)
        output_state = inputs.output_state
        # Output stages need only output_state, not an extra owner of QKV.
        # Tensors saved for backward remain owned by autograd.
        del inputs
        output, output_state = self.restore_attention(output, output_state)
        return self.project_output(output, output_state), weights

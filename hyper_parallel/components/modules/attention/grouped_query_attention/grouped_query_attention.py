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
"""Grouped QKV math and the TP execution boundary, independent of CP scheduling."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import torch
from torch import nn

from hyper_parallel.components.checkpoint import InterleaveQKV
from hyper_parallel.components.checkpoint.weight_conversion import WeightConverter
from hyper_parallel.components.functional import apply_rotary_pos_emb_single, npu_fusion_attention_forward
from hyper_parallel.components.modules.rms_norm import RMSNorm
from hyper_parallel.core.dtensor.placement_types import Partial, Replicate
from hyper_parallel.distributed.module_binding import ParallelModuleSpec
from hyper_parallel.distributed.tensor_parallel.execution import TensorParallelExecution
from .grouped_query_attention_family import validate_gqa_layout


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
    # Preserve the existing eager partial-half path; changing its kernel is independent of CP.
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


class GQAOutputLinear(nn.Linear):
    """Keep the output projection's FSDP boundary and add bias after TP reduction."""

    def __init__(self, source: nn.Linear) -> None:
        """Reuse the source parameters before sharding or FSDP hook installation."""
        nn.Module.__init__(self)
        self.in_features, self.out_features = source.in_features, source.out_features
        self.weight, self.bias = source.weight, source.bias
        self.tensor_parallel = TensorParallelExecution()
        self.train(source.training)

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # pylint: disable=redefined-builtin
        """Reduce partial output before adding the replicated bias once."""
        output = torch.nn.functional.linear(input, self.weight, None)
        output = self.tensor_parallel.finalize_output(output)
        return output if self.bias is None else output + self.bias


class GroupedQueryAttentionModule(nn.Module):
    """Transformers-compatible GQA using a grouped ``linear_qkv`` layout.

    The source module may expose a fused ``qkv_proj`` or separate
    ``q_proj``/``k_proj``/``v_proj`` layers. Its checkpoint layout is converted
    to the per-KV-head grouping consumed by the high-performance kernel.
    Construction only creates the target structure; ``make_transforms``
    declares the checkpoint conversion.
    """

    def _initialize_dimensions(self, module: nn.Module) -> None:
        """Initialize head dimensions and grouped-QKV layout metadata."""
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

    def _initialize_attention_options(self, module: nn.Module) -> None:
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
        if getattr(module, "param_sink_number", 0) > 0:
            raise ValueError(
                "GroupedQueryAttention only implements standard GQA; parameter-sink attention "
                "must use the sink_attention high-performance interface."
            )
        self.attn_groupnorm = bool(getattr(module, "attn_groupnorm", False))
        self.attn_elementwise_gate = bool(getattr(module, "attn_elementwise_gate", False))
        if self.attn_groupnorm or self.attn_elementwise_gate:
            raise ValueError(
                "GroupedQueryAttention only implements standard GQA; group normalization and "
                "elementwise attention gates require a dedicated replacement."
            )

    def _source_qkv(self, module: nn.Module) -> tuple[nn.Linear, tuple[str, ...]]:
        """Validate and return the source QKV projection layout."""
        self._source_is_fused = hasattr(module, "qkv_proj")
        if self._source_is_fused:
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
        if not _projections_can_fuse(projections):
            raise ValueError("Q, K, and V projections cannot be represented by one fused projection")
        return module.q_proj, required

    def _initialize_projections(self, module: nn.Module, source_qkv: nn.Linear) -> None:
        """Create the fused QKV projection and attach the source output projection."""
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
            raise TypeError("GroupedQueryAttention requires o_proj on the source module")
        if module.o_proj.in_features != self.num_heads * self.v_head_dim:
            raise ValueError("o_proj input size is incompatible with the configured GQA dimensions")
        self.o_proj = module.o_proj
        self.q_norm = getattr(module, "q_norm", None)
        self.k_norm = getattr(module, "k_norm", None)

    def __init__(
        self,
        *,
        module: nn.Module | None = None,
        config: Any = None,
        module_fqn: str = "",
        context: Mapping[str, Any] | None = None,
        overlap: bool = False,
        ulysses_degree: int | None = None,
    ) -> None:
        """Build the high-performance module from a Transformers attention module."""
        super().__init__()
        del module_fqn, config
        if module is None:
            raise ValueError("GroupedQueryAttention requires a source attention module")
        self.tp_degree = (context or {}).get("tp_size", 1)
        self.overlap = overlap
        self.ulysses_degree = ulysses_degree
        self.tensor_parallel = TensorParallelExecution()
        self.attention_interface = npu_fusion_attention_forward
        config = module.config
        self.config = config
        self.layer_idx = getattr(module, "layer_idx", getattr(module, "layer_number", None))
        self._initialize_dimensions(module)
        if isinstance(module, GroupedQueryAttentionModule):
            if (isinstance(self, SplitGroupedQueryAttentionModule) !=
                    isinstance(module, SplitGroupedQueryAttentionModule)):
                raise ValueError("GQA promotion cannot change the fused/split parameter representation")
            if hasattr(self, "gated_qkv_group_width") != hasattr(module, "gated_qkv_group_width"):
                raise ValueError("GQA promotion cannot add or remove query gate parameters")
            # Promotion happens before sharding: retain weights, FQNs and conversion provenance.
            for name in ("_source_is_fused", "_source_projection_names"):
                setattr(self, name, getattr(module, name, None))
            for name in ("linear_qkv", "q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
                if hasattr(module, name):
                    setattr(self, name, getattr(module, name))
            self._initialize_attention_options(module)
            self.train(module.training)
            return
        self._initialize_attention_options(module)
        source_qkv, source_projection_names = self._source_qkv(module)
        self._source_projection_names = source_projection_names
        self._initialize_projections(module, source_qkv)
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
                        self.config.num_key_value_heads,
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
                            self.config.num_key_value_heads,
                            self.num_key_value_groups,
                            self.qk_head_dim,
                            self.v_head_dim,
                            self._source_is_fused,
                        )
                    ],
                )
            )
        return transforms

    def parallel_module_spec(self) -> ParallelModuleSpec:
        """Declare the class-owned input and output TP boundaries."""
        return ParallelModuleSpec("attention.gqa.local", "mesh",
                                  {"prepare_input": ("hidden_states",),
                                   "project_output": ("attention_output", "output_state")},
                                  validate=validate_gqa_layout)

    def bind_parallel(self, binding: Any) -> None:
        """Consume the final TP layout without recreating parameters or CP groups."""
        spec = binding.sharding
        input_layout, output_layout = dict(spec.in_src["hidden_states"]), dict(spec.out_dst["output"])
        if spec.params["o_proj.weight"].get("tp", Replicate()).is_shard():
            input_layout["tp"], output_layout["tp"] = Replicate(), Partial()
            if not isinstance(self.o_proj, GQAOutputLinear):
                self.o_proj = GQAOutputLinear(self.o_proj)
        self.tensor_parallel = TensorParallelExecution.from_binding(
            binding, input_layout=input_layout, output_layout=output_layout)
        if isinstance(self.o_proj, GQAOutputLinear):
            self.o_proj.tensor_parallel = self.tensor_parallel
        self.num_heads = self.config.num_attention_heads // self.tp_degree
        self.num_key_value_heads = self.config.num_key_value_heads // self.tp_degree

    def prepare_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Gather SP tokens before the head-sharded projections."""
        return self.tensor_parallel.prepare_input(hidden_states)

    def split_projection(self, hidden_states: torch.Tensor) -> tuple:
        """Produce grouped Q/K/V and an empty output-stage state for ordinary GQA."""
        projected = self.linear_qkv(hidden_states).view(
            *hidden_states.shape[:-1], self.num_key_value_heads, self.qkv_group_width)
        query, key, value = projected.split(self.qkv_split_sizes, dim=-1)
        return query.reshape(*hidden_states.shape[:-1], self.num_heads, self.qk_head_dim), key, value, None

    def normalize_position(self, tensor: torch.Tensor, norm: nn.Module | None, positions: Any) -> torch.Tensor:
        """Prepare a single Q/K branch before its CP launch, preserving child module hooks."""
        if norm is not None:
            tensor = norm(tensor)
        return apply_gqa_position(tensor, positions, self.rotary_interleaved)

    def projection_stages(self, hidden_states: torch.Tensor, positions: Any) -> Iterator[tuple]:
        """Yield finalized Q, K, V in order so CP can launch before the next stage."""
        query, key, value, output_state = self.split_projection(hidden_states)
        yield self.normalize_position(query, self.q_norm, positions), None
        yield self.normalize_position(key, self.k_norm, positions), None
        yield value, output_state

    def project_output(self, attention_output: torch.Tensor, output_state: Any) -> torch.Tensor:
        """Project local tokens; ordinary GQA has no output gate."""
        del output_state
        return self.o_proj(attention_output.flatten(-2).contiguous())

    def forward(self, hidden_states: torch.Tensor, position_embeddings: Any = None,
                attention_mask: torch.Tensor | None = None, past_key_values: Any = None,
                actual_seq_len: torch.Tensor | Sequence[int] | None = None,
                **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run local GQA; packed lengths reach the common FA without being discarded."""
        hidden_states = self.prepare_input(hidden_states)
        (query, _), (key, _), (value, output_state) = self.projection_stages(hidden_states, position_embeddings)
        query, key, value = (tensor.transpose(1, 2) for tensor in (query, key, value))
        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx)
        if attention_mask is not None and attention_mask.ndim == 4:
            # HF cache masks may reserve more KV slots than the cache currently exposes.
            attention_mask = attention_mask[..., :key.shape[2]]
        options = dict(kwargs)
        options.setdefault("sliding_window", self.sliding_window)
        output, weights = self.attention_interface(
            self, query, key, value, attention_mask,
            dropout=self.attention_dropout if self.training else 0.0, scaling=self.scaling,
            actual_seq_len=actual_seq_len, **options)
        return self.project_output(output, output_state), weights


class SplitGroupedQueryAttentionModule(GroupedQueryAttentionModule):
    """Existing three-projection representation; overlap never changes its parameters."""

    def _source_qkv(self, module: nn.Module) -> tuple[nn.Linear, tuple[str, ...]]:
        """Validate independent projections without imposing fused bias/frozen-parameter restrictions."""
        self._source_is_fused = False
        names = ("q_proj", "k_proj", "v_proj")
        widths = (self.num_heads * self.qk_head_dim, self.num_key_value_heads * self.qk_head_dim,
                  self.num_key_value_heads * self.v_head_dim)
        for name, width in zip(names, widths):
            projection = getattr(module, name, None)
            if not isinstance(projection, nn.Linear) or projection.out_features != width:
                raise ValueError(f"Split GQA requires {name} with {width} output features")
        if not isinstance(getattr(module, "o_proj", None), nn.Linear):
            raise ValueError("Split GQA requires a Linear output projection")
        if module.o_proj.in_features != self.num_heads * self.v_head_dim:
            raise ValueError("Split GQA output projection has an incompatible input width")
        return module.q_proj, names

    def _initialize_projections(self, module: nn.Module, source_qkv: nn.Linear) -> None:
        """Retain source Linear boundaries for FSDP and independent Q/K/V submission."""
        if self._source_is_fused:
            raise ValueError("SplitGroupedQueryAttention requires separate Q/K/V source projections")
        del source_qkv
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, name, getattr(module, name))
        for name in ("q_norm", "k_norm"):
            norm = getattr(module, name, None)
            # The former three-projection Qwen path used fused direct-scale RMSNorm.
            # Retain that kernel as a real child Module so FSDP hooks are not bypassed.
            if norm is not None and hasattr(norm, "variance_epsilon"):
                if set(dict(norm.named_parameters())) != {"weight"}:
                    raise ValueError("Split GQA direct-scale RMSNorm requires only a weight parameter")
                norm = RMSNorm(module=norm)
            setattr(self, name, norm)

    def make_transforms(self) -> list[WeightConverter]:
        """Keep source checkpoint names and values without a fusion conversion."""
        return []

    def projection_stages(self, hidden_states: torch.Tensor, positions: Any) -> Iterator[tuple]:
        """Q projection/Norm/RoPE can communicate while K and V are still computed."""
        shape = hidden_states.shape[:-1]
        query = self.q_proj(hidden_states).view(*shape, self.num_heads, self.qk_head_dim)
        yield self.normalize_position(query, self.q_norm, positions), None
        key = self.k_proj(hidden_states).view(*shape, self.num_key_value_heads, self.qk_head_dim)
        yield self.normalize_position(key, self.k_norm, positions), None
        yield self.v_proj(hidden_states).view(*shape, self.num_key_value_heads, self.v_head_dim), None

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
"""Shared tiny MLA fixtures and independent mathematical reference for UT/ST."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from transformers import DeepseekV3Config, DeepseekV3ForCausalLM
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3Attention

from hyper_parallel.components.modules.mla_attention import MLAAttention
from hyper_parallel.distributed._builder.forward_rewriter import (
    _commit_forward_rewrite,
    validate_wrapped_forward,
)
from hyper_parallel.distributed.context_parallel.mla_context_parallel import MLADimensions
from hyper_parallel.models.deepseek_v3.adapter.distributed.context_parallel import mla_cp_wrapper
from hyper_parallel.models.replacement import (
    ModuleReplacementSpec, apply_module_replacements, compile_module_replacements,
)


def attention_options(lengths: tuple[int, ...] | None = None, causal: bool = True) -> dict:
    """Build ordinary model kwargs shared by packed and unpacked test cases."""
    return {"actual_seq_len": lengths, "is_causal": causal}


class TestCPMesh:
    """Minimal mesh protocol backed by a real process group, or a singleton."""

    __test__ = False

    def __init__(self, group: Any = None, singleton: bool = False) -> None:
        """Bind a process group, or describe a communication-free singleton."""
        self.group = group
        self.singleton = singleton

    def size(self) -> int:
        """Return the CP degree."""
        return 1 if self.singleton else dist.get_world_size(self.group)

    def get_local_rank(self) -> int:
        """Return rank within CP."""
        return 0 if self.singleton else dist.get_rank(self.group)

    def get_group(self) -> Any:
        """Return the existing process group."""
        return self.group


class TinyRMSNorm(nn.Module):
    """Trainable RMSNorm with FP32 accumulation for low precision inputs."""

    def __init__(self, width: int) -> None:
        """Create one scale for every latent feature."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize the complete latent feature dimension."""
        variance_input = tensor if tensor.dtype == torch.float64 else tensor.float()
        normalized = variance_input * (variance_input.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
        return normalized.to(tensor.dtype) * self.weight


def make_mla(
    dimensions: MLADimensions | None = None,
    *,
    dtype: torch.dtype = torch.float64,
    interleaved: bool = False,
    bias: bool = True,
    hidden_dim: int | None = None,
) -> MLAAttention:
    """Build the replacement and explicitly load its declared fused down-projection weights."""
    dims = dimensions or MLADimensions(4, 5, 6, 3, 2, 4)
    hidden_dim = hidden_dim or (12 if dims.qk_dim < 32 else 64)
    source = nn.Module()
    source.config = SimpleNamespace(
        num_attention_heads=dims.heads, q_lora_rank=dims.q_rank, kv_lora_rank=dims.kv_rank,
        qk_nope_head_dim=dims.nope_dim, qk_rope_head_dim=dims.rope_dim, v_head_dim=dims.value_dim,
    )
    source.q_a_proj = nn.Linear(hidden_dim, dims.q_rank, bias=bias)
    source.kv_a_proj_with_mqa = nn.Linear(hidden_dim, dims.kv_rank + dims.rope_dim, bias=bias)
    source.q_a_layernorm = TinyRMSNorm(dims.q_rank)
    source.kv_a_layernorm = TinyRMSNorm(dims.kv_rank)
    source.q_b_proj = nn.Linear(dims.q_rank, dims.heads * dims.qk_dim, bias=bias)
    source.kv_b_proj = nn.Linear(dims.kv_rank, dims.heads * (dims.nope_dim + dims.value_dim), bias=bias)
    source.o_proj = nn.Linear(dims.heads * dims.value_dim, hidden_dim, bias=bias)
    source.scaling = dims.qk_dim**-0.5 * 0.73
    source.rotary_interleaved = interleaved
    source = source.to(dtype=dtype)
    result = MLAAttention(module=source)
    with torch.no_grad():
        result.linear_qkv.weight.copy_(torch.cat((source.q_a_proj.weight, source.kv_a_proj_with_mqa.weight)))
        if bias:
            result.linear_qkv.bias.copy_(torch.cat((source.q_a_proj.bias, source.kv_a_proj_with_mqa.bias)))
    return result


def install_mla_cp(module: MLAAttention, mesh: TestCPMesh, strategy: str, backend: str, **options: Any) -> None:
    """Commit the same adapter requests as the recipe rewriter, for isolated tests."""
    requests = mla_cp_wrapper(module, None, None, mesh, None, strategy=strategy, backend=backend, **options)
    for request in requests:
        validate_wrapped_forward(request.target.forward, request.forward, owner="MLA CP test")
    for request in requests:
        _commit_forward_rewrite(request)


def position_embeddings(
    sequence: int,
    rope_dim: int,
    metadata: dict,
    *,
    dtype: torch.dtype,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create nontrivial document-relative RoPE frequencies with a YaRN-like scale."""
    ends = metadata.get("actual_seq_len") or (sequence,)
    positions = torch.cat([torch.arange(end - begin) for begin, end in zip((0,) + ends[:-1], ends)])
    inverse = torch.linspace(0.017, 0.21, rope_dim // 2)
    angles = positions[:, None] * inverse[None]
    angles = torch.cat((angles, angles), dim=-1)
    return tuple((part * 1.07).to(device=device, dtype=dtype).unsqueeze(0) for part in (angles.cos(), angles.sin()))


def mathematical_reference(
    module: MLAAttention,
    hidden: torch.Tensor,
    positions: tuple[torch.Tensor, torch.Tensor] | None,
    metadata: dict,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Independent explicit-score MLA reference; intentionally small and quadratic."""
    batch, sequence, _ = hidden.shape
    latent = F.linear(hidden, module.linear_qkv.weight, module.linear_qkv.bias)  # pylint: disable=not-callable
    query_latent, kv_latent, key_rope = latent.split(
        (module.q_lora_rank, module.kv_lora_rank, module.qk_rope_head_dim), dim=-1,
    )
    query = module.q_b_proj(module.q_a_layernorm(query_latent))
    query = query.view(batch, sequence, module.num_heads, module.qk_head_dim).transpose(1, 2)
    query_content, query_rope = query.split((module.qk_nope_head_dim, module.qk_rope_head_dim), dim=-1)
    key_rope = key_rope.unsqueeze(1)
    if positions is not None:
        cos, sin = (part.unsqueeze(1).to(hidden.dtype) for part in positions)
        if module.rotary_interleaved:
            query_rope = torch.cat((query_rope[..., 0::2], query_rope[..., 1::2]), dim=-1)
            key_rope = torch.cat((key_rope[..., 0::2], key_rope[..., 1::2]), dim=-1)
        query_half, key_half = query_rope.chunk(2, -1), key_rope.chunk(2, -1)
        query_rope = query_rope * cos + torch.cat((-query_half[1], query_half[0]), -1) * sin
        key_rope = key_rope * cos + torch.cat((-key_half[1], key_half[0]), -1) * sin
    expanded = module.kv_b_proj(module.kv_a_layernorm(kv_latent))
    expanded = expanded.view(
        batch, sequence, module.num_heads, module.qk_nope_head_dim + module.v_head_dim,
    ).transpose(1, 2)
    content, value = expanded.split((module.qk_nope_head_dim, module.v_head_dim), dim=-1)
    query = torch.cat((query_content, query_rope), dim=-1)
    key = torch.cat((content, key_rope.expand(-1, module.num_heads, -1, -1)), dim=-1)
    allowed = torch.ones(sequence, sequence, dtype=torch.bool, device=hidden.device)
    if metadata.get("actual_seq_len") is not None:
        document_ids = torch.cat([
            torch.full((end - begin,), index, device=hidden.device)
            for index, (begin, end) in enumerate(zip((0,) + metadata.get("actual_seq_len")[:-1],
                                                    metadata.get("actual_seq_len")))
        ])
        allowed &= document_ids[:, None] == document_ids[None, :]
    if metadata.get("is_causal", True):
        allowed &= torch.ones_like(allowed).tril()
    if attention_mask is not None:
        allowed = allowed & attention_mask
    scores = (query @ key.transpose(-2, -1)) * module.scaling
    valid_rows = allowed.any(dim=-1, keepdim=True)
    scores = scores.masked_fill(~allowed, -torch.inf)
    scores = torch.where(valid_rows, scores, torch.zeros_like(scores))
    probability = scores.softmax(-1).masked_fill(~valid_rows, 0.0)
    output = (probability @ value).transpose(1, 2).reshape(batch, sequence, -1)
    return module.o_proj(output)


def make_deepseek_model(*, dtype: torch.dtype = torch.float32, npu_dimensions: bool = False) -> nn.Module:
    """Build two real dense decoder layers without downloading a checkpoint."""
    config = DeepseekV3Config.from_dict({
        "vocab_size": 128, "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
        "num_attention_heads": 8, "num_key_value_heads": 8, "q_lora_rank": 32, "kv_lora_rank": 32,
        "qk_nope_head_dim": 128 if npu_dimensions else 4, "qk_rope_head_dim": 64 if npu_dimensions else 4,
        "v_head_dim": 128 if npu_dimensions else 4, "first_k_dense_replace": 2,
        "n_routed_experts": 4, "n_shared_experts": 1, "moe_intermediate_size": 32,
        "max_position_embeddings": 256, "attention_dropout": 0.0, "attention_bias": True,
        "tie_word_embeddings": False, "use_cache": False, "rope_interleave": True,
    })
    config.architectures = ["DeepseekV3ForCausalLM"]
    # SDPA accepts an explicit boolean mask in the untouched HF reference.
    config._attn_implementation = "sdpa"  # pylint: disable=protected-access
    return DeepseekV3ForCausalLM(config).to(dtype=dtype)


def replace_and_load_mla(model: nn.Module) -> tuple[nn.Module, list[Any]]:
    """Use the real replacement matcher and its declared checkpoint converters."""
    original = deepcopy(model.state_dict())
    rule = ModuleReplacementSpec(
        match=("model.layers.*.self_attn",), factory=MLAAttention, module_type=DeepseekV3Attention,
    )
    plan = compile_module_replacements(model, [rule])
    model, transforms = apply_module_replacements(model, plan, weights_mapping=[])
    converted = dict(original)
    for template in transforms:
        converter = deepcopy(template)
        target = None
        for name, value in original.items():
            renamed, pattern = converter.rename_source_key(name)
            if pattern is not None:
                converter.add_tensor(renamed, name, pattern, value)
                converted.pop(name)
                target = renamed
        if target is None:
            raise AssertionError("MLA checkpoint conversion did not match its source projection")
        values = converter.convert(target, model=model, config=model.config)
        converted.update({name: value[0] if isinstance(value, list) else value for name, value in values.items()})
    model.load_state_dict(converted, strict=True)
    model._weight_conversions = transforms  # pylint: disable=protected-access
    return model, transforms

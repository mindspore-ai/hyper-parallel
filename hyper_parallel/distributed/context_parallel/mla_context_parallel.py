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
"""Dense MLA Ulysses with expanded QKV or shared latent-KV communication."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Any, Literal

# Native Torch is required, as in the existing CP adapters; the legacy lint rule predates that migration.
import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import
from torch.nn import functional as F  # pylint: disable=forbidden-backend-import

from hyper_parallel.distributed.context_parallel.collectives import ulysses_head_to_seq, ulysses_seq_to_head
from hyper_parallel.components.functional.npu_fusion_attention import (
    PACKED_SEQUENCE_ARGUMENTS, npu_fusion_attention_forward, resolve_packed_sequence_lengths,
)
from hyper_parallel.core.utils.communication import differentiable_all_gather_concat


MLACPStrategy = Literal["expanded_ulysses", "latent_kv_head"]
MLA_CP_STRATEGIES = ("expanded_ulysses", "latent_kv_head")


@dataclass(frozen=True)
class MLADimensions:
    """Global MLA dimensions; the latent ranks are not partitioned across heads."""

    heads: int
    q_rank: int
    kv_rank: int
    nope_dim: int
    rope_dim: int
    value_dim: int

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value}")
        if self.rope_dim % 2:
            raise ValueError("rope_dim must be even")

    @property
    def qk_dim(self) -> int:
        """Return the score dimension, including the decoupled RoPE band."""
        return self.nope_dim + self.rope_dim


@dataclass(frozen=True)
class MLAState:
    """Normalized local-token latent tensors in BSR layout, before up projection."""

    query_latent: torch.Tensor
    kv_latent: torch.Tensor
    key_rope: torch.Tensor


@dataclass(frozen=True)
class MLACPPlan:
    """Validated explicit execution strategy; no uncalibrated automatic selection."""

    dimensions: MLADimensions
    cp_degree: int
    strategy: MLACPStrategy = "expanded_ulysses"
    backend: Literal["npu", "sdpa"] = "npu"
    tp_degree: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.cp_degree, int) or isinstance(self.cp_degree, bool) or self.cp_degree < 1:
            raise ValueError("cp_degree must be a positive integer")
        if self.strategy not in MLA_CP_STRATEGIES:
            raise ValueError(f"Unknown MLA CP strategy: {self.strategy}")
        if self.backend not in ("npu", "sdpa"):
            raise ValueError(f"Unknown MLA CP backend: {self.backend}")
        if not isinstance(self.tp_degree, int) or isinstance(self.tp_degree, bool) or self.tp_degree < 1:
            raise ValueError("tp_degree must be a positive integer")
        if self.dimensions.heads % (self.tp_degree * self.cp_degree):
            raise ValueError("MLA heads must be divisible by TP degree times CP degree")

    @property
    def local_heads(self) -> int:
        """Heads in the persistent TP parameter shard, before CP redistribution."""
        return self.dimensions.heads // self.tp_degree

    @property
    def compute_heads(self) -> int:
        """Heads used by this rank's full-sequence attention kernel."""
        return self.local_heads // self.cp_degree

    def validate_scale(self, scale: float) -> None:
        """Reject invalid scales without replacing an explicit YaRN scale.

        Args:
            scale: The model's attention score scale, including any YaRN factor.

        Raises:
            ValueError: If the scale is nonfinite or nonpositive.
        """
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("MLA attention scaling must be finite and positive")


def mla_all_gather(tensor: torch.Tensor, sequence_dim: int, cp_mesh: Any) -> torch.Tensor:
    """Gather source tokens and sum all consumer gradients back to their owner.

    Args:
        tensor: Equal-sized local sequence shard.
        sequence_dim: Dimension along which source shards are concatenated.
        cp_mesh: CP mesh, or None for communication-free execution.

    Returns:
        Full-sequence tensor in CP group-rank order.
    """
    if cp_mesh is None or cp_mesh.size() == 1:
        return tensor
    return differentiable_all_gather_concat(tensor, cp_mesh.get_group(), cp_mesh.size(), sequence_dim)


def _apply_mla_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    *,
    interleaved: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate local BSND Q and shared K with the existing NPU kernels or CPU math."""
    if position_embeddings is None:
        return query, key
    cos, sin = (part.to(query.dtype) for part in position_embeddings)
    if query.device.type == "npu":
        # The shared NPU RoPE module is optional for CPU reference execution.
        from hyper_parallel.components.functional.rotary_embedding import (  # pylint: disable=C0415
            apply_rotary_pos_emb, apply_rotary_pos_emb_interleave,
        )

        rope = apply_rotary_pos_emb_interleave if interleaved else apply_rotary_pos_emb
        return rope(query, key, cos, sin, unsqueeze_dim=2)
    cos, sin = cos.unsqueeze(2), sin.unsqueeze(2)
    outputs = []
    for tensor in (query, key):
        if interleaved:
            real, imaginary = tensor[..., 0::2], tensor[..., 1::2]
            half_cos, half_sin = cos[..., :tensor.shape[-1] // 2], sin[..., :tensor.shape[-1] // 2]
            outputs.append(torch.cat((real * half_cos - imaginary * half_sin,
                                      imaginary * half_cos + real * half_sin), dim=-1))
        else:
            first, second = tensor.chunk(2, dim=-1)
            outputs.append(tensor * cos + torch.cat((-second, first), dim=-1) * sin)
    return tuple(outputs)


def _run_attention(module, query, key, value, *, backend, lengths, causal, attention_mask):
    """Use the shared FA interface after CP has restored full Q/K token order."""
    if backend == "npu":
        return npu_fusion_attention_forward(
            module, query, key, value, attention_mask, scaling=module.scaling,
            is_causal=causal, actual_seq_len=lengths,
            pre_tokens=2147483647, next_tokens=0 if causal else 2147483647,
        )[0]
    allowed = attention_mask
    if causal and allowed is not None:
        allowed = allowed & torch.ones(query.shape[2], key.shape[2], dtype=torch.bool, device=query.device).tril()
    if lengths is None:
        return F.scaled_dot_product_attention(  # pylint: disable=not-callable
            query, key, value, attn_mask=allowed, dropout_p=0.0,
            is_causal=causal and allowed is None, scale=module.scaling,
        ).transpose(1, 2)
    outputs = [
        F.scaled_dot_product_attention(  # pylint: disable=not-callable
            query[:, :, begin:end], key[:, :, begin:end], value[:, :, begin:end],
            dropout_p=0.0, is_causal=causal, scale=module.scaling,
        ).transpose(1, 2)
        for begin, end in zip([0] + lengths[:-1], lengths)
    ]
    return torch.cat(outputs, dim=1)


def _positions(
    embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    state: MLAState,
    rope_dim: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Validate local-token RoPE frequencies and broadcast only the batch axis."""
    if embeddings is None:
        return None
    if not isinstance(embeddings, (tuple, list)) or len(embeddings) != 2:
        raise ValueError("position_embeddings must be a local (cos, sin) pair")
    batch, sequence = state.query_latent.shape[:2]
    parts = []
    for tensor in embeddings:
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.shape not in ((1, sequence, rope_dim), (batch, sequence, rope_dim)):
            raise ValueError("RoPE cos/sin must describe local tokens with shape [B, S_local, rope_dim]")
        if tensor.requires_grad:
            raise ValueError("MLA CP currently requires non-trainable RoPE frequencies")
        if tensor.device != state.query_latent.device:
            raise ValueError("RoPE and MLA activations must be on the same device")
        parts.append(tensor.expand(batch, -1, -1))
    return tuple(parts)


def _validate_mask(
    mask: torch.Tensor | None,
    lengths: Sequence[int] | None,
    batch: int,
    sequence: int,
    device: torch.device,
) -> None:
    """Require global boolean masks and reject unsupported packed-mask combinations."""
    if mask is None:
        return
    if lengths is not None:
        raise ValueError("packed MLA CP currently accepts document causal/noncausal masks only")
    if mask.dtype != torch.bool:
        raise ValueError("MLA CP masks must be boolean (True=allowed); additive bias is not supported")
    valid = (
        mask.ndim == 2 and mask.shape == (sequence, sequence)
        or mask.ndim == 4 and mask.shape[0] in (1, batch)
        and mask.shape[1] == 1 and mask.shape[-2:] == (sequence, sequence)
    )
    if not valid or mask.device != device:
        raise ValueError("attention_mask must cover global Q/K: [S, S] or [B|1, 1, S, S], on the input device")


def _query_and_key_rope(
    module: nn.Module,
    state: MLAState,
    positions: tuple[torch.Tensor, torch.Tensor] | None,
    heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand TP-local query heads and rotate local Q and shared K before exchange."""
    projected = module.q_b_proj(state.query_latent)
    projected = projected.view(*state.query_latent.shape[:2], heads, module.qk_head_dim)
    content, rotary = projected.split((module.qk_nope_head_dim, module.qk_rope_head_dim), dim=-1)
    rotary, key_rope = _apply_mla_rope(
        rotary, state.key_rope.unsqueeze(2), positions, interleaved=module.rotary_interleaved,
    )
    return torch.cat((content, rotary), dim=-1).transpose(1, 2), key_rope.squeeze(2)


def _key_value(
    module: nn.Module,
    latent: torch.Tensor,
    key_rope: torch.Tensor,
    heads: int,
    head_range: tuple[int, int] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand the requested KV heads through the managed projection module."""
    projected = (module.kv_b_proj(latent) if head_range is None
                 else module.kv_b_proj(latent, mla_head_range=head_range))
    projected = projected.view(*latent.shape[:2], heads, module.qk_nope_head_dim + module.v_head_dim)
    content, value = projected.split((module.qk_nope_head_dim, module.v_head_dim), dim=-1)
    key = torch.cat((content, key_rope.unsqueeze(2).expand(-1, -1, heads, -1)), dim=-1)
    return key.transpose(1, 2), value.transpose(1, 2)


def _validate_forward_kwargs(kwargs, position_embeddings):
    """Validate forwarded model options before routing any latent."""
    position_ids = kwargs.pop("position_ids", None)
    if position_ids is not None and position_embeddings is None:
        raise ValueError("position_ids require precomputed local position_embeddings in MLA CP")
    kwargs.pop("cache_position", None)
    if kwargs.pop("use_cache", False) or kwargs.pop("output_attentions", False):
        raise ValueError("MLA CP does not return KV caches or attention weights")
    if kwargs:
        raise ValueError(f"Unsupported MLA CP forward arguments: {sorted(kwargs)}")


def _validate_state(module: nn.Module, plan: MLACPPlan, state: MLAState) -> None:
    """Check latent shape, dtype and backend requirements before collectives."""
    dimensions = plan.dimensions
    query_latent, kv_latent, key_rope = state.query_latent, state.kv_latent, state.key_rope
    if query_latent.ndim != 3:
        raise ValueError("MLA CP latents must have BSR layout")
    batch, sequence = query_latent.shape[:2]
    if batch < 1 or sequence < 1:
        raise ValueError("MLA CP requires nonempty batches and sequence shards")
    for tensor, width in ((query_latent, dimensions.q_rank), (kv_latent, dimensions.kv_rank),
                          (key_rope, dimensions.rope_dim)):
        if (tensor.shape != (batch, sequence, width) or tensor.device != query_latent.device
                or tensor.dtype != query_latent.dtype):
            raise ValueError("MLA CP latent shapes/devices/dtypes do not match the plan")
    if module.training and module.attention_dropout != 0:
        raise ValueError("MLA CP requires attention dropout=0")
    plan.validate_scale(module.scaling)
    if plan.backend == "npu":
        if query_latent.device.type != "npu" or query_latent.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("NPU MLA CP requires NPU BF16/FP16 activations")
        if dimensions.value_dim > dimensions.qk_dim:
            raise ValueError("NPU MLA CP requires value_dim <= qk_dim")


class MLAContextParallel:
    """Run CP over normalized latents; parameter-gradient reduction belongs to the trainer."""

    def __init__(self, plan: MLACPPlan, cp_mesh: Any) -> None:
        """Bind a validated execution plan to the framework-owned CP mesh."""
        if (1 if cp_mesh is None else cp_mesh.size()) != plan.cp_degree:
            raise ValueError("MLACPPlan degree must match the CP mesh")
        self.plan = plan
        self.cp_mesh = cp_mesh

    def __call__(
        self,
        module: nn.Module,
        query_latent: torch.Tensor,
        kv_latent: torch.Tensor,
        key_rope: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        actual_seq_len: torch.Tensor | Sequence[int] | None = None,
        is_causal: bool | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        """Return local-token BSND attention output before the shared o_proj."""
        packed_kwargs = {name: kwargs.pop(name) for name in PACKED_SEQUENCE_ARGUMENTS if name in kwargs}
        packed_kwargs["actual_seq_len"] = actual_seq_len
        _validate_forward_kwargs(kwargs, position_embeddings)
        state = MLAState(query_latent, kv_latent, key_rope)
        dimensions, degree = self.plan.dimensions, self.plan.cp_degree
        _validate_state(module, self.plan, state)
        batch, sequence = query_latent.shape[:2]
        lengths, key_lengths = resolve_packed_sequence_lengths(
            packed_kwargs, batch * sequence * degree, batch * sequence * degree,
        )
        if lengths is not None and batch != 1:
            raise ValueError("packed MLA CP currently requires batch_size=1")
        if lengths != key_lengths:
            raise ValueError("MLA Ulysses requires matching global query and key document boundaries")
        causal = getattr(module, "is_causal", True) if is_causal is None else is_causal
        if not isinstance(causal, bool):
            raise ValueError("is_causal must be bool")
        _validate_mask(attention_mask, lengths, batch, sequence * degree, query_latent.device)
        positions = _positions(position_embeddings, state, dimensions.rope_dim)
        query, rotated_key = _query_and_key_rope(module, state, positions, self.plan.local_heads)
        if self.plan.strategy == "expanded_ulysses" or degree == 1:
            query, key, value = self._expanded(module, state, query, rotated_key)
        else:
            query, key, value = self._latent(module, state, query, rotated_key)
        output = _run_attention(
            module, query, key, value, backend=self.plan.backend, lengths=lengths,
            causal=causal, attention_mask=attention_mask,
        )
        if degree > 1:
            output = ulysses_head_to_seq(output, 1, 2, self.cp_mesh)
        expected = (batch, sequence, self.plan.local_heads, dimensions.value_dim)
        if tuple(output.shape) != expected:
            raise RuntimeError(f"MLA CP output has shape {tuple(output.shape)}, expected {expected}")
        return output, None

    def _expanded(self, module, state, query, key_rope):
        """Pack unequal-width QKV into one sequence-to-head exchange."""
        key, value = _key_value(module, state.kv_latent, key_rope, self.plan.local_heads, None)
        if self.plan.cp_degree == 1:
            return query, key, value
        widths = (query.shape[-1], key.shape[-1], value.shape[-1])
        # Keep projection outputs in BSND while packing to reduce layout copies.
        # Source-rank/sequence reconstruction is also a view when B=1.
        payload = torch.cat(tuple(tensor.transpose(1, 2) for tensor in (query, key, value)), dim=-1)
        payload = ulysses_seq_to_head(payload, 1, 2, self.cp_mesh)
        return tuple(tensor.transpose(1, 2) for tensor in payload.split(widths, dim=-1))

    def _latent(self, module, state, query, key_rope):
        """Exchange Q heads, gather shared KV latents, then expand only owned heads."""
        dimensions = self.plan.dimensions
        heads = self.plan.compute_heads
        rank = 0 if self.cp_mesh is None else self.cp_mesh.get_local_rank()
        start = rank * heads
        if self.plan.cp_degree > 1:
            query = ulysses_seq_to_head(query, 2, 1, self.cp_mesh)
        payload = mla_all_gather(torch.cat((state.kv_latent, key_rope), dim=-1), 1, self.cp_mesh)
        kv_latent, key_rope = payload.split((dimensions.kv_rank, dimensions.rope_dim), dim=-1)
        # A contiguous latent preserves the native Linear bias-fusion path after payload splitting.
        key, value = _key_value(module, kv_latent.contiguous(), key_rope, heads, (start, start + heads))
        return query, key, value

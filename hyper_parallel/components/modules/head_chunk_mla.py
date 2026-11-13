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
"""Opt-in head-chunk MLA with explicit model-owned integration."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from hyper_parallel.core.activation_memory import PinnedMemoryPool
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.components.functional.npu_fusion_attention import resolve_packed_sequence_lengths
from ._head_chunk_mla_ops import _NpuAttention, _EagerAttention, _PackedNpuAttention
from ._head_chunk_mla_math import _COMPRESSED_COUNTERS, _LatentExecution, head_stage_schedule
from ._head_chunk_mla_offload import _OffloadStatistics
from ._head_chunk_mla_autograd import _HeadChunkMLAFunction


class HeadChunkMLA(nn.Module):
    """Opt-in head-streamed standard MLA with compressed KV communication.

    Args:
        module: Constructed native MLAAttention. Its parameters are reused.
        head_chunk_size: Global heads per stage; must be divisible by CP size.
        cp_mesh: One-dimensional contiguous-sequence Ulysses mesh, or None.
        offload_saved_stats: Store native FA output/max/sum in bounded pinned memory.
        prefetch_slots: One or two invocation-local saved-state device slots.
        max_host_bytes: Pinned memory capacity limit per module, including live forwards.
        attention_backend: Native MLA ``npu`` FA or CPU-only diagnostic ``eager``.
        overlap_backward: Overlap dO/dQ communication with independent projection VJPs.

    Note:
        Bias-free, causal, half-RoPE, zero-dropout full-sequence training only.
        Apply before FSDP. Parameter reduction must cover CP ranks; standalone
        replicated-parameter CP training is not implemented. Do not wrap child
        projections in FSDP/TP, or stack a second CP wrapper. Exclude this module
        from outer checkpoint replay. Offload supports one backward per forward.
        BF16 head grouping changes reduction order; this is not bitwise native MLA.
    """

    def __init__(
        self, *, module: nn.Module, head_chunk_size: int, cp_mesh: Any = None,
        offload_saved_stats: bool = False, prefetch_slots: int = 2,
        max_host_bytes: int = 8 * 1024**3, attention_backend: str = "npu",
        overlap_backward: bool = False,
    ) -> None:
        """Reuse native MLA parameters and validate the supported execution contract."""
        super().__init__()
        if getattr(module, "use_output_gate", False) or hasattr(module, "g_proj"):
            raise ValueError("gated MLA requires a separate stage-local output VJP")
        if getattr(module, "mla_cp_runtime", None) is not None:
            raise ValueError("HeadChunkMLA owns CP; do not stack a second CP wrapper")
        if not isinstance(overlap_backward, bool):
            raise ValueError("overlap_backward must be boolean")
        if cp_mesh is not None and cp_mesh.ndim != 1:
            raise ValueError("HeadChunkMLA requires a one-dimensional CP mesh")
        required = ("linear_qkv", "q_a_layernorm", "kv_a_layernorm", "q_b_proj", "kv_b_proj", "o_proj")
        missing = [name for name in required if not hasattr(module, name)]
        if missing:
            raise ValueError(f"source must be a constructed MLAAttention; missing children: {missing}")
        if attention_backend not in ("npu", "eager"):
            raise ValueError("attention_backend must be npu or eager")
        if not isinstance(offload_saved_stats, bool):
            raise ValueError("offload_saved_stats must be a boolean")
        if isinstance(prefetch_slots, bool) or prefetch_slots not in (1, 2):
            raise ValueError("prefetch_slots must be 1 or 2")
        if isinstance(max_host_bytes, bool) or not isinstance(max_host_bytes, int) or max_host_bytes <= 0:
            raise ValueError("max_host_bytes must be a positive integer")
        dropout = getattr(module, "attention_dropout", 0.0)
        dropout = dropout.p if isinstance(dropout, nn.Dropout) else dropout
        if dropout or not getattr(module, "is_causal", True):
            raise ValueError("experimental MLA requires zero attention dropout and causal attention")
        if getattr(module, "sliding_window", None) is not None or getattr(module, "rotary_interleaved", False):
            raise ValueError("sliding-window and interleaved RoPE are not supported")
        if getattr(module, "use_mome", False) or getattr(module, "param_sink_number", 0):
            raise ValueError("only standard MLA without MOME or parameter sinks is supported")
        for name in required:
            child = getattr(module, name)
            if getattr(child, "bias", None) is not None:
                raise ValueError(f"bias is unsupported: {name}")
            if set(dict(child.named_parameters())) != {"weight"} or dict(child.named_buffers()):
                raise ValueError(f"{name} must expose only a weight parameter and no buffers")
            setattr(self, name, child)
        for name in ("num_heads", "q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim"):
            value = getattr(module, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            setattr(self, name, value)
        if self.qk_rope_head_dim % 2:
            raise ValueError("half-RoPE requires an even qk_rope_head_dim")
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.scaling = float(module.scaling)
        if not math.isfinite(self.scaling) or self.scaling <= 0:
            raise ValueError("scaling must be finite and positive")
        self.config = getattr(module, "config", None)
        self.layer_idx = getattr(module, "layer_idx", None)
        self.head_chunk_size, self.head_chunk_cp_mesh = head_chunk_size, cp_mesh
        self.offload_saved_stats, self.prefetch_slots = offload_saved_stats, prefetch_slots
        self.overlap_backward = overlap_backward
        self.saved_stats_host_limit_bytes = max_host_bytes
        self.attention_backend = attention_backend
        self._backend = _NpuAttention if attention_backend == "npu" else _EagerAttention
        self._saved_stats_pool = None
        self._saved_stats_statistics = _OffloadStatistics()
        self._compressed_statistics = {name: 0 for name in _COMPRESSED_COUNTERS}
        self._validate_projection_shapes()
        self._validate_parameter_contract()
        head_stage_schedule(self.num_heads, head_chunk_size, 1 if cp_mesh is None else cp_mesh.size())
        self.train(module.training)

    def _validate_parameter_contract(self) -> None:
        # Manual sliced GEMMs bypass child forward hooks. A sharded or custom
        # projection cannot be substituted without changing that VJP contract.
        for name in ("linear_qkv", "q_b_proj", "kv_b_proj", "o_proj",
                     "q_a_layernorm", "kv_a_layernorm"):
            child = getattr(self, name)
            if getattr(child, "hsdp_scheduler", None) is not None or isinstance(child.weight, DTensor):
                raise ValueError("HeadChunkMLA does not support child FSDP or TP/DTensor parameters")
            if name not in ("q_a_layernorm", "kv_a_layernorm") and type(child) is not nn.Linear:
                raise ValueError(f"{name} must be an unmodified torch.nn.Linear")

    def _validate_projection_shapes(self) -> None:
        hidden_size = self.linear_qkv.weight.shape[1]
        expected = {
            "linear_qkv": (self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim, hidden_size),
            "q_a_layernorm": (self.q_lora_rank,), "kv_a_layernorm": (self.kv_lora_rank,),
            "q_b_proj": (self.num_heads * self.qk_head_dim, self.q_lora_rank),
            "kv_b_proj": (self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), self.kv_lora_rank),
            "o_proj": (hidden_size, self.num_heads * self.v_head_dim),
        }
        for name, shape in expected.items():
            actual = tuple(getattr(self, name).weight.shape)
            if actual != shape:
                raise ValueError(f"{name}.weight shape {actual} does not match standard MLA {shape}")

    def saved_stats_offload_statistics(self) -> dict[str, int]:
        """Return integer-only resource counters, without retaining any graph."""
        result = self._saved_stats_statistics.snapshot()
        pool = self._saved_stats_pool
        result.update(pool_capacity_bytes=0 if pool is None else pool.total_allocated,
                      pool_limit_bytes=self.saved_stats_host_limit_bytes)
        return result

    def compressed_kv_statistics(self) -> dict[str, int]:
        """Return scalar-only collective counts and logical tensor bytes."""
        return dict(self._compressed_statistics)

    def forward(
        self, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None, past_key_values: Any = None,
        actual_seq_len: Any = None, **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        """Run one FA per stage, with optional global packed document ends.

        Packed execution requires batch one, native NPU FA and cumulative
        document ends covering the global CP sequence (without leading zero).
        Position embeddings must already use document-local positions.

        Args:
            hidden_states: Local sequence shard in BSH layout.
            position_embeddings: Local cosine and sine, or None for no RoPE.
            attention_mask: Unsupported explicit mask; must be None.
            past_key_values: Unsupported cache; must be None.
            actual_seq_len: Global cumulative packed document ends, or None.
            **kwargs: Unsupported extra attention operands are rejected.

        Returns:
            Local projected output and None for attention weights.
        """
        self._validate_parameter_contract()
        if attention_mask is not None or past_key_values is not None or kwargs:
            raise ValueError("experimental MLA does not support masks, cache, or extra kwargs")
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.linear_qkv.weight.shape[1]:
            raise ValueError("hidden_states must have shape [batch, local_sequence, hidden_size]")
        if hidden_states.shape[0] <= 0 or hidden_states.shape[1] <= 0:
            raise ValueError("empty batch or sequence is unsupported")
        if self.attention_backend == "eager" and hidden_states.device.type not in ("cpu", "meta"):
            raise ValueError("eager is a CPU-only diagnostic backend, not an accelerator fallback")
        if self.attention_backend == "npu" and hidden_states.device.type != "npu":
            raise ValueError("npu requires NPU inputs")
        if position_embeddings is None:
            cos, sin = hidden_states.new_empty((0,)), hidden_states.new_empty((0,))
        else:
            if len(position_embeddings) != 2:
                raise ValueError("position_embeddings must contain cos and sin")
            cos, sin = position_embeddings
            if cos.shape != sin.shape or cos.ndim not in (2, 3):
                raise ValueError("cos and sin must have identical [S, D] or [B, S, D] shapes")
            if cos.shape[-2:] != (hidden_states.shape[1], self.qk_rope_head_dim):
                raise ValueError("cos/sin must cover the local sequence and exactly qk_rope_head_dim")
            if cos.ndim == 3 and cos.shape[0] not in (1, hidden_states.shape[0]):
                raise ValueError("cos/sin batch dimension must be 1 or match hidden_states")
            if cos.requires_grad or sin.requires_grad:
                raise ValueError("learnable position embeddings are not supported")
            if cos.device != hidden_states.device or sin.device != hidden_states.device:
                raise ValueError("cos/sin and hidden_states must reside on the same device")
        weights = (self.linear_qkv.weight, self.q_b_proj.weight, self.kv_b_proj.weight,
                   self.o_proj.weight, self.q_a_layernorm.weight, self.kv_a_layernorm.weight)
        retain = torch.is_grad_enabled() and (
            hidden_states.requires_grad or any(weight.requires_grad for weight in weights)
        )
        if self.offload_saved_stats and retain and hidden_states.device.type != "npu":
            raise ValueError("saved-state offload requires NPU execution")
        pool = None
        if self.offload_saved_stats and retain:
            if self._saved_stats_pool is None:
                self._saved_stats_pool = PinnedMemoryPool(self.saved_stats_host_limit_bytes)
            pool = self._saved_stats_pool
        mesh = self.head_chunk_cp_mesh
        backend = self._backend
        if actual_seq_len is not None:
            if hidden_states.shape[0] != 1 or self.attention_backend != "npu":
                raise ValueError("packed MLA requires batch=1 and native NPU FA")
            total = hidden_states.shape[1] * (1 if mesh is None else mesh.size())
            lengths, _ = resolve_packed_sequence_lengths({"actual_seq_len": actual_seq_len}, total, total)
            backend = _PackedNpuAttention(lengths)
        execution = _LatentExecution(
            head_stage_schedule(self.num_heads, self.head_chunk_size, 1 if mesh is None else mesh.size()),
            self.q_lora_rank, self.kv_lora_rank, self.qk_nope_head_dim, self.qk_rope_head_dim, self.v_head_dim,
            self.scaling, self.q_a_layernorm, self.kv_a_layernorm, self.attention_backend, backend,
            mesh, retain, pool, self.prefetch_slots, self._saved_stats_statistics,
            retire_d2h_sources=self.offload_saved_stats,
            compressed_statistics=self._compressed_statistics,
            overlap_backward=self.overlap_backward,
        )
        return _HeadChunkMLAFunction.apply(hidden_states, *weights, cos, sin, execution), None


__all__ = ["HeadChunkMLA"]

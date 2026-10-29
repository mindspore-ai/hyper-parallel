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
"""Shared-KV Head Chunk GQA with packed parameters and saved-stat FA backward."""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from hyper_parallel.components.modules._head_chunk_shared_kv import (
    SharedKVHeadChunkFunction,
    _Execution,
    _device_plan,
    shared_kv_head_schedule,
)
from hyper_parallel.components.modules._head_chunk_saved_stats import _OffloadStatistics
from hyper_parallel.components.modules.gqa_attention import GQAAttention
from hyper_parallel.components.modules.rms_norm import RMSNorm
from hyper_parallel.core.activation_memory import PinnedMemoryPool
from hyper_parallel.models.replacement import module_replacement


def _norm_contract(norm: nn.Module | None, name: str, head_dim: int) -> tuple[str, float]:
    """Accept only explicitly verified direct-scale RMSNorm implementations."""
    # A native Qwen3 instance necessarily loaded its defining module already.
    # Do not import optional Transformers model dependencies for HP RMSNorm.
    qwen_module = sys.modules.get("transformers.models.qwen3.modeling_qwen3")
    qwen_norm_type = getattr(qwen_module, "Qwen3RMSNorm", None)
    norm_type = type(norm)
    if norm_type is RMSNorm:
        kind, eps = "fused", norm.eps
    elif qwen_norm_type is not None and norm_type is qwen_norm_type:
        kind, eps = "qwen3", norm.variance_epsilon
    else:
        raise ValueError(
            f"Head Chunk {name} norm requires exact HP RMSNorm or native Qwen3RMSNorm; "
            "LayerNorm, offset weights and custom normalization are unsupported"
        )
    if getattr(norm.forward, "__func__", None) is not norm_type.forward:
        raise ValueError(f"Head Chunk {name} norm does not support an overridden forward")
    if norm._forward_hooks or norm._forward_pre_hooks:
        raise ValueError(f"Head Chunk {name} norm does not support forward hooks")
    if not isinstance(norm.weight, nn.Parameter) or tuple(norm.weight.shape) != (head_dim,):
        raise ValueError(f"Head Chunk {name} RMSNorm weight must have shape ({head_dim},)")
    if not isinstance(eps, (int, float)) or isinstance(eps, bool) or not math.isfinite(eps) or eps <= 0:
        raise ValueError(f"Head Chunk {name} RMSNorm epsilon must be finite and positive")
    return kind, float(eps)


@module_replacement
class HeadChunkGQAAttention(GQAAttention):
    """Compute GQA in serial query stages sharing each physical KV group.

    This is a standalone, opt-in NPU module. It preserves the master's packed
    ``linear_qkv`` layout and inherits ``make_transforms`` from GQAAttention.
    Construction builds structure only: apply those checkpoint transforms to
    load actual source Q/K/V weights before comparing or training the model.

    Only continuous causal sequences, BF16/FP16, bias-free projections,
    zero dropout and full-head half RoPE are supported. CP uses native GQA
    Ulysses without repeating KV heads. No Trainer or recipe is installed.
    Backward is first-order only. The default device-state path supports
    retained-graph reuse; saved-state offload supports one backward per
    forward and promptly releases its leases after that backward. All
    parameter gradients return through autograd.
    """

    def __init__(
        self,
        *,
        module: nn.Module,
        module_fqn: str = "",
        context: Mapping[str, Any] | None = None,
        head_chunk_size: int,
        offload_saved_stats: bool = False,
        saved_stats_prefetch_slots: int = 2,
        saved_stats_host_limit_bytes: int = 8 * 1024**3,
        pipeline_output_a2a: bool = False,
        pipeline_dq_a2a: bool = False,
        early_saved_stats: bool = False,
        trim_a2a_stream_records: bool = False,
        retire_d2h_sources: bool = False,
        release_a2a_source: bool = False,
    ) -> None:
        """Build a shared-KV replacement with bounded query-stage workspaces.

        Args:
            module: Transformers-style source GQA with verified direct-scale
                HP RMSNorm or native Qwen3RMSNorm for Q and K.
            module_fqn: Fully qualified source name supplied by replacement.
            context: Model-construction context; tensor parallelism is unsupported.
            head_chunk_size: Maximum global query heads per stage. Tail stages
                are allowed. Under CP, this count and physical KV heads must
                each be divisible by the CP degree.
            offload_saved_stats: Move each stage's FA output/max/sum to pinned
                host storage. This mode supports one backward per forward.
            saved_stats_prefetch_slots: One for demand loading, or two for
                bounded one-stage look-ahead; no cross-layer prefetch.
            saved_stats_host_limit_bytes: Hard pinned-pool capacity shared by
                all outstanding forwards of this module, including cached
                buffers and allocation alignment. It is not a model-wide limit.
            pipeline_output_a2a: In each KV group, prelaunch next Q and retire
                current O after the next FA. At most two transport tickets.
            pipeline_dq_a2a: Retire inverse-dQ after the next FA-grad, keeping
                at most two local query graphs. Does not change gradient add order.
            early_saved_stats: With offload enabled, submit D2H immediately
                after FA and double-slot next H2D before current query replay.
                Single-slot loading still waits for the current consumer.
            trim_a2a_stream_records: On private NPU pipeline transports, omit
                redundant post-wait records when launch and wait streams match.
                Stage storage ownership and subsequent same-stream consumers
                follow the internal call-site contract; allocation-stream
                ownership is not introspected. Cross-stream waits retain the
                original records. Packing and collective order are unchanged.
            retire_d2h_sources: With offload enabled, retire each private FA
                source after a one-stage-delayed copy-completion wait. Clear
                the final source before returning, avoiding cross-layer holds.
            release_a2a_source: On private NPU transports, stop holding the
                source after packing into independent send storage. Aliased
                layouts and identity transports retain their source ownership.

        Raises:
            ValueError: If the source or head geometry violates this contract.
        """
        for name, value in (("offload_saved_stats", offload_saved_stats),
                            ("pipeline_output_a2a", pipeline_output_a2a),
                            ("pipeline_dq_a2a", pipeline_dq_a2a),
                            ("early_saved_stats", early_saved_stats),
                            ("trim_a2a_stream_records", trim_a2a_stream_records),
                            ("retire_d2h_sources", retire_d2h_sources),
                            ("release_a2a_source", release_a2a_source)):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean")
        if early_saved_stats and not offload_saved_stats:
            raise ValueError("early_saved_stats requires offload_saved_stats=True")
        if retire_d2h_sources and not offload_saved_stats:
            raise ValueError("retire_d2h_sources requires offload_saved_stats=True")
        if (isinstance(saved_stats_prefetch_slots, bool)
                or not isinstance(saved_stats_prefetch_slots, int)
                or saved_stats_prefetch_slots not in (1, 2)):
            raise ValueError("saved_stats_prefetch_slots must be one or two")
        if (isinstance(saved_stats_host_limit_bytes, bool)
                or not isinstance(saved_stats_host_limit_bytes, int)
                or saved_stats_host_limit_bytes <= 0):
            raise ValueError("saved_stats_host_limit_bytes must be a positive integer")
        if context is not None and context.get("tp", False):
            raise ValueError("Head Chunk GQA does not support tensor parallelism")
        super().__init__(module=module, module_fqn=module_fqn, context=context)
        self.head_chunk_size = head_chunk_size
        self.head_chunk_cp_mesh = None
        self.offload_saved_stats = offload_saved_stats
        self.pipeline_output_a2a = pipeline_output_a2a
        self.pipeline_dq_a2a = pipeline_dq_a2a
        self.early_saved_stats = early_saved_stats
        self.trim_a2a_stream_records = trim_a2a_stream_records
        self.retire_d2h_sources = retire_d2h_sources
        self.release_a2a_source = release_a2a_source
        self.saved_stats_prefetch_slots = saved_stats_prefetch_slots
        self.saved_stats_host_limit_bytes = saved_stats_host_limit_bytes
        self._saved_stats_pool: PinnedMemoryPool | None = None
        self._saved_stats_statistics: _OffloadStatistics | None = None
        self._shared_kv_plans: dict[tuple[int, str], tuple] = {}
        self._validate_configuration()
        self.query_norm_kind, self.query_norm_eps = _norm_contract(self.q_norm, "Q", self.qk_head_dim)
        self.key_norm_kind, self.key_norm_eps = _norm_contract(self.k_norm, "K", self.qk_head_dim)

    def saved_stats_offload_statistics(self) -> dict[str, int]:
        """Return integer-only per-module capacity and transfer counters.

        The pool retains reusable pinned buffers up to its configured limit.
        Live host leases and device slots return to zero after backward or
        after an unused graph is destroyed; cached pool capacity may remain.
        """
        counters = self._saved_stats_statistics
        result = _OffloadStatistics().snapshot() if counters is None else counters.snapshot()
        result.update(
            pool_capacity_bytes=0 if self._saved_stats_pool is None else self._saved_stats_pool.total_allocated,
            pool_limit_bytes=self.saved_stats_host_limit_bytes,
        )
        return result

    def _validate_configuration(self) -> None:
        shared_kv_head_schedule(self.num_heads, self.num_key_value_heads, self.head_chunk_size, 1)
        if self.linear_qkv.bias is not None or self.o_proj.bias is not None:
            raise ValueError("Head Chunk GQA supports bias-free projections only")
        if self.attention_dropout != 0 or self.is_causal is not True:
            raise ValueError("Head Chunk GQA requires causal attention and attention_dropout=0")
        if self.sliding_window is not None:
            raise ValueError("Head Chunk GQA does not support sliding-window attention")
        if self.qk_head_dim != self.v_head_dim or self.qk_head_dim % 2:
            raise ValueError("Head Chunk GQA requires equal, even QK and V head dimensions")
        if self.rotary_interleaved:
            raise ValueError("Head Chunk GQA requires full-head half RoPE, not interleaved RoPE")
        if not math.isfinite(self.scaling) or self.scaling <= 0:
            raise ValueError("Head Chunk GQA scaling must be finite and positive")

    def _validate_position_embeddings(
        self, hidden: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> None:
        if not isinstance(position_embeddings, (tuple, list)) or len(position_embeddings) != 2:
            raise ValueError("Head Chunk GQA requires precomputed cosine and sine tensors")
        cos, sin = position_embeddings
        if not isinstance(cos, torch.Tensor) or not isinstance(sin, torch.Tensor):
            raise ValueError("Head Chunk GQA cosine and sine values must be tensors")
        if cos.shape != sin.shape or cos.ndim not in (2, 3):
            raise ValueError("Head Chunk GQA cosine and sine must have matching [S, D] or [B, S, D] shapes")
        if cos.shape[-2:] != (hidden.shape[1], self.qk_head_dim):
            raise ValueError("Head Chunk GQA requires full-head RoPE matching the local sequence length")
        if cos.ndim == 3 and cos.shape[0] not in (1, hidden.shape[0]):
            raise ValueError("Head Chunk GQA RoPE batch dimension must be one or match hidden_states")
        if cos.device != hidden.device or sin.device != hidden.device:
            raise ValueError("Head Chunk GQA RoPE tensors must share the input device")
        if cos.dtype != hidden.dtype or sin.dtype != hidden.dtype:
            raise ValueError("Head Chunk GQA RoPE tensors must share the input dtype")
        if cos.requires_grad or sin.requires_grad:
            raise ValueError("Head Chunk GQA does not return position-embedding gradients")

    def _validate_chunk_forward(
        self, hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        attention_mask: torch.Tensor | None, past_key_values: Any | None,
        actual_seq_len: torch.Tensor | Sequence[int] | None, kwargs: Mapping[str, Any],
    ) -> None:
        if hidden_states.ndim != 3 or min(hidden_states.shape) <= 0:
            raise ValueError("Head Chunk GQA requires nonempty hidden_states shaped [B, S, H]")
        if hidden_states.shape[-1] != self.linear_qkv.in_features:
            raise ValueError("Head Chunk GQA hidden size does not match the QKV projection")
        if hidden_states.device.type != "npu" or hidden_states.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("Head Chunk GQA requires Ascend NPU float16 or bfloat16 inputs")
        self._validate_position_embeddings(hidden_states, position_embeddings)
        if attention_mask is not None or past_key_values is not None or actual_seq_len is not None:
            raise ValueError("Head Chunk GQA requires continuous causal attention without masks, packed input or cache")
        allowed = {"position_ids", "cache_position", "is_causal", "output_attentions", "use_cache"}
        unsupported = [name for name, value in kwargs.items() if value is not None and name not in allowed]
        if unsupported:
            raise ValueError(f"Head Chunk GQA does not support attention arguments {unsupported}")
        if kwargs.get("is_causal") is not None and kwargs["is_causal"] is not True:
            raise ValueError("Head Chunk GQA requires is_causal=true")
        if kwargs.get("output_attentions", False) or kwargs.get("use_cache", False):
            raise ValueError("Head Chunk GQA does not support attention weights or KV cache")
        weights = (self.linear_qkv.weight, self.o_proj.weight, self.q_norm.weight, self.k_norm.weight)
        if any(weight.device != hidden_states.device or weight.dtype != hidden_states.dtype for weight in weights):
            raise ValueError("Head Chunk GQA inputs and parameters must share device and dtype")
        cp_mesh = self.head_chunk_cp_mesh
        if cp_mesh is not None and getattr(cp_mesh, "ndim", 1) != 1:
            raise ValueError("Head Chunk GQA requires a one-dimensional CP mesh")
        cp_size = 1 if cp_mesh is None else cp_mesh.size()
        shared_kv_head_schedule(self.num_heads, self.num_key_value_heads, self.head_chunk_size, cp_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        actual_seq_len: torch.Tensor | Sequence[int] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        """Run the same shared-KV schedule in training, evaluation and no-grad."""
        self._validate_chunk_forward(
            hidden_states, position_embeddings, attention_mask, past_key_values, actual_seq_len, kwargs,
        )
        cos, sin = position_embeddings
        inputs = (hidden_states, self.linear_qkv.weight, self.o_proj.weight, self.q_norm.weight, self.k_norm.weight)
        retain_for_backward = torch.is_grad_enabled() and any(tensor.requires_grad for tensor in inputs)
        pool = None
        if self.offload_saved_stats and retain_for_backward:
            if self._saved_stats_pool is None:
                self._saved_stats_pool = PinnedMemoryPool(self.saved_stats_host_limit_bytes)
                self._saved_stats_statistics = _OffloadStatistics()
            pool = self._saved_stats_pool
        execution = _Execution(
            _device_plan(self, hidden_states.device), self.qk_head_dim, self.query_norm_eps, self.key_norm_eps,
            self.query_norm_kind, self.key_norm_kind, self.scaling, self.head_chunk_cp_mesh,
            retain_for_backward=retain_for_backward,
            offload_pool=pool, prefetch_slots=self.saved_stats_prefetch_slots,
            offload_statistics=self._saved_stats_statistics if pool is not None else None,
            pipeline_output_a2a=self.pipeline_output_a2a, pipeline_dq_a2a=self.pipeline_dq_a2a,
            early_saved_stats=self.early_saved_stats,
            trim_a2a_stream_records=self.trim_a2a_stream_records,
            retire_d2h_sources=self.retire_d2h_sources,
            release_a2a_source=self.release_a2a_source,
        )
        output = SharedKVHeadChunkFunction.apply(
            *inputs, cos, sin, execution,
        )
        return output, None


__all__ = ["HeadChunkGQAAttention"]

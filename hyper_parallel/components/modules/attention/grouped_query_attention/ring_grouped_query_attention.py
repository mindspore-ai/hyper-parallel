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
"""Ring-attention GQA member for the module-family attention interface."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from hyper_parallel.components.functional.npu_fusion_attention import (
    PACKED_SEQUENCE_ARGUMENTS, resolve_packed_sequence_lengths,
)
from hyper_parallel.distributed.context_parallel.ring_attention import (
    ulysses_ring_attention,
)
from hyper_parallel.distributed.module_binding import ParallelModuleSpec
from .grouped_query_attention import GroupedQueryAttentionModule
from .grouped_query_attention_family import validate_gqa_layout


@dataclass(frozen=True)
class RingGQAInputs:
    """Ring operands and metadata after projection and before attention."""

    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    input_layout: str
    document_lengths: tuple[int, ...] | None
    causal: bool
    attention_mask: torch.Tensor | None
    options: dict[str, Any]
    output_state: Any


def _lengths_from_cumulative(value: Sequence[int]) -> tuple[int, ...]:
    """Convert cumulative packed endpoints to positive document lengths."""
    previous = 0
    result = []
    for endpoint in value:
        length = int(endpoint) - previous
        if length < 1:
            raise ValueError("Ring packed sequence lengths must be strictly increasing")
        result.append(length)
        previous = int(endpoint)
    return tuple(result)


def _validate_ring_options(
    attention_mask: torch.Tensor | None,
    attention_dropout: float,
    kwargs: dict[str, Any],
) -> None:
    """Reject attention semantics that the Ring kernel does not implement."""
    if attention_mask is not None:
        raise ValueError("Ring GQA does not support an explicit attention mask")
    if attention_dropout != 0:
        raise ValueError("Ring GQA requires attention dropout=0")
    unsupported = (
        "sliding_window", "window_size", "softcap", "alibi_slopes", "attention_bias",
    )
    for name in unsupported:
        if kwargs.get(name) is not None:
            raise ValueError(f"Ring GQA does not support {name}")
    if kwargs.get("output_attentions", False):
        raise ValueError("Ring GQA does not return attention weights")
    supported = PACKED_SEQUENCE_ARGUMENTS | {
        "document_lengths", "is_causal", "output_attentions", "scale", "scaling", "trace", "use_cache",
    }
    unknown = sorted(name for name, value in kwargs.items() if name not in supported and value is not None)
    if unknown:
        raise ValueError(f"Ring GQA does not support attention options {unknown}")
    if kwargs.get("use_cache", False):
        raise ValueError("Ring GQA does not support a KV cache")
    if "scale" in kwargs and "scaling" in kwargs and kwargs["scale"] != kwargs["scaling"]:
        raise ValueError("Ring GQA received conflicting scale and scaling values")


class RingGroupedQueryAttentionModule(GroupedQueryAttentionModule):
    """GQA projection module whose attention stage uses differentiable Ring CP."""

    def __init__(
        self,
        *,
        module: nn.Module | None = None,
        config: Any = None,
        module_fqn: str = "",
        context: Any = None,
        fuse_qkv: bool = True,
        overlap: bool = False,
        ulysses_degree: int = 1,
        load_balance: str = "none",
        balance_block_size: int | None = None,
        zigzag_min_length: int | None = None,
    ) -> None:
        """Construct GQA projections and retain Ring scheduling options."""
        del module_fqn
        super().__init__(config=config, module=module, fuse_qkv=fuse_qkv)
        if self.sliding_window is not None:
            raise ValueError("Ring GQA does not support sliding_window")
        if not isinstance(overlap, bool):
            raise ValueError("overlap must be bool")
        if not isinstance(ulysses_degree, int) or isinstance(ulysses_degree, bool) or ulysses_degree < 1:
            raise ValueError("ulysses_degree must be a positive integer")
        if load_balance not in ("none", "zigzag", "greedy", "hybrid"):
            raise ValueError("load_balance must be none, zigzag, greedy or hybrid")
        if balance_block_size is not None and (
            not isinstance(balance_block_size, int)
            or isinstance(balance_block_size, bool)
            or balance_block_size < 1
        ):
            raise ValueError("balance_block_size must be a positive integer")
        if load_balance == "hybrid":
            if not isinstance(zigzag_min_length, int) or isinstance(zigzag_min_length, bool) or zigzag_min_length < 1:
                raise ValueError("hybrid requires a positive zigzag_min_length")
        elif zigzag_min_length is not None:
            raise ValueError("zigzag_min_length requires hybrid load_balance")
        cp_size = (context or {}).get("cp_size", 1)
        if type(cp_size) is not int or cp_size < 1 or cp_size % ulysses_degree:
            raise ValueError("Ring GQA Ulysses degree must divide the CP degree")
        self.overlap = overlap
        self.ulysses_degree = ulysses_degree
        self.load_balance = load_balance
        self.balance_block_size = balance_block_size
        self.zigzag_min_length = zigzag_min_length
        self._parallel_enabled = cp_size > 1
        self.cp_context = None

    def parallel_module_spec(self) -> ParallelModuleSpec:
        """Declare Ring's explicit CP stage protocol when CP is active."""
        if not self._parallel_enabled:
            return ParallelModuleSpec("attention.gqa.ring.local", validate=validate_gqa_layout)
        return ParallelModuleSpec(
            "attention.gqa.ring",
            "cp",
            {
                "prepare_attention_inputs": ("hidden_states", "positions"),
                "exchange_attention": ("prepared",),
                "restore_attention": ("output", "output_state"),
            },
            validate=validate_gqa_layout,
        )

    def bind_parallel(self, binding: Any) -> None:
        """Borrow the framework-owned CP mesh for Ring transport."""
        if binding.cp_context is None:
            raise ValueError("Ring GQA requires a bound CP context")
        self.cp_context = binding.cp_context
        if self.cp_context.size % self.ulysses_degree:
            raise ValueError("Ring GQA Ulysses degree must divide CP degree")
        if self.num_heads % self.ulysses_degree or self.num_key_value_heads % self.ulysses_degree:
            raise ValueError("Ring GQA Ulysses degree must divide Q and KV heads")

    def _packed_lengths(
        self,
        batch: int,
        local_length: int,
        actual_seq_len: Any,
        kwargs: dict[str, Any],
    ) -> tuple[int, ...] | None:
        """Resolve global document metadata accepted by the generic GQA contract."""
        explicit = kwargs.pop("document_lengths", None)
        resolver_kwargs = dict(kwargs)
        if actual_seq_len is not None:
            resolver_kwargs.setdefault("actual_seq_len", actual_seq_len)
        global_tokens = batch * local_length * (self.cp_context.size if self.cp_context is not None else 1)
        query_lengths, key_lengths = resolve_packed_sequence_lengths(
            resolver_kwargs, query_tokens=global_tokens, key_tokens=global_tokens,
        )
        if query_lengths != key_lengths:
            raise ValueError("Ring GQA requires matching global query and KV document lengths")
        resolved = None if query_lengths is None else _lengths_from_cumulative(query_lengths)
        if explicit is not None:
            if resolved is not None and tuple(explicit) != resolved:
                raise ValueError("document_lengths conflicts with packed sequence metadata")
            resolved = tuple(explicit)
        if resolved is None:
            if any(name in kwargs for name in PACKED_SEQUENCE_ARGUMENTS):
                raise ValueError("Ring GQA could not resolve packed sequence metadata")
            return None
        if batch != 1:
            raise ValueError("Ring GQA packed attention requires batch size one")
        if not resolved or any(type(length) is not int or length < 1 for length in resolved):
            raise ValueError("document_lengths must contain positive integers")
        if sum(resolved) != global_tokens:
            raise ValueError("document_lengths must sum to the global token count")
        return resolved

    def prepare_positions(self, query_shape: tuple[int, int], positions: Any) -> Any:
        """Slice global rotary tables to this rank's source-token interval."""
        if positions is None or self.cp_context is None:
            return positions
        local_length = query_shape[1]
        global_length = local_length * self.cp_context.size
        parts = []
        for part in positions:
            if part.shape[-2] == global_length:
                part = part.narrow(-2, self.cp_context.rank * local_length, local_length)
            elif part.shape[-2] != local_length:
                raise ValueError("Ring GQA positions must cover local or global tokens")
            parts.append(part)
        return tuple(parts)

    def prepare_attention_inputs(
        self,
        hidden_states: torch.Tensor,
        positions: Any,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any = None,
        actual_seq_len: Any = None,
        **kwargs: Any,
    ) -> RingGQAInputs:
        """Project local Q/K/V and convert generic packed metadata to Ring layout."""
        if past_key_values is not None:
            raise ValueError("Ring GQA does not support a KV cache")
        _validate_ring_options(attention_mask, self.attention_dropout if self.training else 0.0, kwargs)
        query, key, value, output_state = self.project_qkv(hidden_states, positions)
        batch, local_length = query.shape[:2]
        options = dict(kwargs)
        document_lengths = self._packed_lengths(batch, local_length, actual_seq_len, options)
        for name in PACKED_SEQUENCE_ARGUMENTS:
            options.pop(name, None)
        options.pop("output_attentions", None)
        options.pop("use_cache", None)
        causal = options.pop("is_causal", self.is_causal)
        if not isinstance(causal, bool):
            raise ValueError("Ring GQA is_causal must be bool")
        if document_lengths is None:
            input_layout = "BSND"
        else:
            query, key, value = (tensor.squeeze(0).contiguous() for tensor in (query, key, value))
            input_layout = "TND"
        return RingGQAInputs(
            query, key, value, input_layout, document_lengths, causal,
            attention_mask, options, output_state,
        )

    def exchange_attention(self, prepared: RingGQAInputs) -> RingGQAInputs:
        """Keep local Q/K/V in Ring's input layout until the attention stage."""
        return prepared

    def run_attention(self, inputs: RingGQAInputs) -> tuple[torch.Tensor, None]:
        """Execute Ring or Ulysses + Ring and restore the model's BSND output."""
        cp_mesh = None if self.cp_context is None else self.cp_context.mesh
        output = ulysses_ring_attention(
            inputs.query,
            inputs.key,
            inputs.value,
            cp_mesh,
            ulysses_degree=self.ulysses_degree,
            causal=inputs.causal,
            scale=inputs.options.get("scaling", inputs.options.get("scale", self.scaling)),
            input_layout=inputs.input_layout,
            dropout_p=0.0,
            overlap=self.overlap,
            load_balance=self.load_balance,
            document_lengths=inputs.document_lengths,
            balance_block_size=self.balance_block_size,
            zigzag_min_length=self.zigzag_min_length,
            trace=inputs.options.get("trace"),
        )
        if inputs.input_layout == "TND":
            output = output.unsqueeze(0)
        return output, None

    def restore_attention(self, output: torch.Tensor, output_state: Any) -> tuple[torch.Tensor, Any]:
        """Return Ring output in the local token layout expected by output projection."""
        return output, output_state

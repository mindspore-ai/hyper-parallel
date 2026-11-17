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
"""GQA Ulysses/KV-AllGather execution over shared, differentiable CP transport."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import accumulate
from typing import Any

import torch

from hyper_parallel.components.functional.npu_fusion_attention import (
    PACKED_SEQUENCE_ARGUMENTS, resolve_packed_sequence_lengths,
)
from hyper_parallel.distributed.module_binding import ParallelModuleSpec
from .grouped_query_attention import GQAAttentionInputs, GroupedQueryAttentionModule
from .grouped_query_attention_family import validate_gqa_layout


@lru_cache(maxsize=128)
def _query_documents(lengths: tuple[int, ...], batch: int, global_length: int,
                     query_length: int, query_rank: int, causal: bool) -> tuple:
    """Map each batch-local query interval to its documents and visible KV prefixes.

    Lengths describe the global, batch-major TND sequence. A document may cross
    rank or batch boundaries: each local query fragment gets the correct KV
    prefix (or whole document for noncausal attention), not other documents.
    The returned ranges address a flattened global KV tensor. No device reads
    or S-by-S document mask are needed.
    """
    ranges, query_counts, key_counts = [], [], []
    starts = (0, *lengths[:-1])
    for batch_index in range(batch):
        query_begin = batch_index * global_length + query_rank * query_length
        query_end = query_begin + query_length
        for start, end in zip(starts, lengths):
            begin, stop = max(start, query_begin), min(end, query_end)
            if begin < stop:
                key_end = stop if causal else end
                ranges.append((start, key_end))
                query_counts.append(stop - begin)
                key_counts.append(key_end - start)
    return tuple(ranges), tuple(accumulate(query_counts)), tuple(accumulate(key_counts))


def _select_document_keys(tensor: torch.Tensor, ranges: tuple) -> torch.Tensor:
    """Select batch-major KV document fragments while autograd sums reused prefixes."""
    flat = tensor.flatten(0, 1)
    # Preserve document boundaries in FA metadata without copying a complete
    # contiguous KV span. Keep partial multi-document spans compact: a view
    # would otherwise retain the entire gathered storage until FA backward.
    if (ranges[0][0] == 0 and ranges[-1][1] == flat.shape[0]
            and all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))):
        return flat.unsqueeze(0)
    pieces = [flat.narrow(0, begin, end - begin) for begin, end in ranges]
    return (pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)).unsqueeze(0)


def _local_mask(mask: torch.Tensor, global_length: int, query_length: int,
                query_begin: int, key_end: int) -> torch.Tensor:
    """Slice global Q/K masks; two-dimensional non-square masks are batch padding."""
    if mask.shape[-1] != global_length:
        raise ValueError("GQA CP mask must cover the global key sequence")
    if mask.ndim == 2 and mask.shape == (global_length, global_length):
        mask = mask[None, None]
    if mask.ndim >= 3 and mask.shape[-2] not in (1, query_length):
        if mask.shape[-2] != global_length:
            raise ValueError("GQA CP query mask must cover the local or global query interval")
        mask = mask[..., query_begin:query_begin + query_length, :]
    return mask[..., :key_end]


def _valid_query_rows(mask: torch.Tensor | None, batch: int, global_length: int,
                      query_length: int, query_rank: int) -> torch.Tensor | None:
    """Restore query padding after CP makes the Q and KV lengths unequal.

    The common FA can infer padded queries from a self-attention mask only
    when Q/K cover the same tokens. Here the CP rank owns a query interval
    inside global KV; slice its rows explicitly after FA.
    """
    if mask is None or mask.ndim != 2 or mask.shape == (global_length, global_length):
        return None
    if mask.shape != (batch, global_length):
        raise ValueError("GQA CP padding mask must have shape [batch, global_sequence]")
    begin = query_rank * query_length
    return mask[:, begin:begin + query_length].to(torch.bool)


@dataclass(frozen=True)
class CPGQAAttentionInputs(GQAAttentionInputs):
    """Keep query ownership and padding attached to one attention invocation."""

    query_shape: tuple[int, int]
    valid_queries: torch.Tensor | None


class CPGroupedQueryAttentionModule(GroupedQueryAttentionModule):
    """GQA CP member sharing transport across fixed projection representations.

    The inherited algorithm calls class-defined input and output stages.
    Projection math and the attention kernel are shared with the local member.
    """

    def __init__(self, *, module: torch.nn.Module | None = None, config: Any = None,
                 module_fqn: str = "", context: Any = None,
                 fuse_qkv: bool = True, overlap: bool = False, ulysses_degree: int | None = None) -> None:
        """Keep CP scheduling options on the parallel member, outside local GQA math."""
        del module_fqn, context
        super().__init__(config=config, module=module, fuse_qkv=fuse_qkv)
        self.overlap = overlap
        self.ulysses_degree = ulysses_degree

    def parallel_module_spec(self) -> ParallelModuleSpec:
        """Request borrowed CP resources in addition to the shared TP boundary."""
        return ParallelModuleSpec("attention.gqa.cp", "cp",
                                  {"prepare_attention_inputs": ("hidden_states",),
                                   "restore_attention": ("output", "inputs")},
                                  validate=validate_gqa_layout)

    def bind_parallel(self, binding: Any) -> None:
        """Bind U/A axes once and validate TP-local KV-head divisibility.

        Args:
            binding: Final module layout and the existing job's TP/CP resources.
        """
        if binding.cp_context is None:
            raise ValueError("GQA CP requires a bound CPContext")
        self.cp_context = binding.cp_context
        degree = self.cp_context.size
        self.ulysses_degree = degree if self.ulysses_degree is None else self.ulysses_degree
        if self.ulysses_degree < 1 or degree % self.ulysses_degree:
            raise ValueError("GQA Ulysses degree must divide CP degree")
        if self.num_heads % self.ulysses_degree or self.num_key_value_heads % self.ulysses_degree:
            raise ValueError("GQA Ulysses degree must divide TP-local Q and KV heads; use KV AG or a smaller U")
        self.ulysses_context, self.query_context = self.cp_context.split(self.ulysses_degree)
        self.all_gather_degree = degree // self.ulysses_degree

    def _prepare_distributed_qkv(self, hidden_states: torch.Tensor, positions: Any) -> tuple:
        """Keep identical projection math for sync/overlap; only launch/wait placement differs."""
        pending, tensors = [], []
        output_state = None
        for index, (tensor, output_state) in enumerate(self.projection_stages(hidden_states, positions)):
            if self.ulysses_degree > 1:
                if self.overlap:
                    pending.append(self.ulysses_context.launch_sequence_to_head(tensor))
                else:
                    tensor = self.ulysses_context.sequence_to_head(tensor)
            elif self.overlap and self.all_gather_degree > 1 and index > 0:
                pending.append(self.query_context.launch_gather_sequence(tensor).collective)
            tensors.append(tensor)
        return *self._finish_input_transfers(tensors, pending), output_state

    def _finish_input_transfers(self, tensors: list, pending: list) -> list:
        """Start ready hybrid gathers before waiting for unrelated branches."""
        if self.overlap:
            if self.ulysses_degree > 1:
                if self.all_gather_degree > 1:
                    # A ready K/V can enter its AG group before the other Ulysses
                    # results are reconstructed. Q is needed only at the FA boundary.
                    key = self.query_context.launch_gather_sequence(pending[1].wait())
                    value = self.query_context.launch_gather_sequence(pending[2].wait())
                    tensors = [pending[0].wait(), key.wait()[0], value.wait()[0]]
                else:
                    tensors = [handle.wait() for handle in pending]
            elif self.all_gather_degree > 1:
                tensors[1:] = [handle.wait() for handle in pending]
        elif self.all_gather_degree > 1:
            tensors[1:] = [self.query_context.gather_sequence(tensor)[0] for tensor in tensors[1:]]
        return tensors

    def _prepare_document_inputs(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                                 attention_mask: torch.Tensor | None, lengths: Any, options: dict,
                                 output_state: Any) -> CPGQAAttentionInputs:
        """Convert global packed metadata into the actual FA query/KV document fragments."""
        batch, query_length = query.shape[:2]
        global_length = key.shape[1]
        query_rank = self.query_context.rank if self.all_gather_degree > 1 else 0
        causal = options.pop("is_causal", self.is_causal)
        window = options.pop("sliding_window", self.sliding_window)
        packed_options = {}
        valid_queries = _valid_query_rows(attention_mask, batch, global_length, query_length, query_rank)
        if lengths is not None and self.all_gather_degree == 1:
            # Pure Ulysses already owns the complete sequence. Do not repack KV
            # by document: FA can consume the original batch-major TND directly.
            packed_options = {"actual_seq_len": lengths}
        elif lengths is not None or (self.all_gather_degree > 1 and attention_mask is None):
            if attention_mask is not None:
                raise ValueError("GQA CP packed attention requires document lengths without an explicit mask")
            documents = tuple(lengths or range(global_length, (batch + 1) * global_length, global_length))
            ranges, q_lengths, kv_lengths = _query_documents(
                documents, batch, global_length, query_length, query_rank, causal)
            query = query.flatten(0, 1).unsqueeze(0)
            key, value = (_select_document_keys(tensor, ranges) for tensor in (key, value))
            packed_options = {"actual_seq_qlen": q_lengths, "actual_seq_kvlen": kv_lengths}
        elif attention_mask is not None:
            query_begin = query_rank * query_length
            # Prefix truncation lets the common FA use its ordinary right-aligned causal contract.
            key_end = query_begin + query_length if causal else global_length
            key, value = key[:, :key_end], value[:, :key_end]
            attention_mask = _local_mask(attention_mask, global_length, query_length, query_begin, key_end)
            if query_length != key_end:
                # The explicit mask already includes the global query offset.
                # Kernel-local next_tokens=0 can incorrectly clip unequal-Q/K backward.
                options.setdefault("next_tokens", 2147483647)
        return CPGQAAttentionInputs(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), attention_mask,
            dict(options, is_causal=causal, sliding_window=window, **packed_options),
            output_state, (batch, query_length), valid_queries)

    def restore_attention(self, output: torch.Tensor, inputs: CPGQAAttentionInputs) -> torch.Tensor:
        """Restore packed query shape, padding, and token ownership before projection.

        Args:
            output: Attention result in its computation layout.
            inputs: This invocation's query ownership and padding metadata.
        """
        output = output.reshape(*inputs.query_shape, output.shape[-2], output.shape[-1])
        if inputs.valid_queries is not None:
            output = output.masked_fill(~inputs.valid_queries[:, :, None, None], 0)
        if self.ulysses_degree > 1:
            output = self.ulysses_context.head_to_sequence(output)
        return output

    def prepare_attention_inputs(self, hidden_states: torch.Tensor, position_embeddings: Any = None,
                                 attention_mask: torch.Tensor | None = None, past_key_values: Any = None,
                                 actual_seq_len: Any = None, **kwargs: Any) -> CPGQAAttentionInputs:
        """Prepare distributed operands while retaining the projection overlap windows.

        Args:
            hidden_states: Local sequence input with shape [B, S_local, hidden].
            position_embeddings: Local or global RoPE cosine/sine pair.
            attention_mask: Optional global padding or explicit visibility mask.
            past_key_values: Must be None for context-parallel training.
            actual_seq_len: Optional global cumulative document endpoints.
            **kwargs: Attention options and supported packed-length aliases.

        Returns:
            Final BNSD operands with this call's document and query ownership metadata.
        """
        if past_key_values is not None:
            raise ValueError("GQA CP training does not support a KV cache")
        if not hasattr(self, "cp_context"):
            raise RuntimeError("Bind GQA CP execution before forward")
        batch, local_length = hidden_states.shape[:2]
        global_length = local_length * self.cp_context.size
        options = dict(kwargs, actual_seq_len=actual_seq_len)
        q_lengths, k_lengths = resolve_packed_sequence_lengths(
            options, query_tokens=batch * global_length, key_tokens=batch * global_length)
        if q_lengths != k_lengths:
            raise ValueError("GQA CP self-attention requires matching global query/KV document lengths")
        if q_lengths is not None and attention_mask is not None:
            raise ValueError("GQA CP packed lengths cannot be combined with an explicit mask")
        options = {name: value for name, value in kwargs.items() if name not in PACKED_SEQUENCE_ARGUMENTS}
        if position_embeddings is not None:
            parts = []
            for part in position_embeddings:
                if part.shape[-2] == global_length:
                    part = part.narrow(-2, self.cp_context.rank * local_length, local_length)
                elif part.shape[-2] != local_length:
                    raise ValueError("GQA CP positions must cover the local or global token sequence")
                parts.append(part)
            position_embeddings = tuple(parts)
        query, key, value, output_state = self._prepare_distributed_qkv(hidden_states, position_embeddings)
        return self._prepare_document_inputs(query, key, value, attention_mask, q_lengths, options, output_state)

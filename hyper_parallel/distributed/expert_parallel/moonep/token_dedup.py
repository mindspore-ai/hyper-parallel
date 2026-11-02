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
"""Deduplicate token vectors by destination rank while retaining expert rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch.autograd.function import once_differentiable

from hyper_parallel.distributed.expert_parallel.collectives import ep_all_to_all


@dataclass(frozen=True)
class LocalDedup:
    """One rank's small metadata packet, prepared before collective admission."""

    packet: torch.Tensor
    packet_send: tuple[int, ...]
    packet_recv: tuple[int, ...]
    unique_tokens: torch.Tensor
    unique_send: tuple[int, ...]
    backward_tokens: torch.Tensor
    backward_send: tuple[int, ...]
    source_choices: torch.Tensor
    received_rows: int


@dataclass(frozen=True)
class TokenDedup:
    """Selection-to-vector fanout and peer splits for a saved MoonEP route."""

    unique_tokens: torch.Tensor
    unique_send: tuple[int, ...]
    unique_recv: tuple[int, ...]
    original_send_rows: int
    receive_lookup: torch.Tensor
    backward_tokens: torch.Tensor
    backward_send: tuple[int, ...]
    backward_recv: tuple[int, ...]
    backward_lookup: torch.Tensor
    source_choices: torch.Tensor
    width: int
    score_lanes: int
    group: Any
    target_hidden_rows: torch.Tensor
    target_score_elements: torch.Tensor
    source_split_rows: tuple[int, ...]
    target_split_rows: tuple[int, ...]
    source_pack_order: torch.Tensor
    source_score_pad: torch.Tensor
    backward_source_vector_rows: torch.Tensor
    backward_source_score_elements: torch.Tensor
    backward_source_split_rows: tuple[int, ...]
    backward_target_split_rows: tuple[int, ...]
    backward_target_pack_order: torch.Tensor
    backward_target_score_pad: torch.Tensor

    @property
    def saved_rows(self) -> int:
        """Count vector rows omitted from one forward transfer direction."""
        return self.original_send_rows - sum(self.unique_send)

    def _check_width(self, value: torch.Tensor) -> None:
        """Reject a route compiled for a different hidden-vector width."""
        if value.ndim != 2 or value.shape[-1] != self.width:
            raise ValueError("dedup vector width differs from the compiled peer layout")

    def _pack(self, vectors: torch.Tensor, scores: torch.Tensor,
              pack_order: torch.Tensor, score_pad: torch.Tensor) -> torch.Tensor:
        """Build peer-major payload with two gathers and contiguous concatenation."""
        score_input = torch.cat((scores, scores.new_zeros(1)))
        score_rows = score_input.index_select(0, score_pad).reshape(-1, self.width)
        return torch.cat((vectors, score_rows), dim=0).index_select(0, pack_order)

    def dispatch_with_scores(self, hidden: torch.Tensor, scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack vectors and lossless BF16/FP32 score lanes by peer."""
        self._check_width(hidden)
        return _DispatchWithScores.apply(hidden, scores, self)

    def return_vectors(self, values: torch.Tensor, tokens: int) -> torch.Tensor:
        """Reduce expert rows locally before returning one vector per source token."""
        self._check_width(values)
        return _ReturnVectors.apply(values, tokens, self)


class _DispatchWithScores(torch.autograd.Function):
    """Use the saved forward map while choosing an independent reverse layout."""

    @staticmethod
    def forward(ctx: Any, hidden: torch.Tensor, scores: torch.Tensor,
                plan: TokenDedup) -> tuple[torch.Tensor, torch.Tensor]:
        """Send unique hidden vectors and every expert's router score."""
        ctx.plan, ctx.hidden_shape, ctx.score_shape, ctx.score_dtype = plan, hidden.shape, scores.shape, scores.dtype
        vectors = hidden.index_select(0, plan.unique_tokens)
        if scores.dtype not in (torch.bfloat16, torch.float32) or max(
                1, scores.element_size() // hidden.element_size()) != plan.score_lanes:
            raise ValueError("dedup score dtype differs from the compiled peer layout")
        ctx.transport_score_dtype = hidden.dtype if scores.element_size() < hidden.element_size() else scores.dtype
        selected_scores = scores.reshape(-1).index_select(0, plan.source_choices).contiguous()
        selected_scores = (selected_scores.to(hidden.dtype) if scores.element_size() < hidden.element_size()
                           else selected_scores.view(hidden.dtype))
        payload = plan._pack(vectors, selected_scores, plan.source_pack_order, plan.source_score_pad)
        received = ep_all_to_all(payload, list(plan.source_split_rows), list(plan.target_split_rows), plan.group)
        return (received.index_select(0, plan.target_hidden_rows),
                received.reshape(-1).index_select(
                    0, plan.target_score_elements).contiguous().view(ctx.transport_score_dtype))

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_hidden: torch.Tensor, grad_scores: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor, None]:
        """Return dX using the configured reverse peer layout and preserve every dScore."""
        plan = ctx.plan
        compact = grad_hidden.new_zeros((sum(plan.backward_recv), plan.width))
        compact.index_add_(0, plan.backward_lookup, grad_hidden)
        score_lanes = grad_scores.reshape(-1).contiguous().view(grad_hidden.dtype)
        payload = plan._pack(compact, score_lanes,
                             plan.backward_target_pack_order, plan.backward_target_score_pad)
        returned = ep_all_to_all(payload, list(plan.backward_target_split_rows),
                                 list(plan.backward_source_split_rows), plan.group)
        vector_gradients = returned.index_select(0, plan.backward_source_vector_rows)
        score_gradients = returned.reshape(-1).index_select(
            0, plan.backward_source_score_elements).contiguous().view(ctx.transport_score_dtype)
        hidden_gradient = grad_hidden.new_zeros(ctx.hidden_shape)
        hidden_gradient.index_add_(0, plan.backward_tokens, vector_gradients)
        score_gradient = torch.empty(ctx.score_shape, dtype=ctx.score_dtype, device=grad_scores.device).reshape(-1)
        score_gradient.index_copy_(0, plan.source_choices, score_gradients.to(ctx.score_dtype))
        return hidden_gradient, score_gradient.reshape(ctx.score_shape), None


class _ReturnVectors(torch.autograd.Function):
    """Return weighted expert values with a reverse route independent of forward compression."""

    @staticmethod
    def forward(ctx: Any, values: torch.Tensor, tokens: int, plan: TokenDedup) -> torch.Tensor:
        """Reduce expert rows locally before returning one vector per source token."""
        ctx.plan, ctx.values_shape = plan, values.shape
        compact = values.new_zeros((sum(plan.unique_recv), values.shape[-1]))
        compact.index_add_(0, plan.receive_lookup, values)
        returned = ep_all_to_all(compact, list(plan.unique_recv), list(plan.unique_send), plan.group)
        output = values.new_zeros((tokens, values.shape[-1]))
        return output.index_add(0, plan.unique_tokens, returned)

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        """Send output gradients using the independently selected reverse split."""
        plan = ctx.plan
        payload = grad_output.index_select(0, plan.backward_tokens)
        received = ep_all_to_all(payload, list(plan.backward_send), list(plan.backward_recv), plan.group)
        return received.index_select(0, plan.backward_lookup).reshape(ctx.values_shape), None, None


def _layout_indices(vector_counts: Sequence[int], score_counts: Sequence[int], width: int,
                    *, include_pack: bool = True) -> tuple:
    """Place peer vectors and scalar tails with no per-peer device operations."""
    if width < 1 or len(vector_counts) != len(score_counts):
        raise ValueError("dedup peer layout requires a positive width and matching peer counts")
    vectors = np.asarray(vector_counts, dtype=np.int64)
    scores = np.asarray(score_counts, dtype=np.int64)
    tails = (scores + width - 1) // width
    splits = vectors + tails
    vector_offsets = np.cumsum(vectors) - vectors
    score_offsets = np.cumsum(scores) - scores
    row_offsets = np.cumsum(splits) - splits
    tail_offsets = np.cumsum(tails) - tails
    total_vectors, total_scores = int(vectors.sum()), int(scores.sum())

    if not include_pack:
        vector_peers = np.repeat(np.arange(len(vectors)), vectors)
        vector_local = np.arange(total_vectors) - np.repeat(vector_offsets, vectors)
        vector_rows = row_offsets[vector_peers] + vector_local
        score_peers = np.repeat(np.arange(len(scores)), scores)
        score_local = np.arange(total_scores) - np.repeat(score_offsets, scores)
        score_elements = (row_offsets[score_peers] + vectors[score_peers]) * width + score_local
        return vector_rows, score_elements, tuple(splits.tolist()), None, None

    output_peers = np.repeat(np.arange(len(splits)), splits)
    output_local = np.arange(int(splits.sum())) - np.repeat(row_offsets, splits)
    vector_mask = output_local < vectors[output_peers]
    pack_order = np.where(vector_mask, vector_offsets[output_peers] + output_local,
                          total_vectors + tail_offsets[output_peers] + output_local - vectors[output_peers])

    padded = tails * width
    padded_peers = np.repeat(np.arange(len(padded)), padded)
    padded_local = np.arange(int(padded.sum())) - np.repeat(np.cumsum(padded) - padded, padded)
    valid = padded_local < scores[padded_peers]
    score_pad = np.full(len(padded_local), total_scores, dtype=np.int64)
    score_pad[valid] = score_offsets[padded_peers[valid]] + padded_local[valid]
    return None, None, tuple(splits.tolist()), pack_order, score_pad


def prepare_local_dedup(order: torch.Tensor, token_send: Sequence[int], token_recv: Sequence[int],
                        top_k: int, *, source_rows: Sequence[int] | None = None,
                        forward_ratio: float = 0.75, backward_ratio: float = 1.0) -> LocalDedup:
    """Build CPU metadata without moving hidden vectors or score values.

    Args:
        order: Source selection order, each value being a token/top-k slot.
        token_send: Original per-target selection counts.
        token_recv: Original per-source selection counts.
        top_k: Router fanout per token.
        source_rows: Optional selection indices in target-major send order.
        forward_ratio: Compress forward vectors when the peer's unique fraction is at or below this value.
        backward_ratio: Compress reverse gradients when the unique fraction is below this value.
    Returns:
        A local packet to exchange after collective preparation succeeds.
    """
    ranks = len(token_send)
    if (ranks < 1 or len(token_recv) != ranks or top_k < 1
            or not 0.0 <= forward_ratio <= 1.0 or not 0.0 <= backward_ratio <= 1.0
            or any(count < 0 for count in (*token_send, *token_recv))
            or order.ndim != 1 or order.dtype != torch.int64 or sum(token_send) != order.numel()):
        raise ValueError("dedup source selection geometry does not match peer splits")
    rows = order.numel()
    if source_rows is None:
        source_index = None
    else:
        if len(source_rows) != rows:
            raise ValueError("dedup source row map must cover every selection")
        source_index = np.asarray(source_rows, dtype=np.int64)
        if not np.array_equal(np.sort(source_index), np.arange(rows, dtype=np.int64)):
            raise ValueError("dedup send order must permute the source selections")
    choices = order.detach().cpu().numpy()
    if source_index is not None:
        choices = choices[source_index]
    if np.any((choices < 0) | (choices >= rows)):
        raise ValueError("dedup selection order contains an out-of-range top-k slot")
    send_counts = np.asarray(token_send, dtype=np.int64)
    peers = np.repeat(np.arange(ranks, dtype=np.int64), send_counts)
    token_count = max(1, (rows + top_k - 1) // top_k)
    keys = peers * token_count + choices // top_k
    domain = ranks * token_count
    if domain <= 4 * max(rows, 1) and domain <= 1_048_576:
        # The key domain is small for EP16/TopK8. A direct count table keeps
        # np.unique's sorted-key and inverse semantics without an O(N log N) sort.
        present = np.bincount(keys, minlength=domain) != 0
        unique_keys = np.flatnonzero(present)
        inverse = (np.cumsum(present, dtype=np.int64) - 1)[keys]
    else:
        unique_keys, inverse = np.unique(keys, return_inverse=True)
    raw_unique = np.bincount(unique_keys // token_count, minlength=ranks)
    forward_compressed = raw_unique <= forward_ratio * send_counts
    backward_compressed = raw_unique < backward_ratio * send_counts
    unique_send = np.where(forward_compressed, raw_unique, send_counts)
    backward_send = np.where(backward_compressed, raw_unique, send_counts)
    raw_offsets = np.cumsum(raw_unique) - raw_unique
    send_offsets = np.cumsum(send_counts) - send_counts
    raw_slots = inverse - raw_offsets[peers]

    def selected_tokens(compressed: np.ndarray) -> np.ndarray:
        """Choose either the sorted unique tokens or the original selection rows per peer."""
        return np.concatenate([
            unique_keys[raw_offsets[peer]:raw_offsets[peer] + raw_unique[peer]] % token_count
            if compressed[peer] else choices[send_offsets[peer]:send_offsets[peer] + send_counts[peer]] // top_k
            for peer in range(ranks)
        ])

    unique_tokens = selected_tokens(forward_compressed)
    backward_tokens = selected_tokens(backward_compressed)
    packet_rows = np.empty(rows + ranks, dtype=np.int32)
    packet_rows[np.cumsum(send_counts + 1) - send_counts - 1] = np.where(
        forward_compressed, raw_unique, -raw_unique - 1)
    packet_rows[np.arange(rows) + peers + 1] = raw_slots
    return LocalDedup(torch.from_numpy(packet_rows), tuple(count + 1 for count in token_send),
                      tuple(count + 1 for count in token_recv),
                      torch.from_numpy(unique_tokens), tuple(unique_send.tolist()),
                      torch.from_numpy(backward_tokens), tuple(backward_send.tolist()),
                      torch.from_numpy(choices.copy()), sum(token_recv))


def failed_local_dedup(token_send: Sequence[int], token_recv: Sequence[int]) -> LocalDedup:
    """Publish an invalid header so every peer rejects a local preparation failure after one metadata exchange."""
    ranks = len(token_send)
    packet = np.zeros(sum(token_send) + ranks, dtype=np.int32)
    counts = np.asarray(token_send, dtype=np.int64)
    packet[np.cumsum(counts + 1) - counts - 1] = -1
    return LocalDedup(torch.from_numpy(packet), tuple(count + 1 for count in token_send),
                      tuple(count + 1 for count in token_recv), torch.empty(0, dtype=torch.int64),
                      tuple(0 for _ in token_send), torch.empty(0, dtype=torch.int64),
                      tuple(0 for _ in token_send), torch.empty(0, dtype=torch.int64),
                      sum(token_recv))


def finish_dedup(local: LocalDedup, group: Any, *, device: torch.device, width: int,
                 backward_ratio: float = 1.0, score_lanes: int = 1) -> TokenDedup:
    """Exchange raw indices once and compile independent forward and backward splits."""
    if not 0.0 <= backward_ratio <= 1.0:
        raise ValueError("backward dedup ratio must be between zero and one")
    if score_lanes not in (1, 2):
        raise ValueError("dedup score payload requires one or two vector-element lanes per score")
    packet = local.packet.to(device)
    received = torch.empty(sum(local.packet_recv), dtype=torch.int32, device=device)
    if len(local.unique_send) == 1:
        received.copy_(packet)
    else:
        dist.all_to_all_single(received, packet, output_split_sizes=list(local.packet_recv),
                               input_split_sizes=list(local.packet_send), group=group)
    rows = received.cpu().numpy()
    packet_counts = np.asarray(local.packet_recv, dtype=np.int64)
    peer_counts = packet_counts - 1
    header_positions = np.cumsum(packet_counts) - packet_counts
    headers = rows[header_positions].astype(np.int64)
    forward_compressed = headers >= 0
    raw_unique = np.where(forward_compressed, headers, -headers - 1)
    if np.any((headers == -1) | (raw_unique < 0) | (raw_unique > peer_counts)):
        raise RuntimeError("dedup peer header contains an invalid unique count")
    peers = np.repeat(np.arange(len(packet_counts), dtype=np.int64), peer_counts)
    raw_slots = rows[np.arange(local.received_rows) + peers + 1].astype(np.int64)
    if np.any((raw_slots < 0) | (raw_slots >= raw_unique[peers])):
        raise RuntimeError("dedup peer body contains an invalid compact index")
    peer_offsets = np.cumsum(peer_counts) - peer_counts
    dense_slots = np.arange(local.received_rows, dtype=np.int64) - peer_offsets[peers]
    forward_counts = np.where(forward_compressed, raw_unique, peer_counts)
    backward_compressed = raw_unique < backward_ratio * peer_counts
    backward_counts = np.where(backward_compressed, raw_unique, peer_counts)
    forward_offsets = np.cumsum(forward_counts) - forward_counts
    backward_offsets = np.cumsum(backward_counts) - backward_counts
    lookup = forward_offsets[peers] + np.where(forward_compressed[peers], raw_slots, dense_slots)
    backward_lookup = backward_offsets[peers] + np.where(backward_compressed[peers], raw_slots, dense_slots)
    unique_recv = tuple(forward_counts.tolist())
    backward_recv = tuple(backward_counts.tolist())
    source_rows = tuple(count - 1 for count in local.packet_send)
    target_rows = tuple(count - 1 for count in local.packet_recv)
    source_score_lanes = tuple(count * score_lanes for count in source_rows)
    target_score_lanes = tuple(count * score_lanes for count in target_rows)
    _, _, source_split_rows, source_pack_order, source_score_pad = _layout_indices(
        local.unique_send, source_score_lanes, width)
    target_vector_rows, target_score_elements, target_split_rows, _, _ = _layout_indices(
        unique_recv, target_score_lanes, width, include_pack=False)
    target_hidden_rows = np.asarray(target_vector_rows, dtype=np.int64)[lookup]
    backward_source_vector_rows, backward_source_score_elements, backward_source_split_rows, _, _ = _layout_indices(
        local.backward_send, source_score_lanes, width, include_pack=False)
    _, _, backward_target_split_rows, backward_target_pack_order, backward_target_score_pad = _layout_indices(
        backward_recv, target_score_lanes, width)
    indices = (local.unique_tokens.numpy(), local.backward_tokens.numpy(), local.source_choices.numpy(),
               lookup, backward_lookup, target_hidden_rows, target_score_elements,
               source_pack_order, source_score_pad, backward_source_vector_rows,
               backward_source_score_elements, backward_target_pack_order, backward_target_score_pad)
    lengths = [len(index) for index in indices]
    packed = np.concatenate([np.asarray(index, dtype=np.int64) for index in indices])
    (unique_tokens, backward_tokens, source_choices, receive_lookup, backward_lookup_device,
     target_hidden_rows_device, target_score_elements_device, source_pack_order_device,
     source_score_pad_device, backward_source_vector_rows_device,
     backward_source_score_elements_device, backward_target_pack_order_device,
     backward_target_score_pad_device) = torch.from_numpy(packed).to(device).split(lengths)
    return TokenDedup(
        unique_tokens=unique_tokens, unique_send=local.unique_send, unique_recv=unique_recv,
        original_send_rows=sum(source_rows), receive_lookup=receive_lookup,
        backward_tokens=backward_tokens, backward_send=local.backward_send, backward_recv=backward_recv,
        backward_lookup=backward_lookup_device, source_choices=source_choices, width=width,
        score_lanes=score_lanes, group=group,
        target_hidden_rows=target_hidden_rows_device, target_score_elements=target_score_elements_device,
        source_split_rows=source_split_rows, target_split_rows=target_split_rows,
        source_pack_order=source_pack_order_device, source_score_pad=source_score_pad_device,
        backward_source_vector_rows=backward_source_vector_rows_device,
        backward_source_score_elements=backward_source_score_elements_device,
        backward_source_split_rows=backward_source_split_rows,
        backward_target_split_rows=backward_target_split_rows,
        backward_target_pack_order=backward_target_pack_order_device,
        backward_target_score_pad=backward_target_score_pad_device,
    )

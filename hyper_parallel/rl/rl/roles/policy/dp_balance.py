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
"""Redistribute complete Actor rows within each global optimizer mini-batch."""

from dataclasses import fields, is_dataclass, replace
import pickle
from typing import Any

import torch
import torch.distributed as dist
from torch.nn import functional

from rl.dataset.contracts import ExperienceBatch
from rl.weight_sync import synchronized_call


_ROW_FIELDS = ("trajectories", "responses", "teacher_keys", "router_traces")


def balanced_row_indices(lengths: list[list[int]], mini_batch_size: int) -> list[list[int]]:
    """Assign descending length stripes to the least-loaded ranks, preserving update membership.

    Each stripe contains one row per rank. This keeps equal row counts and similar
    length distributions as well as token loads. Indices address the rank-major input.

    Args:
        lengths: Complete sequence lengths for each rank in original row order.
        mini_batch_size: Number of rows per rank in each optimizer update.

    Returns:
        Global row indices assigned to each destination rank.
    """
    if not lengths or not lengths[0] or mini_batch_size <= 0:
        raise ValueError("Actor balancing requires nonempty ranks and a positive mini-batch size")
    rows, ranks = len(lengths[0]), len(lengths)
    if any(len(rank) != rows or any(length < 2 for length in rank) for rank in lengths):
        raise ValueError("Actor balancing requires equal row counts and complete sequences")
    result = [[] for _ in lengths]
    for start in range(0, rows, mini_batch_size):
        candidates = [(lengths[rank][row], rank * rows + row)
                      for rank in range(ranks) for row in range(start, min(start + mini_batch_size, rows))]
        candidates.sort(key=lambda item: (-item[0], item[1]))
        loads = [0] * ranks
        for offset in range(0, len(candidates), ranks):
            destinations = sorted(range(ranks), key=lambda rank, loads=loads: (loads[rank], rank))
            for rank, (length, index) in zip(destinations, candidates[offset:offset + ranks]):
                result[rank].append(index)
                loads[rank] += length
    return result


def _row_schema(experience: ExperienceBatch) -> dict[str, Any]:
    """Reject unsupported row layouts before entering data collectives."""
    rows, width = experience.sequences.shape
    schema = {}
    for field in fields(experience):
        value = getattr(experience, field.name)
        if isinstance(value, torch.Tensor):
            if value.requires_grad or value.ndim not in (1, 2) or value.shape[0] != rows:
                raise ValueError(f"Actor balancing requires detached row-aligned tensors: {field.name}")
            offset = None if value.ndim == 1 else width - value.shape[1]
            if offset not in (None, 0, 1):
                raise ValueError(f"Actor balancing requires full or next-token alignment: {field.name}")
            schema[field.name] = (str(value.dtype), offset)
        elif field.name in _ROW_FIELDS:
            if value is not None and len(value) != rows:
                raise ValueError(f"Actor balancing requires one {field.name} entry per row")
            schema[field.name] = value is not None
    return schema


def _cpu_metadata(value: Any) -> Any:
    """Detach identity metadata from device storage before serializing across ranks."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if is_dataclass(value) and not isinstance(value, type):
        return replace(value, **{field.name: _cpu_metadata(getattr(value, field.name)) for field in fields(value)})
    if isinstance(value, dict):
        return {key: _cpu_metadata(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu_metadata(item) for item in value)
    return value


def _exchange_metadata(
    experience: ExperienceBatch, send_rows: list[list[int]], group: Any,
) -> list[dict[str, Any]]:
    """Send trusted rank-local identities only to their destination, alongside tensor rows."""
    payloads = []
    for indices in send_rows:
        payload = {name: tuple(_cpu_metadata(getattr(experience, name)[index]) for index in indices)
                   for name in _ROW_FIELDS if getattr(experience, name) is not None}
        payloads.append(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    device = experience.sequences.device
    send_sizes = torch.tensor([len(payload) for payload in payloads], device=device, dtype=torch.int64)
    receive_sizes = torch.empty_like(send_sizes)
    dist.all_to_all_single(receive_sizes, send_sizes, group=group)
    counts = receive_sizes.cpu().tolist()
    send = torch.frombuffer(bytearray(b"".join(payloads)), dtype=torch.uint8).to(device)
    receive = torch.empty(sum(counts), dtype=torch.uint8, device=device)
    dist.all_to_all_single(receive, send, output_split_sizes=counts,
                           input_split_sizes=[len(payload) for payload in payloads], group=group)
    data = receive.cpu().numpy().tobytes()
    result, offset = [], 0
    for count in counts:
        result.append(pickle.loads(data[offset:offset + count]))
        offset += count
    return result


def _restore_metadata(experience: ExperienceBatch, metadata: list[dict], order: list[int]) -> dict:
    """Restore each transported row identity to the same order as the tensors."""
    updates = {}
    for name in _ROW_FIELDS:
        if getattr(experience, name) is not None:
            received = [row for part in metadata for row in part[name]]
            updates[name] = tuple(received[index] for index in order)
    return updates


def balance_experience(
    experience: ExperienceBatch, lengths: list[int], mini_batch_size: int, group: Any,
) -> ExperienceBatch:
    """Rebalance Actor-only data without changing the caller's experience or global update sets.

    Tensor values move through native all-to-all; CPU trajectory metadata uses a
    separate byte exchange. Rank-local padding widths may differ. Batch metadata
    stays local, while policy versions, tensor schemas and row counts must agree.

    Args:
        experience: Detached row-aligned training inputs and prepared advantages.
        lengths: Complete sequence lengths for the local rank.
        mini_batch_size: Local optimizer mini-batch size before redistribution.
        group: Data-parallel process group shared by all participating ranks.

    Returns:
        A new batch with the same global update membership and redistributed rows.
    """
    schema = synchronized_call("Actor balancing schema", lambda: _row_schema(experience))
    ranks = dist.get_world_size(group)
    local_rank = dist.get_rank(group)
    descriptions = [None] * ranks
    dist.all_gather_object(descriptions, (lengths, experience.sequences.shape[1], schema,
                                         experience.worker_policy_version), group=group)
    if any(item[2:] != descriptions[0][2:] for item in descriptions):
        raise ValueError("Actor balancing requires matching tensor schemas and policy versions across DP")
    orders = balanced_row_indices([item[0] for item in descriptions], mini_batch_size)
    rows = len(lengths)
    target = orders[local_rank]
    send_rows = [[index % rows for index in order if index // rows == local_rank] for order in orders]
    received_order = [index for source in range(ranks) for index in target if index // rows == source]
    positions = {index: position for position, index in enumerate(received_order)}
    restore = [positions[index] for index in target]
    receive_counts = [sum(index // rows == source for index in target) for source in range(ranks)]
    send_counts = list(map(len, send_rows))
    device = experience.sequences.device
    selection = torch.tensor([index for group_rows in send_rows for index in group_rows],
                             dtype=torch.long, device=device)
    permutation = torch.tensor(restore, dtype=torch.long, device=device)
    width = max(item[1] for item in descriptions)
    updates = {}
    for field in fields(experience):
        value = getattr(experience, field.name)
        if not isinstance(value, torch.Tensor):
            continue
        outgoing = value.to(device).index_select(0, selection)
        if value.ndim == 2:
            outgoing = functional.pad(outgoing, (0, width - experience.sequences.shape[1]))
        incoming = torch.empty_like(outgoing)
        dist.all_to_all_single(incoming, outgoing, output_split_sizes=receive_counts,
                               input_split_sizes=send_counts, group=group)
        updates[field.name] = incoming.index_select(0, permutation).to(value.device)
    metadata = _exchange_metadata(experience, send_rows, group)
    updates.update(_restore_metadata(experience, metadata, restore))
    return replace(experience, **updates)

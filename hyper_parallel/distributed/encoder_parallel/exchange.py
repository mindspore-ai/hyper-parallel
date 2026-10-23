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
"""Differentiable routing of aligned image features to decoder token owners."""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping, Sequence

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.core.dtensor._utils import differentiable_all_to_all_single, differentiable_variable_all_gather
from hyper_parallel.distributed.encoder_parallel.contracts import (
    EncoderFeature,
    EncoderWorkItem,
)


class FeatureExchangeMode(Enum):
    """Collective selected from static decoder topology."""

    LOCAL = "local"
    TP_VARIABLE_ALL_GATHER = "tp_variable_all_gather"
    TP_VARIABLE_A2A = "tp_variable_a2a"
    PARENT_VARIABLE_A2A = "parent_variable_a2a"


def select_feature_exchange(mesh_context: Any) -> FeatureExchangeMode:
    """Choose a uniform exchange algorithm from decoder CP/TP/SP topology.

    Args:
        mesh_context: Decoder CP, TP, and sequence-parallel coordinates.
    """
    if mesh_context.cp_size > 1:
        return FeatureExchangeMode.PARENT_VARIABLE_A2A
    if mesh_context.tp_size == 1:
        return FeatureExchangeMode.LOCAL
    if mesh_context.sequence_parallel:
        return FeatureExchangeMode.TP_VARIABLE_A2A
    return FeatureExchangeMode.TP_VARIABLE_ALL_GATHER


def cat_or_empty(rows: Sequence[torch.Tensor], *, anchor: torch.Tensor,
                 feature_width: int) -> torch.Tensor:
    """Concatenate rows, preserving an autograd edge for an empty sender.

    Args:
        rows: Feature tensors to concatenate.
        anchor: Tensor supplying the zero-row gradient edge.
        feature_width: Width of an empty feature tensor.
    """
    if rows:
        return torch.cat(tuple(rows), dim=0)
    return anchor.sum().reshape(1, 1).expand(0, feature_width)


def _ordered_source_records(buckets: Sequence[Sequence[EncoderWorkItem]], encoder_cp_size: int
                            ) -> dict[int, tuple[EncoderWorkItem, ...]]:
    return {bucket_rank * encoder_cp_size: tuple(bucket) for bucket_rank, bucket in enumerate(buckets)}


def _route_tp_all_gather(*, local_by_id: Mapping[int, torch.Tensor],
                         source_records: Mapping[int, tuple[EncoderWorkItem, ...]],
                         prototype: torch.Tensor, parent_rank: int,
                         exchange_mesh: Any) -> tuple[torch.Tensor, tuple[EncoderWorkItem, ...]]:
    """Gather each producer's complete records to every decoder TP rank."""
    group_size = exchange_mesh.size()
    local_items = source_records.get(parent_rank, ())
    local_rows = cat_or_empty([local_by_id[item.item_id] for item in local_items],
                              anchor=prototype, feature_width=int(prototype.shape[-1]))
    splits = [sum(item.output_rows for item in source_records.get(rank, ())) for rank in range(group_size)]
    rows = differentiable_variable_all_gather(local_rows, output_splits=splits,
                                               group=exchange_mesh.get_group())
    received_items = tuple(item for rank in range(group_size) for item in source_records.get(rank, ()))
    return rows, received_items


def _route_variable_a2a(*, mode: FeatureExchangeMode,
                        local_by_id: Mapping[int, torch.Tensor],
                        source_records: Mapping[int, tuple[EncoderWorkItem, ...]],
                        prototype: torch.Tensor, parent_rank: int,
                        destinations: Mapping[int, tuple[int, ...]],
                        encoder_cp_size: int, exchange_mesh: Any
                        ) -> tuple[torch.Tensor, tuple[EncoderWorkItem, ...]]:
    """Send each full record only to its exact decoder token owners."""
    group_size = exchange_mesh.size()
    local_rank = exchange_mesh.get_local_rank()
    if group_size != len(source_records) * encoder_cp_size:
        raise ValueError(f"[HP-ENCODER-009] {mode.value} exchange size disagrees with encoder parent")
    width = int(prototype.shape[-1])
    send_chunks = []
    input_splits = []
    for destination_rank in range(group_size):
        items = [item for item in source_records.get(parent_rank, ())
                 if destination_rank in destinations[item.item_id]]
        chunk = cat_or_empty([local_by_id[item.item_id] for item in items],
                             anchor=prototype, feature_width=width)
        send_chunks.append(chunk)
        input_splits.append(chunk.shape[0])
    packed = cat_or_empty(send_chunks, anchor=prototype, feature_width=width)
    output_splits = []
    received_items = []
    for source_rank in range(group_size):
        items = [item for item in source_records.get(source_rank, ())
                 if local_rank in destinations[item.item_id]]
        output_splits.append(sum(item.output_rows for item in items))
        received_items.extend(items)
    rows = differentiable_all_to_all_single(packed, input_splits=input_splits,
                                            output_splits=output_splits, group=exchange_mesh.get_group())
    return rows, tuple(received_items)


def route_features(*, mode: FeatureExchangeMode, producer_records: Sequence[EncoderFeature],
                   prototype: torch.Tensor, parent_rank: int, destinations: Mapping[int, tuple[int, ...]],
                   buckets: Sequence[Sequence[EncoderWorkItem]], encoder_cp_size: int,
                   exchange_mesh: Any) -> tuple[torch.Tensor, tuple[EncoderWorkItem, ...]]:
    """Exchange full feature records using a deterministic plan shared by all ranks."""
    width = int(prototype.shape[-1])
    local_by_id = {item.item_id: feature for item, feature in producer_records}
    source_records = _ordered_source_records(buckets, encoder_cp_size)
    if mode is FeatureExchangeMode.LOCAL:
        received_items = tuple(item for item, _ in producer_records)
        rows = cat_or_empty([feature for _, feature in producer_records], anchor=prototype, feature_width=width)
        return rows, received_items

    if mode is FeatureExchangeMode.TP_VARIABLE_ALL_GATHER:
        return _route_tp_all_gather(local_by_id=local_by_id, source_records=source_records,
                                    prototype=prototype, parent_rank=parent_rank,
                                    exchange_mesh=exchange_mesh)
    if mode in (FeatureExchangeMode.TP_VARIABLE_A2A, FeatureExchangeMode.PARENT_VARIABLE_A2A):
        return _route_variable_a2a(
            mode=mode, local_by_id=local_by_id, source_records=source_records,
            prototype=prototype, parent_rank=parent_rank, destinations=destinations,
            encoder_cp_size=encoder_cp_size, exchange_mesh=exchange_mesh,
        )
    raise ValueError(f"[HP-ENCODER-009] unsupported feature exchange mode: {mode!r}")


def unpack_records(rows: torch.Tensor, received_items: Sequence[EncoderWorkItem]
                   ) -> tuple[EncoderFeature, ...]:
    """Split transported rows using the shared plan's image order.

    Args:
        rows: Received aligned feature rows.
        received_items: Images in source-rank then bucket order.
    """
    records = []
    offset = 0
    for item in received_items:
        stop = offset + item.output_rows
        records.append((item, rows[offset:stop]))
        offset = stop
    if offset != rows.shape[0]:
        raise ValueError(
            "[HP-ENCODER-009] transported rows disagree with planned images; "
            f"rows={rows.shape[0]}; described={offset}"
        )
    return tuple(sorted(records, key=lambda record: (
        record[0].batch_index, record[0].decoder_token_start, record[0].item_id
    )))

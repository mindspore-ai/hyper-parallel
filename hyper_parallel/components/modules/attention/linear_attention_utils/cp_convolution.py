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
"""Shared halo exchange scheduling before local causal convolution."""
from typing import Optional

import torch
import torch.distributed as dist
from torch import nn

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.utils import communication
from hyper_parallel.components.modules.attention.linear_attention_utils.sequence_layout import PackedSequenceInfo
from hyper_parallel.components.functional.linear_attention_convolution import run_causal_short_conv


def causal_convolutions_with_halo(
    projected: tuple[torch.Tensor, ...],
    convolutions: tuple[nn.Conv1d, ...],
    cp_mesh: DeviceMesh,
    cp_rank: int,
    cp_size: int,
    sequence_info: Optional[PackedSequenceInfo] = None,
    backend: str = "eager",
) -> tuple[torch.Tensor, ...]:
    """Run causal depthwise convolutions with one packed halo exchange.

    Args:
        projected: Token-aligned projections before causal convolution.
        convolutions: Causal convolution modules corresponding to the projections.
        cp_mesh: Sequence-ordered CP device mesh.
        cp_rank: Rank index within the CP mesh.
        cp_size: Number of ranks in the CP mesh.
        sequence_info: Immutable document metadata for this local token interval.
        backend: Explicit local operator implementation to dispatch.
    """
    halo_widths = {
        (conv.kernel_size[0] - 1) * conv.dilation[0] for conv in convolutions
    }
    if len(halo_widths) != 1:
        raise ValueError("Linear attention CP requires Q/K/V ShortConv halo widths to match.")
    halo_width = halo_widths.pop()
    if halo_width == 0 or cp_size == 1:
        return tuple(
            run_causal_short_conv(tensor, convolution, sequence_info=sequence_info, backend=backend)
            for tensor, convolution in zip(projected, convolutions)
        )
    if projected[0].shape[1] < halo_width:
        raise ValueError(
            "Linear attention CP ShortConv requires local_seq_len >= halo width, got "
            f"{projected[0].shape[1]} < {halo_width}."
        )

    channel_sizes = [tensor.shape[-1] for tensor in projected]
    packed_tail = torch.cat(
        [tensor[:, -halo_width:, :] for tensor in projected],
        dim=-1,
    ).contiguous()
    packed_halo = _exchange_previous_rank_halo(
        packed_tail,
        cp_mesh,
        cp_rank,
        cp_size,
    )
    halos = torch.split(packed_halo, channel_sizes, dim=-1)
    return tuple(
        run_causal_short_conv(tensor, convolution, halo, sequence_info=sequence_info, backend=backend)
        for tensor, halo, convolution in zip(projected, halos, convolutions)
    )


def _exchange_previous_rank_halo(
    tail: torch.Tensor,
    cp_mesh: DeviceMesh,
    cp_rank: int,
    cp_size: int,
) -> torch.Tensor:
    """Send a causal-convolution halo only to the next CP rank.

    Args:
        tail: Trailing tokens to exchange with the adjacent sequence rank.
        cp_mesh: Sequence-ordered CP device mesh.
        cp_rank: Rank index within the CP mesh.
        cp_size: Number of ranks in the CP mesh.
    """
    if cp_size == 1:
        return torch.zeros_like(tail)

    cp_group = cp_mesh.get_group()
    group_ranks = tuple(int(rank) for rank in dist.get_process_group_ranks(cp_group))
    rank_list = tuple(int(rank) for rank in cp_mesh.rank_list)
    rank_to_group_index = {rank: index for index, rank in enumerate(group_ranks)}
    halo_width = tail.shape[1]

    input_splits = [0] * cp_size
    exchange_input = tail.permute(1, 0, 2).contiguous()
    if cp_rank < cp_size - 1:
        input_splits[rank_to_group_index[rank_list[cp_rank + 1]]] = halo_width
    else:
        exchange_input = exchange_input[:0]

    output_splits = [0] * cp_size
    if cp_rank > 0:
        output_splits[rank_to_group_index[rank_list[cp_rank - 1]]] = halo_width

    exchange_output = communication.differentiable_all_to_all_single(
        exchange_input,
        input_splits,
        output_splits,
        group=cp_group,
    )
    if cp_rank == 0:
        return torch.zeros_like(tail) + exchange_output.sum().to(tail.dtype) * 0
    return exchange_output.permute(1, 0, 2).contiguous()

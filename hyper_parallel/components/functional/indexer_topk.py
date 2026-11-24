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
"""Restore local and returned Indexer selections without intermediate row gathers."""

import torch


@torch.no_grad()
def restore_indexer_topk(
    selected: torch.Tensor, received: torch.Tensor, rows: tuple[torch.Tensor, ...], output: torch.Tensor,
) -> torch.Tensor:
    """Place local and remote integer selections with one Triton-Ascend launch.

    Args:
        selected: Contiguous worker selections [worker Q, TopK], int32 or int64.
        received: Completed reverse-collective output [remote owner Q, TopK].
        rows: Worker source rows, their owner destinations, and remote owner
            destinations. Each is a contiguous int64 vector on the same NPU.
        output: Contiguous [owner Q, TopK] destination, disjoint from the inputs.

    Returns:
        The supplied output tensor. IDs, slot order and negative sentinels are
        copied exactly, without floating-point conversion or sorting.

    Note:
        The route producer must supply in-range indices whose local and remote
        destinations form a disjoint cover of output rows. The caller must wait
        for the collective before calling this function. Values are not copied
        to the host to validate that producer contract in the training hot path.
    """
    validate_indexer_topk(selected, received, rows, output)
    if output.device.type != "npu":
        raise ValueError("Fused TopK restoration requires tensors on NPU")
    if output.numel() == 0:
        return output
    # Triton is optional; importing CPU-only CP planners must not require it.
    from hyper_parallel.components.functional._triton.indexer_topk import (  # pylint: disable=import-outside-toplevel
        restore_topk_rows,
    )

    return restore_topk_rows(selected, received, rows, output)


def validate_indexer_topk(
    selected: torch.Tensor, received: torch.Tensor, rows: tuple[torch.Tensor, ...], output: torch.Tensor,
) -> None:
    """Validate integer placement metadata without copying device indices to CPU."""
    if len(rows) != 3:
        raise ValueError("TopK restoration requires local sources, local destinations and remote destinations")
    sources, local_destinations, remote_destinations = rows
    tensors = (selected, received, output)
    if any(value.device != output.device for value in (*tensors, *rows)):
        raise ValueError("TopK restoration requires tensors on the same device")
    if output.dtype not in (torch.int32, torch.int64) or any(value.dtype != output.dtype for value in tensors):
        raise ValueError("TopK inputs and output must have matching int32 or int64 dtype")
    if any(value.ndim != 2 or not value.is_contiguous() for value in tensors):
        raise ValueError("TopK inputs and output must be contiguous matrices")
    if output.shape[1] == 0 or any(value.shape[1] != output.shape[1] for value in tensors):
        raise ValueError("TopK inputs and output must have the same positive slot count")
    if any(value.ndim != 1 or value.dtype != torch.int64 or not value.is_contiguous() for value in rows):
        raise ValueError("TopK routing rows must be contiguous int64 vectors")
    if (sources.numel() != local_destinations.numel()
            or received.shape[0] != remote_destinations.numel()
            or sources.numel() + remote_destinations.numel() != output.shape[0]):
        raise ValueError("TopK routing row counts must cover local and remote output rows")
    if output.numel() and any(
            output.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
            for value in (selected, received, *rows)):
        raise ValueError("TopK restoration output must not alias its inputs")

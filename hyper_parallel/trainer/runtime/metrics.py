# Copyright 2025-2026 Bytedance Ltd. and/or its affiliates
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Global loss metrics reduction across data/context parallel ranks."""

from __future__ import annotations

__all__ = ["mean_global_loss"]

from typing import TYPE_CHECKING, Union

import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import

from hyper_parallel.core.utils.communication import (
    _is_tracing_or_compiling,
    differentiable_all_reduce,
)
from hyper_parallel.trainer.runtime.distributed import all_reduce

if TYPE_CHECKING:
    from hyper_parallel.distributed.mesh import MeshContext


def _reduce_weighted_loss(cur_loss, cur_token_len, device_mesh, dp_cp_group):
    """Reduce a loss numerator while preserving the local backward graph."""
    local_weighted_loss = cur_loss.float() * cur_token_len
    backward_loss = local_weighted_loss * device_mesh.dp_size * device_mesh.cp_size
    global_weighted_loss = local_weighted_loss.detach().item()
    if device_mesh.dp_cp_mesh is not None and device_mesh.dp_cp_mesh.size() > 1:
        global_weighted_loss = all_reduce(
            global_weighted_loss,
            op="sum",
            group=dp_cp_group,
        )
    global_sum = local_weighted_loss.new_tensor(global_weighted_loss)
    return global_sum + (backward_loss - backward_loss.detach())


def mean_global_loss(
    losses: Union[dict[str, torch.Tensor], torch.Tensor],
    current_token_counts: dict[str, torch.Tensor],
    device_mesh: "MeshContext",
) -> dict[str, torch.Tensor]:
    """Calculate global loss numerators using explicit mesh information.

    FSDP divides gradients over its flattened DP+CP domain, so each local loss
    is weighted by valid tokens and multiplied by ``dp_size * cp_size``.
    Trainer divides accumulated gradients and reported losses by the actual
    optimizer-step token count afterwards. Empty shards must return finite losses.

    Args:
        losses: A loss tensor or mapping of named loss tensors.
        current_token_counts: Token counts for the current micro batch.
        device_mesh: Trainer mesh context containing parallel sizes and groups.

    Returns:
        Token-weighted loss numerators keyed by loss name.

    Raises:
        ValueError: If sequence parallelism is enabled without a TP device mesh.
    """
    loss_dict = {}
    dp_cp_mesh = device_mesh.dp_cp_mesh
    dp_cp_group = dp_cp_mesh.get_group() if dp_cp_mesh is not None else None
    sequence_parallel = device_mesh.sequence_parallel
    sequence_parallel_group = None
    sequence_parallel_size = 1
    if sequence_parallel:
        if device_mesh.device_mesh is None or "tp" not in device_mesh.device_mesh.mesh_dim_names:
            raise ValueError("Sequence parallelism requires a DeviceMesh with a 'tp' dimension.")
        sequence_parallel_group = device_mesh.device_mesh["tp"].get_group()
        sequence_parallel_size = dist.get_world_size(group=sequence_parallel_group)

    if isinstance(losses, torch.Tensor):  # text loss only
        losses = {"foundation_loss": losses}

    if len({key.split("_loss", maxsplit=1)[0] for key in losses}) != 1:
        raise ValueError("Step gradient normalization requires losses sharing one token-count family")

    for key, cur_loss in losses.items():
        loss_name = key.split("_loss", maxsplit=1)[0]  # foundation/image_decoder/**

        cur_token_len = current_token_counts[f"{loss_name}_tokens"]
        if sequence_parallel:
            if _is_tracing_or_compiling():
                # Graph tracing: ``.item()`` is data-dependent and unsupported
                # by the static-shape tracer; aggregate on tensors instead.
                cur_token_len = differentiable_all_reduce(
                    cur_token_len, "sum", sequence_parallel_group
                )
            else:
                cur_token_len = all_reduce(cur_token_len.item(), op="sum", group=sequence_parallel_group)

        if _is_tracing_or_compiling():
            # Graph-traced variant of the aggregation below: identical math
            # expressed with tensor collectives so the static-shape tracer can
            # capture it (no ``.item()``, no data-dependent zero-token guard).
            # ``differentiable_all_reduce`` dispatches to functional collectives
            # under tracing. BaseTrainer passes token-count tensors as explicit
            # graph inputs so packed/ignored-token counts remain live.
            local_weighted_loss = cur_loss.float() * cur_token_len
            backward_loss = local_weighted_loss * device_mesh.dp_size * device_mesh.cp_size
            global_weighted_loss = local_weighted_loss.detach()
            if dp_cp_mesh is not None and dp_cp_mesh.size() > 1:
                global_weighted_loss = differentiable_all_reduce(
                    global_weighted_loss, "sum", dp_cp_group
                )
            cur_loss = global_weighted_loss + (backward_loss - backward_loss.detach())
            if sequence_parallel:
                cur_loss = cur_loss / sequence_parallel_size
            loss_dict[key] = cur_loss
            continue

        cur_loss = _reduce_weighted_loss(
            cur_loss,
            cur_token_len,
            device_mesh,
            dp_cp_group,
        )

        if sequence_parallel:
            cur_loss = cur_loss / sequence_parallel_size

        loss_dict[key] = cur_loss

    return loss_dict

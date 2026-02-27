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
"""State dict utilities for fully_shard (torch-specific)."""
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.checkpoint.state_dict import StateDictOptions

from hyper_parallel.core.dtensor import DTensor


def _gather_full_state_dict(
    state_dict: dict[str, Any], cpu_offload: bool
) -> dict[str, Any]:
    """All-gather every DTensor shard into a full tensor.

    Args:
        state_dict: Model state dict with DTensor or plain tensor values.
        cpu_offload: If True, only rank-0 keeps the result on CPU;
            other ranks return an empty dict to save memory.
    """
    is_rank0 = (not dist.is_initialized()) or (dist.get_rank() == 0)

    gathered: dict[str, Any] = {}
    for key, val in state_dict.items():
        if isinstance(val, DTensor):
            val = val.full_tensor()
        if cpu_offload:
            if not is_rank0:
                del val
                continue
            if isinstance(val, torch.Tensor):
                val = val.cpu()
        gathered[key] = val

    if cpu_offload and not is_rank0:
        return {}
    return gathered


def _offload_sharded_state_dict(
    state_dict: dict[str, Any],
) -> dict[str, Any]:
    """Move each shard to CPU without all-gathering.

    Args:
        state_dict: Model state dict with DTensor or plain tensor values.
    """
    offloaded: dict[str, Any] = {}
    for key, val in state_dict.items():
        if isinstance(val, DTensor):
            val = DTensor.from_local(
                val.to_local().cpu(), val.device_mesh, val.placements,
            )
        elif isinstance(val, torch.Tensor):
            val = val.cpu()
        offloaded[key] = val
    return offloaded


def get_model_state_dict(
    model: nn.Module,
    *,
    options: StateDictOptions | None = None,
) -> dict[str, Any]:
    """Return the model state dict with configurable gathering and offloading.

    Behaviour matrix:

    +-----------------+-------------+--------------------------------------+
    | full_state_dict | cpu_offload | result                               |
    +=================+=============+======================================+
    | False           | False       | DTensor (sharded, as-is)             |
    +-----------------+-------------+--------------------------------------+
    | False           | True        | DTensor local shard offloaded to CPU |
    +-----------------+-------------+--------------------------------------+
    | True            | False       | full Tensor on **every** rank        |
    +-----------------+-------------+--------------------------------------+
    | True            | True        | full Tensor on CPU, **rank 0 only**  |
    +-----------------+-------------+--------------------------------------+

    Args:
        model: The model whose state dict to retrieve.
        options: Controls full_state_dict, cpu_offload,
            ignore_frozen_params, and broadcast_from_rank0 flags.
    """
    options = options or StateDictOptions()

    if options.broadcast_from_rank0 and not options.full_state_dict:
        raise ValueError(
            "full_state_dict must be True when broadcast_from_rank0 is True."
        )

    state_dict: dict[str, Any] = model.state_dict()

    if options.ignore_frozen_params:
        frozen_keys = {
            name for name, p in model.named_parameters()
            if not p.requires_grad
        }
        for key in frozen_keys:
            state_dict.pop(key, None)

    if options.full_state_dict:
        return _gather_full_state_dict(state_dict, options.cpu_offload)

    if options.cpu_offload:
        return _offload_sharded_state_dict(state_dict)

    return state_dict

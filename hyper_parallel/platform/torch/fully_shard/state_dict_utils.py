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
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.checkpoint.state_dict import StateDictOptions

from hyper_parallel.core.dtensor.dtensor import DTensor, distribute_tensor


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
                val.to_local().cpu(), val.device_mesh, val.layout.alias_placements,
            )
        elif isinstance(val, torch.Tensor):
            val = val.cpu()
        offloaded[key] = val
    return offloaded


def get_model_state_dict(
    model: nn.Module,
    *,
    options: Optional[StateDictOptions] = None,
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

    Raises:
        ValueError: If ``broadcast_from_rank0`` is True while
            ``full_state_dict`` is False.
        NotImplementedError: If ``broadcast_from_rank0`` is True.
            ``broadcast_from_rank0`` requires a cross-rank tensor broadcast,
            which is not available. Use ``full_state_dict=True`` with a full
            tensor on every rank instead.
    """
    options = options or StateDictOptions()

    if options.broadcast_from_rank0 and not options.full_state_dict:
        raise ValueError(
            "full_state_dict must be True when broadcast_from_rank0 is True."
        )
    if options.broadcast_from_rank0:
        raise NotImplementedError(
            "broadcast_from_rank0=True is not supported. "
            "broadcast_from_rank0 requires a cross-rank tensor broadcast, "
            "which is not available. Use full_state_dict=True with a full "
            "tensor on every rank instead."
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


def _scatter_model_state_dict(
    model: nn.Module,
    state_dict: dict[str, Any],
    cpu_offload: bool,
    strict: bool,
) -> dict[str, Any]:
    """Scatter full tensors into DTensor shards matching model's layout.

    Inverse of _gather_full_state_dict. Uses distribute_tensor (no communication,
    assumes every rank holds the same global tensor).

    When ``cpu_offload`` is True the input tensors are expected to live on CPU
    (e.g. produced by ``get_model_state_dict(full_state_dict=True, cpu_offload=True)``).
    The scattered local shard is moved onto the target parameter's device so the
    subsequent ``load_state_dict(assign=True)`` does not leave the parameter on
    the wrong device.

    Args:
        model: The model whose layout the scattered tensors must match.
        state_dict: Input state dict whose values are plain (global) tensors.
        cpu_offload: Whether the input tensors live on CPU and must be moved
            back onto the target parameter's device after scattering.
        strict: When True, keys present in ``state_dict`` but absent from the
            model are treated as errors (unexpected keys), mirroring the
            PyTorch ``load_state_dict(strict=True)`` semantics of the
            ``full_state_dict=False`` passthrough path.

    Raises:
        ValueError: If ``strict`` is True and ``state_dict`` contains keys that
            do not exist in the model (unexpected keys).
    """
    target_state_dict = model.state_dict()
    scattered: dict[str, Any] = {}
    unexpected_keys: list[str] = []
    for key, val in state_dict.items():
        target = target_state_dict.get(key)
        if target is None:
            unexpected_keys.append(key)
            continue
        if isinstance(target, DTensor):
            if isinstance(val, DTensor):
                scattered[key] = val
            else:
                # Slice a plain (global) tensor into a DTensor shard.
                placements = (
                    target.layout.alias_placements if target.layout else target.placements
                )
                scattered[key] = distribute_tensor(val, target.device_mesh, placements)
        else:
            scattered[key] = val

        # When the input came from CPU (cpu_offload=True round-trip), move the
        # scattered local shard onto the target parameter's device. assign=True
        # would otherwise bind a CPU DTensor to an on-device parameter.
        if cpu_offload and isinstance(scattered[key], DTensor) and isinstance(target, DTensor):
            dt = scattered[key]
            target_device = target._local_tensor.device  # pylint: disable=protected-access
            scattered[key] = DTensor.from_local(
                dt.to_local().to(target_device),
                dt.device_mesh,
                dt.layout.alias_placements if dt.layout else dt.placements,
            )

    if strict and unexpected_keys:
        raise ValueError(
            f"Unexpected key(s) in state_dict: {unexpected_keys}. "
            f"To allow loading a state_dict with extra keys, pass strict=False."
        )
    return scattered


def set_model_state_dict(
    model: nn.Module,
    state_dict: dict[str, Any],
    *,
    options: Optional[StateDictOptions] = None,
) -> None:
    """Load state dict into model, scattering full tensors to DTensor shards.

    Behaviour matrix:

    +-----------------+-------------+--------------------------------+
    | full_state_dict | cpu_offload | behaviour                      |
    +=================+=============+================================+
    | True            | False       | scatter full -> DTensor on dev |
    +-----------------+-------------+--------------------------------+
    | True            | True        | scatter full -> DTensor on dev |
    +-----------------+-------------+--------------------------------+
    | False (any)     | *           | load sharded DTensors as-is    |
    +-----------------+-------------+--------------------------------+

    Note:
        ``ignore_frozen_params`` is intentionally a no-op on the setter path,
        mirroring torch.distributed.checkpoint.state_dict: frozen parameters
        filtering is a getter-only feature. Callers that strip frozen keys
        from the input should also pass ``strict=False``.

    Args:
        model: The model to load state into.
        state_dict: State dict to load. Values may be plain (global) tensors
            when ``full_state_dict=True`` or sharded DTensors otherwise.
        options: Controls full_state_dict, cpu_offload, strict and
            broadcast_from_rank0 flags.

    Raises:
        ValueError: If ``broadcast_from_rank0`` is True while
            ``full_state_dict`` is False.
        NotImplementedError: If ``broadcast_from_rank0`` is True.
            ``broadcast_from_rank0`` requires a cross-rank tensor broadcast,
            which is not available. Use ``full_state_dict=True`` with a full
            tensor on every rank instead.
    """
    options = options or StateDictOptions()

    if options.broadcast_from_rank0 and not options.full_state_dict:
        raise ValueError(
            "full_state_dict must be True when broadcast_from_rank0 is True."
        )
    if options.broadcast_from_rank0:
        raise NotImplementedError(
            "broadcast_from_rank0=True is not supported. "
            "broadcast_from_rank0 requires a cross-rank tensor broadcast, "
            "which is not available. Use full_state_dict=True with a full "
            "tensor on every rank instead."
        )

    # Scatter full tensors into DTensor shards matching the model's layout.
    if options.full_state_dict:
        scattered = _scatter_model_state_dict(
            model, state_dict, options.cpu_offload, options.strict,
        )
    else:
        scattered = state_dict

    model.load_state_dict(scattered, strict=options.strict, assign=True)

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
"""Optimizer state dict utilities for fully_shard (torch-specific).

Provides ``get_optim_state_dict`` and ``set_optim_state_dict`` that handle
HSDP DTensor parameters, FQN-based keys, full/local/CPU conversion,
broadcast-from-rank0, flatten/unflatten, and DCP load template construction.

This module does **not** call PyTorch DCP's
``get_optimizer_state_dict``/``set_optimizer_state_dict``.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard

logger = logging.getLogger(__name__)

_SCALAR_STATE_KEYS = {"step"}
_TENSOR_STATE_KEYS_ADAM = {"exp_avg", "exp_avg_sq", "max_exp_avg_sq"}
_TENSOR_STATE_KEYS_SGD = {"momentum_buffer"}


class UnsupportedConfigurationError(RuntimeError):
    """Raised when the requested configuration is not supported."""


def _is_dtensor_param(param: Any) -> bool:
    return isinstance(param, DTensor)


def _is_replicate_group_root(
    mesh: DeviceMesh,
    placements: Sequence,
) -> bool:
    """Return True if current rank is the root (coordinate-0) in every
    Replicate dimension of the given mesh+placements.

    For HSDP with mesh (replicate, shard) and placements (Replicate(), Shard(0)),
    only the rank whose replicate-dim coordinate is 0 returns True — even if
    its global rank is not 0 (e.g. PP stage-1 ranks in a 3-D mesh).

    If there are no Replicate dimensions, falls back to ``dist.get_rank() == 0``.
    """
    coord = mesh.get_coordinate()
    if coord is None:
        return False
    replicate_dims = [i for i, p in enumerate(placements) if isinstance(p, Replicate)]
    if not replicate_dims:
        return dist.get_rank() == 0
    return all(coord[d] == 0 for d in replicate_dims)


def _param_to_fqn(model: nn.Module) -> Dict[nn.Parameter, str]:
    """Build a mapping from parameter object to its fully-qualified name."""
    param_to_name: Dict[nn.Parameter, str] = {}
    for name, param in model.named_parameters():
        if param in param_to_name:
            raise ValueError(
                f"Parameter {name} shares the same object as "
                f"{param_to_name[param]}. Duplicate parameter objects are not supported."
            )
        param_to_name[param] = name
    return param_to_name


def _build_id_to_fqn(
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
) -> Tuple[Dict[int, str], Dict[str, int]]:
    """Build mappings between optimizer saved IDs and FQNs.

    Returns:
        Tuple of (saved_id_to_fqn, fqn_to_saved_id).
    """
    param_to_name = _param_to_fqn(model)
    raw_sd = optimizer.state_dict()
    saved_id_to_fqn: Dict[int, str] = {}
    saved_id_to_param: Dict[int, nn.Parameter] = {}

    for runtime_group, saved_group in zip(optimizer.param_groups, raw_sd["param_groups"]):
        for parameter, saved_id in zip(runtime_group["params"], saved_group["params"]):
            if saved_id in saved_id_to_param and saved_id_to_param[saved_id] is not parameter:
                raise ValueError(
                    f"saved_id {saved_id} maps to different parameters; "
                    f"this is not supported."
                )
            saved_id_to_param[saved_id] = parameter

    for saved_id, param in saved_id_to_param.items():
        if param not in param_to_name:
            raise ValueError(
                f"Parameter with saved_id={saved_id} not found in "
                f"model.named_parameters(). Ensure the model matches the optimizer."
            )
        saved_id_to_fqn[saved_id] = param_to_name[param]

    fqn_to_saved_id = {v: k for k, v in saved_id_to_fqn.items()}
    return saved_id_to_fqn, fqn_to_saved_id


def _get_param_dtensor_info(
    param: nn.Parameter,
) -> Optional[Tuple[DeviceMesh, Sequence]]:
    """Return (device_mesh, placements) if param is DTensor, else None."""
    if _is_dtensor_param(param):
        return param.device_mesh, param.placements
    return None


def _convert_state_tensor(
    tensor: torch.Tensor,
    param: nn.Parameter,
    full_state_dict: bool,
    cpu_offload: bool,
    is_rank0: bool,
) -> Optional[torch.Tensor]:
    """Convert a single optimizer state tensor according to options.

    For scalar states (step), only cpu_offload applies.
    For tensor states sharing the parameter shape, full/local conversion applies.

    When the optimizer state value is itself a DTensor (e.g. when
    ``optimizer.step()`` ran without ``SkipDTensorDispatch`` and the
    parameter is a DTensor), we extract the local shard via
    ``.to_local()`` or gather via ``.full_tensor()`` directly rather
    than wrapping it again.

    Args:
        tensor: The optimizer state tensor (plain Tensor or DTensor).
        param: The corresponding model parameter (may be DTensor).
        full_state_dict: If True, gather to full tensor.
        cpu_offload: If True, move to CPU.
        is_rank0: Whether current rank is 0 (used for full+cpu to drop non-rank0 data).

    Returns:
        Converted tensor, or None if this rank should not keep the result.
    """
    dtensor_info = _get_param_dtensor_info(param)
    is_dtensor_value = isinstance(tensor, DTensor)

    if full_state_dict and dtensor_info is not None:
        if is_dtensor_value:
            full_t = tensor.full_tensor()
        else:
            mesh, placements = dtensor_info
            full_t = DTensor.from_local(tensor, mesh, placements).full_tensor()
        if cpu_offload:
            if not is_rank0:
                return None
            return full_t.cpu()
        return full_t

    if is_dtensor_value:
        local_t = tensor.to_local()
        if cpu_offload:
            return local_t.cpu()
        return local_t

    if cpu_offload:
        return tensor.cpu()

    return tensor


def _is_scalar_state(key: str) -> bool:
    return key in _SCALAR_STATE_KEYS


def _convert_state_scalar(
    tensor: torch.Tensor,
    cpu_offload: bool,
) -> torch.Tensor:
    if isinstance(tensor, DTensor):
        tensor = tensor.to_local()
    if cpu_offload:
        return tensor.cpu()
    return tensor


def _determine_is_root(
    full_state_dict: bool,
    cpu_offload: bool,
    dtensor_info: Optional[Tuple[Any, Any]],
) -> bool:
    """Determine if current rank is the replicate-group root."""
    if full_state_dict and cpu_offload and dtensor_info is not None:
        return _is_replicate_group_root(dtensor_info[0], dtensor_info[1])
    if full_state_dict and cpu_offload:
        return (not dist.is_initialized()) or (dist.get_rank() == 0)
    return True


def _convert_state_entries(
    state: Dict[str, Any],
    param: nn.Parameter,
    full_state_dict: bool,
    cpu_offload: bool,
) -> Dict[str, Any]:
    """Convert optimizer state entries for a single parameter."""
    dtensor_info = _get_param_dtensor_info(param)
    is_root = _determine_is_root(full_state_dict, cpu_offload, dtensor_info)
    skip_non_root = full_state_dict and cpu_offload and not is_root

    converted: Dict[str, Any] = {}
    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            if _is_scalar_state(key):
                result = _convert_state_scalar(value, cpu_offload)
                if not skip_non_root:
                    converted[key] = result
            else:
                result = _convert_state_tensor(
                    value, param, full_state_dict, cpu_offload, is_root,
                )
                if result is not None:
                    converted[key] = result
        else:
            if not skip_non_root:
                converted[key] = value
    return converted


def _build_result_param_groups(
    optimizer: torch.optim.Optimizer,
    raw_sd: Dict[str, Any],
    saved_id_to_fqn: Dict[int, str],
) -> List[Dict[str, Any]]:
    """Build result param_groups with FQN keys from saved id-based groups."""
    result_param_groups: List[Dict[str, Any]] = []
    for saved_group in raw_sd["param_groups"]:
        pg: Dict[str, Any] = {}
        for k, v in saved_group.items():
            if k == "params":
                pg["params"] = [saved_id_to_fqn[sid] for sid in v]
            else:
                pg[k] = v
        result_param_groups.append(pg)
    return result_param_groups


def _check_chained_optimizer(optimizer: Any) -> None:
    if type(optimizer).__name__ == "ChainedOptimizer":
        raise ValueError(
            "ChainedOptimizer is not supported by get_optim_state_dict / "
            "set_optim_state_dict. Use ChainedOptimizer.state_dict() and "
            "ChainedOptimizer.load_state_dict() instead, which handle "
            "multi-optimizer merging internally."
        )


def get_optim_state_dict(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    options: Any = None,
) -> Dict[str, Any]:
    """Return the optimizer state dict with FQN keys and configurable conversion.

    Args:
        model: The model whose parameters are optimized.
        optimizer: The optimizer instance.
        options: Options object with fields:
            full_state_dict (bool): Gather DTensor shards to full tensors.
            cpu_offload (bool): Move tensors to CPU.
            flatten_optimizer_state_dict (bool): Flatten to single-level keys.
            strict (bool): Not used in get; reserved for set.
            broadcast_from_rank0 (bool): Not used in get; reserved for set.

    Returns:
        Optimizer state dict with FQN-based keys.
    """
    _check_chained_optimizer(optimizer)

    full_state_dict = getattr(options, "full_state_dict", False)
    cpu_offload = getattr(options, "cpu_offload", False)
    flatten = getattr(options, "flatten_optimizer_state_dict", False)

    raw_sd = optimizer.state_dict()
    saved_id_to_fqn, _ = _build_id_to_fqn(optimizer, model)

    param_by_id: Dict[int, nn.Parameter] = {}
    for runtime_group, saved_group in zip(optimizer.param_groups, raw_sd["param_groups"]):
        for parameter, saved_id in zip(runtime_group["params"], saved_group["params"]):
            param_by_id[saved_id] = parameter

    result_state: Dict[str, Dict[str, Any]] = {}
    for saved_id, state in raw_sd["state"].items():
        fqn = saved_id_to_fqn[saved_id]
        param = param_by_id[saved_id]
        converted = _convert_state_entries(
            state, param, full_state_dict, cpu_offload,
        )
        if converted:
            result_state[fqn] = converted

    result_param_groups = _build_result_param_groups(
        optimizer, raw_sd, saved_id_to_fqn,
    )

    result: Dict[str, Any] = {
        "state": result_state,
        "param_groups": result_param_groups,
    }

    if flatten:
        result = _flatten_optim_state_dict(result)

    return result


def _check_strict_fqns(
    source_fqns: set,
    target_fqns: set,
    strict: bool,
) -> None:
    """Validate FQN compatibility under strict mode."""
    if not strict:
        return
    extra_fqns = source_fqns - target_fqns
    if extra_fqns:
        raise ValueError(
            f"strict=True but checkpoint contains FQNs not in target "
            f"optimizer: {sorted(extra_fqns)}"
        )


def _load_state_values(
    source_state: Dict[str, Any],
    param: nn.Parameter,
    full_state_dict: bool,
    cpu_offload: bool,
    stores_dtensor: bool,
) -> Dict[str, Any]:
    """Convert and load state values for a single FQN."""
    converted: Dict[str, Any] = {}
    for key, value in source_state.items():
        if isinstance(value, torch.Tensor):
            if _is_scalar_state(key):
                converted[key] = _convert_input_scalar_to_target(
                    value, param, cpu_offload, stores_dtensor,
                )
            else:
                converted[key] = _convert_input_tensor_to_target(
                    value, param, full_state_dict, cpu_offload, stores_dtensor,
                )
        else:
            converted[key] = value
    return converted


def _merge_param_groups(
    source_param_groups: List[Dict[str, Any]],
    target_raw_sd: Dict[str, Any],
    target_fqn_to_saved_id: Dict[str, int],
) -> None:
    """Merge source param_group fields into target param_groups in-place."""
    if not source_param_groups:
        return
    target_saved_id_to_fqn = {v: k for k, v in target_fqn_to_saved_id.items()}

    target_pg_by_fqns: Dict[frozenset, Dict[str, Any]] = {}
    for saved_group in target_raw_sd["param_groups"]:
        fqns_in_group = frozenset(
            target_saved_id_to_fqn.get(sid, "")
            for sid in saved_group.get("params", [])
        )
        target_pg_by_fqns[fqns_in_group] = saved_group

    for source_pg in source_param_groups:
        source_fqn_set = frozenset(source_pg.get("params", []))
        matched_target_pg = None
        for target_fqn_set, target_pg in target_pg_by_fqns.items():
            if source_fqn_set & target_fqn_set:
                matched_target_pg = target_pg
                break
        if matched_target_pg is None:
            continue
        for k, v in source_pg.items():
            if k == "params":
                continue
            matched_target_pg[k] = v


def _build_target_fqn_mappings(
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    param_by_fqn: Dict[str, nn.Parameter],
) -> Tuple[Dict[str, int], set]:
    """Build target FQN→saved_id mapping and identify DTensor state IDs.

    Returns:
        Tuple of (fqn_to_saved_id, dtensor_state_ids).
    """
    target_raw_sd = optimizer.state_dict()

    target_id_to_param: Dict[int, nn.Parameter] = {}
    for runtime_group, saved_group in zip(optimizer.param_groups, target_raw_sd["param_groups"]):
        for parameter, saved_id in zip(runtime_group["params"], saved_group["params"]):
            target_id_to_param[saved_id] = parameter

    fqn_to_saved_id: Dict[str, int] = {}
    for saved_id, param in target_id_to_param.items():
        for name, p in model.named_parameters():
            if p is param:
                fqn_to_saved_id[name] = saved_id
                break

    dtensor_state_ids: set = set()
    for saved_id, state in target_raw_sd["state"].items():
        for value in state.values():
            if isinstance(value, DTensor):
                dtensor_state_ids.add(saved_id)
                break

    return fqn_to_saved_id, dtensor_state_ids


def _load_fqn_states(
    optim_state_dict: Dict[str, Any],
    target_raw_sd: Dict[str, Any],
    target_fqn_to_saved_id: Dict[str, int],
    param_by_fqn: Dict[str, nn.Parameter],
    dtensor_state_ids: set,
    full_state_dict: bool,
    cpu_offload: bool,
    strict: bool,
) -> Dict[int, Dict[str, Any]]:
    """Load source FQN states into the target state dict format.

    Returns:
        Updated state dict with saved_id keys.
    """
    new_state: Dict[int, Dict[str, Any]] = dict(target_raw_sd["state"])

    for fqn, source_state in optim_state_dict.get("state", {}).items():
        if fqn not in target_fqn_to_saved_id:
            if strict:
                raise ValueError(
                    f"strict=True but FQN '{fqn}' not found in target optimizer."
                )
            continue

        target_saved_id = target_fqn_to_saved_id[fqn]
        param = param_by_fqn[fqn]

        if target_saved_id not in new_state:
            new_state[target_saved_id] = {}

        new_state[target_saved_id].update(
            _load_state_values(
                source_state, param, full_state_dict, cpu_offload,
                target_saved_id in dtensor_state_ids,
            )
        )

    return new_state


def set_optim_state_dict(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    optim_state_dict: Dict[str, Any],
    *,
    options: Any = None,
) -> None:
    """Load an optimizer state dict with FQN keys into the optimizer.

    Args:
        model: The model whose parameters are optimized.
        optimizer: The target optimizer instance.
        optim_state_dict: State dict with FQN-based keys (possibly flatten format).
        options: Options object with fields:
            full_state_dict (bool): Input contains full (non-sharded) tensors.
            cpu_offload (bool): Input tensors are on CPU.
            flatten_optimizer_state_dict (bool): Input is in flatten format.
            strict (bool): If True, reject FQNs in the checkpoint that do not
                exist in the target optimizer (extra FQNs).  Missing FQNs are
                always allowed because the target optimizer may have
                untrained/empty state for some parameters.  If False, extra
                FQNs in the checkpoint are silently ignored.
            broadcast_from_rank0 (bool): If True and full_state_dict+cpu_offload,
                broadcast from the replicate-group root to all ranks within
                the same replicate subgroup.  In PP+HSDP, each PP stage
                broadcasts independently within its own HSDP subgroup.
    """
    full_state_dict = getattr(options, "full_state_dict", False)
    cpu_offload = getattr(options, "cpu_offload", False)
    flatten = getattr(options, "flatten_optimizer_state_dict", False)
    strict = getattr(options, "strict", True)
    broadcast_from_rank0 = getattr(options, "broadcast_from_rank0", False)

    _check_chained_optimizer(optimizer)

    if flatten:
        optim_state_dict = _unflatten_optim_state_dict(optim_state_dict, model, strict=strict)

    if full_state_dict and cpu_offload and not broadcast_from_rank0:
        has_any_state = bool(optim_state_dict.get("state", {}))
        if not has_any_state:
            raise ValueError(
                "Received empty state dict with full_state_dict=True "
                "and cpu_offload=True but broadcast_from_rank0=False. "
                "Set broadcast_from_rank0=True to allow the replicate-group "
                "root to broadcast."
            )

    if broadcast_from_rank0:
        optim_state_dict = _broadcast_state_from_rank0(
            optim_state_dict, model, full_state_dict, cpu_offload,
        )

    param_by_fqn: Dict[str, nn.Parameter] = {}
    for name, param in model.named_parameters():
        param_by_fqn[name] = param

    target_fqn_to_saved_id, dtensor_state_ids = _build_target_fqn_mappings(
        optimizer, model, param_by_fqn,
    )

    source_fqns = set(optim_state_dict.get("state", {}).keys())
    target_fqns = set(target_fqn_to_saved_id.keys())
    _check_strict_fqns(source_fqns, target_fqns, strict)

    target_raw_sd = optimizer.state_dict()
    new_state = _load_fqn_states(
        optim_state_dict, target_raw_sd, target_fqn_to_saved_id,
        param_by_fqn, dtensor_state_ids, full_state_dict, cpu_offload, strict,
    )

    target_raw_sd["state"] = new_state

    _merge_param_groups(
        optim_state_dict.get("param_groups", []),
        target_raw_sd,
        target_fqn_to_saved_id,
    )

    optimizer.load_state_dict(target_raw_sd)


def _convert_input_tensor_to_target(
    tensor: torch.Tensor,
    param: nn.Parameter,
    full_state_dict: bool,
    cpu_offload: bool,
    stores_dtensor: bool = False,
) -> Union[torch.Tensor, DTensor]:
    """Convert an incoming state tensor to the target optimizer's format.

    For DTensor params whose optimizer stores DTensor state values (i.e. when
    ``optimizer.step()`` ran without ``SkipDTensorDispatch``), we return a
    DTensor so that ``optimizer.load_state_dict()`` receives the correct type.

    For DTensor params whose optimizer stores plain local-shard tensors (i.e.
    when ``optimizer.step()`` ran inside ``SkipDTensorDispatch``), we return
    a plain local-shard tensor.

    For plain (non-DTensor) params, we return a plain tensor on the correct
    device.

    Args:
        tensor: Incoming state tensor (plain Tensor, possibly on CPU).
        param: The corresponding model parameter (may be DTensor).
        full_state_dict: If True, input tensor represents the full (non-sharded) value.
        cpu_offload: If True, input tensor is on CPU and may need device transfer.
        stores_dtensor: If True, the target optimizer stores DTensor state values
            for this parameter (detected from existing ``optimizer.state_dict()``).

    Returns:
        Tensor or DTensor matching the target optimizer's expected format.
    """
    dtensor_info = _get_param_dtensor_info(param)

    if not dtensor_info:
        target_device = param.data.device
        result = tensor
        if cpu_offload and result.device != target_device:
            result = result.to(target_device)
        return result

    mesh, placements = dtensor_info
    target_device = param.to_local().device

    from hyper_parallel.core.dtensor.dtensor import distribute_tensor  # pylint: disable=C0415

    if isinstance(tensor, DTensor):
        if tensor.device_mesh == mesh and tensor.placements == placements:
            if stores_dtensor:
                return tensor
            return tensor.to_local()
        redistributed = tensor.redistribute(mesh, placements)
        if stores_dtensor:
            return redistributed
        return redistributed.to_local()

    if full_state_dict:
        if cpu_offload:
            tensor = tensor.to(target_device)
        dt = distribute_tensor(tensor, mesh, placements)
        if stores_dtensor:
            return dt
        return dt.to_local()

    if cpu_offload and tensor.device != target_device:
        tensor = tensor.to(target_device)

    if stores_dtensor:
        if tensor.shape == param.to_local().shape:
            return DTensor.from_local(tensor, mesh, placements)
        dt = distribute_tensor(tensor, mesh, placements)
        return dt

    if tensor.shape == param.to_local().shape:
        return tensor

    dt = distribute_tensor(tensor, mesh, placements)
    return dt.to_local()


def _convert_input_scalar_to_target(
    tensor: torch.Tensor,
    param: nn.Parameter,
    cpu_offload: bool,
    stores_dtensor: bool = False,
) -> Union[torch.Tensor, DTensor]:
    """Convert an incoming scalar state tensor (e.g. step) to the target format.

    Scalar states are not sharded — they are replicated on all ranks. When the
    target optimizer stores DTensor values (``stores_dtensor=True``), we wrap
    the scalar as a replicated DTensor; otherwise we return a plain tensor on
    the correct device.

    Args:
        tensor: Incoming scalar tensor (plain Tensor, possibly on CPU).
        param: The corresponding model parameter (may be DTensor).
        cpu_offload: If True, input tensor is on CPU and may need device transfer.
        stores_dtensor: If True, the target optimizer stores DTensor state values.

    Returns:
        Plain tensor or replicated DTensor matching the target's expected format.
    """
    dtensor_info = _get_param_dtensor_info(param)

    if not dtensor_info or not stores_dtensor:
        target_device = param.data.device if not dtensor_info else param.to_local().device
        result = tensor
        if cpu_offload and result.device != target_device:
            result = result.to(target_device)
        return result

    mesh, placements = dtensor_info
    target_device = param.to_local().device

    scalar_placements = [Replicate()] * mesh.ndim
    if cpu_offload and tensor.device != target_device:
        tensor = tensor.to(target_device)
    return DTensor.from_local(tensor, mesh, scalar_placements)


def _get_broadcast_groups(
    model: nn.Module,
) -> List[Dict[str, Any]]:
    """Identify broadcast groups from the model's DTensor parameters.

    A broadcast group is a set of ranks that share the same model
    parameters (same FQNs).  In HSDP, ranks within the same replicate
    subgroup share parameters.  In PP, each PP stage is a different
    broadcast group because different stages own different parameters.

    Returns:
        List of dicts, each with keys:
            fqns: list of FQN strings owned by this group
            pg: ProcessGroup for broadcast within the group
            src_rank: global rank of the replicate-group root
            is_root: True if current rank is the root of this group
    """
    param_by_fqn: Dict[str, nn.Parameter] = {}
    for name, param in model.named_parameters():
        param_by_fqn[name] = param

    model_fqns = list(param_by_fqn.keys())

    if not model_fqns:
        return []

    first_param = param_by_fqn[model_fqns[0]]
    dtensor_info = _get_param_dtensor_info(first_param)

    if dtensor_info is not None:
        mesh, placements = dtensor_info
        is_root = _is_replicate_group_root(mesh, placements)
        replicate_dims = [i for i, p in enumerate(placements) if isinstance(p, Replicate)]
        if replicate_dims:
            pg = mesh.get_group(replicate_dims[0])
            coord = mesh.get_coordinate()
            if coord is not None:
                root_coord = list(coord)
                for d in replicate_dims:
                    root_coord[d] = 0
                src_rank = int(mesh.mesh[tuple(root_coord)])
            else:
                src_rank = 0
        else:
            pg = dist.group.WORLD
            src_rank = 0
    else:
        is_root = dist.get_rank() == 0
        pg = dist.group.WORLD
        src_rank = 0

    return [{
        "fqns": model_fqns,
        "pg": pg,
        "src_rank": src_rank,
        "is_root": is_root,
        "param_by_fqn": param_by_fqn,
    }]


def _broadcast_fqn_list(
    is_root: bool,
    fqns: List[str],
    pg: dist.ProcessGroup,
    src_rank: int,
) -> List[str]:
    """Broadcast FQN list from root to all ranks in the replicate group."""
    fqn_list = fqns if is_root else []
    obj = [fqn_list]
    dist.broadcast_object_list(obj, src=src_rank, group=pg)
    return obj[0]


def _build_state_schema(
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """Build schema dict describing the types and shapes of state entries."""
    schema: Dict[str, Any] = {}
    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            schema[key] = {
                "shape": tuple(value.shape),
                "dtype": str(value.dtype),
                "is_scalar": _is_scalar_state(key),
            }
        else:
            schema[key] = {"type": type(value).__name__, "value": value}
    return schema


def _broadcast_schema_per_fqn(
    fqn_list: List[str],
    optim_state_dict: Dict[str, Any],
    is_root: bool,
    pg: dist.ProcessGroup,
    src_rank: int,
) -> Dict[str, Dict[str, Any]]:
    """Broadcast schema for each FQN from root to all ranks."""
    fqn_schema: Dict[str, Dict[str, Any]] = {}
    for fqn in fqn_list:
        if is_root and fqn in optim_state_dict.get("state", {}):
            schema = _build_state_schema(optim_state_dict["state"][fqn])
        else:
            schema = {}
        schema_list = [schema] if is_root else [None]
        dist.broadcast_object_list(schema_list, src=src_rank, group=pg)
        fqn_schema[fqn] = schema_list[0]
    return fqn_schema


def _get_local_device() -> torch.device:
    """Get the device for the current local rank."""
    local_rank = dist.get_rank() if dist.is_initialized() else 0
    if torch.npu.is_available():
        return torch.device(f"npu:{local_rank}")
    return torch.device(f"cuda:{local_rank}")


def _broadcast_scalar_entry(
    key: str,
    fqn: str,
    info: Dict[str, Any],
    optim_state_dict: Dict[str, Any],
    is_root: bool,
    src_rank: int,
    pg: dist.ProcessGroup,
    cpu_offload: bool,
) -> torch.Tensor:
    """Broadcast a scalar state entry and return the result tensor."""
    dtype_str = info.get("dtype", "torch.float32")
    dtype = _resolve_dtype(dtype_str)
    device = _get_local_device()

    if is_root and fqn in optim_state_dict.get("state", {}):
        scalar_val = optim_state_dict["state"][fqn][key]
        t = scalar_val.clone() if isinstance(scalar_val, torch.Tensor) else torch.tensor(scalar_val)
        if t.dim() == 0:
            t = t.reshape(1)
    else:
        t = torch.zeros(1, dtype=dtype)

    if t.device.type == "cpu":
        t = t.to(device)
    dist.broadcast(t, src=src_rank, group=pg)
    return t.reshape(()).cpu() if cpu_offload else t.reshape(())


def _broadcast_tensor_entry(
    key: str,
    fqn: str,
    info: Dict[str, Any],
    optim_state_dict: Dict[str, Any],
    is_root: bool,
    src_rank: int,
    pg: dist.ProcessGroup,
    full_state_dict: bool,
    cpu_offload: bool,
) -> torch.Tensor:
    """Broadcast a tensor state entry and return the result tensor."""
    shape = info.get("shape", ())
    dtype_str = info.get("dtype", "torch.float32")
    dtype = _resolve_dtype(dtype_str)
    device = _get_local_device()

    if is_root and fqn in optim_state_dict.get("state", {}):
        src_tensor = optim_state_dict["state"][fqn][key]
        if src_tensor.device.type == "cpu":
            src_tensor = src_tensor.to(device)
    else:
        src_tensor = torch.zeros(shape, dtype=dtype, device=device)

    dist.broadcast(src_tensor, src=src_rank, group=pg)

    if full_state_dict and cpu_offload:
        return src_tensor.cpu() if is_root else src_tensor
    return src_tensor.cpu() if cpu_offload else src_tensor


def _broadcast_tensor_data_per_fqn(
    fqn_list: List[str],
    fqn_schema: Dict[str, Dict[str, Any]],
    optim_state_dict: Dict[str, Any],
    is_root: bool,
    src_rank: int,
    pg: dist.ProcessGroup,
    full_state_dict: bool,
    cpu_offload: bool,
) -> Dict[str, Dict[str, Any]]:
    """Broadcast tensor data for each FQN from root to all ranks."""
    result_state: Dict[str, Dict[str, Any]] = {}
    for fqn in fqn_list:
        schema = fqn_schema[fqn]
        if not schema:
            continue
        result_state[fqn] = {}
        for key, info in schema.items():
            is_scalar = info.get("is_scalar", False)
            if is_scalar:
                result_state[fqn][key] = _broadcast_scalar_entry(
                    key, fqn, info, optim_state_dict,
                    is_root, src_rank, pg, cpu_offload,
                )
            else:
                result_state[fqn][key] = _broadcast_tensor_entry(
                    key, fqn, info, optim_state_dict,
                    is_root, src_rank, pg, full_state_dict, cpu_offload,
                )
    return result_state


def _broadcast_state_from_rank0(
    optim_state_dict: Dict[str, Any],
    model: nn.Module,
    full_state_dict: bool,
    cpu_offload: bool,
) -> Dict[str, Any]:
    """Broadcast full optimizer state from replicate-group root to all ranks.

    Only meaningful when full_state_dict=True and cpu_offload=True.
    The root rank (coordinate-0 in every Replicate dimension) has the full
    data; other ranks in the same replicate subgroup receive it via
    broadcast on the parameter's replicate-dim process group.

    For non-DTensor parameters (pure FSDP), falls back to global rank 0
    as the broadcast source.

    **MPMD / Pipeline Parallelism**: each rank uses its own model's FQN
    list, which only contains the parameters of its own PP stage.  The
    broadcast is scoped to the replicate-dim process group (HSDP
    subgroup), so different PP stages broadcast independently without
    cross-stage interference.  This avoids the problem where global
    rank 0's FQN list does not include other stages' parameters.
    """
    if not dist.is_initialized():
        return optim_state_dict

    groups = _get_broadcast_groups(model)

    if not groups:
        return {
            "state": {},
            "param_groups": optim_state_dict.get("param_groups", []),
        }

    group = groups[0]
    pg = group["pg"]
    src_rank = group["src_rank"]
    is_root = group["is_root"]
    fqns = group["fqns"]

    fqn_list = _broadcast_fqn_list(is_root, fqns, pg, src_rank)
    fqn_schema = _broadcast_schema_per_fqn(
        fqn_list, optim_state_dict, is_root, pg, src_rank,
    )
    result_state = _broadcast_tensor_data_per_fqn(
        fqn_list, fqn_schema, optim_state_dict,
        is_root, src_rank, pg, full_state_dict, cpu_offload,
    )

    return {
        "state": result_state,
        "param_groups": optim_state_dict.get("param_groups", []),
    }


def _flatten_optim_state_dict(
    state_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Flatten nested state dict to single-level dot-separated keys.

    Format:
        state.<FQN>.<state_key> = value
        param_group.<FQN>.<group_field> = value

    **MPMD / Pipeline Parallelism caveat**: in PP, different ranks may
    own parameters with the same local FQN (e.g. ``layers.0.net1.weight``
    exists on both stage 0 and stage 1).  The flatten format does not
    include a PP-rank or stage identifier, so these FQNs would collide
    when combined into a single flat dict.  The recommended PP checkpoint
    pattern is for each rank to independently save/load its own stages'
    optimizer state dicts using ``no_dist=True`` (see P6 test pattern),
    rather than using a coordinated DCP save with flatten format across
    all ranks.
    """
    flat: Dict[str, Any] = {}

    for fqn, state in state_dict.get("state", {}).items():
        for key, value in state.items():
            flat[f"state.{fqn}.{key}"] = value

    for group_idx, group in enumerate(state_dict.get("param_groups", [])):
        params_list = group.get("params", [])
        if not params_list:
            raise UnsupportedConfigurationError(
                f"Cannot flatten param_groups[{group_idx}]: empty 'params' list. "
                f"Empty param groups cannot be represented in flatten format "
                f"because there is no FQN to prefix the group fields. "
                f"Use non-flatten format or provide a stable group_name."
            )
        for fqn in params_list:
            for key, value in group.items():
                if key == "params":
                    continue
                flat[f"param_group.{fqn}.{key}"] = value

    return flat


def _check_inconsistent_pg_fields(
    existing: Dict[str, Any],
    incoming: Dict[str, Any],
    fqn: str,
    strict: bool,
) -> None:
    """Check and raise/warn on inconsistent param_group fields."""
    common_keys = set(existing.keys()) & set(incoming.keys())
    for k in common_keys:
        if existing[k] != incoming[k]:
            if strict:
                raise ValueError(
                    f"strict=True but param_group field '{k}' is "
                    f"inconsistent within the same group: "
                    f"existing={existing[k]!r}, "
                    f"incoming(from {fqn})={incoming[k]!r}"
                )
            logger.warning(
                "param_group field '%s' is inconsistent within the "
                "same group: existing=%r, incoming(from %s)=%r. "
                "Keeping existing value (strict=False).",
                k, existing[k], fqn, incoming[k],
            )


def _parse_unflatten_entries(
    flat_dict: Dict[str, Any],
    known_fqns_sorted: List[str],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Parse flat dict entries into state and param_group_fields dicts.

    Returns:
        Tuple of (state, param_group_fields).
    """
    state: Dict[str, Dict[str, Any]] = {}
    param_group_fields: Dict[str, Dict[str, Any]] = {}

    for key, value in flat_dict.items():
        if key.startswith("state."):
            remainder = key[len("state."):]
            fqn = _match_fqn_from_remainder(remainder, known_fqns_sorted)
            if fqn is None:
                raise ValueError(
                    f"Cannot match FQN from flat key '{key}'. "
                    f"Known FQNs: {known_fqns_sorted}"
                )
            state_key = remainder[len(fqn) + 1:]
            state.setdefault(fqn, {})[state_key] = value

        elif key.startswith("param_group."):
            remainder = key[len("param_group."):]
            fqn = _match_fqn_from_remainder(remainder, known_fqns_sorted)
            if fqn is None:
                raise ValueError(
                    f"Cannot match FQN from flat key '{key}'. "
                    f"Known FQNs: {known_fqns_sorted}"
                )
            field_name = remainder[len(fqn) + 1:]
            param_group_fields.setdefault(fqn, {})[field_name] = value

    return state, param_group_fields


def _assemble_param_groups(
    known_fqns: List[str],
    state: Dict[str, Dict[str, Any]],
    param_group_fields: Dict[str, Dict[str, Any]],
    strict: bool,
) -> List[Dict[str, Any]]:
    """Assemble param_groups list from FQN-based state and fields."""
    param_groups: List[Dict[str, Any]] = []
    current_group: Dict[str, Any] = {"params": []}

    for fqn in known_fqns:
        if fqn in param_group_fields or fqn in state:
            fields = param_group_fields.get(fqn, {})
            if not current_group["params"]:
                current_group["params"].append(fqn)
                current_group.update(fields)
            else:
                existing_fields = {
                    k: v for k, v in current_group.items() if k != "params"
                }
                if existing_fields == fields or not fields:
                    current_group["params"].append(fqn)
                else:
                    _check_inconsistent_pg_fields(existing_fields, fields, fqn, strict)
                    current_group["params"].append(fqn)

    if current_group["params"]:
        param_groups.append(current_group)

    if not param_groups:
        raise UnsupportedConfigurationError(
            "Cannot unflatten: empty param_group in flatten format. "
            "Provide a stable group_name or use non-flatten format."
        )

    return param_groups


def _unflatten_optim_state_dict(
    flat_dict: Dict[str, Any],
    model: nn.Module,
    strict: bool = True,
) -> Dict[str, Any]:
    """Unflatten a flat state dict back to nested format.

    Uses longest-prefix matching against model FQNs to handle FQNs
    containing dots.

    Args:
        flat_dict: The flattened state dict.
        model: The model whose FQNs are used for prefix matching.
        strict: If True, raise ValueError when param_group fields are
            inconsistent within the same group. If False, keep the
            current group's values and log a warning.

    Returns:
        The nested state dict with FQN keys.
    """
    known_fqns = [name for name, _ in model.named_parameters()]
    known_fqns_sorted = sorted(known_fqns, key=len, reverse=True)

    state, param_group_fields = _parse_unflatten_entries(flat_dict, known_fqns_sorted)
    param_groups = _assemble_param_groups(known_fqns, state, param_group_fields, strict)

    return {
        "state": state,
        "param_groups": param_groups,
    }


def _build_optim_state_dict_load_template(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    storage_reader: Any,
    options: Any = None,
) -> Dict[str, Any]:
    """Build an empty optimizer state dict template for DCP load.

    Reads checkpoint metadata from ``storage_reader`` to determine
    key, shape, and dtype for each tensor. This is needed when loading
    into a new optimizer that has no state yet (optimizer.step() has not
    been called), so ``optimizer.state_dict()`` returns empty state.

    The template uses the checkpoint metadata to infer the correct keys
    and shapes, then creates zero-filled tensors so that ``dcp.load``
    has a valid target to write into.

    Args:
        model: The model whose parameters are optimized.
        optimizer: The target optimizer (may have empty state).
        storage_reader: DCP StorageReader with ``load_metadata()`` method.
        options: Options for full_state_dict, cpu_offload,
            flatten_optimizer_state_dict, etc.

    Returns:
        State dict template with empty tensors of correct shape/dtype.
    """
    full_state_dict = getattr(options, "full_state_dict", False)
    cpu_offload = getattr(options, "cpu_offload", False)
    flatten = getattr(options, "flatten_optimizer_state_dict", False)

    try:
        metadata = storage_reader.load_metadata()
    except FileNotFoundError:
        rank = dist.get_rank() if dist.is_initialized() else 0
        metadata = storage_reader.load_metadata(rank=rank)

    param_by_fqn: Dict[str, nn.Parameter] = {}
    for name, param in model.named_parameters():
        param_by_fqn[name] = param

    if flatten:
        return _build_flatten_template_from_metadata(
            metadata, param_by_fqn, full_state_dict, cpu_offload,
        )

    return _build_nested_template_from_metadata(
        metadata, optimizer, param_by_fqn, full_state_dict, cpu_offload,
    )


def _resolve_dtype(dtype_str: str) -> torch.dtype:
    """Resolve a dtype string like 'torch.float32' to torch.dtype."""
    try:
        return getattr(torch, dtype_str.replace("torch.", ""))
    except AttributeError:
        return torch.float32


def _match_fqn_from_remainder(
    remainder: str,
    known_fqns_sorted: List[str],
) -> Optional[str]:
    """Match a remainder string against known FQNs using longest-prefix match."""
    for fqn in known_fqns_sorted:
        if remainder == fqn or remainder.startswith(fqn + "."):
            return fqn
    return None


def _build_saved_id_to_fqn(
    optimizer: torch.optim.Optimizer,
    param_by_fqn: Dict[str, nn.Parameter],
) -> Dict[int, str]:
    """Build mapping from optimizer saved IDs to FQNs."""
    raw_sd = optimizer.state_dict()
    saved_id_to_fqn: Dict[int, str] = {}
    for runtime_group, saved_group in zip(optimizer.param_groups, raw_sd["param_groups"]):
        for parameter, saved_id in zip(runtime_group["params"], saved_group["params"]):
            for name, p in param_by_fqn.items():
                if p is parameter:
                    saved_id_to_fqn[saved_id] = name
                    break
    return saved_id_to_fqn


def _parse_nested_state_entry(
    meta_key: str,
    meta_val: Any,
    param_by_fqn: Dict[str, nn.Parameter],
    full_state_dict: bool,
    cpu_offload: bool,
) -> Optional[Tuple[str, str, torch.Tensor]]:
    """Parse a state.* metadata entry and return (fqn, state_key, tensor) or None."""
    from hyper_parallel.core.distributed_checkpoint.metadata import (
        TensorStorageMetadata,
    )

    remainder = meta_key[len("state."):]
    fqns_sorted = sorted(param_by_fqn.keys(), key=len, reverse=True)
    matched_fqn = _match_fqn_from_remainder(remainder, fqns_sorted)
    if matched_fqn is None:
        return None

    state_key = remainder[len(matched_fqn) + 1:] if len(remainder) > len(matched_fqn) else None
    if state_key is None:
        return None

    if not isinstance(meta_val, TensorStorageMetadata):
        return None

    dtype = _resolve_dtype(meta_val.properties.dtype)
    global_shape = meta_val.size
    param = param_by_fqn.get(matched_fqn)

    if _is_scalar_state(state_key):
        device = torch.device("cpu")
        t = torch.zeros(global_shape, dtype=dtype, device=device)
    else:
        t = _create_empty_state_tensor(param, global_shape, dtype, full_state_dict, cpu_offload)

    return matched_fqn, state_key, t


def _parse_nested_param_group_entry(
    meta_key: str,
    meta_val: Any,
) -> Optional[Tuple[int, str, torch.Tensor]]:
    """Parse a param_group.* metadata entry and return (group_idx, field_name, tensor) or None."""
    from hyper_parallel.core.distributed_checkpoint.metadata import (
        TensorStorageMetadata,
    )

    remainder = meta_key[len("param_group."):]
    parts = remainder.split(".", 1)
    if len(parts) < 2:
        return None
    try:
        group_idx = int(parts[0])
    except ValueError:
        return None
    field_name = parts[1]

    if not isinstance(meta_val, TensorStorageMetadata):
        return None

    dtype = _resolve_dtype(meta_val.properties.dtype)
    return group_idx, field_name, torch.zeros(meta_val.size, dtype=dtype)


def _build_nested_template_from_metadata(
    metadata: Any,
    optimizer: torch.optim.Optimizer,
    param_by_fqn: Dict[str, nn.Parameter],
    full_state_dict: bool,
    cpu_offload: bool,
) -> Dict[str, Any]:
    """Build nested (non-flatten) template from checkpoint metadata.

    Non-flatten checkpoint keys look like:
        state.<FQN>.<state_key>  -> TensorStorageMetadata
        param_group.<group_idx>.<field> -> TensorStorageMetadata or BytesStorageMetadata

    We reconstruct:
        {
            "state": { "<FQN>": { "<state_key>": empty_tensor, ... }, ... },
            "param_groups": [ { "params": [...], "<field>": value, ... }, ... ]
        }
    """
    state: Dict[str, Dict[str, Any]] = {}
    param_groups_raw: Dict[int, Dict[str, Any]] = {}
    param_groups_fqns: Dict[int, List[str]] = {}

    saved_id_to_fqn = _build_saved_id_to_fqn(optimizer, param_by_fqn)
    raw_sd = optimizer.state_dict()

    for group_idx, saved_group in enumerate(raw_sd["param_groups"]):
        param_groups_raw[group_idx] = {
            k: v for k, v in saved_group.items() if k != "params"
        }
        param_groups_fqns[group_idx] = [
            saved_id_to_fqn[sid] for sid in saved_group["params"]
        ]

    for meta_key, meta_val in metadata.state_dict_metadata.items():
        if meta_key.startswith("state."):
            result = _parse_nested_state_entry(
                meta_key, meta_val, param_by_fqn, full_state_dict, cpu_offload,
            )
            if result is not None:
                fqn, state_key, t = result
                state.setdefault(fqn, {})[state_key] = t
        elif meta_key.startswith("param_group."):
            result = _parse_nested_param_group_entry(meta_key, meta_val)
            if result is not None:
                group_idx, field_name, t = result
                param_groups_raw.setdefault(group_idx, {})[field_name] = t

    result_param_groups: List[Dict[str, Any]] = []
    for group_idx in sorted(param_groups_fqns.keys()):
        pg: Dict[str, Any] = {"params": param_groups_fqns[group_idx]}
        pg.update(param_groups_raw.get(group_idx, {}))
        result_param_groups.append(pg)

    return {
        "state": state,
        "param_groups": result_param_groups,
    }


def _build_flatten_template_from_metadata(
    metadata: Any,
    param_by_fqn: Dict[str, nn.Parameter],
    full_state_dict: bool,
    cpu_offload: bool,
) -> Dict[str, Any]:
    """Build flatten template from checkpoint metadata.

    Flatten checkpoint keys look like:
        state.<FQN>.<state_key>  -> TensorStorageMetadata
        param_group.<FQN>.<field> -> TensorStorageMetadata or BytesStorageMetadata
    """
    from hyper_parallel.core.distributed_checkpoint.metadata import (
        TensorStorageMetadata,
    )

    flat: Dict[str, Any] = {}
    fqns_sorted = sorted(param_by_fqn.keys(), key=len, reverse=True)

    for meta_key, meta_val in metadata.state_dict_metadata.items():
        if meta_key.startswith("state."):
            remainder = meta_key[len("state."):]
            matched_fqn = _match_fqn_from_remainder(remainder, fqns_sorted)
            if matched_fqn is None:
                continue

            state_key = remainder[len(matched_fqn) + 1:] if len(remainder) > len(matched_fqn) else None
            if state_key is None:
                continue

            if isinstance(meta_val, TensorStorageMetadata):
                dtype = _resolve_dtype(meta_val.properties.dtype)
                global_shape = meta_val.size
                param = param_by_fqn.get(matched_fqn)

                if _is_scalar_state(state_key):
                    t = torch.zeros(global_shape, dtype=dtype)
                else:
                    t = _create_empty_state_tensor(
                        param, global_shape, dtype,
                        full_state_dict, cpu_offload,
                    )
                flat[meta_key] = t

        elif meta_key.startswith("param_group."):
            if isinstance(meta_val, TensorStorageMetadata):
                dtype = _resolve_dtype(meta_val.properties.dtype)
                flat[meta_key] = torch.zeros(meta_val.size, dtype=dtype)

    return flat


def _create_empty_state_tensor(
    param: Optional[nn.Parameter],
    global_shape: tuple,
    dtype: torch.dtype,
    full_state_dict: bool,
    cpu_offload: bool,
) -> torch.Tensor:
    """Create an empty state tensor with appropriate shape and device.

    For DTensor params, determines whether to use full or local shape
    based on full_state_dict. For plain params, uses the global shape.

    Args:
        param: The model parameter (may be DTensor or None).
        global_shape: Global shape from checkpoint metadata.
        dtype: Tensor dtype.
        full_state_dict: If True, use global shape; otherwise use local.
        cpu_offload: If True, place on CPU.

    Returns:
        Empty tensor with correct shape, dtype, and device.
    """
    dtensor_info = _get_param_dtensor_info(param) if param is not None else None

    if dtensor_info is not None:
        mesh, placements = dtensor_info
        if full_state_dict:
            shape = global_shape
            device = torch.device("cpu") if cpu_offload else param.to_local().device
        else:
            shape = param.to_local().shape
            device = torch.device("cpu") if cpu_offload else param.to_local().device
    else:
        shape = global_shape
        device = torch.device("cpu") if cpu_offload else (
            param.data.device if param is not None else torch.device("cpu")
        )

    return torch.zeros(shape, dtype=dtype, device=device)


def _infer_state_keys(optimizer: torch.optim.Optimizer) -> List[str]:
    """Infer expected state keys from optimizer defaults.

    Supports Adam, AdamW, and SGD. Returns keys that would appear
    after at least one optimizer.step() call.
    """
    opt_cls = type(optimizer)
    opt_name = opt_cls.__name__.lower()

    if "adam" in opt_name:
        keys = ["step", "exp_avg", "exp_avg_sq"]
        if optimizer.defaults.get("amsgrad", False):
            keys.append("max_exp_avg_sq")
        return keys

    if "sgd" in opt_name:
        keys = []
        if optimizer.defaults.get("momentum", 0) != 0:
            keys.append("momentum_buffer")
        return keys

    logger.warning(
        "Cannot infer state keys for optimizer type '%s'. "
        "DCP load template may be incomplete.",
        opt_cls.__name__,
    )
    return []

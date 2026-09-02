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
"""Trainer-owned model initialization dtype conversion."""

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel import DTensor
from hyper_parallel.core.fully_shard.hsdp_utils import get_hsdp_state


_MODEL_INIT_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


@dataclass
class _BufferSnapshot:
    """Non-persistent buffer state that should survive dtype conversion."""

    module: torch.nn.Module
    name: str
    tensor: torch.Tensor


def _resolve_model_init_dtype(
        model_init_dtype: Optional[Literal["float16", "bfloat16", "float32"]],
) -> Optional[torch.dtype]:
    """Resolve the configured final model initialization dtype."""
    if model_init_dtype is None:
        return None
    if model_init_dtype not in _MODEL_INIT_DTYPES:
        raise ValueError(
            "model_init_dtype must be one of float16, bfloat16, float32, "
            f"or null; got {model_init_dtype!r}"
        )
    return _MODEL_INIT_DTYPES[model_init_dtype]


def _model_tensor_identities(model: torch.nn.Module) -> Dict[str, int]:
    """Capture Parameter and DTensor-buffer identities by FQN."""
    identities = {
        f"parameter:{name}": id(parameter)
        for name, parameter in model.named_parameters(remove_duplicate=False)
    }
    identities.update({
        f"buffer:{name}": id(buffer)
        for name, buffer in model.named_buffers(remove_duplicate=False)
        if isinstance(buffer, DTensor)
    })
    return identities


def _dtensor_layouts(model: torch.nn.Module) -> Dict[int, tuple[Any, tuple[Any, ...]]]:
    """Capture DTensor mesh and placements before dtype conversion."""
    return {
        id(tensor): (tensor.device_mesh, tuple(tensor.placements))
        for tensor in list(model.parameters()) + list(model.buffers())
        if isinstance(tensor, DTensor)
    }


def _snapshot_nonpersistent_buffers(model: torch.nn.Module) -> list[_BufferSnapshot]:
    """Clone config-derived buffers before global dtype conversion."""
    snapshots = []
    for module in model.modules():
        for name in module._non_persistent_buffers_set:  # pylint: disable=protected-access
            buffer = module._buffers.get(name)  # pylint: disable=protected-access
            if buffer is None or isinstance(buffer, DTensor) or buffer.is_meta:
                continue
            snapshots.append(_BufferSnapshot(module, name, buffer.detach().clone()))
    return snapshots


def _restore_nonpersistent_buffers(snapshots: list[_BufferSnapshot]) -> None:
    """Restore non-persistent buffers with their original values and dtypes."""
    for snapshot in snapshots:
        current = snapshot.module._buffers.get(snapshot.name)  # pylint: disable=protected-access
        device = snapshot.tensor.device if current is None or current.is_meta else current.device
        snapshot.module._buffers[snapshot.name] = snapshot.tensor.to(device=device)  # pylint: disable=protected-access


def _nonpersistent_buffer_names(model: torch.nn.Module) -> set[str]:
    """Return fully qualified names of non-persistent buffers."""
    names = set()
    for module_name, module in model.named_modules(remove_duplicate=False):
        for tensor_name in module._non_persistent_buffers_set:  # pylint: disable=protected-access
            if module._buffers.get(tensor_name) is None:  # pylint: disable=protected-access
                continue
            names.add(f"{module_name}.{tensor_name}" if module_name else tensor_name)
    return names


def _refresh_hsdp_precision_state(model: torch.nn.Module) -> None:
    """Refresh FSDP storage and dtype metadata after model conversion."""
    visited_states = set()
    for module in model.modules():
        hsdp_state = get_hsdp_state(module)
        if hsdp_state is None or id(hsdp_state) in visited_states:
            continue
        visited_states.add(id(hsdp_state))
        for hsdp_param in hsdp_state.hsdp_params:
            hsdp_param.reset_sharded_param()
            hsdp_param.init_dtype_attrs(hsdp_state.mp_policy)


def _validate_model_init_dtype(
        model: torch.nn.Module,
        target_dtype: torch.dtype,
) -> None:
    """Validate floating model parameters and buffers after conversion."""
    preserved_buffers = _nonpersistent_buffer_names(model)
    mismatched = [
        name
        for name, tensor in (
            list(model.named_parameters(remove_duplicate=False))
            + list(model.named_buffers(remove_duplicate=False))
        )
        if tensor.is_floating_point() and tensor.dtype != target_dtype and name not in preserved_buffers
    ]
    if mismatched:
        raise RuntimeError(
            "Model initialization dtype conversion failed for: "
            f"{', '.join(sorted(mismatched))}"
        )


def apply_model_init_dtype(
        model: torch.nn.Module,
        model_init_dtype: Optional[Literal["float16", "bfloat16", "float32"]],
) -> None:
    """Convert initialized model floating state to the configured final dtype.

    Args:
        model: Model whose loaded or newly initialized floating state is converted.
        model_init_dtype: Final initialization dtype, or ``None`` for no conversion.
    """
    target_dtype = _resolve_model_init_dtype(model_init_dtype)
    if target_dtype is None:
        return

    identities_before = _model_tensor_identities(model)
    layouts_before = _dtensor_layouts(model)
    nonpersistent_buffers = _snapshot_nonpersistent_buffers(model)
    swap_on_conversion = torch.__future__.get_swap_module_params_on_conversion()
    torch.__future__.set_swap_module_params_on_conversion(True)
    try:
        model.to(dtype=target_dtype)
    finally:
        torch.__future__.set_swap_module_params_on_conversion(swap_on_conversion)
    _restore_nonpersistent_buffers(nonpersistent_buffers)

    if _model_tensor_identities(model) != identities_before:
        raise RuntimeError(
            "Model initialization dtype conversion replaced a Parameter or DTensor identity"
        )
    for tensor in list(model.parameters()) + list(model.buffers()):
        previous_layout = layouts_before.get(id(tensor))
        if previous_layout is None:
            continue
        if (
            tensor.device_mesh is not previous_layout[0]
            or tuple(tensor.placements) != previous_layout[1]
        ):
            raise RuntimeError(
                "Model initialization dtype conversion changed a DTensor layout"
            )

    _refresh_hsdp_precision_state(model)
    _validate_model_init_dtype(model, target_dtype)


__all__ = ["apply_model_init_dtype"]

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
"""Optimizer and checkpoint-layout observations for Trainer validation."""
# pylint: disable=forbidden-backend-import

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

from hyper_parallel.tools.model_integration.schemas import (
    FindingSeverity,
    ModelIntegrationFinding,
    SCHEMA_VERSION,
)


def tensor_layout(value: Any) -> dict[str, Any]:
    """Return logical/local shape and distributed layout without custom chunks."""
    to_local = getattr(value, "to_local", None)
    local = to_local() if callable(to_local) else value
    layout = getattr(value, "layout", None)
    mesh = getattr(value, "device_mesh", None)
    placements = getattr(value, "placements", None)
    if layout is not None:
        mesh = getattr(layout, "mesh", mesh)
        placements = getattr(layout, "placements", placements)
    mesh_tensor = getattr(mesh, "mesh", None)
    return {
        "global_shape": [int(size) for size in getattr(value, "shape", ())],
        "local_shape": [int(size) for size in getattr(local, "shape", ())],
        "dtype": str(getattr(value, "dtype", None)),
        "device": str(getattr(local, "device", None)),
        "mesh_dim_names": list(getattr(mesh, "mesh_dim_names", ()) or ()),
        "mesh_shape": [int(size) for size in getattr(mesh_tensor, "shape", ())],
        "placements": [repr(placement) for placement in (placements or ())],
        "is_dtensor": callable(to_local),
    }


def _same_logical_layout(first: Any, second: Any) -> bool:
    first_layout = tensor_layout(first)
    second_layout = tensor_layout(second)
    return all(
        first_layout[name] == second_layout[name]
        for name in ("global_shape", "mesh_dim_names", "mesh_shape", "placements")
    )


def optimizer_leaves(optimizer: Any) -> tuple[Any, ...]:
    """Return optimizer leaves through list, chain, and wrapper containers."""
    chained = getattr(optimizer, "chained_optimizers", None)
    if chained is not None:
        return tuple(leaf for item in chained for leaf in optimizer_leaves(item))
    inner = getattr(optimizer, "optimizer", None)
    if inner is not None and inner is not optimizer:
        return optimizer_leaves(inner)
    if isinstance(optimizer, (list, tuple)):
        return tuple(leaf for item in optimizer for leaf in optimizer_leaves(item))
    return (optimizer,)


def optimizer_state_for_parameter(optimizer: Any, parameter: Any) -> Mapping[str, Any]:
    """Find one parameter's state through supported optimizer wrappers."""
    for leaf in optimizer_leaves(optimizer):
        state = getattr(leaf, "state", None)
        if isinstance(state, Mapping) and parameter in state:
            value = state[parameter]
            return value if isinstance(value, Mapping) else {}
    return {}


def _layout_mismatch_finding(
        parameter_fqn: str,
        owner_name: str,
        expected: Any,
        actual: Any,
) -> ModelIntegrationFinding:
    return ModelIntegrationFinding(
        code="HP-OPT-001",
        phase="D6",
        owner_fqn=parameter_fqn,
        severity=FindingSeverity.ERROR,
        message=f"{owner_name} does not preserve its parameter's global layout",
        facts={
            "parameter": tensor_layout(expected),
            owner_name: tensor_layout(actual),
        },
        why_unsafe=(
            "optimizer state and gradients must describe the same logical tensor; "
            "a local-only or differently placed value cannot be resharded safely"
        ),
        remediation=(
            "preserve the parameter DTensor global shape, mesh, and placements when "
            "creating main gradients, main parameters, and optimizer moments"
        ),
        related_config=("optimizer.fp32_main_params", "checkpoint.restore_optimizer"),
    )


def validate_optimizer_layout(model: Any, optimizer: Any) -> list[ModelIntegrationFinding]:
    """Validate param/main-param/main-grad/moment layouts from live objects."""
    findings = []
    for parameter_fqn, model_parameter in model.named_parameters(remove_duplicate=False):
        main_parameter = getattr(model_parameter, "main_param", model_parameter)
        if not _same_logical_layout(model_parameter, main_parameter):
            findings.append(
                _layout_mismatch_finding(
                    parameter_fqn,
                    "main_parameter",
                    model_parameter,
                    main_parameter,
                )
            )
            continue
        gradient = getattr(model_parameter, "main_grad", None)
        if gradient is None:
            gradient = getattr(main_parameter, "grad", None)
        if gradient is not None and not _same_logical_layout(main_parameter, gradient):
            findings.append(
                _layout_mismatch_finding(
                    parameter_fqn,
                    "main_gradient",
                    main_parameter,
                    gradient,
                )
            )
        for state_name, state_value in optimizer_state_for_parameter(
                optimizer, main_parameter
        ).items():
            if torch.is_tensor(state_value) and state_value.ndim > 0:
                if not _same_logical_layout(main_parameter, state_value):
                    findings.append(
                        _layout_mismatch_finding(
                            parameter_fqn,
                            f"optimizer_state.{state_name}",
                            main_parameter,
                            state_value,
                        )
                    )
        layout = tensor_layout(main_parameter)
        if (
            math.prod(layout["global_shape"])
            and not math.prod(layout["local_shape"])
            and not layout["is_dtensor"]
        ):
            findings.append(
                ModelIntegrationFinding(
                    code="HP-OPT-002",
                    phase="D8",
                    owner_fqn=parameter_fqn,
                    severity=FindingSeverity.ERROR,
                    message="empty optimizer shard has no global DTensor layout",
                    facts={"main_parameter": layout},
                    why_unsafe=(
                        "DCP cannot place an empty local tensor into its logical global tensor "
                        "without mesh and placement metadata"
                    ),
                    remediation="keep uneven FSDP shards as DTensors through optimizer state creation",
                    related_config=("parallel.fsdp", "optimizer.fp32_main_params"),
                )
            )
    return findings


def snapshot_training_layout(model: Any, optimizer: Any) -> dict[str, Any]:
    """Build model/main-gradient/optimizer-state layout evidence."""
    records = {}
    for parameter_fqn, model_parameter in model.named_parameters(remove_duplicate=False):
        main_parameter = getattr(model_parameter, "main_param", model_parameter)
        record = {
            "model_parameter": tensor_layout(model_parameter),
            "main_parameter": tensor_layout(main_parameter),
        }
        gradient = getattr(model_parameter, "main_grad", None)
        if gradient is None:
            gradient = getattr(main_parameter, "grad", None)
        if gradient is not None:
            record["main_gradient"] = tensor_layout(gradient)
        state = optimizer_state_for_parameter(optimizer, main_parameter)
        record["optimizer_state"] = {
            name: tensor_layout(value)
            for name, value in sorted(state.items())
            if torch.is_tensor(value)
        }
        records[parameter_fqn] = record
    return {"schema_version": SCHEMA_VERSION, "parameters": records}


def flatten_tensor_layouts(value: Any, prefix: str = "") -> dict[str, dict[str, Any]]:
    """Flatten tensor leaves from a checkpoint payload into layout records."""
    records = {}
    if torch.is_tensor(value):
        records[prefix or "<root>"] = tensor_layout(value)
    elif isinstance(value, Mapping):
        for name, child in value.items():
            child_prefix = f"{prefix}.{name}" if prefix else str(name)
            records.update(flatten_tensor_layouts(child, child_prefix))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}.{index}" if prefix else str(index)
            records.update(flatten_tensor_layouts(child, child_prefix))
    return records


__all__ = [
    "flatten_tensor_layouts",
    "optimizer_leaves",
    "optimizer_state_for_parameter",
    "snapshot_training_layout",
    "tensor_layout",
    "validate_optimizer_layout",
]

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
"""Source ownership and invocation metadata shared by linear-attention families."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any

import torch
from torch import nn

from hyper_parallel.components.modules.attention.linear_attention.sequence_layout import (
    resolve_packed_sequence,
)
from hyper_parallel.core.dtensor.dtensor import DTensor


def adopt_source_parameters(target: nn.Module, source: nn.Module, names: tuple[str, ...]) -> None:
    """Retain source registration order, aliases, frozen weights and checkpoint keys.

    Args:
        target: Concrete family member, never a wrapper owning the source module.
        source: Supported source layer or local family being promoted to CP.
        names: Algorithm-owned top-level parameter and submodule names.
    """
    unknown = {name.split('.')[0] for name, _ in source.named_parameters()} - set(names)
    if unknown:
        raise ValueError(f"Unsupported linear-attention parameter owners: {sorted(unknown)}")
    # Raw registries preserve aliases, None slots and buffer persistence that
    # named_parameters()/named_buffers() cannot expose together.
    for name, parameter in source._parameters.items():  # pylint: disable=protected-access
        if name in names:
            target.register_parameter(name, parameter)
    for name, child in source._modules.items():  # pylint: disable=protected-access
        # Parameterless source helpers also belong to the registered-module contract.
        # Unknown parameter owners were rejected above, so preserving them is safe.
        target.add_module(name, child)
    for name, buffer in source._buffers.items():  # pylint: disable=protected-access
        persistent = name not in source._non_persistent_buffers_set  # pylint: disable=protected-access
        target.register_buffer(name, buffer, persistent=persistent)
    # Do not call train() recursively: source children may intentionally differ.
    target.training = source.training


def prepare_linear_attention_input(hidden_states: torch.Tensor, options: dict[str, Any],
                                   cp_rank: int = 0, cp_size: int = 1) -> tuple:
    """Resolve per-call metadata without storing the active batch on a module.

    Args:
        hidden_states: Local [B,T,C] or flattened [T,C] activations.
        options: Explicit document offsets or an existing PackedSequenceInfo.
        cp_rank: Logical sequence rank, independent of global rank numbering.
        cp_size: Number of equal contiguous token shards.
    """
    if isinstance(hidden_states, DTensor):
        hidden_states = hidden_states.to_local()
    is_flat_input = hidden_states.ndim == 2
    if is_flat_input:
        hidden_states = hidden_states.unsqueeze(0)
    if hidden_states.ndim != 3:
        raise ValueError("Linear attention expects [B,T,C] or [T,C] inputs")
    info = resolve_packed_sequence(options, hidden_states.shape[1], cp_rank=cp_rank, cp_size=cp_size)
    if info is not None and hidden_states.shape[0] != 1:
        raise ValueError("Packed linear attention requires batch dimension one")
    return hidden_states, info, is_flat_input


def copy_unbound_linear_attention(module: nn.Module, memo: dict) -> nn.Module:
    """Copy weights before binding, retaining immutable family request metadata.

    Args:
        module: Local or not-yet-bound CP family member.
        memo: Python deepcopy alias map.

    Returns:
        An independent parameter copy with the same logical family request.
    """
    if getattr(module, "cp_context", None) is not None or getattr(module, "_hp_parallel_binding", None) is not None:
        raise RuntimeError("Copy linear-attention weights before binding communication resources")
    result = type(module).__new__(type(module))
    memo[id(module)] = result
    state = dict(module.__dict__)
    info = state.pop("_hp_family_build_info", None)
    result.__dict__.update(deepcopy(state, memo))
    if info is not None:
        if info.layout is not None:
            raise RuntimeError("Copy linear-attention weights before applying a distributed layout")
        # Requests/selection contain only immutable execution options. Configuration
        # snapshots are independent so changing a copied config cannot affect promotion.
        # This copy retains the same private record owned by module_family.
        result._hp_family_build_info = replace(  # pylint: disable=protected-access
            info, config_snapshot=deepcopy(info.config_snapshot, memo))
    return result


def validate_backend_options(options: Any) -> str:
    """Validate the local algorithm independently of the CP topology.

    Args:
        options: Explicit backend options or invocation document metadata.
    """
    backend = options.get("backend", "eager")
    if backend not in ("eager", "triton", "ascendc"):
        raise ValueError("backend must be eager, triton or ascendc")
    conv = options.get("conv_backend")
    if conv is not None and conv not in ("eager", "triton", "ascendc"):
        raise ValueError("conv_backend must be eager, triton or ascendc")
    chunk = options.get("chunk_size", 64)
    if (not isinstance(chunk, int) or isinstance(chunk, bool)) or chunk < 1 or (backend != "eager" and chunk != 64):
        raise ValueError("chunk_size must be positive; fused backends require 64")
    if options.get("input_layout", "BSH") not in ("BSH", "TH"):
        raise ValueError("input_layout must be BSH or TH; TH declares flat packed token axis 0")
    return backend

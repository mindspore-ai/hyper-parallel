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
"""Check independent expert parameter ownership without restoring full weights."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from hyper_parallel.core.dtensor.dtensor import DTensor


@dataclass(frozen=True)
class _Span:
    device: torch.device
    begin: int
    end: int
    label: str


def _dense_span(parameter: torch.Tensor, label: str) -> _Span | None:
    """Describe addressed bytes without reading values or invoking a collective."""
    local = parameter.to_local() if isinstance(parameter, DTensor) else parameter
    if not isinstance(local, torch.Tensor) or local.layout != torch.strided or local.is_meta:
        raise ValueError(f"{label} must have materialized strided local storage")
    if local.numel() == 0:
        return None
    expected = 1
    for stride, size in sorted((stride, size) for size, stride in zip(local.shape, local.stride()) if size > 1):
        if stride != expected:
            raise ValueError(f"{label} must have a dense non-overlapping local layout")
        expected *= size
    itemsize = local.element_size()
    begin = local.data_ptr()
    end = begin + local.numel() * itemsize
    storage_end = local.untyped_storage().data_ptr() + local.untyped_storage().nbytes()
    if not begin or local.storage_offset() < 0 or end > storage_end:
        raise ValueError(f"{label} has invalid or undersized local storage")
    return _Span(local.device, begin, end, label)


def _managed_shards(owner: torch.nn.Module, parameters: tuple[torch.nn.Parameter, ...]) -> tuple:
    """Match current owner parameters to this unit while returning stable shards."""
    scheduler = getattr(owner, "hsdp_scheduler", None)
    if scheduler is None:
        return parameters
    if len(scheduler.modules) != 1 or scheduler.cell is not owner or scheduler.modules[0] is not owner:
        raise ValueError("expert owner must be its own independent FSDP unit")
    state = scheduler.hsdp_state
    managed = tuple(state.hsdp_params)
    current = tuple(item.sharded_param if state.is_shard else item.unsharded_param for item in managed)
    if len(current) != len(parameters) or {id(value) for value in current} != {id(value) for value in parameters}:
        raise ValueError("FSDP manager must cover exactly this owner's parameters")
    return tuple(item.sharded_param for item in managed)


def validate_parameter_ownership(owners: Sequence[torch.nn.Module]) -> None:
    """Reject tied or overlapping expert parameters before any W submission.

    Args:
        owners: Registered expert modules, after fully_shard wrapping if used.

    Raises:
        ValueError: An owner is repeated, parameters are tied, FSDP coverage is
            incomplete, local storage is unsupported, or parameter bytes overlap.

    Note:
        This checks only metadata and stable local shards. It does not restore
        full weights, move tensors, access their values, or create streams.
        The caller must converge local failures before starting payload work.
        Changing parameter ownership or rebinding storage requires new admission.
    """
    identities, shard_identities, spans = {}, {}, []
    if len({id(owner) for owner in owners}) != len(owners):
        raise ValueError("expert owners must be distinct modules")
    for index, owner in enumerate(owners):
        named = tuple(owner.named_parameters(recurse=True, remove_duplicate=False))
        for name, parameter in named:
            label = f"owner[{index}].{name}"
            previous = identities.get(id(parameter))
            if previous is not None:
                raise ValueError(f"tied parameter: {previous} and {label}")
            identities[id(parameter)] = label
        shards = _managed_shards(owner, tuple(parameter for _, parameter in named))
        for position, parameter in enumerate(shards):
            label = f"owner[{index}].shard[{position}]"
            previous = shard_identities.get(id(parameter))
            if previous is not None:
                raise ValueError(f"tied sharded parameter: {previous} and {label}")
            shard_identities[id(parameter)] = label
            span = _dense_span(parameter, label)
            if span is not None:
                spans.append(span)
    spans.sort(key=lambda span: (str(span.device), span.begin, span.end))
    for previous, current in zip(spans, spans[1:]):
        if previous.device == current.device and current.begin < previous.end:
            raise ValueError(f"overlapping parameter storage: {previous.label} and {current.label}")

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
"""Gather and export rollout weights using the existing model export rules."""

from collections.abc import Iterator
from typing import Any

import torch
from torch import Tensor, nn
from verl.utils.device import get_device_id, get_device_name

from hyper_parallel import DTensor, hsdp_sync_stream
from hyper_parallel.core.fully_shard.api import get_model_state_dict
from hyper_parallel.components.checkpoint.weight_conversion import revert_weight_conversion
from hyper_parallel.components.checkpoint.weight_verifier import _InverseModel
from hyper_parallel.models.registry import resolve_model_adapter


def _revert_weight_conversion(
    model: nn.Module,
    weights: dict[str, Tensor],
    conversions: Any,
) -> dict[str, Tensor]:
    """Apply explicit rules through HyperParallel's native API without mutating the model."""
    return revert_weight_conversion(_InverseModel(model, conversions), weights)


def get_per_tensor_param(model: nn.Module) -> Iterator[tuple[str, Tensor]]:
    """Gather and restore each parameter to rollout format without changing residency."""
    hsdp_sync_stream()
    device = torch.device(get_device_name(), get_device_id())
    adapter = resolve_model_adapter(model)
    replacements = getattr(model, "_hp_used_replacement_weight_conversions", None)
    base = getattr(model, "_hp_used_base_weight_conversions", None)
    conversion_steps = [rules for rules in (replacements, base) if rules]

    for name, tensor in get_model_state_dict(model).items():
        tensor = tensor.to(device, non_blocking=True)
        tensor = tensor.full_tensor() if isinstance(tensor, DTensor) else tensor
        converted = {name: tensor}
        if adapter is not None and adapter.checkpoint_export is not None:
            converted, _ = adapter.checkpoint_export(model, converted, True)
        for rules in conversion_steps:
            converted = _revert_weight_conversion(model, converted, rules)
        yield from converted.items()

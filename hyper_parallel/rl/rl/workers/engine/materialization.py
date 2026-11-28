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
"""Restore Qwen3 model-owned RoPE buffers after meta storage materialization."""

from typing import Any, Callable

import torch
from torch import nn

from hyper_parallel.models.materialization import MaterializationContext, register_rebuildable_buffer


def _rope_factory(rotary: nn.Module) -> Callable[[MaterializationContext], torch.Tensor]:
    """Use the installed authoritative HF RoPE implementation and effective model config."""
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS  # pylint: disable=C0415

    rope_function = (rotary.compute_default_rope_parameters if rotary.rope_type == "default"
                     else ROPE_INIT_FUNCTIONS[rotary.rope_type])

    def rebuild(context: MaterializationContext) -> torch.Tensor:
        """Compute canonical FP32 frequencies on CPU, then transfer to the target device."""
        # Both model construction paths use canonical frequencies on CPU. NPU pow
        # differs by FP32 ULPs that can flip BF16 cos/sin at later positions.
        # HF may create intermediate tensors without an explicit device before
        # applying its device argument. Keep those factories out of a caller's
        # meta/NPU default-device context as well.
        with torch.device("cpu"):
            value, _ = rope_function(rotary.config, device=torch.device("cpu"))
        return value.to(device=context.device)

    return rebuild


def register_qwen3_materialization(model: Any) -> None:
    """Declare native non-persistent buffers before sharding and to_empty lose their values."""
    from rl.unified_execution.models import get_model_family  # pylint: disable=C0415

    rotary_type = get_model_family(model.config.model_type).execution_types()["rotary"]

    if getattr(model, "_rl_rope_materialization_registered", False):
        return
    for module in model.modules():
        if isinstance(module, rotary_type):
            factory = _rope_factory(module)
            for name in ("inv_freq", "original_inv_freq"):
                if name in module._buffers:  # pylint: disable=protected-access
                    register_rebuildable_buffer(module, name, factory=factory)
                    buffer = getattr(module, name)
                    if not buffer.is_meta:
                        # Single-rank from_config keeps live storage and never
                        # enters HP's post-meta rebuild. Apply the same recipe
                        # now without changing buffer identity or parameters.
                        context = MaterializationContext(reason="random_init", device=buffer.device)
                        with torch.no_grad():
                            buffer.copy_(factory(context))

    setattr(model, "_rl_rope_materialization_registered", True)

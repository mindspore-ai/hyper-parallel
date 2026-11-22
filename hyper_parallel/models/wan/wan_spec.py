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
"""Wan source boundaries and its official Diffusers attention-family execution."""

from collections.abc import Mapping
from typing import Any

from torch import nn

from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.replacement import ModuleReplacementRequest
from hyper_parallel.models.wan.components.attention import WanAttention


def get_replacement_rules(*, model: nn.Module, context: Mapping[str, Any]) -> tuple[ModuleReplacementRequest, ...]:
    """Select the NPU processor when applicable without storing replacement globs in recipes."""
    del model
    backend = "_native_npu" if context.get("device_type") == "npu" else None
    return (ModuleReplacementRequest(match=("blocks.*.attn1", "blocks.*.attn2"),
                                     target=WanAttention, kwargs={"backend": backend}),)


def _load_dit_provider() -> Any:
    """Keep optional Diffusers model and condition encoders out of registry discovery."""
    from hyper_parallel.models.wan.dit_adapter import WanDiTModelProvider  # pylint: disable=C0415

    return WanDiTModelProvider


MODEL_SPEC = ModelAdapterSpec(
    architecture="WanTransformer3DModel", model_type="wan", replacements=get_replacement_rules, dit=_load_dit_provider,
)

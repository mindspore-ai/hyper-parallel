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
"""Register the upstream Diffusers Wan model and its HyperParallel adapters."""

from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.registry import register_model_adapter


def _load_replacements():
    """Return Wan module-replacement declarations lazily."""
    from hyper_parallel.models.wan.adapter import replacements  # pylint: disable=C0415
    return replacements


def _load_dit_provider():
    """Return the Wan DiT construction provider lazily."""
    from hyper_parallel.models.wan.dit_adapter import WanDiTModelProvider  # pylint: disable=C0415
    return WanDiTModelProvider


WAN_ADAPTER_SPEC = ModelAdapterSpec(
    architecture="WanTransformer3DModel",
    model_type="wan",
    replacements=_load_replacements,
    dit=_load_dit_provider,
)

register_model_adapter(WAN_ADAPTER_SPEC)

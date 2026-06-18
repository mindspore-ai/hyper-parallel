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
"""Context-parallel preparation entrypoint for the LlamaFactory integration."""
from torch import nn

from .loss import _enable_context_parallel_loss_patch
from .models import get_context_parallel_model_patches


def cp_prepare_model(model: nn.Module, accelerator, hp_args) -> nn.Module:
    """Apply all trainer-owned context-parallel runtime preparation to one model."""
    if getattr(hp_args, "cp_size", 1) <= 1:
        return model

    matched_patches = [patch for patch in get_context_parallel_model_patches() if patch.supports(model)]
    mesh = None

    def _get_mesh():
        from hyper_parallel.integration.llamafactory.utils import _build_device_mesh  # pylint: disable=import-outside-toplevel

        nonlocal mesh
        if mesh is None:
            mesh = _build_device_mesh(accelerator, hp_args)
        return mesh

    for patch in matched_patches:
        patch.prepare(model, hp_args, _get_mesh)

    _enable_context_parallel_loss_patch(model, hp_args)
    return model

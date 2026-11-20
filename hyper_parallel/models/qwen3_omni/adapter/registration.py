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
"""Qwen3-Omni architecture identity and distributed capabilities."""

import torch

from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.registry import register_model_adapter


def _load_stock_model_patches():
    """Return the entrypoint applying the family's stock-Transformers fixes."""

    from hyper_parallel.models.qwen3_omni.adapter.model_patches import (  # pylint: disable=C0415
        apply_stock_model_patches,
    )
    return apply_stock_model_patches


def _load_materialization():
    """Return the derived positional-buffer rebuild provider."""

    from hyper_parallel.models.qwen3_omni.adapter.materialization import (  # pylint: disable=C0415
        register_omni_derived_buffer_materialization,
    )
    return register_omni_derived_buffer_materialization


def _load_fsdp_wrap_modules():
    """Return the provider declaring the vision tower's FSDP2 child units.

    The omni ``visual`` branch (patch embed / blocks / merger) lives outside
    the HF decoder-container convention; declare its projection modules as
    explicit FSDP units so their all-gather hooks fire on their own
    ``__call__`` (a merger iterates an internal ``ModuleList`` by hand, so
    wrapping the grandchildren would register hooks that never fire and
    leave parameters sharded at compute time).
    """

    def _fsdp_wrap_modules(model):
        visual = getattr(model, "visual", None)
        if visual is None:
            thinker = getattr(model, "thinker", None)
            visual = getattr(thinker, "visual", None) if thinker is not None else None
        if visual is None:
            return []
        # ``model.visual`` exists only on the bare thinker; the conditional-
        # generation wrapper nests it under ``thinker``.
        prefix = "visual" if hasattr(model, "visual") else "thinker.visual"
        fqns = []
        for name in ("patch_embed", "merger"):
            child = getattr(visual, name, None)
            if isinstance(child, torch.nn.Module) and not isinstance(child, torch.nn.ModuleList):
                fqns.append(f"{prefix}.{name}")
        return fqns

    return _fsdp_wrap_modules


QWEN3_OMNI_ADAPTER_SPEC = ModelAdapterSpec(
    architecture="Qwen3OmniMoeForConditionalGeneration",
    model_type="qwen3_omni_moe",
    materialization=_load_materialization,
    stock_model_patches=_load_stock_model_patches,
    fsdp_wrap_modules=_load_fsdp_wrap_modules,
)

register_model_adapter(QWEN3_OMNI_ADAPTER_SPEC)

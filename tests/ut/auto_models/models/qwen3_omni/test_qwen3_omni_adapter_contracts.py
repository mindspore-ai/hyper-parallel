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
"""CPU contracts for the Qwen3-Omni family adapter registration."""
# pylint: disable=protected-access

import torch
from torch import nn

from hyper_parallel.models import get_model_adapter
from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.materialization import (
    fp32_rebuildable_buffer_ids,
    register_rebuildable_buffer,
    restore_fp32_rebuildable_buffers,
)
from hyper_parallel.models.qwen3_omni.adapter.materialization import (
    register_omni_derived_buffer_materialization,
)
from hyper_parallel.models.qwen3_omni.adapter.model_patches import apply_stock_model_patches
from tests.common.mark_utils import arg_mark


@arg_mark(
    plat_marks=["cpu_linux", "cpu_macos"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_model_adapter_resolves_for_every_identity_spelling() -> None:
    """The registry maps model_type, architecture and normalized forms to one spec."""
    for identity in ("qwen3_omni_moe", "Qwen3OmniMoeForConditionalGeneration", "qwen3omnimoe"):
        spec = get_model_adapter(identity)
        assert isinstance(spec, ModelAdapterSpec), f"no adapter for {identity}"
        assert spec.architecture == "Qwen3OmniMoeForConditionalGeneration"
        assert spec.model_type == "qwen3_omni_moe"


@arg_mark(
    plat_marks=["cpu_linux", "cpu_macos"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_model_adapter_declares_its_providers() -> None:
    """Materialization, stock patches and vision FSDP units are family-declared."""
    spec = get_model_adapter("qwen3_omni_moe")

    assert spec.materialization is not None
    assert spec.stock_model_patches is not None
    assert callable(spec.stock_model_patches())
    assert spec.fsdp_wrap_modules is not None
    assert callable(spec.fsdp_wrap_modules())


class _VisualStub(nn.Module):
    """Minimal omni-style vision tower for FSDP unit declarations."""

    def __init__(self) -> None:
        """Build the projection modules the provider must declare."""
        super().__init__()
        self.patch_embed = nn.Linear(4, 4)
        self.merger = nn.Linear(4, 4)


def test_fsdp_wrap_modules_declares_vision_projection_units() -> None:
    """The provider names patch_embed and merger on the nested vision tower."""
    spec = get_model_adapter("qwen3_omni_moe")
    provider = spec.fsdp_wrap_modules()

    thinker = nn.Module()
    thinker.visual = _VisualStub()  # type: ignore[attr-defined]
    wrapper = nn.Module()
    wrapper.thinker = thinker  # type: ignore[attr-defined]

    assert provider(wrapper) == ["thinker.visual.patch_embed", "thinker.visual.merger"]
    assert provider(nn.Module()) == []


def test_stock_model_patches_entry_is_idempotent_without_transformers() -> None:
    """Applying the patch set is safe on any install (guards handle absence)."""
    apply_stock_model_patches()
    apply_stock_model_patches()


def test_derived_buffer_materialization_keeps_fp32_through_dtype_cast() -> None:
    """Rebuild recipes restore fp32 positional buffers after a model dtype cast."""
    dim = 8
    module = nn.Module()
    module.register_buffer(
        "inv_freq",
        1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)),
        persistent=False,
    )

    register_rebuildable_buffer(module, "inv_freq")
    register_omni_derived_buffer_materialization(module)
    assert "inv_freq" in module._hp_rebuildable_buffer_specs

    model = nn.Module()
    model.sub = module  # type: ignore[attr-defined]
    model.to(torch.bfloat16)
    assert module.inv_freq.dtype == torch.bfloat16

    restored = restore_fp32_rebuildable_buffers(model)
    assert restored == ["inv_freq"]
    assert module.inv_freq.dtype == torch.float32
    assert id(module.inv_freq) in fp32_rebuildable_buffer_ids(model)

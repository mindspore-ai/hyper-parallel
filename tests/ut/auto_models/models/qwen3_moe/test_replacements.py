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
"""Qwen3-MoE experts replacement: the Transformers version matrix (plan §6.1.1).

The ``*.mlp.experts`` boundary supports only the batched Transformers
layout (``gate_up_proj``/``down_proj`` parameters); the legacy 4.57
``ModuleList`` layout is explicitly rejected with a clear error. Both
layouts are faked by the shared conftest fixtures so the support matrix is
asserted explicitly instead of depending on whichever Transformers version
happens to be installed (plan §14.5). The contract validation is pure
Python and runs before the lazy NPU-only ``modules`` import, so the whole
matrix is testable on CPU-only checkouts (Gate-1).
"""
# pylint: disable=wrong-import-position

import sys
import types
import unittest.mock

import pytest
from torch import nn

from hyper_parallel.models.qwen3_moe.adapter.conversion.module_replacement import (
    _validate_batched_experts_contract,
    replace_qwen3_moe_flash_attention,
    replace_qwen3_moe_grouped_experts,
)
from hyper_parallel.models.registry import get_model_adapter


def test_batched_layout_is_accepted(batched_experts):
    """The batched gate_up_proj/down_proj layout passes validation."""
    _validate_batched_experts_contract(batched_experts, "model.layers.0.mlp.experts")


def test_legacy_module_list_layout_is_rejected(legacy_experts):
    """The legacy 4.57 ModuleList layout fails with a clear error."""
    with pytest.raises(TypeError, match="legacy Transformers expert layout"):
        _validate_batched_experts_contract(legacy_experts, "model.layers.0.mlp")


def test_bare_module_list_target_is_rejected():
    """A bare ModuleList target (legacy ``*.mlp.experts``) is rejected."""
    with pytest.raises(TypeError, match="legacy Transformers expert layout"):
        _validate_batched_experts_contract(nn.ModuleList(), "model.layers.0.mlp.experts")


def test_missing_batched_parameters_are_named():
    """A module without the batched parameters names what is missing."""
    with pytest.raises(TypeError, match="gate_up_proj"):
        _validate_batched_experts_contract(nn.Module(), "model.layers.0.mlp.experts")


def test_factory_rejects_legacy_before_npu_import(legacy_experts):
    """The factory rejects the legacy layout on CPU-only checkouts.

    The rejection must fire before the lazy ``modules`` import so the
    version matrix holds even where the NPU-only functional backends
    cannot be imported at all.
    """
    with pytest.raises(TypeError, match="legacy Transformers expert layout"):
        replace_qwen3_moe_grouped_experts(
            module=legacy_experts, module_fqn="model.layers.0.mlp", context={}
        )


def test_factory_delegates_batched_layout_to_grouped_experts(batched_experts):
    """The accepted batched layout is handed to ``modules.GroupedExperts``."""
    fake_modules = types.ModuleType("hyper_parallel.components.modules")

    class _FakeGroupedExperts(nn.Module):
        """Stand-in recording the delegation arguments."""

        def __init__(self, *, module, module_fqn, context):
            super().__init__()
            self.delegation = (module, module_fqn, context)

    fake_modules.GroupedExperts = _FakeGroupedExperts
    with unittest.mock.patch.dict(
        sys.modules, {"hyper_parallel.components.modules": fake_modules}
    ):
        replaced = replace_qwen3_moe_grouped_experts(
            module=batched_experts,
            module_fqn="model.layers.0.mlp.experts",
            context={"tp": None},
        )
    assert isinstance(replaced, _FakeGroupedExperts), "case: delegation"
    module, module_fqn, context = replaced.delegation
    assert module is batched_experts, "case: module"
    assert module_fqn == "model.layers.0.mlp.experts", "case: module_fqn"
    assert context == {"tp": None}, "case: context"


def test_attention_replaces_q_and_k_norms() -> None:
    """Apply both native norm replacements beneath the new GQA module."""
    source = nn.Module()
    attention = nn.Module()
    attention.q_norm = nn.Identity()
    attention.k_norm = nn.Identity()
    original_norms = (attention.q_norm, attention.k_norm)
    new_norms = (nn.Identity(), nn.Identity())
    module_name = "hyper_parallel.components.modules"
    components = types.ModuleType(module_name)
    build_attention = unittest.mock.Mock(return_value=attention)
    build_norm = unittest.mock.Mock(side_effect=new_norms)
    vars(components).update(GroupedQueryAttention=build_attention, RMSNorm=build_norm)
    context = {}
    with unittest.mock.patch.dict(sys.modules, {module_name: components}):
        result = replace_qwen3_moe_flash_attention(module=source, module_fqn="layer.attn", context=context)
    assert result is attention, "replacement must return the GQA module"
    assert (attention.q_norm, attention.k_norm) == new_norms, "both norm children must be replaced"
    build_attention.assert_called_once_with(module=source, module_fqn="layer.attn", context=context)
    build_norm.assert_has_calls([
        unittest.mock.call(module=module, module_fqn=f"layer.attn.{name}", context=context)
        for name, module in zip(("q_norm", "k_norm"), original_norms)
    ])
    assert build_norm.call_count == 2, "only Q and K norms should be replaced"


def test_lazy_loss_and_codegen_providers() -> None:
    """Resolve the registered loss and shared codegen implementations on demand."""
    spec = get_model_adapter("qwen3_moe")
    assert spec.loss().__name__ == "hyper_parallel.models.qwen3_moe.adapter.runtime.chunked_loss", "Qwen3-MoE loss"
    assert spec.codegen().__name__ == "hyper_parallel.models.qwen3.adapter.codegen", "shared Qwen3 codegen"

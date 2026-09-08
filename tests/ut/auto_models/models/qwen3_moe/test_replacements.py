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

import os
import sys
import types
import unittest.mock

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

import pytest
from torch import nn

from hyper_parallel.models.qwen3_moe.adapter.replacements import (
    _validate_batched_experts_contract,
    replace_qwen3_moe_grouped_experts,
)
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_batched_layout_is_accepted(batched_experts):
    """The batched gate_up_proj/down_proj layout passes validation."""
    _validate_batched_experts_contract(batched_experts, "model.layers.0.mlp.experts")


@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_legacy_module_list_layout_is_rejected(legacy_experts):
    """The legacy 4.57 ModuleList layout fails with a clear error."""
    with pytest.raises(TypeError, match="legacy Transformers expert layout"):
        _validate_batched_experts_contract(legacy_experts, "model.layers.0.mlp")


@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_bare_module_list_target_is_rejected():
    """A bare ModuleList target (legacy ``*.mlp.experts``) is rejected."""
    with pytest.raises(TypeError, match="legacy Transformers expert layout"):
        _validate_batched_experts_contract(nn.ModuleList(), "model.layers.0.mlp.experts")


@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_missing_batched_parameters_are_named():
    """A module without the batched parameters names what is missing."""
    with pytest.raises(TypeError, match="gate_up_proj"):
        _validate_batched_experts_contract(nn.Module(), "model.layers.0.mlp.experts")


@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
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


@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
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

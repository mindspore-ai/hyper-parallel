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
"""Tests for model-agnostic generated-source structural validation."""

import pytest

from hyper_parallel.codegen.check.structure import verify_generated_source_structure
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_structure_check_accepts_unseen_class_and_strategy_name():
    """No registry entry is needed for a future model or enable flag."""
    source = """
class FutureModelBlock:
    future_parallel_enable = False

    def _forward_impl(self, value):
        if self.future_parallel_enable:
            return future_parallel_forward(self, value)
        return super().forward(value)

    def forward(self, value):
        return self._forward_impl(value)
"""

    verify_generated_source_structure({"modeling_future.py": source})


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_structure_check_rejects_duplicate_forward_impl():
    """Duplicate ownership is rejected independently of the class name."""
    source = """
class PreviouslyUnknownBlock:
    def _forward_impl(self, value):
        return value

    def _forward_impl(self, value):
        return value + 1
"""

    with pytest.raises(RuntimeError, match="defines methods more than once"):
        verify_generated_source_structure({"modeling_new.py": source})


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_structure_check_rejects_recursive_forward_impl():
    """A shadowed fallback cannot silently recurse after composition."""
    source = """
class AnyAttention:
    def _forward_impl(self, value):
        return self._forward_impl(value)
"""

    with pytest.raises(RuntimeError, match="_forward_impl calls itself"):
        verify_generated_source_structure({"modeling_new.py": source})


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_structure_check_rejects_unused_enable_flag():
    """Every emitted strategy flag must participate in the forward pipeline."""
    source = """
class AnyBlock:
    arbitrary_enable = False

    def forward(self, value):
        return value
"""

    with pytest.raises(RuntimeError, match="does not read it"):
        verify_generated_source_structure({"modeling_new.py": source})

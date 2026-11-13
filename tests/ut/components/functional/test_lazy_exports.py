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
"""Lazily exported functions must survive direct imports of their modules."""

import importlib
import sys
import unittest
from types import ModuleType
from unittest.mock import patch

from hyper_parallel.components import functional
from tests.common.mark_utils import arg_mark


class TestLazyExports(unittest.TestCase):
    """Check the lazily resolved function exports of the functional package."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_swiglu_survives_importing_its_module_first(self) -> None:
        """
        Feature: Lazy swiglu export of hyper_parallel.components.functional.
        Description: Import the SwiGLU implementation module directly, then import swiglu from the package.
        Expectation: The package import returns the function, not the implementation module.
        """
        exports = functional._EXPORT_TO_MODULE  # pylint: disable=protected-access
        module_name = f"{functional.__name__}.{exports['swiglu']}"
        with patch.dict(sys.modules, {"torch_npu": ModuleType("torch_npu")}), patch.dict(functional.__dict__):
            sys.modules.pop(module_name, None)
            functional.__dict__.pop("swiglu", None)
            implementation = importlib.import_module(module_name)
            from hyper_parallel.components.functional import swiglu  # pylint: disable=import-outside-toplevel
        self.assertIs(swiglu, implementation.swiglu)

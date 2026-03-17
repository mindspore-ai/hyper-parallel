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
"""Unit tests for fully_shard hsdp_utils (no NPU required)."""
import os
import unittest

# Force torch platform before any hyper_parallel imports
os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

# pylint: disable=C0413
import torch
from torch import nn

from hyper_parallel.core.fully_shard.hsdp_utils import (
    ParamModuleInfo,
    _get_param_module_infos,
    _named_parameters_with_duplicates,
)


class SimpleLinear(nn.Module):
    """Simple linear module for testing."""

    def __init__(self, in_features=4, out_features=4):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return x @ self.weight + self.bias


class SharedParamModule(nn.Module):
    """Module with shared parameter (tied weights)."""

    def __init__(self, dim=4):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dim))
        self.linear = nn.Linear(dim, dim)
        self.linear.weight = self.weight  # shared

    def forward(self, x):
        return self.linear(x)


class TestParamModuleInfo(unittest.TestCase):
    """Unit tests for ParamModuleInfo."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
        self.device = torch.device("cpu")

    def test_basic_param_module_info(self):
        """ParamModuleInfo stores module and param_name correctly."""
        # Arrange
        mod = SimpleLinear(4, 4)
        # Act
        info = ParamModuleInfo(module=mod, param_name="weight")
        # Assert
        self.assertIs(info.module, mod)
        self.assertEqual(info.param_name, "weight")
        self.assertEqual(info.shared_modules, [])
        self.assertEqual(info.shared_param_names, [])

    def test_param_module_info_with_shared(self):
        """ParamModuleInfo tracks shared modules."""
        # Arrange
        mod1 = SimpleLinear(4, 4)
        mod2 = SimpleLinear(4, 4)
        # Act
        info = ParamModuleInfo(
            module=mod1,
            param_name="weight",
            shared_modules=[mod2],
            shared_param_names=["weight"],
        )
        # Assert
        self.assertEqual(len(info.shared_modules), 1)
        self.assertIs(info.shared_modules[0], mod2)
        self.assertEqual(info.shared_param_names, ["weight"])


class TestNamedParametersWithDuplicates(unittest.TestCase):
    """Unit tests for _named_parameters_with_duplicates."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
        self.device = torch.device("cpu")

    def test_returns_named_params(self):
        """_named_parameters_with_duplicates returns (name, param) tuples."""
        # Arrange
        mod = SimpleLinear(4, 4)
        # Act
        result = _named_parameters_with_duplicates(mod, recurse=False)
        # Assert
        self.assertIsInstance(result, list)
        names = [name for name, _ in result]
        self.assertIn("weight", names)
        self.assertIn("bias", names)

    def test_remove_duplicate_raises(self):
        """_named_parameters_with_duplicates raises if remove_duplicate in kwargs."""
        # Arrange
        mod = SimpleLinear(4, 4)
        # Act & Assert
        with self.assertRaises(AssertionError) as ctx:
            _named_parameters_with_duplicates(mod, remove_duplicate=True)
        self.assertIn("remove_duplicate", str(ctx.exception))


class TestGetParamModuleInfos(unittest.TestCase):
    """Unit tests for _get_param_module_infos."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
        self.device = torch.device("cpu")

    def test_single_module_single_param(self):
        """_get_param_module_infos returns correct info for single param."""
        # Arrange
        mod = SimpleLinear(4, 4)
        params = list(mod.parameters())
        # Act
        infos = _get_param_module_infos(params, (mod,))
        # Assert
        self.assertEqual(len(infos), len(params))
        for info in infos:
            self.assertIsInstance(info, ParamModuleInfo)
            self.assertIs(info.module, mod)

    def test_param_not_in_module_tree_raises(self):
        """_get_param_module_infos raises when param not in module tree."""
        # Arrange
        mod = SimpleLinear(4, 4)
        orphan_param = nn.Parameter(torch.ones(2, 2))
        # Act & Assert
        with self.assertRaises(AssertionError) as ctx:
            _get_param_module_infos([orphan_param], (mod,))
        self.assertIn("not in the module tree", str(ctx.exception))

    def test_multiple_modules(self):
        """_get_param_module_infos works with multiple modules."""
        # Arrange
        mod1 = SimpleLinear(4, 4)
        mod2 = SimpleLinear(4, 4)
        params = list(mod1.parameters()) + list(mod2.parameters())
        # Act
        infos = _get_param_module_infos(params, (mod1, mod2))
        # Assert
        self.assertEqual(len(infos), len(params))


if __name__ == "__main__":
    unittest.main()

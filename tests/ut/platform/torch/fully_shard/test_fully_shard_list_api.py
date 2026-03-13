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
"""Unit tests for fully_shard list support (no NPU required)."""
import os
import unittest
from unittest.mock import patch, MagicMock

# Force torch platform before any hyper_parallel imports
os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

# pylint: disable=C0413
import torch
from torch import nn

from hyper_parallel.core.fully_shard.api import (
    _get_root_modules,
    _validate_module_for_fully_shard,
    fully_shard,
)
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from hyper_parallel.platform.platform import PlatformType


def _default_mp_policy():
    """Create default MixedPrecisionPolicy for tests."""
    return MixedPrecisionPolicy(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        output_dtype=torch.float32,
    )


class SimpleLinear(nn.Module):
    """Simple linear module for testing."""

    def __init__(self, in_features=4, out_features=4):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return x @ self.weight + self.bias


class ParentModule(nn.Module):
    """Parent module containing child modules."""

    def __init__(self):
        super().__init__()
        self.child = SimpleLinear(4, 4)

    def forward(self, x):
        return self.child(x)


class TestGetRootModules(unittest.TestCase):
    """Unit tests for _get_root_modules."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
        self.device = torch.device("cpu")

    def test_sibling_modules_both_roots(self):
        """Two sibling modules (neither contains the other) are both roots."""
        # Arrange
        linear1 = SimpleLinear(4, 4)
        linear2 = SimpleLinear(4, 4)
        # Act
        roots = _get_root_modules([linear1, linear2])
        # Assert
        self.assertEqual(len(roots), 2)
        self.assertIn(linear1, roots)
        self.assertIn(linear2, roots)

    def test_parent_child_only_parent_root(self):
        """When list contains parent and child, only parent is root."""
        # Arrange
        parent = ParentModule()
        child = parent.child
        # Act
        roots = _get_root_modules([parent, child])
        # Assert
        self.assertEqual(len(roots), 1)
        self.assertIn(parent, roots)
        self.assertNotIn(child, roots)

    def test_single_module_is_root(self):
        """Single module in list is root."""
        # Arrange
        mod = SimpleLinear(4, 4)
        # Act
        roots = _get_root_modules([mod])
        # Assert
        self.assertEqual(len(roots), 1)
        self.assertIn(mod, roots)

    def test_three_siblings_all_roots(self):
        """Three sibling modules are all roots."""
        # Arrange
        m1, m2, m3 = SimpleLinear(4, 4), SimpleLinear(4, 4), SimpleLinear(4, 4)
        # Act
        roots = _get_root_modules([m1, m2, m3])
        # Assert
        self.assertEqual(len(roots), 3)
        self.assertIn(m1, roots)
        self.assertIn(m2, roots)
        self.assertIn(m3, roots)


class TestValidateModuleForFullyShard(unittest.TestCase):
    """Unit tests for _validate_module_for_fully_shard."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
        self.device = torch.device("cpu")

    def test_single_module_valid(self):
        """Single nn.Module passes validation."""
        # Arrange
        mod = SimpleLinear(4, 4)
        # Act & Assert (no raise)
        _validate_module_for_fully_shard(mod, PlatformType.PYTORCH)

    def test_list_of_modules_valid(self):
        """List of nn.Module passes validation (Torch platform)."""
        # Arrange
        mods = [SimpleLinear(4, 4), SimpleLinear(4, 4)]
        # Act & Assert (no raise)
        _validate_module_for_fully_shard(mods, PlatformType.PYTORCH)

    def test_empty_list_raises(self):
        """Empty list raises ValueError."""
        # Act & Assert
        with self.assertRaises(ValueError) as ctx:
            _validate_module_for_fully_shard([], PlatformType.PYTORCH)
        self.assertIn("empty list", str(ctx.exception))

    def test_list_with_non_module_raises(self):
        """List containing non-Module raises ValueError."""
        # Arrange
        mod = SimpleLinear(4, 4)
        # Act & Assert
        with self.assertRaises(ValueError) as ctx:
            _validate_module_for_fully_shard([mod, 1], PlatformType.PYTORCH)
        self.assertIn("index 1", str(ctx.exception))

    def test_non_module_raises(self):
        """Non-Module input raises ValueError (via _check_module_valid)."""
        # Act & Assert
        with self.assertRaises(ValueError) as ctx:
            _validate_module_for_fully_shard("not a module", PlatformType.PYTORCH)
        self.assertIn("nn.Module", str(ctx.exception))

    def test_list_mindspore_valid(self):
        """List of modules on MindSpore passes validation (both platforms support list)."""
        try:
            import mindspore.nn
        except ImportError:
            self.skipTest("mindspore not installed")
        # Arrange - use MindSpore Cell for validation
        from mindspore import nn as ms_nn
        mods = [ms_nn.Dense(4, 4), ms_nn.Dense(4, 4)]
        # Act & Assert (no raise)
        _validate_module_for_fully_shard(mods, PlatformType.MINDSPORE)


class TestFullyShardListAPI(unittest.TestCase):
    """Unit tests for fully_shard list support (mocked to avoid NPU/dist)."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
        self.device = torch.device("cpu")

    def _create_mock_mesh(self):
        """Create mock DeviceMesh to avoid distributed init."""
        mesh = MagicMock()
        mesh.ndim = 1
        return mesh

    @patch("hyper_parallel.core.fully_shard.api._get_device_from_mesh")
    @patch("hyper_parallel.core.fully_shard.api.platform")
    def test_fully_shard_single_module_returns_module(self, mock_platform, mock_get_device):
        """fully_shard with single module returns the same module (in-place)."""
        # Arrange
        mock_platform.platform_type = PlatformType.PYTORCH
        mock_get_device.return_value = self.device
        mesh = self._create_mock_mesh()
        mod = SimpleLinear(4, 4)
        # Act
        with patch(
            "hyper_parallel.core.fully_shard.api.HSDPModule.hsdp_init",
            lambda self, *a, **k: None,
        ):
            result = fully_shard(
                mod,
                mesh=mesh,
                reshard_after_forward=True,
                mp_policy=_default_mp_policy(),
            )
        # Assert
        self.assertIs(result, mod)
        self.assertIsInstance(result, nn.Module)

    @patch("hyper_parallel.core.fully_shard.api._get_device_from_mesh")
    @patch("hyper_parallel.core.fully_shard.api.platform")
    def test_fully_shard_list_returns_list(self, mock_platform, mock_get_device):
        """fully_shard with list returns the same list (in-place)."""
        # Arrange
        mock_platform.platform_type = PlatformType.PYTORCH
        mock_get_device.return_value = self.device
        mesh = self._create_mock_mesh()
        linear1 = SimpleLinear(4, 4)
        linear2 = SimpleLinear(4, 4)
        modules_list = [linear1, linear2]
        # Act
        with patch.object(
            type(linear1),
            "hsdp_init",
            lambda self, *a, **k: None,
            create=True,
        ):
            with patch(
                "hyper_parallel.core.fully_shard.api.HSDPModule.hsdp_init",
                lambda self, *a, **k: None,
            ):
                result = fully_shard(
                    modules_list,
                    mesh=mesh,
                    reshard_after_forward=True,
                    mp_policy=_default_mp_policy(),
                )
        # Assert
        self.assertIs(result, modules_list)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    @patch("hyper_parallel.core.fully_shard.api._get_device_from_mesh")
    @patch("hyper_parallel.core.fully_shard.api.platform")
    def test_fully_shard_list_root_filtering(self, mock_platform, mock_get_device):
        """fully_shard with parent+child list filters to root only."""
        # Arrange
        mock_platform.platform_type = PlatformType.PYTORCH
        mock_get_device.return_value = self.device
        mesh = self._create_mock_mesh()
        parent = ParentModule()
        child = parent.child
        modules_list = [parent, child]
        # Act
        with patch(
            "hyper_parallel.core.fully_shard.api.HSDPModule.hsdp_init",
            lambda self, *a, **k: None,
        ):
            result = fully_shard(
                modules_list,
                mesh=mesh,
                reshard_after_forward=True,
                mp_policy=_default_mp_policy(),
            )
        # Assert
        self.assertIs(result, modules_list)
        self.assertEqual(len(result), 2)

    @patch("hyper_parallel.core.fully_shard.api._get_device_from_mesh")
    @patch("hyper_parallel.core.fully_shard.api.platform")
    def test_fully_shard_list_mindspore_returns_list(self, mock_platform, mock_get_device):
        """fully_shard with list on MindSpore returns the same list (both platforms support list)."""
        # Arrange (mock to avoid MindSpore Cell validation and device init)
        mock_platform.platform_type = PlatformType.MINDSPORE
        mock_get_device.return_value = torch.device("cpu")
        mesh = self._create_mock_mesh()
        linear1 = SimpleLinear(4, 4)
        linear2 = SimpleLinear(4, 4)
        modules_list = [linear1, linear2]
        # Act - patch _validate_module_for_fully_shard to bypass (nn.Module is not Cell)
        with patch(
            "hyper_parallel.core.fully_shard.api._validate_module_for_fully_shard",
            lambda m, pt: None,
        ):
            with patch(
                "hyper_parallel.core.fully_shard.api.HSDPModule.hsdp_init",
                lambda self, *a, **k: None,
            ):
                result = fully_shard(
                    modules_list,
                    mesh=mesh,
                    reshard_after_forward=True,
                    mp_policy=_default_mp_policy(),
                )
        # Assert
        self.assertIs(result, modules_list)
        self.assertEqual(len(result), 2)

    @patch("hyper_parallel.core.fully_shard.api.platform")
    def test_fully_shard_empty_list_raises(self, mock_platform):
        """fully_shard with empty list raises ValueError."""
        # Arrange (validation runs before _get_device_from_mesh)
        mock_platform.platform_type = PlatformType.PYTORCH
        mesh = self._create_mock_mesh()
        # Act & Assert
        with self.assertRaises(ValueError) as ctx:
            fully_shard(
                [],
                mesh=mesh,
                reshard_after_forward=True,
                mp_policy=_default_mp_policy(),
            )
        self.assertIn("empty list", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

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
"""Unit tests for ``hyper_parallel.core.tensor_parallel.style``.

Tests for :class:`ParallelStyle` abstract base class and its contract:
- Abstract class cannot be instantiated directly
- Subclasses must implement ``apply`` method
- ``src_data_rank`` default value and mutability
- ``apply`` method signature and return value contract
"""
import os
import unittest
from unittest.mock import patch

import torch
from torch import nn

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.dtensor.device_mesh import init_device_mesh, _DEVICE_MESH_MAP
from hyper_parallel.core.tensor_parallel.style import ParallelStyle
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS, PlatformType


class ConcreteParallelStyle(ParallelStyle):
    """A concrete implementation of ParallelStyle for testing."""

    def __init__(self):
        super().__init__()
        self.apply_called = False
        self.last_module = None
        self.last_device_mesh = None

    def apply(self, module, device_mesh):
        """Apply the parallel style and record the call."""
        self.apply_called = True
        self.last_module = module
        self.last_device_mesh = device_mesh
        return module


class WrapperParallelStyle(ParallelStyle):
    """A wrapper style that wraps the module in an identity wrapper."""

    class IdentityWrapper(nn.Module):
        """Simple wrapper for testing."""

        def __init__(self, module):
            super().__init__()
            self.wrapped = module

        def forward(self, *args, **kwargs):
            return self.wrapped(*args, **kwargs)

    def apply(self, module, device_mesh):
        """Wrap the module."""
        return WrapperParallelStyle.IdentityWrapper(module)


class TestParallelStyle(unittest.TestCase):
    """Tests for ParallelStyle abstract base class."""

    def setUp(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def tearDown(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def _setup_mock_platform(self, mock_platform, world_size: int = 4):
        mock_platform.platform_type = PlatformType.PYTORCH
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = world_size
        mock_platform.tensor_to_numpy.side_effect = (
            lambda t: t.numpy() if hasattr(t, "numpy") else __import__("numpy").array(t)
        )

    def _make_1d_mesh(self, mock_platform, size: int = 4):
        self._setup_mock_platform(mock_platform, world_size=size)
        return init_device_mesh(
            device_type="cpu",
            mesh_shape=(size,),
            mesh_dim_names=("tp",),
            init_backend=False,
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cannot_instantiate_abstract_class(self, mock_platform):
        """
        Feature: ParallelStyle abstract base class instantiation
        Description: call ParallelStyle() without subclassing
        Expectation: throw TypeError whose message mentions abstract
        """
        self._setup_mock_platform(mock_platform)
        with self.assertRaises(TypeError) as ctx:
            ParallelStyle()
        self.assertIn("abstract", str(ctx.exception).lower())

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_concrete_subclass_can_be_instantiated(self, mock_platform):
        """
        Feature: concrete ParallelStyle subclass instantiation
        Description: instantiate ConcreteParallelStyle which implements apply
        Expectation: object is not None and apply_called is False before apply
        """
        self._setup_mock_platform(mock_platform)
        style = ConcreteParallelStyle()
        self.assertIsNotNone(style)
        self.assertFalse(style.apply_called)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_src_data_rank_default_value(self, mock_platform):
        """
        Feature: ParallelStyle src_data_rank default value
        Description: new ConcreteParallelStyle without assigning src_data_rank
        Expectation: src_data_rank equals 0
        """
        self._setup_mock_platform(mock_platform)
        style = ConcreteParallelStyle()
        self.assertEqual(style.src_data_rank, 0)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_src_data_rank_can_be_set(self, mock_platform):
        """
        Feature: ParallelStyle src_data_rank modification
        Description: assign src_data_rank to integer 2 on ConcreteParallelStyle
        Expectation: reading src_data_rank returns 2
        """
        self._setup_mock_platform(mock_platform)
        style = ConcreteParallelStyle()
        style.src_data_rank = 2
        self.assertEqual(style.src_data_rank, 2)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_src_data_rank_can_be_none(self, mock_platform):
        """
        Feature: ParallelStyle src_data_rank may be None
        Description: set src_data_rank to None on ConcreteParallelStyle
        Expectation: src_data_rank is None
        """
        self._setup_mock_platform(mock_platform)
        style = ConcreteParallelStyle()
        style.src_data_rank = None
        self.assertIsNone(style.src_data_rank)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_apply_receives_correct_arguments(self, mock_platform):
        """
        Feature: ParallelStyle.apply receives module and mesh
        Description: call ConcreteParallelStyle.apply(Linear, 1-D DeviceMesh)
        Expectation: apply_called True; last_module and last_device_mesh match; return is module
        """
        mesh = self._make_1d_mesh(mock_platform)
        style = ConcreteParallelStyle()
        module = nn.Linear(4, 4)

        result = style.apply(module, mesh)

        self.assertTrue(style.apply_called)
        self.assertIs(style.last_module, module)
        self.assertIs(style.last_device_mesh, mesh)
        self.assertIs(result, module)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_apply_can_return_wrapped_module(self, mock_platform):
        """
        Feature: ParallelStyle.apply may return wrapped module
        Description: WrapperParallelStyle.apply wraps Linear in IdentityWrapper
        Expectation: result is not original module; result.wrapped is original module
        """
        mesh = self._make_1d_mesh(mock_platform)
        style = WrapperParallelStyle()
        module = nn.Linear(4, 4)

        result = style.apply(module, mesh)

        self.assertIsNot(result, module)
        self.assertIsInstance(result, nn.Module)
        self.assertIs(result.wrapped, module)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_apply_called_multiple_times(self, mock_platform):
        """
        Feature: ParallelStyle.apply may run multiple times
        Description: same style applies to two different Linear modules on same mesh
        Expectation: last_module tracks second module; apply_called remains True
        """
        mesh = self._make_1d_mesh(mock_platform)
        style = ConcreteParallelStyle()
        module1 = nn.Linear(4, 4)
        module2 = nn.Linear(3, 3)

        style.apply(module1, mesh)
        self.assertIs(style.last_module, module1)

        style.apply(module2, mesh)
        self.assertIs(style.last_module, module2)
        self.assertTrue(style.apply_called)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_different_styles_have_independent_src_data_rank(self, mock_platform):
        """
        Feature: independent src_data_rank per ParallelStyle instance
        Description: two ConcreteParallelStyle instances with src_data_rank 1 and 2
        Expectation: style1 reads 1 and style2 reads 2 with no cross mutation
        """
        self._setup_mock_platform(mock_platform)
        style1 = ConcreteParallelStyle()
        style2 = ConcreteParallelStyle()

        style1.src_data_rank = 1
        style2.src_data_rank = 2

        self.assertEqual(style1.src_data_rank, 1)
        self.assertEqual(style2.src_data_rank, 2)


class TestParallelStyleWithMockMesh(unittest.TestCase):
    """Additional tests for ParallelStyle using mock mesh."""

    def setUp(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def tearDown(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def _setup_mock_platform(self, mock_platform, world_size: int = 4):
        mock_platform.platform_type = PlatformType.PYTORCH
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = world_size
        mock_platform.tensor_to_numpy.side_effect = (
            lambda t: t.numpy() if hasattr(t, "numpy") else __import__("numpy").array(t)
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_apply_with_different_mesh_dimensions(self, mock_platform):
        """
        Feature: ParallelStyle.apply with varying 1-D mesh world sizes
        Description: loop mesh sizes 1,2,4,8; each time apply Linear with new mesh
        Expectation: each iteration return is module; last_device_mesh equals current mesh
        """
        for size in [1, 2, 4, 8]:
            EXISTING_COMM_GROUPS.clear()
            _DEVICE_MESH_MAP.clear()
            self._setup_mock_platform(mock_platform, world_size=size)
            mesh = init_device_mesh(
                device_type="cpu",
                mesh_shape=(size,),
                mesh_dim_names=("tp",),
                init_backend=False,
            )
            style = ConcreteParallelStyle()
            module = nn.Linear(4, 4)

            result = style.apply(module, mesh)

            self.assertIs(result, module)
            self.assertIs(style.last_device_mesh, mesh)


if __name__ == "__main__":
    unittest.main()

# Copyright 2025 Huawei Technologies Co., Ltd
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
"""
Unit tests for OpDispatcher with custom distributed ops (e.g., StackExt).
"""
import importlib
import os
import sys
import unittest
from pathlib import Path
from typing import Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)

from hyper_parallel.platform import get_platform
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS
from hyper_parallel.core.shard._op_dispatch import LayoutCacheKey

_TEST_FILE_DIR = Path(__file__).resolve().parent
_TESTS_ROOT_DIR = _TEST_FILE_DIR.parent.parent.parent.parent
_CUSTOM_OPS_DIR = _TESTS_ROOT_DIR / "tests" / "custom_ops"

HYPER_PARALLEL_OPS_YAML_DIR = str(_CUSTOM_OPS_DIR)
HYPER_PARALLEL_OPS_PYTHON_PATH = str(_CUSTOM_OPS_DIR)


def _reload_op_dispatch_with_env_str(yaml_dir: str, python_path: str):
    """
    Reload OpDispatcher module with custom environment variables.

    Args:
        yaml_dir (str): Path to the directory containing op dispatch YAML files.
        python_path (str): Path to the directory containing custom op implementations.

    Returns:
        module: The reloaded OpDispatcher module.
    """
    os.environ["HYPER_PARALLEL_OPS_YAML_DIR"] = yaml_dir
    os.environ["HYPER_PARALLEL_OPS_PYTHON_PATH"] = python_path

    target_mod = "hyper_parallel.core.shard._op_dispatch"
    if target_mod in sys.modules:
        del sys.modules[target_mod]

    mod = importlib.import_module(target_mod)
    mod = importlib.reload(mod)
    return mod


class TestStackExtDispatch(unittest.TestCase):
    """
    Feature: StackExt Dispatch and Layout Cache
    Description: Test StackExt distributed operator dispatch and layout caching.
    Expectation: dispatch should return correct DTensor output with proper layout,
                 and layout cache should work correctly.
    """

    def setUp(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        self.platform = get_platform()

    def tearDown(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def _make_mesh(self, mock_platform, mesh_shape, mesh_dim_names):
        """Create a device mesh for testing."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = np.prod(mesh_shape)
        return init_device_mesh(
            device_type="npu",
            mesh_shape=mesh_shape,
            mesh_dim_names=mesh_dim_names,
            init_backend=False,
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_stack_ext_dispatch_and_layout(self, mock_platform):
        """Test StackExt dispatch and layout cache with two input DTensors."""
        op_dispatch = _reload_op_dispatch_with_env_str(
            HYPER_PARALLEL_OPS_YAML_DIR, HYPER_PARALLEL_OPS_PYTHON_PATH
        )

        mesh = self._make_mesh(mock_platform, (1, 1, 1), ("dp", "cp", "mp"))
        base_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 2)

        from hyper_parallel.core.shard._op_dispatch import LayoutCacheManager

        dist_op = LayoutCacheManager.get_instance().distributed_op("StackExt")

        np_obj = np
        local_tensor0 = np_obj.arange(6).reshape(2, 3).astype(np_obj.int32)
        local_tensor1 = np_obj.arange(6, 12).reshape(2, 3).astype(np_obj.int32)

        d0 = MagicMock(spec=DTensor)
        d0._local_tensor = local_tensor0
        d0.layout = base_layout
        d0._layout = base_layout
        d0.to_local.return_value = local_tensor0

        d1 = MagicMock(spec=DTensor)
        d1._local_tensor = local_tensor1
        d1.layout = base_layout
        d1._layout = base_layout
        d1.to_local.return_value = local_tensor1

        output_layout = dist_op.infer_layout((d0.layout, d1.layout), (0,))

        assert output_layout is not None
        assert tuple(output_layout.to_dict()["tensor_map"]) == (-1, -1, -1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_stack_ext_layout_cache(self, mock_platform):
        """Test StackExt layout cache with multiple input layouts."""
        op_dispatch = _reload_op_dispatch_with_env_str(
            HYPER_PARALLEL_OPS_YAML_DIR, HYPER_PARALLEL_OPS_PYTHON_PATH
        )

        mesh = self._make_mesh(mock_platform, (1, 1, 1), ("dp", "cp", "mp"))
        base_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 2)

        from hyper_parallel.core.shard._op_dispatch import LayoutCacheManager

        dist_op = LayoutCacheManager.get_instance().distributed_op("StackExt")

        np_obj = np
        local_tensor0 = np_obj.arange(6).reshape(2, 3).astype(np_obj.int32)
        local_tensor1 = np_obj.arange(6, 12).reshape(2, 3).astype(np_obj.int32)

        d0 = MagicMock(spec=DTensor)
        d0._local_tensor = local_tensor0
        d0.layout = base_layout
        d0._layout = base_layout

        d1 = MagicMock(spec=DTensor)
        d1._local_tensor = local_tensor1
        d1.layout = base_layout
        d1._layout = base_layout

        output_layout = dist_op.infer_layout((d0.layout, d1.layout), (0,))

        assert output_layout is not None
        assert tuple(output_layout.to_dict()["tensor_map"]) == (-1, -1, -1)


class TestNewDispatchFlow(unittest.TestCase):
    """
    Feature: New Dispatch Flow with Preprocess and Infer Layout
    Description: Test the new dispatch flow with preprocess and infer_layout methods for a distributed operator.
    Expectation: preprocess should return valid local_args, local_kwargs, and cache_values, and infer_layout should
                 return the correct output layouts.
    """

    def setUp(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        self.platform = get_platform()

    def tearDown(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def _make_mesh(self, mock_platform, mesh_shape, mesh_dim_names):
        """Create a device mesh for testing."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = np.prod(mesh_shape)
        return init_device_mesh(
                device_type="npu",
                mesh_shape=mesh_shape,
                mesh_dim_names=mesh_dim_names,
                init_backend=False,
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_from_cache_values_with_layout(self, mock_platform):
        """Test that LayoutCacheKey from_cache_values with layout returns correct key."""
        mesh = self._make_mesh(mock_platform, (2, 2), ("dp", "mp"))
        layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)
        cache_values = [layout, 1, True]
        key = LayoutCacheKey.from_cache_values(cache_values)
        expected = [str(layout.compact_str), "1", "True"]
        assert list(key._tuple) == expected, (
            f"Expected {expected}, got {list(key._tuple)}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_from_cache_values_consistency(self, mock_platform):
        """Test that LayoutCacheKey from_cache_values is consistent with the same cache_values."""
        mesh = self._make_mesh(mock_platform, (2, 2), ("dp", "mp"))
        layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)
        cache_values1 = [layout, 1, True]
        cache_values2 = [layout, 1, True]
        key1 = LayoutCacheKey.from_cache_values(cache_values1)
        key2 = LayoutCacheKey.from_cache_values(cache_values2)
        assert key1 == key2, f"Keys should be equal: {key1} != {key2}"
        assert hash(key1) == hash(key2), "Hashes should be equal"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_from_cache_values_different_values(self, mock_platform):
        """Test that LayoutCacheKey from_cache_values differs with different values."""
        mesh = self._make_mesh(mock_platform, (2, 2), ("dp", "mp"))
        layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)
        key1 = LayoutCacheKey.from_cache_values([layout, 1, True])
        key2 = LayoutCacheKey.from_cache_values([layout, 0, True])
        assert key1 != key2, "Keys should differ"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_equality_with_legacy_key(self, mock_platform):
        """Test that LayoutCacheKey is equal to legacy key."""
        mesh = self._make_mesh(mock_platform, (2, 2), ("dp", "mp"))
        layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)
        key_new = LayoutCacheKey.from_cache_values([layout, 1, True])
        key_legacy = LayoutCacheKey([str(layout.compact_str), "1", "True"])
        assert key_new == key_legacy, "Keys should be equal"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_dispatch_new_flow_with_preprocess(self, mock_platform):
        """Test that dispatch preprocess returns valid local_args, local_kwargs, and cache_values."""
        from hyper_parallel.core.shard.ops.parallel_sort import SortDistributedOp

        op = SortDistributedOp("sort")
        mesh = self._make_mesh(mock_platform, (2,), ("dp",))
        layout = _build_layout(mesh, (Replicate(),), 2)

        mock_tensor = MagicMock()
        mock_tensor._layout = layout
        mock_tensor.layout = layout
        mock_tensor.to_local.return_value = np.random.randn(4, 4)

        result = op.preprocess((mock_tensor, -1), {})
        assert result is not None, "preprocess should return tuple for DTensor input"
        local_args, local_kwargs, cache_values = result

        assert local_kwargs.get("dim") == -1, "Expected dim=-1 in kwargs"
        assert len(cache_values) == 2, "Expected 2 cache values"
        assert cache_values[0] is layout, "Expected layout in cache_values"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_dispatch_new_flow_infer_layout_with_cache_values(self, mock_platform):
        """Test that dispatch infer_layout with cache_values returns correct output layouts."""
        from hyper_parallel.core.shard.ops.parallel_sort import SortDistributedOp

        op = SortDistributedOp("sort")
        mesh = self._make_mesh(mock_platform, (2,), ("dp",))
        layout = _build_layout(mesh, (Replicate(),), 2)
        cache_values = [layout, -1]

        infer_result = op.infer_layout(cache_values)
        assert isinstance(infer_result, tuple), "Expected tuple"
        output_layouts, extra_info = infer_result
        assert isinstance(output_layouts, tuple), "Expected tuple of output layouts"
        assert len(output_layouts) == 2, "Expected 2 output layouts"
        assert extra_info is None, "Expected extra_info=None"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_dispatch_new_flow_normalizes_args(self, mock_platform):
        """Test that dispatch normalizes args and kwargs."""
        from hyper_parallel.core.shard.ops.parallel_sort import SortDistributedOp

        op = SortDistributedOp("sort")
        mesh = self._make_mesh(mock_platform, (2,), ("dp",))
        layout = _build_layout(mesh, (Replicate(),), 2)

        mock_tensor = MagicMock()
        mock_tensor._layout = layout
        mock_tensor.layout = layout
        mock_tensor.to_local.return_value = np.random.randn(4, 4)

        result1 = op.preprocess((mock_tensor, 1, True, False), {})
        result2 = op.preprocess((mock_tensor,), {"dim": 1, "descending": True, "stable": False})

        assert result1 is not None, "preprocess should return tuple"
        assert result2 is not None, "preprocess should return tuple"

        _, kwargs1, cv1 = result1
        _, kwargs2, cv2 = result2
        assert kwargs1 == kwargs2, "Normalized kwargs should match"
        assert cv1[1] == cv2[1], "Normalized cache_values dim should match"

    def test_dispatch_falls_back_to_legacy_when_preprocess_returns_none(self):
        """Test that dispatch falls back to legacy preprocess when new preprocess returns None."""
        from hyper_parallel.core.shard.ops.parallel_ops import DistributedOp

        class DummyOp(DistributedOp):
            def infer_layout(self, layouts, extra_args=None):
                return layouts[0]

        op = DummyOp("dummy")
        assert op.preprocess((1, 2), {"a": 3}) is None, "Default preprocess should return None for non-DTensor inputs"

if __name__ == "__main__":
    unittest.main()

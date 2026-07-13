# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
from unittest.mock import MagicMock, patch

import numpy as np
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)

from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS
from hyper_parallel.core.shard._op_dispatch import LayoutCacheKey

_TEST_FILE_DIR = Path(__file__).resolve().parent
_TESTS_ROOT_DIR = _TEST_FILE_DIR.parent.parent.parent.parent
_CUSTOM_OPS_DIR = _TESTS_ROOT_DIR / "tests" / "ut" / "core" / "shard" / "custom_parallel_ops"

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

    def tearDown(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def _make_mesh(self, mock_platform, mesh_shape, mesh_dim_names):
        """Create a device mesh for testing."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = np.prod(mesh_shape)
        mock_platform.tensor_to_numpy.side_effect = (
            lambda t: t.numpy() if hasattr(t, "numpy") else np.array(t)
        )
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

        output_layouts, _ = dist_op.infer_layout([(d0.layout, d1.layout), 0])
        output_layout = output_layouts[0]

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

        output_layouts, _ = dist_op.infer_layout([(d0.layout, d1.layout), 0])
        output_layout = output_layouts[0]

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

    def tearDown(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def _make_mesh(self, mock_platform, mesh_shape, mesh_dim_names):
        """Create a device mesh for testing."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = np.prod(mesh_shape)
        mock_platform.tensor_to_numpy.side_effect = (
            lambda t: t.numpy() if hasattr(t, "numpy") else np.array(t)
        )
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

class TestUnwrapArgsAndKwargs(unittest.TestCase):
    """
    Feature: OpDispatcher._unwrap_value / _unwrap_args / _unwrap_kwargs
    Description: Test that DTensor wrappers are correctly stripped from args
                 and kwargs while preserving tuple/list container structure.
    Expectation: DTensor instances are replaced by their local tensors;
                 plain tensors, scalars, and nested containers are preserved.
    """

    def setUp(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def tearDown(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def test_unwrap_value_plain_scalar(self):
        """Test _unwrap_value passes plain scalars through unchanged."""
        from hyper_parallel.core.shard._op_dispatch import OpDispatcher

        assert OpDispatcher._unwrap_value(42) == 42, \
            (f"Expected 42, "
             f"got {OpDispatcher._unwrap_value(42)}")
        assert OpDispatcher._unwrap_value("hello") == "hello", \
            (f"Expected 'hello', "
             f"got {OpDispatcher._unwrap_value('hello')}")
        assert OpDispatcher._unwrap_value(None) is None, \
            "Expected None"

    def test_unwrap_value_plain_tensor(self):
        """Test _unwrap_value passes plain (non-DTensor) tensors through unchanged."""
        from hyper_parallel.core.shard._op_dispatch import OpDispatcher

        t = np.array([1.0, 2.0])
        result = OpDispatcher._unwrap_value(t)
        assert result is t, \
            (f"Expected same object, "
             f"got {result}")

    def test_unwrap_value_dtensor(self):
        """Test _unwrap_value replaces DTensor with its local tensor."""
        from hyper_parallel.core.shard._op_dispatch import OpDispatcher

        local = np.array([1.0, 2.0])
        dt = MagicMock(spec=DTensor)
        dt.to_local.return_value = local

        result = OpDispatcher._unwrap_value(dt)
        dt.to_local.assert_called_once()
        assert result is local, \
            (f"Expected local tensor, "
             f"got {result}")

    def test_unwrap_value_tuple_with_dtensor(self):
        """Test _unwrap_value recursively unwraps DTensors inside tuples."""
        from hyper_parallel.core.shard._op_dispatch import OpDispatcher

        local0 = np.array([1.0])
        local1 = np.array([2.0])
        dt0 = MagicMock(spec=DTensor)
        dt0.to_local.return_value = local0
        dt1 = MagicMock(spec=DTensor)
        dt1.to_local.return_value = local1

        result = OpDispatcher._unwrap_value((dt0, 3, dt1))
        assert isinstance(result, tuple), \
            (f"Expected tuple, "
             f"got {type(result)}")
        assert len(result) == 3, \
            (f"Expected length 3, "
             f"got {len(result)}")
        assert result[0] is local0, \
            "First element should be local0"
        assert result[1] == 3, \
            "Second element should be scalar 3"
        assert result[2] is local1, \
            "Third element should be local1"

    def test_unwrap_value_list_with_dtensor(self):
        """Test _unwrap_value recursively unwraps DTensors inside lists."""
        from hyper_parallel.core.shard._op_dispatch import OpDispatcher

        local = np.array([3.0])
        dt = MagicMock(spec=DTensor)
        dt.to_local.return_value = local

        result = OpDispatcher._unwrap_value([dt, "abc"])
        assert isinstance(result, list), \
            (f"Expected list, "
             f"got {type(result)}")
        assert len(result) == 2, \
            (f"Expected length 2, "
             f"got {len(result)}")
        assert result[0] is local, \
            "First element should be local"
        assert result[1] == "abc", \
            "Second element should be 'abc'"

    def test_unwrap_args_mixed(self):
        """Test _unwrap_args with mixed DTensor, scalar, and tuple args."""
        from hyper_parallel.core.shard._op_dispatch import OpDispatcher

        local = np.array([1.0])
        dt = MagicMock(spec=DTensor)
        dt.to_local.return_value = local

        result = OpDispatcher._unwrap_args((dt, 5, (dt, 2)))
        assert len(result) == 3, \
            (f"Expected 3 args, "
             f"got {len(result)}")
        assert result[0] is local, \
            "First arg should be local tensor"
        assert result[1] == 5, \
            "Second arg should be scalar 5"
        assert isinstance(result[2], tuple), \
            (f"Third arg should be tuple, "
             f"got {type(result[2])}")
        assert result[2][0] is local, \
            "Nested tuple first element should be local tensor"
        assert result[2][1] == 2, \
            "Nested tuple second element should be 2"

    def test_unwrap_kwargs_mixed(self):
        """Test _unwrap_kwargs with mixed DTensor, scalar, and list kwargs."""
        from hyper_parallel.core.shard._op_dispatch import OpDispatcher

        local = np.array([1.0])
        dt = MagicMock(spec=DTensor)
        dt.to_local.return_value = local

        kwargs = {"input": dt, "dim": -1, "extra": [dt, True]}
        result = OpDispatcher._unwrap_kwargs(kwargs)
        assert result["input"] is local, \
            "kwargs 'input' should be local tensor"
        assert result["dim"] == -1, \
            (f"kwargs 'dim' should be -1, "
             f"got {result['dim']}")
        assert isinstance(result["extra"], list), \
            (f"kwargs 'extra' should be list, "
             f"got {type(result['extra'])}")
        assert result["extra"][0] is local, \
            "kwargs 'extra'[0] should be local tensor"
        assert result["extra"][1] is True, \
            "kwargs 'extra'[1] should be True"

    def test_unwrap_kwargs_does_not_mutate_original(self):
        """Test _unwrap_kwargs returns a new dict without modifying the original."""
        from hyper_parallel.core.shard._op_dispatch import OpDispatcher

        local = np.array([1.0])
        dt = MagicMock(spec=DTensor)
        dt.to_local.return_value = local

        original = {"x": dt, "y": 10}
        result = OpDispatcher._unwrap_kwargs(original)
        assert result is not original, \
            "Returned dict should be a new object"
        assert isinstance(original["x"], MagicMock), \
            "Original dict should not be mutated"

    @patch("hyper_parallel.core.shard._op_dispatch.platform")
    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_dispatch_bypass_unwraps_kwargs(self, mock_device_platform, mock_dispatch_platform):
        """Test dispatch bypass path unwraps kwargs containing DTensors."""
        from hyper_parallel.core.shard._op_dispatch import OpDispatcher

        mock_dispatch_platform.get_op_name.return_value = "fake_op"

        local = np.array([1.0, 2.0])
        dt = MagicMock(spec=DTensor)
        dt.to_local.return_value = local

        captured_args = {}
        def fake_op(*args, **kwargs):
            captured_args["args"] = args
            captured_args["kwargs"] = kwargs
            return np.array([3.0])

        dispatcher = OpDispatcher.__new__(OpDispatcher)
        dispatcher.whitelist = ["fake_op"]
        dispatcher._random_ops = set()
        dispatcher._random_ms_ops = set()
        dispatcher.layout_infer_ops = {}

        result = dispatcher.dispatch(fake_op, (dt,), {"bias": dt, "alpha": 0.5})
        assert captured_args["args"] == (local,), \
            (f"Expected args={(local,)}, "
             f"got {captured_args['args']}")
        assert captured_args["kwargs"]["bias"] is local, \
            (f"Expected kwargs['bias']=local, "
             f"got {captured_args['kwargs']['bias']}")
        assert captured_args["kwargs"]["alpha"] == 0.5, \
            (f"Expected kwargs['alpha']=0.5, "
             f"got {captured_args['kwargs']['alpha']}")


class TestOpDispatchSimpleFunctions(unittest.TestCase):
    """
    Feature: Module-level toggle functions and LayoutCacheKey/Manager methods.
    Description: Cover SkipDTensorDispatch (including nesting), get_no_skip_ops,
                 get_dtensor_dispatch, LayoutCacheKey.__eq__/__repr__, and
                 LayoutCacheManager accessors.
    Expectation: State changes are visible; nesting is safe; repr format correct; cache cleared.
    """

    def test_get_no_skip_ops_returns_frozenset(self):
        """get_no_skip_ops returns the current no-skip frozenset."""
        from hyper_parallel.core.shard._op_dispatch import get_no_skip_ops
        result = get_no_skip_ops()
        self.assertIsInstance(result, frozenset)

    def test_skip_dtensor_dispatch_context_manager(self):
        """SkipDTensorDispatch sets dispatch state via ContextVar and restores on exit."""
        from hyper_parallel.core.shard._op_dispatch import get_dtensor_dispatch, get_no_skip_ops
        from hyper_parallel.core.dtensor.dtensor import SkipDTensorDispatch
        self.assertTrue(get_dtensor_dispatch())
        with SkipDTensorDispatch(no_skip={"_TestOp_coverage"}):
            self.assertFalse(get_dtensor_dispatch())
            self.assertIn("_TestOp_coverage", get_no_skip_ops())
        self.assertTrue(get_dtensor_dispatch())
        self.assertNotIn("_TestOp_coverage", get_no_skip_ops())

    def test_skip_dtensor_dispatch_nested(self):
        """Nested SkipDTensorDispatch correctly restores state at each level."""
        from hyper_parallel.core.shard._op_dispatch import get_dtensor_dispatch, get_no_skip_ops
        from hyper_parallel.core.dtensor.dtensor import SkipDTensorDispatch
        self.assertTrue(get_dtensor_dispatch())
        with SkipDTensorDispatch(no_skip={"op_a"}):
            self.assertFalse(get_dtensor_dispatch())
            self.assertIn("op_a", get_no_skip_ops())
            with SkipDTensorDispatch(no_skip={"op_b"}):
                self.assertFalse(get_dtensor_dispatch())
                self.assertIn("op_a", get_no_skip_ops())
                self.assertIn("op_b", get_no_skip_ops())
            # inner exit should not re-enable dispatch or remove outer's no_skip ops
            self.assertFalse(get_dtensor_dispatch())
            self.assertIn("op_a", get_no_skip_ops())
            self.assertNotIn("op_b", get_no_skip_ops())
        self.assertTrue(get_dtensor_dispatch())
        self.assertEqual(get_no_skip_ops(), frozenset())

    def test_layout_cache_key_eq_and_repr(self):
        """LayoutCacheKey.__eq__ compares tuples; __repr__ includes class name."""
        k1 = LayoutCacheKey(["a", "b"])
        k2 = LayoutCacheKey(["a", "b"])
        k3 = LayoutCacheKey(["a", "c"])
        self.assertTrue(k1 == k2)
        self.assertFalse(k1 == k3)
        self.assertFalse(k1 == "not_a_key")
        self.assertIn("LayoutCacheKey", repr(k1))

    def test_layout_cache_manager_get_and_clear(self):
        """get_layout_cache returns dict; clear_cache empties it."""
        from hyper_parallel.core.shard._op_dispatch import LayoutCacheManager
        mgr = LayoutCacheManager.get_instance()
        cache = mgr.get_layout_cache()
        self.assertIsInstance(cache, dict)
        original_keys = set(cache.keys())
        cache["_test_coverage_op"] = {}
        mgr.clear_cache()
        self.assertNotIn("_test_coverage_op", mgr.get_layout_cache())
        for key in original_keys:
            cache[key] = {}


class TestOpDispatcherSetupYamlDir(unittest.TestCase):
    """
    Feature: OpDispatcher._setup_yaml_dir absolute path branch.
    Description: When env_yaml_dir is an absolute path, work_dir is set to ''.
    Expectation: yaml_dir = absolute path, work_dir = ''.
    """

    def test_absolute_path_sets_work_dir_empty(self):
        """_setup_yaml_dir with absolute path sets work_dir='' and yaml_dir=path."""
        from hyper_parallel.core.shard._op_dispatch import OpDispatcher

        class _Stub:
            yaml_dir = ""
            work_dir = ""

        stub = _Stub()
        OpDispatcher._setup_yaml_dir(stub, "/absolute/path/to/yaml")
        self.assertEqual(stub.yaml_dir, "/absolute/path/to/yaml")
        self.assertEqual(stub.work_dir, "")

    def test_relative_path_sets_work_dir(self):
        """_setup_yaml_dir with relative path sets work_dir from __file__."""
        from hyper_parallel.core.shard._op_dispatch import OpDispatcher

        class _Stub:
            yaml_dir = ""
            work_dir = ""

        stub = _Stub()
        OpDispatcher._setup_yaml_dir(stub, "relative/yaml")
        self.assertEqual(stub.yaml_dir, "relative/yaml")
        self.assertNotEqual(stub.work_dir, "")

    def test_none_path_sets_default_yaml_dir(self):
        """_setup_yaml_dir with None uses default 'shard/ops/yaml'."""
        from hyper_parallel.core.shard._op_dispatch import OpDispatcher

        class _Stub:
            yaml_dir = ""
            work_dir = ""

        stub = _Stub()
        OpDispatcher._setup_yaml_dir(stub, None)
        self.assertEqual(stub.yaml_dir, "shard/ops/yaml")


class TestDispatchLayoutInfer(unittest.TestCase):
    """
    Feature: OpDispatcher._dispatch_layout_infer routing paths.
    Description: Routes unregistered ops through _handle_unregistered_op; registered
                 ops go through preprocess → layout lookup/infer → execute → wrap.
    Expectation: RuntimeError for unregistered ops or when preprocess returns None.
    """

    def _make_dispatcher(self, layout_infer_ops=None):
        """Create a bare OpDispatcher-like object."""
        from hyper_parallel.core.shard._op_dispatch import OpDispatcher
        d = object.__new__(OpDispatcher)
        d.layout_infer_ops = layout_infer_ops or {}
        return d

    def test_unregistered_op_raises(self):
        """Op not in layout_infer_ops raises RuntimeError."""
        from hyper_parallel.core.shard._op_dispatch import OpDispatcher
        d = self._make_dispatcher()
        with self.assertRaises(RuntimeError):
            d._dispatch_layout_infer("NoSuchOp", lambda: None, (), {})

    @patch("hyper_parallel.core.shard._op_dispatch.LayoutCacheManager")
    def test_preprocess_non_none_completes_dispatch(self, mock_cache_cls):
        """preprocess returning non-None triggers full dispatch: lookup → execute → wrap."""
        from hyper_parallel.core.shard._op_dispatch import OpDispatcher

        mock_dist_op = MagicMock()
        mock_dist_op.preprocess.return_value = ([], {}, [])
        mock_dist_op.infer_layout.return_value = (MagicMock(), None)
        mock_dist_op.get_expand_impl.return_value = None
        mock_dist_op.wrap_output.return_value = "dispatch_result"
        mock_cache_cls.get_instance.return_value.distributed_op.return_value = mock_dist_op

        d = self._make_dispatcher(layout_infer_ops={"TestOp": {}})
        result = d._dispatch_layout_infer("TestOp", lambda: None, (), {})
        self.assertEqual(result, "dispatch_result")
        mock_dist_op.preprocess.assert_called_once()
        mock_dist_op.wrap_output.assert_called_once()

    @patch("hyper_parallel.core.shard._op_dispatch.LayoutCacheManager")
    def test_preprocess_returns_none_raises(self, mock_cache_cls):
        """preprocess returning None raises RuntimeError (three-phase dispatch required)."""
        from hyper_parallel.core.shard._op_dispatch import OpDispatcher

        mock_dist_op = MagicMock()
        mock_dist_op.preprocess.return_value = None
        mock_cache_cls.get_instance.return_value.distributed_op.return_value = mock_dist_op

        d = self._make_dispatcher(layout_infer_ops={"TestOp": {}})

        with self.assertRaises(RuntimeError):
            d._dispatch_layout_infer("TestOp", lambda: None, (1,), {})


class TestLookupOrInferLayout(unittest.TestCase):
    """
    Feature: OpDispatcher._lookup_or_infer_layout
    Description: Cache lookup, infer_layout, get_expand_impl, wrap_output.
    Expectation: Cache hit returns cached result; miss calls infer/expand; output wrapped.
    """

    @patch("hyper_parallel.core.shard._op_dispatch.LayoutCacheManager")
    @patch("hyper_parallel.core.shard._op_dispatch.platform")
    def test_lookup_cache_miss_path(self, mock_platform, mock_cache_cls):
        """Cache miss: infer_layout and get_expand_impl called; wrap_output called via dispatch."""
        from hyper_parallel.core.shard._op_dispatch import OpDispatcher

        mock_platform.get_op_name.return_value = "TestOp"

        mock_output_layouts = (MagicMock(),)
        mock_dist_op = MagicMock()
        mock_dist_op.preprocess.return_value = ([], {}, [])
        mock_dist_op.infer_layout.return_value = (mock_output_layouts, None)
        mock_dist_op.get_expand_impl.return_value = None
        mock_dist_op.wrap_output.return_value = "wrapped_output"

        layout_cache = {}
        mock_cache_cls.get_instance.return_value.get_layout_cache.return_value = layout_cache
        mock_cache_cls.get_instance.return_value.distributed_op.return_value = mock_dist_op

        def mock_func(*args, **kwargs):
            return "py_output"

        d = object.__new__(OpDispatcher)
        d.layout_infer_ops = {"TestOp": {}}
        result = d._dispatch_layout_infer("TestOp", mock_func, (), {})
        self.assertEqual(result, "wrapped_output")
        mock_dist_op.infer_layout.assert_called_once()
        mock_dist_op.wrap_output.assert_called_once()

    @patch("hyper_parallel.core.shard._op_dispatch.LayoutCacheManager")
    @patch("hyper_parallel.core.shard._op_dispatch.platform")
    def test_lookup_cache_hit_path(self, mock_platform, mock_cache_cls):
        """Cache hit: infer_layout NOT called; cached op_impl and infer_result used."""
        from hyper_parallel.core.shard._op_dispatch import OpDispatcher
        # LayoutCacheKey is re-imported here because _reload_op_dispatch_with_env_str may have
        # reloaded the module, making the module-level LayoutCacheKey a stale class definition
        # that no longer matches what OpDispatcher uses internally.
        from hyper_parallel.core.shard._op_dispatch import LayoutCacheKey  # pylint: disable=W0404,W0621

        mock_platform.get_op_name.return_value = "CachedOp"

        mock_output_layouts = (MagicMock(),)
        mock_dist_op = MagicMock()
        mock_dist_op.preprocess.return_value = (["arg"], {}, ["cached_val"])
        mock_dist_op.wrap_output.return_value = "cached_wrapped"

        cache_values = ["cached_val"]
        cache_key = LayoutCacheKey.from_cache_values(cache_values)

        mock_impl = MagicMock(return_value="cached_py_output")
        cached_infer = ((mock_output_layouts,), None)

        layout_cache = {"CachedOp": {cache_key: (cached_infer, mock_impl)}}
        mock_cache_cls.get_instance.return_value.get_layout_cache.return_value = layout_cache
        mock_cache_cls.get_instance.return_value.distributed_op.return_value = mock_dist_op

        def mock_func(*args, **kwargs):
            return "should_not_be_called"

        d = object.__new__(OpDispatcher)
        d.layout_infer_ops = {"CachedOp": {}}
        result = d._dispatch_layout_infer("CachedOp", mock_func, (), {})
        mock_dist_op.infer_layout.assert_not_called()
        mock_impl.assert_called_once()


class TestDispatchRandomPath(unittest.TestCase):
    """
    Feature: OpDispatcher.dispatch random op and auto-register paths.
    Description: Random ops route to _dispatch_random_op; auto-registered ops route
                 to _dispatch_layout_infer.
    Expectation: Correct routing for each case.
    """

    @patch("hyper_parallel.core.shard._op_dispatch.platform")
    def test_dispatch_random_op_path(self, mock_platform):
        """dispatch routes random ops to _dispatch_random_op."""
        from hyper_parallel.core.shard._op_dispatch import OpDispatcher

        mock_platform.get_op_name.return_value = "random_test_op_coverage"

        d = object.__new__(OpDispatcher)
        d.whitelist = []
        d._random_ops = {"random_test_op_coverage"}
        d._random_ms_ops = set()
        d.layout_infer_ops = {}
        d._dispatch_random_op = MagicMock(return_value="random_result")

        fake_op = MagicMock()
        result = d.dispatch(fake_op, (1,), {})
        self.assertEqual(result, "random_result")
        d._dispatch_random_op.assert_called_once()

    @patch("hyper_parallel.core.shard._op_dispatch.get_distributed_op")
    @patch("hyper_parallel.core.shard._op_dispatch.platform")
    def test_dispatch_auto_register_path(self, mock_platform, mock_get_dist_op):
        """dispatch auto-registers programmatically registered ops."""
        from hyper_parallel.core.shard._op_dispatch import OpDispatcher

        mock_platform.get_op_name.return_value = "auto_reg_op"
        mock_get_dist_op.return_value = MagicMock()

        d = object.__new__(OpDispatcher)
        d.whitelist = []
        d._random_ops = set()
        d._random_ms_ops = set()
        d.layout_infer_ops = {}
        d._dispatch_layout_infer = MagicMock(return_value="layout_result")

        fake_op = MagicMock()
        result = d.dispatch(fake_op, (), {})
        self.assertIn("auto_reg_op", d.layout_infer_ops)
        d._dispatch_layout_infer.assert_called_once()

    @patch("hyper_parallel.core.shard._op_dispatch.platform")
    def test_dispatch_ms_random_op_path(self, mock_platform):
        """dispatch routes MindSpore random ops to _dispatch_random_op."""
        from hyper_parallel.core.shard._op_dispatch import OpDispatcher

        mock_platform.get_op_name.return_value = "BernoulliExt"

        d = object.__new__(OpDispatcher)
        d.whitelist = []
        d._random_ops = set()
        d._random_ms_ops = {"BernoulliExt"}
        d.layout_infer_ops = {}
        d._dispatch_random_op = MagicMock(return_value="ms_random_result")

        fake_op = MagicMock()
        result = d.dispatch(fake_op, (), {})
        self.assertEqual(result, "ms_random_result")

    @patch("hyper_parallel.core.shard._op_dispatch.platform")
    def test_dispatch_new_ms_random_ops_path(self, mock_platform):
        """Newly whitelisted MindSpore random kernels route to _dispatch_random_op."""
        from hyper_parallel.core.shard._op_dispatch import OpDispatcher

        for op_name in ("NormalTensorTensor", "UniformExt", "RandExt"):
            with self.subTest(op_name=op_name):
                mock_platform.get_op_name.return_value = op_name
                d = object.__new__(OpDispatcher)
                d.whitelist = []
                d._random_ops = set()
                d._random_ms_ops = {op_name}
                d.layout_infer_ops = {}
                d._dispatch_random_op = MagicMock(return_value=f"{op_name}_result")
                result = d.dispatch(MagicMock(), (), {})
                self.assertEqual(result, f"{op_name}_result")
                d._dispatch_random_op.assert_called_once()


class TestRandomOpReturnsSelf(unittest.TestCase):
    """
    Feature: OpDispatcher._random_op_returns_self.
    Description: Explicit MindSpore inplace set, FuncDropoutExt special case,
                 and Torch '_' suffix drive inplace return semantics.
    Expectation: Correct True/False for representative op names and arguments.
    """

    def test_random_inplace_ms_ops_return_true(self):
        from hyper_parallel.core.shard._op_dispatch import OpDispatcher

        for op_name in OpDispatcher._RANDOM_INPLACE_MS_OPS:
            with self.subTest(op_name=op_name):
                self.assertTrue(OpDispatcher._random_op_returns_self(op_name, (), {}))

    def test_random_inplace_ms_ops_subset_of_random_ms_ops(self):
        from hyper_parallel.core.shard._op_dispatch import OpDispatcher

        d = OpDispatcher()
        self.assertTrue(
            OpDispatcher._RANDOM_INPLACE_MS_OPS.issubset(d._random_ms_ops),
            "Every MindSpore inplace random kernel must also be in _random_ms_ops.",
        )

    def test_ms_out_of_place_random_ops_return_false(self):
        """MindSpore out-of-place random kernels must not return self."""
        from hyper_parallel.core.shard._op_dispatch import OpDispatcher

        for op_name in (
            "BernoulliExt",
            "MultinomialExt",
            "NormalTensorTensor",
            "UniformExt",
            "RandExt",
            "FuncDropoutExt",
        ):
            with self.subTest(op_name=op_name):
                self.assertFalse(OpDispatcher._random_op_returns_self(op_name, (MagicMock(),), {}))

    def test_torch_inplace_suffix_returns_true(self):
        from hyper_parallel.core.shard._op_dispatch import OpDispatcher

        self.assertTrue(OpDispatcher._random_op_returns_self("normal_", (), {}))
        self.assertTrue(OpDispatcher._random_op_returns_self("bernoulli_", (), {}))

    def test_torch_out_of_place_random_ops_return_false(self):
        from hyper_parallel.core.shard._op_dispatch import OpDispatcher

        for op_name in ("bernoulli", "randn", "rand", "native_dropout"):
            with self.subTest(op_name=op_name):
                self.assertFalse(OpDispatcher._random_op_returns_self(op_name, (), {}))

    def test_non_random_inplace_name_does_not_match_prefix_heuristic(self):
        from hyper_parallel.core.shard._op_dispatch import OpDispatcher

        self.assertFalse(OpDispatcher._random_op_returns_self("InplaceAddExt", (), {}))

    def test_func_dropout_ext_inplace_via_args(self):
        from hyper_parallel.core.shard._op_dispatch import OpDispatcher

        args = (MagicMock(), 0.5, True, True, MagicMock(), MagicMock())
        self.assertTrue(OpDispatcher._random_op_returns_self("FuncDropoutExt", args, {}))
        self.assertTrue(OpDispatcher._func_dropout_ext_inplace(args, {}))

    def test_func_dropout_ext_inplace_via_kwargs(self):
        from hyper_parallel.core.shard._op_dispatch import OpDispatcher

        self.assertTrue(
            OpDispatcher._random_op_returns_self(
                "FuncDropoutExt", (MagicMock(), 0.5, True), {"inplace": True}
            )
        )
        self.assertTrue(
            OpDispatcher._func_dropout_ext_inplace((MagicMock(), 0.5, True), {"inplace": True})
        )

    def test_func_dropout_ext_out_of_place_returns_false(self):
        from hyper_parallel.core.shard._op_dispatch import OpDispatcher

        args = (MagicMock(), 0.5, True, False, MagicMock(), MagicMock())
        self.assertFalse(OpDispatcher._random_op_returns_self("FuncDropoutExt", args, {}))


class TestDispatchRandomInplaceReturnsSelf(unittest.TestCase):
    """
    Feature: OpDispatcher._dispatch_random_op return value for in-place random ops.
    Description: In-place random ops must return the input DTensor itself, not a new
                 wrapper. Torch in-place names end with '_' (e.g. normal_);
                 MindSpore in-place random kernels are listed in
                 ``OpDispatcher._RANDOM_INPLACE_MS_OPS``. ``FuncDropoutExt`` uses
                 its ``inplace`` argument instead. Non-in-place random ops (e.g.
                 Randn) return a freshly wrapped DTensor.
    Expectation: _dispatch_random_op returns self for in-place names and a new wrapper
                 for non-in-place names.
    """

    def _dispatch(self, op_name, op_call, args=(), kwargs=None):
        """Run _dispatch_random_op for op_name with RNG support disabled; return (result, first_arg)."""
        from hyper_parallel.core.shard._op_dispatch import OpDispatcher

        if kwargs is None:
            kwargs = {}
        d = object.__new__(OpDispatcher)
        d._rng_tracker = None
        if not args:
            first_arg = MagicMock(spec=DTensor)
            args = (first_arg,)
        else:
            first_arg = args[0]
        with patch("hyper_parallel.core.shard._op_dispatch.is_rng_supported_mesh", return_value=False):
            result = d._dispatch_random_op(op_name, op_call, args, kwargs)

        op_call.assert_called_once()
        return result, first_arg

    def test_inplace_ops_return_self(self):
        """Torch '_' suffix and _RANDOM_INPLACE_MS_OPS kernels return self."""
        from hyper_parallel.core.shard._op_dispatch import OpDispatcher

        for op_name in ("normal_", *OpDispatcher._RANDOM_INPLACE_MS_OPS):
            with self.subTest(op_name=op_name):
                result, first_arg = self._dispatch(op_name, MagicMock(return_value=MagicMock()))
                self.assertIs(result, first_arg)

    def test_non_inplace_op_returns_new_wrapper(self):
        """Out-of-place random ops wrap the result in a new DTensor."""
        from hyper_parallel.core.shard._op_dispatch import OpDispatcher, Tensor

        out_of_place_ops = (
            "Randn",
            "BernoulliExt",
            "UniformExt",
            "NormalTensorTensor",
            "RandExt",
        )
        for op_name in out_of_place_ops:
            with self.subTest(op_name=op_name):
                self.assertNotIn(op_name, OpDispatcher._RANDOM_INPLACE_MS_OPS)
                op_call = MagicMock(return_value=MagicMock(spec=Tensor))
                with patch.object(DTensor, "from_local", return_value="new_wrapper"):
                    result, first_arg = self._dispatch(op_name, op_call)
                self.assertIsNot(result, first_arg)
                self.assertEqual(result, "new_wrapper")

    def test_func_dropout_ext_inplace_returns_self(self):
        """FuncDropoutExt(inplace=True) returns the input DTensor, not a new wrapper."""
        first_arg = MagicMock(spec=DTensor)
        op_call = MagicMock(return_value=MagicMock())
        result, returned_first_arg = self._dispatch(
            "FuncDropoutExt",
            op_call,
            args=(first_arg, 0.5, True, True, MagicMock(), MagicMock()),
        )
        self.assertIs(result, returned_first_arg)
        self.assertIs(result, first_arg)

    def test_func_dropout_ext_inplace_via_kwargs_returns_self(self):
        """FuncDropoutExt(inplace=True) via kwargs returns the input DTensor."""
        first_arg = MagicMock(spec=DTensor)
        op_call = MagicMock(return_value=MagicMock())
        result, returned_first_arg = self._dispatch(
            "FuncDropoutExt",
            op_call,
            args=(first_arg, 0.5, True),
            kwargs={"inplace": True},
        )
        self.assertIs(result, returned_first_arg)
        self.assertIs(result, first_arg)

    def test_func_dropout_ext_out_of_place_wraps(self):
        """FuncDropoutExt(inplace=False) wraps the local result in a new DTensor."""
        from hyper_parallel.core.shard._op_dispatch import Tensor

        first_arg = MagicMock(spec=DTensor)
        op_call = MagicMock(return_value=MagicMock(spec=Tensor))
        with patch.object(DTensor, "from_local", return_value="new_wrapper"):
            result, returned_first_arg = self._dispatch(
                "FuncDropoutExt",
                op_call,
                args=(first_arg, 0.5, True, False, MagicMock(), MagicMock()),
            )
        self.assertIsNot(result, returned_first_arg)
        self.assertEqual(result, "new_wrapper")


class TestRegisterSingleDistributedOp(unittest.TestCase):
    """
    Feature: OpDispatcher._register_single_distributed_op branching.
    Description: Cover the distributed_op_module fast path and the re-raise path.
    Expectation: Module imported and class instantiated; ImportError re-raised when
                 no fallback python path is configured.
    """

    def _make_stub(self, env_python_path=""):
        """Return a minimal OpDispatcher-like stub."""
        from hyper_parallel.core.shard._op_dispatch import OpDispatcher
        stub = object.__new__(OpDispatcher)
        stub._env_python_path = env_python_path
        return stub

    def test_distributed_op_module_fast_path(self):
        """When config has 'distributed_op_module', it imports and instantiates the class."""
        from hyper_parallel.core.shard._op_dispatch import OpDispatcher

        stub = self._make_stub()
        config = {
            "distributed_op_module": "some.module",
            "distributed_op_class": "SomeClass",
        }

        mock_cls = MagicMock()
        mock_module = MagicMock(spec=["SomeClass"])
        mock_module.SomeClass = mock_cls

        with patch.object(importlib, "import_module", return_value=mock_module):
            OpDispatcher._register_single_distributed_op(stub, "TestOp", config)

        mock_cls.assert_called_once_with("TestOp")

    def test_reraise_when_no_env_python_path(self):
        """ImportError is re-raised when _env_python_path is empty."""
        from hyper_parallel.core.shard._op_dispatch import OpDispatcher

        stub = self._make_stub(env_python_path="")
        config = {
            "distributed_op_file": "nonexistent_module",
            "distributed_op_class": "SomeClass",
        }

        with patch.object(importlib, "import_module", side_effect=ModuleNotFoundError("no module")):
            with self.assertRaises(ModuleNotFoundError):
                OpDispatcher._register_single_distributed_op(stub, "TestOp", config)


class TestLoadYamlDictErrors(unittest.TestCase):
    """
    Feature: OpDispatcher._load_yaml_dict error paths.
    Description: Cover invalid directory path ValueError and duplicate key ValueError.
    Expectation: ValueError raised for bad path; ValueError raised for duplicate keys.
    """

    def _make_stub(self, yaml_dir, work_dir=""):
        from hyper_parallel.core.shard._op_dispatch import OpDispatcher
        stub = object.__new__(OpDispatcher)
        stub.yaml_dir = yaml_dir
        stub.work_dir = work_dir
        return stub

    def test_invalid_yaml_dir_raises(self):
        """Non-existent yaml_dir raises ValueError."""
        from hyper_parallel.core.shard._op_dispatch import OpDispatcher
        stub = self._make_stub("/nonexistent/path/to/yaml_dir_coverage_test_12345")
        with self.assertRaises(ValueError):
            OpDispatcher.safe_load_yaml_from_dir(stub)

    def test_duplicate_yaml_key_raises(self):
        """Two YAML files with same top-level key raises ValueError."""
        import tempfile
        from hyper_parallel.core.shard._op_dispatch import OpDispatcher

        with tempfile.TemporaryDirectory() as tmpdir:
            stub = self._make_stub(tmpdir)
            yaml_content = "DupOp:\n  op_type: test\n"
            for fname in ("a.yaml", "b.yaml"):
                with open(os.path.join(tmpdir, fname), "w", encoding="utf-8") as f:
                    f.write(yaml_content)
            with self.assertRaises((ValueError, Exception)):
                OpDispatcher.safe_load_yaml_from_dir(stub)


if __name__ == "__main__":
    unittest.main()

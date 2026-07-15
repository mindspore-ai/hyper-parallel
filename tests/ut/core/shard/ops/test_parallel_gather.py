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
"""parallel_gather test"""
import os
import unittest
from unittest.mock import patch
import numpy as np

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.dtensor import _LAYOUT_CACHE
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate, Partial
from hyper_parallel.core.shard.ops.parallel_gather import GatherDDistributedOp, GatherNdDistributedOp, IndexSelectDistributedOp
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

op = IndexSelectDistributedOp("index_select")


def _infer_one(distributed_op, cache_values):
    """Run new infer_layout(cache_values) and return the single output layout."""
    output_layouts, _ = distributed_op.infer_layout(cache_values)
    return output_layouts[0]


def _expand_impl(distributed_op, func, output_layout, cache_values):
    """Run new get_expand_impl(func, infer_result, cache_values)."""
    return distributed_op.get_expand_impl(func, ((output_layout,), None), cache_values)


def _index_select_cache(layouts, extra_args):
    """Build IndexSelect cache_values from legacy test fixtures."""
    return [layouts[0], layouts[2], extra_args[0]]


class _MockDTensor:
    """Small DTensor-like object for preprocess tests."""

    def __init__(self, layout, local_value=None, shape=None):
        self.layout = layout
        self._layout = layout
        self._local_value = local_value if local_value is not None else object()
        self.shape = shape

    def to_local(self):
        return self._local_value


class TestParallelGatherD(unittest.TestCase):
    """Unit tests for GatherDDistributedOp."""
    def setUp(self):
        """Set up test fixtures before each test method.

        Clears global caches to ensure test isolation and initializes
        the platform for testing.
        """
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()
        self.op = GatherDDistributedOp("GatherD")

    def tearDown(self):
        """Clean up after each test method."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def _setup_mock_platform(self, mock_platform, platform_type=None, world_size=8):
        """Configure common mock-platform attributes used across tests.

        Args:
            mock_platform: The MagicMock object injected by @patch.
            platform_type: Optional PlatformType to set on the mock.
            world_size: Value returned by mock_platform.get_world_size().
        """
        if platform_type is not None:
            mock_platform.platform_type = platform_type
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = world_size
        mock_platform.tensor_to_numpy.side_effect = (
            lambda t: t.numpy() if hasattr(t, "numpy") else np.array(t)
        )

    def _make_2x4_mesh(self, mock_platform):
        """Set up mock and return a standard 2x4 (dp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(
            device_type="npu",
            mesh_shape=(2, 4),
            mesh_dim_names=("dp", "mp"),
            init_backend=False
        )

    def _make_2x2x2_mesh(self, mock_platform, mesh_dim_names=("dp", "tp", "cp")):
        """Set up mock and return a standard 2x2x2 mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(
            device_type="npu",
            mesh_shape=(2, 2, 2),
            mesh_dim_names=mesh_dim_names,
            init_backend=False
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_gatherd_preprocess(self, mock_platform):
        """Verify GatherD preprocess builds positional local args and cache_values."""
        mesh = self._make_2x4_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
        index_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        input_tensor = _MockDTensor(input_layout, local_value="input")
        index_tensor = _MockDTensor(index_layout, local_value="index")

        local_args, local_kwargs, cache_values = self.op.preprocess((input_tensor, 0, index_tensor), {})

        assert local_args == ("input", 0, "index")
        assert not local_kwargs
        assert cache_values == [input_layout, index_layout, 0]

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_gatherd_data_parallel_dim0(self, mock_platform):
        """
        Feature: Data Parallel for GatherD
        Description: Input is sharded on the gather dimension, while index is replicated.
        Expectation: Output layout becomes fully replicated and enters partial sum state.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        input_placements = (Shard(0), Replicate())
        input_layout = _build_layout(mesh, input_placements, 2)
        index_placements = (Replicate(), Replicate())
        index_layout = _build_layout(mesh, index_placements, 2)
        dim = 0
        output_layout = _infer_one(self.op, [input_layout, index_layout, dim])
        expected_map = (-1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"Data parallel dim0 inference failed. Expected {expected_map}, got {output_layout.tensor_map}"
        )
        assert output_layout.is_partial(), "Sharded gather dimension should generate partial state"
        assert output_layout.partial == ['sum', None], (
            f"Expected partial ['sum', None], got {output_layout.partial}"
        )
        impl = _expand_impl(self.op, None, output_layout, [input_layout, index_layout, dim])
        assert impl is not None, "Data parallel dim0 should have expand implementation"
        assert callable(impl), "Returned impl should be a callable function"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_gatherd_data_parallel_dim1(self, mock_platform):
        """
        Feature: GatherD dim-axis sharding inference
        Description: Input is sharded on the gather dimension, while index is replicated.
        Expectation: Output layout becomes fully replicated and enters partial sum state.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        input_placements = (Replicate(), Shard(1))
        input_layout = _build_layout(mesh, input_placements, 2)
        index_placements = (Replicate(), Replicate())
        index_layout = _build_layout(mesh, index_placements, 2)
        dim = 1
        output_layout = _infer_one(self.op, [input_layout, index_layout, dim])
        expected_map = (-1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"Data parallel dim1 inference failed. Expected {expected_map}, got {output_layout.tensor_map}"
        )
        assert output_layout.is_partial(), "Sharded gather dimension should generate partial state"
        assert output_layout.partial == [None, 'sum'], (
            f"Expected partial [None, 'sum'], got {output_layout.partial}"
        )
        impl = _expand_impl(self.op, None, output_layout, [input_layout, index_layout, dim])
        assert impl is not None, "Data parallel dim1 should have expand implementation"
        assert callable(impl), "Returned impl should be a callable function"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_gatherd_input_both_shard_replicate_index(self, mock_platform):
        """
        Feature: GatherD multi-dim sharding validation
        Description: Input is sharded on both batch and sequence dimensions, while index is replicated.
        Expectation: infer_layout should reject mismatched non-dim sharding.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        input_placements = (Shard(0), Shard(1))
        input_layout = _build_layout(mesh, input_placements, 2)
        index_placements = (Replicate(), Replicate())
        index_layout = _build_layout(mesh, index_placements, 2)
        dim = 0
        with self.assertRaisesRegex(ValueError, "same sharding on non-dim axis 1"):
            _infer_one(self.op, [input_layout, index_layout, dim])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_gatherd_column_parallel(self, mock_platform):
        """
        Feature: GatherD cross-axis sharding validation
        Description: Input is sharded on the gather dimension, and index is sharded on another dimension.
        Expectation: infer_layout should reject mismatched non-dim sharding.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        input_placements = (Shard(0), Replicate())
        input_layout = _build_layout(mesh, input_placements, 2)
        index_placements = (Replicate(), Shard(1))
        index_layout = _build_layout(mesh, index_placements, 2)
        dim = 0
        with self.assertRaisesRegex(ValueError, "same sharding on non-dim axis 1"):
            _infer_one(self.op, [input_layout, index_layout, dim])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_gatherd_row_parallel(self, mock_platform):
        """
        Feature: Enhanced Model Parallel for GatherD
        Description: Both input and index sharded identically on dim axis.
        Expectation: Output follows the sharded batch mapping and has Partial Sum state.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        input_placements = (Shard(0), Replicate())
        input_layout = _build_layout(mesh, input_placements, 2)
        index_placements = (Replicate(), Shard(0))
        index_layout = _build_layout(mesh, index_placements, 2)
        dim = 0
        output_layout = _infer_one(self.op, [input_layout, index_layout, dim])
        expected_map = (0, -1)
        assert output_layout.tensor_map == expected_map, (
            f"Row parallel expand failed. Expected {expected_map}, got {output_layout.tensor_map}"
        )
        assert output_layout.is_partial(), "Row parallel should generate partial state"
        assert output_layout.partial == ['sum', None], (
            f"Expected partial ['sum', None], got {output_layout.partial}"
        )
        impl = _expand_impl(self.op, None, output_layout, [input_layout, index_layout, dim])
        assert impl is not None, "Row parallel should have expand implementation"
        assert callable(impl), "Returned impl should be a callable function"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_gatherd_input_shard_dp_index_both_shard_conflict(self, mock_platform):
        """
        Feature: GatherD conflicting shard and partial inference
        Description: Input is sharded on batch with dp, while index is sharded on both batch and
                     sequence. This makes the output require dp for both sharding and partial reduction.
        Expectation: infer_layout should raise ValueError: "Partial dim must be replicate."
        """
        mesh = self._make_2x4_mesh(mock_platform)
        input_placements = (Shard(0), Replicate())
        input_layout = _build_layout(mesh, input_placements, 2)
        index_placements = (Shard(0), Replicate())
        index_layout = _build_layout(mesh, index_placements, 2)
        dim = 0
        with self.assertRaisesRegex(ValueError, "Partial dim must be replicate."):
            _infer_one(self.op, [input_layout, index_layout, dim])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_gatherd_input_multi_shard_with_matched_index_non_dim_shard(self, mock_platform):
        """
        Feature: GatherD multi-shard inference with matched non-dim axis
        Description: Input is sharded on both batch and sequence, while index matches the input sharding
                     on the non-gather axis and stays replicated on the gather axis.
        Expectation: Output follows the index layout and enters partial sum state.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        input_placements = (Shard(0), Shard(1))
        input_layout = _build_layout(mesh, input_placements, 2)
        index_placements = (Replicate(), Shard(1))
        index_layout = _build_layout(mesh, index_placements, 2)
        dim = 0
        output_layout = _infer_one(self.op, [input_layout, index_layout, dim])
        expected_map = (-1, 0)
        assert output_layout.tensor_map == expected_map, (
            f"Matched non-dim shard inference failed. Expected {expected_map}, got {output_layout.tensor_map}"
        )
        assert output_layout.is_partial(), "Sharded gather dimension should generate partial state"
        assert output_layout.partial == ['sum', None], (
            f"Expected partial ['sum', None], got {output_layout.partial}"
        )
        impl = _expand_impl(self.op, None, output_layout, [input_layout, index_layout, dim])
        assert impl is not None, "Matched non-dim shard case should have expand implementation"
        assert callable(impl), "Returned impl should be a callable function"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_gatherd_3d_mesh_input_multi_shard(self, mock_platform):
        """
        Feature: GatherD 3D mesh sharding inference
        Description: Mesh shape (2,2,2) with dims ("dp","tp","cp").
                     Input is sharded on the gather axis by tp, and index is sharded on the same
                     gather axis by cp, while non-dim axes remain replicated.
        Expectation: Output follows the index layout and enters partial sum state.
        """
        mesh = self._make_2x2x2_mesh(mock_platform, mesh_dim_names=("dp", "tp", "cp"))
        input_placements = (Replicate(), Shard(1), Replicate())
        input_layout = _build_layout(mesh, input_placements, 3)
        index_placements = (Replicate(), Replicate(), Shard(1))
        index_layout = _build_layout(mesh, index_placements, 3)
        dim = 1
        output_layout = _infer_one(self.op, [input_layout, index_layout, dim])
        expected_map = (-1, 0, -1)
        assert output_layout.tensor_map == expected_map, (
            f"3D mesh multi-shard inference failed. Expected {expected_map}, got {output_layout.tensor_map}"
        )
        assert output_layout.is_partial(), "3D mesh with sharded index should generate partial state"
        assert output_layout.partial == [None, 'sum', None], (
            f"Expected partial [None, 'sum', None], got {output_layout.partial}"
        )
        impl = _expand_impl(self.op, None, output_layout, [input_layout, index_layout, dim])
        assert impl is not None, "3D mesh with sharded index should have expand implementation"
        assert callable(impl), "Returned impl should be a callable function"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_gatherd_3d_mesh_matched_non_dim_shard(self, mock_platform):
        """
        Feature: GatherD 3D mesh sharding inference
        Description: Mesh shape (2,2,2) with dims ("dp","tp","cp").
                     Input and index match on non-gather axes, while the gather axis is sharded on
                     different mesh axes.
        Expectation: Output follows the index layout and enters partial sum state.
        """
        mesh = self._make_2x2x2_mesh(mock_platform, mesh_dim_names=("dp", "tp", "cp"))
        input_placements = (Shard(0), Shard(1), Replicate())
        input_layout = _build_layout(mesh, input_placements, 3)
        index_placements = (Shard(0), Replicate(), Shard(1))
        index_layout = _build_layout(mesh, index_placements, 3)
        dim = 1
        output_layout = _infer_one(self.op, [input_layout, index_layout, dim])
        expected_map = (2, 0, -1)
        assert output_layout.tensor_map == expected_map, (
            f"3D mesh matched non-dim shard inference failed. Expected {expected_map}, got {output_layout.tensor_map}"
        )
        assert output_layout.is_partial(), "3D mesh with sharded gather axis should generate partial state"
        assert output_layout.partial == [None, 'sum', None], (
            f"Expected partial [None, 'sum', None], got {output_layout.partial}"
        )
        impl = _expand_impl(self.op, None, output_layout, [input_layout, index_layout, dim])
        assert impl is not None, "3D mesh matched non-dim shard case should have expand implementation"
        assert callable(impl), "Returned impl should be a callable function"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_gatherd_rank_mismatch_error(self, mock_platform):
        """
        Feature: GatherD layout inference error handling
        Description: Input and index have different ranks.
        Expectation: Raise ValueError.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        input_placements = (Shard(0), Replicate(), Replicate())
        input_layout = _build_layout(mesh, input_placements, 3)
        index_placements = (Shard(0), Replicate())
        index_layout = _build_layout(mesh, index_placements, 2)
        dim = 1
        with self.assertRaisesRegex(ValueError, "same number of dimensions"):
            _infer_one(self.op, [input_layout, index_layout, dim])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_gatherd_invalid_dim_error(self, mock_platform):
        """
        Feature: GatherD layout inference error handling
        Description: Dim value is out of valid range.
        Expectation: Raise ValueError.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        input_placements = (Shard(0), Replicate())
        input_layout = _build_layout(mesh, input_placements, 2)
        index_layout = _build_layout(mesh, input_placements, 2)
        dim = 5
        with self.assertRaisesRegex(ValueError, "out of valid range"):
            _infer_one(self.op, [input_layout, index_layout, dim])


class TestParallelGatherNd(unittest.TestCase):
    """Unit tests for GatherNdDistributedOp."""
    def setUp(self):
        """Set up test fixtures before each test method.

        Clears global caches to ensure test isolation and initializes
        the platform for testing.
        """
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()
        self.op = GatherNdDistributedOp("GatherNd")

    def tearDown(self):
        """Clean up after each test method."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def _setup_mock_platform(self, mock_platform, platform_type=None, world_size=8):
        """Configure common mock-platform attributes used across tests.

        Args:
            mock_platform: The MagicMock object injected by @patch.
            platform_type: Optional PlatformType to set on the mock.
            world_size: Value returned by mock_platform.get_world_size().
        """
        if platform_type is not None:
            mock_platform.platform_type = platform_type
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = world_size
        mock_platform.tensor_to_numpy.side_effect = (
            lambda t: t.numpy() if hasattr(t, "numpy") else np.array(t)
        )

    def _make_1d_mesh(self, mock_platform, world_size=8):
        """Set up mock and return a standard 1D mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=world_size)
        return init_device_mesh(device_type="npu", mesh_shape=(world_size,), mesh_dim_names=("dp",))

    def _make_2x4_mesh(self, mock_platform):
        """Set up mock and return a standard 2x4 (dp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "mp"))

    def _make_2x2x2_mesh(self, mock_platform, mesh_dim_names=("dp", "cp", "mp")):
        """Set up mock and return a standard 2x2x2 mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2, 2), mesh_dim_names=mesh_dim_names)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_gathernd_preprocess_with_shape(self, mock_platform):
        """Verify GatherNd preprocess caches layouts and shapes."""
        mesh = self._make_1d_mesh(mock_platform, world_size=8)
        input_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        indices_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
        input_tensor = _MockDTensor(input_layout, local_value="input", shape=(16, 64))
        indices = _MockDTensor(indices_layout, local_value="indices", shape=(16, 2))

        local_args, local_kwargs, cache_values = self.op.preprocess((input_tensor, indices), {})

        assert local_args == ("input", "indices")
        assert not local_kwargs
        assert cache_values == [input_layout, indices_layout, (16, 64), (16, 2)]

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_gathernd_preprocess_packed_call(self, mock_platform):
        """Verify GatherNd preprocess receives clean unpacked args.

        NOTE: aclop packed-args normalization now happens upstream in
        ``OpDispatcher._dispatch_layout_infer`` via ``_normalize_aclop_args``.
        The re-packing for the kernel call is handled by ``_dispatch_layout_infer``.
        This test validates that ``preprocess`` works correctly with
        the already-unpacked (input, indices) args.
        """
        mesh = self._make_1d_mesh(mock_platform, world_size=8)
        input_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        indices_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
        input_tensor = _MockDTensor(input_layout, local_value="input", shape=(16, 64))
        indices = _MockDTensor(indices_layout, local_value="indices", shape=(16, 2))

        local_args, local_kwargs, cache_values = self.op.preprocess(
            (input_tensor, indices), {}
        )

        assert local_args == ("input", "indices")
        assert not local_kwargs
        assert cache_values == [input_layout, indices_layout, (16, 64), (16, 2)]

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_gathernd_data_parallel(self, mock_platform):
        """
        Feature: GatherNd layout inference.
        Description: Params is replicated; indices is sharded on the first dimension and replicated on the last dimension.
        Expectation: Output inherits sharding from indices[:-1].
        """
        mesh = self._make_1d_mesh(mock_platform, world_size=8)

        p_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        i_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

        out_layout = _infer_one(self.op, [p_layout, i_layout, [16, 64], [16, 2]])
        expected_map = (0,)
        assert out_layout.tensor_map == expected_map, (
            f"GatherNd data parallel failed. Expected {expected_map}, got {out_layout.tensor_map}"
        )

        # Since `get_expand_impl` is not overridden, it returns None by default.
        # The same applies to other test classes, so it is unnecessary to test its return value.
        assert _expand_impl(self.op, None, out_layout, [p_layout, i_layout, [16, 64], [16, 2]]) is None, (
            f"get_expand_impl test failed. Expected None, "
            f"got {_expand_impl(self.op, None, out_layout, [p_layout, i_layout, [16, 64], [16, 2]])}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_gathernd_model_parallel(self, mock_platform):
        """
        Feature: GatherNd layout inference.
        Description: Params is replicated; indices is sharded on the first dimension by model-parallel axis.
        Expectation: Output inherits sharding from indices[:-1].
        """
        mesh = self._make_2x4_mesh(mock_platform)

        p_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        i_layout = _build_layout(mesh, (Replicate(), Shard(0)), 2)

        out_layout = _infer_one(self.op, [p_layout, i_layout, [16, 64], [16, 2]])
        expected_map = (0,)
        assert out_layout.tensor_map == expected_map, (
            f"GatherNd model parallel failed. Expected {expected_map}, got {out_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_gathernd_replicated(self, mock_platform):
        """
        Feature: GatherNd layout inference.
        Description: Params and indices are both replicated.
        Expectation: Output is replicated.
        """
        mesh = self._make_1d_mesh(mock_platform, world_size=8)

        p_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        i_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

        out_layout = _infer_one(self.op, [p_layout, i_layout, [16, 64], [16, 2]])
        expected_map = (-1,)
        assert out_layout.tensor_map == expected_map, (
            f"GatherNd replicated failed. Expected {expected_map}, got {out_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_gathernd_indices_last_dim_sharded_error(self, mock_platform):
        """
        Feature: GatherNd layout inference.
        Description: Indices last dimension is sharded, which is not allowed.
        Expectation: Raise ValueError.
        """
        mesh = self._make_2x4_mesh(mock_platform)

        p_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        i_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)

        with self.assertRaises(ValueError):
            _infer_one(self.op, [p_layout, i_layout, [16, 64], [16, 2]])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_gathernd_params_sharded_error(self, mock_platform):
        """
        Feature: GatherNd layout inference.
        Description: Params is sharded, which is not allowed.
        Expectation: Raise ValueError.
        """
        mesh = self._make_1d_mesh(mock_platform, world_size=8)

        p_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
        i_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

        with self.assertRaises(ValueError):
            _infer_one(self.op, [p_layout, i_layout, [16, 64], [16, 2]])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_gathernd_3d_indices_no_extra_args(self, mock_platform):
        """
        Feature: GatherNd layout inference.
        Description: 3D indices without extra_args.
        Expectation: Output inherits only indices[:-1] sharding.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)

        p_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)
        i_layout = _build_layout(mesh, (Shard(0), Shard(1), Replicate()), 3)

        out_layout = _infer_one(self.op, [p_layout, i_layout, [16, 64, 32], [16, 8, 2]])
        expected_map = (2, 1, -1)
        assert out_layout.tensor_map == expected_map, (
            f"GatherNd 3D indices failed. Expected {expected_map}, got {out_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_gathernd_params_is_none_layout(self, mock_platform):
        """
        Feature: GatherNd layout inference.
        Description: Params layout is None (plain Tensor), indices is sharded.
        Expectation: Output inherits sharding from indices[:-1].
        """
        mesh = self._make_1d_mesh(mock_platform, world_size=8)

        p_layout = None
        i_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

        out_layout = _infer_one(self.op, [p_layout, i_layout, [16, 64], [16, 2]])
        expected_map = (0,)
        assert out_layout.tensor_map == expected_map, (
            f"GatherNd params None failed. Expected {expected_map}, got {out_layout.tensor_map}"
        )


class TestParallelIndexSelect(unittest.TestCase):
    """Unit tests for IndexSelectDistributedOp."""
    def setUp(self):
        """Set up test fixtures before each test method.

        Clears global caches to ensure test isolation and initializes
        the platform for testing.
        """
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def tearDown(self):
        """Clean up after each test method."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def _setup_mock_platform(self, mock_platform, platform_type=None, world_size=8):
        """Configure common mock-platform attributes used across tests.

        Args:
            mock_platform: The MagicMock object injected by @patch.
            platform_type: Optional PlatformType to set on the mock.
            world_size: Value returned by mock_platform.get_world_size().
        """
        if platform_type is not None:
            mock_platform.platform_type = platform_type
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = world_size
        mock_platform.tensor_to_numpy.side_effect = (
            lambda t: t.numpy() if hasattr(t, "numpy") else np.array(t)
        )

    def _make_2x4_mesh(self, mock_platform, mesh_dim_names=("dp", "tp")):
        """Set up mock and return a standard 2x4 mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=mesh_dim_names)

    def _make_2x2x2_mesh(self, mock_platform, mesh_dim_names=("dp", "cp", "tp")):
        """Set up mock and return a standard 2x2x2 mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2, 2), mesh_dim_names=mesh_dim_names)

    def _make_1d_mesh(self, mock_platform, world_size=8):
        """Set up mock and return a standard 1D mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=world_size)
        return init_device_mesh(device_type="npu", mesh_shape=(world_size,), mesh_dim_names=("dp",))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_preprocess(self, mock_platform):
        """Verify IndexSelect preprocess builds positional local args and cache_values."""
        mesh = self._make_2x4_mesh(mock_platform)
        p_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
        i_layout = _build_layout(mesh, (Replicate(), Replicate()), 1)
        input_tensor = _MockDTensor(p_layout, local_value="input")
        index = _MockDTensor(i_layout, local_value="index")

        local_args, local_kwargs, cache_values = op.preprocess((input_tensor, 0, index), {})

        assert local_args == ("input", 0, "index")
        assert not local_kwargs
        assert cache_values == [p_layout, i_layout, 0]

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_valid_axis_0(self, mock_platform):
        """
        Feature: Valid index_select on axis 0
        Description: Param is unsharded on axis 0 and sharded on axis 1. Index is 1D and sharded.
        Expectation: Output layout correctly splices the index alias map and the param alias map.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        p_placements = (Replicate(), Shard(1))
        p_layout = _build_layout(mesh, p_placements, 2)

        i_placements = (Shard(0), Replicate())
        i_layout = _build_layout(mesh, i_placements, 1)

        layouts = (p_layout, None, i_layout)
        extra_args = (0,)

        output_layout = _infer_one(op, _index_select_cache(layouts, extra_args))

        expected_map = (1, 0)
        assert output_layout.tensor_map == expected_map, (
            f"Axis 0 inference failed. Expected {expected_map}, got {output_layout.tensor_map}"
        )

        assert _expand_impl(op, None, output_layout, _index_select_cache(layouts, extra_args)) is None, (
            f"get_expand_impl test failed. Expected None, "
            f"got {_expand_impl(op, None, output_layout, _index_select_cache(layouts, extra_args))}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_valid_axis_1(self, mock_platform):
        """
        Feature: Valid index_select on axis 1
        Description: Param is sharded on axis 0 and unsharded on axis 1. Index is 1D and sharded.
        Expectation: Output layout combines the correct sharding strategies.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        p_placements = (Shard(0), Replicate())
        p_layout = _build_layout(mesh, p_placements, 2)

        i_placements = (Replicate(), Shard(0))
        i_layout = _build_layout(mesh, i_placements, 1)

        layouts = (p_layout, None, i_layout)
        extra_args = (1,)

        output_layout = _infer_one(op, _index_select_cache(layouts, extra_args))

        expected_map = (1, 0)
        assert output_layout.tensor_map == expected_map, (
            f"Axis 1 inference failed. Expected {expected_map}, got {output_layout.tensor_map}"
        )

        assert _expand_impl(op, None, output_layout, _index_select_cache(layouts, extra_args)) is None, (
            f"get_expand_impl test failed. Expected None, "
            f"got {_expand_impl(op, None, output_layout, _index_select_cache(layouts, extra_args))}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_valid_negative_axis(self, mock_platform):
        """
        Feature: Valid index_select with negative axis
        Description: Pass a negative axis (-1) for a 2D parameter tensor.
        Expectation: Operation calculates the correct positive axis and proceeds without errors.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        p_placements = (Shard(0), Replicate())
        p_layout = _build_layout(mesh, p_placements, 2)

        i_placements = (Replicate(), Shard(0))
        i_layout = _build_layout(mesh, i_placements, 1)

        layouts = (p_layout, None, i_layout)
        extra_args = (-1,)

        output_layout = _infer_one(op, _index_select_cache(layouts, extra_args))

        expected_map = (1, 0)
        assert output_layout.tensor_map == expected_map, (
            f"Negative axis inference failed. Expected {expected_map}, got {output_layout.tensor_map}"
        )

        assert _expand_impl(op, None, output_layout, _index_select_cache(layouts, extra_args)) is None, (
            f"get_expand_impl test failed. Expected None, "
            f"got {_expand_impl(op, None, output_layout, _index_select_cache(layouts, extra_args))}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_invalid_index_ndim(self, mock_platform):
        """
        Feature: Invalid multi-dimensional index tensor
        Description: Provide an index tensor that is 2D instead of the required 1D.
        Expectation: Raises ValueError regarding index dimension.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        p_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        i_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)

        layouts = (p_layout, None, i_layout)
        extra_args = (0,)

        with self.assertRaisesRegex(ValueError, "index is not a one-dimensional Tensor"):
            _infer_one(op, _index_select_cache(layouts, extra_args))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_invalid_axis_positive(self, mock_platform):
        """
        Feature: Invalid positive axis
        Description: Pass a positive axis value that exceeds the dimensions of the parameter tensor.
        Expectation: Raises ValueError for index out of bounds.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        p_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        i_layout = _build_layout(mesh, (Replicate(), Replicate()), 1)

        layouts = (p_layout, None, i_layout)
        extra_args = (2,)

        with self.assertRaisesRegex(ValueError, "is out of valid range"):
            _infer_one(op, _index_select_cache(layouts, extra_args))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_invalid_axis_negative(self, mock_platform):
        """
        Feature: Invalid negative axis
        Description: Pass a negative axis value that exceeds the negative dimension range.
        Expectation: Raises ValueError for index out of bounds.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        p_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        i_layout = _build_layout(mesh, (Replicate(), Replicate()), 1)

        layouts = (p_layout, None, i_layout)
        extra_args = (-3,)

        with self.assertRaisesRegex(ValueError, "is out of valid range"):
            _infer_one(op, _index_select_cache(layouts, extra_args))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_unsharded_axis_unsharded_index(self, mock_platform):
        """
        Feature: Index select layout inference
        Description: Unsharded axis with an unsharded index tensor.
        Expectation: Output layout preserves sharding on non-axis dims, axis dim remains unsharded.
        """
        mesh = self._make_2x4_mesh(mock_platform, mesh_dim_names=("dp", "mp"))

        p_placements = [Shard(0), Replicate()]
        p_layout = _build_layout(mesh, p_placements, 2)

        i_placements = [Replicate(), Replicate()]
        i_layout = _build_layout(mesh, i_placements, 1)

        output_layout = _infer_one(op, [p_layout, i_layout, 1])

        expected_map = (1, -1)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"Unsharded axis failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

        assert _expand_impl(op, None, output_layout, [p_layout, i_layout, 1]) is None, (
            f"get_expand_impl test failed. Expected None, "
            f"got {_expand_impl(op, None, output_layout, [p_layout, i_layout, 1])}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_sharded_axis_unsharded_index(self, mock_platform):
        """
        Feature: Index select layout inference
        Description: Sharded axis with an unsharded index tensor.
        Expectation: Output layout drops sharding on the axis, replacing it with the unsharded index layout.
        """
        mesh = self._make_2x4_mesh(mock_platform, mesh_dim_names=("dp", "mp"))

        p_placements = [Shard(0), Shard(1)]
        p_layout = _build_layout(mesh, p_placements, 2)

        i_placements = [Replicate(), Replicate()]
        i_layout = _build_layout(mesh, i_placements, 1)

        output_layout = _infer_one(op, [p_layout, i_layout, 1])

        expected_map = (1, -1)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"Sharded axis failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )
        impl = _expand_impl(op, None, output_layout, [p_layout, i_layout, 1])
        assert impl is not None, (
            f"get_expand_impl test failed. Expected non-None, got {impl}"
        )
        assert callable(impl), "Returned impl should be a callable function"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_negative_axis(self, mock_platform):
        """
        Feature: Index select layout inference
        Description: Using a negative axis value.
        Expectation: Output layout processes the negative axis correctly as a positive index internally.
        """
        mesh = self._make_2x4_mesh(mock_platform, mesh_dim_names=("dp", "mp"))
        p_placements = [Shard(0), Shard(1)]
        p_layout = _build_layout(mesh, p_placements, 2)

        i_placements = [Replicate(), Replicate()]
        i_layout = _build_layout(mesh, i_placements, 1)

        output_layout = _infer_one(op, [p_layout, i_layout, -1])

        expected_map = (1, -1)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"Negative axis failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )
        impl = _expand_impl(op, None, output_layout, [p_layout, i_layout, 1])
        assert impl is not None, (
            f"get_expand_impl test failed. Expected non-None, got {impl}"
        )
        assert callable(impl), "Returned impl should be a callable function"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_axis_out_of_bounds_positive(self, mock_platform):
        """
        Feature: Index select layout inference
        Description: Axis value exceeds the input tensor dimensions limits.
        Expectation: ValueError is raised with clear range context.
        """
        mesh = self._make_2x4_mesh(mock_platform, mesh_dim_names=("dp", "mp"))
        p_placements = [Shard(0), Replicate()]
        p_layout = _build_layout(mesh, p_placements, 2)
        i_placements = [Replicate(), Replicate()]
        i_layout = _build_layout(mesh, i_placements, 1)

        with self.assertRaisesRegex(ValueError, "dim value 2 is out of valid range"):
            _infer_one(op, [p_layout, i_layout, 2])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_axis_out_of_bounds_negative(self, mock_platform):
        """
        Feature: Index select layout inference
        Description: Negative axis value is smaller than -ndim.
        Expectation: ValueError is raised.
        """
        mesh = self._make_2x4_mesh(mock_platform, mesh_dim_names=("dp", "mp"))
        p_placements = [Shard(0), Replicate()]
        p_layout = _build_layout(mesh, p_placements, 2)
        i_placements = [Replicate(), Replicate()]
        i_layout = _build_layout(mesh, i_placements, 1)

        with self.assertRaisesRegex(ValueError, "dim value -3 is out of valid range"):
            _infer_one(op, [p_layout, i_layout, -3])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_invalid_index_ndim_2(self, mock_platform):
        """
        Feature: Index select layout inference
        Description: Index tensor is provided as a 2D tensor instead of 1D.
        Expectation: ValueError is raised enforcing 1D index requirement.
        """
        mesh = self._make_2x4_mesh(mock_platform, mesh_dim_names=("dp", "mp"))
        p_placements = [Shard(0), Replicate()]
        p_layout = _build_layout(mesh, p_placements, 2)

        i_placements = [Replicate(), Replicate()]
        i_layout = _build_layout(mesh, i_placements, 2)

        with self.assertRaisesRegex(ValueError, "index is not a one-dimensional Tensor"):
            _infer_one(op, [p_layout, i_layout, 0])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_invalid_partial_input(self, mock_platform):
        """
        Feature: Index select layout inference
        Description: Input layout has a Partial status, which is not supported for this op.
        Expectation: ValueError is raised via _check_partial_inputs blocking.
        """
        mesh = self._make_2x4_mesh(mock_platform, mesh_dim_names=("dp", "mp"))

        p_placements = [Partial("sum"), Replicate()]
        p_layout = _build_layout(mesh, p_placements, 2)
        i_placements = [Replicate(), Replicate()]
        i_layout = _build_layout(mesh, i_placements, 1)

        with self.assertRaisesRegex(ValueError, "Partial status which is not allowed"):
            _infer_one(op, [p_layout, i_layout, 0])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_3d_input_axis_0(self, mock_platform):
        """
        Feature: Index select layout inference
        Description: 3D input tensor, selecting on axis 0, with an unsharded index.
        Expectation: The output layout correctly replaces the first dimension's sharding with the index's sharding.
        """
        mesh = self._make_2x4_mesh(mock_platform, mesh_dim_names=("dp", "mp"))

        p_placements = [Shard(0), Shard(1), Replicate()]
        p_layout = _build_layout(mesh, p_placements, 3)

        i_placements = [Replicate(), Replicate()]
        i_layout = _build_layout(mesh, i_placements, 1)

        output_layout = _infer_one(op, [p_layout, i_layout, 0])

        expected_map = (-1, 0, -1)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"3D input axis 0 failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )
        impl = _expand_impl(op, None, output_layout, [p_layout, i_layout, 1])
        assert impl is not None, (
            f"get_expand_impl test failed. Expected non-None, got {impl}"
        )
        assert callable(impl), "Returned impl should be a callable function"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_1d_input_axis_0(self, mock_platform):
        """
        Feature: Index select layout inference
        Description: 1D input tensor and 1D unsharded index tensor.
        Expectation: Output layout is 1D and reflects the index tensor's unsharded layout.
        """
        mesh = self._make_2x4_mesh(mock_platform, mesh_dim_names=("dp", "mp"))

        p_placements = [Shard(0), Replicate()]
        p_layout = _build_layout(mesh, p_placements, 1)

        i_placements = [Replicate(), Replicate()]
        i_layout = _build_layout(mesh, i_placements, 1)

        output_layout = _infer_one(op, [p_layout, i_layout, 0])

        expected_map = (-1,)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"1D input axis 0 failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

        impl = _expand_impl(op, None, output_layout, [p_layout, i_layout, 0])
        assert impl is not None, (
            f"get_expand_impl test failed. Expected non-None, got {impl}"
        )
        assert callable(impl), "Returned impl should be a callable function"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_scalar_index_invalid(self, mock_platform):
        """
        Feature: Index select layout inference
        Description: The index tensor is a 0D scalar (invalid for PyTorch index_select).
        Expectation: ValueError is raised ensuring the index is strictly 1-dimensional.
        """
        mesh = self._make_2x4_mesh(mock_platform, mesh_dim_names=("dp", "mp"))

        p_placements = [Shard(0), Replicate()]
        p_layout = _build_layout(mesh, p_placements, 2)

        i_placements = [Replicate(), Replicate()]
        i_layout = _build_layout(mesh, i_placements, 0)

        with self.assertRaisesRegex(ValueError, "index is not a one-dimensional Tensor"):
            _infer_one(op, [p_layout, i_layout, 0])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_get_expand_impl_unsharded(self, mock_platform):
        """
        Feature: Index select expand implementation
        Description: The selected axis is not sharded ("None").
        Expectation: The implementation falls back to the dispatcher default.
        """
        mesh = self._make_2x4_mesh(mock_platform, mesh_dim_names=("dp", "mp"))

        p_placements = [Replicate(), Shard(1)]
        p_layout = _build_layout(mesh, p_placements, 2)

        def dummy_func():
            pass

        impl = op.get_expand_impl(dummy_func, None, [p_layout, None, 0])

        assert impl is None, "Should return None when axis is unsharded"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_get_expand_impl_sharded(self, mock_platform):
        """
        Feature: Index select expand implementation
        Description: The selected axis is sharded across a device mesh dimension.
        Expectation: The implementation returns a custom wrapper function for distributed execution.
        """
        mesh = self._make_2x4_mesh(mock_platform, mesh_dim_names=("dp", "mp"))

        p_placements = [Shard(0), Replicate()]
        p_layout = _build_layout(mesh, p_placements, 2)

        def dummy_func():
            pass

        impl = op.get_expand_impl(dummy_func, None, [p_layout, None, 0])

        assert impl is not dummy_func, "Should return custom wrapper when axis is sharded"
        assert impl.__name__ == "expand_impl", "Wrapper function should be named 'expand_impl'"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_negative_axis_on_3d_tensor(self, mock_platform):
        """
        Feature: Index select layout inference
        Description: Use a negative axis (-2) on a 3D tensor to verify robust boundary and translation mapping.
        Expectation: Axis -2 correctly translates to positive axis 1, yielding correct layout mapping.
        """
        mesh = self._make_2x4_mesh(mock_platform, mesh_dim_names=("dp", "mp"))

        p_placements = [Shard(0), Shard(1), Replicate()]
        p_layout = _build_layout(mesh, p_placements, 3)

        i_placements = [Replicate(), Replicate()]
        i_layout = _build_layout(mesh, i_placements, 1)

        output_layout = _infer_one(op, [p_layout, i_layout, -2])

        expected_map = (1, -1, -1)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"Negative axis on 3D tensor failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

        impl = _expand_impl(op, None, output_layout, [p_layout, i_layout, -2])
        assert impl is not None, (
            f"get_expand_impl test failed. Expected non-None, got {impl}"
        )
        assert callable(impl), "Returned impl should be a callable function"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_4d_param_axis_2(self, mock_platform):
        """
        Feature: Index select on 4D tensor
        Description: 4D parameter tensor sharded on dim 0 and dim 3. Index select on unsharded dim 2.
        Expectation: The output layout correctly drops the index mapping for dim 2 and inserts the index tensor's map.
        """
        mesh = self._make_2x4_mesh(mock_platform)

        p_placements = [Shard(0), Shard(3)]
        p_layout = _build_layout(mesh, p_placements, 4)

        i_placements = [Replicate(), Replicate()]
        i_layout = _build_layout(mesh, i_placements, 1)

        output_layout = _infer_one(op, [p_layout, i_layout, 2])

        expected_map = (1, -1, -1, 0)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"4D tensor axis 2 failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

        assert _expand_impl(op, None, output_layout, [p_layout, i_layout, 2]) is None, (
            f"get_expand_impl test failed. Expected None,"
            f"got {_expand_impl(op, None, output_layout, [p_layout, i_layout, 2])}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_3d_mesh_fully_sharded_axis_1(self, mock_platform):
        """
        Feature: Index select with a 3D DeviceMesh
        Description: Use a 3D device mesh ("dp", "cp", "tp"). 3D Param is sharded across all 3 mesh dimensions.
        Expectation: Selecting axis 1 replaces its sharding with the index tensor's replicated layout.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)

        p_placements = [Shard(0), Shard(1), Shard(2)]
        p_layout = _build_layout(mesh, p_placements, 3)

        i_placements = [Replicate(), Replicate(), Replicate()]
        i_layout = _build_layout(mesh, i_placements, 1)

        output_layout = _infer_one(op, [p_layout, i_layout, 1])

        expected_map = (2, -1, 0)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"3D mesh fully sharded failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

        impl = _expand_impl(op, None, output_layout, [p_layout, i_layout, 1])
        assert impl is not None, (
            f"get_expand_impl test failed. Expected non-None, got {impl}"
        )
        assert callable(impl), "Returned impl should be a callable function"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_1d_mesh_param_sharded(self, mock_platform):
        """
        Feature: Index select with 1D DeviceMesh
        Description: 1D device mesh ("dp"). 2D Param is sharded on dim 0. Index is replicated.
        Expectation: Output layout processes 1D mesh correctly without crashing.
        """
        mesh = self._make_1d_mesh(mock_platform, world_size=8)

        p_placements = [Shard(0)]
        p_layout = _build_layout(mesh, p_placements, 2)

        i_placements = [Replicate()]
        i_layout = _build_layout(mesh, i_placements, 1)

        output_layout = _infer_one(op, [p_layout, i_layout, 0])

        expected_map = (-1, -1)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"1D mesh param sharded failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

        impl = _expand_impl(op, None, output_layout, [p_layout, i_layout, 0])
        assert impl is not None, (
            f"get_expand_impl test failed. Expected non-None, got {impl}"
        )
        assert callable(impl), "Returned impl should be a callable function"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_1d_mesh_index_sharded(self, mock_platform):
        """
        Feature: Index select with 1D DeviceMesh and sharded index
        Description: 1D device mesh ("dp"). 2D Param is replicated. Index is sharded on dim 0.
        Expectation: The output inherits the index's sharding on the selected axis.
        """
        mesh = self._make_1d_mesh(mock_platform, world_size=8)

        p_placements = [Replicate()]
        p_layout = _build_layout(mesh, p_placements, 2)

        i_placements = [Shard(0)]
        i_layout = _build_layout(mesh, i_placements, 1)

        output_layout = _infer_one(op, [p_layout, i_layout, 1])

        expected_map = (-1, 0)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"1D mesh index sharded failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

        assert _expand_impl(op, None, output_layout, [p_layout, i_layout, 1]) is None, (
            f"get_expand_impl test failed. Expected None,"
            f"got {_expand_impl(op, None, output_layout, [p_layout, i_layout, 1])}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_1d_param_negative_axis(self, mock_platform):
        """
        Feature: Index select on 1D tensor with negative axis
        Description: 1D Param tensor, axis is -1.
        Expectation: Correctly resolves -1 to 0 and computes layout correctly.
        """
        mesh = self._make_2x4_mesh(mock_platform)

        p_placements = [Shard(0), Replicate()]
        p_layout = _build_layout(mesh, p_placements, 1)

        i_placements = [Replicate(), Replicate()]
        i_layout = _build_layout(mesh, i_placements, 1)

        output_layout = _infer_one(op, [p_layout, i_layout, -1])

        expected_map = (-1,)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"1D param negative axis failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

        impl = _expand_impl(op, None, output_layout, [p_layout, i_layout, -1])
        assert impl is not None, (
            f"get_expand_impl test failed. Expected non-None, got {impl}"
        )
        assert callable(impl), "Returned impl should be a callable function"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_5d_param_last_axis(self, mock_platform):
        """
        Feature: Index select on 5D tensor
        Description: 5D parameter tensor, selecting the last axis (axis=4).
        Expectation: The layout maps the first 4 dimensions strictly and replaces the 5th.
        """
        mesh = self._make_2x4_mesh(mock_platform)

        p_placements = [Shard(0), Shard(4)]
        p_layout = _build_layout(mesh, p_placements, 5)

        i_placements = [Replicate(), Replicate()]
        i_layout = _build_layout(mesh, i_placements, 1)

        output_layout = _infer_one(op, [p_layout, i_layout, 4])

        expected_map = (1, -1, -1, -1, -1)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"5D param last axis failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

        impl = _expand_impl(op, None, output_layout, [p_layout, i_layout, 4])
        assert impl is not None, (
            f"get_expand_impl test failed. Expected non-None, got {impl}"
        )
        assert callable(impl), "Returned impl should be a callable function"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_output_mesh_shape_preservation(self, mock_platform):
        """
        Feature: Output layout properties
        Description: Verify the output layout correctly preserves the input mesh shape.
        Expectation: Output layout mesh shape is identical to input layout mesh shape.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        p_layout = _build_layout(mesh, [Shard(0), Replicate()], 2)
        i_layout = _build_layout(mesh, [Replicate(), Replicate()], 1)

        output_layout = _infer_one(op, [p_layout, i_layout, 0])

        assert output_layout.mesh_shape == p_layout.mesh_shape, (
            "Output mesh shape does not match input mesh shape."
        )

        impl = _expand_impl(op, None, output_layout, [p_layout, i_layout, 0])
        assert impl is not None, (
            f"get_expand_impl test failed. Expected non-None, got {impl}"
        )
        assert callable(impl), "Returned impl should be a callable function"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_output_alias_name_preservation(self, mock_platform):
        """
        Feature: Output layout properties
        Description: Verify the output layout correctly preserves the input alias names.
        Expectation: Output layout alias name tuple is identical to input alias name tuple.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        p_layout = _build_layout(mesh, [Shard(0), Replicate()], 2)
        i_layout = _build_layout(mesh, [Replicate(), Replicate()], 1)

        output_layout = _infer_one(op, [p_layout, i_layout, 0])

        assert output_layout.alias_name == p_layout.alias_name, (
            "Output alias name does not match input alias name."
        )

        impl = _expand_impl(op, None, output_layout, [p_layout, i_layout, 0])
        assert impl is not None, (
            f"get_expand_impl test failed. Expected non-None, got {impl}"
        )
        assert callable(impl), "Returned impl should be a callable function"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_output_rank_list_preservation(self, mock_platform):
        """
        Feature: Output layout properties
        Description: Verify the output layout correctly preserves the process rank list.
        Expectation: Output layout rank list is identical to input layout rank list.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        p_layout = _build_layout(mesh, [Shard(0), Replicate()], 2)
        i_layout = _build_layout(mesh, [Replicate(), Replicate()], 1)

        output_layout = _infer_one(op, [p_layout, i_layout, 0])

        assert output_layout.rank_list == p_layout.rank_list, (
            "Output rank list does not match input rank list."
        )

        impl = _expand_impl(op, None, output_layout, [p_layout, i_layout, 0])
        assert impl is not None, (
            f"get_expand_impl test failed. Expected non-None, got {impl}"
        )
        assert callable(impl), "Returned impl should be a callable function"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_expand_impl_negative_sharded_axis(self, mock_platform):
        """
        Feature: get_expand_impl with negative axis
        Description: Check if get_expand_impl correctly identifies a sharded axis when axis is negative.
        Expectation: Returns a custom `expand_impl` closure rather than the original function.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        p_layout = _build_layout(mesh, [Replicate(), Shard(1)], 2)

        def dummy_func():
            pass

        impl = op.get_expand_impl(dummy_func, None, [p_layout, None, -1])

        assert impl is not dummy_func, "Should return custom wrapper for negative sharded axis."
        assert impl.__name__ == "expand_impl", "Wrapper function should be named 'expand_impl'."

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_expand_impl_other_dims_sharded_only(self, mock_platform):
        """
        Feature: get_expand_impl with unsharded axis but other sharded dims
        Description: The target axis is unsharded, but a different dimension of the tensor is sharded.
        Expectation: Returns None because the selected axis itself requires no cross-device sync.
        """
        mesh = self._make_2x4_mesh(mock_platform)

        p_layout = _build_layout(mesh, [Shard(0), Replicate()], 2)

        def dummy_func():
            pass

        impl = op.get_expand_impl(dummy_func, None, [p_layout, None, 1])

        assert impl is None, "Should return None when target axis is unsharded, regardless of other dims."

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_partial_index_layout_invalid(self, mock_platform):
        """
        Feature: Partial layout rejection
        Description: The index layout contains a Partial placement.
        Expectation: Raises ValueError blocking Partial inputs.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        p_layout = _build_layout(mesh, [Replicate(), Replicate()], 2)

        i_placements = [Partial("sum"), Replicate()]
        i_layout = _build_layout(mesh, i_placements, 1)

        with self.assertRaisesRegex(ValueError, "Partial status which is not allowed"):
            _infer_one(op, [p_layout, i_layout, 0])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_0d_param_invalid(self, mock_platform):
        """
        Feature: 0D scalar parameter layout
        Description: The parameter tensor is a 0D scalar (which has no dimensions to index).
        Expectation: Raises ValueError regarding out-of-bounds axis.
        """
        mesh = self._make_2x4_mesh(mock_platform)

        p_layout = _build_layout(mesh, [Replicate(), Replicate()], 0)
        i_layout = _build_layout(mesh, [Replicate(), Replicate()], 1)

        with self.assertRaisesRegex(ValueError, "dim value 0 is out of valid range"):
            _infer_one(op, [p_layout, i_layout, 0])

if __name__ == "__main__":
    unittest.main()

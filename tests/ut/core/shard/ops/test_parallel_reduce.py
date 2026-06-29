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
"""Unit tests for parallel reduce distributed operators."""
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from hyper_parallel.core.dtensor.dtensor import _build_layout, _LAYOUT_CACHE
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_reduce import (
    ReduceMaxDistributedOp,
    MaxDistributedOp,
    SumExtDistributedOp,
    MeanExtDistributedOp,
    _normalize_reduce_args,
)
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP,
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS


class TestParallelReduceMax(unittest.TestCase):
    """Unit tests for ReduceMaxDistributedOp (base reduce family)."""

    def setUp(self) -> None:
        """Set up test fixtures before each test method.

        Clears global caches to ensure test isolation.
        """
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def tearDown(self) -> None:
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

    def _make_2x2x2_mesh(self, mock_platform):
        """Set up mock and return a standard 2x2x2 (dp, cp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2, 2), mesh_dim_names=("dp", "cp", "mp"))

    # ------------------------------------------------------------------
    # ReduceMaxDistributedOp — reduction scenarios
    # ------------------------------------------------------------------

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_reduce_max_data_parallel_1(self, mock_platform):
        """
        Feature: Data parallel.
        Description: reduce dp axis, keepdim=False.
        Expectation: dp axis reduced.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate()), 3)

        op = ReduceMaxDistributedOp("ReduceMax")
        cache_values = [x_layout, 0, False]
        output_layouts, _ = op.infer_layout(cache_values)
        output_layout = output_layouts[0]

        assert output_layout.tensor_map == (-1, -1), (
            f"ReduceMax failed. Expected (-1, -1), got {output_layout.tensor_map}"
        )
        # get_expand_impl is not overridden for reduce ops — returns None.
        # Verified once here; other reduction tests skip it.
        assert op.get_expand_impl(None, (output_layouts, None), cache_values) is None, (
            f"get_expand_impl should return None, "
            f"got {op.get_expand_impl(None, (output_layouts, None), cache_values)}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_reduce_max_model_parallel_2(self, mock_platform):
        """
        Feature: Model parallel.
        Description: reduce mp axis, keepdim=True.
        Expectation: mp axis -> None.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate(), Shard(2)), 3)

        op = ReduceMaxDistributedOp("ReduceMax")
        cache_values = [x_layout, 2, True]
        output_layouts, _ = op.infer_layout(cache_values)
        output_layout = output_layouts[0]

        assert output_layout.tensor_map == (-1, -1, -1), (
            f"ReduceMax failed. Expected (-1, -1, -1), got {output_layout.tensor_map}"
        )
        # get_expand_impl already verified in test_reduce_max_data_parallel_1

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_reduce_max_hybrid_parallel_3(self, mock_platform):
        """
        Feature: Hybrid parallel.
        Description: reduce cp axis, keepdim=False.
        Expectation: cp reduced, dp/mp kept.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1), Shard(2)), 3)

        op = ReduceMaxDistributedOp("ReduceMax")
        cache_values = [x_layout, 1, False]
        output_layouts, _ = op.infer_layout(cache_values)
        output_layout = output_layouts[0]

        assert output_layout.tensor_map == (2, 0), (
            f"ReduceMax failed. Expected (2, 0), got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_reduce_max_reduce_multiple_dims_4(self, mock_platform):
        """
        Feature: Reduce over multiple dims.
        Description: reduce (0, 2), keepdim=True.
        Expectation: dp/mp -> None, cp kept.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1), Shard(2)), 3)

        op = ReduceMaxDistributedOp("ReduceMax")
        cache_values = [x_layout, (0, 2), True]
        output_layouts, _ = op.infer_layout(cache_values)
        output_layout = output_layouts[0]

        assert output_layout.tensor_map == (-1, 1, -1), (
            f"ReduceMax failed. Expected (-1, 1, -1), got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_reduce_max_reduce_all_dims_5(self, mock_platform):
        """
        Feature: Reduce over all dims.
        Description: dim=None, keepdim=False.
        Expectation: all reduced.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1), Shard(2)), 3)

        op = ReduceMaxDistributedOp("ReduceMax")
        cache_values = [x_layout, None, False]
        output_layouts, _ = op.infer_layout(cache_values)
        output_layout = output_layouts[0]

        assert output_layout.tensor_map == (), (
            f"ReduceMax failed. Expected (), got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_reduce_max_negative_dim(self, mock_platform):
        """Reduce with negative dim index — -1 means last axis (mp)."""
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1), Shard(2)), 3)

        op = ReduceMaxDistributedOp("ReduceMax")
        cache_values = [x_layout, -1, False]
        output_layouts, _ = op.infer_layout(cache_values)
        output_layout = output_layouts[0]

        # dim=-1 → axis 2 (mp, tensor_map value 0); dp(2)+cp(1) kept
        assert output_layout.tensor_map == (2, 1), (
            f"ReduceMax failed. Expected (2, 1), got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_reduce_max_dim_as_list(self, mock_platform):
        """Reduce with dim as list — [0, 2], keepdim=True."""
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1), Shard(2)), 3)

        op = ReduceMaxDistributedOp("ReduceMax")
        cache_values = [x_layout, [0, 2], True]
        output_layouts, _ = op.infer_layout(cache_values)
        output_layout = output_layouts[0]

        assert output_layout.tensor_map == (-1, 1, -1), (
            f"ReduceMax failed. Expected (-1, 1, -1), got {output_layout.tensor_map}"
        )

    # ------------------------------------------------------------------
    # Preprocess tests
    # ------------------------------------------------------------------

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_preprocess_reduce_max(self, mock_platform):
        """preprocess for ReduceMax builds correct cache_values and local_args."""
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate()), 3)

        # Create a mock DTensor
        mock_tensor = MagicMock()
        mock_tensor.layout = x_layout
        mock_tensor.to_local.return_value = "local_tensor"

        op = ReduceMaxDistributedOp("ReduceMax")
        local_args, local_kwargs, cache_values = op.preprocess(
            (mock_tensor, 0, False), {}
        )

        assert local_args == ("local_tensor", 0, False), (
            f"Expected ('local_tensor', 0, False), got {local_args}"
        )
        assert not local_kwargs, (
            f"Expected empty kwargs, got {local_kwargs}"
        )
        assert cache_values[0] is x_layout, (
            f"Expected layout in cache_values[0], got {cache_values[0]}"
        )
        assert cache_values[1] == 0, (
            f"Expected dim=0 in cache_values[1], got {cache_values[1]}"
        )
        assert cache_values[2] is False, (
            f"Expected keepdim=False in cache_values[2], got {cache_values[2]}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_preprocess_max_element_wise(self, mock_platform):
        """preprocess for MaxDistributedOp routes element-wise mode correctly."""
        mesh = self._make_2x2x2_mesh(mock_platform)
        a_layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate()), 3)
        b_layout = _build_layout(mesh, (Replicate(), Shard(1), Replicate()), 3)

        mock_a = MagicMock()
        mock_a.layout = a_layout
        mock_a.to_local.return_value = "local_a"
        mock_b = MagicMock()
        mock_b.layout = b_layout
        mock_b.to_local.return_value = "local_b"

        op = MaxDistributedOp("max")
        local_args, local_kwargs, cache_values = op.preprocess(
            (mock_a, mock_b), {}
        )

        assert local_args == ("local_a", "local_b"), (
            f"Expected ('local_a', 'local_b'), got {local_args}"
        )
        assert cache_values == [a_layout, b_layout], (
            f"Expected two layouts in cache_values, got {cache_values}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_preprocess_max_reduction(self, mock_platform):
        """preprocess for MaxDistributedOp routes reduction mode correctly."""
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate()), 3)

        mock_tensor = MagicMock()
        mock_tensor.layout = x_layout
        mock_tensor.to_local.return_value = "local_tensor"

        op = MaxDistributedOp("max")
        local_args, local_kwargs, cache_values = op.preprocess(
            (mock_tensor, 1, True), {}
        )

        assert local_args == ("local_tensor", 1, True), (
            f"Expected ('local_tensor', 1, True), got {local_args}"
        )
        assert cache_values == [x_layout, 1, True], (
            f"Expected [layout, 1, True], got {cache_values}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_preprocess_maxdim_primitive(self, mock_platform):
        """preprocess for MaxDim routes positional args with empty kwargs."""
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate()), 3)

        mock_tensor = MagicMock()
        mock_tensor.layout = x_layout
        mock_tensor.to_local.return_value = "local_tensor"

        op = MaxDistributedOp("MaxDim")
        local_args, local_kwargs, cache_values = op.preprocess(
            (mock_tensor, 0, False), {}
        )

        assert local_args == ("local_tensor", 0, False), (
            f"Expected ('local_tensor', 0, False), got {local_args}"
        )
        assert not local_kwargs, (
            f"MaxDim Primitive should have empty kwargs, got {local_kwargs}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_preprocess_default_dim(self, mock_platform):
        """preprocess with torch.sum(tensor) — dim defaults to None, keepdim to False."""
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate()), 3)

        mock_tensor = MagicMock()
        mock_tensor.layout = x_layout
        mock_tensor.to_local.return_value = "local_tensor"

        op = SumExtDistributedOp("sum")
        local_args, local_kwargs, cache_values = op.preprocess(
            (mock_tensor,), {}
        )

        assert local_args == ("local_tensor", None, False), (
            f"Expected ('local_tensor', None, False), got {local_args}"
        )
        assert cache_values[1] is None, (
            f"Expected dim=None in cache_values[1], got {cache_values[1]}"
        )
        assert cache_values[2] is False, (
            f"Expected keepdim=False in cache_values[2], got {cache_values[2]}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_preprocess_sum_ext_routes_none_dtype_positional(self, mock_platform):
        """SumExt requires the dtype slot even when dtype is None."""
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate()), 3)

        mock_tensor = MagicMock()
        mock_tensor.layout = x_layout
        mock_tensor.to_local.return_value = "local_tensor"

        op = SumExtDistributedOp("SumExt")
        local_args, local_kwargs, cache_values = op.preprocess(
            (mock_tensor, None, False, None), {}
        )

        assert local_args == ("local_tensor", None, False, None), (
            f"Expected ('local_tensor', None, False, None), got {local_args}"
        )
        assert not local_kwargs, (
            f"SumExt should route dtype as positional, got {local_kwargs}"
        )
        assert cache_values == [x_layout, None, False], (
            f"Expected [layout, None, False], got {cache_values}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_preprocess_sum_ext_routes_explicit_dtype_positional(self, mock_platform):
        """preprocess routes explicit SumExt dtype as a positional argument."""
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate()), 3)

        mock_tensor = MagicMock()
        mock_tensor.layout = x_layout
        mock_tensor.to_local.return_value = "local_tensor"

        dtype = object()
        op = SumExtDistributedOp("SumExt")
        local_args, local_kwargs, cache_values = op.preprocess(
            (mock_tensor, None, False, dtype), {}
        )

        assert local_args == ("local_tensor", None, False, dtype), (
            f"Expected ('local_tensor', None, False, dtype), got {local_args}"
        )
        assert not local_kwargs, (
            f"SumExt should route explicit dtype as positional, got {local_kwargs}"
        )
        assert cache_values == [x_layout, None, False], (
            f"Expected [layout, None, False], got {cache_values}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_preprocess_mean_ext_routes_none_dtype_positional(self, mock_platform):
        """MeanExt requires the dtype slot even when dtype is None."""
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate()), 3)

        mock_tensor = MagicMock()
        mock_tensor.layout = x_layout
        mock_tensor.to_local.return_value = "local_tensor"

        op = MeanExtDistributedOp("MeanExt")
        local_args, local_kwargs, cache_values = op.preprocess(
            (mock_tensor, 1, False, None), {}
        )

        assert local_args == ("local_tensor", 1, False, None), (
            f"Expected ('local_tensor', 1, False, None), got {local_args}"
        )
        assert not local_kwargs, (
            f"MeanExt should route dtype as positional, got {local_kwargs}"
        )
        assert cache_values == [x_layout, 1, False], (
            f"Expected [layout, 1, False], got {cache_values}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_preprocess_reduce_max_rejects_dtype(self, mock_platform):
        """ReduceMax raises TypeError when dtype argument is provided."""
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate()), 3)

        mock_tensor = MagicMock()
        mock_tensor.layout = x_layout
        mock_tensor.to_local.return_value = "local_tensor"

        op = ReduceMaxDistributedOp("ReduceMax")
        with self.assertRaisesRegex(TypeError, "the `dtype` argument is not supported"):
            op.preprocess((mock_tensor, 1, False, None), {})

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_preprocess_maxdim_rejects_dtype(self, mock_platform):
        """MaxDim raises TypeError when dtype argument is provided."""
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate()), 3)

        mock_tensor = MagicMock()
        mock_tensor.layout = x_layout
        mock_tensor.to_local.return_value = "local_tensor"

        op = MaxDistributedOp("MaxDim")
        with self.assertRaisesRegex(TypeError, "the `dtype` argument is not supported"):
            op.preprocess((mock_tensor, 1, False, None), {})

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_preprocess_torch_sum_routes_dtype_keyword(self, mock_platform):
        """preprocess keeps dtype as keyword-only for torch-style reduce ops."""
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate()), 3)

        mock_tensor = MagicMock()
        mock_tensor.layout = x_layout
        mock_tensor.to_local.return_value = "local_tensor"

        dtype = object()
        op = SumExtDistributedOp("sum")
        local_args, local_kwargs, cache_values = op.preprocess(
            (mock_tensor, 1, True), {"dtype": dtype}
        )

        assert local_args == ("local_tensor", 1, True), (
            f"Expected ('local_tensor', 1, True), got {local_args}"
        )
        assert local_kwargs == {"dtype": dtype}, (
            f"Expected dtype keyword, got {local_kwargs}"
        )
        assert cache_values == [x_layout, 1, True], (
            f"Expected [layout, 1, True], got {cache_values}"
        )

    # ------------------------------------------------------------------
    # _normalize_reduce_args
    # ------------------------------------------------------------------

    def test_normalize_reduce_args_all_positional(self):
        """_normalize_reduce_args returns all-positional with empty kwargs."""
        args, kwargs = _normalize_reduce_args("tensor", 0, True)
        assert args == ("tensor", 0, True, None), (
            f"Expected ('tensor', 0, True, None), got {args}"
        )
        assert not kwargs, f"Expected empty kwargs, got {kwargs}"

    def test_normalize_reduce_args_defaults(self):
        """_normalize_reduce_args defaults dim=None, keepdim=False."""
        args, kwargs = _normalize_reduce_args("tensor")
        assert args == ("tensor", None, False, None), (
            f"Expected ('tensor', None, False, None), got {args}"
        )
        assert not kwargs, f"Expected empty kwargs, got {kwargs}"

    def test_normalize_reduce_args_dim_only(self):
        """_normalize_reduce_args with only dim specified."""
        args, kwargs = _normalize_reduce_args("tensor", dim=-1)
        assert args == ("tensor", -1, False, None), (
            f"Expected ('tensor', -1, False, None), got {args}"
        )

    def test_normalize_reduce_args_dtype_placeholder(self):
        """_normalize_reduce_args accepts frontend dtype placeholders."""
        args, kwargs = _normalize_reduce_args("tensor", None, False, None)
        assert args == ("tensor", None, False, None), (
            f"Expected ('tensor', None, False, None), got {args}"
        )
        assert not kwargs, f"Expected empty kwargs, got {kwargs}"

    # ------------------------------------------------------------------
    # Error cases
    # ------------------------------------------------------------------

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_invalid_dim_index(self, mock_platform):
        """infer_layout raises ValueError for out-of-range dim."""
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate()), 3)

        op = ReduceMaxDistributedOp("ReduceMax")
        with self.assertRaises(ValueError):
            op.infer_layout([x_layout, 5, False])


if __name__ == "__main__":
    unittest.main()
    
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
"""Unit tests for Conv1dDistributedOp."""
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from hyper_parallel.core.dtensor.dtensor import _build_layout, _LAYOUT_CACHE
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_conv1d import (
    Conv1dDistributedOp,
    _normalize_conv1d_args,
)
from hyper_parallel.core.shard.ops.parallel_ops_register import get_distributed_op
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP,
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS


class TestParallelConv1D(unittest.TestCase):
    """Unit tests for Conv1dDistributedOp."""

    def setUp(self) -> None:
        """Clear distributed state before each test."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def tearDown(self) -> None:
        """Clear distributed state after each test."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    @staticmethod
    def _get_op():
        return get_distributed_op("conv1d")

    @staticmethod
    def _setup_mock_platform(mock_platform, world_size=8):
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = world_size
        mock_platform.tensor_to_numpy.side_effect = (
            lambda tensor: tensor.numpy() if hasattr(tensor, "numpy") else np.array(tensor)
        )
        mock_platform.platform_type = MagicMock()

    def _make_2x2_mesh(self, mock_platform, mesh_dim_names=("dp", "tp")):
        self._setup_mock_platform(mock_platform, world_size=4)
        return init_device_mesh(device_type="cpu", mesh_shape=(2, 2),
                                mesh_dim_names=mesh_dim_names, init_backend=False)

    def _make_1d_mesh(self, mock_platform, world_size=4, mesh_dim_names=("tp",)):
        self._setup_mock_platform(mock_platform, world_size=world_size)
        return init_device_mesh(device_type="cpu", mesh_shape=(world_size,),
                                mesh_dim_names=mesh_dim_names, init_backend=False)

    def _make_2x2x2_mesh(self, mock_platform, mesh_dim_names=("dp", "tp", "sp")):
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="cpu", mesh_shape=(2, 2, 2),
                                mesh_dim_names=mesh_dim_names, init_backend=False)

    @staticmethod
    def _cache_values(in_layout, w_layout, b_layout=None, groups=1, bias_present=None):
        if bias_present is None:
            bias_present = b_layout is not None
        return [in_layout, w_layout, b_layout, bias_present, groups]

    def _infer_output_layout(self, in_layout, w_layout, b_layout=None, groups=1):
        """Infer Conv1d output layout and unwrap the single-output result."""
        op_inst = self._get_op()
        cache = self._cache_values(in_layout, w_layout, b_layout, groups)
        output_layouts, _ = op_inst.infer_layout(cache)
        return output_layouts[0]

    # ---- Argument normalization tests ----

    def test_all_positional_defaults(self):
        """All positional args with default values."""
        x, w = object(), object()
        args, kwargs = _normalize_conv1d_args(x, w)
        self.assertEqual(args, (x, w, None, 1, 0, 1, 1))
        self.assertEqual(kwargs, {})

    def test_all_explicit_args(self):
        """All args explicitly set."""
        x, w, b = object(), object(), object()
        args, kwargs = _normalize_conv1d_args(x, w, bias=b, stride=2, padding=1, dilation=3, groups=4)
        self.assertEqual(args, (x, w, b, 2, 1, 3, 4))
        self.assertEqual(kwargs, {})

    def test_kwargs(self):
        """Keyword arguments passed through, captured into positional tuple."""
        x, w, b = object(), object(), object()
        args, kwargs = _normalize_conv1d_args(
            input_tensor=x, weight=w, bias=b, stride=3, padding="same", dilation=2, groups=1
        )
        self.assertEqual(args, (x, w, b, 3, "same", 2, 1))
        self.assertEqual(kwargs, {})

    def test_preprocess_accepts_input_keyword(self):
        """Conv1d preprocess accepts the native input= keyword."""
        input_tensor = MagicMock()
        weight = MagicMock()

        local_args, local_kwargs, cache_values = self._get_op().preprocess(
            (), {"input": input_tensor, "weight": weight}
        )

        self.assertEqual(
            local_args,
            (input_tensor.to_local.return_value, weight.to_local.return_value, None, 1, 0, 1, 1),
        )
        self.assertEqual(local_kwargs, {})
        self.assertEqual(cache_values, [input_tensor.layout, weight.layout, None, False, 1])

    # ---- Registration test ----

    def test_conv1d_registered(self):
        """YAML registers Conv1dDistributedOp for 'conv1d'."""
        registered = get_distributed_op("conv1d")
        self.assertIsInstance(registered, Conv1dDistributedOp)

    # ---- Positive tests ----

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv1d_all_replicated(self, mock_platform):
        """All replicated on 1D mesh."""
        mesh = self._make_1d_mesh(mock_platform, world_size=4)
        in_layout = _build_layout(mesh, (Replicate(),), 3)
        w_layout = _build_layout(mesh, (Replicate(),), 3)
        output_layout = self._infer_output_layout(in_layout, w_layout)
        expected_map = (-1, -1, -1)
        self.assertEqual(output_layout.to_dict()["tensor_map"], expected_map)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv1d_data_parallel(self, mock_platform):
        """DP: input N sharded, weight replicated."""
        mesh = self._make_2x2_mesh(mock_platform)
        in_placements = (Shard(0), Replicate())
        in_layout = _build_layout(mesh, in_placements, 3)
        w_layout = _build_layout(mesh, (Replicate(), Replicate()), 3)
        output_layout = self._infer_output_layout(in_layout, w_layout)
        expected_map = (1, -1, -1)
        self.assertEqual(output_layout.to_dict()["tensor_map"], expected_map)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv1d_column_parallel(self, mock_platform):
        """CP: weight C_out sharded via tp, bias aligned."""
        mesh = self._make_2x2_mesh(mock_platform)
        in_layout = _build_layout(mesh, (Replicate(), Replicate()), 3)
        w_placements = (Replicate(), Shard(0))
        w_layout = _build_layout(mesh, w_placements, 3)
        b_layout = _build_layout(mesh, (Replicate(), Shard(0)), 1)
        output_layout = self._infer_output_layout(in_layout, w_layout, b_layout)
        expected_map = (-1, 0, -1)
        self.assertEqual(output_layout.to_dict()["tensor_map"], expected_map)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv1d_column_parallel_no_bias(self, mock_platform):
        """CP without bias."""
        mesh = self._make_1d_mesh(mock_platform, world_size=2, mesh_dim_names=("tp",))
        in_layout = _build_layout(mesh, (Replicate(),), 3)
        w_layout = _build_layout(mesh, (Shard(0),), 3)
        output_layout = self._infer_output_layout(in_layout, w_layout)
        expected_map = (-1, 0, -1)
        self.assertEqual(output_layout.to_dict()["tensor_map"], expected_map)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv1d_row_parallel(self, mock_platform):
        """RP: input/weight C_in sharded, bias=None, output Partial."""
        mesh = self._make_2x2_mesh(mock_platform)
        in_placements = (Replicate(), Shard(1))
        in_layout = _build_layout(mesh, in_placements, 3)
        w_placements = (Replicate(), Shard(1))
        w_layout = _build_layout(mesh, w_placements, 3)
        output_layout = self._infer_output_layout(in_layout, w_layout)
        expected_map = (-1, -1, -1)
        self.assertEqual(output_layout.to_dict()["tensor_map"], expected_map)
        self.assertEqual(output_layout.partial, [None, "sum"])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv1d_dp_cp(self, mock_platform):
        """DP + CP: N via dp, C_out via tp."""
        mesh = self._make_2x2_mesh(mock_platform)
        in_placements = (Shard(0), Replicate())
        in_layout = _build_layout(mesh, in_placements, 3)
        w_placements = (Replicate(), Shard(0))
        w_layout = _build_layout(mesh, w_placements, 3)
        b_layout = _build_layout(mesh, (Replicate(), Shard(0)), 1)
        output_layout = self._infer_output_layout(in_layout, w_layout, b_layout)
        expected_map = (1, 0, -1)
        self.assertEqual(output_layout.to_dict()["tensor_map"], expected_map)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv1d_dp_rp(self, mock_platform):
        """DP + RP: N via dp, C_in via tp, bias=None, output Partial."""
        mesh = self._make_2x2_mesh(mock_platform)
        in_placements = (Shard(0), Shard(1))
        in_layout = _build_layout(mesh, in_placements, 3)
        w_placements = (Replicate(), Shard(1))
        w_layout = _build_layout(mesh, w_placements, 3)
        output_layout = self._infer_output_layout(in_layout, w_layout)
        expected_map = (1, -1, -1)
        self.assertEqual(output_layout.to_dict()["tensor_map"], expected_map)
        self.assertEqual(output_layout.partial, [None, "sum"])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv1d_grouped_column_parallel(self, mock_platform):
        """Grouped CP: groups=4, C_out via tp on 1D mesh."""
        mesh = self._make_1d_mesh(mock_platform, world_size=2, mesh_dim_names=("tp",))
        in_layout = _build_layout(mesh, (Replicate(),), 3)
        w_layout = _build_layout(mesh, (Shard(0),), 3)
        b_layout = _build_layout(mesh, (Shard(0),), 1)
        output_layout = self._infer_output_layout(in_layout, w_layout, b_layout, groups=4)
        expected_map = (-1, 0, -1)
        self.assertEqual(output_layout.to_dict()["tensor_map"], expected_map)

        # Verify expand_impl exists
        op_inst = self._get_op()
        cache = self._cache_values(in_layout, w_layout, b_layout, groups=4)
        infer_result = ((output_layout,), None)
        expand_fn = op_inst.get_expand_impl(MagicMock(), infer_result, cache)
        self.assertIsNotNone(expand_fn)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv1d_depthwise_group_aligned(self, mock_platform):
        """Depthwise input and C_out sharding on the same axis produces complete shards."""
        mesh = self._make_1d_mesh(mock_platform, world_size=2, mesh_dim_names=("tp",))
        in_layout = _build_layout(mesh, (Shard(1),), 3)
        w_layout = _build_layout(mesh, (Shard(0),), 3)
        b_layout = _build_layout(mesh, (Shard(0),), 1)

        output_layout = self._infer_output_layout(in_layout, w_layout, b_layout, groups=4)

        self.assertEqual(output_layout.to_dict()["tensor_map"], (-1, 0, -1))
        self.assertFalse(output_layout.is_partial())

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv1d_multi_axis_group_aligned(self, mock_platform):
        """Group-aligned sharding requires and supports identical multi-axis aliases."""
        mesh = self._make_2x2_mesh(mock_platform)
        in_layout = _build_layout(mesh, (Shard(1), Shard(1)), 3)
        w_layout = _build_layout(mesh, (Shard(0), Shard(0)), 3)
        b_layout = _build_layout(mesh, (Shard(0), Shard(0)), 1)

        output_layout = self._infer_output_layout(in_layout, w_layout, b_layout, groups=4)

        self.assertEqual(output_layout.alias_tensor_map, ("None", ("dp", "tp"), "None"))
        self.assertFalse(output_layout.is_partial())

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv1d_dp_grouped_column_parallel(self, mock_platform):
        """DP + Grouped CP: N via dp, C_out via tp, groups=2."""
        mesh = self._make_2x2_mesh(mock_platform)
        in_placements = (Shard(0), Replicate())
        in_layout = _build_layout(mesh, in_placements, 3)
        w_placements = (Replicate(), Shard(0))
        w_layout = _build_layout(mesh, w_placements, 3)
        b_layout = _build_layout(mesh, (Replicate(), Shard(0)), 1)
        output_layout = self._infer_output_layout(in_layout, w_layout, b_layout, groups=2)
        expected_map = (1, 0, -1)
        self.assertEqual(output_layout.to_dict()["tensor_map"], expected_map)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv1d_fully_replicated_2d_mesh(self, mock_platform):
        """All replicated on 2D mesh."""
        mesh = self._make_2x2_mesh(mock_platform)
        in_layout = _build_layout(mesh, (Replicate(), Replicate()), 3)
        w_layout = _build_layout(mesh, (Replicate(), Replicate()), 3)
        output_layout = self._infer_output_layout(in_layout, w_layout)
        expected_map = (-1, -1, -1)
        self.assertEqual(output_layout.to_dict()["tensor_map"], expected_map)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv1d_negative_dim_placement(self, mock_platform):
        """Shard(-3) on N and Shard(-2) on C_in equivalent to positive indices."""
        mesh = self._make_2x2_mesh(mock_platform)
        in_placements = (Shard(-3), Shard(-2))
        in_layout = _build_layout(mesh, in_placements, 3)
        w_placements = (Replicate(), Shard(-2))
        w_layout = _build_layout(mesh, w_placements, 3)
        output_layout = self._infer_output_layout(in_layout, w_layout)
        expected_map = (1, -1, -1)
        self.assertEqual(output_layout.to_dict()["tensor_map"], expected_map)
        self.assertEqual(output_layout.partial, [None, "sum"])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv1d_multi_axis_grouped_column_parallel(self, mock_platform):
        """Grouped CP on 3D mesh — C_out sharded on 'sp' axis."""
        mesh = self._make_2x2x2_mesh(mock_platform, mesh_dim_names=("dp", "tp", "sp"))
        in_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)
        w_placements = (Replicate(), Replicate(), Shard(0))
        w_layout = _build_layout(mesh, w_placements, 3)
        output_layout = self._infer_output_layout(in_layout, w_layout, groups=4)
        expected_map = (-1, 0, -1)
        self.assertEqual(output_layout.to_dict()["tensor_map"], expected_map)

        # Verify expand_impl uses _as_axes to compute tp_size from C_out axes
        op_inst = self._get_op()
        cache = self._cache_values(in_layout, w_layout, groups=4)
        infer_result = ((output_layout,), None)
        expand_fn = op_inst.get_expand_impl(MagicMock(), infer_result, cache)
        self.assertIsNotNone(expand_fn)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv1d_groups_no_expand(self, mock_platform):
        """Groups=2 but C_out not sharded → expand_impl returns None."""
        mesh = self._make_2x2_mesh(mock_platform)
        in_layout = _build_layout(mesh, (Replicate(), Replicate()), 3)
        w_layout = _build_layout(mesh, (Replicate(), Replicate()), 3)
        output_layout = self._infer_output_layout(in_layout, w_layout, groups=2)
        expected_map = (-1, -1, -1)
        self.assertEqual(output_layout.to_dict()["tensor_map"], expected_map)

        op_inst = self._get_op()
        cache = self._cache_values(in_layout, w_layout, groups=2)
        infer_result = ((output_layout,), None)
        expand_fn = op_inst.get_expand_impl(MagicMock(), infer_result, cache)
        self.assertIsNone(expand_fn)

    # ---- Error tests ----

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv1d_error_not_3d(self, mock_platform):
        """Input not 3D → ValueError."""
        mesh = self._make_1d_mesh(mock_platform, world_size=4)
        in_layout = _build_layout(mesh, (Replicate(),), 2)
        w_layout = _build_layout(mesh, (Replicate(),), 3)
        with self.assertRaisesRegex(ValueError, "Input and weight must be 3D."):
            self._get_op().infer_layout(self._cache_values(in_layout, w_layout))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv1d_error_mesh_mismatch(self, mock_platform):
        """Input and weight mesh_shape mismatch → ValueError."""
        self._setup_mock_platform(mock_platform, world_size=4)
        mesh1 = init_device_mesh(device_type="cpu", mesh_shape=(2, 2),
                                 mesh_dim_names=("dp", "tp"), init_backend=False)
        mesh2 = init_device_mesh(device_type="cpu", mesh_shape=(4,),
                                 mesh_dim_names=("tp",), init_backend=False)
        in_layout = _build_layout(mesh1, (Replicate(), Replicate()), 3)
        w_layout = _build_layout(mesh2, (Replicate(),), 3)
        with self.assertRaisesRegex(ValueError, "mesh_shape"):
            self._get_op().infer_layout(self._cache_values(in_layout, w_layout))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv1d_error_partial_input(self, mock_platform):
        """Input with Partial → ValueError."""
        mesh = self._make_2x2_mesh(mock_platform)
        in_layout = _build_layout(mesh, (Replicate(), Replicate()), 3)
        in_layout.set_partial_by_dev_axis("dp", "sum")
        w_layout = _build_layout(mesh, (Replicate(), Replicate()), 3)
        with self.assertRaisesRegex(ValueError, "Partial status which is not allowed"):
            self._get_op().infer_layout(self._cache_values(in_layout, w_layout))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv1d_error_partial_bias(self, mock_platform):
        """Bias with Partial → ValueError."""
        mesh = self._make_2x2_mesh(mock_platform)
        in_layout = _build_layout(mesh, (Replicate(), Replicate()), 3)
        w_layout = _build_layout(mesh, (Replicate(), Replicate()), 3)
        b_layout = _build_layout(mesh, (Replicate(), Replicate()), 1)
        b_layout.set_partial_by_dev_axis("dp", "sum")
        with self.assertRaisesRegex(ValueError, "Partial status which is not allowed"):
            self._get_op().infer_layout(self._cache_values(in_layout, w_layout, b_layout))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv1d_error_shard_l(self, mock_platform):
        """L dimension sharded → ValueError."""
        mesh = self._make_2x2_mesh(mock_platform)
        in_placements = (Replicate(), Shard(2))
        in_layout = _build_layout(mesh, in_placements, 3)
        w_layout = _build_layout(mesh, (Replicate(), Replicate()), 3)
        with self.assertRaisesRegex(ValueError, "Sharding on L dimension is not supported."):
            self._get_op().infer_layout(self._cache_values(in_layout, w_layout))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv1d_error_shard_kernel_size(self, mock_platform):
        """kernel_size dimension sharded → ValueError."""
        mesh = self._make_2x2_mesh(mock_platform)
        in_layout = _build_layout(mesh, (Replicate(), Replicate()), 3)
        # weight kernel_size (dim 2) on tp
        w_placements = (Replicate(), Shard(2))
        w_layout = _build_layout(mesh, w_placements, 3)
        with self.assertRaisesRegex(ValueError, "Sharding on kernel_size dimension is not supported."):
            self._get_op().infer_layout(self._cache_values(in_layout, w_layout))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv1d_error_bias_not_1d(self, mock_platform):
        """Bias not 1D → ValueError."""
        mesh = self._make_2x2_mesh(mock_platform)
        in_layout = _build_layout(mesh, (Replicate(), Replicate()), 3)
        w_layout = _build_layout(mesh, (Replicate(), Replicate()), 3)
        b_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)  # 2D
        with self.assertRaisesRegex(ValueError, "1D"):
            self._get_op().infer_layout(self._cache_values(in_layout, w_layout, b_layout))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv1d_error_row_parallelism_groups(self, mock_platform):
        """Row Parallelism with groups > 1 → ValueError."""
        mesh = self._make_2x2_mesh(mock_platform)
        in_placements = (Replicate(), Shard(1))
        in_layout = _build_layout(mesh, in_placements, 3)
        w_placements = (Replicate(), Shard(1))
        w_layout = _build_layout(mesh, w_placements, 3)
        with self.assertRaisesRegex(ValueError, "Sharding on C_in with groups > 1 is not supported."):
            self._get_op().infer_layout(self._cache_values(in_layout, w_layout, groups=2))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv1d_error_grouped_weight_cin_sharded_only(self, mock_platform):
        """Grouped convolution with only weight C_in sharded raises ValueError."""
        mesh = self._make_2x2_mesh(mock_platform)
        in_layout = _build_layout(mesh, (Replicate(), Replicate()), 3)
        w_layout = _build_layout(mesh, (Replicate(), Shard(1)), 3)
        with self.assertRaisesRegex(ValueError, "Sharding on C_in with groups > 1 is not supported."):
            self._get_op().infer_layout(self._cache_values(in_layout, w_layout, groups=2))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv1d_error_weight_cin_sharded_only(self, mock_platform):
        """Only weight C_in sharded raises a layout mismatch ValueError."""
        mesh = self._make_2x2_mesh(mock_platform)
        in_layout = _build_layout(mesh, (Replicate(), Replicate()), 3)
        w_layout = _build_layout(mesh, (Replicate(), Shard(1)), 3)
        with self.assertRaisesRegex(ValueError, "Input C_in and Weight C_in must be sharded on the same axis."):
            self._get_op().infer_layout(self._cache_values(in_layout, w_layout))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv1d_error_cin_mismatch(self, mock_platform):
        """Input C_in and weight C_in sharded on different axes → ValueError."""
        mesh = self._make_2x2_mesh(mock_platform)
        # input: C_in (dim 1) sharded on tp
        in_placements = (Replicate(), Shard(1))
        in_layout = _build_layout(mesh, in_placements, 3)
        # weight: C_in (dim 1) sharded on dp
        w_placements = (Shard(1), Replicate())
        w_layout = _build_layout(mesh, w_placements, 3)
        with self.assertRaisesRegex(ValueError, "Input C_in and Weight C_in must be sharded on the same axis."):
            self._get_op().infer_layout(self._cache_values(in_layout, w_layout))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv1d_error_bias_mismatch(self, mock_platform):
        """Weight C_out and bias C_out sharded on different axes → ValueError."""
        mesh = self._make_2x2_mesh(mock_platform)
        in_layout = _build_layout(mesh, (Replicate(), Replicate()), 3)
        # weight C_out (dim 0) on tp
        w_placements = (Replicate(), Shard(0))
        w_layout = _build_layout(mesh, w_placements, 3)
        # bias C_out (dim 0) on dp
        b_placements = (Shard(0), Replicate())
        b_layout = _build_layout(mesh, b_placements, 1)
        with self.assertRaisesRegex(ValueError, "Weight C_out and Bias C_out must be sharded on the same axis."):
            self._get_op().infer_layout(self._cache_values(in_layout, w_layout, b_layout))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv1d_error_row_with_bias(self, mock_platform):
        """Row Parallelism with bias raises ValueError."""
        mesh = self._make_2x2_mesh(mock_platform)
        in_placements = (Replicate(), Shard(1))
        in_layout = _build_layout(mesh, in_placements, 3)
        w_placements = (Replicate(), Shard(1))
        w_layout = _build_layout(mesh, w_placements, 3)
        b_layout = _build_layout(mesh, (Replicate(), Replicate()), 1)
        with self.assertRaisesRegex(ValueError, "Row Parallelism requires bias=None."):
            self._get_op().infer_layout(self._cache_values(in_layout, w_layout, b_layout))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv1d_error_cp_rp_simultaneous(self, mock_platform):
        """Row TP + Column TP simultaneously → ValueError."""
        mesh = self._make_2x2_mesh(mock_platform)
        in_placements = (Replicate(), Shard(1))
        in_layout = _build_layout(mesh, in_placements, 3)
        w_placements = (Shard(0), Shard(1))
        w_layout = _build_layout(mesh, w_placements, 3)
        with self.assertRaisesRegex(ValueError, "Simultaneous Row and Column Parallelism is not supported."):
            self._get_op().infer_layout(self._cache_values(in_layout, w_layout))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv1d_error_groups_not_divisible(self, mock_platform):
        """groups not divisible by tp_size → ValueError."""
        mesh = self._make_1d_mesh(mock_platform, world_size=2, mesh_dim_names=("tp",))
        in_layout = _build_layout(mesh, (Replicate(),), 3)
        w_layout = _build_layout(mesh, (Shard(0),), 3)
        with self.assertRaisesRegex(ValueError, "divisible"):
            self._get_op().infer_layout(self._cache_values(in_layout, w_layout, groups=3))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv1d_error_group_aligned_groups_not_divisible(self, mock_platform):
        """Group-aligned sharding still requires groups divisible by TP size."""
        mesh = self._make_1d_mesh(mock_platform, world_size=2, mesh_dim_names=("tp",))
        in_layout = _build_layout(mesh, (Shard(1),), 3)
        w_layout = _build_layout(mesh, (Shard(0),), 3)

        with self.assertRaisesRegex(ValueError, "divisible"):
            self._get_op().infer_layout(self._cache_values(in_layout, w_layout, groups=3))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv1d_error_grouped_input_cin_cout_different_axes(self, mock_platform):
        """Grouped input C_in and weight C_out sharding on different axes is rejected."""
        mesh = self._make_2x2_mesh(mock_platform)
        in_layout = _build_layout(mesh, (Replicate(), Shard(1)), 3)
        w_layout = _build_layout(mesh, (Shard(0), Replicate()), 3)

        with self.assertRaisesRegex(ValueError, "Sharding on C_in with groups > 1"):
            self._get_op().infer_layout(self._cache_values(in_layout, w_layout, groups=2))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv1d_error_output_axis_conflict(self, mock_platform):
        """Layout rejects N and C_out using the same mesh axis."""
        mesh = self._make_1d_mesh(mock_platform, world_size=2, mesh_dim_names=("tp",))
        in_placements = (Shard(0), Replicate())
        in_layout = _build_layout(mesh, in_placements, 3)
        w_placements = (Shard(0), Replicate())
        w_layout = _build_layout(mesh, w_placements, 3)
        with self.assertRaisesRegex(ValueError, "has been set more than one"):
            self._get_op().infer_layout(self._cache_values(in_layout, w_layout))

    def test_conv1d_error_missing_layout(self):
        """Missing input or weight layout → ValueError."""
        with self.assertRaisesRegex(ValueError, "Requires at least input and weight layouts."):
            self._get_op().infer_layout([None, None, None, False, 1])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv1d_error_bias_sharded_alone(self, mock_platform):
        """Weight replicated, bias sharded alone → ValueError."""
        mesh = self._make_2x2_mesh(mock_platform)
        in_layout = _build_layout(mesh, (Replicate(), Replicate()), 3)
        w_layout = _build_layout(mesh, (Replicate(), Replicate()), 3)
        b_placements = (Replicate(), Shard(0))
        b_layout = _build_layout(mesh, b_placements, 1)
        with self.assertRaisesRegex(ValueError, "Weight C_out and Bias C_out must be sharded on the same axis."):
            self._get_op().infer_layout(self._cache_values(in_layout, w_layout, b_layout))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv1d_error_column_parallel_plain_bias(self, mock_platform):
        """Column Parallelism rejects a full non-DTensor bias explicitly."""
        mesh = self._make_2x2_mesh(mock_platform)
        in_layout = _build_layout(mesh, (Replicate(), Replicate()), 3)
        w_layout = _build_layout(mesh, (Replicate(), Shard(0)), 3)
        cache_values = self._cache_values(
            in_layout, w_layout, b_layout=None, bias_present=True
        )

        with self.assertRaisesRegex(ValueError, "bias should be a DTensor"):
            self._get_op().infer_layout(cache_values)

    # ---- Expand implementation tests ----

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv1d_expand_impl_groups_cp(self, mock_platform):
        """groups=2, CP on (2,) — expand_impl exists and produces correct values."""
        mesh = self._make_1d_mesh(mock_platform, world_size=2, mesh_dim_names=("tp",))
        in_layout = _build_layout(mesh, (Replicate(),), 3)
        w_layout = _build_layout(mesh, (Shard(0),), 3)
        b_layout = _build_layout(mesh, (Shard(0),), 1)

        op_inst = self._get_op()
        cache = self._cache_values(in_layout, w_layout, b_layout, groups=2)
        out_layouts, _ = op_inst.infer_layout(cache)

        conv1d_fn = MagicMock(wraps=torch.nn.functional.conv1d)
        expand_fn = op_inst.get_expand_impl(conv1d_fn, (out_layouts, None), cache)
        self.assertIsNotNone(expand_fn)

        # Verify actual numerical correctness
        torch.manual_seed(1234)

        # Input: N=2, C_in=4, L=8
        x = torch.randn(2, 4, 8)
        # Weight: C_out=4, C_in/groups=2, k=3
        w = torch.randn(4, 2, 3)
        b = torch.randn(4)

        # Native conv1d result with groups=2
        native_result = conv1d_fn(x, w, b, groups=2)

        # Rank 0 owns the first half of C_out and its corresponding bias.
        expand_result_rank0 = expand_fn(x, w[:2], b[:2], groups=2)

        # The output should match native_result[:, :2, :] for the handled groups
        expected = native_result[:, :2, :]
        torch.testing.assert_close(expand_result_rank0, expected, rtol=1e-4, atol=1e-4)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv1d_expand_impl_group_aligned_does_not_slice_input(self, mock_platform):
        """Group-aligned expand passes the local input unchanged and adjusts groups."""
        mesh = self._make_1d_mesh(mock_platform, world_size=2, mesh_dim_names=("tp",))
        in_layout = _build_layout(mesh, (Shard(1),), 3)
        w_layout = _build_layout(mesh, (Shard(0),), 3)
        b_layout = _build_layout(mesh, (Shard(0),), 1)
        cache = self._cache_values(in_layout, w_layout, b_layout, groups=4)
        output_layouts, _ = self._get_op().infer_layout(cache)
        func = MagicMock(return_value=object())
        expand_fn = self._get_op().get_expand_impl(func, (output_layouts, None), cache)
        self.assertIsNotNone(expand_fn)

        local_input = torch.randn(2, 2, 8)
        local_weight = torch.randn(2, 1, 3)
        local_bias = torch.randn(2)
        expand_fn(local_input, local_weight, local_bias, 1, 0, 1, 4)

        func.assert_called_once_with(local_input, local_weight, local_bias, 1, 0, 1, 2)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv1d_expand_impl_no_cp(self, mock_platform):
        """No CP → expand_impl returns None."""
        mesh = self._make_1d_mesh(mock_platform, world_size=2, mesh_dim_names=("tp",))
        in_layout = _build_layout(mesh, (Replicate(),), 3)
        w_layout = _build_layout(mesh, (Replicate(),), 3)
        cache = self._cache_values(in_layout, w_layout, groups=2)
        out_layouts, _ = self._get_op().infer_layout(cache)
        expand_fn = self._get_op().get_expand_impl(None, (out_layouts, None), cache)
        self.assertIsNone(expand_fn)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv1d_expand_impl_groups1(self, mock_platform):
        """CP but groups=1 → expand_impl returns None."""
        mesh = self._make_1d_mesh(mock_platform, world_size=2, mesh_dim_names=("tp",))
        in_layout = _build_layout(mesh, (Replicate(),), 3)
        w_layout = _build_layout(mesh, (Shard(0),), 3)
        cache = self._cache_values(in_layout, w_layout, groups=1)
        out_layouts, _ = self._get_op().infer_layout(cache)
        expand_fn = self._get_op().get_expand_impl(None, (out_layouts, None), cache)
        self.assertIsNone(expand_fn)

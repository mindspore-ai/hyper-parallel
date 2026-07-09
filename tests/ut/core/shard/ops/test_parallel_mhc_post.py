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
"""Unit tests for NpuMhcPostDistributedOp."""
import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from hyper_parallel.core.dtensor.dtensor import _build_layout, _LAYOUT_CACHE
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.shard.ops.parallel_mhc_post import (
    NpuMhcPostDistributedOp,
    _normalize_mhc_post_args,
)
from hyper_parallel.core.shard.ops.parallel_ops_register import get_distributed_op
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh, _DEVICE_MESH_MAP
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS, PlatformType


class TestNpuMhcPost(unittest.TestCase):
    """Unit tests for NpuMhcPostDistributedOp."""

    def setUp(self):
        """Clear global state before each test."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def tearDown(self):
        """Clear global state after each test."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def _setup_mock_platform(self, mock_platform, world_size=8):
        """Configure mock platform for mesh creation."""
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = world_size
        mock_platform.split_group.return_value = MagicMock()
        mock_platform.tensor_to_numpy.side_effect = (
            lambda t: t.asnumpy() if hasattr(t, "asnumpy") else np.array(t)
        )

    def _make_1d_mesh(self, mock_platform, size=8, name="dp"):
        """Return a 1-D mesh of the given size."""
        self._setup_mock_platform(mock_platform, world_size=size)
        return init_device_mesh(device_type="npu", mesh_shape=(size,), mesh_dim_names=(name,))

    def _make_2d_mesh(self, mock_platform, shape=(2, 2), names=("dp", "cp_tp")):
        """Return a 2-D mesh; cp_tp is mesh dim index 1 by default."""
        self._setup_mock_platform(mock_platform, world_size=shape[0] * shape[1])
        return init_device_mesh(device_type="npu", mesh_shape=shape, mesh_dim_names=names)

    @staticmethod
    def _get_op():
        return get_distributed_op("npu_mhc_post")

    # ------------------------------------------------------------------
    # _normalize_mhc_post_args
    # ------------------------------------------------------------------

    def test_normalize_args_positional_1(self):
        """
        Feature: _normalize_mhc_post_args normalises positional arguments.
        Description: Call with 4 positional args.
        Expectation: Each arg is at the expected index; kwargs empty.
        """
        x, h_res, h_out, h_post = object(), object(), object(), object()
        args, kwargs = _normalize_mhc_post_args(x, h_res, h_out, h_post)
        self.assertIs(args[0], x)
        self.assertIs(args[1], h_res)
        self.assertIs(args[2], h_out)
        self.assertIs(args[3], h_post)
        self.assertEqual(kwargs, {})

    # ------------------------------------------------------------------
    # YAML registration
    # ------------------------------------------------------------------

    def test_yaml_registration_2(self):
        """
        Feature: YAML registry resolves npu_mhc_post to a DistributedOp.
        Description: Call get_distributed_op with the op name.
        Expectation: Returns a non-None DistributedOp instance.
        """
        op = self._get_op()
        self.assertIsNotNone(op)

    # ------------------------------------------------------------------
    # preprocess
    # ------------------------------------------------------------------

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_preprocess_ms_path_3(self, mock_platform):
        """
        Feature: preprocess extracts local tensors on MindSpore path.
        Description: All 4 inputs are DTensor mocks; platform is MINDSPORE.
        Expectation: local_args has 4 elements; local_kwargs is empty.
        """
        mock_platform.platform_type = PlatformType.MINDSPORE
        self._setup_mock_platform(mock_platform)

        def _mk():
            dt = MagicMock(spec=DTensor)
            dt.to_local.return_value = MagicMock()
            dt.layout = MagicMock()
            return dt

        dtensor_x, dtensor_h_res, dtensor_h_out, dtensor_h_post = _mk(), _mk(), _mk(), _mk()

        op = self._get_op()
        local_args, local_kwargs, cache_values = op.preprocess(
            (dtensor_x, dtensor_h_res, dtensor_h_out, dtensor_h_post), {}
        )

        assert len(local_args) == 4, (
            f"Expected 4 local_args, got {len(local_args)}"
        )
        assert local_kwargs == {}, f"Expected empty kwargs, got {local_kwargs}"

    # ------------------------------------------------------------------
    # infer_layout — success cases
    # ------------------------------------------------------------------

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_all_replicated_4(self, mock_platform):
        """
        Feature: infer_layout with fully replicated inputs.
        Description: All 4 inputs replicated on 1-D mesh; npu_mhc_post has 1 output.
        Expectation: Returns 1 output layout; tensor_map all -1; extra is None.
        """
        mesh = self._make_1d_mesh(mock_platform, size=8)

        # BSND: x [B,S,N,D]=4D, h_res [B,S,N,N]=4D, h_out [B,S,D]=3D, h_post [B,S,N]=3D
        x_layout = _build_layout(mesh, (Replicate(),), 4)
        h_res_layout = _build_layout(mesh, (Replicate(),), 4)
        h_out_layout = _build_layout(mesh, (Replicate(),), 3)
        h_post_layout = _build_layout(mesh, (Replicate(),), 3)

        for layout in (x_layout, h_res_layout, h_out_layout, h_post_layout):
            layout.is_partial = MagicMock(return_value=False)

        op = self._get_op()
        out_layouts, extra = op.infer_layout(
            [x_layout, h_res_layout, h_out_layout, h_post_layout]
        )

        assert len(out_layouts) == 1, (
            f"npu_mhc_post emits 1 output; expected 1 layout, got {len(out_layouts)}"
        )
        assert out_layouts[0].tensor_map == (-1, -1, -1, -1), (
            f"Expected all-replicated tensor_map, got {out_layouts[0].tensor_map}"
        )
        assert extra is None, f"Expected extra=None, got {extra}"
        # get_expand_impl is not overridden — verify once here.
        assert op.get_expand_impl(None, (out_layouts, extra), [x_layout]) is None, (
            f"get_expand_impl should return None, "
            f"got {op.get_expand_impl(None, (out_layouts, extra), [x_layout])}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_data_parallel_5(self, mock_platform):
        """
        Feature: infer_layout with B-dim sharded (data parallel).
        Description: x sharded on dim 0 (batch); N and D replicated.
        Expectation: Output tensor_map mirrors x sharding on B.
        """
        mesh = self._make_1d_mesh(mock_platform, size=4, name="dp")

        # BSND: x [B,S,N,D]=4D, h_res [B,S,N,N]=4D, h_out [B,S,D]=3D, h_post [B,S,N]=3D
        x_layout = _build_layout(mesh, (Shard(0),), 4)
        h_res_layout = _build_layout(mesh, (Shard(0),), 4)
        h_out_layout = _build_layout(mesh, (Shard(0),), 3)
        h_post_layout = _build_layout(mesh, (Shard(0),), 3)

        for layout in (x_layout, h_res_layout, h_out_layout, h_post_layout):
            layout.is_partial = MagicMock(return_value=False)

        op = self._get_op()
        out_layouts, extra = op.infer_layout(
            [x_layout, h_res_layout, h_out_layout, h_post_layout]
        )

        assert len(out_layouts) == 1, (
            f"Expected 1 output layout, got {len(out_layouts)}"
        )
        assert out_layouts[0].tensor_map == x_layout.tensor_map, (
            f"Output tensor_map {out_layouts[0].tensor_map} should match "
            f"x tensor_map {x_layout.tensor_map}"
        )
        assert out_layouts[0].tensor_map[0] == 0, (
            f"B dimension (dim 0) should be sharded on mesh dim 0, "
            f"got tensor_map[0]={out_layouts[0].tensor_map[0]}"
        )
        assert extra is None, f"Expected extra=None, got {extra}"

    # ------------------------------------------------------------------
    # infer_layout — error cases
    # ------------------------------------------------------------------

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_partial_input_raises_6(self, mock_platform):
        """
        Feature: infer_layout rejects inputs in Partial state.
        Description: x_layout.is_partial() returns True.
        Expectation: ValueError is raised.
        """
        mesh = self._make_1d_mesh(mock_platform, size=8)

        x_layout = _build_layout(mesh, (Replicate(),), 4)
        x_layout.is_partial = MagicMock(return_value=True)

        op = self._get_op()
        with self.assertRaises(ValueError):
            op.infer_layout([x_layout])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_sharded_n_dim_raises_7(self, mock_platform):
        """
        Feature: infer_layout rejects N-dimension sharding.
        Description: x sharded on dim 2 (N); N must be replicated per spec.
        Expectation: ValueError mentioning 'N dimension'.
        """
        mesh = self._make_1d_mesh(mock_platform, size=8)

        # BSND: x [B,S,N,D]=4D, h_res [B,S,N,N]=4D, h_out [B,S,D]=3D, h_post [B,S,N]=3D
        x_layout = _build_layout(mesh, (Shard(2),), 4)
        h_res_layout = _build_layout(mesh, (Replicate(),), 4)
        h_out_layout = _build_layout(mesh, (Replicate(),), 3)
        h_post_layout = _build_layout(mesh, (Replicate(),), 3)

        for layout in (x_layout, h_res_layout, h_out_layout, h_post_layout):
            layout.is_partial = MagicMock(return_value=False)

        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "N dimension"):
            op.infer_layout([x_layout, h_res_layout, h_out_layout, h_post_layout])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_tnd_replicated_10(self, mock_platform):
        """
        Feature: infer_layout with TND format (3D inputs), all replicated.
        Description: x is [T,N,D] (3D tensor_map); all inputs replicated.
        Expectation: 1 output layout; tensor_map all -1; extra is None.
        """
        mesh = self._make_1d_mesh(mock_platform, size=8)

        x_layout = _build_layout(mesh, (Replicate(),), 3)
        h_res_layout = _build_layout(mesh, (Replicate(),), 3)
        h_out_layout = _build_layout(mesh, (Replicate(),), 2)
        h_post_layout = _build_layout(mesh, (Replicate(),), 2)

        for layout in (x_layout, h_res_layout, h_out_layout, h_post_layout):
            layout.is_partial = MagicMock(return_value=False)

        op = self._get_op()
        out_layouts, extra = op.infer_layout(
            [x_layout, h_res_layout, h_out_layout, h_post_layout]
        )

        assert len(out_layouts) == 1, (
            f"Expected 1 output layout for TND format, got {len(out_layouts)}"
        )
        assert out_layouts[0].tensor_map == (-1, -1, -1), (
            f"Expected all-replicated TND tensor_map (-1,-1,-1), "
            f"got {out_layouts[0].tensor_map}"
        )
        assert extra is None, f"Expected extra=None, got {extra}"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_tnd_data_parallel_11(self, mock_platform):
        """
        Feature: infer_layout with TND format, T-dim data parallel.
        Description: x [T,N,D] sharded on T (dim 0); all replicated otherwise.
        Expectation: Output tensor_map mirrors x's T sharding.
        """
        mesh = self._make_1d_mesh(mock_platform, size=4, name="dp")

        x_layout = _build_layout(mesh, (Shard(0),), 3)
        h_res_layout = _build_layout(mesh, (Shard(0),), 3)
        h_out_layout = _build_layout(mesh, (Shard(0),), 2)
        h_post_layout = _build_layout(mesh, (Shard(0),), 2)

        for layout in (x_layout, h_res_layout, h_out_layout, h_post_layout):
            layout.is_partial = MagicMock(return_value=False)

        op = self._get_op()
        out_layouts, extra = op.infer_layout(
            [x_layout, h_res_layout, h_out_layout, h_post_layout]
        )

        assert len(out_layouts) == 1, (
            f"Expected 1 output layout, got {len(out_layouts)}"
        )
        assert out_layouts[0].tensor_map == x_layout.tensor_map, (
            f"TND DP output tensor_map {out_layouts[0].tensor_map} should match "
            f"x tensor_map {x_layout.tensor_map}"
        )
        assert extra is None, f"Expected extra=None, got {extra}"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_tnd_n_axis_raises_12(self, mock_platform):
        """
        Feature: infer_layout rejects N-axis sharding on TND format.
        Description: x [T,N,D] sharded on N (dim 1); N must be replicated.
        Expectation: ValueError mentioning 'N dimension'.
        """
        mesh = self._make_2d_mesh(mock_platform)

        # x [T,N,D] with N sharded on cp_tp (dim 1)
        x_layout = _build_layout(mesh, (Replicate(), Shard(1)), 3)
        h_res_layout = _build_layout(mesh, (Replicate(),), 3)
        h_out_layout = _build_layout(mesh, (Replicate(),), 2)
        h_post_layout = _build_layout(mesh, (Replicate(),), 2)

        for layout in (x_layout, h_res_layout, h_out_layout, h_post_layout):
            layout.is_partial = MagicMock(return_value=False)

        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "N dimension"):
            op.infer_layout([x_layout, h_res_layout, h_out_layout, h_post_layout])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_tnd_h_res_wrong_ndim_13(self, mock_platform):
        """
        Feature: infer_layout rejects TND h_res with wrong number of dimensions.
        Description: TND format; h_res passed as 4D instead of required 3D.
        Expectation: ValueError mentioning h_res dimensions.
        """
        mesh = self._make_1d_mesh(mock_platform, size=8)

        x_layout = _build_layout(mesh, (Replicate(),), 3)
        h_res_layout = _build_layout(mesh, (Replicate(),), 4)  # 4D — wrong for TND
        h_out_layout = _build_layout(mesh, (Replicate(),), 2)
        h_post_layout = _build_layout(mesh, (Replicate(),), 2)

        for layout in (x_layout, h_res_layout, h_out_layout, h_post_layout):
            layout.is_partial = MagicMock(return_value=False)

        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "h_res tensor should have 3 dimensions"):
            op.infer_layout([x_layout, h_res_layout, h_out_layout, h_post_layout])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_sharded_d_dim_raises_14(self, mock_platform):
        """
        Feature: infer_layout rejects D-dimension sharding for BSND format.
        Description: x sharded on dim 3 (D); D must be replicated per spec.
        Expectation: ValueError mentioning 'D dimension'.
        """
        mesh = self._make_1d_mesh(mock_platform, size=8)

        x_layout = _build_layout(mesh, (Shard(3),), 4)
        h_res_layout = _build_layout(mesh, (Replicate(),), 3)
        h_out_layout = _build_layout(mesh, (Replicate(),), 2)
        h_post_layout = _build_layout(mesh, (Replicate(),), 2)

        for layout in (x_layout, h_res_layout, h_out_layout, h_post_layout):
            layout.is_partial = MagicMock(return_value=False)

        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "D dimension"):
            op.infer_layout([x_layout, h_res_layout, h_out_layout, h_post_layout])


if __name__ == "__main__":
    unittest.main()

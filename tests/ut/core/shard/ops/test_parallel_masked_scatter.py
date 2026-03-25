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
"""parallel_masked_scatter test"""
import os
import unittest
from unittest.mock import patch
import numpy as np
os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_masked_scatter import MaskedScatterDistributedOp
from hyper_parallel.platform import get_platform
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS, PlatformType

op = MaskedScatterDistributedOp("masked_scatter")


class TestParallelMaskedScatter(unittest.TestCase):
    """Unit tests for MaskedScatterDistributedOp."""
    def setUp(self):
        """Set up test fixtures before each test method.

        Clears global caches to ensure test isolation and initializes
        the platform for testing.
        """
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        self.platform = get_platform()

    def tearDown(self):
        """Clean up after each test method."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

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
        return init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "mp"))

    def _make_1d_mesh(self, mock_platform, world_size=8):
        """Set up mock and return a standard 1D mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=world_size)
        return init_device_mesh(device_type="npu", mesh_shape=(world_size,), mesh_dim_names=("dp",))

    def _make_2x2x2_mesh(self, mock_platform, mesh_dim_names=("dp", "tp", "pp")):
        """Set up mock and return a standard 2x2x2 mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2, 2), mesh_dim_names=mesh_dim_names)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_masked_scatter_success(self, mock_platform):
        """
        Feature: MaskedScatter with fully replicated inputs
        Description: All inputs (input, mask, source) are Replicated
        Expectation: Output layout is the same as input layout (Replicated)
        """
        mesh = self._make_2x4_mesh(mock_platform)

        input_placements = (Replicate(), Replicate())
        input_layout = _build_layout(mesh, input_placements, 2)

        mask_placements = (Replicate(), Replicate())
        mask_layout = _build_layout(mesh, mask_placements, 2)

        source_placements = (Replicate(),)
        source_layout = _build_layout(mesh, source_placements, 1)

        output_layout = op.infer_layout((input_layout, mask_layout, source_layout))

        expected_map = (-1, -1)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"Expected fully replicated output (-1, -1), but got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_masked_scatter_fail_sharded_input(self, mock_platform):
        """
        Feature: MaskedScatter with sharded input
        Description: The 'input' tensor is sharded
        Expectation: ValueError raised (masked_scatter does not support sharded inputs)
        """
        mesh = self._make_2x4_mesh(mock_platform)

        input_placements = (Shard(0), Replicate())
        input_layout = _build_layout(mesh, input_placements, 2)

        mask_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        source_layout = _build_layout(mesh, (Replicate(),), 1)

        with self.assertRaisesRegex(ValueError, r"(?s)Input 0 .* is sharded"):
            op.infer_layout((input_layout, mask_layout, source_layout))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_masked_scatter_fail_sharded_mask(self, mock_platform):
        """
        Feature: MaskedScatter with sharded mask
        Description: The 'mask' tensor is sharded
        Expectation: ValueError raised (masked_scatter does not support sharded mask)
        """
        mesh = self._make_2x4_mesh(mock_platform)

        input_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

        mask_placements = (Replicate(), Shard(1))
        mask_layout = _build_layout(mesh, mask_placements, 2)

        source_layout = _build_layout(mesh, (Replicate(),), 1)

        with self.assertRaisesRegex(ValueError, r"(?s)Input 1 .* is sharded"):
            op.infer_layout((input_layout, mask_layout, source_layout))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_masked_scatter_fail_sharded_source(self, mock_platform):
        """
        Feature: MaskedScatter with sharded source
        Description: The 'source' tensor is sharded
        Expectation: ValueError raised (masked_scatter does not support sharded source)
        """
        mesh = self._make_2x4_mesh(mock_platform)

        input_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        mask_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

        source_placements = (Shard(0),)
        source_layout = _build_layout(mesh, source_placements, 1)

        with self.assertRaisesRegex(ValueError, r"(?s)Input 2 .* is sharded"):
            op.infer_layout((input_layout, mask_layout, source_layout))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_masked_scatter_success_scalar(self, mock_platform):
        """
        Feature: MaskedScatter with scalar inputs
        Description: Inputs are 0-D scalars (fully replicated implicitly)
        Expectation: Success, output is scalar layout
        """
        mesh = self._make_2x4_mesh(mock_platform)

        scalar_layout = _build_layout(mesh, (), 0)

        output_layout = op.infer_layout((scalar_layout, scalar_layout, scalar_layout))

        assert output_layout.to_dict()["tensor_map"] == (), "Expected scalar output (empty tensor_map)"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_masked_scatter_success_high_dim2(self, mock_platform):
        """
        Feature: MaskedScatter with high-dimensional tensors
        Description: 4D tensors (e.g. NCHW image batch) fully replicated
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)

        placements = (Replicate(), Replicate(), Replicate(), Replicate())
        layout_4d = _build_layout(mesh, placements, 4)

        layout_1d = _build_layout(mesh, (Replicate(),), 1)

        output_layout = op.infer_layout((layout_4d, layout_4d, layout_1d))

        expected_map = (-1, -1, -1, -1)
        assert output_layout.to_dict()["tensor_map"] == expected_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_masked_scatter_success_mixed_ranks(self, mock_platform):
        """
        Feature: MaskedScatter with mixed rank inputs
        Description: Input/Mask are 3D, Source is 1D (typical PyTorch usage)
        Expectation: Success, output follows input layout
        """
        mesh = self._make_2x4_mesh(mock_platform)

        layout_3d = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)
        layout_1d = _build_layout(mesh, (Replicate(),), 1)

        output_layout = op.infer_layout((layout_3d, layout_3d, layout_1d))

        expected_map = (-1, -1, -1)
        assert output_layout.to_dict()["tensor_map"] == expected_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_masked_scatter_fail_shard_last_dim(self, mock_platform):
        """
        Feature: MaskedScatter fail on last dim sharding
        Description: Sharding the last dimension (common in Tensor Parallelism)
        Expectation: ValueError raised
        """
        mesh = self._make_2x4_mesh(mock_platform)

        input_placements = (Replicate(), Shard(1))
        input_layout = _build_layout(mesh, input_placements, 2)

        mask_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        source_layout = _build_layout(mesh, (Replicate(),), 1)

        with self.assertRaisesRegex(ValueError, r"(?s)Input 0 .* is sharded"):
            op.infer_layout((input_layout, mask_layout, source_layout))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_masked_scatter_success_1d_mesh(self, mock_platform):
        """
        Feature: MaskedScatter on 1D Mesh
        Description: Verify logic works on a simple 1D device mesh
        Expectation: Success if replicated
        """
        mesh = self._make_1d_mesh(mock_platform, world_size=8)

        layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        source = _build_layout(mesh, (Replicate(),), 1)

        output_layout = op.infer_layout((layout, layout, source))
        assert output_layout.to_dict()["tensor_map"] == (-1, -1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_masked_scatter_success_3d_mesh(self, mock_platform):
        """
        Feature: MaskedScatter on 3D Mesh
        Description: Verify logic works on complex 3D device mesh
        Expectation: Success if replicated
        """
        mesh = self._make_2x2x2_mesh(mock_platform)

        layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        source = _build_layout(mesh, (Replicate(),), 1)

        output_layout = op.infer_layout((layout, layout, source))
        assert output_layout.to_dict()["tensor_map"] == (-1, -1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_masked_scatter_fail_multiple_sharded(self, mock_platform):
        """
        Feature: MaskedScatter with multiple sharded inputs
        Description: Both Input and Mask are sharded
        Expectation: ValueError raised (should fail fast on the first one)
        """
        mesh = self._make_2x4_mesh(mock_platform)

        input_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
        mask_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
        source_layout = _build_layout(mesh, (Replicate(),), 1)

        with self.assertRaisesRegex(ValueError, r"(?s)Input 0 .* is sharded"):
            op.infer_layout((input_layout, mask_layout, source_layout))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_masked_scatter_ignore_extra_args(self, mock_platform):
        """
        Feature: MaskedScatter ignoring extra args
        Description: Verify that passing extra arguments doesn't break inference
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)

        layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        source = _build_layout(mesh, (Replicate(),), 1)

        output_layout = op.infer_layout(
            (layout, layout, source),
            extra_args={"some_arg": 123}
        )

        assert output_layout.to_dict()["tensor_map"] == (-1, -1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_masked_scatter_success_scalar_inputs(self, mock_platform):
        """
        Feature: MaskedScatter with scalar inputs
        Description: All inputs are 0-D scalars.
        Expectation: Success, output is 0-D scalar layout (empty tensor_map).
        """
        mesh = self._make_2x4_mesh(mock_platform)

        scalar_layout = _build_layout(mesh, (), 0)

        output_layout = op.infer_layout((scalar_layout, scalar_layout, scalar_layout))

        assert output_layout.to_dict()["tensor_map"] == ()

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_masked_scatter_success_1d_tensors(self, mock_platform):
        """
        Feature: MaskedScatter with 1D tensors
        Description: All inputs are 1D vectors, fully replicated.
        Expectation: Success.
        """
        mesh = self._make_2x4_mesh(mock_platform)

        layout_1d = _build_layout(mesh, (Replicate(),), 1)

        output_layout = op.infer_layout((layout_1d, layout_1d, layout_1d))

        assert output_layout.to_dict()["tensor_map"] == (-1,)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_masked_scatter_success_source_broadcast(self, mock_platform):
        """
        Feature: MaskedScatter with source broadcast
        Description: Input and Mask are 3D, but Source is 1D (Common PyTorch usage).
        Expectation: Success, output layout matches Input layout.
        """
        mesh = self._make_2x4_mesh(mock_platform)

        layout_3d = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)
        layout_source = _build_layout(mesh, (Replicate(),), 1)

        output_layout = op.infer_layout((layout_3d, layout_3d, layout_source))

        assert output_layout.to_dict()["tensor_map"] == (-1, -1, -1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_masked_scatter_success_high_dim(self, mock_platform):
        """
        Feature: MaskedScatter with high-dimensional tensors
        Description: 5D tensors (e.g. video batch), fully replicated.
        Expectation: Success.
        """
        mesh = self._make_2x4_mesh(mock_platform)

        placements = (Replicate(),) * 5
        layout_5d = _build_layout(mesh, placements, 5)

        output_layout = op.infer_layout((layout_5d, layout_5d, layout_5d))

        assert output_layout.to_dict()["tensor_map"] == (-1, -1, -1, -1, -1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_masked_scatter_fail_1d_mesh_sharded(self, mock_platform):
        """
        Feature: MaskedScatter failure on 1D Mesh
        Description: Sharding on a simple 1D mesh.
        Expectation: ValueError raised.
        """
        mesh = self._make_1d_mesh(mock_platform, world_size=8)

        input_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
        replicated_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

        with self.assertRaisesRegex(ValueError, r"(?s)Input 0 .* is sharded"):
            op.infer_layout((input_layout, replicated_layout, replicated_layout))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_masked_scatter_fail_3d_mesh_sharded_inner(self, mock_platform):
        """
        Feature: MaskedScatter failure on 3D Mesh inner dimension
        Description: Sharding on the 'tp' dimension of a (dp, tp, pp) mesh.
        Expectation: ValueError raised.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)

        input_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
        replicated_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

        with self.assertRaisesRegex(ValueError, r"(?s)Input 0 .* is sharded"):
            op.infer_layout((input_layout, replicated_layout, replicated_layout))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_masked_scatter_fail_last_dim_sharding(self, mock_platform):
        """
        Feature: MaskedScatter failure on last dimension sharding
        Description: Input is sharded on the last dimension (common in Tensor Parallelism).
        Expectation: ValueError raised.
        """
        mesh = self._make_2x4_mesh(mock_platform)

        input_placements = (Replicate(), Shard(1))
        input_layout = _build_layout(mesh, input_placements, 2)

        layout_ok = _build_layout(mesh, (Replicate(), Replicate()), 2)

        with self.assertRaisesRegex(ValueError, r"(?s)Input 0 .* is sharded"):
            op.infer_layout((input_layout, layout_ok, layout_ok))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_masked_scatter_fail_sharded_mask_only(self, mock_platform):
        """
        Feature: MaskedScatter failure when only Mask is sharded
        Description: Input is Replicated, but Mask is sharded.
        Expectation: ValueError raised identifying Input 1 (Mask).
        """
        mesh = self._make_2x4_mesh(mock_platform)

        input_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        mask_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
        source_layout = _build_layout(mesh, (Replicate(),), 1)

        with self.assertRaisesRegex(ValueError, r"(?s)Input 1 .* is sharded"):
            op.infer_layout((input_layout, mask_layout, source_layout))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_masked_scatter_fail_multiple_bad_inputs(self, mock_platform):
        """
        Feature: MaskedScatter failure order
        Description: Input and Source are both sharded.
        Expectation: ValueError raised for Input 0 (Fail Fast).
        """
        mesh = self._make_2x4_mesh(mock_platform)

        bad_layout_2d = _build_layout(mesh, (Shard(0), Replicate()), 2)
        bad_layout_1d = _build_layout(mesh, (Shard(0),), 1)

        with self.assertRaisesRegex(ValueError, r"(?s)Input 0 .* is sharded"):
            op.infer_layout((bad_layout_2d, bad_layout_2d, bad_layout_1d))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_masked_scatter_ignore_extra_args2(self, mock_platform):
        """
        Feature: MaskedScatter robustness with extra_args
        Description: Verify that passing unknown extra arguments does not cause failure.
        Expectation: Success.
        """
        mesh = self._make_2x4_mesh(mock_platform)

        layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        source = _build_layout(mesh, (Replicate(),), 1)

        output_layout = op.infer_layout(
            (layout, layout, source),
            extra_args={"some_flag": True, "value": 100}
        )

        assert output_layout.to_dict()["tensor_map"] == (-1, -1)


if __name__ == "__main__":
    unittest.main()

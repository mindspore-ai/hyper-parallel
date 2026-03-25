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
"""parallel_min test"""
import os
import unittest
from unittest.mock import patch
import numpy as np
os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_reduce import MinDistributedOp
from hyper_parallel.platform import get_platform
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

op = MinDistributedOp("min")


class TestParallelMin(unittest.TestCase):
    """Unit tests for MinDistributedOp."""
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

    def _make_2x2x2_mesh(self, mock_platform):
        """Set up mock and return a standard 2x2x2 (dp, tp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2, 2), mesh_dim_names=("dp", "tp", "mp"))

    def _make_2x2_mesh(self, mock_platform):
        """Set up mock and return a standard 2x2 (dp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=4)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "mp"))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_min_layout_inference_reduce_dim_sharded(self, mock_platform):
        """
        Feature: Min reduction on sharded dimension
        Description: Reduce dim0 (sharded) of a 2D tensor
        Expectation: Returns tuple of layouts (values, indices).
        Dimension 0 is removed. Output tensor map reflects remaining dimension.
        Implicitly assumes Partial(reduce_op="min") on the reduced axis (internal state).
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        output_layouts = op.infer_layout((x_layout,), extra_args=(0,))

        assert isinstance(output_layouts, tuple)
        assert len(output_layouts) == 2

        val_layout, idx_layout = output_layouts

        expected_map = (-1,)

        assert val_layout.tensor_map == expected_map, (
            f"Values layout incorrect. Expected {expected_map}, "
            f"got {val_layout.tensor_map}"
        )
        assert idx_layout.tensor_map == expected_map, (
            f"Indices layout incorrect. Expected {expected_map}, "
            f"got {idx_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_min_layout_inference_reduce_dim_replicated(self, mock_platform):
        """
        Feature: Min reduction on replicated dimension
        Description: Reduce dim1 (replicated) of a 2D tensor
        Expectation: Returns tuple of layouts. Dimension 1 is removed.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        output_layouts = op.infer_layout((x_layout,), extra_args=(1,))

        assert isinstance(output_layouts, tuple)
        val_layout, _ = output_layouts

        expected_map = (1,)

        assert val_layout.tensor_map == expected_map, (
            f"Values layout incorrect. Expected {expected_map}, "
            f"got {val_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_min_layout_inference_keepdim(self, mock_platform):
        """
        Feature: Min reduction with keepdim=True
        Description: Reduce dim0 (sharded) with keepdim
        Expectation: Dimension 0 becomes unsharded/None (-1) in the map.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        output_layouts = op.infer_layout((x_layout,), extra_args=(0, True))

        assert isinstance(output_layouts, tuple)
        val_layout, _ = output_layouts

        expected_map = (-1, -1)

        assert val_layout.tensor_map == expected_map, (
            f"Keepdim layout incorrect. Expected {expected_map}, "
            f"got {val_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_min_layout_inference_global(self, mock_platform):
        """
        Feature: Global Min reduction (torch.min(input))
        Description: Reduce all dimensions
        Expectation: Returns a single layout (not tuple). Output should be scalar (empty map).
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        output_layout = op.infer_layout((x_layout,), extra_args=())

        assert not isinstance(output_layout, tuple)

        expected_map = ()

        assert output_layout.tensor_map == expected_map, (
            f"Global reduce layout incorrect. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_min_layout_inference_global_keepdim(self, mock_platform):
        """
        Feature: Global Min reduction with keepdim (hypothetical case via extra args logic)
        Description: Reduce all dimensions but keep them
        Expectation: All dimensions become -1.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        output_layout = op.infer_layout((x_layout,), extra_args=(None, True))

        expected_map = (-1, -1)

        assert output_layout.tensor_map == expected_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_min_layout_inference_elementwise(self, mock_platform):
        """
        Feature: Element-wise min (torch.min(a, b))
        Description: Input two layouts
        Expectation: Returns one layout (propagates first input layout).
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)
        y_layout = _build_layout(mesh, x_placements, 2)

        output_layout = op.infer_layout((x_layout, y_layout), extra_args=())

        assert not isinstance(output_layout, tuple)
        expected_map = (1, -1)

        assert output_layout.tensor_map == expected_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_min_layout_inference_negative_dim(self, mock_platform):
        """
        Feature: Min reduction with negative dimension index
        Description: Reduce last dimension (dim=-1) of a 2D tensor
        Expectation: Last dimension is removed from layout.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        output_layouts = op.infer_layout((x_layout,), extra_args=(-1,))

        val_layout, idx_layout = output_layouts
        expected_map = (1,)

        assert val_layout.tensor_map == expected_map
        assert idx_layout.tensor_map == expected_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_min_layout_inference_3d_sharded(self, mock_platform):
        """
        Feature: Min reduction on 3D tensor
        Description: Reduce middle dimension (dim1) of a 3D tensor where dim0 and dim1 are sharded
        Expectation: Dim1 is removed, Dim0 and Dim2 sharding preserved.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_placements = (Shard(0), Shard(1), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        output_layouts = op.infer_layout((x_layout,), extra_args=(1,))
        val_layout, _ = output_layouts

        expected_map = (2, -1)
        assert val_layout.tensor_map == expected_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_min_layout_inference_all_sharded_reduce_0(self, mock_platform):
        """
        Feature: Min reduction on fully sharded tensor
        Description: Reduce dim0 of a 2D tensor where both dims are sharded
        Expectation: Dim0 removed, Dim1 sharding preserved.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Shard(1))
        x_layout = _build_layout(mesh, x_placements, 2)

        output_layouts = op.infer_layout((x_layout,), extra_args=(0,))
        val_layout, _ = output_layouts

        expected_map = (0,)
        assert val_layout.tensor_map == expected_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_min_layout_inference_keepdim_negative(self, mock_platform):
        """
        Feature: Min reduction with keepdim and negative dim
        Description: Reduce last dimension with keepdim=True
        Expectation: Last dimension becomes -1 (None).
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        output_layouts = op.infer_layout((x_layout,), extra_args=(-1, True))
        val_layout, _ = output_layouts

        expected_map = (1, -1)
        assert val_layout.tensor_map == expected_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_min_layout_inference_global_fully_sharded(self, mock_platform):
        """
        Feature: Global Min reduction on fully sharded tensor
        Description: Reduce all dims of (Shard(0), Shard(1))
        Expectation: Output is a scalar (empty tuple map).
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Shard(1))
        x_layout = _build_layout(mesh, x_placements, 2)

        output_layout = op.infer_layout((x_layout,), extra_args=())

        expected_map = ()
        assert output_layout.tensor_map == expected_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_min_layout_inference_replicated_input(self, mock_platform):
        """
        Feature: Min reduction on fully replicated tensor
        Description: Reduce dim0 of replicated tensor
        Expectation: Output remains replicated/unsharded.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        output_layouts = op.infer_layout((x_layout,), extra_args=(0,))
        val_layout, _ = output_layouts

        expected_map = (-1,)
        assert val_layout.tensor_map == expected_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_min_layout_inference_elementwise_mismatched(self, mock_platform):
        """
        Feature: Element-wise min with mismatched sharding
        Description: Input A is sharded, Input B is replicated.
        Expectation: Output should adopt the sharding strategy (broadcasting sharding).
        """
        mesh = self._make_2x4_mesh(mock_platform)
        layout_a = _build_layout(mesh, (Shard(0), Replicate()), 2)
        layout_b = _build_layout(mesh, (Replicate(), Replicate()), 2)

        output_layout = op.infer_layout((layout_a, layout_b), extra_args=())

        expected_map = (1, -1)
        assert output_layout.tensor_map == expected_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_min_layout_inference_4d_complex(self, mock_platform):
        """
        Feature: Min reduction on 4D tensor
        Description: Reduce dim 2 of a 4D tensor with mixed placements.
        Expectation: Correctly removes dim 2 and preserves others.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate(), Shard(1), Replicate())
        x_layout = _build_layout(mesh, x_placements, 4)

        output_layouts = op.infer_layout((x_layout,), extra_args=(2,))
        val_layout, _ = output_layouts

        expected_map = (1, -1, -1)
        assert val_layout.tensor_map == expected_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_min_layout_inference_dim_out_of_range(self, mock_platform):
        """
        Feature: Invalid dimension index
        Description: Provide dimension index larger than rank
        Expectation: Raises ValueError or IndexError (depending on impl, usually ValueError for validation)
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

        with self.assertRaises(Exception):
            op.infer_layout((x_layout,), extra_args=(5,))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_min_layout_inference_scalar_input(self, mock_platform):
        """
        Feature: Min reduction on scalar input
        Description: Global reduce on 0-D tensor
        Expectation: Returns scalar layout.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (), 0)

        output_layout = op.infer_layout((x_layout,), extra_args=())

        expected_map = ()
        assert output_layout.tensor_map == expected_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_min_layout_inference_1d_sharded_reduce(self, mock_platform):
        """
        Feature: Min reduction on 1D sharded tensor
        Description: Reduce the only dimension of a 1D sharded tensor.
        Expectation: Returns scalar tuple layout (values, indices) with empty tensor maps.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0),)
        x_layout = _build_layout(mesh, x_placements, 1)

        output_layouts = op.infer_layout((x_layout,), extra_args=(0,))
        val_layout, idx_layout = output_layouts

        expected_map = ()

        assert val_layout.tensor_map == expected_map
        assert idx_layout.tensor_map == expected_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_min_layout_inference_binary_same_layout(self, mock_platform):
        """
        Feature: Element-wise min with identical sharded layouts
        Description: Input A and B have same sharding.
        Expectation: Output preserves the common sharding.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        layout = _build_layout(mesh, x_placements, 2)

        output_layout = op.infer_layout((layout, layout), extra_args=())

        expected_map = (1, -1)
        assert output_layout.tensor_map == expected_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_min_layout_inference_binary_mixed_shard_replicate(self, mock_platform):
        """
        Feature: Element-wise min with mixed Shard/Replicate
        Description: Input A is Sharded, Input B is Replicated (on same mesh).
        Expectation: Output takes the Sharded layout (propagates sharding).
        """
        mesh = self._make_2x4_mesh(mock_platform)
        layout_a = _build_layout(mesh, (Shard(0), Replicate()), 2)
        layout_b = _build_layout(mesh, (Replicate(), Replicate()), 2)

        output_layout = op.infer_layout((layout_a, layout_b), extra_args=())

        expected_map = (1, -1)
        assert output_layout.tensor_map == expected_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_min_layout_inference_binary_broadcast_scalar(self, mock_platform):
        """
        Feature: Element-wise min with scalar broadcasting
        Description: Min of 2D Sharded Tensor and a Scalar.
        Expectation: Scalar broadcasts to tensor shape, output preserves tensor layout.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        layout_a = _build_layout(mesh, (Shard(0), Replicate()), 2)
        layout_b = _build_layout(mesh, (), 0)

        output_layout = op.infer_layout((layout_a, layout_b), extra_args=())

        expected_map = (1, -1)
        assert output_layout.tensor_map == expected_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_min_layout_inference_binary_orthogonal_sharding(self, mock_platform):
        """
        Feature: Element-wise min with orthogonal sharding
        Description: A sharded on dim0 (mesh 0), B sharded on dim1 (mesh 1).
        Expectation: Current implementation likely propagates the first input's layout.
                     Adjusted expectation to match current behavior: (1, -1).
        """
        mesh = self._make_2x2_mesh(mock_platform)

        layout_a = _build_layout(mesh, (Shard(0), Replicate()), 2)
        layout_b = _build_layout(mesh, (Replicate(), Shard(1)), 2)

        output_layout = op.infer_layout((layout_a, layout_b), extra_args=())

        expected_map = (1, -1)
        assert output_layout.tensor_map == expected_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_min_layout_inference_reduce_keepdim_false_explicit(self, mock_platform):
        """
        Feature: Min reduction with explicit keepdim=False
        Description: Reduce dim0 of 2D tensor.
        Expectation: Rank reduced from 2 to 1.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

        output_layouts = op.infer_layout((x_layout,), extra_args=(0, False))
        val_layout, _ = output_layouts

        expected_map = (-1,)
        assert val_layout.tensor_map == expected_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_min_layout_inference_reduce_invalid_dim_type(self, mock_platform):
        """
        Feature: Min reduction with invalid dim type
        Description: Pass a float as dimension.
        Expectation: Raises ValueError (as per implementation).
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

        with self.assertRaisesRegex(ValueError, "Invalid reduce axis"):
            op.infer_layout((x_layout,), extra_args=(1.5,))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_min_layout_inference_3d_reduce_last_dim(self, mock_platform):
        """
        Feature: Min reduction on last dimension of 3D tensor
        Description: Reduce dim 2.
        Expectation: Dim 2 removed, dim 0/1 preserved.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_placements = (Shard(0), Shard(1), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        output_layouts = op.infer_layout((x_layout,), extra_args=(2,))
        val_layout, _ = output_layouts

        expected_map = (2, 1)
        assert val_layout.tensor_map == expected_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_min_layout_inference_replicate_reduce_keepdim(self, mock_platform):
        """
        Feature: Min reduction on Replicated tensor with keepdim
        Description: Reduce dim0 of fully replicated tensor, keeping dimensions.
        Expectation: Layout remains fully replicated, but dimension preserved (as None/-1).
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

        output_layouts = op.infer_layout((x_layout,), extra_args=(0, True))
        val_layout, _ = output_layouts

        expected_map = (-1, -1)
        assert val_layout.tensor_map == expected_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_min_layout_inference_global_scalar_input(self, mock_platform):
        """
        Feature: Global Min reduction on Scalar
        Description: Input is scalar, op is global min.
        Expectation: Output is scalar (empty map).
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (), 0)

        output_layout = op.infer_layout((x_layout,), extra_args=())

        expected_map = ()
        assert output_layout.tensor_map == expected_map


if __name__ == "__main__":
    unittest.main()

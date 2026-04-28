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
"""parallel_linear test"""
import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.dtensor.dtensor import _build_layout, _LAYOUT_CACHE
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_matmul import LinearDistributedOp
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS
op = LinearDistributedOp("Linear")


class TestParallelLinear(unittest.TestCase):
    """Test Parallel Linear Distributed Operator."""

    def setUp(self) -> None:
        """Clear global caches before each test to ensure isolation."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def tearDown(self) -> None:
        """Restore global cache state after each test."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def _make_2x4_mesh(self, mock_platform):
        """Mock a 2x4 device mesh."""
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 8
        mock_platform.tensor_to_numpy.side_effect = (
            lambda t: t.numpy() if hasattr(t, "numpy") else np.array(t)
        )
        mock_platform.platform_type = MagicMock()
        return init_device_mesh(
            device_type="cpu",
            mesh_shape=(2, 4),
            mesh_dim_names=("dp", "mp"),
            init_backend=False
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_linear_layout_data_parallel(self, mock_platform):
        """
        Feature: LinearDistributedOp layout inference with data parallel sharding.
        Description: Input x is sharded on batch dim (Shard(0)), weight is fully replicated.
        Expectation: Output layout inherits batch sharding; get_expand_impl returns None.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
        w_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        cache_values = [x_layout, w_layout, None]
        output_layouts, extra_info = op.infer_layout(cache_values)
        output_layout = output_layouts[0]
        expected_map = (1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"Data Parallel test failed. Expected {expected_map},"
            f" got {output_layout.tensor_map}"
        )
        assert extra_info is None, f"extra_info should be None, got {extra_info}"
        # LinearDistributedOp overrides get_expand_impl; no contract dim sharding here → None.
        assert op.get_expand_impl(None, (output_layouts, None), cache_values) is None, (
            f"get_expand_impl should be None when contract dim is not sharded, "
            f"got {op.get_expand_impl(None, (output_layouts, None), cache_values)}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_linear_layout_hybrid_parallel(self, mock_platform):
        """
        Feature: LinearDistributedOp layout inference with hybrid parallel (DP + TP on output dim).
        Description: x sharded on dim 0, weight sharded on output dim (Shard(0)), bias matches.
        Expectation: Output is sharded on both dims; get_expand_impl returns None (no contract sharding).
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
        w_layout = _build_layout(mesh, (Replicate(), Shard(0)), 2)
        bias_layout = _build_layout(mesh, (Replicate(), Shard(0)), 1)
        cache_values = [x_layout, w_layout, bias_layout]
        output_layouts, extra_info = op.infer_layout(cache_values)
        output_layout = output_layouts[0]
        expected_map = (1, 0)
        assert output_layout.tensor_map == expected_map, (
            f"Hybrid Parallel test failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )
        assert extra_info is None, f"extra_info should be None, got {extra_info}"
        # Output dim sharded but contract dim not sharded → no bias scaling needed.
        assert op.get_expand_impl(None, (output_layouts, None), cache_values) is None, (
            f"get_expand_impl should be None when contract dim is not sharded, "
            f"got {op.get_expand_impl(None, (output_layouts, None), cache_values)}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_linear_layout_hybrid_tensor_parallel(self, mock_platform):
        """
        Feature: LinearDistributedOp layout inference with tensor parallel on the contract dim.
        Description: Both x and weight are sharded on the contracting dimension (in_features); no bias.
        Expectation: Output inherits x's batch sharding; partial sum set; get_expand_impl returns None.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)
        w_layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)
        cache_values = [x_layout, w_layout, None]
        output_layouts, extra_info = op.infer_layout(cache_values)
        output_layout = output_layouts[0]
        expected_map = (1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"Hybrid Tensor Parallel test failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )
        assert extra_info is None, f"extra_info should be None, got {extra_info}"
        # Contract dim is sharded but bias is None → no scaling closure needed.
        assert op.get_expand_impl(None, (output_layouts, None), cache_values) is None, (
            f"get_expand_impl should be None when bias is absent, "
            f"got {op.get_expand_impl(None, (output_layouts, None), cache_values)}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_linear_layout_hybrid_tensor_parallel_with_bias_mismatch(self, mock_platform):
        """
        Feature: LinearDistributedOp error handling for mismatched bias sharding.
        Description: Contract dim is sharded; bias is sharded on a different dim than the weight output dim.
        Expectation: ValueError raised with message about bias output dim sharding mismatch.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)
        w_layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)
        bias_layout = _build_layout(mesh, (Shard(0),), 1)
        cache_values = [x_layout, w_layout, bias_layout]
        with self.assertRaisesRegex(ValueError, "bias output dim sharding must match"):
            _ = op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_linear_layout_partial_with_sharded_contract_dim(self, mock_platform):
        """
        Feature: LinearDistributedOp sets Partial status when contracting dimension is sharded.
        Description: x and weight are both sharded on in_features (contract dim); no bias.
        Expectation: Output layout carries partial='sum' on the mp axis; get_expand_impl returns None.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)
        w_layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)
        cache_values = [x_layout, w_layout, None]
        output_layouts, extra_info = op.infer_layout(cache_values)
        output_layout = output_layouts[0]
        expected_partial = [None, 'sum']
        assert output_layout.partial == expected_partial, (
            f"Partial status test failed. Expected {expected_partial}, "
            f"got {output_layout.partial}"
        )
        assert extra_info is None, f"extra_info should be None, got {extra_info}"
        # Bias is None → no expand impl even though contract dim is sharded.
        assert op.get_expand_impl(None, (output_layouts, None), cache_values) is None, (
            f"get_expand_impl should be None when bias is absent, "
            f"got {op.get_expand_impl(None, (output_layouts, None), cache_values)}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_linear_layout_partial_without_sharded_contract_dim(self, mock_platform):
        """
        Feature: LinearDistributedOp produces no Partial status when contract dim is replicated.
        Description: x is sharded on batch dim only; contract dim (in_features) is fully replicated.
        Expectation: Output partial is all None; get_expand_impl returns None.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
        w_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        cache_values = [x_layout, w_layout, None]
        output_layouts, extra_info = op.infer_layout(cache_values)
        output_layout = output_layouts[0]
        expected_partial = [None, None]
        assert output_layout.partial == expected_partial, (
            f"No-partial status test failed. Expected {expected_partial}, "
            f"got {output_layout.partial}"
        )
        assert extra_info is None, f"extra_info should be None, got {extra_info}"
        assert op.get_expand_impl(None, (output_layouts, None), cache_values) is None, (
            f"get_expand_impl should be None when contract dim is not sharded, "
            f"got {op.get_expand_impl(None, (output_layouts, None), cache_values)}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_linear_get_expand_impl_with_bias_and_sharded_contract_dim(self, mock_platform):
        """
        Feature: LinearDistributedOp get_expand_impl returns callable when bias scaling is needed.
        Description: Contract dim is sharded and a replicated bias DTensor is provided.
        Expectation: get_expand_impl returns a callable that pre-scales bias by the shard factor.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)
        w_layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)
        bias_layout = _build_layout(mesh, (Replicate(), Replicate()), 1)
        cache_values = [x_layout, w_layout, bias_layout]
        output_layouts, _ = op.infer_layout(cache_values)
        impl = op.get_expand_impl(None, (output_layouts, None), cache_values)
        assert callable(impl), (
            f"get_expand_impl should return callable when contract dim is sharded "
            f"and bias is present, got {type(impl)}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_linear_partial_input_raises_error(self, mock_platform):
        """
        Feature: LinearDistributedOp rejects partial-status inputs.
        Description: Input x has Partial status set on mp axis; this must be reduced before Linear.
        Expectation: ValueError is raised with message about Partial status.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
        x_layout.set_partial_by_dev_axis("mp", "sum")
        w_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        cache_values = [x_layout, w_layout, None]
        with self.assertRaisesRegex(ValueError, "Partial status"):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_linear_weight_not_2d_raises_error(self, mock_platform):
        """
        Feature: LinearDistributedOp validates weight dimensionality.
        Description: Weight tensor is 3D, which is not supported by Linear.
        Expectation: ValueError is raised with message about weight being 2D.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate()), 3)
        w_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)
        cache_values = [x_layout, w_layout, None]
        with self.assertRaisesRegex(ValueError, "weight should be 2D"):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_linear_mismatched_mesh_raises_error(self, mock_platform):
        """
        Feature: LinearDistributedOp validates that x and weight share the same mesh.
        Description: x uses a 2x2 mesh while weight uses a 2x4 mesh.
        Expectation: ValueError is raised with message about same mesh_shape requirement.
        """
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 4
        mock_platform.tensor_to_numpy.side_effect = (
            lambda t: t.numpy() if hasattr(t, "numpy") else np.array(t)
        )
        mock_platform.platform_type = MagicMock()
        mesh_2x2 = init_device_mesh(device_type="cpu", mesh_shape=(2, 2),
                                    mesh_dim_names=("dp", "mp"), init_backend=False)
        mock_platform.get_world_size.return_value = 8
        mesh_2x4 = init_device_mesh(device_type="cpu", mesh_shape=(2, 4),
                                    mesh_dim_names=("dp2", "mp2"), init_backend=False)
        x_layout = _build_layout(mesh_2x2, (Shard(0), Replicate()), 2)
        w_layout = _build_layout(mesh_2x4, (Replicate(), Replicate()), 2)
        cache_values = [x_layout, w_layout, None]
        with self.assertRaisesRegex(ValueError, "same mesh_shape"):
            op.infer_layout(cache_values)


if __name__ == "__main__":
    unittest.main()

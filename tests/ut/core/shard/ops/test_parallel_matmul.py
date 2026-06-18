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
"""parallel_matmul test"""
import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from hyper_parallel.core.dtensor.dtensor import _build_layout, _LAYOUT_CACHE
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_matmul import (
    MatMulDistributedOp,
    MatMulExtDistributedOp,
    LinearDistributedOp,
    BatchMatMulDistributedOp,
    BatchMatMulExtDistributedOp,
)
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

op = MatMulDistributedOp("MatMul")
linear_op = LinearDistributedOp("Linear")


class TestParallelMatMul(unittest.TestCase):
    """Unit tests for MatMulDistributedOp."""
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

    def _make_2x4_mesh(self, mock_platform):
        """Set up mock and return a standard 2x4 (dp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "mp"))

    def _make_2x2x2_mesh(self, mock_platform):
        """Set up mock and return a standard 2x2x2 (dp, tp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2, 2), mesh_dim_names=("dp", "tp", "mp"))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_matmul_layout_data_parallel(self, mock_platform):
        """
        Feature: MatMul data parallel
        Description: Data parallel scenario
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
        w_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

        output_layout = op.infer_layout((x_layout, w_layout), (False, True))
        expected_map = (1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"Data Parallel with transpose_a test failed. Expected {expected_map},"
            f" got {output_layout.tensor_map}"
        )

        # Since `get_expand_impl` is not overridden, it returns None by default.
        # The same applies to other test classes, so it is unnecessary to test its return value.
        assert op.get_expand_impl(None, output_layout, (x_layout, w_layout), (False, True)) is None, (
            f"get_expand_impl test failed. Expected None, "
            f"got {op.get_expand_impl(None, output_layout, (x_layout, w_layout), (False, True))}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_matmul_layout_hybrid_parallel(self, mock_platform):
        """
        Feature: MatMul hybrid parallel
        Description: Hybrid parallel scenario
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
        w_layout = _build_layout(mesh, (Replicate(), Shard(0)), 2)

        output_layout = op.infer_layout((x_layout, w_layout), (False, True))
        expected_map = (1, 0)
        assert output_layout.tensor_map == expected_map, (
            f"Hybrid Parallel test failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_matmul_layout_tensor_parallel(self, mock_platform):
        """
        Feature: MatMul tensor parallel
        Description: Tensor parallel scenario
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Shard(0)), 2)
        w_layout = _build_layout(mesh, (Replicate(), Shard(0)), 2)

        output_layout = op.infer_layout((x_layout, w_layout), (True, False))
        expected_map = (-1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"Tensor Parallel test failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_matmul_layout_hybrid_tensor_parallel(self, mock_platform):
        """
        Feature: MatMul hybrid tensor parallel
        Description: Hybrid tensor parallel scenario
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(1), Shard(0)), 2)
        w_layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)

        output_layout = op.infer_layout((x_layout, w_layout), (True, True))
        expected_map = (1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"Hybrid Tensor Parallel test failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_matmul_layout_multi_shard_tensor_parallel(self, mock_platform):
        """
        Feature: MatMul multi shard tensor parallel
        Description: Multi shard tensor tensor parallel scenario
        Expectation: Success
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate(), Shard(1)), 2)
        w_layout = _build_layout(mesh, (Replicate(), Shard(0), Shard(1)), 2)

        output_layout = op.infer_layout((x_layout, w_layout), (False, True))
        expected_map = (2, 1)
        assert output_layout.tensor_map == expected_map, (
            f"Multi-Shard Tensor Parallel test failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )

    def _make_matmul_ext(self):
        """Create a MatMulExtDistributedOp instance."""
        return MatMulExtDistributedOp("MatMulExt")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_matmul_ext_partial_x_propagated(self, mock_platform):
        """
        Feature: MatMulExtDistributedOp propagates Partial from input x.
        Description: x has Partial on tp axis, w is Replicate. Output should inherit Partial.
        Expectation: output_layout.partial == [None, 'sum'] on mp axis.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        x_layout.set_partial_by_dev_axis("mp", "sum")
        _LAYOUT_CACHE.clear()
        w_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

        op_ext = self._make_matmul_ext()
        output_layout = op_ext.infer_layout((x_layout, w_layout))
        expected_partial = [None, 'sum']
        assert output_layout.partial == expected_partial, (
            f"Partial propagation from x failed. Expected {expected_partial}, "
            f"got {output_layout.partial}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_matmul_ext_partial_w_propagated(self, mock_platform):
        """
        Feature: MatMulExtDistributedOp propagates Partial from input w.
        Description: w has Partial on dp axis, x is Replicate. Output should inherit Partial.
        Expectation: output_layout.partial == ['sum', None] on dp axis.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        _LAYOUT_CACHE.clear()
        w_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        w_layout.set_partial_by_dev_axis("dp", "sum")

        op_ext = self._make_matmul_ext()
        output_layout = op_ext.infer_layout((x_layout, w_layout))
        expected_partial = ['sum', None]
        assert output_layout.partial == expected_partial, (
            f"Partial propagation from w failed. Expected {expected_partial}, "
            f"got {output_layout.partial}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_matmul_ext_partial_both_different_axes(self, mock_platform):
        """
        Feature: MatMulExtDistributedOp propagates Partial from both inputs on different axes.
        Description: x has Partial on mp, w has Partial on dp. Output should have both.
        Expectation: output_layout.partial == ['sum', 'sum'].
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        x_layout.set_partial_by_dev_axis("mp", "sum")
        _LAYOUT_CACHE.clear()
        w_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        w_layout.set_partial_by_dev_axis("dp", "sum")

        op_ext = self._make_matmul_ext()
        output_layout = op_ext.infer_layout((x_layout, w_layout))
        expected_partial = ['sum', 'sum']
        assert output_layout.partial == expected_partial, (
            f"Partial propagation from both axes failed. Expected {expected_partial}, "
            f"got {output_layout.partial}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_matmul_ext_partial_both_same_axis_different_op_raises(self, mock_platform):
        """
        Feature: MatMulExtDistributedOp rejects both inputs Partial on same axis with different ops.
        Description: x has Partial('sum') on mp, w has Partial('avg') on mp. Should raise.
        Expectation: ValueError raised.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        x_layout.set_partial_by_dev_axis("mp", "sum")
        _LAYOUT_CACHE.clear()
        w_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        w_layout.set_partial_by_dev_axis("mp", "avg")

        op_ext = self._make_matmul_ext()
        with self.assertRaisesRegex(ValueError, "Partial on the same axis"):
            op_ext.infer_layout((x_layout, w_layout))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_matmul_ext_partial_x_plus_contract_dim_same_axis_not_conflict(self, mock_platform):
        """
        Feature: MatMulExtDistributedOp with x Partial and contract dim sharded on same axis.
        Description: x has Partial on mp (same axis as contracting dim). This is a valid
                     scenario where x was output of a prior matmul that also had contract
                     dim sharded on mp. Both set the same Partial op, no conflict.
        Expectation: output_layout.partial == [None, 'sum'].
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        x_layout.set_partial_by_dev_axis("mp", "sum")
        _LAYOUT_CACHE.clear()
        w_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

        op_ext = self._make_matmul_ext()
        output_layout = op_ext.infer_layout((x_layout, w_layout))
        expected_partial = [None, 'sum']
        assert output_layout.partial == expected_partial, (
            f"Partial + contract dim same axis failed. Expected {expected_partial}, "
            f"got {output_layout.partial}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_matmul_ext_partial_x_plus_contract_dim_different_axes(self, mock_platform):
        """
        Feature: MatMulExtDistributedOp with x Partial on dp and contract dim sharded on mp.
        Description: Both axes independently produce Partial.
        Expectation: output_layout.partial == ['sum', 'sum'].
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)
        x_layout.set_partial_by_dev_axis("dp", "sum")
        _LAYOUT_CACHE.clear()
        w_layout = _build_layout(mesh, (Replicate(), Shard(0)), 2)

        op_ext = self._make_matmul_ext()
        output_layout = op_ext.infer_layout((x_layout, w_layout))
        expected_partial = ['sum', 'sum']
        assert output_layout.partial == expected_partial, (
            f"Partial + contract dim different axes failed. Expected {expected_partial}, "
            f"got {output_layout.partial}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_matmul_ext_partial_without_contract_sharding(self, mock_platform):
        """
        Feature: MatMulExtDistributedOp with Partial x but contract dim not sharded.
        Description: x has Partial on mp, contract dim is Replicate.
                     Only input Partial should appear.
        Expectation: output_layout.partial == [None, 'sum'] and no extra Partial from contract.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        x_layout.set_partial_by_dev_axis("mp", "sum")
        _LAYOUT_CACHE.clear()
        w_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

        op_ext = self._make_matmul_ext()
        output_layout = op_ext.infer_layout((x_layout, w_layout))
        expected_partial = [None, 'sum']
        assert output_layout.partial == expected_partial, (
            f"Partial without contract sharding failed. Expected {expected_partial}, "
            f"got {output_layout.partial}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_matmul_ext_partial_x_shard_w_same_axis_output_dim_raises(self, mock_platform):
        """
        Feature: MatMulExtDistributedOp rejects Partial x + Shard w on same axis in output dim.
        Description: x has Partial on 'mp', w is Shard on 'mp' in the output dimension.
                     Layout.set_partial_by_dev_axis should raise "Partial dim must be replicate".
        Expectation: ValueError raised with message about Partial dim must be replicate.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        x_layout.set_partial_by_dev_axis("mp", "sum")
        _LAYOUT_CACHE.clear()
        w_layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)

        op_ext = self._make_matmul_ext()
        with self.assertRaisesRegex(ValueError, "Partial dim must be replicate"):
            op_ext.infer_layout((x_layout, w_layout))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_matmul_distributed_partial_x_propagated(self, mock_platform):
        """
        Feature: MatMulDistributedOp propagates Partial from input x with transpose.
        Description: x has Partial on dp, w is Replicate, transpose_a=True.
                     Input Partial should propagate regardless of transpose.
        Expectation: output_layout.partial == ['sum', None].
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        x_layout.set_partial_by_dev_axis("dp", "sum")
        _LAYOUT_CACHE.clear()
        w_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

        output_layout = op.infer_layout((x_layout, w_layout), (True, False))
        expected_partial = ['sum', None]
        assert output_layout.partial == expected_partial, (
            f"MatMulDistributedOp partial x propagation failed. Expected {expected_partial}, "
            f"got {output_layout.partial}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_matmul_distributed_partial_w_propagated(self, mock_platform):
        """
        Feature: MatMulDistributedOp propagates Partial from input w.
        Description: w has Partial on mp, x is Replicate, transpose_b=True.
        Expectation: output_layout.partial == [None, 'sum'].
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        _LAYOUT_CACHE.clear()
        w_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        w_layout.set_partial_by_dev_axis("mp", "sum")

        output_layout = op.infer_layout((x_layout, w_layout), (False, True))
        expected_partial = [None, 'sum']
        assert output_layout.partial == expected_partial, (
            f"MatMulDistributedOp partial w propagation failed. Expected {expected_partial}, "
            f"got {output_layout.partial}"
        )


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
        output_layouts, extra_info = linear_op.infer_layout(cache_values)
        output_layout = output_layouts[0]
        expected_map = (1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"Data Parallel test failed. Expected {expected_map},"
            f" got {output_layout.tensor_map}"
        )
        assert extra_info is None, f"extra_info should be None, got {extra_info}"
        # LinearDistributedOp overrides get_expand_impl; no contract dim sharding here → None.
        assert linear_op.get_expand_impl(None, (output_layouts, None), cache_values) is None, (
            f"get_expand_impl should be None when contract dim is not sharded, "
            f"got {linear_op.get_expand_impl(None, (output_layouts, None), cache_values)}"
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
        output_layouts, extra_info = linear_op.infer_layout(cache_values)
        output_layout = output_layouts[0]
        expected_map = (1, 0)
        assert output_layout.tensor_map == expected_map, (
            f"Hybrid Parallel test failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )
        assert extra_info is None, f"extra_info should be None, got {extra_info}"
        # Output dim sharded but contract dim not sharded → no bias scaling needed.
        assert linear_op.get_expand_impl(None, (output_layouts, None), cache_values) is None, (
            f"get_expand_impl should be None when contract dim is not sharded, "
            f"got {linear_op.get_expand_impl(None, (output_layouts, None), cache_values)}"
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
        output_layouts, extra_info = linear_op.infer_layout(cache_values)
        output_layout = output_layouts[0]
        expected_map = (1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"Hybrid Tensor Parallel test failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )
        assert extra_info is None, f"extra_info should be None, got {extra_info}"
        # Contract dim is sharded but bias is None → no scaling closure needed.
        assert linear_op.get_expand_impl(None, (output_layouts, None), cache_values) is None, (
            f"get_expand_impl should be None when bias is absent, "
            f"got {linear_op.get_expand_impl(None, (output_layouts, None), cache_values)}"
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
            _ = linear_op.infer_layout(cache_values)

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
        output_layouts, extra_info = linear_op.infer_layout(cache_values)
        output_layout = output_layouts[0]
        expected_partial = [None, 'sum']
        assert output_layout.partial == expected_partial, (
            f"Partial status test failed. Expected {expected_partial}, "
            f"got {output_layout.partial}"
        )
        assert extra_info is None, f"extra_info should be None, got {extra_info}"
        # Bias is None → no expand impl even though contract dim is sharded.
        assert linear_op.get_expand_impl(None, (output_layouts, None), cache_values) is None, (
            f"get_expand_impl should be None when bias is absent, "
            f"got {linear_op.get_expand_impl(None, (output_layouts, None), cache_values)}"
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
        output_layouts, extra_info = linear_op.infer_layout(cache_values)
        output_layout = output_layouts[0]
        expected_partial = [None, None]
        assert output_layout.partial == expected_partial, (
            f"No-partial status test failed. Expected {expected_partial}, "
            f"got {output_layout.partial}"
        )
        assert extra_info is None, f"extra_info should be None, got {extra_info}"
        assert linear_op.get_expand_impl(None, (output_layouts, None), cache_values) is None, (
            f"get_expand_impl should be None when contract dim is not sharded, "
            f"got {linear_op.get_expand_impl(None, (output_layouts, None), cache_values)}"
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
        output_layouts, _ = linear_op.infer_layout(cache_values)
        impl = linear_op.get_expand_impl(None, (output_layouts, None), cache_values)
        assert callable(impl), (
            f"get_expand_impl should return callable when contract dim is sharded "
            f"and bias is present, got {type(impl)}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_linear_partial_input_propagated(self, mock_platform):
        """
        Feature: LinearDistributedOp propagates Partial from input x.
        Description: Input x has Partial status set on mp axis; it should be
                     propagated to the output like other matmul operators.
        Expectation: output_layout.partial == [None, 'sum'].
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
        x_layout.set_partial_by_dev_axis("mp", "sum")
        _LAYOUT_CACHE.clear()
        w_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        cache_values = [x_layout, w_layout, None]
        output_layouts, extra_info = linear_op.infer_layout(cache_values)
        output_layout = output_layouts[0]
        expected_partial = [None, 'sum']
        assert output_layout.partial == expected_partial, (
            f"Linear partial input propagation failed. Expected {expected_partial}, "
            f"got {output_layout.partial}"
        )

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
            linear_op.infer_layout(cache_values)

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
            linear_op.infer_layout(cache_values)


class TestParallelBatchMatMul(unittest.TestCase):
    """Unit tests for BatchMatMulDistributedOp."""
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

    def _make_2x2x2_mesh(self, mock_platform):
        """Set up mock and return a standard 2x2x2 (dp, cp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2, 2), mesh_dim_names=("dp", "cp", "mp"))

    def _run_scenario(self, bmm_op, x_layout, w_layout, expected_map, transpose_a=False, transpose_b=False):
        """Infer layout of BatchMatMul"""
        output_layout = bmm_op.infer_layout((x_layout, w_layout), (transpose_a, transpose_b))
        assert output_layout.tensor_map == expected_map, (
            f"Test BatchMatMul failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )
        assert bmm_op.get_expand_impl(
            None, output_layout, (x_layout, w_layout), (transpose_a, transpose_b)
        ) is None, (
            f"get_expand_impl test failed. Expected None, "
            f"got {bmm_op.get_expand_impl(None, output_layout, (x_layout, w_layout), (transpose_a, transpose_b))}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_bmm_tensor_parallel(self, mock_platform):
        """
        Feature: Tensor parallel in python shard.
        Description: Test tensor parallel in python shard.
        Expectation: Run success.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate(), Shard(2)), 3)
        w_layout = _build_layout(mesh, (Replicate(), Replicate(), Shard(1)), 3)

        bmm_op = BatchMatMulDistributedOp("BatchMatMul")
        self._run_scenario(bmm_op, x_layout, w_layout, expected_map=(2, -1, -1))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_bmm_transpose_tensor_parallel(self, mock_platform):
        """
        Feature: Tensor parallel in python shard.
        Description: Test tensor parallel in python shard, transpose=True.
        Expectation: Run success.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate(), Shard(1)), 3)
        w_layout = _build_layout(mesh, (Replicate(), Replicate(), Shard(1)), 3)

        bmm_op = BatchMatMulDistributedOp("BatchMatMul")
        self._run_scenario(
            bmm_op, x_layout, w_layout,
            expected_map=(2, -1, -1),
            transpose_a=True,
            transpose_b=False
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_bmm_partial_x_propagated(self, mock_platform):
        """
        Feature: BatchMatMulDistributedOp propagates Partial from input x.
        Description: x has Partial on cp axis, both inputs are Replicate.
                     Output should inherit Partial.
        Expectation: output_layout.partial == [None, 'sum', None].
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)
        x_layout.set_partial_by_dev_axis("cp", "sum")
        _LAYOUT_CACHE.clear()
        w_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)

        bmm_op = BatchMatMulDistributedOp("BatchMatMul")
        output_layout = bmm_op.infer_layout((x_layout, w_layout), (False, False))
        expected_partial = [None, 'sum', None]
        assert output_layout.partial == expected_partial, (
            f"BatchMatMul partial x propagation failed. Expected {expected_partial}, "
            f"got {output_layout.partial}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_bmm_partial_x_plus_contract_dim_sharded(self, mock_platform):
        """
        Feature: BatchMatMulDistributedOp with x Partial on cp and contract dim sharded on mp.
        Description: x has Partial on cp (different axis from contract dim sharding on mp).
                     Output should inherit both.
        Expectation: output_layout.partial == [None, 'sum', 'sum'].
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate(), Shard(2)), 3)
        x_layout.set_partial_by_dev_axis("cp", "sum")
        _LAYOUT_CACHE.clear()
        w_layout = _build_layout(mesh, (Replicate(), Replicate(), Shard(1)), 3)

        bmm_op = BatchMatMulDistributedOp("BatchMatMul")
        output_layout = bmm_op.infer_layout((x_layout, w_layout), (False, False))
        expected_partial = [None, 'sum', 'sum']
        assert output_layout.partial == expected_partial, (
            f"BatchMatMul partial x + contract sharding failed. Expected {expected_partial}, "
            f"got {output_layout.partial}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_bmm_partial_x_transpose_a_propagated(self, mock_platform):
        """
        Feature: BatchMatMulDistributedOp with transpose propagates Partial from x.
        Description: x has Partial on cp, transpose_a=True. Partial should propagate.
        Expectation: output_layout.partial == [None, 'sum', None].
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)
        x_layout.set_partial_by_dev_axis("cp", "sum")
        w_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)

        bmm_op = BatchMatMulDistributedOp("BatchMatMul")
        output_layout = bmm_op.infer_layout((x_layout, w_layout), (True, False))
        expected_partial = [None, 'sum', None]
        assert output_layout.partial == expected_partial, (
            f"BatchMatMul partial x transpose_a failed. Expected {expected_partial}, "
            f"got {output_layout.partial}"
        )


class TestParallelBatchMatMulExt(unittest.TestCase):
    """Unit tests for BatchMatMulExtDistributedOp."""
    def setUp(self):
        """Set up test fixtures before each test method."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def tearDown(self):
        """Clean up after each test method."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def _setup_mock_platform(self, mock_platform, platform_type=None, world_size=8):
        """Configure common mock-platform attributes used across tests."""
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

    def _run_scenario(self, x_layout, w_layout, expected_map):
        """Infer layout of BatchMatmul operator"""
        bmm_ext_op = BatchMatMulExtDistributedOp("BatchMatMulExt")
        output_layout = bmm_ext_op.infer_layout((x_layout, w_layout), None)
        assert output_layout.tensor_map == expected_map, (
            f"BatchMatMulExt failed. Expected {expected_map}, got {output_layout.tensor_map}"
        )
        assert bmm_ext_op.get_expand_impl(None, output_layout, (x_layout, w_layout), None) is None, (
            f"get_expand_impl test failed. Expected None, "
            f"got {bmm_ext_op.get_expand_impl(None, output_layout, (x_layout, w_layout), None)}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_bmm_ext_data_parallel(self, mock_platform):
        """
        Feature: Data parallel in python shard.
        Description: Test data parallel in python shard.
        Expectation: Run success.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate()), 3)
        w_layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate()), 3)

        self._run_scenario(x_layout, w_layout, expected_map=(2, -1, -1))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_bmm_ext_model_parallel(self, mock_platform):
        """
        Feature: Model parallel in python shard.
        Description: Test model parallel in python shard.
        Expectation: Run success.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)
        w_layout = _build_layout(mesh, (Replicate(), Replicate(), Shard(2)), 3)

        self._run_scenario(x_layout, w_layout, expected_map=(-1, -1, 0))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_bmm_ext_hybrid_parallel(self, mock_platform):
        """
        Feature: Hybrid parallel in python shard.
        Description: Test hybrid parallel in python shard.
        Expectation: Run success.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1), Replicate()), 3)
        w_layout = _build_layout(mesh, (Shard(0), Replicate(), Shard(2)), 3)

        self._run_scenario(x_layout, w_layout, expected_map=(2, 1, 0))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_bmm_ext_tensor_parallel(self, mock_platform):
        """
        Feature: Tensor parallel in python shard.
        Description: Test tensor parallel in python shard.
        Expectation: Run success.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate(), Shard(2)), 3)
        w_layout = _build_layout(mesh, (Replicate(), Replicate(), Shard(1)), 3)

        self._run_scenario(x_layout, w_layout, expected_map=(2, -1, -1))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_bmm_ext_hybrid_tensor_parallel(self, mock_platform):
        """
        Feature: Tensor parallel in python shard.
        Description: Test tensor parallel in python shard.
        Expectation: Run success.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1), Shard(2)), 3)
        w_layout = _build_layout(mesh, (Replicate(), Replicate(), Shard(1)), 3)

        self._run_scenario(x_layout, w_layout, expected_map=(2, 1, -1))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_bmm_ext_multi_shard_tensor_parallel(self, mock_platform):
        """
        Feature: Multi shard tensor parallel in python shard.
        Description: Test multi shard tensor parallel in python shard.
        Expectation: Run success.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate(), Shard(2)), 3)
        w_layout = _build_layout(mesh, (Replicate(), Shard(2), Shard(1)), 3)

        self._run_scenario(x_layout, w_layout, expected_map=(2, -1, 1))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_bmm_ext_partial_x_propagated(self, mock_platform):
        """
        Feature: BatchMatMulExtDistributedOp propagates Partial from input x.
        Description: x has Partial on cp axis, w is Replicate. Output should inherit Partial.
        Expectation: output_layout.partial == [None, 'sum', None] on cp axis.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)
        x_layout.set_partial_by_dev_axis("cp", "sum")
        w_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)

        bmm_ext_op = BatchMatMulExtDistributedOp("BatchMatMulExt")
        output_layout = bmm_ext_op.infer_layout((x_layout, w_layout), None)
        expected_partial = [None, 'sum', None]
        assert output_layout.partial == expected_partial, (
            f"BatchMatMulExt partial x propagation failed. Expected {expected_partial}, "
            f"got {output_layout.partial}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_bmm_ext_partial_w_propagated(self, mock_platform):
        """
        Feature: BatchMatMulExtDistributedOp propagates Partial from input w.
        Description: w has Partial on dp axis, x is Replicate. Output should inherit Partial.
        Expectation: output_layout.partial == ['sum', None, None].
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)
        w_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)
        w_layout.set_partial_by_dev_axis("dp", "sum")

        bmm_ext_op = BatchMatMulExtDistributedOp("BatchMatMulExt")
        output_layout = bmm_ext_op.infer_layout((x_layout, w_layout), None)
        expected_partial = ['sum', None, None]
        assert output_layout.partial == expected_partial, (
            f"BatchMatMulExt partial w propagation failed. Expected {expected_partial}, "
            f"got {output_layout.partial}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_bmm_ext_partial_x_plus_contract_sharding(self, mock_platform):
        """
        Feature: BatchMatMulExtDistributedOp with x Partial on cp and contract dim sharded on mp.
        Description: x has Partial on cp (different axis from contract dim sharding on mp).
                     Output should have both.
        Expectation: output_layout.partial == [None, 'sum', 'sum'].
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate(), Shard(2)), 3)
        x_layout.set_partial_by_dev_axis("cp", "sum")
        _LAYOUT_CACHE.clear()
        w_layout = _build_layout(mesh, (Replicate(), Replicate(), Shard(1)), 3)

        bmm_ext_op = BatchMatMulExtDistributedOp("BatchMatMulExt")
        output_layout = bmm_ext_op.infer_layout((x_layout, w_layout), None)
        expected_partial = [None, 'sum', 'sum']
        assert output_layout.partial == expected_partial, (
            f"BatchMatMulExt partial x + contract sharding failed. Expected {expected_partial}, "
            f"got {output_layout.partial}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_bmm_ext_partial_x_shard_w_same_output_axis_raises(self, mock_platform):
        """
        Feature: BatchMatMulExtDistributedOp rejects Partial x + Shard w on same axis in output dim.
        Description: x has Partial on mp, w is Shard on mp in the output column dimension.
                     set_partial_by_dev_axis should raise "Partial dim must be replicate".
        Expectation: ValueError raised.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)
        x_layout.set_partial_by_dev_axis("mp", "sum")
        _LAYOUT_CACHE.clear()
        w_layout = _build_layout(mesh, (Replicate(), Replicate(), Shard(2)), 3)

        bmm_ext_op = BatchMatMulExtDistributedOp("BatchMatMulExt")
        with self.assertRaisesRegex(ValueError, "Partial dim must be replicate"):
            bmm_ext_op.infer_layout((x_layout, w_layout), None)


if __name__ == "__main__":
    unittest.main()

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
"""Unit tests for MindSpore fully_shard packing helpers."""

import os
import unittest
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mindspore")

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"
from tests.ut.platform.mindspore._ensure_mindspore_platform import (  # noqa: E402
    ensure_mindspore_platform_for_fully_shard,
)

ensure_mindspore_platform_for_fully_shard()

import mindspore as ms

from hyper_parallel.core.dtensor.placement_types import Shard, StridedShard
from hyper_parallel.core.fully_shard.utils import FSDPMeshInfo
from hyper_parallel.platform.mindspore.fully_shard.pack_utils import (
    ReduceScatterPlan,
    build_rs_plan,
    pack_for_reduce_scatter,
    supports_same_dim_strided_layout,
    unpack_from_all_gather,
)


class TestMindSporePackUtils(unittest.TestCase):
    """Cover the V1 same-dim StridedShard packing helpers."""

    def test_build_rs_plan_rejects_invalid_world_size(self):
        """world_size must be positive."""
        local_tensor = ms.Tensor(np.ones((4, 4), dtype=np.float32))

        with self.assertRaisesRegex(ValueError, "world_size must be positive"):
            build_rs_plan(None, local_tensor, world_size=0, shard_dim=0)

    def test_build_rs_plan_requires_shard_dim_without_param(self):
        """A plain tensor path needs an explicit shard dimension."""
        local_tensor = ms.Tensor(np.ones((4, 4), dtype=np.float32))

        with self.assertRaisesRegex(ValueError, "requires either hsdp_param or shard_dim"):
            build_rs_plan(None, local_tensor, world_size=2)

    def test_build_rs_plan_rejects_invalid_shard_dim(self):
        """The shard dimension must index the unpacked tensor shape."""
        local_tensor = ms.Tensor(np.ones((4, 4), dtype=np.float32))

        with self.assertRaisesRegex(ValueError, "Invalid shard dim"):
            build_rs_plan(None, local_tensor, world_size=2, shard_dim=2)

    def test_build_rs_plan_rejects_scalar_gradients(self):
        """Scalar gradients have no valid shard dimension."""
        local_tensor = ms.Tensor(np.array(1.0, dtype=np.float32))

        with self.assertRaisesRegex(ValueError, "Invalid shard dim"):
            build_rs_plan(None, local_tensor, world_size=2, shard_dim=0)

    def test_build_rs_plan_rejects_uneven_sharding(self):
        """The local tensor must split evenly on the fully_shard dimension."""
        local_tensor = ms.Tensor(np.ones((3, 4), dtype=np.float32))

        with self.assertRaisesRegex(NotImplementedError, "even sharding"):
            build_rs_plan(None, local_tensor, world_size=2, shard_dim=0)

    def test_build_rs_plan_world_size_one_uses_identity_layout(self):
        """world_size=1 should keep the unpacked tensor layout unchanged."""
        local_tensor = ms.Tensor(np.arange(6, dtype=np.float32).reshape(2, 3))

        plan = build_rs_plan(None, local_tensor, world_size=1, shard_dim=1)

        self.assertEqual(plan.pack_kind, "identity_dim0")
        self.assertEqual(plan.packed_shape, (1, 6))
        self.assertEqual(plan.packed_tensor_shape, (2, 3))
        self.assertEqual(plan.unpacked_shape, (2, 3))

    def test_build_rs_plan_supports_dim0_fully_shard(self):
        """Plain dim0 fully_shard should use identity packing."""
        local_tensor = ms.Tensor(np.arange(16, dtype=np.float32).reshape(4, 4))

        plan = build_rs_plan(None, local_tensor, world_size=2, shard_dim=0)

        self.assertEqual(plan.pack_kind, "identity_dim0")
        self.assertEqual(plan.packed_shape, (2, 8))
        self.assertEqual(plan.packed_tensor_shape, (4, 4))
        self.assertEqual(plan.unpacked_shape, (4, 4))

    def test_build_rs_plan_supports_non_dim0_fully_shard(self):
        """Plain non-dim0 fully_shard should use the chunk-cat packing path."""
        local_tensor = ms.Tensor(np.arange(16, dtype=np.float32).reshape(4, 4))

        plan = build_rs_plan(None, local_tensor, world_size=2, shard_dim=1)

        self.assertEqual(plan.pack_kind, "chunk_cat_non_dim0")
        self.assertEqual(plan.packed_shape, (2, 8))
        self.assertEqual(plan.packed_tensor_shape, (8, 2))
        self.assertEqual(plan.unpacked_shape, (4, 4))

    def test_pack_and_unpack_roundtrip_for_non_dim0_layout(self):
        """chunk_cat_non_dim0 pack/unpack should recover the original local tensor."""
        local_tensor = ms.Tensor(np.arange(16, dtype=np.float32).reshape(4, 4))
        plan = build_rs_plan(None, local_tensor, world_size=2, shard_dim=1)

        packed = pack_for_reduce_scatter(local_tensor, plan)
        unpacked = unpack_from_all_gather(packed.reshape(-1), plan)

        expected_packed = np.concatenate(
            np.array_split(np.arange(16, dtype=np.float32).reshape(4, 4), 2, axis=1),
            axis=0,
        )
        np.testing.assert_allclose(packed.asnumpy(), expected_packed.reshape(2, 8))
        np.testing.assert_allclose(unpacked.asnumpy(), local_tensor.asnumpy())

    def test_supports_same_dim_strided_layout(self):
        """The helper should recognize the supported V1 same-dim StridedShard subset."""
        hsdp_param = SimpleNamespace(
            mesh_info=object.__new__(FSDPMeshInfo),
            _orig_param_is_dtensor=True,
            hsdp_placement=Shard(1),
            _spmd_shard_mesh_dim=0,
            _spmd_placements=(StridedShard(1, split_factor=2), Shard(1)),
            _orig_dtensor_placements=(Shard(1),),
        )

        self.assertTrue(supports_same_dim_strided_layout(hsdp_param))

    def test_same_dim_strided_dim0_uses_identity_layout(self):
        """Supported same-dim StridedShard on dim0 has a distinct identity marker."""
        hsdp_param = SimpleNamespace(
            mesh_info=object.__new__(FSDPMeshInfo),
            _orig_param_is_dtensor=True,
            hsdp_placement=Shard(0),
            _spmd_shard_mesh_dim=0,
            _spmd_placements=(StridedShard(0, split_factor=2), Shard(0)),
            _orig_dtensor_placements=(Shard(0),),
            _orig_size=(4, 4),
        )
        local_tensor = ms.Tensor(np.ones((4, 4), dtype=np.float32))

        plan = build_rs_plan(hsdp_param, local_tensor, world_size=2)

        self.assertEqual(plan.pack_kind, "same_dim_strided_identity_dim0")
        self.assertEqual(plan.shard_dim, 0)

    def test_same_dim_strided_non_dim0_uses_chunk_cat_layout(self):
        """Supported same-dim StridedShard on non-dim0 should use chunk-cat packing."""
        hsdp_param = SimpleNamespace(
            mesh_info=object.__new__(FSDPMeshInfo),
            _orig_param_is_dtensor=True,
            hsdp_placement=Shard(1),
            _spmd_shard_mesh_dim=0,
            _spmd_placements=(StridedShard(1, split_factor=2), Shard(1)),
            _orig_dtensor_placements=(Shard(1),),
            _orig_size=(4, 4),
        )
        local_tensor = ms.Tensor(np.ones((4, 4), dtype=np.float32))

        plan = build_rs_plan(hsdp_param, local_tensor, world_size=2)

        self.assertEqual(plan.pack_kind, "chunk_cat_non_dim0")
        self.assertEqual(plan.shard_dim, 1)

    def test_build_rs_plan_rejects_unsupported_strided_layout(self):
        """Unsupported StridedShard layouts should fail fast instead of packing a wrong buffer."""
        hsdp_param = SimpleNamespace(
            mesh_info=object.__new__(FSDPMeshInfo),
            _orig_param_is_dtensor=True,
            hsdp_placement=Shard(0),
            _spmd_shard_mesh_dim=0,
            _spmd_placements=(StridedShard(0, split_factor=2), Shard(1)),
            _orig_dtensor_placements=(Shard(0),),
            _orig_size=(4, 4),
        )
        local_tensor = ms.Tensor(np.ones((4, 4), dtype=np.float32))

        with self.assertRaisesRegex(NotImplementedError, "same-dim StridedShard"):
            build_rs_plan(hsdp_param, local_tensor, world_size=2)

    def test_supports_same_dim_strided_layout_rejects_incomplete_context(self):
        """Missing DTensor/placement metadata should be reported as unsupported."""
        incomplete_params = [
            SimpleNamespace(_spmd_placements=()),
            SimpleNamespace(
                mesh_info=object(),
                _spmd_placements=(StridedShard(1, split_factor=2), Shard(1)),
            ),
            SimpleNamespace(
                mesh_info=object.__new__(FSDPMeshInfo),
                _orig_param_is_dtensor=False,
                _spmd_placements=(StridedShard(1, split_factor=2), Shard(1)),
            ),
            SimpleNamespace(
                mesh_info=object.__new__(FSDPMeshInfo),
                _orig_param_is_dtensor=True,
                hsdp_placement=Shard(1),
                _spmd_shard_mesh_dim=4,
                _spmd_placements=(StridedShard(1, split_factor=2), Shard(1)),
                _orig_dtensor_placements=(Shard(1),),
            ),
        ]

        for hsdp_param in incomplete_params:
            self.assertFalse(supports_same_dim_strided_layout(hsdp_param))

    def test_pack_and_unpack_roundtrip_for_dim0_layout(self):
        """identity_dim0 pack/unpack should recover the original local tensor."""
        local_tensor = ms.Tensor(np.arange(16, dtype=np.float32).reshape(4, 4))
        plan = build_rs_plan(None, local_tensor, world_size=2, shard_dim=0)

        packed = pack_for_reduce_scatter(local_tensor, plan)
        unpacked = unpack_from_all_gather(packed.reshape(-1), plan)

        np.testing.assert_allclose(packed.asnumpy(), local_tensor.asnumpy().reshape(2, 8))
        np.testing.assert_allclose(unpacked.asnumpy(), local_tensor.asnumpy())

    def test_pack_for_reduce_scatter_rejects_unknown_pack_kind(self):
        """Unknown pack kinds should fail before producing a misleading buffer."""
        local_tensor = ms.Tensor(np.ones((2, 2), dtype=np.float32))
        plan = ReduceScatterPlan("unknown", 0, 2, (2, 2), (2, 2), (2, 2))

        with self.assertRaisesRegex(NotImplementedError, "Unsupported reduce-scatter pack kind"):
            pack_for_reduce_scatter(local_tensor, plan)

    def test_pack_for_reduce_scatter_rejects_shape_mismatch(self):
        """The input tensor shape must match the plan's unpacked shape."""
        local_tensor = ms.Tensor(np.ones((2, 2), dtype=np.float32))
        plan = ReduceScatterPlan("identity_dim0", 0, 2, (2, 2), (4, 1), (4, 1))

        with self.assertRaisesRegex(AssertionError, "plan.unpacked_shape"):
            pack_for_reduce_scatter(local_tensor, plan)

    def test_unpack_from_all_gather_rejects_unknown_pack_kind(self):
        """Unknown unpack kinds should fail before reshaping the fused buffer."""
        full_packed = ms.Tensor(np.ones((4,), dtype=np.float32))
        plan = ReduceScatterPlan("unknown", 0, 2, (2, 2), (2, 2), (2, 2))

        with self.assertRaisesRegex(NotImplementedError, "Unsupported all-gather unpack kind"):
            unpack_from_all_gather(full_packed, plan)


if __name__ == "__main__":
    unittest.main()

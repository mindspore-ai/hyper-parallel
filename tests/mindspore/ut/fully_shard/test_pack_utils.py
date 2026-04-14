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

import mindspore as ms

from hyper_parallel.core.dtensor.placement_types import Shard, StridedShard
from hyper_parallel.platform.mindspore.fully_shard.pack_utils import (
    build_rs_plan,
    pack_for_reduce_scatter,
    supports_same_dim_strided_layout,
    unpack_from_all_gather,
)


class TestMindSporePackUtils(unittest.TestCase):
    """Cover the V1 same-dim StridedShard packing helpers."""

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
            uses_param_shard=True,
            _orig_param_is_dtensor=True,
            hsdp_placement=Shard(1),
            _spmd_shard_mesh_dim=0,
            _spmd_placements=(StridedShard(1, split_factor=2), Shard(1)),
            _orig_dtensor_placements=(Shard(1),),
        )

        self.assertTrue(supports_same_dim_strided_layout(hsdp_param))

    def test_build_rs_plan_rejects_unsupported_strided_layout(self):
        """Unsupported StridedShard layouts should fail fast instead of packing a wrong buffer."""
        hsdp_param = SimpleNamespace(
            uses_param_shard=True,
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


if __name__ == "__main__":
    unittest.main()

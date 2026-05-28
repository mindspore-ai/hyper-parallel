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
"""Unit tests for torch fully_shard packing helpers."""
# pylint: disable=protected-access

import os
import unittest
from types import SimpleNamespace

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

# pylint: disable=wrong-import-position
import torch

from hyper_parallel.core.dtensor.placement_types import Shard, StridedShard
from hyper_parallel.platform.torch.fully_shard.pack_utils import (
    ReduceScatterPlan,
    build_rs_plan,
    pack_for_reduce_scatter,
    supports_same_dim_strided_layout,
    unpack_from_all_gather,
)


def _same_dim_param(target_dim):
    return SimpleNamespace(
        uses_param_shard=True,
        _orig_param_is_dtensor=True,
        hsdp_placement=Shard(target_dim),
        _spmd_shard_mesh_dim=0,
        _spmd_placements=(StridedShard(target_dim, split_factor=2), Shard(target_dim)),
        _orig_dtensor_placements=(Shard(target_dim),),
        _orig_size=(4, 4),
    )


class TestTorchPackUtils(unittest.TestCase):
    """Cover reduce-scatter packing plans and roundtrips on CPU tensors."""

    def test_build_rs_plan_rejects_invalid_inputs(self):
        """Invalid world-size and shard-dim inputs should fail early."""
        tensor = torch.ones(4, 4)

        cases = [
            ({"world_size": 0, "shard_dim": 0}, ValueError, "world_size must be positive"),
            ({"world_size": 2}, ValueError, "requires either hsdp_param or shard_dim"),
            ({"world_size": 2, "shard_dim": 2}, ValueError, "Invalid shard dim"),
        ]
        for kwargs, exc_type, msg in cases:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(exc_type, msg):
                build_rs_plan(None, tensor, **kwargs)

    def test_build_rs_plan_rejects_scalar_uneven_and_noncontiguous(self):
        """Reduce-scatter plan creation should reject scalar, uneven, and noncontiguous inputs."""
        with self.assertRaisesRegex(ValueError, "Invalid shard dim"):
            build_rs_plan(None, torch.tensor(1.0), world_size=2, shard_dim=0)
        with self.assertRaisesRegex(NotImplementedError, "even sharding"):
            build_rs_plan(None, torch.ones(3, 4), world_size=2, shard_dim=0)
        with self.assertRaisesRegex(NotImplementedError, "contiguous"):
            build_rs_plan(None, torch.ones(4, 4).t(), world_size=2, shard_dim=0)

    def test_build_rs_plan_identity_and_non_dim0(self):
        """Build plans should distinguish identity and non-dim0 packing."""
        tensor = torch.arange(16, dtype=torch.float32).reshape(4, 4)

        world_one = build_rs_plan(None, tensor, world_size=1, shard_dim=1)
        dim0 = build_rs_plan(None, tensor, world_size=2, shard_dim=0)
        non_dim0 = build_rs_plan(None, tensor, world_size=2, shard_dim=1)

        self.assertEqual(world_one.pack_kind, "identity_dim0")
        self.assertEqual(world_one.packed_shape, torch.Size((1, 16)))
        self.assertEqual(dim0.pack_kind, "identity_dim0")
        self.assertEqual(non_dim0.pack_kind, "chunk_cat_non_dim0")
        self.assertEqual(non_dim0.packed_tensor_shape, torch.Size((8, 2)))

    def test_same_dim_strided_layouts(self):
        """Same-dim strided layouts should select identity or non-dim0 plans."""
        dim0 = build_rs_plan(_same_dim_param(0), torch.ones(4, 4), world_size=2)
        non_dim0 = build_rs_plan(_same_dim_param(1), torch.ones(4, 4), world_size=2)

        self.assertEqual(dim0.pack_kind, "same_dim_strided_identity_dim0")
        self.assertEqual(non_dim0.pack_kind, "chunk_cat_non_dim0")
        self.assertTrue(supports_same_dim_strided_layout(_same_dim_param(1)))

    def test_same_dim_strided_layout_rejects_incomplete_or_unsupported_context(self):
        """Unsupported same-dim strided layouts should be rejected."""
        incomplete = [
            SimpleNamespace(_spmd_placements=()),
            SimpleNamespace(
                uses_param_shard=False,
                _spmd_placements=(StridedShard(1, split_factor=2), Shard(1)),
            ),
            SimpleNamespace(
                uses_param_shard=True,
                _orig_param_is_dtensor=False,
                _spmd_placements=(StridedShard(1, split_factor=2), Shard(1)),
            ),
            SimpleNamespace(
                uses_param_shard=True,
                _orig_param_is_dtensor=True,
                hsdp_placement=Shard(1),
                _spmd_shard_mesh_dim=0,
                _spmd_placements=(StridedShard(1, split_factor=1), Shard(1)),
                _orig_dtensor_placements=(Shard(1),),
            ),
        ]

        for hsdp_param in incomplete:
            self.assertFalse(supports_same_dim_strided_layout(hsdp_param))

        unsupported = _same_dim_param(1)
        unsupported._orig_dtensor_placements = (Shard(0),)
        with self.assertRaisesRegex(NotImplementedError, "same-dim StridedShard"):
            build_rs_plan(unsupported, torch.ones(4, 4), world_size=2)

    def test_pack_and_unpack_roundtrips(self):
        """Pack and unpack should roundtrip for dim0 and non-dim0 shard axes."""
        tensor = torch.arange(16, dtype=torch.float32).reshape(4, 4)

        for shard_dim in (0, 1):
            with self.subTest(shard_dim=shard_dim):
                plan = build_rs_plan(None, tensor, world_size=2, shard_dim=shard_dim)
                packed = pack_for_reduce_scatter(tensor, plan)
                unpacked = unpack_from_all_gather(packed.reshape(-1), plan)
                torch.testing.assert_close(unpacked, tensor)

    def test_pack_and_unpack_reject_invalid_plan(self):
        """Pack and unpack helpers should reject unsupported or mismatched plans."""
        tensor = torch.ones(2, 2)
        bad_plan = ReduceScatterPlan("bad", 0, 2, torch.Size((2, 2)), torch.Size((2, 2)), torch.Size((2, 2)))
        shape_mismatch = ReduceScatterPlan(
            "identity_dim0", 0, 2, torch.Size((2, 2)), torch.Size((4, 1)), torch.Size((4, 1))
        )

        with self.assertRaisesRegex(NotImplementedError, "Unsupported reduce-scatter"):
            pack_for_reduce_scatter(tensor, bad_plan)
        with self.assertRaisesRegex(AssertionError, "plan.unpacked_shape"):
            pack_for_reduce_scatter(tensor, shape_mismatch)
        with self.assertRaisesRegex(NotImplementedError, "Unsupported all-gather"):
            unpack_from_all_gather(torch.ones(4), bad_plan)


if __name__ == "__main__":
    unittest.main()

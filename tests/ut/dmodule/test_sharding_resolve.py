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
"""Unit tests for ShardingConfig and resolve_placements."""

import unittest
from unittest.mock import MagicMock

from tests.ut.dmodule._ensure_torch_dmodule import ensure_torch_platform_for_dmodule

ensure_torch_platform_for_dmodule()

from hyper_parallel.core.dtensor.placement_types import Partial, Replicate, Shard
from hyper_parallel.dmodule.sharding import LocalMapConfig, ShardingConfig, resolve_placements
from hyper_parallel.dmodule.types import MeshAxisName


class TestResolvePlacements(unittest.TestCase):
    """resolve_placements axis order and validation (hyper semantics)."""

    def test_missing_axis_defaults_replicate(self):
        named = {MeshAxisName.TP: Shard(0)}
        out = resolve_placements(named, ("dp", "tp"))
        self.assertTrue(out[0].is_replicate())
        self.assertTrue(out[1].is_shard(0))

    def test_extra_axis_raises(self):
        named = {MeshAxisName.TP: Shard(0), MeshAxisName.CP: Replicate()}
        with self.assertRaises(ValueError) as ctx:
            resolve_placements(named, ("tp",))
        self.assertIn("not in mesh", str(ctx.exception))

    def test_output_order_follows_mesh(self):
        named = {MeshAxisName.TP: Shard(1), MeshAxisName.DP: Replicate()}
        out = resolve_placements(named, ("dp", "tp"))
        self.assertTrue(out[0].is_replicate())
        self.assertTrue(out[1].is_shard(1))

    def test_string_axis_keys_accepted(self):
        named = {"tp": Shard(0)}
        out = resolve_placements(named, ("tp",))
        self.assertTrue(out[0].is_shard(0))

    def test_case_mismatch_named_raises(self):
        """Axis keys are matched case-sensitively against mesh_dim_names."""
        with self.assertRaises(ValueError) as ctx:
            resolve_placements({"TP": Shard(0)}, ("dp", "tp"))
        self.assertIn("not in mesh", str(ctx.exception))
        self.assertIn("TP", str(ctx.exception))

    def test_case_mismatch_mesh_raises(self):
        with self.assertRaises(ValueError) as ctx:
            resolve_placements({MeshAxisName.TP: Shard(0)}, ("DP", "TP"))
        self.assertIn("not in mesh", str(ctx.exception))

    def test_empty_mesh_returns_empty_list(self):
        self.assertEqual(resolve_placements({}, ()), [])

    def test_named_on_empty_mesh_raises(self):
        with self.assertRaises(ValueError) as ctx:
            resolve_placements({MeshAxisName.TP: Shard(0)}, ())
        self.assertIn("not in mesh", str(ctx.exception))

    def test_partial_placement(self):
        named = {MeshAxisName.TP: Partial("sum")}
        out = resolve_placements(named, ("tp",))
        self.assertTrue(out[0].is_partial("sum"))

    def test_mixed_replicate_shard_partial(self):
        named = {
            MeshAxisName.DP: Replicate(),
            MeshAxisName.TP: Shard(-1),
            MeshAxisName.CP: Partial("sum"),
        }
        out = resolve_placements(named, ("dp", "tp", "cp"))
        self.assertTrue(out[0].is_replicate())
        self.assertTrue(out[1].is_shard(-1))
        self.assertTrue(out[2].is_partial("sum"))


class TestShardingConfig(unittest.TestCase):
    """ShardingConfig dataclass helpers."""

    def test_defaults_empty_state_shardings(self):
        config = ShardingConfig()
        self.assertEqual(config.state_shardings, {})
        self.assertIsNone(config.in_dst_shardings)
        self.assertIsNone(config.local_map)

    def test_to_dict_serializable(self):
        config = ShardingConfig(
            state_shardings={"weight": {MeshAxisName.TP: Shard(0)}},
        )
        payload = config.to_dict()
        self.assertIn("repr", payload)

    def test_local_map_config_to_dict(self):
        local_map = LocalMapConfig(in_grad_placements=({MeshAxisName.TP: Replicate()},))
        self.assertIn("repr", local_map.to_dict())


class TestResolvePlacementsPartialNamed(unittest.TestCase):
    """Missing mesh axes in NamedPlacement default to Replicate."""

    def test_partial_named_placements_allowed(self):
        """Omitted axes are filled with Replicate in mesh axis order."""
        mesh = MagicMock()
        mesh.mesh_dim_names = ("dp", "tp")
        named = {MeshAxisName.TP: Shard(0)}
        placements = resolve_placements(named, mesh.mesh_dim_names)
        self.assertEqual(len(placements), 2)
        self.assertTrue(placements[0].is_replicate())


if __name__ == "__main__":
    unittest.main()

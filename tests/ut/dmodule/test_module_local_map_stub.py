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
"""Unit tests for local_map stub in Module.parallelize."""

import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock

from tests.ut.dmodule._ensure_torch_dmodule import ensure_torch_platform_for_dmodule

ensure_torch_platform_for_dmodule()

from hyper_parallel.dmodule.module import Module
from hyper_parallel.core.dtensor.placement_types import Replicate
from hyper_parallel.dmodule.sharding import LocalMapConfig, ShardingConfig
from hyper_parallel.dmodule.types import MeshAxisName


class TestModuleLocalMapStub(unittest.TestCase):
    """local_map must raise NotImplementedError until M9."""

    @dataclass(kw_only=True, slots=True)
    class ToyConfig(Module.Config):
        pass

    class Toy(Module):
        def __init__(self, config: "TestModuleLocalMapStub.ToyConfig"):
            super().__init__()
            self.config = config
            self._sharding_config = ShardingConfig(
                local_map=LocalMapConfig(
                    in_grad_placements=({MeshAxisName.TP: Replicate()},),
                ),
            )

        def forward(self, x):
            return x

    def test_parallelize_raises_for_local_map(self):
        mod = self.Toy(self.ToyConfig())
        mesh = MagicMock()
        mesh.mesh_dim_names = ("tp",)
        with self.assertRaises(NotImplementedError) as ctx:
            mod.parallelize(mesh)
        self.assertIn("M9", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

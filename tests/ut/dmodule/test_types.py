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
"""Unit tests for protocol types (MeshAxisName)."""

import unittest

from tests.ut.dmodule._ensure_torch_dmodule import ensure_torch_platform_for_dmodule

ensure_torch_platform_for_dmodule()

from hyper_parallel.dmodule.types import MeshAxisName, StrEnum


class TestMeshAxisName(unittest.TestCase):
    """Mesh axis names align with hyper ParallelDims mesh naming."""

    def test_values_match_hyper_mesh_names(self):
        self.assertEqual(MeshAxisName.DP.value, "dp")
        self.assertEqual(MeshAxisName.DP_REPLICATE.value, "dp_replicate")
        self.assertEqual(MeshAxisName.DP_SHARD.value, "dp_shard")
        self.assertEqual(MeshAxisName.FSDP.value, "fsdp")
        self.assertEqual(MeshAxisName.TP.value, "tp")
        self.assertEqual(MeshAxisName.CP.value, "cp")
        self.assertEqual(MeshAxisName.PP.value, "pp")
        self.assertEqual(MeshAxisName.EP.value, "ep")
        self.assertEqual(MeshAxisName.EFSDP.value, "efsdp")

    def test_str_enum_coercion(self):
        self.assertEqual(MeshAxisName("tp"), MeshAxisName.TP)
        self.assertTrue(issubclass(MeshAxisName, StrEnum))


if __name__ == "__main__":
    unittest.main()

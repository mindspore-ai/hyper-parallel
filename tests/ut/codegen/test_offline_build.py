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
"""Unit tests for process-group-free Codegen planning topology."""

import unittest
from types import SimpleNamespace

from hyper_parallel.codegen.offline_build import (
    PlanningTopology,
    create_offline_distributed_setup,
)


class TestOfflineBuild(unittest.TestCase):
    """Validate logical topology projection without distributed setup."""

    @staticmethod
    def _config(*, tp_size, cp_size, ep_size, dp_shard_size, edp_shard_size):
        return SimpleNamespace(
            accelerator=SimpleNamespace(
                tp_size=tp_size,
                cp_size=cp_size,
                pp_size=1,
                ep_size=ep_size,
                sequence_parallel=tp_size > 1,
                loss_parallel=cp_size > 1,
            ),
            fsdp_config=SimpleNamespace(
                dp_shard_size=dp_shard_size,
                edp_shard_size=edp_shard_size,
            ),
            training=SimpleNamespace(low_precision=None),
            optimizer=SimpleNamespace(fp32_main_params=True),
        )

    def test_main_model_topologies_satisfy_yaml_constraints(self):
        """Qwen3-MoE and DeepSeek-V4.1 produce pure-data planning meshes."""
        cases = (
            (self._config(tp_size=2, cp_size=2, ep_size=2, dp_shard_size=2, edp_shard_size=2),
             (1, 2, 2)),
            (self._config(tp_size=1, cp_size=1, ep_size=8, dp_shard_size=8, edp_shard_size=1),
             (8, 1, 1)),
        )

        for config, expected_shape in cases:
            with self.subTest(expected_shape=expected_shape):
                setup = create_offline_distributed_setup(config)
                topology = setup.mesh_context.device_mesh

                self.assertIsInstance(topology, PlanningTopology)
                self.assertEqual(topology.mesh_dim_names, ("dp", "cp", "tp"))
                self.assertEqual(topology.mesh_shape, expected_shape)
                self.assertEqual(topology.shape, expected_shape)
                self.assertIsNone(setup.strategy_config)

    def test_virtual_ep_divides_logical_dense_domain(self):
        """The projected dense domain always covers the configured EP group."""
        config = self._config(
            tp_size=1,
            cp_size=1,
            ep_size=16,
            dp_shard_size=2,
            edp_shard_size=1,
        )

        setup = create_offline_distributed_setup(config)
        dense_size = setup.mesh_context.dp_size * setup.mesh_context.cp_size * setup.mesh_context.tp_size

        self.assertEqual(dense_size, 16)
        self.assertEqual(dense_size % setup.mesh_context.ep_size, 0)


if __name__ == "__main__":
    unittest.main()

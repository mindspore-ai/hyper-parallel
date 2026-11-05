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
from unittest import mock

from hyper_parallel.codegen.build import create_offline_distributed_setup
from hyper_parallel.distributed.mesh import DistributedSetup, LogicalDeviceMesh
from hyper_parallel.trainer.runtime import distributed as runtime_distributed


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

                self.assertIsInstance(topology, LogicalDeviceMesh)
                self.assertEqual(topology.mesh_dim_names, ("dp", "cp", "tp"))
                self.assertEqual(topology.mesh_shape, expected_shape)
                self.assertEqual(topology.shape, expected_shape)
                self.assertEqual(
                    topology.size(),
                    expected_shape[0] * expected_shape[1] * expected_shape[2],
                )
                self.assertIs(setup.strategy_config, config.fsdp_config)

    def test_model_build_policies_match_runtime_setup(self):
        """Both entry points assemble the same non-runtime policy objects."""
        config = self._config(
            tp_size=1, cp_size=1, ep_size=1, dp_shard_size=1, edp_shard_size=1,
        )
        config.training.low_precision = SimpleNamespace(enabled=True)
        offline = create_offline_distributed_setup(config)
        with mock.patch.object(runtime_distributed.dist, "is_initialized", return_value=True), \
                mock.patch.object(runtime_distributed.dist, "get_world_size", return_value=1), \
                mock.patch.object(runtime_distributed, "get_device_type", return_value="cpu"), \
                mock.patch.object(runtime_distributed.MeshContext, "build_meshs"), \
                mock.patch.object(runtime_distributed, "_populate_mesh_ranks"):
            runtime = runtime_distributed.create_distributed_setup_from_config(config)
        offline_fields = vars(offline).copy()
        runtime_fields = vars(runtime).copy()
        offline_fields.pop("mesh_context")
        runtime_fields.pop("mesh_context")
        self.assertEqual(offline_fields, runtime_fields)
        self.assertEqual(runtime.mesh_context.tp_size, offline.mesh_context.tp_size)
        self.assertIs(offline.low_precision_config, config.training.low_precision)
        self.assertTrue(offline.fp32_main_params)

    def test_runtime_without_process_group_retains_disabled_strategy(self):
        """Shared assembly preserves the Trainer's uninitialized behavior."""
        config = self._config(
            tp_size=1, cp_size=1, ep_size=1, dp_shard_size=1, edp_shard_size=1,
        )
        offline = create_offline_distributed_setup(config)
        runtime = DistributedSetup.from_config(
            config, offline.mesh_context, strategy_enabled=False,
        )
        self.assertIsNone(runtime.strategy_config)
        self.assertIsNone(runtime.plan_overrides)
        self.assertIsNone(runtime.module_replacements)

    def test_runtime_without_accelerator_preserves_empty_setup(self):
        """Missing accelerator keeps the native early-return contract."""
        setup = runtime_distributed.create_distributed_setup_from_config(None)
        self.assertEqual(setup, DistributedSetup())

    def test_nonpositive_degrees_preserve_trainer_compatibility(self):
        """Offline projection keeps the existing minimum-degree semantics."""
        for degree in (0, -1):
            with self.subTest(degree=degree):
                config = self._config(
                    tp_size=degree, cp_size=degree, ep_size=degree,
                    dp_shard_size=degree, edp_shard_size=degree,
                )
                setup = create_offline_distributed_setup(config)
                self.assertEqual(setup.mesh_context.device_mesh.shape, (1, 1, 1))

    def test_planning_topology_supports_native_preflight_submeshes(self):
        """Native CP preflight can select a process-group-free logical axis."""
        topology = LogicalDeviceMesh(
            mesh_dim_names=("dp", "cp", "tp"),
            mesh_shape=(2, 4, 8),
        )

        cp_mesh = topology["cp"]
        active_mesh = topology[("cp", "tp")]

        self.assertEqual(cp_mesh.mesh_dim_names, ("cp",))
        self.assertEqual(cp_mesh.shape, (4,))
        self.assertEqual(cp_mesh.size(), 4)
        self.assertEqual(active_mesh.mesh_dim_names, ("cp", "tp"))
        self.assertEqual(active_mesh.shape, (4, 8))
        self.assertEqual(active_mesh.size("tp"), 8)

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

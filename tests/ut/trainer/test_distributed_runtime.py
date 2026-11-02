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
"""Unit tests for trainer distributed-runtime lifecycle helpers."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from hyper_parallel.core.pipeline_parallel._p2p import _P2P_MULTI_STREAM_GROUPS
from hyper_parallel.distributed.mesh import MeshContext
from hyper_parallel.models.build_options import CompileConfig, FSDP2Config, FSDP2MixedPrecisionConfig
from hyper_parallel.trainer.config.parallelism import AcceleratorConfig
from hyper_parallel.trainer.runtime import distributed as runtime
from hyper_parallel.trainer.runtime.distributed import destroy_process_group
from tests.common.mark_utils import arg_mark


class TestDestroyProcessGroup(unittest.TestCase):
    """Verify process-group teardown clears caches from their current owners."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    @patch("hyper_parallel.trainer.runtime.distributed.dist.is_initialized", return_value=False)
    def test_clears_pipeline_p2p_group_cache(self, mock_is_initialized) -> None:
        """Teardown imports and clears the cache from core pipeline parallel."""
        del mock_is_initialized
        _P2P_MULTI_STREAM_GROUPS[("send", 0, 1)] = object()

        destroy_process_group()

        self.assertEqual(_P2P_MULTI_STREAM_GROUPS, {})


class TestSingleRankMixedPrecision(unittest.TestCase):
    """Explicit precision policies must survive the single-rank setup boundary."""

    @staticmethod
    def _config(**precision):
        policy = FSDP2MixedPrecisionConfig(**precision)
        return SimpleNamespace(
            accelerator=AcceleratorConfig(),
            fsdp_config=FSDP2Config(mix_precision=policy),
            optimizer=SimpleNamespace(fp32_main_params=True),
            compile=CompileConfig(),
        )

    @staticmethod
    def _setup(config, *, initialized=True, world_size=1, device="npu", opt_in=True):
        with patch.object(runtime.dist, "is_initialized", return_value=initialized), \
             patch.object(runtime.dist, "get_world_size", return_value=world_size), \
             patch.object(runtime, "get_device_type", return_value=device), \
             patch.object(runtime, "_build_device_mesh_from_accelerator", return_value=(MeshContext(), None)):
            return runtime.create_distributed_setup_from_config(config, allow_single_rank_precision=opt_in)

    def test_default_single_rank_keeps_strategy_disabled(self):
        """Default cast flags alone do not request a policy."""
        for precision in ({}, {"cast_forward_inputs": False}):
            with self.subTest(precision=precision):
                self.assertIsNone(self._setup(self._config(**precision)).strategy_config)

    def test_explicit_single_rank_policy_reaches_model_builder(self):
        """Each public dtype field survives the opt-in Trainer boundary."""
        for precision in (
            {"param_dtype": "bfloat16"}, {"reduce_dtype": "float32"}, {"output_dtype": "bfloat16"},
        ):
            config = self._config(**precision)
            with self.subTest(precision=precision):
                setup = self._setup(config)
                self.assertIs(setup.strategy_config, config.fsdp_config)
                self.assertTrue(setup.fp32_main_params)

    def test_existing_callers_keep_single_rank_strategy_disabled(self):
        """Programmatic dtype defaults do not opt a shared runtime into size-1 FSDP."""
        config = self._config(param_dtype="bfloat16", reduce_dtype="float32")
        for device in ("cpu", "cuda", "npu"):
            with self.subTest(device=device):
                self.assertIsNone(self._setup(config, device=device, opt_in=False).strategy_config)

    def test_existing_multi_rank_and_uninitialized_behavior(self):
        """The change does not infer a strategy without a process group."""
        config = self._config(param_dtype="bfloat16")
        self.assertIsNone(self._setup(config, initialized=False).strategy_config)
        for world_size in (2, 4):
            with self.subTest(world_size=world_size):
                self.assertIs(self._setup(config, world_size=world_size).strategy_config, config.fsdp_config)

    def test_unsupported_single_rank_device_and_graph_fail_clearly(self):
        """Do not silently drop an explicit policy at a later unsupported boundary."""
        config = self._config(param_dtype="bfloat16")
        with self.assertRaisesRegex(ValueError, "CUDA or NPU"):
            self._setup(config, device="cpu")
        config.compile = CompileConfig(enabled=True, use_joint_graph=True, backend="torchair")
        with self.assertRaisesRegex(ValueError, "eager"):
            self._setup(config)


if __name__ == "__main__":
    unittest.main()

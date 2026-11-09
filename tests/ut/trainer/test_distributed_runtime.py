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
from hyper_parallel.models.build_options import FSDP2Config, FSDP2MixedPrecisionConfig
from hyper_parallel.trainer.config.parallelism import AcceleratorConfig
from hyper_parallel.trainer.runtime import distributed as runtime
from hyper_parallel.trainer.runtime.distributed import destroy_process_group


class TestDestroyProcessGroup(unittest.TestCase):
    """Verify process-group teardown clears caches from their current owners."""

    @patch("hyper_parallel.trainer.runtime.distributed.dist.is_initialized", return_value=False)
    def test_clears_pipeline_p2p_group_cache(self, mock_is_initialized) -> None:
        """Teardown imports and clears the cache from core pipeline parallel."""
        del mock_is_initialized
        _P2P_MULTI_STREAM_GROUPS[("send", 0, 1)] = object()

        destroy_process_group()

        self.assertEqual(_P2P_MULTI_STREAM_GROUPS, {})


class TestFSDPSetup(unittest.TestCase):
    """Verify FSDP configuration reaches the model builder at every DP size."""

    def test_strategy_depends_on_initialization_instead_of_parallel_size(self) -> None:
        """Feature: Size-independent Trainer FSDP setup.
        Description: Configure default, global and custom-only precision in single-rank, DP and TP domains.
        Expectation: Initialized runtimes retain the configured strategy; uninitialized runtimes do not enable it.
        """
        for precision in (
            {},
            {"param_dtype": "bfloat16", "reduce_dtype": "float32"},
            {"custom_params": {"*.weight": {"param_dtype": "float32"}}},
        ):
            for world_size, tp_size, shard_size, initialized in (
                (1, 1, 1, False), (1, 1, 1, True), (2, 2, 1, True), (2, 1, 1, True), (2, 1, 2, True),
            ):
                with self.subTest(precision=precision, topology=(world_size, tp_size, shard_size, initialized)):
                    config = SimpleNamespace(
                        accelerator=AcceleratorConfig(tp_size=tp_size),
                        fsdp_config=FSDP2Config(
                            dp_shard_size=shard_size, mix_precision=FSDP2MixedPrecisionConfig(**precision)
                        ),
                        optimizer=SimpleNamespace(fp32_main_params=True),
                    )
                    with patch.object(runtime.dist, "is_initialized", return_value=initialized), \
                            patch.object(runtime.dist, "get_world_size", return_value=world_size), \
                            patch.object(MeshContext, "build_meshs"):
                        setup = runtime.create_distributed_setup_from_config(config)

                    self.assertIs(setup.strategy_config, config.fsdp_config if initialized else None)

    def test_non_positive_parallel_degree_keeps_compatibility_clamp(self) -> None:
        """Shared topology preserves the Trainer's historical minimum of one."""
        config = SimpleNamespace(
            accelerator=AcceleratorConfig(tp_size=0),
            fsdp_config=FSDP2Config(),
            optimizer=SimpleNamespace(fp32_main_params=False),
        )

        setup = runtime.create_distributed_setup_from_config(config)

        self.assertEqual(setup.mesh_context.tp_size, 1)


if __name__ == "__main__":
    unittest.main()

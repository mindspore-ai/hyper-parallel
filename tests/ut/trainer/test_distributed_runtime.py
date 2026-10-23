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
from hyper_parallel.models._transformers.pipeline import PipelineConfig
from hyper_parallel.models.build_options import FSDP2Config
from hyper_parallel.trainer.config.parallelism import AcceleratorConfig
from hyper_parallel.trainer.runtime.distributed import create_distributed_setup_from_config, destroy_process_group
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


class TestPipelineDistributedSetup(unittest.TestCase):
    """Pipeline batches use DP degree and never overwrite the public micro-batch size."""

    @staticmethod
    def _config(global_batch_size=8, microbatch_size=2, enabled=True, pp_size=2):
        """Build a configuration without constructing any process groups."""
        return SimpleNamespace(
            accelerator=AcceleratorConfig(pp_size=pp_size, tp_size=2, cp_size=2),
            fsdp_config=FSDP2Config(dp_shard_size=2),
            pipeline=PipelineConfig(enabled=enabled, pp_microbatch_size=microbatch_size),
            training=SimpleNamespace(global_batch_size=global_batch_size, micro_batch_size=1),
            optimizer=SimpleNamespace(fp32_main_params=False),
        )

    @patch("hyper_parallel.trainer.runtime.distributed._populate_mesh_ranks")
    @patch("hyper_parallel.trainer.runtime.distributed._build_device_mesh_from_accelerator")
    @patch("hyper_parallel.trainer.runtime.distributed.dist")
    def test_dp_local_batch_and_independent_pipeline_microbatch(self, mock_dist, mock_build, mock_populate):
        """World=16, PP=2, TP=2, CP=2 gives DP=2 and a local batch of four."""
        mock_dist.is_initialized.return_value = True
        mock_dist.get_world_size.return_value = 16
        mock_build.return_value = (SimpleNamespace(), ())
        cfg = self._config()

        setup = create_distributed_setup_from_config(cfg)

        self.assertEqual(setup.pipeline_batch_size, 4)
        self.assertEqual(setup.pipeline_config.pp_microbatch_size, 2)
        self.assertEqual(cfg.training.micro_batch_size, 1)
        mock_populate.assert_called_once()
        self.assertEqual(mock_build.call_args.args[2], 2)

    @patch("hyper_parallel.trainer.runtime.distributed.dist")
    def test_rejects_inconsistent_enablement_and_batch_divisibility(self, mock_dist):
        """Invalid combinations fail before any mesh is built."""
        mock_dist.is_initialized.return_value = True
        mock_dist.get_world_size.return_value = 16
        cases = [
            ({"enabled": False}, "pipeline.enabled"),
            ({"enabled": True, "pp_size": 1}, "pipeline.enabled"),
            ({"global_batch_size": 7}, "divisible by dp_size"),
            ({"microbatch_size": 3}, "divisible by pipeline.pp_microbatch_size"),
            ({"global_batch_size": 0}, "must be positive"),
        ]
        for values, message in cases:
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, message):
                create_distributed_setup_from_config(self._config(**values))


if __name__ == "__main__":
    unittest.main()

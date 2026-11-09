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
"""Trainer DataLoader balancing configuration and serialization contracts."""

import unittest
from types import SimpleNamespace

import torch
import yaml
from torch.utils.data import default_collate

from hyper_parallel.data.batching import FixedBatchDataLoader, build_dataloader
from hyper_parallel.distributed_data import DistributedDataLoader, WorkloadCost
from hyper_parallel.trainer.config.data import DataLoaderConfig
from hyper_parallel.trainer.config.resolver import ConfigResolutionError, replace_override_path, resolve_component


def _resolve(suffix: str) -> DataLoaderConfig:
    """Resolve a real YAML loader section without building a Dataset."""
    node = yaml.safe_load("_target_: hyper_parallel.data.batching.FixedBatchDataLoader\n" + suffix)
    return resolve_component(node, annotation=DataLoaderConfig, path="$.dataloader")


class TestDataLoaderBalanceConfig(unittest.TestCase):
    """Keep optional balance blocks separate from native DataLoader arguments."""

    def test_yaml_presence_and_round_trip(self) -> None:
        """Feature: Presence-based load balancing.
        Description: Resolve omitted, empty and populated YAML blocks, then serialize and reload.
        Expectation: Empty blocks enable balancing and omitted blocks stay absent on reload.
        """
        for suffix, expected in (
            ("", None), ("load_balance: {}", {}), ("load_balance:", {}),
            ("load_balance:\n  balance_window_size: 3\n  repack_sequences: true\n  pad_token_id: 151645",
             {"balance_window_size": 3, "repack_sequences": True, "pad_token_id": 151645}),
        ):
            with self.subTest(suffix=suffix):
                config = _resolve(suffix)
                self.assertEqual(config.load_balance, expected)
                self.assertNotIn("load_balance", config.target.to_dict())
                serialized = config.to_dict()
                self.assertEqual("load_balance" in serialized, expected is not None)
                restored = resolve_component(
                    yaml.safe_load(yaml.safe_dump(serialized)), annotation=DataLoaderConfig, path="$.dataloader",
                )
                self.assertEqual(restored.load_balance, expected)

    def test_scalar_balance_blocks_are_rejected(self) -> None:
        """Feature: Balancing block validation.
        Description: Use obsolete boolean switches and other scalar values.
        Expectation: Parsing reports the load_balance path instead of silently disabling balancing.
        """
        for value in ("true", "false", "1", "yes", "[]"):
            with self.subTest(value=value), self.assertRaisesRegex(ConfigResolutionError, "dataloader.load_balance"):
                _resolve(f"load_balance: {value}")

    def test_balance_options_support_config_overrides(self) -> None:
        """Feature: DataLoader balancing overrides.
        Description: Add a balance block or update its existing window option through the CLI resolver.
        Expectation: Overrides preserve the original config and appear in the saved loader block.
        """
        original = _resolve("")
        enabled = replace_override_path(
            original, ["load_balance"], {"balance_window_size": 3}, path="dataloader",
        )
        updated = replace_override_path(
            enabled, ["load_balance", "balance_window_size"], 5, path="dataloader",
        )
        self.assertIsNone(original.load_balance)
        self.assertEqual(enabled.load_balance, {"balance_window_size": 3})
        self.assertEqual(updated.to_dict()["load_balance"], {"balance_window_size": 5})

    def test_empty_balance_block_selects_collective_loader(self) -> None:
        """Feature: YAML to DataLoader integration.
        Description: Consume identical samples with an omitted or empty balance block.
        Expectation: Both preserve data while only the collective loader owns sampler progress.
        """
        records = [{"input_ids": torch.full((4,), index), "labels": torch.full((4,), index + 1)}
                   for index in range(4)]
        context = SimpleNamespace(
            dp_rank=0, dp_size=1, pp_size=1,
            device_mesh=SimpleNamespace(mesh_shape=(1,), mesh_dim_names=("dp",), rank_list=(0,)),
        )
        for suffix in ("", "load_balance: {}"):
            with self.subTest(suffix=suffix):
                config = _resolve(suffix)
                loaders, samplers = build_dataloader(
                    config, datasets=(records, None, None), collate_fn=default_collate, mesh_context=context,
                    training_config=SimpleNamespace(micro_batch_size=1, global_batch_size=2, seed=7),
                    data_config={"seq_length": 4}, cost_model=lambda item: WorkloadCost(llm=item.pack_tokens),
                )
                loader = loaders[0]
                try:
                    self.assertIsInstance(loader, DistributedDataLoader if suffix else FixedBatchDataLoader)
                    self.assertEqual(samplers[0] is None, bool(suffix))
                    self.assertEqual([batch["input_ids"][0, 0].item() for batch in loader], list(range(4)))
                finally:
                    if isinstance(loader, DistributedDataLoader):
                        loader.close()

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
"""Dataset-owned metadata, collation, field placement and ordinary iteration."""

import unittest
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hyper_parallel.distributed_data import (
    DistributedDatasetConfig,
    SampleMetadata,
    WorkloadCost,
    build_distributed_dataloader,
    build_distributed_dataset,
)
from hyper_parallel.distributed_data.dataset_dataloader import DatasetDataLoader
from tests.common.mark_utils import arg_mark


def _mesh() -> SimpleNamespace:
    return SimpleNamespace(mesh_shape=(1,), mesh_dim_names=("dp",), rank_list=(0,))


def _samples() -> list[dict]:
    return [
        {"input_ids": torch.tensor([index, index + 1]),
         "metadata": SampleMetadata(2, cost=WorkloadCost(llm=index + 1), features={"images": index})}
        for index in range(4)
    ]


def _collate(samples: list[dict]) -> dict:
    return {"input_ids": torch.cat([sample["input_ids"] for sample in samples])}


class TestDistributedDataset(unittest.TestCase):
    """Test the new facade without accelerator hardware or process groups."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_embedded_metadata_is_removed_only_for_collation(self) -> None:
        """Feature: Dataset-owned metadata.
        Description: Collate samples containing embedded metadata.
        Expectation: Collators see payload fields without mutating source mappings.
        """
        samples = _samples()
        collate = Mock(wraps=_collate)
        dataset = build_distributed_dataset([[samples]], metadata="metadata", collate_fn=collate)
        self.assertIs(dataset.sample_metadata(samples[0]), samples[0]["metadata"])
        packed = dataset.pack(samples, 8)
        self.assertEqual(packed["input_ids"].tolist(), [0, 1, 1, 2, 2, 3, 3, 4])
        self.assertTrue(all("metadata" not in sample for sample in collate.call_args.args[0]))
        self.assertTrue(all("metadata" in sample for sample in samples))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_callable_metadata_and_log_features(self) -> None:
        """Feature: Dataset log counters.
        Description: Read callback metadata and aggregate declared features.
        Expectation: Numeric features are summed and invalid counters are rejected.
        """
        samples = _samples()
        dataset = build_distributed_dataset(
            [], metadata=lambda sample: sample["metadata"], collate_fn=list, log_fields=("images",),
        )
        self.assertEqual(dataset.pack(samples, 8), samples)
        self.assertEqual(dataset.summarize(map(dataset.sample_metadata, samples)), {"images": 6})
        with self.assertRaisesRegex(ValueError, "must be numeric"):
            dataset.summarize([SampleMetadata(1, features={"images": "one"})])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_cpu_fields_exclude_only_declared_top_level_fields(self) -> None:
        """Feature: Dataset field placement.
        Description: Use meta tensors to model device transfer without an accelerator.
        Expectation: Only declared CPU fields remain unmoved and tensor dtypes survive.
        """
        dataset = build_distributed_dataset(
            [], metadata="metadata", collate_fn=list, cpu_fields=("offsets",),
        )
        batch = {"offsets": torch.tensor([0, 2]), "inputs": [torch.ones(2, dtype=torch.int64)]}
        moved = dataset.move_to_device(batch, torch.device("meta"))
        self.assertIs(moved["offsets"], batch["offsets"])
        self.assertEqual(moved["inputs"][0].device.type, "meta")
        self.assertEqual(moved["inputs"][0].dtype, torch.int64)
        self.assertEqual(batch["inputs"][0].device.type, "cpu")

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_enabled_balance_stops_prefetch_at_limit_and_reports_custom_cost(self) -> None:
        """Feature: Buffered local balancing.
        Description: Apply a custom cost model and an explicit one-step limit.
        Expectation: Logging uses modeled costs and prefetch does not cross the limit.
        """
        samples = _samples()
        reads = []

        def source() -> Iterator[list[list[dict]]]:
            """Record every physical source read, including speculative reads."""
            for step in range(2):
                reads.append(step)
                yield [samples[step * 2:step * 2 + 2]]

        dataset = build_distributed_dataset(source(), metadata="metadata", collate_fn=_collate, log_fields=("images",))
        config = DistributedDatasetConfig(seq_len=8, local_batch_size=1)
        with build_distributed_dataloader(
                dataset, _mesh(), config, device="cpu", max_steps=1,
                cost_model=lambda metadata: WorkloadCost(llm=metadata.cost.llm * 10),
        ) as loader:
            batch = next(loader)
            self.assertEqual(batch[0]["input_ids"].tolist(), [0, 1, 1, 2])
            stats = dict(loader.last_balance_stats)
            self.assertEqual(stats["cost_before"], (30,))
            self.assertEqual(stats["bins_after"][0][0]["images"], 1)
            self.assertEqual(loader.group_ranks, (0,))
            with self.assertRaises(StopIteration):
                next(loader)
        self.assertEqual(reads, [0])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_default_balancing_requires_backbone_dimensions(self) -> None:
        """Feature: Default cost model.
        Description: Build enabled balancing without model dimensions or a callback.
        Expectation: Startup rejects the missing configuration.
        """
        dataset = build_distributed_dataset([], metadata="metadata", collate_fn=_collate)
        with self.assertRaisesRegex(ValueError, "model_config"):
            build_distributed_dataloader(
                dataset, _mesh(), DistributedDatasetConfig(seq_len=8, local_batch_size=1),
                device="cpu",
            )

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_epoch_and_checkpoint_contract(self) -> None:
        """Feature: Dataset lifecycle.
        Description: Change epochs and request unsupported local checkpoint operations.
        Expectation: Epochs reach the source and checkpoint operations fail explicitly.
        """
        source = Mock()
        source.__iter__ = Mock(side_effect=lambda: iter([[_samples()]]))
        dataset = build_distributed_dataset(source, metadata="metadata", collate_fn=_collate)
        with build_distributed_dataloader(
                dataset, _mesh(), DistributedDatasetConfig(seq_len=8, local_batch_size=1),
                device="cpu", cost_model=lambda metadata: metadata.cost,
        ) as loader:
            first = next(loader)[0]["input_ids"]
            loader.set_epoch(3)
            source.set_epoch.assert_called_once_with(3)
            self.assertEqual(loader.step, 0)
            self.assertIsNone(loader.last_host_batch)
            torch.testing.assert_close(next(loader)[0]["input_ids"], first)
            with self.assertRaisesRegex(NotImplementedError, "checkpoint"):
                loader.state_dict()
            with self.assertRaisesRegex(NotImplementedError, "checkpoint"):
                loader.load_state_dict({})

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_background_collator_error_reaches_consumer(self) -> None:
        """Feature: Background failure propagation.
        Description: Raise from a user collator on the producer thread.
        Expectation: The waiting training thread receives the original exception.
        """
        failure = RuntimeError("collation failed")
        dataset = build_distributed_dataset(
            [[_samples()]], metadata="metadata", collate_fn=Mock(side_effect=failure),
        )
        with build_distributed_dataloader(
                dataset, _mesh(), DistributedDatasetConfig(seq_len=8, local_batch_size=1),
                device="cpu", cost_model=lambda metadata: metadata.cost,
        ) as loader:
            with self.assertRaises(RuntimeError) as caught:
                next(loader)
            self.assertIs(caught.exception, failure)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_facade_rejects_conflicting_builder_callbacks(self) -> None:
        """Feature: Dataset API ownership.
        Description: Configure collation on both the dataset and loader builder.
        Expectation: The builder rejects competing data interpretation contracts.
        """
        dataset = build_distributed_dataset([], metadata="metadata", collate_fn=_collate)
        with self.assertRaisesRegex(ValueError, "Configure.*on DistributedDataset"):
            build_distributed_dataloader(
                dataset, _mesh(), DistributedDatasetConfig(seq_len=8, local_batch_size=1),
                device="cpu", collate_fn=list,
            )

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_iteration_hands_off_each_staged_microbatch(self) -> None:
        """Feature: Ordinary device-ready iteration.
        Description: Consume a staged step from a mocked producer.
        Expectation: Each device view is handed off on the caller without another copy.
        """
        dataset = build_distributed_dataset([], metadata="metadata", collate_fn=_collate)
        host_batch = [{"host": 1}, {"host": 2}]
        device_batch = [{"device": 1}, {"device": 2}]
        local_loader = Mock(last_host_batch=host_batch)
        local_loader.__next__ = Mock(return_value=device_batch)
        loader = DatasetDataLoader(dataset, local_loader, torch.device("cpu"))
        with patch.object(dataset, "move_to_device") as move:
            self.assertEqual(next(loader), device_batch)
            self.assertIs(loader.last_host_batch, host_batch)
            move.assert_not_called()

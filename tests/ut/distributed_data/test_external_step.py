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
"""Tests for the source-only external step API."""

import unittest
from collections.abc import Iterator
from contextlib import closing

import torch

from hyper_parallel.distributed_data import (
    DistributedDatasetConfig,
    SampleMetadata,
    WorkloadCost,
    build_distributed_dataloader,
)
from tests.common.mark_utils import arg_mark


class _Mesh:
    mesh_shape = (1,)
    mesh_dim_names = ("dp",)
    rank_list = (0,)


class _Source:
    """Source that emits one raw-sample bin per step without checkpoint hooks."""

    def __init__(self) -> None:
        """Initialize two deterministic source steps and their cursor."""
        self.steps = [[[{"id": 0, "tokens": 4}]], [[{"id": 1, "tokens": 5}]]]
        self.position = 0

    def __iter__(self) -> Iterator[list[list[dict[str, int]]]]:
        """Yield complete local steps while advancing the source cursor."""
        while self.position < len(self.steps):
            step = self.steps[self.position]
            self.position += 1
            yield step

    def set_epoch(self, epoch: int) -> None:
        """Restart the deterministic source for any epoch.

        Args:
            epoch: Requested epoch; this fixture repeats the same samples.
        """
        del epoch
        self.position = 0


def _build(source):
    return build_distributed_dataloader(
        None,
        _Mesh(),
        DistributedDatasetConfig(seq_len=16, local_batch_size=1),
        metadata_fn=lambda sample: SampleMetadata(sample["tokens"], sample_id=sample["id"]),
        pack_fn=lambda samples, seq_len: tuple(samples) if sum(item["tokens"] for item in samples) <= seq_len else None,
        collate_fn=list,
        external_step_source=source,
        device="cpu", cost_model=lambda metadata: metadata.cost,
    )


def _samples() -> list[dict]:
    return [
        {"input_ids": torch.tensor([index, index + 1]),
         "metadata": SampleMetadata(2, cost=WorkloadCost(llm=index + 1))}
        for index in range(4)
    ]


def _pack(samples: list[dict], seq_len: int) -> dict:
    del seq_len
    return {"input_ids": torch.cat([sample["input_ids"] for sample in samples])}


class _CountingSource:
    """Expose child-owned reads and epoch state through the checkpoint protocol."""

    def __init__(self, steps: list) -> None:
        """Store immutable steps and a consumed source cursor."""
        self.steps, self.position, self.epoch = steps, 0, 0

    def __iter__(self) -> Iterator:
        """Advance only inside the source-owning process."""
        while self.position < len(self.steps):
            value = self.steps[self.position]
            self.position += 1
            yield value

    def state_dict(self) -> dict:
        """Expose actual reads, including the single speculative step."""
        return {"position": self.position, "epoch": self.epoch}

    def load_state_dict(self, state: dict) -> None:
        """Restore source-owned progress."""
        self.position, self.epoch = state["position"], state["epoch"]

    def set_epoch(self, epoch: int) -> None:
        """Reset data while retaining the selected epoch for verification."""
        self.position, self.epoch = 0, epoch


class TestExternalStepSource(unittest.TestCase):
    """Verify source iteration, metadata, and externally owned recovery."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_source_steps_and_optional_epoch_hook(self) -> None:
        """Feature: External complete-step sources.
        Description: Consume two selected steps with the automatic balancing pipeline.
        Expectation: Membership is preserved and set_epoch restarts the external source.
        """
        source = _Source()
        loader = _build(source)
        try:
            expected = list(loader)
            self.assertEqual(expected, [[({"id": 0, "tokens": 4},)], [({"id": 1, "tokens": 5},)]])
            loader.set_epoch(1)
            self.assertEqual(list(loader), expected)
        finally:
            loader.close()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_source_does_not_require_checkpoint_hooks(self) -> None:
        """Feature: External source lifecycle validation.
        Description: Consume plain iterable steps, then wrap a caller-restored source.
        Expectation: HP consumes the supplied position without requiring checkpoint or epoch hooks.
        """
        source = [[[{"id": 0, "tokens": 4}]], [[{"id": 1, "tokens": 5}]]]
        loader = _build(source)
        try:
            self.assertEqual(list(loader), [[({"id": 0, "tokens": 4},)], [({"id": 1, "tokens": 5},)]])
        finally:
            loader.close()
        # The caller reconstructs its source at the next step before wrapping it.
        resumed = _build(source[1:])
        try:
            self.assertEqual(list(resumed), [[({"id": 1, "tokens": 5},)]])
        finally:
            resumed.close()


class TestExternalStepCallbacks(unittest.TestCase):
    """Verify raw-step callbacks, buffering and source lifecycle without process groups."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_enabled_balance_stops_prefetch_at_limit_and_reports_custom_cost(self) -> None:
        """Feature: Buffered local balancing.
        Description: Apply a custom cost model and an explicit one-step limit.
        Expectation: Logging uses modeled costs and prefetch does not cross the limit.
        """
        samples = _samples()
        source = _CountingSource([[samples[:2]], [samples[2:]]])

        config = DistributedDatasetConfig(seq_len=8, local_batch_size=1)
        with closing(build_distributed_dataloader(
                None, _Mesh(), config, device="cpu", max_steps=1,
                external_step_source=source, metadata_fn=lambda sample: sample["metadata"],
                pack_fn=_pack, collate_fn=list,
                cost_model=lambda metadata: WorkloadCost(llm=metadata.cost.llm * 10),
        )) as loader:
            with self.assertLogs("hyper_parallel.distributed_data.balance_logging", level="INFO") as logs:
                batch = next(loader)
            self.assertEqual(batch[0]["input_ids"].tolist(), [0, 1, 1, 2])
            stats = dict(loader.last_balance_stats)
            self.assertEqual(stats["cost_before"], (30,))
            self.assertEqual(stats["bins_before"][0][0], {"samples": 2, "seq_len": 4, "cost": 30})
            self.assertEqual(stats["bins_after"][0][0], {"samples": 2, "seq_len": 4, "cost": 30})
            self.assertIn("mb0: seq_len=4, samples=2, cost=30", logs.output[0])
            self.assertIn("dp0: send 0 samples, recv 0 samples", logs.output[0])
            self.assertEqual(loader.group_ranks, (0,))
            with self.assertRaises(StopIteration):
                next(loader)
            self.assertEqual(loader.state_dict()["producer"]["source"]["position"], 1)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_default_balancing_requires_backbone_dimensions(self) -> None:
        """Feature: Default cost model.
        Description: Build enabled balancing without model dimensions or a callback.
        Expectation: Startup rejects the missing configuration.
        """
        with self.assertRaisesRegex(ValueError, "model_config"):
            build_distributed_dataloader(
                None, _Mesh(), DistributedDatasetConfig(seq_len=8, local_batch_size=1),
                external_step_source=[], metadata_fn=lambda sample: sample["metadata"],
                pack_fn=_pack, collate_fn=list, device="cpu",
            )

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_epoch_and_checkpoint_contract(self) -> None:
        """Feature: External source lifecycle.
        Description: Change epochs and restore the delivered cursor after source EOF.
        Expectation: Epochs reach the child source and restore preserves the consumed cursor.
        """
        source = _CountingSource([[_samples()]])
        with closing(build_distributed_dataloader(
                None, _Mesh(), DistributedDatasetConfig(seq_len=8, local_batch_size=1),
                external_step_source=source, metadata_fn=lambda sample: sample["metadata"],
                pack_fn=_pack, collate_fn=list, device="cpu", cost_model=lambda metadata: metadata.cost,
        )) as loader:
            first = next(loader)[0]["input_ids"]
            loader.set_epoch(3)
            self.assertEqual(loader.state_dict()["producer"]["source"]["epoch"], 3)
            self.assertEqual(loader.step, 0)
            self.assertIsNone(loader.last_host_batch)
            torch.testing.assert_close(next(loader)[0]["input_ids"], first)
            checkpoint = loader.state_dict()
            self.assertEqual(checkpoint["producer"]["step"], 1)
            loader.load_state_dict(checkpoint)
            with self.assertRaises(StopIteration):
                next(loader)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_background_collator_error_reaches_consumer(self) -> None:
        """Feature: Background failure propagation.
        Description: Raise from a user collator in the CPU producer process.
        Expectation: The waiting consumer receives the producer traceback.
        """
        def fail(samples: list, seq_len: int) -> None:
            """Raise a recognizable error inside the producer."""
            del samples, seq_len
            raise RuntimeError("collation failed")
        with closing(build_distributed_dataloader(
                None, _Mesh(), DistributedDatasetConfig(seq_len=8, local_batch_size=1),
                external_step_source=[[_samples()]], metadata_fn=lambda sample: sample["metadata"],
                pack_fn=fail, collate_fn=list,
                device="cpu", cost_model=lambda metadata: metadata.cost,
        )) as loader:
            with self.assertRaisesRegex(RuntimeError, "collation failed"):
                next(loader)

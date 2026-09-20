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
"""Tests for the source-only external step adapter."""

import unittest
from collections.abc import Iterator

from hyper_parallel.distributed_data import DistributedDatasetConfig, SampleMetadata, build_distributed_dataloader
from tests.common.mark_utils import arg_mark


class _Mesh:
    mesh_shape = (1,)
    mesh_dim_names = ("dp",)
    rank_list = (0,)


class _Source:
    """Checkpointable source that emits one raw-sample bin per step."""

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

    def state_dict(self) -> dict[str, int]:
        """Save the cursor after the most recently emitted step."""
        return {"position": self.position}

    def load_state_dict(self, state: dict[str, int]) -> None:
        """Restore the source cursor.

        Args:
            state: Saved source position.
        """
        self.position = state["position"]

    def set_epoch(self, epoch: int) -> None:
        """Restart the deterministic source for any epoch.

        Args:
            epoch: Requested epoch; this fixture repeats the same samples.
        """
        del epoch
        self.position = 0


def _build(source: _Source):
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


class TestExternalStepAdapter(unittest.TestCase):
    """Verify that HP owns source metadata and checkpoint lifecycle."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_source_only_steps_and_checkpoint_contract(self) -> None:
        """Feature: External complete-step sources.
        Description: Consume two selected steps with the automatic balancing pipeline.
        Expectation: Membership is preserved; unsupported resume fails explicitly.
        """
        source = _Source()
        loader = _build(source)
        self.assertEqual(next(loader), [({"id": 0, "tokens": 4},)])
        with self.assertRaisesRegex(NotImplementedError, "checkpoint"):
            loader.state_dict()
        self.assertEqual(next(loader), [({"id": 1, "tokens": 5},)])

        with self.assertRaisesRegex(NotImplementedError, "checkpoint"):
            loader.load_state_dict({})
        loader.close()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_source_does_not_require_checkpoint_hooks(self) -> None:
        """Feature: External source lifecycle validation.
        Description: Build an HP loader around an iterable without checkpoint hooks.
        Expectation: A plain iterable is accepted and exhausts normally.
        """
        class Incomplete:
            def __iter__(self) -> Iterator[None]:
                """Return an empty stream without implementing checkpoint hooks."""
                return iter(())

        loader = _build(Incomplete())
        self.assertEqual(list(loader), [])
        loader.close()

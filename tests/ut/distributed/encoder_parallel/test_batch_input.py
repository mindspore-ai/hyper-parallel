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
"""Encoder Parallel planning through the common runtime-input contract."""

import unittest
from collections.abc import Mapping
from typing import Any

import torch

from hyper_parallel.data.batching.runtime_input import RuntimeInputAdapter, RuntimeInputContext
from hyper_parallel.distributed.encoder_parallel.batch_input import EncoderParallelInputAdapter


class _RecordedRuntime:
    """Record planning order without distributed initialization."""

    def __init__(self, events: list[str]) -> None:
        """Keep the shared call-order recorder."""
        self.events = events

    def preflight_batch(self, batch: Mapping[str, Any]) -> tuple[str, str]:
        """Return local image and layout markers."""
        self.events.append("preflight")
        return batch.get("image", "text"), "layout"

    def finalize_plan(self, image: str, layout: str) -> tuple[str, str]:
        """Record the point at which collective planning would begin."""
        self.events.append("finalize")
        return image, layout


class _RecordedInputs(RuntimeInputAdapter):
    """Stand in for the model's packed-sequence input provider."""

    def __init__(self, events: list[str], field: str = "packed_seq_params") -> None:
        """Choose the model field and the shared call-order recorder."""
        self.events = events
        self.field = field

    def runtime_input_fields(self) -> tuple[str, ...]:
        """Declare the field returned by this model provider."""
        return (self.field,)

    def build_runtime_inputs(self, *, batch: Mapping[str, Any],
                             context: RuntimeInputContext) -> Mapping[str, Any]:
        """Record local input production and return one forward field."""
        del batch, context
        self.events.append("inner")
        return {self.field: "packed"}


class TestEncoderParallelInputAdapter(unittest.TestCase):
    """Keep local validation before the first planning collective."""

    @staticmethod
    def _context() -> RuntimeInputContext:
        """Provide the context normally built by OmniParallelBatch."""
        return RuntimeInputContext(
            local_input_shape=(1, 8), parallel_ranks={"tp": 0, "cp": 0},
            parallel_sizes={"tp": 1, "cp": 1}, options={},
        )

    def test_composes_model_inputs_and_one_plan_in_order(self):
        """An image batch and text-only batch each get exactly one ordered plan."""
        events = []
        adapter = EncoderParallelInputAdapter(
            runtime=_RecordedRuntime(events), inner=_RecordedInputs(events),
        )
        self.assertEqual(adapter.runtime_input_fields(), ("packed_seq_params", "encoder_parallel_plan"))
        for batch, expected in (({"image": "image-0"}, "image-0"), ({}, "text")):
            inputs = adapter.build_runtime_inputs(batch=batch, context=self._context())
            self.assertEqual(inputs, {"packed_seq_params": "packed",
                                      "encoder_parallel_plan": (expected, "layout")})
        self.assertEqual(events, ["preflight", "inner", "finalize"] * 2)

    def test_collision_stops_before_collective_planning(self):
        """A model provider cannot overwrite the batch or visual plan field."""
        events = []
        adapter = EncoderParallelInputAdapter(
            runtime=_RecordedRuntime(events), inner=_RecordedInputs(events, "input_ids"),
        )
        with self.assertRaisesRegex(ValueError, "runtime inputs cannot replace Omni model inputs"):
            adapter.build_runtime_inputs(batch={"input_ids": torch.zeros(1, 8)}, context=self._context())
        self.assertEqual(events, ["preflight", "inner"])

        events.clear()
        with self.assertRaisesRegex(ValueError, "encoder_parallel_plan has two owners"):
            EncoderParallelInputAdapter(
                runtime=_RecordedRuntime(events), inner=_RecordedInputs(events, "encoder_parallel_plan"),
            )
        self.assertEqual(events, [])

    def test_standalone_provider_rejects_existing_plan(self):
        """A caller-provided plan is rejected before finalization."""
        events = []
        adapter = EncoderParallelInputAdapter(runtime=_RecordedRuntime(events), inner=None)
        self.assertEqual(adapter.runtime_input_fields(), ("encoder_parallel_plan",))
        with self.assertRaisesRegex(ValueError, "runtime inputs cannot replace Omni model inputs"):
            adapter.build_runtime_inputs(batch={"encoder_parallel_plan": object()}, context=self._context())
        self.assertEqual(events, ["preflight"])

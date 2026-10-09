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
"""Unit tests for model-owned runtime input extensions."""

import unittest
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import torch

from hyper_parallel.data.batching.get_batch import OmniParallelBatch
from hyper_parallel.data.batching.runtime_input import (
    RuntimeInputAdapter,
    RuntimeInputContext,
)
from tests.common.mark_utils import arg_mark


class _ModalityRuntimeAdapter(RuntimeInputAdapter):
    """Test adapter proving that runtime inputs are not attention-specific."""

    def runtime_input_fields(self) -> tuple[str, ...]:
        """Declare the fields produced for every batch."""
        return ("runtime_modality_route", "runtime_cp_rank")

    def build_runtime_inputs(
            self,
            *,
            batch: Mapping[str, Any],
            context: RuntimeInputContext,
    ) -> Mapping[str, Any]:
        """Forward a modality route and record generic execution context."""
        return {
            "runtime_modality_route": batch["modality_route"],
            "runtime_cp_rank": context.parallel_ranks["cp"],
        }


class _CollidingRuntimeAdapter(RuntimeInputAdapter):
    """Produce one invalid field already owned by the Omni batch."""

    def build_runtime_inputs(
            self,
            *,
            batch: Mapping[str, Any],
            context: RuntimeInputContext,
    ) -> Mapping[str, Any]:
        """Attempt to replace canonical token IDs."""
        del context
        return {"input_ids": batch["input_ids"]}


class TestRuntimeInputAdapter(unittest.TestCase):
    """Generic adapters may add arbitrary non-colliding forward inputs."""

    @staticmethod
    def _parallel_context() -> SimpleNamespace:
        """Build the topology fields consumed by RuntimeInputContext."""
        return SimpleNamespace(
            tp_rank=0,
            tp_world_size=2,
            cp_rank=1,
            cp_world_size=2,
        )

    @staticmethod
    def _omni_batch_runtime(adapter: RuntimeInputAdapter) -> OmniParallelBatch:
        """Build an uninitialized runtime for pure input-contract tests."""
        batch_runtime = OmniParallelBatch.__new__(OmniParallelBatch)
        batch_runtime.runtime_input_adapter = adapter
        batch_runtime.parallel_context = TestRuntimeInputAdapter._parallel_context()
        return batch_runtime

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_omni_batch_merges_non_attention_runtime_inputs(self):
        """Exercise a model-owned extension of an Omni batch.

        Feature: Generic runtime-input adapter support.
        Description: Build modality metadata from topology and batch fields.
        Expectation: Non-colliding fields are forwarded without an allowlist.
        """
        batch_runtime = self._omni_batch_runtime(_ModalityRuntimeAdapter())
        parallel_batch = {
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "labels": torch.ones(1, 4, dtype=torch.long),
            "loss_mask": torch.ones(1, 4, dtype=torch.long),
            "modality_route": torch.tensor([[0, 1, 1, 0]]),
        }

        runtime_inputs = batch_runtime._build_runtime_inputs(  # pylint: disable=protected-access
            parallel_batch
        )
        model_inputs, _ = batch_runtime._split_model_and_loss_inputs(  # pylint: disable=protected-access
            parallel_batch
        )
        model_inputs.update(runtime_inputs)

        torch.testing.assert_close(
            model_inputs["runtime_modality_route"],
            parallel_batch["modality_route"],
        )
        self.assertEqual(model_inputs["runtime_cp_rank"], 1)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_omni_batch_rejects_runtime_field_collision(self):
        """Reject a runtime adapter that replaces encoded Omni fields.

        Feature: Runtime-input field ownership.
        Description: Return ``input_ids`` from a model runtime adapter.
        Expectation: The merge fails with ``HP-DATA-001`` before forward.
        """
        batch_runtime = self._omni_batch_runtime(_CollidingRuntimeAdapter())
        parallel_batch = {
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "labels": torch.ones(1, 4, dtype=torch.long),
        }

        with self.assertRaisesRegex(ValueError, "HP-DATA-001"):
            batch_runtime._build_runtime_inputs(  # pylint: disable=protected-access
                parallel_batch
            )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_runtime_context_is_feature_neutral(self):
        """Keep model-specific options opaque to the generic context.

        Feature: Feature-neutral runtime context.
        Description: Construct context with generic topology and one opaque option.
        Expectation: The option survives without adding attention-specific state.
        """
        context = RuntimeInputContext(
            local_input_shape=(1, 8),
            parallel_ranks={"tp": 1, "cp": 0},
            parallel_sizes={"tp": 2, "cp": 1},
            options={"model_feature": "value"},
        )

        self.assertEqual(context.options["model_feature"], "value")
        self.assertFalse(hasattr(context, "attention_mask"))


if __name__ == "__main__":
    unittest.main()

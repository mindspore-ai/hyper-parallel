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

from hyper_parallel.data.batching.get_batch import ParallelBatch
from hyper_parallel.data.batching.build_collate_fn import (
    BatchConstraints,
    DataBatchAdapter,
    DataBatchContext,
)
from hyper_parallel.data.batching.build_dataloader import TextTokenBatcher
from hyper_parallel.data.batching.runtime_input import (
    RuntimeInputAdapter,
    RuntimeInputContext,
)
from tests.common.mark_utils import arg_mark


class _ModalityRuntimeAdapter(RuntimeInputAdapter):
    """Test adapter proving that runtime inputs are not attention-specific."""

    def build_runtime_inputs(
            self,
            *,
            batch: Mapping[str, Any],
            context: RuntimeInputContext,
    ) -> Mapping[str, Any]:
        """Forward a modality route and record generic execution context."""
        return {
            "modality_route": batch["modality_route"],
            "runtime_cp_rank": context.parallel_ranks["cp"],
        }


class _AlignedDataBatchAdapter(DataBatchAdapter):
    """Charge physical length rounded to a four-token model constraint."""

    def item_cost(
            self,
            item: Mapping[str, Any],
            context: DataBatchContext,
    ) -> int:
        """Return the aligned physical token count."""
        del context
        item_length = int(item["input_ids"].shape[-1])
        return item_length + (-item_length) % 4

    def constraints(self, context: DataBatchContext) -> BatchConstraints:
        """Declare the same final sequence alignment."""
        del context
        return BatchConstraints(sequence_multiple=4)


class TestRuntimeInputAdapter(unittest.TestCase):
    """Generic adapters may add arbitrary non-colliding forward inputs."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_parallel_batch_merges_non_attention_runtime_inputs(self):
        """
        Feature: runtime input
        Description: A model adapter can extend a dense batch with modality metadata.
        Expectation: Parallel batch merges non attention runtime inputs.
        """
        batch_runtime = ParallelBatch.__new__(ParallelBatch)
        batch_runtime.runtime_input_adapter = _ModalityRuntimeAdapter()
        batch_runtime.source_type = "online"
        batch_runtime.attention_mode = "dense"
        batch_runtime.cp_algorithm = "ulysses"
        batch_runtime.causal = True
        batch_runtime.sliding_window = None
        batch_runtime.labels_are_shifted = False
        batch_runtime.parallel_context = SimpleNamespace(
            tp_rank=0,
            tp_world_size=2,
            cp_rank=1,
            cp_world_size=2,
        )
        parallel_batch = {
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "labels": torch.ones(1, 4, dtype=torch.long),
            "position_ids": torch.arange(4).unsqueeze(0),
            "attention_mask": None,
            "swa_mask": None,
            "loss_mask": torch.ones(1, 4, dtype=torch.long),
            "modality_route": torch.tensor([[0, 1, 1, 0]]),
        }

        runtime_inputs = batch_runtime._build_runtime_inputs(  # pylint: disable=protected-access
            parallel_batch
        )
        model_inputs, _ = batch_runtime._split_model_and_loss_inputs(  # pylint: disable=protected-access
            parallel_batch,
            runtime_inputs,
        )

        torch.testing.assert_close(
            model_inputs["modality_route"],
            parallel_batch["modality_route"],
        )
        self.assertEqual(model_inputs["runtime_cp_rank"], 1)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_runtime_context_is_feature_neutral(self):
        """
        Feature: runtime input
        Description: The public context stores generic topology and opaque options.
        Expectation: Runtime context is feature neutral.
        """
        context = RuntimeInputContext(
            source_type="online",
            local_input_shape=(1, 8),
            parallel_ranks={"tp": 1, "cp": 0},
            parallel_sizes={"tp": 2, "cp": 1},
            options={"model_feature": "value"},
        )

        self.assertEqual(context.options["model_feature"], "value")
        self.assertFalse(hasattr(context, "attention_mask"))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_dynamic_batching_uses_adapter_physical_item_cost(self):
        """
        Feature: runtime input
        Description: Selection budgets transformed physical tokens, not raw lengths.
        Expectation: Dynamic batching uses adapter physical item cost.
        """
        context = DataBatchContext(
            source_type="online",
            token_budget=6,
        )
        batcher = TextTokenBatcher(
            token_budget=6,
            min_buffered_samples=2,
            batch_adapter=_AlignedDataBatchAdapter(),
            batch_context=context,
        )
        first = {"input_ids": torch.tensor([1, 2, 3])}
        second = {"input_ids": torch.tensor([4, 5])}
        batcher.put_item(first)
        batcher.put_item(second)

        selected = batcher.get_micro_batch()

        self.assertEqual(len(selected), 1)
        self.assertIs(selected[0], first)
        self.assertEqual(batcher.buffer_token_count, 4)


if __name__ == "__main__":
    unittest.main()

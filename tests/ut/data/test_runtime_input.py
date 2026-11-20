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
from unittest.mock import patch

import torch

from hyper_parallel.data.batching.get_batch import OmniParallelBatch, TextParallelBatch
from hyper_parallel.data.constants import OMNI_CP_TOKEN_FIELDS
from hyper_parallel.data.parallel import CPBatchSharder
from hyper_parallel.data.batching.runtime_input import (
    AttentionRuntime,
    PackedAttentionRuntime,
    RuntimeInputAdapter,
    RuntimeInputContext,
)


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
        batch_runtime.sequence_parallel = False
        return batch_runtime

    def test_omni_chunk_targets_preserve_cp_boundaries(self):
        """Preserve a cross-shard target in the same document and mask document ends."""
        batch = {
            "input_ids": torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17]]),
            "labels": torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17]]),
            "loss_mask": torch.tensor([[1, 1, 0, 1, 1, 1, 0, 1]]),
            "cu_seq_lens": torch.tensor([0, 3, 8]),
        }
        OmniParallelBatch._align_chunk_loss(batch)  # pylint: disable=protected-access
        for rank, targets in enumerate(([[11, 12, -100, 14]], [[15, 16, 17, -100]])):
            context = SimpleNamespace(cp_rank=rank, cp_world_size=2, tp_world_size=1)
            local_batch = CPBatchSharder(context, token_fields=OMNI_CP_TOKEN_FIELDS).shard(batch)
            model_inputs, loss_inputs = OmniParallelBatch._split_model_and_loss_inputs(  # pylint: disable=protected-access
                local_batch
            )
            self.assertNotIn("shift_labels", model_inputs)
            torch.testing.assert_close(loss_inputs["shift_labels"], torch.tensor(targets))
        torch.testing.assert_close(batch["loss_mask"], torch.tensor([[1, 0, 0, 1, 1, 0, 1, 0]]))

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

    def test_omni_batch_rejects_runtime_field_collision(self):
        """Reject a runtime adapter that replaces encoded Omni fields.

        Feature: Runtime-input field ownership.
        Description: Return ``input_ids`` from a model runtime adapter.
        Expectation: The merge fails before forward.
        """
        batch_runtime = self._omni_batch_runtime(_CollidingRuntimeAdapter())
        parallel_batch = {
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "labels": torch.ones(1, 4, dtype=torch.long),
        }

        with self.assertRaisesRegex(ValueError, "runtime inputs cannot replace Omni model inputs"):
            batch_runtime._build_runtime_inputs(  # pylint: disable=protected-access
                parallel_batch
            )

    def test_packed_text_keeps_global_bounds_without_a_dense_mask(self):
        """Feature: Compressed Trainer inputs preserve B2/CP document coordinates.
        Description: Build the real text runtime fields and reject malformed bounds.
        Expectation: Q/K share one host tuple and no dense mask builder runs.
        """
        runtime = TextParallelBatch.__new__(TextParallelBatch)
        runtime.parallel_context = self._parallel_context()
        runtime.runtime_input_adapter = PackedAttentionRuntime()
        runtime.attention_runtime = AttentionRuntime(mode="compressed", create_mask=True,
                                                     reset_mask=True, sliding_window=None)
        runtime.source_type, runtime.causal, runtime.sliding_window = "online", True, None
        batch = {"input_ids": torch.zeros(2, 4, dtype=torch.long), "cu_seq_lens": torch.tensor([0, 3, 8, 10, 16])}
        with patch.object(AttentionRuntime, "_build_dense_attention_masks", side_effect=AssertionError("dense mask")):
            inputs = runtime._build_runtime_inputs(batch)  # pylint: disable=protected-access
        self.assertIsNone(inputs["attention_mask"])
        self.assertEqual(inputs["cu_seq_lens_q"], (0, 3, 8, 10, 16))
        self.assertIs(inputs["cu_seq_lens_q"], inputs["cu_seq_lens_k"])
        for invalid in ([0, 3, 8], [0, 0, 16], [0.0, 16.0]):
            with self.subTest(boundaries=invalid), self.assertRaises(ValueError):
                runtime._build_runtime_inputs(dict(batch, cu_seq_lens=invalid))  # pylint: disable=protected-access

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

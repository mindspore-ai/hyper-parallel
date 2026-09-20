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
"""Unit tests for model-neutral VLM collation."""
# pylint: disable=wrong-import-position

import unittest

import pytest
import torch

pytest.importorskip("torchdata.stateful_dataloader")

from hyper_parallel.data.batching.build_collate_fn import (
    DataBatchAdapter,
    DataBatchContext,
)
from hyper_parallel.data.vlm.collator import VLMCollator
from hyper_parallel.data.vlm.get_batch import VLMBatchProcessor


class TestVLMCollator(unittest.TestCase):
    """VLM collation delegates field semantics to the batch adapter."""

    def test_default_adapter_collates_fixed_shape_mappings(self):
        """The default adapter retains PyTorch collation for generic fields."""
        collator = VLMCollator()

        batch = collator([
            {
                "input_ids": torch.tensor([1, 2]),
                "labels": torch.tensor([-100, 2]),
                "model_feature": torch.tensor([3]),
            },
            {
                "input_ids": torch.tensor([4, 5]),
                "labels": torch.tensor([-100, 5]),
                "model_feature": torch.tensor([6]),
            },
        ])

        torch.testing.assert_close(batch["input_ids"], torch.tensor([[1, 2], [4, 5]]))
        torch.testing.assert_close(batch["model_feature"], torch.tensor([[3], [6]]))

    def test_collator_runs_adapter_lifecycle_in_order(self):
        """Preparation, custom collation, and finalization share one context."""
        events = []

        class _LifecycleAdapter(DataBatchAdapter):
            """Record each adapter lifecycle hook invoked by the collator."""

            def prepare_items(self, items, context):
                events.append(("prepare", context.source_type))
                return items

            def collate_items(self, items, context):
                events.append(("collate", context.source_type))
                return {
                    "input_ids": torch.stack([item["input_ids"] for item in items]),
                    "labels": torch.stack([item["labels"] for item in items]),
                }

            def finalize_batch(self, batch, context):
                events.append(("finalize", context.source_type))
                return {**batch, "adapter_marker": context.source_type}

        context = DataBatchContext(source_type="multimodal")
        collator = VLMCollator(
            context=context,
            batch_adapter=_LifecycleAdapter(),
        )

        batch = collator([
            {"input_ids": torch.tensor([1]), "labels": torch.tensor([1])},
        ])

        self.assertEqual(
            events,
            [("prepare", "multimodal"), ("collate", "multimodal"), ("finalize", "multimodal")],
        )
        self.assertEqual(batch["adapter_marker"], "multimodal")

    def test_batch_processor_forwards_adapter_defined_model_fields(self):
        """Unknown modality fields reach the model without a framework allowlist."""
        model_inputs, loss_inputs = VLMBatchProcessor.prepare_batch({
            "input_ids": torch.tensor([[1, 2]]),
            "labels": torch.tensor([[-100, 2]]),
            "loss_mask": torch.tensor([[False, True]]),
            "custom_modality_route": torch.tensor([[3, 4]]),
        })

        self.assertIn("custom_modality_route", model_inputs)
        self.assertNotIn("loss_mask", model_inputs)
        self.assertEqual(set(loss_inputs), {"labels", "loss_mask"})


if __name__ == "__main__":
    unittest.main()

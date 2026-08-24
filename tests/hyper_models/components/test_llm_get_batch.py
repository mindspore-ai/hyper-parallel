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
"""Tests for the LLM runtime batch label contract."""

import torch

from hyper_parallel.auto_models.components.datasets.llm.get_batch import LLMBatchProcessor


def _processor(*, labels_are_shifted: bool) -> LLMBatchProcessor:
    """Build a processor for runtime-field-only tests."""
    return LLMBatchProcessor(
        None,
        eod_token_id=99,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        create_attention_mask=False,
        labels_are_shifted=labels_are_shifted,
    )


def test_pre_shifted_labels_are_forwarded_to_model_loss() -> None:
    """Indexed next-token targets should bypass the model's causal shift."""
    input_ids = torch.tensor([[10, 11, 12]])
    labels = torch.tensor([[11, 12, 13]])

    runtime_batch = _processor(labels_are_shifted=True).build_runtime_batch(
        {"input_ids": input_ids, "labels": labels}
    )
    model_inputs, _ = LLMBatchProcessor.prepare_runtime_batch(runtime_batch)

    assert torch.equal(model_inputs["labels"], labels)
    assert torch.equal(model_inputs["shift_labels"], labels)


def test_unshifted_labels_keep_model_owned_causal_shift() -> None:
    """Plain-text labels should retain the default Transformers contract."""
    input_ids = torch.tensor([[10, 11, 12]])

    runtime_batch = _processor(labels_are_shifted=False).build_runtime_batch(
        {"input_ids": input_ids, "labels": input_ids.clone()}
    )
    model_inputs, _ = LLMBatchProcessor.prepare_runtime_batch(runtime_batch)

    assert "shift_labels" not in model_inputs

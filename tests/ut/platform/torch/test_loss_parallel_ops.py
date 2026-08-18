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
"""Unit tests for PyTorch loss parallel operations."""

import os
from types import SimpleNamespace

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
import pytest  # pylint: disable=C0413
import torch  # pylint: disable=C0413

from hyper_parallel.platform.torch.loss_parallel_ops import (  # pylint: disable=C0413
    DistributedCrossEntropyFunction,
)


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
@pytest.mark.parametrize("use_weight", [False, True])
def test_distributed_cross_entropy_backward_matches_closed_form(reduction: str, use_weight: bool) -> None:
    """Backward matches the sharded cross-entropy gradient for all reductions."""
    torch.manual_seed(7)
    batch_size = 5
    vocab_size = 8
    vocab_start = 4
    local_vocab_size = 4
    ignore_index = -100

    logits = torch.randn(batch_size, vocab_size, dtype=torch.float64)
    log_probs_local = logits.log_softmax(dim=-1)[:, vocab_start:]
    target = torch.tensor([0, 4, 7, ignore_index, 2])
    weight = torch.linspace(0.5, 1.5, vocab_size, dtype=torch.float64) if use_weight else None
    valid_target = target[target != ignore_index]
    total_weight = weight[valid_target].sum() if weight is not None else valid_target.numel()
    total_weight = torch.as_tensor(total_weight, dtype=logits.dtype).reshape(1)
    grad_output = (
        torch.randn(batch_size, dtype=logits.dtype)
        if reduction == "none"
        else torch.randn(1, dtype=logits.dtype)
    )
    ctx = SimpleNamespace(
        saved_tensors=(None, log_probs_local, target, weight, total_weight, None, None),
        reduction=reduction,
        ignore_index=ignore_index,
        local_vocab_size=local_vocab_size,
        vocab_start=vocab_start,
    )

    grad_input = DistributedCrossEntropyFunction.backward(ctx, grad_output)[0]

    local_vocab_indices = torch.arange(vocab_start, vocab_size)
    local_target_mask = (target.unsqueeze(-1) == local_vocab_indices).to(logits.dtype)
    sample_weight = torch.ones(batch_size, dtype=logits.dtype) if weight is None else weight[target.clamp_min(0)]
    grad_scale = grad_output / total_weight if reduction == "mean" else grad_output
    row_scale = grad_scale * sample_weight * (target != ignore_index)
    expected = (log_probs_local.exp() - local_target_mask) * row_scale.unsqueeze(-1)
    torch.testing.assert_close(grad_input, expected)

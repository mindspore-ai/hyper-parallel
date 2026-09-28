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
"""CPU regression tests for fused Indexer/KL adapter boundary handling."""

import unittest
from unittest.mock import patch

import torch

from hyper_parallel.components.functional.compressed_indexer_ops import _stable_prediction_log
from hyper_parallel.core.activation_memory import checkpoint
from hyper_parallel.distributed.activation_checkpoint import make_selective_checkpoint_context_fn
from tests.common.mark_utils import arg_mark


class TestCompressedIndexerOps(unittest.TestCase):
    """Keep probability recovery compatible with strict selective recomputation."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_underflow_recovery_preserves_selective_checkpoint_cache(self):
        """Recover underflowed probabilities during strict SAC replay.

        Feature: Sparse Indexer KL numerical recovery.
        Description: Replay a cached contraction used to recover a zero probability.
        Expectation: The cache remains valid and loss/input gradients match the analytic values.
        """
        query = torch.tensor([[[[1., 1.]]]], requires_grad=True)
        key = torch.tensor([[[1., 1.], [-1., -1.]]])
        weights = torch.tensor([[[1000.]]])
        indices = torch.tensor([[[0, 1]]])
        prediction = torch.tensor([[[1., 0.]]])

        def _contraction(equation, queries, keys):
            """Expose the same cache/version aliasing on a CPU matmul backend."""
            self.assertEqual(equation, "bqhd,bqkd->bqhk")
            return torch.mm(queries[0, 0], keys[0, 0].T).unsqueeze(0).unsqueeze(0)

        def _region(value):
            return _stable_prediction_log(prediction, value, key, weights, indices, 1).sum()

        # CPU einsum can hide mutation behind an unsafe view. Regular views
        # preserve the cached matmul version counter, as in the NPU failure.
        with patch("torch.einsum", side_effect=_contraction):
            loss = checkpoint(
                _region, query, use_reentrant=False,
                context_fn=make_selective_checkpoint_context_fn(),
            )
            loss.backward()
        torch.testing.assert_close(loss, torch.tensor(-2000.), rtol=0, atol=0)
        torch.testing.assert_close(query.grad, torch.full_like(query, -1000.), rtol=0, atol=0)

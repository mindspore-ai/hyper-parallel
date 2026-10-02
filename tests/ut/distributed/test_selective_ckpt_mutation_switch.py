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
"""Unit tests for the SAC cache-mutation switch and indexer chunk plumbing."""

from __future__ import annotations

import unittest

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from hyper_parallel.components.modules.shared_compressed_dsa_attention import (
    SharedCompressedDSAIndexer,
)
from hyper_parallel.distributed.activation_checkpoint import (
    make_selective_checkpoint_context_fn,
)
from hyper_parallel.trainer.config.parallelism import ActivationCheckpointConfig


class CachedTensorMutationTest(unittest.TestCase):
    """Reproduce, on CPU, what the switch is for.

    Selective activation checkpointing saves the matmul output and rejects any
    in-place rewrite of it. An in-place ``relu_`` on that output is exactly
    such a rewrite, and it is deterministic: recompute replays the same
    values. The switch is the escape hatch for that case.
    """

    @staticmethod
    def _step(allow_cache_entry_mutation: bool) -> None:
        """Run one checkpointed forward/backward with an in-place rewrite."""
        torch.manual_seed(0)
        left = torch.randn(8, 8, requires_grad=True)
        right = torch.randn(8, 8, requires_grad=True)

        def region(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
            """Mutate the cached matmul output in place, as the indexer did."""
            scores = torch.matmul(first, second)
            scores.relu_()
            return scores.sum()

        loss = checkpoint(
            region, left, right, use_reentrant=False,
            context_fn=make_selective_checkpoint_context_fn(allow_cache_entry_mutation),
        )
        loss.backward()

    def test_strict_check_rejects_the_in_place_rewrite(self):
        """The default keeps the platform's strict check, so backward fails."""
        with self.assertRaises(RuntimeError) as caught:
            self._step(False)
        self.assertIn("mutated", str(caught.exception))

    def test_switch_accepts_the_deterministic_rewrite(self):
        """With the switch on, the same region completes."""
        self._step(True)


class ActivationCheckpointConfigTest(unittest.TestCase):
    """Validate the YAML-facing switch surface."""

    def test_defaults_keep_the_mutation_check_enabled(self):
        """The strict cached-tensor mutation check stays the default."""
        config = ActivationCheckpointConfig()
        self.assertFalse(config.allow_cache_entry_mutation)

    def test_non_bool_switch_is_rejected(self):
        """Ambiguous YAML values must fail fast."""
        with self.assertRaises(TypeError):
            ActivationCheckpointConfig(allow_cache_entry_mutation="yes")


class SelectiveContextFactoryTest(unittest.TestCase):
    """The switch must reach the SAC dispatch mode via the context factory.

    Regression for the silent-drop bug: the trainer forwarded the switch,
    but a build-entry signature without the keyword discarded it and every
    SAC context was constructed with the default strict check.
    """

    def test_switch_reaches_the_cached_dispatch_mode(self):
        """Factory closures must carry the configured switch value."""
        for value in (False, True):
            factory = make_selective_checkpoint_context_fn(
                allow_cache_entry_mutation=value)
            _forward_ctx, recompute_ctx = factory()
            self.assertEqual(recompute_ctx.allow_cache_entry_mutation, value,
                             f"switch value {value} lost on the way to SAC")


class _TinyIndexerSource(nn.Module):
    """Minimal indexer parameter holder accepted by the shared wrapper."""

    def __init__(self, query_chunk_size: int | None) -> None:
        """Create the projection surface the wrapper transfers."""
        super().__init__()
        self.q_b_proj = nn.Linear(8, 8, bias=False)
        self.weights_proj = nn.Linear(8, 1, bias=False)
        self.compress_ratio = 4
        self.num_heads = 1
        self.head_dim = 8
        self.index_topk = 2
        self.owns_key = True
        self.is_candidate_source = False
        self.uses_candidates = False
        self.candidate_topk_blocks = 0
        self.candidate_block_size = 1
        self.loss_coeff = 0.0
        if query_chunk_size is not None:
            self.query_chunk_size = query_chunk_size


class IndexerChunkPlumbingTest(unittest.TestCase):
    """The configured chunk size must reach the accelerator indexer."""

    def test_configured_chunk_size_is_transferred(self):
        """A source-module value overrides the wrapper default."""
        wrapper = SharedCompressedDSAIndexer(_TinyIndexerSource(1024))
        self.assertEqual(wrapper.query_chunk_size, 1024)

    def test_missing_chunk_size_falls_back_to_the_released_default(self):
        """Sources without the attribute keep the released 256 default."""
        wrapper = SharedCompressedDSAIndexer(_TinyIndexerSource(None))
        self.assertEqual(wrapper.query_chunk_size, 256)


if __name__ == "__main__":
    unittest.main()

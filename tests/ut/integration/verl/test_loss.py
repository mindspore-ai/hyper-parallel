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
"""Existing entropy regression coverage for the optional VeRL adapter."""

import unittest
from functools import partial
from importlib.util import find_spec
from unittest.mock import Mock, patch

import torch

_VERL_AVAILABLE = find_spec("verl") is not None
if _VERL_AVAILABLE:
    from hyper_parallel.integration.verl.engine import transformer_impl as ENGINE_MODULE
else:
    ENGINE_MODULE = None


def _mesh(size: int = 1, rank: int = 0) -> Mock:
    return Mock(ndim=1, get_group=Mock(return_value=None),
                get_local_rank=Mock(return_value=rank), size=Mock(return_value=size))


def _identity_reduce(value: torch.Tensor, **_kwargs) -> torch.Tensor:
    return value


def _record_saved_tensor(saved_tensors: list[torch.Tensor], tensor: torch.Tensor) -> torch.Tensor:
    """Record retained tensors without changing what autograd saves."""
    saved_tensors.append(tensor)
    return tensor


@unittest.skipUnless(_VERL_AVAILABLE, "requires the optional VeRL dependency")
class TestVocabParallelEntropy(unittest.TestCase):
    """Check entropy values, recomputed gradients and retained activation memory."""

    def test_recompute_matches_native_entropy_without_fp32_vocab_caches(self) -> None:
        """Cross chunk boundaries while retaining only input logits and per-token entropy."""
        for options in ({}, {"chunk_size": 1024}):
            with self.subTest(options=options):
                generator = torch.Generator().manual_seed(43)
                logits = torch.randn(2051, 11, generator=generator, requires_grad=True)
                reference_logits = logits.detach().clone().requires_grad_()
                saved = []

                pack = partial(_record_saved_tensor, saved)

                with patch("hyper_parallel.components.losses._vocab_parallel_cross_entropy._differentiable_all_reduce",
                              side_effect=_identity_reduce), \
                        patch.object(ENGINE_MODULE.torch.distributed, "all_reduce"), \
                        torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                    actual = ENGINE_MODULE.entropy_from_logits(logits, mesh=_mesh(), **options)
                    gradient = torch.randn(actual.shape, generator=generator)
                    actual.backward(gradient)
                reference_log_probs = reference_logits.log_softmax(-1)
                reference = -(reference_log_probs.exp() * reference_log_probs).sum(-1)
                reference.backward(gradient)
                torch.testing.assert_close(actual, reference)
                torch.testing.assert_close(logits.grad, reference_logits.grad, atol=1e-6, rtol=1e-5)
                retained = sum(tensor.numel() for tensor in saved)
                limit = logits.numel() + logits.shape[0]
                self.assertLessEqual(retained, limit, f"entropy retained {retained} elements, limit={limit}")

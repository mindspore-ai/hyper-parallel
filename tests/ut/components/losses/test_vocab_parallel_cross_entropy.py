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
"""CPU regression tests for the local vocab-parallel cross-entropy kernel."""

import unittest
from unittest.mock import Mock, patch

import torch
from torch.nn import functional

from hyper_parallel.components.losses import _vocab_parallel_cross_entropy as loss_module


def _mesh(size: int = 1, rank: int = 0) -> Mock:
    """Describe a vocab-sharding mesh without starting a process group."""
    return Mock(
        ndim=1,
        get_group=Mock(return_value=None),
        get_local_rank=Mock(return_value=rank),
        size=Mock(return_value=size),
    )


def _identity_reduce(value: torch.Tensor, **_kwargs) -> torch.Tensor:
    """Emulate a collective in a single-rank mesh."""
    return value


class TestVocabParallelCrossEntropy(unittest.TestCase):
    """Compare real forward/backward calculations with native cross-entropy."""

    def _check_single_rank(self, target: torch.Tensor, weight, reduction: str) -> None:
        """Compare single-rank loss values and scaled gradients with native cross-entropy."""
        logits = torch.randn(4, 5, generator=torch.Generator().manual_seed(17), dtype=torch.float64)
        local_logits = logits.clone().requires_grad_()
        reference_logits = logits.clone().requires_grad_()
        with patch.object(loss_module, "_differentiable_all_reduce", side_effect=_identity_reduce):
            actual = loss_module.DistributedCrossEntropyFunction.apply(
                local_logits, target, weight, -100, reduction, 5, _mesh(), 0,
            )
        reference = functional.cross_entropy(reference_logits, target, weight=weight, reduction=reduction)
        torch.testing.assert_close(actual.reshape(reference.shape), reference, equal_nan=True)

        scale = torch.linspace(0.3, 1.2, actual.numel(), dtype=actual.dtype)
        actual.backward(scale.reshape(actual.shape))
        reference.backward(scale.reshape(reference.shape))
        torch.testing.assert_close(local_logits.grad, reference_logits.grad, equal_nan=True)

    def test_all_reductions_and_class_weights(self) -> None:
        """Preserve ignored targets, weighted normalization, and upstream gradient scaling."""
        target = torch.tensor([0, -100, 4, 2])
        for reduction in ("none", "sum", "mean"):
            for weight in (None, torch.tensor([0.5, 1.0, 2.0, 3.0, 1.5], dtype=torch.float64)):
                with self.subTest(reduction=reduction, weighted=weight is not None):
                    self._check_single_rank(target, weight, reduction)

    def test_all_ignored_targets(self) -> None:
        """Match PyTorch's zero gradients and NaN mean for an entirely ignored batch."""
        for reduction in ("none", "sum", "mean"):
            with self.subTest(reduction=reduction):
                self._check_single_rank(torch.full((4,), -100), None, reduction)

    def test_uneven_vocab_shards_match_reference_gradients(self) -> None:
        """Respect nonzero shard offsets and the shorter final vocab shard."""
        logits = torch.randn(4, 5, generator=torch.Generator().manual_seed(23), dtype=torch.float64)
        target = torch.tensor([0, -100, 4, 2])
        reference_logits = logits.clone().requires_grad_()
        reference = functional.cross_entropy(reference_logits, target, reduction="sum")
        (reference * 0.7).backward()
        maximum = logits.max(dim=-1, keepdim=True).values
        exp_sum = (logits - maximum).exp().sum(dim=-1, keepdim=True)
        local_losses = []

        for rank, shard in enumerate(logits.chunk(2, dim=-1)):
            with self.subTest(rank=rank):
                local_logits = shard.clone().requires_grad_()
                with patch.object(
                    loss_module, "_differentiable_all_reduce",
                    side_effect=[maximum, exp_sum, reference.detach().reshape(1)],
                ) as reduce_mock:
                    loss = loss_module.vocab_parallel_cross_entropy_local(
                        local_logits, target, vocab_size=5, mesh=_mesh(2, rank), reduction="sum",
                    )
                local_losses.append(reduce_mock.call_args_list[2].args[0])
                # Custom autograd installs its backward even when the mocked
                # collective returns a tensor detached from the local logits.
                self.assertTrue(loss.requires_grad)
                self.assertIsNotNone(loss.grad_fn)
                (loss * 0.7).backward()
                torch.testing.assert_close(local_logits.grad, reference_logits.grad.chunk(2, dim=-1)[rank])
        torch.testing.assert_close(sum(local_losses).reshape(()), reference)

    def test_rank_without_targets_still_receives_softmax_gradient(self) -> None:
        """Keep remote-target contributions on a shard with no locally selected classes."""
        logits = torch.tensor([[0.1, -0.2, 0.4, 0.3, 0.6]], dtype=torch.float64, requires_grad=True)
        target = torch.tensor([0])
        reference = functional.cross_entropy(logits, target, reduction="sum")
        reference.backward()
        local_logits = logits.detach()[:, 3:].clone().requires_grad_()
        maximum = logits.detach().max(dim=-1, keepdim=True).values
        with patch.object(
            loss_module, "_differentiable_all_reduce",
            side_effect=[
                maximum,
                (logits.detach() - maximum).exp().sum(-1, keepdim=True),
                reference.detach().reshape(1),
            ],
        ):
            loss = loss_module.vocab_parallel_cross_entropy_local(
                local_logits, target, vocab_size=5, mesh=_mesh(2, 1), reduction="sum",
            )
        self.assertTrue(loss.requires_grad)
        self.assertIsNotNone(loss.grad_fn)
        loss.backward()
        torch.testing.assert_close(local_logits.grad, logits.grad[:, 3:])

    def test_local_loss_validates_mesh_targets_and_reduction(self) -> None:
        """Accept the named or explicit TP axis and reject ambiguous input."""
        logits = torch.randn(2, 3, 4)
        targets = torch.zeros((2, 3), dtype=torch.long)
        mesh = _mesh()
        mesh.ndim = 2
        mesh.mesh_dim_names = ("dp", "tp")
        for mesh_dim in (None, 1):
            with self.subTest(mesh_dim=mesh_dim), patch.object(
                loss_module, "_differentiable_all_reduce", side_effect=_identity_reduce,
            ):
                loss = loss_module.vocab_parallel_cross_entropy_local(
                    logits, targets, vocab_size=4, mesh=mesh, mesh_dim=mesh_dim,
                )
                torch.testing.assert_close(loss.squeeze(), functional.cross_entropy(
                    logits.flatten(0, 1), targets.flatten(), reduction="sum",
                ))
                mesh.get_group.assert_called_with(1)

        with self.assertRaisesRegex(ValueError, "mesh_dim=2"):
            loss_module.vocab_parallel_cross_entropy_local(logits, targets, vocab_size=4, mesh=mesh, mesh_dim=2)
        with self.assertRaisesRegex(ValueError, "Probabilistic target"):
            loss_module.vocab_parallel_cross_entropy_local(logits, targets.float(), vocab_size=4, mesh=mesh)
        with self.assertRaisesRegex(ValueError, "Invalid reduction"):
            loss_module.vocab_parallel_cross_entropy_local(logits, targets, vocab_size=4, mesh=mesh, reduction="max")
        mesh.mesh_dim_names = ("dp", "cp")
        with self.assertRaisesRegex(ValueError, "name the sharding axis"):
            loss_module.vocab_parallel_cross_entropy_local(logits, targets, vocab_size=4, mesh=mesh)

    def test_collective_materializes_noncontiguous_input(self) -> None:
        """The autograd-aware collective receives a contiguous tensor and the requested op."""
        values = torch.arange(6).reshape(2, 3).T
        with patch.object(loss_module.dist_func, "all_reduce", side_effect=_identity_reduce) as reduce:
            result = loss_module._differentiable_all_reduce(values, "max", group=None)
        torch.testing.assert_close(result, values)
        self.assertTrue(reduce.call_args.args[0].is_contiguous())
        self.assertEqual(reduce.call_args.kwargs, {"op": torch.distributed.ReduceOp.MAX, "group": None})

    def test_entropy_validates_mesh_and_chunk_size(self) -> None:
        """Reject ambiguous vocabulary axes and invalid chunk lengths."""
        logits = torch.zeros(2, 3)
        mesh = _mesh()
        with self.assertRaisesRegex(ValueError, "chunk_size"):
            loss_module.entropy_from_logits(logits, mesh=mesh, chunk_size=0)
        mesh.ndim = 2
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            loss_module.entropy_from_logits(logits, mesh=mesh)

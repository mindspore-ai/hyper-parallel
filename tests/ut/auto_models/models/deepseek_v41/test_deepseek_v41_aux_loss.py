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
"""Sequence balancing, expert-bias updates and Indexer gradient checks."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from torch.nn import functional
from torch.utils.checkpoint import DefaultDeviceType, checkpoint

from hyper_parallel.components.functional.aux_loss import (
    aux_loss_auto_scale,
    aux_loss_scale_context,
    bind_aux_loss_scale,
)
from hyper_parallel.components.modules.attention.compressed_sparse_attention2.indexer_kl_loss import (
    shared_compressed_indexer_kl_loss,
)
from hyper_parallel.components.modules.moe.moe_utils import update_router_bias
from hyper_parallel.models.deepseek_v41.adapter.aux_loss import (
    accumulate_router_load, sequence_load_balancing_loss, update_modality_bias,
)
from tests.common.mark_utils import arg_mark


def reference_loss(
        scores: torch.Tensor, indices: torch.Tensor, sample_ids: torch.Tensor,
        num_samples: int, mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate a dense routing map separately for each logical sample."""
    losses = []
    for sample in range(num_samples):
        selected = sample_ids == sample
        if mask is not None:
            selected = selected & mask
        if not selected.any():
            continue
        routing = functional.one_hot(indices[selected], scores.shape[-1]).sum(1).float()  # pylint: disable=not-callable
        fraction = routing.mean(0) / indices.shape[-1]
        probability = functional.normalize(scores[selected].float(), p=1, dim=-1).mean(0)
        losses.append(scores.shape[-1] * torch.dot(fraction, probability))
    return torch.stack(losses).mean() if losses else scores.sum() * 0


class TestSequenceLoadBalancingLoss(unittest.TestCase):
    """Validate logical sample boundaries, masking and distributed gradients."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_sequence_imbalance_cannot_cancel_across_samples(self) -> None:
        """
        Feature: Auxiliary-loss training lifecycle.
        Description: Two individually collapsed sequences remain penalized despite batch balance.
        Expectation: Per-sequence loss is 1.8 while pooled loss is 1.0; only the former has a nonzero gradient.
        """
        scores = torch.tensor([[0.9, 0.1], [0.9, 0.1], [0.1, 0.9], [0.1, 0.9]], requires_grad=True)
        indices = torch.tensor([[0], [0], [1], [1]])
        sample_ids = torch.tensor([0, 0, 1, 1])
        actual = sequence_load_balancing_loss(scores, indices, sample_ids, 2)
        pooled = sequence_load_balancing_loss(scores, indices, torch.zeros_like(sample_ids), 1)
        torch.testing.assert_close(actual, torch.tensor(1.8))
        torch.testing.assert_close(pooled, torch.tensor(1.0))
        torch.testing.assert_close(torch.autograd.grad(pooled, scores, retain_graph=True)[0], torch.zeros_like(scores))
        self.assertGreater(torch.autograd.grad(actual, scores)[0].abs().sum().item(), 0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_unequal_packed_samples_mask_and_mixed_precision(self) -> None:
        """
        Feature: Auxiliary-loss training lifecycle.
        Description: Samples have equal weight despite different token counts; masks exclude padding.
        Expectation: The loss and gradients match a separate per-sample dense oracle.
        """
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                scores = torch.tensor([[4, 2, 1], [2, 3, 1], [1, 1, 4], [1, 2, 3]],
                                      dtype=dtype, requires_grad=True)
                indices = torch.tensor([[0, 1], [0, 1], [0, 2], [1, 2]])
                ids = torch.tensor([0, 1, 1, 2])
                mask = torch.tensor([True, True, True, False])
                actual = sequence_load_balancing_loss(scores, indices, ids, 3, mask)
                expected = reference_loss(scores, indices, ids, 3, mask)
                torch.testing.assert_close(actual, expected)
                grad = torch.autograd.grad(actual, scores, retain_graph=True)[0]
                torch.testing.assert_close(grad, torch.autograd.grad(expected, scores)[0])
                self.assertNotEqual(grad[0, 2].item(), 0)
                torch.testing.assert_close(grad[3], torch.zeros_like(grad[3]))
                self.assertEqual(actual.dtype, torch.float32)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_empty_and_fully_masked_samples(self) -> None:
        """
        Feature: Auxiliary-loss training lifecycle.
        Description: Empty shards preserve autograd and still enter all sequence collectives.
        Expectation: Every rank participates in reductions and the resulting zero remains differentiable.
        """
        for tokens in (0, 3):
            scores = torch.ones(tokens, 4, requires_grad=True)
            indices = torch.zeros(tokens, 2, dtype=torch.long)
            ids = torch.zeros(tokens, dtype=torch.long)
            group = object()
            module = "hyper_parallel.components.losses.moe_aux_loss"
            with patch(f"{module}.dist.all_reduce") as reduce_counts, patch(
                    f"{module}.differentiable_all_reduce", side_effect=lambda tensor, *args: tensor,
            ) as reduce_probs:
                loss = sequence_load_balancing_loss(scores, indices, ids, 2,
                                                    torch.zeros(tokens, dtype=torch.bool), (group,))
                loss.backward()
            torch.testing.assert_close(loss, torch.tensor(0.0))
            torch.testing.assert_close(scores.grad, torch.zeros_like(scores))
            reduce_counts.assert_called_once()
            reduce_probs.assert_called_once()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_partitioned_statistics_match_full_sequences(self) -> None:
        """
        Feature: Auxiliary-loss training lifecycle.
        Description: SUM backward plus replica averaging recovers the full-sample gradient.
        Expectation: Rank-average gradients equal the full-sequence reference.
        """
        values = torch.tensor([[5., 2., 1.], [3., 1., 2.], [1., 4., 2.]])
        indices = torch.tensor([[0], [0], [1]])
        ids = torch.tensor([0, 1, 1])
        full = values.clone().requires_grad_()
        expected = reference_loss(full, indices, ids, 2)
        expected.backward()
        for local_slice, remote_slice in ((slice(0, 1), slice(1, 3)), (slice(1, 3), slice(0, 1))):
            local = values[local_slice].clone().requires_grad_()
            remote_ids = ids[remote_slice]
            remote_counts = torch.zeros(2, 3).index_add(
                0, remote_ids, functional.one_hot(indices[remote_slice, 0], 3).float(),  # pylint: disable=not-callable
            )
            remote_sum = torch.zeros(2, 3).index_add(
                0, remote_ids, functional.normalize(values[remote_slice], p=1, dim=-1),
            )
            module = "hyper_parallel.components.losses.moe_aux_loss"
            with (
                patch(f"{module}.dist.all_reduce",
                      side_effect=lambda tensor, counts=remote_counts, **kwargs: tensor.add_(counts)),
                patch(f"{module}.differentiable_all_reduce",
                      side_effect=lambda tensor, *args, probabilities=remote_sum: (
                          2 * tensor + probabilities - tensor.detach()
                      )),
            ):
                actual = sequence_load_balancing_loss(local, indices[local_slice], ids[local_slice], 2,
                                                     sequence_partition_groups=(object(),))
                actual.backward()
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(local.grad / 2, full.grad[local_slice])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_invalid_metadata(self) -> None:
        """
        Feature: Auxiliary-loss training lifecycle.
        Description: Malformed sample IDs and masks fail before communication.
        Expectation: Invalid IDs, shapes, top-k or masks raise ValueError.
        """
        scores = torch.ones(2, 3)
        indices = torch.zeros(2, 1, dtype=torch.long)
        for ids, samples, mask in ((torch.zeros(2), 1, None), (torch.zeros(2).long(), 0, None),
                                   (torch.zeros(2).long(), 1, torch.ones(2))):
            with self.assertRaises(ValueError):
                sequence_load_balancing_loss(scores, indices, ids, samples, mask)


class TestModalityLoad(unittest.TestCase):
    """Keep V4.1 modality counts separate from standard router state."""

    def setUp(self) -> None:
        """Keep checkpoint execution and RNG local to these CPU cases."""
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        self.addCleanup(DefaultDeviceType.set_device_type, DefaultDeviceType.get_device_type())
        DefaultDeviceType.set_device_type("cpu")
        torch.manual_seed(12)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_modality_loads_match_dense_reference_with_checkpoint(self) -> None:
        """Feature: V4.1 auxiliary-loss-free load accounting.
        Description: Count masked text/image routes over two backwards with either checkpoint mode.
        Expectation: Counts match a dense reference exactly and routing weights/gradients are unchanged.
        """
        indices = torch.tensor([[0, 2], [1, 3], [0, 1]], dtype=torch.int32)
        assignments = functional.one_hot(indices.long(), 4).sum(1)  # pylint: disable=not-callable
        for image_mask in (None, torch.tensor([[False, True, True]])):
            for valid in (torch.tensor([[True, True, False]]), torch.zeros(1, 3, dtype=torch.bool)):
                for replay in (None, False, True):
                    with self.subTest(image_mask=image_mask, valid=valid, replay=replay):
                        counts = torch.zeros(2, 4, dtype=torch.int64)

                        def forward(weights: torch.Tensor, modality_mask=image_mask, token_mask=valid,
                                    load_counts=counts) -> torch.Tensor:
                            """Attach model-owned modality masks to the common backward accountant."""
                            return accumulate_router_load(weights, indices, modality_mask, token_mask, load_counts)

                        for _ in range(2):
                            weights = torch.randn(3, 2, requires_grad=True)
                            upstream = torch.randn_like(weights)
                            output = forward(weights) if replay is None else checkpoint(
                                forward, weights, use_reentrant=replay,
                            )
                            torch.testing.assert_close(output, weights, rtol=0, atol=0)
                            (output * upstream).sum().backward()
                            torch.testing.assert_close(weights.grad, upstream, rtol=0, atol=0)
                        image = torch.zeros(3, dtype=torch.bool) if image_mask is None else image_mask.flatten()
                        text = assignments[valid.flatten() & ~image].sum(0)
                        vision = assignments[valid.flatten() & image].sum(0)
                        torch.testing.assert_close(counts, 2 * torch.stack((text, vision)), rtol=0, atol=0)


class TestModalityBias(unittest.TestCase):
    """Preserve exact load comparisons when updating expert biases."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_adjacent_large_counts_preserve_update_direction(self) -> None:
        """
        Feature: Auxiliary-loss-free expert-bias updates.
        Description: Adjacent counts at the FP32 integer boundary must remain distinguishable.
        Expectation: The overloaded expert decreases its bias and the underloaded expert increases it.
        """
        counts = torch.tensor([2**24 + 1, 2**24], dtype=torch.int64)
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                bias = torch.zeros(2, dtype=dtype)
                update_router_bias(bias, counts, 0.001)
                torch.testing.assert_close(bias, torch.tensor([-0.001, 0.001], dtype=dtype))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_fractional_mean_preserves_update_direction(self) -> None:
        """
        Feature: Auxiliary-loss-free expert-bias updates.
        Description: Individually exact FP32 counts can still have a mean that rounds away their imbalance.
        Expectation: All 255 underloaded experts increase their biases despite the tiny mean difference.
        """
        counts = torch.tensor([2000001] + [2000000] * 255, dtype=torch.int64)
        bias = torch.zeros(256)
        update_router_bias(bias, counts, 0.001)
        torch.testing.assert_close(bias, torch.tensor([-0.001] + [0.001] * 255))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_modality_bias_uses_independent_loads(self) -> None:
        """Feature: V4.1 modality-specific expert-bias updates.
        Description: Text and image prefer opposite experts, including absent image state or loads.
        Expectation: Each bias uses its own load mean; absent modalities stay unchanged and counts are retained.
        """
        for image_load in ([0, 8], [0, 0]):
            for vision_enabled in (False, True):
                with self.subTest(image_load=image_load, vision_enabled=vision_enabled):
                    bias = torch.zeros(2)
                    vision_bias = torch.zeros(2) if vision_enabled else None
                    counts = torch.tensor([[8, 0], image_load], dtype=torch.int64)
                    original_counts = counts.clone()
                    update_modality_bias(bias, counts, .001, vision_bias=vision_bias)
                    torch.testing.assert_close(bias, torch.tensor([-.001, .001]), rtol=0, atol=0)
                    if vision_bias is not None:
                        expected = torch.tensor([.001, -.001]) if image_load[1] else torch.zeros(2)
                        torch.testing.assert_close(vision_bias, expected, rtol=0, atol=0)
                    torch.testing.assert_close(counts, original_counts, rtol=0, atol=0)


class TestV41IndexerAuxLossScaling(unittest.TestCase):
    """Verify the V4.1 Indexer consumer of the shared injection mechanism."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_compressed_indexer_kl_obeys_scale_and_detaches_teacher(self) -> None:
        """
        Feature: Auxiliary-loss training lifecycle.
        Description: The real V4.1 KL primitive uses the context's multiplier.
        Expectation: Indexer gradients follow the loss multiplier; teacher gradients remain absent.
        """
        torch.manual_seed(91)
        inputs = [torch.rand(shape, requires_grad=True) for shape in ((1, 4, 2, 3), (1, 3, 3), (1, 4, 2))]
        teacher = [torch.rand(shape, requires_grad=True) for shape in ((1, 2, 4, 3), (1, 3, 3), (2,))]
        indices = torch.tensor([[[0, 1], [0, 2], [1, 2], [-1, -1]]])
        def compute_loss() -> torch.Tensor:
            """Use the original production KL without replacing its backward."""
            return shared_compressed_indexer_kl_loss(
                *inputs, teacher[0], teacher[1], indices, teacher[2], attention_scale=0.5, loss_coeff=0.1,
            )

        expected_grads = torch.autograd.grad(compute_loss(), inputs)
        with aux_loss_scale_context():
            carrier = aux_loss_auto_scale(torch.zeros((), requires_grad=True), compute_loss())
            bind_aux_loss_scale(carrier)
            (carrier * 0.25).backward()
        for tensor, expected in zip(inputs, expected_grads):
            torch.testing.assert_close(tensor.grad, expected * 0.25)
        for tensor in teacher:
            self.assertIsNone(tensor.grad)

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
"""PyTorch distributed implementation of loss_parallel accuracy tests.

This file is executed by torchrun.
Tests compare:
  - Single-card reference (no parallelism)
  - Multi-card with loss_parallel (TP=2, vocab sharded)
"""
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel import init_device_mesh  # pylint: disable=C0413
from hyper_parallel.core.dtensor.dtensor import DTensor  # pylint: disable=C0413
from hyper_parallel.core.dtensor.placement_types import Shard  # pylint: disable=C0413
from hyper_parallel.core.tensor_parallel import loss_parallel, is_loss_parallel_active  # pylint: disable=C0413


np.random.seed(42)

_BATCH_SIZE = 2
_SEQ_LEN = 4
_VOCAB_SIZE = 16
_HIDDEN_SIZE = 8


def setup_module():
    """Initialize distributed backend."""
    dist.init_process_group(backend="gloo")


def teardown_module():
    """Cleanup distributed backend."""
    dist.destroy_process_group()


def _simple_linear_layer_torch(x: torch.Tensor, weight: torch.Tensor,
                                bias: torch.Tensor = None) -> torch.Tensor:
    """Simple linear transformation.

    Args:
        x: Input tensor of shape [..., in_features]
        weight: Weight tensor of shape [out_features, in_features]
        bias: Optional bias tensor of shape [out_features]

    Returns:
        output: Tensor of shape [..., out_features]
    """
    output = torch.matmul(x, weight.T)
    if bias is not None:
        output = output + bias
    return output


class TestLossParallelAccuracyPyTorch:
    """Accuracy tests for loss_parallel functionality (PyTorch backend)."""

    def test_single_vs_multi_card_loss_parity(self):
        """Compare single-card loss vs multi-card loss_parallel loss.

        Expected: loss_parallel should produce same numerical result as single-card,
        within floating-point tolerance.
        """
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        vocab_size = _VOCAB_SIZE * world_size

        np.random.seed(42)
        weight_np = np.random.randn(vocab_size, _HIDDEN_SIZE).astype(np.float32) * 0.1
        input_np = np.random.randn(_BATCH_SIZE * _SEQ_LEN, _HIDDEN_SIZE).astype(np.float32) * 0.1
        targets_np = np.random.randint(0, vocab_size, (_BATCH_SIZE * _SEQ_LEN,)).astype(np.int64)

        weight_single = torch.from_numpy(weight_np)
        input_single = torch.from_numpy(input_np)
        targets_single = torch.from_numpy(targets_np)

        logits_single = _simple_linear_layer_torch(input_single, weight_single)
        loss_single = F.cross_entropy(logits_single, targets_single, reduction='mean')

        mesh = init_device_mesh("cpu", (world_size,))

        weight_shard_np = weight_np[rank * _VOCAB_SIZE:(rank + 1) * _VOCAB_SIZE, :]
        weight_shard = torch.from_numpy(weight_shard_np)

        input_replicate = torch.from_numpy(input_np)
        targets_replicate = torch.from_numpy(targets_np)

        logits_shard = _simple_linear_layer_torch(input_replicate, weight_shard)

        logits_dtensor = DTensor.from_local(logits_shard, mesh, [Shard(-1)])

        with loss_parallel(mesh=mesh):
            assert is_loss_parallel_active(), "loss_parallel context should be active"

            loss_parallel_value = F.cross_entropy(logits_dtensor, targets_replicate, reduction='mean')

        rtol = 1e-3
        atol = 1e-5
        np.testing.assert_allclose(
            loss_single.item(),
            loss_parallel_value.item(),
            rtol=rtol,
            atol=atol,
            err_msg="loss_parallel loss does not match single-card reference"
        )

        print(f"[Rank {rank}] Single-card loss: {loss_single.item():.6f}")
        print(f"[Rank {rank}] Multi-card loss_parallel loss: {loss_parallel_value.item():.6f}")
        print(f"[Rank {rank}] Absolute difference: {abs(loss_single.item() - loss_parallel_value.item()):.6e}")

    def test_loss_parallel_context_correctness(self):
        """Verify loss_parallel context manager works correctly.

        Expected: Context should be active inside with block, inactive outside.
        """
        assert is_loss_parallel_active() is False, "Should be inactive before context"

        with loss_parallel():
            assert is_loss_parallel_active() is True, "Should be active inside context"

        assert is_loss_parallel_active() is False, "Should be inactive after context"

        with loss_parallel():
            assert is_loss_parallel_active() is True
            with loss_parallel():
                assert is_loss_parallel_active() is True
            assert is_loss_parallel_active() is True

        assert is_loss_parallel_active() is False

        mesh = init_device_mesh("cpu", (dist.get_world_size(),))
        with loss_parallel(mesh=mesh, strict=True):
            assert is_loss_parallel_active() is True

        print(f"[Rank {dist.get_rank()}] Context manager tests passed")

    def _reference_gradients(self, input_np, weight_np, targets_np, reduction, weight, ignore_index):
        """Compute single-card reference gradients on the full (un-sharded) logits.

        Args:
            input_np: Input of shape [N, H].
            weight_np: Full vocab weight of shape [V, H].
            targets_np: Target class indices of shape [N].
            reduction: 'mean', 'sum' or 'none'.
            weight: Optional class weights of shape [V].
            ignore_index: Index to ignore.

        Returns:
            Tuple of (weight_grad, input_grad) from the reference model.
        """
        input_ref = torch.from_numpy(input_np).requires_grad_(True)
        weight_ref = torch.from_numpy(weight_np).requires_grad_(True)
        targets_ref = torch.from_numpy(targets_np)

        logits_ref = _simple_linear_layer_torch(input_ref, weight_ref)
        loss_ref = F.cross_entropy(
            logits_ref,
            targets_ref,
            weight=weight,
            ignore_index=ignore_index,
            reduction=reduction,
        )
        if reduction == "none":
            loss_ref.backward(torch.ones_like(loss_ref))
        else:
            loss_ref.backward()
        return weight_ref.grad, input_ref.grad

    def _distributed_gradients(self, input_np, weight_shard_np, targets_np, reduction, weight, ignore_index):
        """Compute distributed loss_parallel gradients on this rank's vocab shard.

        Args:
            input_np: Replicated input of shape [N, H].
            weight_shard_np: This rank's vocab shard of shape [V/world, H].
            targets_np: Target class indices of shape [N].
            reduction: 'mean', 'sum' or 'none'.
            weight: Optional class weights of shape [V].
            ignore_index: Index to ignore.

        Returns:
            Tuple of (weight_shard_grad, input_grad) from the loss_parallel model.
        """
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        mesh = init_device_mesh("cpu", (world_size,))

        weight_shard = torch.from_numpy(weight_shard_np.copy()).requires_grad_(True)
        input_shard = torch.from_numpy(input_np.copy()).requires_grad_(True)
        targets_shard = torch.from_numpy(targets_np.copy())

        def forward_with_loss_parallel():
            logits_shard = _simple_linear_layer_torch(input_shard, weight_shard)
            logits_dtensor = DTensor.from_local(logits_shard, mesh, [Shard(-1)])

            with loss_parallel(mesh=mesh):
                return F.cross_entropy(
                    logits_dtensor,
                    targets_shard,
                    weight=weight,
                    ignore_index=ignore_index,
                    reduction=reduction,
                )

        loss = forward_with_loss_parallel()
        if reduction == "none":
            # reduction="none" returns each rank's local loss vector; sum across
            # ranks to reconstruct the full loss before backprop.
            full_loss = loss.clone()
            dist.all_reduce(full_loss)
            full_loss.backward(torch.ones_like(full_loss))
        else:
            loss.backward()
        return weight_shard.grad, input_shard.grad

    def _assert_gradients_match(self, label, input_np, weight_np, weight_shard_np, targets_np,
                                reduction, weight, ignore_index):
        """Assert distributed gradients match the single-card reference on every rank."""
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        weight_ref_grad, input_ref_grad = self._reference_gradients(
            input_np, weight_np, targets_np, reduction, weight, ignore_index
        )
        weight_dist_grad, input_dist_grad = self._distributed_gradients(
            input_np, weight_shard_np, targets_np, reduction, weight, ignore_index
        )

        # The weight gradient is sharded by vocab: compare this rank's slice.
        shard_start = rank * _VOCAB_SIZE
        expected_weight = weight_ref_grad[shard_start:shard_start + _VOCAB_SIZE, :]
        np.testing.assert_allclose(
            weight_dist_grad.numpy(),
            expected_weight.numpy(),
            rtol=1e-3,
            atol=1e-5,
            err_msg=f"{label}: weight gradient mismatch on rank {rank}",
        )

        # The input is replicated; all-reduce the per-rank partial gradients to
        # reconstruct the full input gradient before comparing with the reference.
        input_full_grad = input_dist_grad.clone()
        dist.all_reduce(input_full_grad)
        np.testing.assert_allclose(
            input_full_grad.numpy(),
            input_ref_grad.numpy(),
            rtol=1e-3,
            atol=1e-5,
            err_msg=f"{label}: input gradient mismatch on rank {rank}",
        )
        print(f"[Rank {rank}] {label}: gradients match single-card reference", flush=True)

    def test_gradient_correctness_with_loss_parallel(self):
        """Verify gradients from the loss_parallel path match single-card references.

        Expected: The loss_parallel backward must reproduce the reference gradients
        (softmax - one_hot, scaled by weight/total_weight) for every rank's vocab
        shard, across mean/sum/none reductions, optional class weights and
        ignore_index.
        """
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        vocab_size = _VOCAB_SIZE * world_size
        batch_size = _BATCH_SIZE * _SEQ_LEN

        np.random.seed(123)
        weight_np = np.random.randn(vocab_size, _HIDDEN_SIZE).astype(np.float32) * 0.1
        input_np = np.random.randn(batch_size, _HIDDEN_SIZE).astype(np.float32) * 0.1
        targets_np = np.random.randint(0, vocab_size, (batch_size,)).astype(np.int64)

        weight_shard_np = weight_np[rank * _VOCAB_SIZE:(rank + 1) * _VOCAB_SIZE, :]

        # Force some targets to be ignored so the ignore path is exercised.
        ignored_targets = targets_np.copy()
        ignored_targets[0] = -100
        ignored_targets[1] = -100

        # Derived from the seeded numpy RNG so the weight vector is identical on
        # every rank (the distributed loss mixes per-rank shards).
        class_weight = torch.from_numpy(np.random.rand(vocab_size).astype(np.float32) + 0.5)

        self._assert_gradients_match(
            "mean", input_np, weight_np, weight_shard_np, targets_np,
            reduction="mean", weight=None, ignore_index=-100,
        )
        self._assert_gradients_match(
            "sum", input_np, weight_np, weight_shard_np, targets_np,
            reduction="sum", weight=None, ignore_index=-100,
        )
        self._assert_gradients_match(
            "none", input_np, weight_np, weight_shard_np, targets_np,
            reduction="none", weight=None, ignore_index=-100,
        )
        self._assert_gradients_match(
            "mean+weight", input_np, weight_np, weight_shard_np, targets_np,
            reduction="mean", weight=class_weight, ignore_index=-100,
        )
        self._assert_gradients_match(
            "sum+weight", input_np, weight_np, weight_shard_np, targets_np,
            reduction="sum", weight=class_weight, ignore_index=-100,
        )
        self._assert_gradients_match(
            "mean+ignore", input_np, weight_np, weight_shard_np, ignored_targets,
            reduction="mean", weight=None, ignore_index=-100,
        )
        self._assert_gradients_match(
            "mean+weight+ignore", input_np, weight_np, weight_shard_np, ignored_targets,
            reduction="mean", weight=class_weight, ignore_index=-100,
        )

        print(f"[Rank {rank}] Gradient test passed (7 configurations)")


def test_single_vs_multi_card_loss_parity():
    """Wrapper for pytest."""
    TestLossParallelAccuracyPyTorch().test_single_vs_multi_card_loss_parity()


def test_loss_parallel_context_correctness():
    """Wrapper for pytest."""
    TestLossParallelAccuracyPyTorch().test_loss_parallel_context_correctness()


def test_gradient_correctness_with_loss_parallel():
    """Wrapper for pytest."""
    TestLossParallelAccuracyPyTorch().test_gradient_correctness_with_loss_parallel()


if __name__ == "__main__":
    setup_module()

    print("=" * 80)
    print("Test 1: Single vs Multi-card Loss Parity (PyTorch)")
    print("=" * 80)
    test_single_vs_multi_card_loss_parity()
    print("PASS\n")

    print("=" * 80)
    print("Test 2: Loss Parallel Context Correctness (PyTorch)")
    print("=" * 80)
    test_loss_parallel_context_correctness()
    print("PASS\n")

    print("=" * 80)
    print("Test 3: Gradient Correctness with Loss Parallel (PyTorch)")
    print("=" * 80)
    test_gradient_correctness_with_loss_parallel()
    print("PASS\n")

    print("=" * 80)
    print("All PyTorch tests passed!")
    print("=" * 80)

    teardown_module()

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
"""PyTorch reference implementation for loss_parallel accuracy comparison.

This file generates reference values using PyTorch that can be compared
against MindSpore implementation.

Usage:
    python _pytorch_reference.py
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel import init_device_mesh  # pylint: disable=C0413
from hyper_parallel.core.dtensor.dtensor import DTensor  # pylint: disable=C0413
from hyper_parallel.core.dtensor.layout import Layout  # pylint: disable=C0413
from hyper_parallel.core.dtensor.placement_types import Shard  # pylint: disable=C0413
from hyper_parallel.core.tensor_parallel import loss_parallel  # pylint: disable=C0413


np.random.seed(42)

_BATCH_SIZE = 2
_SEQ_LEN = 4
_VOCAB_SIZE = 16
_HIDDEN_SIZE = 8
_TP_SIZE = 4


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


def generate_reference_data() -> tuple:
    """Generate reference data and save to numpy files.

    Returns:
        Tuple of (weight, input, targets) numpy arrays.
    """
    vocab_size = _VOCAB_SIZE * _TP_SIZE

    np.random.seed(42)
    weight_np = np.random.randn(vocab_size, _HIDDEN_SIZE).astype(np.float32) * 0.1
    input_np = np.random.randn(_BATCH_SIZE * _SEQ_LEN, _HIDDEN_SIZE).astype(np.float32) * 0.1
    targets_np = np.random.randint(0, vocab_size, (_BATCH_SIZE * _SEQ_LEN,)).astype(np.int64)

    return weight_np, input_np, targets_np


def pytorch_single_card_reference() -> dict:
    """Generate PyTorch single-card reference.

    Returns:
        Dictionary containing weight, input, targets, loss, and logits.
    """
    weight_np, input_np, targets_np = generate_reference_data()

    weight_torch = torch.from_numpy(weight_np)
    input_torch = torch.from_numpy(input_np)
    targets_torch = torch.from_numpy(targets_np)

    logits_torch = _simple_linear_layer_torch(input_torch, weight_torch)
    loss_torch = F.cross_entropy(logits_torch, targets_torch, reduction='mean')

    print("=" * 80)
    print("PyTorch Single-Card Reference")
    print("=" * 80)
    print(f"Logits shape: {logits_torch.shape}")
    print(f"Targets shape: {targets_torch.shape}")
    print(f"Loss: {loss_torch.item():.6f}")
    print()

    return {
        'weight': weight_np,
        'input': input_np,
        'targets': targets_np,
        'loss': loss_torch.item(),
        'logits': logits_torch.numpy(),
    }


def pytorch_distributed_reference() -> dict:
    """Generate PyTorch distributed reference with loss_parallel.

    Returns:
        Dictionary containing rank, world_size, weight_shard, input, targets, and loss.
        Returns None if distributed is not initialized.
    """
    if not torch.distributed.is_initialized():
        print("Distributed not initialized, running in single-process mode")
        return None

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    weight_np, input_np, targets_np = generate_reference_data()

    input_torch = torch.from_numpy(input_np)
    targets_torch = torch.from_numpy(targets_np)

    mesh = init_device_mesh("cuda", (world_size,))

    weight_shard_np = weight_np[rank * _VOCAB_SIZE:(rank + 1) * _VOCAB_SIZE, :]
    weight_shard = torch.from_numpy(weight_shard_np)

    logits_shard = _simple_linear_layer_torch(input_torch, weight_shard)

    logits_dtensor = DTensor(
        local_tensor=logits_shard,
        layout=Layout(mesh, (Shard(-1),)),
    )

    with loss_parallel(mesh=mesh):
        loss_dist = F.cross_entropy(logits_dtensor, targets_torch, reduction='mean')

    print("=" * 80)
    print(f"PyTorch Distributed Reference (Rank {rank})")
    print("=" * 80)
    print(f"Logits shape (sharded): {logits_shard.shape}")
    print(f"Targets shape: {targets_torch.shape}")
    print(f"Loss: {loss_dist.item():.6f}")
    print()

    return {
        'rank': rank,
        'world_size': world_size,
        'weight_shard': weight_shard_np,
        'input': input_np,
        'targets': targets_np,
        'loss': loss_dist.item(),
    }


if __name__ == "__main__":
    ref_data = pytorch_single_card_reference()

    output_file = "loss_parallel_reference.npz"
    np.savez(
        output_file,
        weight=ref_data['weight'],
        input=ref_data['input'],
        targets=ref_data['targets'],
        loss=ref_data['loss'],
        logits=ref_data['logits'],
    )
    print(f"Reference data saved to {output_file}")
    print("=" * 80)

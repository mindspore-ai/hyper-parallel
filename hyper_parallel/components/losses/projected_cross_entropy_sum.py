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
"""Chunked cross-entropy through a caller-owned output projection boundary."""

from __future__ import annotations

from functools import partial

# The output projection and its checkpointed first-order gradient use Torch autograd.
# pylint: disable=forbidden-backend-import
import torch
from torch import nn
from torch.func import functional_call
from torch.nn import functional
from torch.utils.checkpoint import checkpoint

from hyper_parallel.components.losses.chunked_cross_entropy import (
    _validate_chunk_dtypes_and_device,
    _validate_chunk_options,
    _validate_chunk_shapes,
)


def _projected_chunk_loss(
    hidden: torch.Tensor,
    weight_master: torch.Tensor,
    targets: torch.Tensor,
    *,
    projection: nn.Module,
    vocab_size: int,
    ignore_index: int,
) -> torch.Tensor:
    """Evaluate one chunk through the module's configured parallel boundary."""
    logits = functional_call(projection, {"weight": weight_master.to(hidden.dtype)}, (hidden,))
    if not isinstance(logits, torch.Tensor) or getattr(logits, "placements", None) is not None:
        raise TypeError("projection must return a local Tensor containing the full vocabulary")
    if logits.shape != (*hidden.shape[:2], vocab_size):
        raise ValueError(
            "projection must return full-vocabulary logits aligned with the local hidden states; "
            f"expected {(*hidden.shape[:2], vocab_size)}, got {tuple(logits.shape)}"
        )
    return functional.cross_entropy(
        logits.reshape(-1, vocab_size).float(), targets.reshape(-1),
        ignore_index=ignore_index, reduction="sum",
    )


def projected_cross_entropy_sum(
    hidden_states: torch.Tensor,
    targets: torch.Tensor,
    projection: nn.Module,
    *,
    vocab_size: int,
    chunk_size: int = 1024,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Return a summed CE while preserving the projection module's forward boundary.

    This path supports a bias-free Linear whose parallel wrapper gathers each
    chunk's vocabulary logits. Its weight may remain a local vocabulary shard.
    The weight must be a local Tensor, as exposed inside source-layout FSDP.
    A DTensor weight requires a distributed-autograd integration and is rejected.
    The wrapper owns communication and the surrounding training runtime owns
    replica-gradient normalization. No additional loss or gradient scaling is
    introduced here. A loss-parallel, class-sharded output is not supported.

    Args:
        hidden_states: Local hidden states in [batch, sequence, hidden] layout.
        targets: Global vocabulary indices aligned with the local positions.
        projection: Bias-free Linear with its caller-configured forward wrapper.
        vocab_size: Global class count, independently of local weight metadata.
        chunk_size: Maximum local sequence length evaluated per checkpoint.
        ignore_index: Target value excluded from the summed cross-entropy.

    Returns:
        A scalar FP32 loss sum connected to the original hidden states and weight.

    Raises:
        TypeError: If scalar options, tensors, or the projection are unsupported.
        ValueError: If tensor dimensions or projection output violate the contract.
    """
    _validate_chunk_options(chunk_size, ignore_index)
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
        raise TypeError("vocab_size must be an integer")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if not isinstance(projection, nn.Linear) or projection.bias is not None:
        raise TypeError("projection must be a bias-free Linear")
    weight = projection.weight
    if getattr(weight, "placements", None) is not None:
        raise NotImplementedError("projected_cross_entropy_sum requires a local unsharded projection weight")
    _validate_chunk_shapes(hidden_states, targets, weight)
    _validate_chunk_dtypes_and_device(hidden_states, targets, weight)
    if weight.shape[0] > vocab_size:
        raise ValueError("local projection weight exceeds the global vocabulary")

    # Per-chunk low-precision weight gradients accumulate in FP32 before one
    # cast at the original weight boundary.
    master_dtype = torch.float32 if weight.dtype in (torch.float16, torch.bfloat16) else weight.dtype
    weight_master = weight.to(master_dtype)
    project_loss = partial(
        _projected_chunk_loss, projection=projection, vocab_size=vocab_size, ignore_index=ignore_index,
    )
    loss_sum = hidden_states.new_zeros((), dtype=torch.float32)
    for hidden_chunk, target_chunk in zip(
        hidden_states.split(chunk_size, dim=1), targets.split(chunk_size, dim=1), strict=True,
    ):
        loss_sum = loss_sum + checkpoint(
            project_loss, hidden_chunk, weight_master, target_chunk, use_reentrant=False,
        )
    return loss_sum

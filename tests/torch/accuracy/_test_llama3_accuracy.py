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
"""Llama3 accuracy comparison: parallel scenarios vs single-card baseline.

The harness compares **every step's** training loss of a parallelized Llama3
demo against an in-process single-card reference. Three cases are exercised:

* :func:`test_single_card_baseline` runs a single-rank reference loss trajectory.
* :func:`test_tp_fully_shard_matches_single_card` covers ``TP + FSDP`` on a
  ``(dp=2, tp=2)`` mesh (4 ranks).
* :func:`test_tp_cp_fully_shard_matches_single_card` covers ``TP + CP + FSDP``
  on a ``(dp=2, cp=2, tp=2)`` mesh (8 ranks) with Colossal-style context
  parallel attached to every BSHD SDPA core.

Strict per-step alignment relies on these properties:

1. **Identical initialization** — all ranks build the model from the same seed
   and rank 0 broadcasts parameters/buffers so weights start identical.
2. **Identical global batch** — the ``(tokens, targets)`` mini-batch is built
   from the same seed and broadcast from rank 0.
3. **``reduction="sum"`` cross-entropy** — the partial loss on each rank is a
   sum over its own ``(B/dp, S/cp)`` token slice; summing across the ``(dp, cp)``
   plane recovers the single-card sum-loss over the full ``(B, S)`` batch.
4. **``set_reduce_op_type("sum")``** — FSDP gradient reduction is configured
   as SUM, matching how a sum-loss backward accumulates per-rank partial
   gradients into a single global gradient.
5. **TP backward normalization** — the per-rank scalar loss is replicated
   across the TP submesh, so backward through TP would otherwise count the
   same gradient ``tp_size`` times. We pass ``1.0 / tp_size`` as the gradient
   seed to ``loss.backward`` (mirroring ``_test_tp_fully_shard_e2e.py``).
6. **CP+RoPE alignment** — each CP rank passes
   ``rope_seq_start = cp_rank * (S / cp_size)`` to ``Llama3Model.forward`` so
   RoPE positions match the global token positions of its sequence slice.
"""
# pylint: disable=W0611,C0413,C0412,W0613,W0612
from __future__ import annotations

import os

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu
from torch import optim

from hyper_parallel import (
    ContextParallel,
    SkipDTensorDispatch,
    fully_shard,
    init_device_mesh,
)

from tests.torch.accuracy.model import Llama3DemoConfig, Llama3Model
from tests.torch.accuracy.parallelize import broadcast_state_dict_from_rank0, parallelize_llama3
from tests.torch.utils import init_dist


_BATCH_SIZE = 4
_SEQ_LEN = 16
_STEPS = 10
_LR = 1e-4
_INIT_SEED = 1234
_DATA_SEED = 2026
_RTOL = 1e-3
_ATOL = 1e-3


def _build_config() -> Llama3DemoConfig:
    """Return the small Llama3 config used across the accuracy cases."""
    return Llama3DemoConfig(
        dim=256,
        n_layers=2,
        n_heads=8,
        n_kv_heads=4,
        vocab_size=1024,
        max_seq_len=128,
    )


def _build_model(cfg: Llama3DemoConfig, device: torch.device) -> Llama3Model:
    """Construct an ``Llama3Model`` deterministically from a fixed seed on ``device``."""
    torch.manual_seed(_INIT_SEED)
    model = Llama3Model(cfg).to(device=device)
    return model


def _build_global_batch(
    cfg: Llama3DemoConfig, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a deterministic ``(tokens, targets)`` global batch on ``device``."""
    torch.manual_seed(_DATA_SEED)
    tokens = torch.randint(0, cfg.vocab_size, (_BATCH_SIZE, _SEQ_LEN), device=device)
    targets = torch.randint(0, cfg.vocab_size, (_BATCH_SIZE, _SEQ_LEN), device=device)
    return tokens, targets


def _broadcast_batch(tokens: torch.Tensor, targets: torch.Tensor) -> None:
    """Broadcast ``tokens``/``targets`` from rank 0 so every rank trains on identical data."""
    if dist.is_initialized():
        dist.broadcast(tokens, src=0)
        dist.broadcast(targets, src=0)


def _sum_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, vocab_size: int
) -> torch.Tensor:
    """Return ``F.cross_entropy(logits, targets, reduction="sum")`` cast to float32."""
    return F.cross_entropy(
        logits.float().reshape(-1, vocab_size),
        targets.reshape(-1),
        reduction="sum",
    )


def _run_single_card_training(
    cfg: Llama3DemoConfig,
    device: torch.device,
    tokens: torch.Tensor,
    targets: torch.Tensor,
) -> List[float]:
    """Train an unparallelized model with sum-CE and return the per-step global sum-loss.

    Builds a fresh model from :data:`_INIT_SEED` so it can be called multiple times in one
    process and stay deterministic.
    """
    model = _build_model(cfg, device)
    optimizer = optim.SGD(model.parameters(), lr=_LR)
    losses: List[float] = []
    for _ in range(_STEPS):
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens)
        loss = _sum_cross_entropy(logits, targets, cfg.vocab_size)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().float().cpu().item())
    return losses


def _build_distributed_model(cfg: Llama3DemoConfig, device: torch.device) -> Llama3Model:
    """Build the distributed model and broadcast rank-0 weights so every rank starts identically."""
    model = _build_model(cfg, device)
    broadcast_state_dict_from_rank0(model)
    return model


def _all_reduce_sum_scalar(
    scalar: torch.Tensor, groups: List[Optional[dist.ProcessGroup]]
) -> torch.Tensor:
    """Sum-reduce ``scalar`` across each ``group`` in turn.

    Reducing on independent groups one after the other is equivalent to reducing on a single
    group whose ranks are the cartesian product of those axes (used to collapse partial losses
    over both DP and CP).
    """
    out = scalar.detach().clone()
    for group in groups:
        if group is None:
            continue
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)
    return out


def _assert_loss_trajectory_matches(
    case_name: str,
    rank: int,
    distributed_losses: List[float],
    baseline_losses: List[float],
) -> None:
    """Strictly compare every step's distributed global loss against the single-card baseline."""
    assert len(distributed_losses) == len(baseline_losses), (
        f"{case_name}: step count mismatch (dist={len(distributed_losses)}, "
        f"baseline={len(baseline_losses)})."
    )
    for step_idx, (dist_loss, ref_loss) in enumerate(zip(distributed_losses, baseline_losses)):
        if not np.isfinite(dist_loss):
            raise AssertionError(
                f"{case_name}: rank {rank} step {step_idx} produced non-finite loss {dist_loss}."
            )
        if not np.allclose(dist_loss, ref_loss, rtol=_RTOL, atol=_ATOL):
            raise AssertionError(
                f"{case_name}: rank {rank} step {step_idx} distributed loss "
                f"{dist_loss:.6f} != single-card baseline {ref_loss:.6f} "
                f"(rtol={_RTOL}, atol={_ATOL})."
            )


def test_single_card_baseline() -> None:
    """
    Feature: deterministic Llama3 single-card training reference.
    Description: Run :data:`_STEPS` SGD steps on the full batch with sum-CE and ensure every
        loss is finite, exercising the same in-process baseline used by the parallel cases.
    Expectation: Loss trajectory contains :data:`_STEPS` finite values.
    """
    rank, device_id = init_dist()
    if rank != 0:
        return
    device = torch.device("npu", device_id)
    cfg = _build_config()
    tokens, targets = _build_global_batch(cfg, device)
    losses = _run_single_card_training(cfg, device, tokens, targets)
    for step_idx, loss_val in enumerate(losses):
        assert np.isfinite(loss_val), (
            f"single_card_baseline: step {step_idx} produced non-finite loss {loss_val}."
        )
    print(
        f"[Rank {rank}] single_card_baseline ok, losses="
        f"{[round(loss_val, 6) for loss_val in losses]}"
    )


def test_tp_fully_shard_matches_single_card() -> None:
    """
    Feature: ``TP + fully_shard`` Llama3 accuracy vs single-card.
    Description:
        1. Build a 2-D ``(dp=2, tp=2)`` mesh on 4 NPU ranks.
        2. Apply ``parallelize_llama3`` on ``mesh["tp"]`` and ``fully_shard`` on ``mesh["dp"]``,
           setting ``reduce_op_type="sum"`` so the FSDP gradient reduction matches the implicit
           ``sum`` reduction of the ``reduction="sum"`` cross-entropy.
        3. Each DP rank trains on its own ``batch / dp_size`` rows (TP ranks share the same
           rows). The per-rank sum-loss is reduced across the DP group to recover the
           single-card sum-loss.
        4. Backward is seeded with ``1.0 / tp_size`` to undo TP-replication of the scalar loss.
    Expectation:
        Every step's reconstructed global loss equals the single-card baseline within
        ``rtol=1e-3``/``atol=1e-3``.
    """
    rank, device_id = init_dist()
    world_size = dist.get_world_size()
    if world_size != 4:
        if rank == 0:
            print(
                f"Skip tp_fully_shard_matches_single_card: requires world_size=4, got {world_size}."
            )
        return

    device = torch.device("npu", device_id)
    cfg = _build_config()
    tp_size = 2
    dp_size = world_size // tp_size

    if _BATCH_SIZE % dp_size != 0:
        raise AssertionError(
            f"_BATCH_SIZE={_BATCH_SIZE} must be divisible by dp_size={dp_size}."
        )

    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(dp_size, tp_size),
        mesh_dim_names=("dp", "tp"),
    )
    tp_mesh = mesh["tp"]
    dp_mesh = mesh["dp"]
    dp_rank = mesh.get_local_rank("dp")

    tokens, targets = _build_global_batch(cfg, device)
    _broadcast_batch(tokens, targets)

    baseline_losses = _run_single_card_training(cfg, device, tokens, targets)

    model = _build_distributed_model(cfg, device)
    parallelize_llama3(model, tp_mesh)
    for layer in model.layers:
        fully_shard(layer, mesh=dp_mesh)
    fully_shard(model, mesh=dp_mesh)
    model.set_reduce_op_type("sum")

    rows_per_dp = _BATCH_SIZE // dp_size
    tokens_dp = tokens[dp_rank * rows_per_dp : (dp_rank + 1) * rows_per_dp].contiguous()
    targets_dp = targets[dp_rank * rows_per_dp : (dp_rank + 1) * rows_per_dp].contiguous()

    optimizer = optim.SGD(model.parameters(), lr=_LR)
    distributed_losses: List[float] = []
    dp_group = dp_mesh.get_group()
    backward_seed = torch.tensor(1.0 / tp_size, device=device)
    for _ in range(_STEPS):
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens_dp)
        partial_loss = _sum_cross_entropy(logits, targets_dp, cfg.vocab_size)
        with SkipDTensorDispatch():
            partial_loss.backward(backward_seed)
            optimizer.step()
        global_loss = _all_reduce_sum_scalar(partial_loss.detach(), [dp_group])
        distributed_losses.append(global_loss.float().cpu().item())

    _assert_loss_trajectory_matches(
        "tp_fully_shard_matches_single_card", rank, distributed_losses, baseline_losses
    )

    if rank == 0:
        print(
            f"[Rank {rank}] tp_fully_shard_matches_single_card ok "
            f"(dp={dp_size}, tp={tp_size}), losses="
            f"{[round(loss_val, 6) for loss_val in distributed_losses]}"
        )


def test_tp_cp_fully_shard_matches_single_card() -> None:
    """
    Feature: ``TP + CP + fully_shard`` Llama3 accuracy vs single-card.
    Description:
        1. Build a 3-D ``(dp=2, cp=2, tp=2)`` mesh on 8 NPU ranks.
        2. Apply ``parallelize_llama3`` on ``mesh["tp"]``, attach Colossal
           ``ContextParallel`` (``ulysses_degree=1``) to every ``layer.attention.sdpa_core`` on
           ``mesh["cp"]``, and ``fully_shard`` the model on ``mesh["dp"]``.
        3. Each DP rank takes ``batch / dp_size`` rows; each CP rank then consumes the slice
           ``tokens[:, cp_rank * S/cp : (cp_rank+1) * S/cp]`` and ``Llama3Model.forward(...,
           rope_seq_start=cp_rank * S/cp)`` so RoPE matches the global window. With
           ``reduction="sum"`` and ``set_reduce_op_type("sum")``, summing partial losses across
           the ``(dp, cp)`` plane recovers the single-card sum-loss over the full ``(B, S)``
           batch.
        4. Backward is seeded with ``1.0 / tp_size`` to undo TP-replication of the scalar loss.
    Expectation:
        Every step's reconstructed global loss equals the single-card baseline within
        ``rtol=1e-3``/``atol=1e-3``.
    """
    rank, device_id = init_dist()
    world_size = dist.get_world_size()
    if world_size != 8:
        if rank == 0:
            print(
                f"Skip tp_cp_fully_shard_matches_single_card: requires world_size=8, "
                f"got {world_size}."
            )
        return

    device = torch.device("npu", device_id)
    cfg = _build_config()
    dp_size, cp_size, tp_size = 2, 2, 2

    if _BATCH_SIZE % dp_size != 0:
        raise AssertionError(
            f"_BATCH_SIZE={_BATCH_SIZE} must be divisible by dp_size={dp_size}."
        )
    if _SEQ_LEN % cp_size != 0:
        raise AssertionError(f"_SEQ_LEN={_SEQ_LEN} must be divisible by cp_size={cp_size}.")
    seq_per_cp = _SEQ_LEN // cp_size
    if seq_per_cp % tp_size != 0:
        raise AssertionError(
            f"(seq_len/cp)={seq_per_cp} must be divisible by tp_size={tp_size}."
        )

    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(dp_size, cp_size, tp_size),
        mesh_dim_names=("dp", "cp", "tp"),
    )
    dp_mesh = mesh["dp"]
    cp_mesh = mesh["cp"]
    tp_mesh = mesh["tp"]
    dp_rank = mesh.get_local_rank("dp")
    cp_rank = mesh.get_local_rank("cp")

    tokens, targets = _build_global_batch(cfg, device)
    _broadcast_batch(tokens, targets)

    baseline_losses = _run_single_card_training(cfg, device, tokens, targets)

    model = _build_distributed_model(cfg, device)
    parallelize_llama3(model, tp_mesh)

    cp_plan = ContextParallel(seq_dim=1, head_dim=2, ulysses_degree=1)
    for layer in model.layers:
        cp_plan.apply(layer.attention.sdpa_core, cp_mesh)

    for layer in model.layers:
        fully_shard(layer, mesh=dp_mesh)
    fully_shard(model, mesh=dp_mesh)
    model.set_reduce_op_type("sum")

    rows_per_dp = _BATCH_SIZE // dp_size
    tokens_dp = tokens[dp_rank * rows_per_dp : (dp_rank + 1) * rows_per_dp]
    targets_dp = targets[dp_rank * rows_per_dp : (dp_rank + 1) * rows_per_dp]
    tokens_dp_cp = tokens_dp[:, cp_rank * seq_per_cp : (cp_rank + 1) * seq_per_cp].contiguous()
    targets_dp_cp = targets_dp[:, cp_rank * seq_per_cp : (cp_rank + 1) * seq_per_cp].contiguous()
    rope_seq_start = cp_rank * seq_per_cp

    optimizer = optim.SGD(model.parameters(), lr=_LR)
    distributed_losses: List[float] = []
    reduce_groups = [dp_mesh.get_group(), cp_mesh.get_group()]
    backward_seed = torch.tensor(1.0 / tp_size, device=device)
    for _ in range(_STEPS):
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens_dp_cp, rope_seq_start=rope_seq_start)
        partial_loss = _sum_cross_entropy(logits, targets_dp_cp, cfg.vocab_size)
        with SkipDTensorDispatch():
            partial_loss.backward(backward_seed)
            optimizer.step()
        global_loss = _all_reduce_sum_scalar(partial_loss.detach(), reduce_groups)
        distributed_losses.append(global_loss.float().cpu().item())

    _assert_loss_trajectory_matches(
        "tp_cp_fully_shard_matches_single_card", rank, distributed_losses, baseline_losses
    )

    if rank == 0:
        print(
            f"[Rank {rank}] tp_cp_fully_shard_matches_single_card ok "
            f"(dp={dp_size}, cp={cp_size}, tp={tp_size}), losses="
            f"{[round(loss_val, 6) for loss_val in distributed_losses]}"
        )

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
"""Llama3 PP + FSDP + TP composite test (1F1B) vs full-model serial reference.

Mirrors ``tests/mindspore/st/pipeline_parallel/_test_pp_composite.py`` driver logic
(per-step loss parity against an in-process serial reference before ``optimizer.step()``)
and ``examples/torch/llama3`` Llama3 PP stage layout:

* **Distributed** — 3-D mesh ``(pp=2, fsdp=2, tp=2)`` on 8 ranks, ``Schedule1F1B``,
  ``parallelize_llama3`` on ``mesh["tp"]``, and nested ``fully_shard`` on every trainable
  submodule (``tok_embeddings``, decoder blocks, ``norm``, ``output``) over
  ``mesh["fsdp"]``.  Stage root stays unwrapped (see ``pp_fsdp_tp_cp_sp_example``) so
  Torch PP scheduler FSDP MetaSteps do not conflict with nested autograd hooks.
* **Reference** — full ``Llama3Model`` on every rank; each step runs one forward +
  sum-CE + backward on the **same** global ``(tokens, targets)`` mini-batch.

Level0 (10 steps) and level1 (1000 steps) workers compare the reconstructed global sum-loss
from the last PP stage against an in-process single-card ``Llama3Model`` baseline before
``optimizer.step()``, with a full loss-trajectory check at the end.
"""
# pylint: disable=W0611,C0413,C0412,W0613,W0612
from __future__ import annotations

import os

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from typing import Iterable, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu
from torch import nn, optim

from hyper_parallel import DTensor, SkipDTensorDispatch, fully_shard, init_device_mesh
from hyper_parallel.core.dtensor.placement_types import Replicate
from hyper_parallel.core.pipeline_parallel.scheduler import Schedule1F1B

from tests.torch.accuracy.model import Llama3DemoConfig, Llama3Model
from tests.torch.accuracy.parallelize import parallelize_llama3
from tests.torch.pipeline_parallel.pipeline import (
    build_llama3_pp_chunk,
    build_pipeline_stage,
    load_pp_chunk_from_reference,
    split_batch_dim0,
)
from tests.torch.utils import init_dist

_BATCH_SIZE = 8
_SEQ_LEN = 16
_STEPS_LEVEL0 = 10
_STEPS_LEVEL1 = 1000
_LR = 1e-4
_INIT_SEED = 1234
_DATA_SEED = 2026
_RTOL_LEVEL0 = 1e-3
_ATOL_LEVEL0 = 1e-3
# Long trajectories: FSDP/TP grad reduce vs dense SGD drift slowly; allow slightly looser bounds.
_RTOL_LEVEL1 = 2e-3
_ATOL_LEVEL1 = 2.0
_PP_SIZE = 2
_FSDP_SIZE = 2
_TP_SIZE = 2
_MICRO_BATCH_NUM = 4


def _build_config() -> Llama3DemoConfig:
    """Return the small Llama3 config aligned with ``examples/torch/llama3`` demos."""
    return Llama3DemoConfig(
        dim=256,
        n_layers=4,
        n_heads=8,
        n_kv_heads=4,
        vocab_size=1024,
        max_seq_len=128,
    )


def _build_reference_model(cfg: Llama3DemoConfig, device: torch.device) -> Llama3Model:
    """Build the full reference ``Llama3Model`` from a fixed seed."""
    torch.manual_seed(_INIT_SEED)
    return Llama3Model(cfg).to(device=device)


def _build_global_batch(
    cfg: Llama3DemoConfig, device: torch.device, step: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return deterministic ``(tokens, targets)`` for ``step`` on ``device``."""
    torch.manual_seed(_DATA_SEED + step)
    tokens = torch.randint(0, cfg.vocab_size, (_BATCH_SIZE, _SEQ_LEN), device=device)
    targets = torch.randint(0, cfg.vocab_size, (_BATCH_SIZE, _SEQ_LEN), device=device)
    return tokens, targets


def _broadcast_batch(tokens: torch.Tensor, targets: torch.Tensor) -> None:
    """Broadcast the global mini-batch from rank 0."""
    if dist.is_initialized():
        dist.broadcast(tokens, src=0)
        dist.broadcast(targets, src=0)


def _sum_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, vocab_size: int
) -> torch.Tensor:
    """Return sum cross-entropy in float32."""
    return F.cross_entropy(
        logits.float().reshape(-1, vocab_size),
        targets.reshape(-1),
        reduction="sum",
    )


def _microbatch_loss_value(loss: torch.Tensor) -> float:
    """Extract a scalar sum-loss (handles ``DTensor``)."""
    if hasattr(loss, "to_local"):
        return float(loss.to_local().detach().cpu().item())
    return float(loss.detach().cpu().item())


def _step_global_loss(losses: List[torch.Tensor]) -> float:
    """Sum per-micro-batch sum-CE losses from the last PP stage."""
    return sum(_microbatch_loss_value(loss) for loss in losses)


def _reporter_global_rank(pp_size: int, fsdp_size: int, tp_size: int) -> int:
    """Global rank of ``(pp=pp_size-1, fsdp=0, tp=0)`` in row-major ``(pp, fsdp, tp)`` mesh."""
    return (pp_size - 1) * fsdp_size * tp_size


def _shard_module(module: nn.Module, fsdp_mesh) -> None:
    """Apply ``fully_shard`` to ``module`` and configure sum gradient reduction."""
    fully_shard(module, mesh=fsdp_mesh, reshard_after_forward=True)
    if hasattr(module, "set_reduce_op_type"):
        module.set_reduce_op_type("sum")


def _fsdp_trainable_modules(stage_module: nn.Module) -> Iterable[nn.Module]:
    """Yield every trainable submodule that should be wrapped with ``fully_shard``."""
    if hasattr(stage_module, "tok_embeddings"):
        yield stage_module.tok_embeddings
    yield from stage_module.layers
    if hasattr(stage_module, "norm"):
        yield stage_module.norm
    if hasattr(stage_module, "output"):
        yield stage_module.output


def _apply_fsdp_all_params(stage_module: nn.Module, fsdp_mesh) -> None:
    """``fully_shard`` every trainable submodule so all parameters become ``DTensor``.

    The stage root stays a plain ``nn.Module`` (see ``pp_fsdp_tp_cp_sp_example``) so
    ``Schedule1F1B`` does not inject conflicting root-level FSDP MetaSteps.
    """
    for module in _fsdp_trainable_modules(stage_module):
        _shard_module(module, fsdp_mesh)


def _assert_all_params_are_dtensor(stage_module: nn.Module, pp_rank: int) -> None:
    """Verify every trainable parameter on this PP rank is a sharded ``DTensor``."""
    for name, param in stage_module.named_parameters():
        if not isinstance(param, DTensor):
            raise AssertionError(
                f"pp_rank={pp_rank}: parameter {name!r} is {type(param).__name__}, "
                "expected DTensor after fully_shard."
            )


def _run_single_card_step(
    ref_model: Llama3Model,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    vocab_size: int,
) -> float:
    """One single-card reference step: forward + sum-CE + backward; return scalar loss."""
    ref_model.zero_grad(set_to_none=True)
    logits = ref_model(tokens)
    loss = _sum_cross_entropy(logits, targets, vocab_size)
    loss.backward()
    return float(loss.detach().cpu().item())


def _assert_step_loss_matches_single_card(
    case_name: str,
    step_idx: int,
    rank: int,
    distributed_loss: float,
    single_card_loss: float,
    *,
    rtol: float,
    atol: float,
) -> None:
    """Assert one training step's distributed sum-loss matches the single-card baseline."""
    if not np.isfinite(distributed_loss):
        raise AssertionError(
            f"{case_name}: rank {rank} step {step_idx} "
            f"non-finite distributed loss {distributed_loss}."
        )
    if not np.isfinite(single_card_loss):
        raise AssertionError(
            f"{case_name}: rank {rank} step {step_idx} "
            f"non-finite single-card loss {single_card_loss}."
        )
    if not np.allclose(distributed_loss, single_card_loss, rtol=rtol, atol=atol):
        abs_err = abs(distributed_loss - single_card_loss)
        denom = max(abs(single_card_loss), 1e-8)
        raise AssertionError(
            f"{case_name}: rank {rank} step {step_idx} "
            f"distributed_loss={distributed_loss:.6f} != single_card_loss={single_card_loss:.6f} "
            f"(abs_err={abs_err:.6f}, rel_err={abs_err / denom:.6e}, "
            f"rtol={rtol}, atol={atol})."
        )


def _assert_loss_trajectory_matches_single_card(
    case_name: str,
    rank: int,
    num_steps: int,
    distributed_losses: List[float],
    single_card_losses: List[float],
    *,
    rtol: float,
    atol: float,
) -> None:
    """Assert every step in the training trajectory matches the single-card baseline."""
    if len(distributed_losses) != num_steps:
        raise AssertionError(
            f"{case_name}: rank {rank} expected {num_steps} distributed losses, "
            f"got {len(distributed_losses)}."
        )
    if len(single_card_losses) != num_steps:
        raise AssertionError(
            f"{case_name}: rank {rank} expected {num_steps} single-card losses, "
            f"got {len(single_card_losses)}."
        )
    for step_idx, (dist_loss, single_loss) in enumerate(
        zip(distributed_losses, single_card_losses)
    ):
        _assert_step_loss_matches_single_card(
            case_name, step_idx, rank, dist_loss, single_loss, rtol=rtol, atol=atol
        )


def _print_single_card_loss_comparison(
    case_name: str,
    num_steps: int,
    distributed_losses: List[float],
    single_card_losses: List[float],
    *,
    rtol: float,
    atol: float,
) -> None:
    """Print per-step distributed vs single-card sum-loss (rank 0 only)."""
    print(
        f"[{case_name}] single-card loss comparison "
        f"({num_steps} steps, rtol={rtol}, atol={atol}):"
    )
    print(" step | distributed_loss | single_card_loss |   abs_err |   rel_err")
    row_limit = 20
    indices = list(range(num_steps))
    if num_steps > row_limit:
        indices = list(range(5)) + [-1] + list(range(num_steps - 5, num_steps))
    for step_idx in indices:
        if step_idx == -1:
            print("  ... |                ... |              ... |       ... |       ...")
            continue
        dist_loss = distributed_losses[step_idx]
        single_loss = single_card_losses[step_idx]
        abs_err = abs(dist_loss - single_loss)
        rel_err = abs_err / max(abs(single_loss), 1e-8)
        print(
            f" {step_idx:4d} | {dist_loss:18.6f} | {single_loss:16.6f} | "
            f"{abs_err:9.6f} | {rel_err:9.3e}"
        )
    max_abs_err = max(
        abs(d - s) for d, s in zip(distributed_losses, single_card_losses)
    )
    max_step = max(
        range(num_steps),
        key=lambda i: abs(distributed_losses[i] - single_card_losses[i]),
    )
    print(
        f"[{case_name}] max abs_err={max_abs_err:.6f} at step {max_step} "
        f"(distributed={distributed_losses[max_step]:.6f}, "
        f"single_card={single_card_losses[max_step]:.6f})."
    )


def _count_steps_within_tolerance(
    distributed_losses: List[float],
    single_card_losses: List[float],
    *,
    rtol: float,
    atol: float,
) -> int:
    """Return how many steps satisfy ``np.allclose`` with the given tolerances."""
    return sum(
        1
        for dist_loss, single_loss in zip(distributed_losses, single_card_losses)
        if np.allclose(dist_loss, single_loss, rtol=rtol, atol=atol)
    )


def _print_loss_drift_summary(
    case_name: str,
    num_steps: int,
    distributed_losses: List[float],
    single_card_losses: List[float],
) -> None:
    """Print distributed vs single-card loss error stats (no assertion; rank 0 only)."""
    abs_errors = np.array(
        [abs(d - s) for d, s in zip(distributed_losses, single_card_losses)],
        dtype=np.float64,
    )
    rel_errors = abs_errors / np.maximum(np.abs(single_card_losses), 1e-8)
    max_step = int(np.argmax(abs_errors))
    within_level0 = _count_steps_within_tolerance(
        distributed_losses, single_card_losses, rtol=_RTOL_LEVEL0, atol=_ATOL_LEVEL0
    )
    within_level1 = _count_steps_within_tolerance(
        distributed_losses, single_card_losses, rtol=_RTOL_LEVEL1, atol=_ATOL_LEVEL1
    )
    print(f"[{case_name}] loss drift summary ({num_steps} steps):")
    print(
        f"  abs_err: max={abs_errors.max():.6f} (step {max_step}), "
        f"mean={abs_errors.mean():.6f}, median={np.median(abs_errors):.6f}, "
        f"p99={np.percentile(abs_errors, 99):.6f}"
    )
    print(
        f"  rel_err: max={rel_errors.max():.6e} (step {max_step}), "
        f"mean={rel_errors.mean():.6e}, median={np.median(rel_errors):.6e}, "
        f"p99={np.percentile(rel_errors, 99):.6e}"
    )
    print(
        f"  within level0 tol (rtol={_RTOL_LEVEL0}, atol={_ATOL_LEVEL0}): "
        f"{within_level0}/{num_steps} steps"
    )
    print(
        f"  within level1 tol (rtol={_RTOL_LEVEL1}, atol={_ATOL_LEVEL1}): "
        f"{within_level1}/{num_steps} steps"
    )
    _print_single_card_loss_comparison(
        case_name,
        num_steps,
        distributed_losses,
        single_card_losses,
        rtol=_RTOL_LEVEL0,
        atol=_ATOL_LEVEL0,
    )
    sample_stride = max(1, num_steps // 10)
    print(f"[{case_name}] sampled abs_err every {sample_stride} steps:")
    for step_idx in range(0, num_steps, sample_stride):
        print(
            f"  step {step_idx:5d}: abs_err={abs_errors[step_idx]:.6f}, "
            f"rel_err={rel_errors[step_idx]:.6e}"
        )


def _assert_pp_fsdp_tp_1f1b_match_reference(
    num_steps: int,
    case_name: str,
    *,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    print_drift_summary: bool = False,
) -> None:
    """Drive ``Schedule1F1B`` PP+FSDP+TP; optionally assert loss matches single-card baseline.

    When ``rtol`` and ``atol`` are both ``None``, only record per-step losses and print a
    drift summary on rank 0 (no tolerance assertion).  When ``print_drift_summary`` is
    ``True`` and tolerances are set, assert first then print error statistics on rank 0.
    """
    rank, device_id = init_dist()
    world_size = dist.get_world_size()
    expected_world = _PP_SIZE * _FSDP_SIZE * _TP_SIZE
    if world_size != expected_world:
        if rank == 0:
            print(
                f"Skip pp_fsdp_tp_1f1b_composite: requires world_size={expected_world}, "
                f"got {world_size}."
            )
        return

    device = torch.device("npu", device_id)
    cfg = _build_config()
    if cfg.n_layers < _PP_SIZE:
        raise AssertionError(f"n_layers ({cfg.n_layers}) must be >= pp_size ({_PP_SIZE}).")
    if cfg.n_heads % _TP_SIZE != 0 or cfg.n_kv_heads % _TP_SIZE != 0:
        raise AssertionError("n_heads and n_kv_heads must be divisible by TP size.")
    if _SEQ_LEN % _TP_SIZE != 0:
        raise AssertionError(f"seq_len ({_SEQ_LEN}) must be divisible by tp_size ({_TP_SIZE}).")
    if _BATCH_SIZE % _MICRO_BATCH_NUM != 0:
        raise AssertionError("batch_size must divide micro_batch_num.")

    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(_PP_SIZE, _FSDP_SIZE, _TP_SIZE),
        mesh_dim_names=("pp", "fsdp", "tp"),
    )
    pp_mesh = mesh["pp"]
    fsdp_mesh = mesh["fsdp"]
    tp_mesh = mesh["tp"]
    pp_rank = mesh.get_local_rank("pp")
    fsdp_rank = mesh.get_local_rank("fsdp")
    tp_rank = mesh.get_local_rank("tp")
    is_last = pp_rank == _PP_SIZE - 1
    is_reporter = is_last and fsdp_rank == 0 and tp_rank == 0
    reporter_rank = _reporter_global_rank(_PP_SIZE, _FSDP_SIZE, _TP_SIZE)

    reference = _build_reference_model(cfg, device)
    stage_chunk = build_llama3_pp_chunk(cfg, pp_rank, _PP_SIZE).to(device=device)
    load_pp_chunk_from_reference(stage_chunk, reference, pp_rank, _PP_SIZE)

    parallelize_llama3(stage_chunk, tp_mesh)
    _apply_fsdp_all_params(stage_chunk, fsdp_mesh)
    _assert_all_params_are_dtensor(stage_chunk, pp_rank)

    pipeline_stage = build_pipeline_stage(
        stage_chunk,
        pp_rank=pp_rank,
        pp_size=_PP_SIZE,
        device=device,
        pp_mesh=pp_mesh,
        use_microbatch_loss=is_last,
    )
    schedule = Schedule1F1B(pipeline_stage, _MICRO_BATCH_NUM)

    dist_optimizer = optim.SGD(stage_chunk.parameters(), lr=_LR)
    single_card_model = _build_reference_model(cfg, device)
    single_card_optimizer = optim.SGD(single_card_model.parameters(), lr=_LR)

    distributed_losses: List[float] = []
    single_card_losses: List[float] = []

    for step in range(num_steps):
        tokens, targets = _build_global_batch(cfg, device, step)
        _broadcast_batch(tokens, targets)

        dist_optimizer.zero_grad(set_to_none=True)
        if is_last:
            stage_chunk.set_micro_targets(split_batch_dim0(targets, _MICRO_BATCH_NUM))

        if pp_rank == 0:
            d_tokens = DTensor.from_local(tokens, tp_mesh, (Replicate(),))
            losses = schedule.run(d_tokens)
        else:
            losses = schedule.run()

        loss_buf = torch.zeros(1, dtype=torch.float64, device=device)
        if is_reporter:
            loss_buf[0] = _step_global_loss(losses)
        dist.broadcast(loss_buf, src=reporter_rank)
        dist_loss = float(loss_buf.item())

        single_card_loss = _run_single_card_step(
            single_card_model, tokens, targets, cfg.vocab_size
        )
        if not np.isfinite(dist_loss) or not np.isfinite(single_card_loss):
            raise AssertionError(
                f"{case_name}: rank {rank} step {step} non-finite loss "
                f"(distributed={dist_loss}, single_card={single_card_loss})."
            )
        if rtol is not None and atol is not None:
            _assert_step_loss_matches_single_card(
                case_name, step, rank, dist_loss, single_card_loss, rtol=rtol, atol=atol
            )
        distributed_losses.append(dist_loss)
        single_card_losses.append(single_card_loss)

        with SkipDTensorDispatch():
            dist_optimizer.step()
            single_card_optimizer.step()

    if rtol is not None and atol is not None:
        _assert_loss_trajectory_matches_single_card(
            case_name,
            rank,
            num_steps,
            distributed_losses,
            single_card_losses,
            rtol=rtol,
            atol=atol,
        )

    if rank == 0:
        if print_drift_summary or (rtol is None and atol is None):
            _print_loss_drift_summary(
                case_name, num_steps, distributed_losses, single_card_losses
            )
        elif rtol is not None and atol is not None:
            _print_single_card_loss_comparison(
                case_name,
                num_steps,
                distributed_losses,
                single_card_losses,
                rtol=rtol,
                atol=atol,
            )
        print(
            f"[Rank {rank}] {case_name} ok "
            f"(pp={_PP_SIZE}, fsdp={_FSDP_SIZE}, tp={_TP_SIZE}, "
            f"micro_batches={_MICRO_BATCH_NUM}, steps={num_steps})"
        )


def test_pp_fsdp_tp_1f1b_composite_matches_reference() -> None:
    """
    Feature: Llama3 ``PP + FSDP + TP`` under ``Schedule1F1B`` vs single-card baseline (10 steps).
    Description:
        1. Build mesh ``(pp=2, fsdp=2, tp=2)`` on 8 ranks with Llama3 PP stage chunks from
           ``pipeline.py`` (layout aligned with ``examples/torch/llama3``).
        2. Apply TP + nested ``fully_shard`` on every trainable submodule (stage root
           unwrapped, aligned with ``examples/torch/llama3``).
        3. For each of 10 training steps, compare global sum-loss against an in-process
           single-card ``Llama3Model`` baseline on the same batch before ``optimizer.step()``,
           then verify the full loss trajectory.
    Expectation:
        Every step's global sum-loss matches the single-card baseline within
        ``rtol=1e-3``/``atol=1e-3``.
    """
    _assert_pp_fsdp_tp_1f1b_match_reference(
        _STEPS_LEVEL0,
        "pp_fsdp_tp_1f1b_composite",
        rtol=_RTOL_LEVEL0,
        atol=_ATOL_LEVEL0,
    )


def test_pp_fsdp_tp_1f1b_composite_matches_reference_1000_steps() -> None:
    """
    Feature: Llama3 ``PP + FSDP + TP`` under ``Schedule1F1B`` vs single-card baseline (1000 steps).
    Description:
        Same as :func:`test_pp_fsdp_tp_1f1b_composite_matches_reference`, but runs 1000
        training steps.  Asserts per-step sum-loss against the single-card baseline, then
        rank 0 prints abs/rel error drift statistics (max/mean/p99, level0/level1 pass counts).
    Expectation:
        Every step's global sum-loss matches the single-card baseline within
        ``rtol=2e-3``/``atol=2.0``; drift summary is printed on success.
    """
    _assert_pp_fsdp_tp_1f1b_match_reference(
        _STEPS_LEVEL1,
        "pp_fsdp_tp_1f1b_composite_1000_steps",
        rtol=_RTOL_LEVEL1,
        atol=_ATOL_LEVEL1,
        print_drift_summary=True,
    )

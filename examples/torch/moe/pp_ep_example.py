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
"""MoE demo: pipeline parallelism + expert parallelism (1F1B) correctness verification.

This script runs the same MoE model in two modes and compares their forward
losses to verify that PP + EP produces numerically identical results to a
standalone (non-distributed) reference:

1. **Standalone** — single-rank reference with the full model on each rank.
   No PP splitting, no EP sharding.  All ranks produce identical losses.
2. **PP + EP distributed** — model split across PP ranks with experts
   sharded across EP ranks via :class:`ExpertParallel`, driven by
   :class:`Schedule1F1B`.

Verification strategy:
    * Both paths use **identical inputs** (broadcast from rank 0) and
      **identical initial weights** (broadcast from rank 0).
    * At each step we compare the **forward-only loss** (no optimizer step)
      so the two paths always start from the same weights.
    * The standalone path computes full-batch cross-entropy; the PP+EP path
      averages per-micro-batch CE losses on the last stage.  These are
      mathematically equal because CE averages over all tokens.

Mesh layout (4 ranks, ``pp=2``, ``ep=2``)::

          ep →
    pp ↓  rank 0   rank 1
          rank 2   rank 3

``pp`` splits the model across rows; ``ep`` shards experts across columns.

Run (from this directory)::

    torchrun --nproc_per_node=4 pp_ep_example.py

Optional environment variables:

* ``MOE_PP_SIZE`` — pipeline width (default ``2``).
* ``MOE_EP_SIZE`` — expert-parallel width (default ``2``).
* ``MOE_NUM_STEPS`` — verification steps (default ``5``).
* ``MOE_DEVICE_TYPE`` — ``npu`` or ``cuda`` (default ``npu``).

Requirements:
    * ``world_size == MOE_PP_SIZE * MOE_EP_SIZE``.
    * ``num_experts`` must be divisible by ``MOE_EP_SIZE``.
    * ``n_layers`` must be >= ``MOE_PP_SIZE``.
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from demo_utils import init_dist
from model import MoEDemoConfig, MoEDemoModel
from parallelize import broadcast_state_dict_from_rank0, parallelize_moe_ep
from pipeline import (
    build_moe_pp_chunk,
    build_pipeline_stage,
    count_moe_parameters,
    extract_stage_state_dict,
    split_batch_dim0,
)

from hyper_parallel import init_device_mesh
from hyper_parallel.core.pipeline_parallel.scheduler import Schedule1F1B


SEED_MODEL = 42
SEED_INPUT_BASE = 1000
RTOL = 1e-3
ATOL = 1e-3
TRAIN_STEPS_DEFAULT = 5


def _parallel_sizes_from_env(world: int) -> tuple[int, int]:
    """Read ``MOE_PP_SIZE`` and ``MOE_EP_SIZE`` and validate ``world``."""
    pp_raw = os.environ.get("MOE_PP_SIZE", "2").strip()
    ep_raw = os.environ.get("MOE_EP_SIZE", "2").strip()
    try:
        pp_size = int(pp_raw)
        ep_size = int(ep_raw)
    except ValueError as exc:
        raise ValueError(
            f"MOE_PP_SIZE and MOE_EP_SIZE must be integers, got "
            f"pp={pp_raw!r}, ep={ep_raw!r}"
        ) from exc
    if pp_size < 1 or ep_size < 1:
        raise ValueError("MOE_PP_SIZE and MOE_EP_SIZE must be >= 1.")
    if world != pp_size * ep_size:
        raise ValueError(
            f"world_size ({world}) must equal pp ({pp_size}) * ep ({ep_size})."
        )
    return pp_size, ep_size


def _train_steps_override() -> int:
    """Return verification step count (``MOE_NUM_STEPS``, default ``5``)."""
    raw = os.environ.get("MOE_NUM_STEPS", str(TRAIN_STEPS_DEFAULT)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"MOE_NUM_STEPS must be an integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"MOE_NUM_STEPS must be >= 1 (got {value}).")
    return value


def _build_full_model(
    cfg: MoEDemoConfig, device: torch.device,
) -> MoEDemoModel:
    """Create a full MoE model with a fixed seed on *device*.

    Args:
        cfg: Model configuration.
        device: Target device.

    Returns:
        Initialised MoEDemoModel on *device*.
    """
    torch.manual_seed(SEED_MODEL)
    if device.type == "npu":
        torch.npu.manual_seed(SEED_MODEL)
    elif device.type == "cuda":
        torch.cuda.manual_seed(SEED_MODEL)
    model = MoEDemoModel(cfg).to(device=device)
    return model


def _make_inputs(
    cfg: MoEDemoConfig,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    step: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate identical tokens/targets on all ranks via broadcast.

    Args:
        cfg: Model configuration.
        batch_size: Batch dimension.
        seq_len: Sequence length.
        device: Target device.
        step: Current step (used as seed offset).

    Returns:
        Tuple of (tokens, targets) identical on all ranks.
    """
    torch.manual_seed(SEED_INPUT_BASE + step)
    tokens = torch.randint(
        0, cfg.vocab_size, (batch_size, seq_len), device=device,
    )
    targets = torch.randint(
        0, cfg.vocab_size, (batch_size, seq_len), device=device,
    )
    dist.broadcast(tokens, src=0)
    dist.broadcast(targets, src=0)
    return tokens, targets


def _compare_losses(
    losses: list,
    s_loss: float,
    step: int,
    ep_rank: int,
    is_last: bool,
) -> bool:
    """Compare PP+EP losses against standalone loss; return *True* if close."""
    if not is_last:
        return True
    mean_loss = sum(
        loss.item() if not hasattr(loss, "to_local") else loss.to_local().item()
        for loss in losses
    ) / len(losses)
    passed = abs(mean_loss - s_loss) < max(ATOL, RTOL * abs(s_loss))
    if ep_rank == 0:
        status = "PASS" if passed else "FAIL"
        print(
            f"[pp_ep step {step}] {status}  "
            f"standalone_loss={s_loss:.4f}  pp_ep_loss={mean_loss:.4f}  "
            f"loss_diff={abs(mean_loss - s_loss):.6f}  "
            f"(rtol={RTOL}, atol={ATOL})"
        )
    return passed


def _print_summary(
    all_passed: bool,
    num_steps: int,
    pp_size: int,
    ep_size: int,
    cfg: MoEDemoConfig,
    is_last: bool,
    ep_rank: int,
) -> None:
    """Print final verification summary on the last stage, EP rank 0."""
    if not (is_last and ep_rank == 0):
        return
    if all_passed:
        print(
            f"\n=== PP+EP correctness verification PASSED === "
            f"({num_steps} steps, pp={pp_size}, ep={ep_size}, "
            f"{cfg.num_experts} experts, top_k={cfg.top_k})"
        )
    else:
        print(
            "\n=== PP+EP correctness verification FAILED === "
            "Check per-step diagnostics above."
        )


def main() -> None:
    """Entry point: set up PP+EP distributed mesh, verify correctness against standalone model."""
    rank, world, device_type = init_dist()
    device = torch.device(device_type, rank)
    pp_size, ep_size = _parallel_sizes_from_env(world)

    mesh = init_device_mesh(
        device_type=device_type,
        mesh_shape=(pp_size, ep_size),
        mesh_dim_names=("pp", "ep"),
    )
    pp_mesh = mesh["pp"]
    ep_mesh = mesh["ep"]
    pp_rank = pp_mesh.get_local_rank()
    ep_rank = ep_mesh.get_local_rank()
    is_last = pp_rank == pp_size - 1

    cfg = MoEDemoConfig(
        dim=256,
        n_layers=4,
        n_heads=8,
        n_kv_heads=4,
        vocab_size=1024,
        max_seq_len=128,
        num_experts=4,
        moe_hidden_dim=512,
        top_k=2,
    )

    if cfg.n_layers < pp_size:
        raise ValueError(
            f"n_layers ({cfg.n_layers}) must be >= pp_size ({pp_size})."
        )
    if cfg.num_experts % ep_size != 0:
        raise ValueError(
            f"num_experts ({cfg.num_experts}) must divide ep_size ({ep_size})."
        )

    micro_batch_num = 4
    batch_size = 8
    seq_len = 16
    if batch_size % micro_batch_num != 0:
        raise ValueError("batch_size must divide micro_batch_num.")

    num_steps = _train_steps_override()

    if rank == 0:
        ref_params = count_moe_parameters(cfg)
        print(
            f"[pp_ep] world={world}, pp={pp_size}, ep={ep_size}, "
            f"micro_batches={micro_batch_num}, "
            f"full-model params≈{ref_params:,}"
        )

    standalone_model = _build_full_model(cfg, device)
    broadcast_state_dict_from_rank0(standalone_model)

    reference_model = _build_full_model(cfg, device)
    broadcast_state_dict_from_rank0(reference_model)

    stage_module = build_moe_pp_chunk(cfg, pp_rank, pp_size).to(device=device)
    stage_sd = extract_stage_state_dict(reference_model, cfg, pp_rank, pp_size)
    stage_module.load_state_dict(stage_sd, strict=False)
    del reference_model

    parallelize_moe_ep(stage_module, ep_mesh)

    pipeline_stage = build_pipeline_stage(
        stage_module,
        pp_rank=pp_rank,
        pp_size=pp_size,
        device=device,
        pp_mesh=pp_mesh,
        use_microbatch_loss=is_last,
    )
    schedule = Schedule1F1B(pipeline_stage, micro_batch_num)

    all_passed = True
    for step in range(num_steps):
        tokens, targets = _make_inputs(cfg, batch_size, seq_len, device, step)
        s_logits = standalone_model(tokens)
        s_loss = F.cross_entropy(
            s_logits.float().reshape(-1, cfg.vocab_size),
            targets.reshape(-1),
        ).item()
        if is_last:
            stage_module.set_micro_targets(
                split_batch_dim0(targets, micro_batch_num)
            )
        losses = schedule.run(tokens) if pp_rank == 0 else schedule.run()
        if not _compare_losses(losses, s_loss, step, ep_rank, is_last):
            all_passed = False

    _print_summary(all_passed, num_steps, pp_size, ep_size, cfg, is_last, ep_rank)
    dist.barrier()


if __name__ == "__main__":
    main()

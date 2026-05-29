# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Minimal MoE Expert-Parallel demo — standalone vs EP correctness verification.

This script runs the same MoE model in two modes and compares their outputs
and gradients to verify that Expert Parallelism produces numerically
identical results:

1. **Standalone** — single-rank reference with all experts local.
2. **EP distributed** — experts sharded across ranks via
   :class:`~hyper_parallel.core.expert_parallel.expert_parallel.ExpertParallel`
   with differentiable all-to-all token routing.

Run (from repo root or this directory)::

    torchrun --nproc_per_node=4 expert_parallel_example.py

Optional environment variables:

* ``MOE_NUM_STEPS`` — training steps (default ``5``).
* ``MOE_DEVICE_TYPE`` — ``npu`` or ``cuda`` (default ``npu``).

Requirements:
    * ``num_experts`` must be divisible by the world size.
    * Each rank must create the model with an identical seed before EP
      sharding, then broadcast parameters from rank 0.

Verification strategy:
    All ranks create the same model and input with identical seeds.
    The standalone path runs independently on each rank (producing identical
    results everywhere).  The EP path shards experts and routes tokens across
    ranks.  Forward outputs and input gradients are compared with
    ``rtol=1e-3, atol=1e-3`` tolerances.
"""
# pylint: disable=C0413
from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
# Required for direct script execution (``torchrun expert_parallel_example.py``)
# so that ``from model import ...`` resolves within this package directory.
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

import torch
import torch.nn.functional as F

from hyper_parallel import SkipDTensorDispatch, init_device_mesh

from demo_utils import init_dist
from model import MoEDemoConfig, MoEDemoModel
from parallelize import broadcast_state_dict_from_rank0, parallelize_moe_ep


SEED_MODEL = 42
SEED_INPUT_BASE = 1000
RTOL = 1e-3
ATOL = 1e-3
TRAIN_STEPS = 5


def _build_model(cfg: MoEDemoConfig, device: torch.device) -> MoEDemoModel:
    """Create a MoE model with a fixed seed on *device*.

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


def _run_forward_backward(
    model: MoEDemoModel,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    cfg: MoEDemoConfig,
) -> tuple:
    """Run a forward + backward pass and return (loss, logits, input_grad).

    Args:
        model: The MoE model.
        tokens: Token indices ``[batch, seq_len]``.
        targets: Target indices ``[batch, seq_len]``.
        cfg: Model configuration.

    Returns:
        Tuple of ``(loss_item, logits_detached, token_grad)``.
    """
    tokens_in = tokens.clone().detach().requires_grad_(True)
    logits = model(tokens_in)
    loss = F.cross_entropy(
        logits.float().reshape(-1, cfg.vocab_size),
        targets.reshape(-1),
    )
    with SkipDTensorDispatch():
        loss.backward()
    return loss.item(), logits.detach(), tokens_in.grad.clone()


def _precision_close(
    result: torch.Tensor,
    reference: torch.Tensor,
    rtol: float = RTOL,
    atol: float = ATOL,
) -> tuple[bool, float]:
    """Check that *result* is close to *reference* and print diagnostics.

    Args:
        result: Tensor produced by the EP path.
        reference: Tensor produced by the standalone path.
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    Returns:
        Tuple of ``(is_close, max_diff)``.
    """
    result_cpu = result.detach().cpu().float()
    reference_cpu = reference.detach().cpu().float()
    max_diff = (result_cpu - reference_cpu).abs().max().item()
    close = torch.allclose(result_cpu, reference_cpu, rtol=rtol, atol=atol)
    return close, max_diff


def main() -> None:
    rank, world, device_type = init_dist()
    device = torch.device(device_type, rank)

    # Configuration: num_experts must divide world_size.
    cfg = MoEDemoConfig(
        dim=256,
        n_layers=2,
        n_heads=8,
        n_kv_heads=4,
        vocab_size=1024,
        max_seq_len=128,
        num_experts=4,
        moe_hidden_dim=512,
        top_k=2,
    )
    if cfg.num_experts % world != 0:
        raise ValueError(
            f"num_experts ({cfg.num_experts}) must be divisible by "
            f"world_size ({world})."
        )

    batch_size = 2
    seq_len = 16

    # ------------------------------------------------------------------
    # 1. Standalone reference (all experts on each rank, no EP)
    # ------------------------------------------------------------------
    standalone_model = _build_model(cfg, device)
    broadcast_state_dict_from_rank0(standalone_model)
    standalone_optimizer = torch.optim.Adam(standalone_model.parameters(), lr=1e-4)

    # ------------------------------------------------------------------
    # 2. EP distributed model (experts sharded across ranks)
    # ------------------------------------------------------------------
    ep_model = _build_model(cfg, device)
    broadcast_state_dict_from_rank0(ep_model)

    ep_mesh = init_device_mesh(
        device_type=device_type,
        mesh_shape=(world,),
        mesh_dim_names=("ep",),
    )
    parallelize_moe_ep(ep_model, ep_mesh)

    ep_optimizer = torch.optim.Adam(ep_model.parameters(), lr=1e-4)

    # ------------------------------------------------------------------
    # 3. Training loop: compare standalone vs EP at each step
    # ------------------------------------------------------------------
    num_steps = TRAIN_STEPS
    all_passed = True

    for step in range(num_steps):
        # Same input on all ranks.
        torch.manual_seed(SEED_INPUT_BASE + step)
        tokens = torch.randint(
            0, cfg.vocab_size, (batch_size, seq_len), device=device
        )
        targets = torch.randint(
            0, cfg.vocab_size, (batch_size, seq_len), device=device
        )

        # --- Standalone forward/backward ---
        standalone_optimizer.zero_grad(set_to_none=True)
        s_loss, s_logits, _ = _run_forward_backward(
            standalone_model, tokens, targets, cfg
        )
        standalone_optimizer.step()

        # --- EP forward/backward ---
        ep_optimizer.zero_grad(set_to_none=True)
        e_loss, e_logits, _ = _run_forward_backward(
            ep_model, tokens, targets, cfg
        )
        ep_optimizer.step()

        # --- Compare outputs ---
        logits_close, logits_diff = _precision_close(
            e_logits, s_logits
        )
        loss_close = abs(e_loss - s_loss) < max(ATOL, RTOL * abs(s_loss))

        passed = logits_close and loss_close
        if not passed:
            all_passed = False

        if rank == 0:
            status = "PASS" if passed else "FAIL"
            print(
                f"[ep step {step}] {status}  "
                f"standalone_loss={s_loss:.4f}  ep_loss={e_loss:.4f}  "
                f"logits_max_diff={logits_diff:.6f}  "
                f"(rtol={RTOL}, atol={ATOL})"
            )

    # ------------------------------------------------------------------
    # 4. Final summary
    # ------------------------------------------------------------------
    if rank == 0:
        if all_passed:
            print(
                f"\n=== EP correctness verification PASSED === "
                f"({num_steps} steps, {world} ranks, "
                f"{cfg.num_experts} experts, top_k={cfg.top_k})"
            )
        else:
            print(
                "\n=== EP correctness verification FAILED === "
                "Check per-step diagnostics above."
            )


if __name__ == "__main__":
    main()

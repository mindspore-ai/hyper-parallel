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
"""Activation checkpoint memory comparison: None vs Recompute vs Save vs Swap"""
import copy
import torch

from hyper_parallel.core.activation_checkpoint import CheckpointPolicy, SwapManager, checkpoint_wrapper
from tests.torch.common_net import SimpleTransformer
from tests.torch.activation_checkpoint.utils import prepare_data, seed_memory_time_context, train_one_mode


def apply_recompute(model, mode):
    """Apply activation checkpointing based on the specified mode."""
    op_non_recompute = {
        torch.ops.aten.matmul.default,
        torch.ops.aten.addmm.default,
        torch.ops.aten.bmm.default
    }
    if mode == "none":
        return model

    if mode == "recompute":
        for i, layer in enumerate(model.layers):
            model.layers[i] = checkpoint_wrapper(layer)
    elif mode == "swap":
        def policy_fn(ctx, op, *args, **kwargs):  # pylint: disable=W0613
            if op in op_non_recompute:
                return CheckpointPolicy.MUST_SWAP
            return CheckpointPolicy.MUST_RECOMPUTE

        for i, layer in enumerate(model.layers):
            model.layers[i] = checkpoint_wrapper(layer, policy_fn=policy_fn, swap_inputs=True)

        for i in range(len(model.layers) - 1):
            SwapManager().set_forward_prefetch_layer(model.layers[i], model.layers[i + 1])
    elif mode == 'save':
        def policy_fn(ctx, op, *args, **kwargs):  # pylint: disable=W0613
            if op in op_non_recompute:
                return CheckpointPolicy.MUST_SAVE
            return CheckpointPolicy.MUST_RECOMPUTE

        for i, layer in enumerate(model.layers):
            model.layers[i] = checkpoint_wrapper(layer, policy_fn=policy_fn)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return model


def test_ac_memory_comparison():
    """
    Feature: Activation Checkpointing and Swapping Memory Behavior
    Description: Compare peak memory usage across four modes:
                 'none' (baseline), 'recompute' (full activation checkpointing),
                 'save' (partial activation saving), and 'swap' (fine-grained tensor swapping).
                 Validates that losses are numerically identical at every training step
                 and that peak memory follows the expected hierarchy:
                 NONE > SAVE > RECOMPUTE ≈ SWAP.
    Expectation: All modes produce consistent losses (within 1e-5 tolerance),
                 the memory usage trend is satisfied, and no OOM occurs.
    """
    print("🚀 Starting memory and time comparison: none vs recompute vs save vs swap")
    dataloader = prepare_data()
    train_steps = 3

    modes = ["none", "recompute", "save", "swap"]
    results = {}

    for mode in modes:
        print(f"\n--- Running mode: {mode.upper()} ---")
        with seed_memory_time_context() as stats:
            base_model = SimpleTransformer(vocab_size=32000, dim=2048, depth=16).npu()
            model = apply_recompute(base_model, mode)
            losses = train_one_mode(model, dataloader, train_steps)
        peak_mem, duration = stats.get("peak_mem"), stats.get("exec_time")
        results[mode] = {
            "losses": losses,
            "peak_mem_gb": peak_mem,
            "time_sec": duration
        }
        print(f"{mode}: Loss={losses[-1]:.4f}, Peak Mem={peak_mem:.5f} GB, Time={duration:.5f}s")

    print("\n" + "="*70)
    print("📊 FINAL COMPARISON")
    print("="*70)
    print(f"{'Mode':<12} | {'Peak Mem (GB)':<15} | {'Time (s)':<10} | {'Final Loss':<12}")
    print("-"*70)
    for mode in modes:
        r = results[mode]
        print(f"{mode.upper():<12} | {r['peak_mem_gb']:<15.5f} | {r['time_sec']:<10.5f} | {r['losses'][-1]:<12.4f}")

    # loss assert
    base_losses = results["none"]["losses"]
    tol = 1e-5
    for step in range(train_steps):
        base_val = base_losses[step]
        for mode in ["recompute", "save", "swap"]:
            val = results[mode]["losses"][step]
            diff = abs(val - base_val)
            assert diff < tol, (
                f"Loss mismatch at step {step} in mode '{mode}': "
                f"none={base_val:.8f}, {mode}={val:.8f}, diff={diff:.2e}"
            )
    print(f"\n✅ All {train_steps} steps: losses are consistent across modes (tol={tol}).")

    # mem assert
    mem_none = results["none"]["peak_mem_gb"]
    mem_save = results["save"]["peak_mem_gb"]
    mem_recompute = results["recompute"]["peak_mem_gb"]
    mem_swap = results["swap"]["peak_mem_gb"]

    # none > save > recompute ≈ swap
    # none > save
    assert mem_none > mem_save, f"Expected NONE ({mem_none:.5f}) > SAVE ({mem_save:.5f})"
    print(f"✅ Verified: NONE ({mem_none:.5f}) > SAVE ({mem_save:.5f})")
    # save > recompute
    assert mem_save > mem_recompute, f"Expected SAVE ({mem_save:.5f}) > RECOMPUTE ({mem_recompute:.5f})"
    print(f"✅ Verified: SAVE ({mem_save:.5f}) > RECOMPUTE ({mem_recompute:.5f})")
    # recompute ≈ swap
    tol_mem = 0.15
    assert abs(mem_recompute - mem_swap) < tol_mem, \
        f"Expected RECOMPUTE ({mem_recompute:.5f}) ≈ SWAP ({mem_swap:.5f})"
    print(f"✅ Verified: RECOMPUTE ({mem_recompute:.5f}) ≈ SWAP ({mem_swap:.5f}) within tolerance ({tol_mem:.5f} GB)")


class _SmallNet(torch.nn.Module):
    """Two-layer MLP where the hidden activation is exposed as a plain function."""

    def __init__(self, dim: int, use_ckpt: bool):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, dim)
        self.fc2 = torch.nn.Linear(dim, dim)

        def _hidden(x):
            return torch.relu(self.fc1(x))

        self.hidden = checkpoint_wrapper(_hidden) if use_ckpt else _hidden

    def forward(self, x):
        return self.fc2(self.hidden(x))


def test_checkpoint_wrapper_accepts_func():
    """
    Feature: checkpoint_wrapper accepts plain callable (func) as module argument
    Description: Build a small two-layer MLP whose hidden activation is a plain
                 Python function.  Run several training steps on two identical
                 copies of the network — one with the hidden function wrapped by
                 checkpoint_wrapper, one without — and verify that the per-step
                 losses are numerically identical.
    Expectation: Losses across all training steps match within 1e-6 tolerance,
                 confirming that wrapping a func does not alter forward/backward
                 semantics.
    """

    torch.manual_seed(42)
    dim = 32
    batch, train_steps = 8, 5
    tol = 1e-6

    # reference model (no checkpoint)
    ref_model = _SmallNet(dim, use_ckpt=False)
    # wrapped model starts from identical weights
    ckpt_model = _SmallNet(dim, use_ckpt=True)
    ckpt_model.fc1.load_state_dict(copy.deepcopy(ref_model.fc1.state_dict()))
    ckpt_model.fc2.load_state_dict(copy.deepcopy(ref_model.fc2.state_dict()))

    ref_opt = torch.optim.SGD(ref_model.parameters(), lr=0.01)
    ckpt_opt = torch.optim.SGD(ckpt_model.parameters(), lr=0.01)

    torch.manual_seed(42)
    for step in range(train_steps):
        x = torch.randn(batch, dim)
        target = torch.randn(batch, dim)

        # reference forward/backward
        ref_opt.zero_grad()
        ref_loss = torch.nn.functional.mse_loss(ref_model(x), target)
        ref_loss.backward()
        ref_opt.step()

        # checkpoint-wrapped forward/backward (same input & target)
        ckpt_opt.zero_grad()
        ckpt_loss = torch.nn.functional.mse_loss(ckpt_model(x), target)
        ckpt_loss.backward()
        ckpt_opt.step()

        diff = abs(ref_loss.item() - ckpt_loss.item())
        assert diff < tol, (
            f"Loss mismatch at step {step}: "
            f"ref={ref_loss.item():.8f}, ckpt={ckpt_loss.item():.8f}, diff={diff:.2e}"
        )

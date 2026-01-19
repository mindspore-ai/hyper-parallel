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
"""Test Activation checkpoint memory comparison: None vs Recompute vs Save vs Swap"""
import random
import time
from contextlib import contextmanager

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hyper_parallel.platform.torch.activation_checkpoint import (
    CheckpointPolicy, SwapManager, checkpoint_wrapper)
from tests.common.mark_utils import arg_mark


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.npu.is_available():
        torch.npu.manual_seed(seed)
        torch.npu.manual_seed_all(seed)


@contextmanager
def seed_memory_time_context(seed=42):
    """Context manager to set seed, track peak memory and execution time."""
    set_seed(seed)
    torch.npu.reset_peak_memory_stats()
    torch.npu.empty_cache()
    start_time = time.time()

    stats = {}

    try:
        yield stats
    finally:
        exec_time = time.time() - start_time
        peak_mem_gb = torch.npu.max_memory_allocated() / (1024 ** 3)
        torch.npu.empty_cache()
        stats["peak_mem"] = peak_mem_gb
        stats["exec_time"] = exec_time


class TransformerBlock(nn.Module):
    """A simple Transformer block for testing purposes."""

    def __init__(self, dim=256, num_heads=4):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = self.norm1(x)
        x = x + self.ffn(x)
        x = self.norm2(x)
        return x


class SimpleTransformer(nn.Module):
    """A simple Transformer model for testing purposes."""

    def __init__(self, vocab_size=1000, dim=2048, depth=6):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([TransformerBlock(dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        for block in self.layers:
            x = block(x)
        x = self.norm(x)
        return self.head(x)


def prepare_data(batch_size=8, seq_len=512, num_samples=64):
    dataset = TensorDataset(
        torch.randint(0, 10000, (num_samples, seq_len)),
        torch.randint(0, 10000, (num_samples, seq_len))
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader


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
            model.layers[i] = checkpoint_wrapper(layer, policy_fn=policy_fn)

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


def train_one_mode(mode, dataloader, train_steps=5):
    """Run training for one mode and return metrics."""
    with seed_memory_time_context() as stats:
        model = SimpleTransformer(vocab_size=32000, dim=2048, depth=16).npu()
        model = apply_recompute(model, mode)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        losses = []
        for step, (x, y) in enumerate(dataloader):
            if step >= train_steps:
                break
            x, y = x.npu(), y.npu()
            optimizer.zero_grad()
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    return losses, stats.get("peak_mem"), stats.get("exec_time")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
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
        losses, peak_mem, duration = train_one_mode(mode, dataloader, train_steps)
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

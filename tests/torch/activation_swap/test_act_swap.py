# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Test Activation Swap memory comparison: None vs Swap"""
import time
import random
from contextlib import contextmanager
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tests.common.mark_utils import arg_mark
from hyper_parallel.platform.torch.activation_checkpoint import swap_wrapper, ActivationPolicy, SwapManager


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
        stats["peak_mem"] =  peak_mem_gb
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
    """Apply activation swap to the model based on the specified mode."""
    if mode == "none":
        return model

    if mode == "swap":
        for i, layer in enumerate(model.layers):
            model.layers[i].attn = swap_wrapper(layer.attn)

        for i in range(len(model.layers) - 1):
            SwapManager().set_forward_prefetch_layer(model.layers[i], model.layers[i + 1])
    elif mode == "swap_with_policy":
        def policy_fn(x):
            if x.storage().size() <= 32*512*512:
                return ActivationPolicy.SAVE
            return ActivationPolicy.SWAP

        for i, layer in enumerate(model.layers):
            model.layers[i].attn = swap_wrapper(layer.attn, policy_fn)

        for i in range(len(model.layers) - 1):
            SwapManager().set_forward_prefetch_layer(model.layers[i], model.layers[i + 1])
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
def test_act_swap_memory_comparison():
    """
    Feature: Activation Swap Memory Behavior
    Description: Compare peak memory usage across four modes:
                 'none' (baseline),
                 'swap' (swap all tensors),
                 'swap_with_policy' (swap part of tensors according to the policy function).
                 Validates that losses are numerically identical at every training step
                 and that peak memory follows the expected hierarchy:
    Expectation: NONE > SWAP_WITH_POLICY > SWAP,
                 the memory usage trend is satisfied, and no OOM occurs.
    """
    print("🚀 Starting memory and time comparison: none vs swap vs swap_with_policy")
    dataloader = prepare_data()
    train_steps=3

    modes = ["none", "swap", "swap_with_policy"]
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
    print(f"{'Mode':<15} | {'Peak Mem (GB)':<15} | {'Time (s)':<10} | {'Final Loss':<12}")
    print("-"*70)
    for mode in modes:
        r = results[mode]
        print(f"{mode.upper():<20} | {r['peak_mem_gb']:<15.5f} | {r['time_sec']:<10.5f} | {r['losses'][-1]:<12.4f}")

    # loss assert
    base_losses = results["none"]["losses"]
    tol = 1e-5
    for step in range(train_steps):
        base_val = base_losses[step]
        for mode in ["swap", "swap_with_policy"]:
            val = results[mode]["losses"][step]
            diff = abs(val - base_val)
            assert diff < tol, (
                f"Loss mismatch at step {step} in mode '{mode}': "
                f"none={base_val:.8f}, {mode}={val:.8f}, diff={diff:.2e}"
            )
    print(f"\n✅ All {train_steps} steps: losses are consistent across modes (tol={tol}).")

    # memory assert
    mem_none = results["none"]["peak_mem_gb"]
    mem_swap = results["swap"]["peak_mem_gb"]
    mem_swap_with_policy = results["swap_with_policy"]["peak_mem_gb"]

    # mem_none > mem_swap_with_policy
    assert mem_none > mem_swap_with_policy, \
        f"Expected NONE ({mem_none:.5f}) > SWAP_WITH_POLICY ({mem_swap_with_policy:.5f})"
    print(f"✅ Verified: NONE ({mem_none:.5f}) > SWAP_WITH_POLICY ({mem_swap_with_policy:.5f})")
    # mem_swap_with_policy > mem_swap
    assert mem_swap_with_policy > mem_swap, \
        f"Expected SWAP_WITH_POLICY ({mem_swap_with_policy:.5f}) > SWAP ({mem_swap:.5f})"
    print(f"✅ Verified: SWAP_WITH_POLICY ({mem_swap_with_policy:.5f}) > SWAP ({mem_swap:.5f})")

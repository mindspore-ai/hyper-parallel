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
import gc
import json
from pathlib import Path
import subprocess
import sys
import time
from contextlib import contextmanager

import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, mint
from hyper_parallel.core.activation_checkpoint import CheckpointPolicy, SwapManager, checkpoint_wrapper
from hyper_parallel.platform.mindspore.autograd_compat import enable_mindspore_backward_compat
enable_mindspore_backward_compat()


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Cell):
    """A simple Transformer block for testing purposes."""

    def __init__(self, dim=256, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm1 = mint.nn.LayerNorm((dim,))
        self.qkv = nn.Dense(dim, dim * 3, has_bias=False)
        self.out_proj = nn.Dense(dim, dim, has_bias=False)
        self.norm2 = mint.nn.LayerNorm((dim,))
        self.ffn1 = nn.Dense(dim, dim * 4, has_bias=False)
        self.ffn2 = nn.Dense(dim * 4, dim, has_bias=False)

    def construct(self, x):
        """TransformerBlock construct"""
        b, s, d = x.shape
        residual = x
        x = self.norm1(x)
        qkv = self.qkv(x).view(b, s, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # (3, B, heads, S, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = float(self.head_dim) ** -0.5
        attn = mint.bmm(
            q.reshape(-1, s, self.head_dim),
            k.reshape(-1, s, self.head_dim).transpose(-2, -1),
        ) * scale
        attn = mint.nn.functional.softmax(attn, dim=-1)
        out = mint.bmm(attn, v.reshape(-1, s, self.head_dim))
        out = out.view(b, self.num_heads, s, self.head_dim).transpose(0, 2, 1, 3).reshape(b, s, d)
        out = self.out_proj(out)
        x = residual + out
        residual = x
        x = self.norm2(x)
        x = self.ffn2(ms.ops.relu(self.ffn1(x)))
        x = residual + x
        return x


class SimpleTransformer(nn.Cell):
    """A simple Transformer model for testing purposes."""

    def __init__(self, vocab_size=32000, dim=512, depth=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.CellList([TransformerBlock(dim) for _ in range(depth)])
        self.norm = mint.nn.LayerNorm((dim,))
        self.head = nn.Dense(dim, vocab_size, has_bias=False)

    def construct(self, x):
        x = self.embed(x)
        num = range(len(self.layers))
        for i in num:
            x = self.layers[i](x)
        x = self.norm(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    ms.set_seed(seed)


@contextmanager
def seed_memory_time_context(seed=42):
    """Context manager to set seed, track peak memory and execution time."""
    set_seed(seed)
    ms.runtime.empty_cache()
    ms.runtime.reset_peak_memory_stats()
    start_time = time.time()

    stats = {}
    try:
        yield stats
    finally:
        exec_time = time.time() - start_time
        peak_mem_gb = ms.runtime.max_memory_allocated() / (1024 ** 3)
        stats["peak_mem"] = peak_mem_gb
        stats["exec_time"] = exec_time


def prepare_data(batch_size=4, seq_len=256, num_samples=32):
    """Create a list of (input_ids, label_ids) tensor pairs."""
    return [
        (
            Tensor(np.random.randint(0, 10000, (batch_size, seq_len)).astype(np.int32)),
            Tensor(np.random.randint(0, 10000, (batch_size, seq_len)).astype(np.int32)),
        )
        for _ in range(num_samples)
    ]


def train_one_mode(net, data_list, train_steps=3):
    """Run training for one mode and return per-step losses."""
    vocab_size = net.head.out_channels
    optimizer = nn.Adam(net.trainable_params(), learning_rate=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    def get_forward_fn(net):
        def forward_fn(x, y):
            logits = net(x)
            return loss_fn(logits.view(-1, vocab_size), y.view(-1))
        return forward_fn

    params = tuple(net.trainable_params())
    losses = []
    for step, (x, y) in enumerate(data_list):
        if step >= train_steps:
            break
        for param in params:
            param.grad = None
        loss = get_forward_fn(net)(x, y)
        loss.backward()
        grads = tuple(param.grad for param in params)
        optimizer(grads)
        losses.append(float(loss.asnumpy()))
    return losses


def run_one_mode(mode, train_steps=3, seed=42):
    """Build a fresh model and measure one mode in a clean interpreter."""
    ms.set_context(mode=ms.PYNATIVE_MODE)
    set_seed(seed)
    data_list = prepare_data()
    try:
        with seed_memory_time_context(seed=seed) as stats:
            base_model = SimpleTransformer(vocab_size=32000, dim=512, depth=16)
            model = apply_recompute(base_model, mode)
            losses = train_one_mode(model, data_list, train_steps)
        return {
            "mode": mode,
            "losses": losses,
            "peak_mem_gb": stats["peak_mem"],
            "time_sec": stats["exec_time"],
        }
    finally:
        gc.collect()
        ms.runtime.empty_cache()


def run_one_mode_in_subprocess(mode, train_steps=3, seed=42):
    """Run one mode in an isolated subprocess to avoid cross-mode cache residue."""
    project_root = Path(__file__).resolve().parents[4]
    command = [
        sys.executable,
        "-c",
        (
            "import json; "
            "from tests.mindspore.st.activation_checkpoint.ckpt_activation import run_one_mode; "
            f"result = run_one_mode({mode!r}, train_steps={train_steps}, seed={seed}); "
            "print('__AC_RESULT__' + json.dumps(result, sort_keys=True))"
        ),
    ]

    completed = subprocess.run(
        command,
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    for line in reversed(completed.stdout.splitlines()):
        marker = "__AC_RESULT__"
        if marker in line:
            start_index = line.find(marker)
            json_str = line[start_index + len(marker):]
            return json.loads(json_str)

    raise RuntimeError(
        f"Mode {mode!r} did not produce a result marker.\n"
        f"STDOUT:\n{completed.stdout}\n"
        f"STDERR:\n{completed.stderr}"
    )


# ---------------------------------------------------------------------------
# Activation checkpoint application
# ---------------------------------------------------------------------------

def apply_recompute(model, mode):
    """Apply activation checkpointing based on the specified mode."""
    if mode == "none":
        return model

    if mode == "recompute":
        def policy_fn(ctx, op, *args, **kwargs):  # pylint: disable=W0613
            return CheckpointPolicy.MUST_RECOMPUTE

        for i, layer in enumerate(model.layers):
            model.layers[i] = checkpoint_wrapper(layer, policy_fn=policy_fn)

    elif mode == "save":
        def policy_fn(ctx, op, *args, **kwargs):  # pylint: disable=W0613
            return CheckpointPolicy.MUST_SAVE

        for i, layer in enumerate(model.layers):
            model.layers[i] = checkpoint_wrapper(layer, policy_fn=policy_fn)


    elif mode == "swap":
        no_swap_ops = {
            "LayerNormExt"
        }
        def policy_fn(ctx, op, *args, **kwargs):  # pylint: disable=W0613
            if op.name in no_swap_ops:
                return CheckpointPolicy.MUST_RECOMPUTE
            return CheckpointPolicy.MUST_SWAP
        for i, layer in enumerate(model.layers):
            model.layers[i] = checkpoint_wrapper(layer, policy_fn=policy_fn, swap_inputs=True)

        for i in range(len(model.layers) - 1):
            SwapManager().set_forward_prefetch_layer(model.layers[i], model.layers[i + 1])

    elif mode == "funcrecompute":
        def policy_fn(ctx, op, *args, **kwargs):  # pylint: disable=W0613
            return CheckpointPolicy.MUST_RECOMPUTE
        model.layers[0].norm1 = checkpoint_wrapper(model.layers[0].norm1, policy_fn=policy_fn)
        model.layers[1].ffn1.matmul = checkpoint_wrapper(model.layers[1].ffn1.matmul, policy_fn=policy_fn)
        model.layers[2].ffn2.reshape = checkpoint_wrapper(model.layers[2].ffn2.reshape, policy_fn=policy_fn)

    elif mode == "funcsave":
        def policy_fn(ctx, op, *args, **kwargs):  # pylint: disable=W0613
            return CheckpointPolicy.MUST_SAVE
        model.layers[1].ffn1.matmul = checkpoint_wrapper(model.layers[1].ffn1.matmul, policy_fn=policy_fn)
        model.layers[1].ffn1.reshape = checkpoint_wrapper(model.layers[1].ffn1.reshape, policy_fn=policy_fn)
        model.layers[2] = checkpoint_wrapper(model.layers[2], policy_fn=policy_fn)
        model.layers[3].qkv.matmul = checkpoint_wrapper(model.layers[3].qkv.matmul, policy_fn=policy_fn)

    elif mode == "funcswap":
        def policy_fn(ctx, op, *args, **kwargs):  # pylint: disable=W0613
            return CheckpointPolicy.MUST_SWAP
        model.layers[0] = checkpoint_wrapper(model.layers[0], policy_fn=policy_fn)
        model.layers[1].ffn1.matmul = checkpoint_wrapper(model.layers[1].ffn1.matmul, policy_fn=policy_fn)
        model.layers[2].ffn1 = checkpoint_wrapper(model.layers[2].ffn1, policy_fn=policy_fn)
        model.layers[2].ffn2.reshape = checkpoint_wrapper(model.layers[2].ffn2.reshape, policy_fn=policy_fn)
        model.layers[3].qkv.matmul = checkpoint_wrapper(model.layers[3].qkv.matmul, policy_fn=policy_fn)

        for i in range(len(model.layers) - 1):
            SwapManager().set_forward_prefetch_layer(model.layers[i], model.layers[i + 1])

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return model


# ---------------------------------------------------------------------------
# Test case
# ---------------------------------------------------------------------------

def test_ac_memory_comparison():
    """
    Feature: Activation Checkpointing and Swapping Memory Behavior
    Description: Compare peak memory usage across four modes:
                 'none' (baseline), 'recompute' (full activation checkpointing),
                 'save' (partial activation saving), and 'swap' (fine-grained tensor swapping).
                 Validates that losses are numerically identical at every training step
                 and that peak memory follows the expected hierarchy:
                 NONE > SAVE > RECOMPUTE ≈ SWAP.
    Expectation: All modes produce consistent losses (within 1e-4 tolerance),
                 the memory usage trend is satisfied, and no OOM occurs.
    """
    print("Starting memory and time comparison: none vs recompute vs save vs swap")
    train_steps = 3

    modes = ["none", "recompute", "funcrecompute", "save", "funcsave", "swap", "funcswap"]
    results = {}

    for mode in modes:
        print(f"\n--- Running mode: {mode.upper()} ---")
        results[mode] = run_one_mode_in_subprocess(mode, train_steps=train_steps)
        peak_mem = results[mode]["peak_mem_gb"]
        duration = results[mode]["time_sec"]
        losses = results[mode]["losses"]
        print(f"{mode}: Loss={losses[-1]:.4f}, Peak Mem={peak_mem:.5f} GB, Time={duration:.5f}s")

    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(f"{'Mode':<12} | {'Peak Mem (GB)':<15} | {'Time (s)':<10} | {'Final Loss':<12}")
    print("-" * 70)
    for mode in modes:
        r = results[mode]
        print(f"{mode.upper():<12} | {r['peak_mem_gb']:<15.5f} | {r['time_sec']:<10.5f} | {r['losses'][-1]:<12.4f}")

    # loss consistency assertion
    base_losses = results["none"]["losses"]
    tol = 1e-4
    for step in range(train_steps):
        base_val = base_losses[step]
        for mode in ["recompute", "funcrecompute", "save", "funcsave", "swap", "funcswap"]:
            val = results[mode]["losses"][step]
            diff = abs(val - base_val)
            assert diff < tol, (
                f"Loss mismatch at step {step} in mode '{mode}': "
                f"none={base_val:.8f}, {mode}={val:.8f}, diff={diff:.2e}"
            )
    print(f"\nAll {train_steps} steps: losses are consistent across modes (tol={tol}).")

    # memory hierarchy assertion: none > recompute ≈ swap
    mem_none = results["none"]["peak_mem_gb"]
    mem_recompute = results["recompute"]["peak_mem_gb"]
    mem_swap = results["swap"]["peak_mem_gb"]

    assert mem_none > mem_recompute, \
        f"Expected SAVE ({mem_none:.5f} GB) > RECOMPUTE ({mem_recompute:.5f} GB)"
    print(f"Verified: SAVE ({mem_none:.5f}) > RECOMPUTE ({mem_recompute:.5f})")

    tol_mem = 0.15
    assert abs(mem_recompute - mem_swap) < tol_mem, \
        f"Expected RECOMPUTE ({mem_recompute:.5f} GB) ≈ SWAP ({mem_swap:.5f} GB)"
    print(f"Verified: RECOMPUTE ({mem_recompute:.5f}) ≈ SWAP ({mem_swap:.5f}) within {tol_mem:.2f} GB")

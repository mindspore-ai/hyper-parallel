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
"""Test activation swap memory comparison: none vs swap vs swap_with_policy."""
import gc
import json
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional

import mindspore as ms
import numpy as np
from hyper_parallel.core.activation_checkpoint import SwapManager, swap_wrapper
from hyper_parallel.core.activation_checkpoint.activation_checkpoint import CheckpointPolicy, swap
from hyper_parallel.platform.mindspore.autograd_compat import enable_mindspore_backward_compat
from mindspore import Tensor, mint, nn
enable_mindspore_backward_compat()


class SelfAttention(nn.Cell):
    """A simple attention cell that can be wrapped by activation swap."""

    def __init__(self, dim=256, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Dense(dim, dim * 3, has_bias=False)
        self.out_proj = nn.Dense(dim, dim, has_bias=False)

    def construct(self, x):
        """SelfAttention construct"""
        batch_size, seq_len, dim = x.shape
        qkv = self.qkv(x).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = float(self.head_dim) ** -0.5
        attn = mint.bmm(
            q.reshape(-1, seq_len, self.head_dim),
            k.reshape(-1, seq_len, self.head_dim).transpose(-2, -1),
        ) * scale
        attn = mint.nn.functional.softmax(attn, dim=-1)
        out = mint.bmm(attn, v.reshape(-1, seq_len, self.head_dim))
        out = out.view(batch_size, self.num_heads, seq_len, self.head_dim)
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, dim)
        return self.out_proj(out)


class TransformerBlock(nn.Cell):
    """A simple Transformer block for testing purposes."""

    def __init__(self, dim=256, num_heads=4):
        super().__init__()
        self.attn = SelfAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm((dim,))
        self.norm2 = nn.LayerNorm((dim,))
        self.ffn = nn.SequentialCell(
            nn.Dense(dim, dim * 4, has_bias=True),
            nn.ReLU(),
            nn.Dense(dim * 4, dim, has_bias=True),
        )

    def construct(self, x):
        x = x + self.attn(x)
        x = self.norm1(x)
        x = x + self.ffn(x)
        x = self.norm2(x)
        return x


class SimpleTransformer(nn.Cell):
    """A simple Transformer model for testing purposes."""

    def __init__(self, vocab_size=32000, dim=2048, depth=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.CellList([TransformerBlock(dim) for _ in range(depth)])
        self.norm = nn.LayerNorm((dim,))
        self.head = nn.Dense(dim, vocab_size, has_bias=False)

    def construct(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x)


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
        stats["exec_time"] = time.time() - start_time
        stats["peak_mem"] = ms.runtime.max_memory_allocated() / (1024 ** 3)


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


def apply_swap(model, mode):
    """Apply activation swap to the model based on the specified mode."""
    if mode == "none":
        return model

    if mode == "swap":
        for i, layer in enumerate(model.layers):
            model.layers[i] = swap_wrapper(layer)

    elif mode == "swap_with_policy":
        policy_threshold = 2048

        def policy_fn(x):
            if x.size <= policy_threshold:
                return CheckpointPolicy.MUST_SAVE
            return CheckpointPolicy.MUST_SWAP

        for i, layer in enumerate(model.layers):
            model.layers[i] = swap_wrapper(layer, policy_fn)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    for i in range(len(model.layers) - 1):
        SwapManager().set_forward_prefetch_layer(model.layers[i], model.layers[i + 1])
    return model


def run_one_mode(mode, train_steps=3, seed=42):
    """Build a fresh model and measure one mode in a clean interpreter."""
    ms.set_context(mode=ms.PYNATIVE_MODE)
    set_seed(seed)
    data_list = prepare_data()
    try:
        with seed_memory_time_context(seed=seed) as stats:
            base_model = SimpleTransformer(vocab_size=32000, dim=512, depth=16)
            model = apply_swap(base_model, mode)
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
            "from tests.mindspore.st.activation_checkpoint.swap_activation import run_one_mode; "
            f"result = run_one_mode({mode!r}, train_steps={train_steps}, seed={seed}); "
            "print('__ACT_SWAP_RESULT__' + json.dumps(result, sort_keys=True))"
        ),
    ]
    completed = subprocess.run(
        command,
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    marker = "__ACT_SWAP_RESULT__"
    for line in reversed(completed.stdout.splitlines()):
        if marker in line:
            return json.loads(line[line.find(marker) + len(marker):])

    raise RuntimeError(
        f"Mode {mode!r} did not produce a result marker.\n"
        f"STDOUT:\n{completed.stdout}\n"
        f"STDERR:\n{completed.stderr}"
    )


def test_act_swap_memory_comparison():
    """
    Feature: Activation Swap Memory Behavior
    Description: Compare peak memory usage across three modes:
                 'none' (baseline),
                 'swap' (swap all attention saved tensors),
                 'swap_with_policy' (swap part of tensors according to the policy function).
                 Validate that losses are numerically identical at every training step
                 and that peak memory follows the expected hierarchy.
    Expectation: The peak memory usage should follow the hierarchy:
                 NONE > SWAP_WITH_POLICY > SWAP,
                 and all modes should produce consistent losses without OOM.
    """
    print("Starting memory and time comparison: none vs swap vs swap_with_policy")
    train_steps = 3

    modes = ["none", "swap", "swap_with_policy"]
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
    print(f"{'Mode':<20} | {'Peak Mem (GB)':<15} | {'Time (s)':<10} | {'Final Loss':<12}")
    print("-" * 70)
    for mode in modes:
        result = results[mode]
        print(
            f"{mode.upper():<20} | {result['peak_mem_gb']:<15.5f} | "
            f"{result['time_sec']:<10.5f} | {result['losses'][-1]:<12.4f}"
        )

    base_losses = results["none"]["losses"]
    tol = 1e-4
    for step in range(train_steps):
        base_val = base_losses[step]
        for mode in ["swap", "swap_with_policy"]:
            val = results[mode]["losses"][step]
            diff = abs(val - base_val)
            assert diff < tol, (
                f"Loss mismatch at step {step} in mode '{mode}': "
                f"none={base_val:.8f}, {mode}={val:.8f}, diff={diff:.2e}"
            )
    print(f"\nAll {train_steps} steps: losses are consistent across modes (tol={tol}).")

    mem_none = results["none"]["peak_mem_gb"]
    mem_swap = results["swap"]["peak_mem_gb"]
    mem_swap_with_policy = results["swap_with_policy"]["peak_mem_gb"]

    assert mem_none > mem_swap_with_policy, (
        f"Expected NONE ({mem_none:.5f}) > SWAP_WITH_POLICY ({mem_swap_with_policy:.5f})"
    )
    print(f"Verified: NONE ({mem_none:.5f}) > SWAP_WITH_POLICY ({mem_swap_with_policy:.5f})")

    assert mem_swap_with_policy > mem_swap, (
        f"Expected SWAP_WITH_POLICY ({mem_swap_with_policy:.5f}) > SWAP ({mem_swap:.5f})"
    )
    print(f"Verified: SWAP_WITH_POLICY ({mem_swap_with_policy:.5f}) > SWAP ({mem_swap:.5f})")



class _SwapFnTransformer(SimpleTransformer):
    """SimpleTransformer that uses the swap() interface per layer call in construct.

    Overrides construct() to call swap(layer, x) instead of layer(x), enabling
    activation offload to CPU via the async_save_on_cpu context manager.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        dim: int = 2048,
        depth: int = 16,
        policy_fn: Optional[Callable] = None,
    ):
        super().__init__(vocab_size, dim, depth)
        self._policy_fn = policy_fn

    def construct(self, x):
        """Forward pass using swap() for each transformer layer.

        Args:
            x: Input token ids tensor of shape (batch, seq_len).

        Returns:
            Logit tensor of shape (batch, seq_len, vocab_size).
        """
        x = self.embed(x)
        for layer in self.layers:
            x = swap(layer, x, policy_fn=self._policy_fn)
        x = self.norm(x)
        return self.head(x)


def run_one_swap_fn_mode(mode: str, train_steps: int = 3, seed: int = 42) -> dict:
    """Build a fresh swap-function model and run training for one mode.

    Args:
        mode: One of 'none', 'swap_fn', or 'swap_fn_with_policy'.
        train_steps: Number of training steps to run.
        seed: Random seed for reproducibility.

    Returns:
        dict with keys 'mode', 'losses', 'peak_mem_gb', 'time_sec'.

    Raises:
        ValueError: If mode is not recognised.
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    set_seed(seed)
    data_list = prepare_data()
    try:
        with seed_memory_time_context(seed=seed) as stats:
            vocab_size, dim, depth = 32000, 2048, 6

            if mode == "none":
                model = SimpleTransformer(vocab_size=vocab_size, dim=dim, depth=depth)
            elif mode == "swap_fn":
                model = _SwapFnTransformer(vocab_size=vocab_size, dim=dim, depth=depth)
                for i in range(len(model.layers) - 1):
                    SwapManager().set_forward_prefetch_layer(model.layers[i], model.layers[i + 1])
            elif mode == "swap_fn_with_policy":
                policy_threshold = 32 * 512 * 512

                def _size_policy(x):
                    if x.size <= policy_threshold:
                        return CheckpointPolicy.MUST_SAVE
                    return CheckpointPolicy.MUST_SWAP

                model = _SwapFnTransformer(
                    vocab_size=vocab_size, dim=dim, depth=depth, policy_fn=_size_policy
                )
                for i in range(len(model.layers) - 1):
                    SwapManager().set_forward_prefetch_layer(model.layers[i], model.layers[i + 1])
            else:
                raise ValueError(f"Unknown mode: {mode!r}")

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


def run_one_swap_fn_mode_in_subprocess(mode: str, train_steps: int = 3, seed: int = 42) -> dict:
    """Run a single swap-function mode in an isolated subprocess.

    Subprocess isolation prevents cross-mode device memory residue from
    affecting peak memory statistics.

    Args:
        mode: Training mode to run.
        train_steps: Number of training steps.
        seed: Random seed.

    Returns:
        dict with training results parsed from subprocess stdout.

    Raises:
        RuntimeError: If the subprocess exits with a non-zero return code or
            does not emit the expected result marker.
    """
    project_root = Path(__file__).resolve().parents[4]
    command = [
        sys.executable,
        "-c",
        (
            "import json; "
            "from tests.mindspore.st.activation_checkpoint.swap_activation import "
            "run_one_swap_fn_mode; "
            f"result = run_one_swap_fn_mode({mode!r}, train_steps={train_steps}, seed={seed}); "
            "print('__ACT_SWAP_FN_RESULT__' + json.dumps(result, sort_keys=True))"
        ),
    ]
    completed = subprocess.run(
        command,
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    marker = "__ACT_SWAP_FN_RESULT__"
    for line in reversed(completed.stdout.splitlines()):
        if marker in line:
            return json.loads(line[line.find(marker) + len(marker):])

    raise RuntimeError(
        f"Mode {mode!r} did not produce a result marker.\n"
        f"STDOUT:\n{completed.stdout}\n"
        f"STDERR:\n{completed.stderr}"
    )


def test_act_swap_function_mode():
    """
    Feature: swap() Function Interface
    Description: Test the swap() interface by comparing training across three modes:
                 'none' (baseline), 'swap_fn' (offload all eligible tensors per layer
                 call via swap()), 'swap_fn_with_policy' (offload only tensors whose
                 element count exceeds a size threshold).
                 Validate that losses are numerically identical at every training step.
    Expectation: All modes produce consistent losses (within 1e-4 tolerance) and no OOM.
    """
    print("Starting swap() function interface test: none vs swap_fn vs swap_fn_with_policy")
    train_steps = 3

    modes = ["none", "swap_fn", "swap_fn_with_policy"]
    results = {}

    for mode in modes:
        print(f"\n--- Running mode: {mode.upper()} ---")
        results[mode] = run_one_swap_fn_mode_in_subprocess(mode, train_steps=train_steps)
        peak_mem = results[mode]["peak_mem_gb"]
        duration = results[mode]["time_sec"]
        losses = results[mode]["losses"]
        print(f"{mode}: Loss={losses[-1]:.4f}, Peak Mem={peak_mem:.5f} GB, Time={duration:.5f}s")

    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(f"{'Mode':<25} | {'Peak Mem (GB)':<15} | {'Time (s)':<10} | {'Final Loss':<12}")
    print("-" * 70)
    for mode in modes:
        result = results[mode]
        print(
            f"{mode.upper():<25} | {result['peak_mem_gb']:<15.5f} | "
            f"{result['time_sec']:<10.5f} | {result['losses'][-1]:<12.4f}"
        )

    base_losses = results["none"]["losses"]
    tol = 1e-4
    for step in range(train_steps):
        base_val = base_losses[step]
        for mode in ["swap_fn", "swap_fn_with_policy"]:
            val = results[mode]["losses"][step]
            diff = abs(val - base_val)
            assert diff < tol, (
                f"Loss mismatch at step {step} in mode '{mode}': "
                f"none={base_val:.8f}, {mode}={val:.8f}, diff={diff:.2e}"
            )
    print(f"\nAll {train_steps} steps: losses are consistent across modes (tol={tol}).")

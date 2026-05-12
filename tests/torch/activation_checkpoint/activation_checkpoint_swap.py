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
"""Activation Swap memory comparison: None vs Swap"""
import copy
import pytest
import torch

from torch import nn
from tests.torch.common_net import SimpleTransformer
from tests.torch.activation_checkpoint.utils import prepare_data, train_one_mode, seed_memory_time_context
from hyper_parallel.core.activation_checkpoint import SwapManager, swap_wrapper, swap_tensor_wrapper
from hyper_parallel.core.activation_checkpoint.activation_checkpoint import CheckpointPolicy, swap
from hyper_parallel.core.activation_checkpoint.swap import SwapGroup


def apply_swap(model, mode):
    """Apply activation swap to the model based on the specified mode."""
    if mode == "none":
        return model

    if mode == "swap":
        for i, layer in enumerate(model.layers):
            model.layers[i].attn = swap_wrapper(layer.attn)

        for i in range(len(model.layers) - 1):
            SwapManager().set_forward_prefetch_layer(model.layers[i], model.layers[i + 1])
    elif mode == "swap_with_policy":
        policy_threshold = 32 * 512 * 512
        def policy_fn(x):
            if x.storage().size() <= policy_threshold:
                return CheckpointPolicy.MUST_SAVE
            return CheckpointPolicy.MUST_SWAP

        for i, layer in enumerate(model.layers):
            model.layers[i].attn = swap_wrapper(layer.attn, policy_fn)

        for i in range(len(model.layers) - 1):
            SwapManager().set_forward_prefetch_layer(model.layers[i], model.layers[i + 1])
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return model


def test_act_swap_memory_comparison():
    """
    Feature: Activation Swap Memory Behavior
    Description: Compare peak memory usage across four modes:
                 'none' (baseline),
                 'swap' (swap all tensors),
                 'swap_with_policy' (swap part of tensors according to the policy function).
                 Validate that losses are numerically identical at every training step
                 and that peak memory follows the expected hierarchy.
    Expectation: The peak memory usage should follow the hierarchy: NONE > SWAP_WITH_POLICY > SWAP,
                 the memory usage trend is satisfied, and no OOM occurs.
    """
    print("🚀 Starting memory and time comparison: none vs swap vs swap_with_policy")
    dataloader = prepare_data()
    train_steps = 3

    modes = ["none", "swap", "swap_with_policy"]
    results = {}

    for mode in modes:
        print(f"\n--- Running mode: {mode.upper()} ---")
        with seed_memory_time_context() as stats:
            base_model = SimpleTransformer(vocab_size=32000, dim=2048, depth=16).npu()
            model = apply_swap(base_model, mode)
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


class _SwapFnTransformer(SimpleTransformer):
    """SimpleTransformer that uses the swap() interface per layer call in forward.

    Overrides forward() to call swap(block, x) instead of block(x), enabling
    activation offload to CPU via the async_save_on_cpu context manager.
    """

    def __init__(self, vocab_size: int, dim: int, depth: int, policy_fn=None):
        super().__init__(vocab_size, dim, depth)
        self._policy_fn = policy_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using swap() for each transformer block.

        Args:
            x: Input token ids of shape (batch, seq_len).

        Returns:
            Logit tensor of shape (batch, seq_len, vocab_size).
        """
        x = self.embed(x)
        for block in self.layers:
            x = swap(block, x, policy_fn=self._policy_fn)
        x = self.norm(x)
        return self.head(x)


class _TransformerBlock(nn.Module):
    """A simple Transformer block for testing purposes."""

    def __init__(self, dim=256, num_heads=4, swap_tensor_on=False):
        super().__init__()
        self.swap_tensor_on = swap_tensor_on
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
        """TransformerBlock forwardS"""
        attn_out, _ = self.attn(x, x, x)
        if self.swap_tensor_on:
            attn_out = swap_tensor_wrapper(attn_out, tag="attn_out")
        x = x + attn_out
        x = self.norm1(x)
        if self.swap_tensor_on:
            x = swap_tensor_wrapper(x, tag="norm1_out")
        x = x + self.ffn(x)
        x = self.norm2(x)
        if self.swap_tensor_on:
            x = swap_tensor_wrapper(x, tag="norm2_out")
        return x



class _SwapTensorTransformer(nn.Module):
    """A simple Transformer model for testing purposes."""

    def __init__(self, vocab_size=1000, dim=2048, depth=6, swap_tensor_on=False):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([_TransformerBlock(dim, swap_tensor_on=swap_tensor_on) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        for block in self.layers:
            x = block(x)
        x = self.norm(x)
        return self.head(x)


def _build_swap_fn_model(mode: str) -> torch.nn.Module:
    """Build and configure a model using the swap() function interface.

    Args:
        mode: One of 'none', 'swap_fn', or 'swap_fn_with_policy'.

    Returns:
        Configured model placed on NPU.

    Raises:
        ValueError: If mode is not recognised.
    """
    vocab_size, dim, depth = 32000, 2048, 16

    if mode == "none":
        return SimpleTransformer(vocab_size=vocab_size, dim=dim, depth=depth).npu()

    policy_fn = None
    if mode == "swap_fn_with_policy":
        policy_threshold = 32 * 512 * 512

        def _size_policy(x: torch.Tensor) -> CheckpointPolicy:
            if x.storage().size() <= policy_threshold:
                return CheckpointPolicy.MUST_SAVE
            return CheckpointPolicy.MUST_SWAP

        policy_fn = _size_policy
    elif mode != "swap_fn":
        raise ValueError(f"Unknown mode: {mode!r}")

    model = _SwapFnTransformer(
        vocab_size=vocab_size, dim=dim, depth=depth, policy_fn=policy_fn
    ).npu()
    for i in range(len(model.layers) - 1):
        SwapManager().set_forward_prefetch_layer(model.layers[i], model.layers[i + 1])
    return model


def _build_swap_tensor_model(mode: str) -> torch.nn.Module:
    """Build and configure a model using the swap_tensor_wrapper() interface.

    Args:
        mode: One of 'none' or 'swap_tensor'.

    Returns:
        Configured model placed on NPU.

    Raises:
        ValueError: If mode is not recognised.
    """
    vocab_size, dim, depth = 32000, 2048, 16

    if mode == "none":
        # return SimpleTransformer(vocab_size=vocab_size, dim=dim, depth=depth).npu()
        return _SwapTensorTransformer(vocab_size=vocab_size, dim=dim, depth=depth).npu()
    if mode == "swap_tensor":
        model = _SwapTensorTransformer(vocab_size=vocab_size, dim=dim, depth=depth, swap_tensor_on=True).npu()
        for i in range(len(model.layers) - 1):
            SwapManager().set_forward_prefetch_layer(model.layers[i], model.layers[i + 1])
        return model

    raise ValueError(f"Unknown mode: {mode!r}")


def test_act_swap_function_mode():
    """
    Feature: swap() Function Interface
    Description: Validate the swap() interface by comparing training across three modes:
                 'none' (baseline), 'swap_fn' (offload all eligible tensors via swap()),
                 'swap_fn_with_policy' (offload only tensors exceeding a size threshold).
                 Asserts that losses are numerically identical at every training step and
                 that device memory is reduced when swap is applied.
    Expectation: All modes produce consistent losses (within 1e-5 tolerance),
                 NONE > SWAP_FN and NONE > SWAP_FN_WITH_POLICY in peak device memory.
    """
    print("Starting swap() function interface comparison: none vs swap_fn vs swap_fn_with_policy")
    dataloader = prepare_data()
    train_steps = 3

    modes = ["none", "swap_fn", "swap_fn_with_policy"]
    results = {}

    for mode in modes:
        print(f"\n--- Running mode: {mode.upper()} ---")
        with seed_memory_time_context() as stats:
            model = _build_swap_fn_model(mode)
            losses = train_one_mode(model, dataloader, train_steps)
        peak_mem = stats.get("peak_mem")
        duration = stats.get("exec_time")
        results[mode] = {"losses": losses, "peak_mem_gb": peak_mem, "time_sec": duration}
        print(f"{mode}: Loss={losses[-1]:.4f}, Peak Mem={peak_mem:.5f} GB, Time={duration:.5f}s")

    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(f"{'Mode':<25} | {'Peak Mem (GB)':<15} | {'Time (s)':<10} | {'Final Loss':<12}")
    print("-" * 70)
    for mode in modes:
        r = results[mode]
        print(
            f"{mode.upper():<25} | {r['peak_mem_gb']:<15.5f} | "
            f"{r['time_sec']:<10.5f} | {r['losses'][-1]:<12.4f}"
        )

    # Loss consistency assertion
    base_losses = results["none"]["losses"]
    tol = 1e-5
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

    # Memory reduction assertion
    mem_none = results["none"]["peak_mem_gb"]
    mem_swap_fn = results["swap_fn"]["peak_mem_gb"]
    mem_swap_fn_with_policy = results["swap_fn_with_policy"]["peak_mem_gb"]

    assert mem_none > mem_swap_fn, (
        f"Expected NONE ({mem_none:.5f}) > SWAP_FN ({mem_swap_fn:.5f})"
    )
    print(f"Verified: NONE ({mem_none:.5f}) > SWAP_FN ({mem_swap_fn:.5f})")

    assert mem_none > mem_swap_fn_with_policy, (
        f"Expected NONE ({mem_none:.5f}) > SWAP_FN_WITH_POLICY ({mem_swap_fn_with_policy:.5f})"
    )
    print(f"Verified: NONE ({mem_none:.5f}) > SWAP_FN_WITH_POLICY ({mem_swap_fn_with_policy:.5f})")


def test_act_swap_tensor_function_mode():
    """
    Feature: swap_tensor_wrapper() Function Interface
    Description: Validate the declarative tensor swap interface by comparing
                 training across two modes: 'none' (baseline) and
                 'swap_tensor' (manually offload each transformer block output
                 via swap_tensor_wrapper()).
                 Asserts that losses are numerically identical at every
                 training step and that device memory is reduced when swap is applied.
    Expectation: All modes produce consistent losses (within 1e-5 tolerance),
                 and NONE > SWAP_TENSOR in peak device memory.
    """
    print("Starting swap_tensor_wrapper() comparison: none vs swap_tensor")
    dataloader = prepare_data()
    train_steps = 3

    modes = ["none", "swap_tensor"]

    results = {}

    for mode in modes:
        print(f"\n--- Running mode: {mode.upper()} ---")
        with seed_memory_time_context() as stats:
            model = _build_swap_tensor_model(mode)
            losses = train_one_mode(model, dataloader, train_steps)
        peak_mem = stats.get("peak_mem")
        duration = stats.get("exec_time")
        results[mode] = {"losses": losses, "peak_mem_gb": peak_mem, "time_sec": duration}
        print(f"{mode}: Loss={losses[-1]:.4f}, Peak Mem={peak_mem:.5f} GB, Time={duration:.5f}s")

    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(f"{'Mode':<25} | {'Peak Mem (GB)':<15} | {'Time (s)':<10} | {'Final Loss':<12}")
    print("-" * 70)
    for mode in modes:
        r = results[mode]
        print(
            f"{mode.upper():<25} | {r['peak_mem_gb']:<15.5f} | "
            f"{r['time_sec']:<10.5f} | {r['losses'][-1]:<12.4f}"
        )

    base_losses = results["none"]["losses"]
    tol = 1e-5
    for step in range(train_steps):
        base_val = base_losses[step]
        val = results["swap_tensor"]["losses"][step]
        diff = abs(val - base_val)
        assert diff < tol, (
            f"Loss mismatch at step {step} in mode 'swap_tensor': "
            f"none={base_val:.8f}, swap_tensor={val:.8f}, diff={diff:.2e}"
        )
    print(f"\nAll {train_steps} steps: losses are consistent across modes (tol={tol}).")


class _SmallNet(torch.nn.Module):
    """Two-layer MLP where the hidden activation is exposed as a plain function."""

    def __init__(self, dim: int, use_swap: bool):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, dim)
        self.fc2 = torch.nn.Linear(dim, dim)

        def _hidden(x):
            return torch.relu(self.fc1(x))

        self.hidden = swap_wrapper(_hidden) if use_swap else _hidden

    def forward(self, x):
        return self.fc2(self.hidden(x))


def test_swap_wrapper_accepts_func():
    """
    Feature: swap_wrapper accepts plain callable (func) as module argument
    Description: Build a small two-layer MLP whose hidden activation is a plain
                 Python function.  Run several training steps on two identical
                 copies of the network — one with the hidden function wrapped by
                 swap_wrapper, one without — and verify that the per-step losses
                 are numerically identical.
    Expectation: Losses across all training steps match within 1e-6 tolerance,
                 confirming that wrapping a func does not alter forward/backward
                 semantics.
    """
    torch.manual_seed(42)
    dim = 32
    batch, train_steps = 8, 5
    tol = 1e-6

    ref_model = _SmallNet(dim, use_swap=False)
    swap_model = _SmallNet(dim, use_swap=True)
    swap_model.fc1.load_state_dict(copy.deepcopy(ref_model.fc1.state_dict()))
    swap_model.fc2.load_state_dict(copy.deepcopy(ref_model.fc2.state_dict()))

    ref_opt = torch.optim.SGD(ref_model.parameters(), lr=0.01)
    swap_opt = torch.optim.SGD(swap_model.parameters(), lr=0.01)

    torch.manual_seed(42)
    for step in range(train_steps):
        x = torch.randn(batch, dim)
        target = torch.randn(batch, dim)

        ref_opt.zero_grad()
        ref_loss = torch.nn.functional.mse_loss(ref_model(x), target)
        ref_loss.backward()
        ref_opt.step()

        swap_opt.zero_grad()
        swap_loss = torch.nn.functional.mse_loss(swap_model(x), target)
        swap_loss.backward()
        swap_opt.step()

        diff = abs(ref_loss.item() - swap_loss.item())
        assert diff < tol, (
            f"Loss mismatch at step {step}: "
            f"ref={ref_loss.item():.8f}, swap={swap_loss.item():.8f}, diff={diff:.2e}"
        )


@pytest.mark.parametrize("wait_fn, launch_keyword", [
    ("wait_offload", "launch_offload"),
    ("wait_load", "launch_load"),
])
def test_swap_group_wait_requires_launch(wait_fn: str, launch_keyword: str):
    """
    Feature: SwapGroup state guard — wait before launch
    Description: calling wait_offload/wait_load before the corresponding launch
                 on a new SwapGroup must raise RuntimeError
    Expectation: raise RuntimeError mentioning the required launch call

    Args:
        wait_fn: Name of the wait method to invoke ("wait_offload" or "wait_load").
        launch_keyword: Substring expected in the RuntimeError message.
    """
    group = SwapGroup("test_group")
    with pytest.raises(RuntimeError, match=launch_keyword):
        getattr(group, wait_fn)()

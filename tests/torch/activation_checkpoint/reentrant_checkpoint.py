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
"""Torch correctness, performance, and memory checks for reentrant exclude."""
import gc
import statistics
import time

import psutil
import pytest
import torch
from torch import nn

from hyper_parallel.platform.torch.activation_checkpoint.checkpoint_wrapper import ckpt_wrapper
from hyper_parallel.platform.torch.activation_checkpoint.reentrant_checkpoint import (
    reentrant_checkpoint_exclude_wrapper,
    reentrant_checkpoint_wrapper,
)


_MIB = 1024 * 1024


def _get_device() -> torch.device:
    """Return the active accelerator, falling back to CPU for correctness."""
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _require_accelerator() -> torch.device:
    """Return an accelerator device or skip the hardware-only test."""
    device = _get_device()
    if device.type == "cpu":
        pytest.skip("device-memory validation requires an NPU or GPU")
    return device


def _device_module(device: torch.device) -> object:
    """Return the Torch device module exposing memory statistics."""
    return getattr(torch, device.type)


def _synchronize(device: torch.device) -> None:
    """Wait for outstanding accelerator work when needed."""
    if device.type != "cpu":
        _device_module(device).synchronize()


class _Middle(nn.Module):
    """Small parameterized nonlinear region that can be excluded."""

    def __init__(self, calls: dict) -> None:
        """Initialize a deterministic scale and shared call counter."""
        super().__init__()
        self.calls = calls
        self.scale = nn.Parameter(torch.tensor(0.9, dtype=torch.float32))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Apply the excluded nonlinear operation.

        Args:
            value: Input activation.
        """
        self.calls["middle"] += 1
        return torch.tanh(value * self.scale)


class _Workload(nn.Module):
    """Pointwise workload with an excluded region between two op sequences."""

    def __init__(self, calls: dict, depth: int, exclude_middle: bool) -> None:
        """Initialize the workload."""
        super().__init__()
        middle = _Middle(calls)
        self.middle = reentrant_checkpoint_exclude_wrapper(middle) if exclude_middle else middle
        self.depth = depth

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Return a scalar loss with enough saved activations for memory checks.

        Args:
            value: Input activation.
        """
        for _ in range(self.depth):
            value = torch.sin(value * 1.01) + value * 0.01
        value = self.middle(value)
        for _ in range(self.depth):
            value = torch.cos(value * 0.99) + value * 0.01
        return (value * value).mean()


class _DispatchWorkload(nn.Module):
    """Small-tensor workload dominated by eager host dispatch."""

    def __init__(self, depth: int = 40) -> None:
        """Store the number of pointwise operator pairs."""
        super().__init__()
        self.depth = depth

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Dispatch a fixed sequence of lightweight operators.

        Args:
            value: Input activation.
        """
        for _ in range(self.depth):
            value = torch.sin(value) + value * 0.01
        return value


def _run_step(module: nn.Module, device: torch.device, shape: tuple[int, ...]) -> None:
    """Run one complete backward step and promptly release graph roots."""
    module.zero_grad(set_to_none=True)
    value = torch.linspace(-1.0, 1.0, steps=int(torch.tensor(shape).prod()), device=device)
    value = value.reshape(shape).requires_grad_(True)
    loss = module(value)
    loss.backward()
    del loss
    del value


def _forward_peak_bytes(module: nn.Module, device: torch.device, shape: tuple[int, ...]) -> int:
    """Measure device allocations added by one forward graph."""
    device_api = _device_module(device)
    gc.collect()
    device_api.empty_cache()
    value = torch.linspace(-1.0, 1.0, steps=int(torch.tensor(shape).prod()), device=device)
    value = value.reshape(shape).requires_grad_(True)
    _synchronize(device)
    baseline = device_api.memory_allocated()
    device_api.reset_peak_memory_stats()
    output = module(value)
    _synchronize(device)
    peak_delta = device_api.max_memory_allocated() - baseline
    del output
    del value
    gc.collect()
    device_api.empty_cache()
    return peak_delta


def _measure_dispatch_seconds(module: nn.Module, value: torch.Tensor, iterations: int) -> float:
    """Return median host submission time for forward-only iterations."""
    device = value.device
    for _ in range(5):
        output = module(value)
        del output
    _synchronize(device)
    samples = []
    for _ in range(3):
        start = time.perf_counter()
        for _ in range(iterations):
            output = module(value)
            del output
        samples.append(time.perf_counter() - start)
        _synchronize(device)
        gc.collect()
    return statistics.median(samples)


def test_reentrant_exclude_result_correctness() -> None:
    """Values, input gradients, and parameter gradients should match eager mode."""
    device = _get_device()
    reference_calls = {"middle": 0}
    actual_calls = {"middle": 0}
    reference = _Workload(reference_calls, depth=3, exclude_middle=False).to(device)
    actual = reentrant_checkpoint_wrapper(_Workload(actual_calls, depth=3, exclude_middle=True)).to(device)
    actual.middle.scale.data.copy_(reference.middle.scale.data)
    reference_input = torch.linspace(-1.0, 1.0, steps=1024, device=device).reshape(16, 64).requires_grad_(True)
    actual_input = reference_input.detach().clone().requires_grad_(True)

    reference_loss = reference(reference_input)
    reference_loss.backward()
    actual_loss = actual(actual_input)
    actual_loss.backward()

    torch.testing.assert_close(actual_loss, reference_loss, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual_input.grad, reference_input.grad, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(actual.middle.scale.grad, reference.middle.scale.grad, atol=1e-6, rtol=1e-5)
    assert reference_calls == {"middle": 1}
    assert actual_calls == {"middle": 1}


def test_reentrant_checkpoint_device_peak_memory() -> None:
    """Reentrant forward should retain materially less device activation memory."""
    device = _require_accelerator()
    reference = _Workload({"middle": 0}, depth=8, exclude_middle=False).to(device)
    actual = reentrant_checkpoint_wrapper(_Workload({"middle": 0}, depth=8, exclude_middle=True)).to(device)
    shape = (1024, 1024)

    reference_peak = _forward_peak_bytes(reference, device, shape)
    actual_peak = _forward_peak_bytes(actual, device, shape)
    print(
        "Torch reentrant forward peak: "
        f"actual={actual_peak / _MIB:.2f} MiB, reference={reference_peak / _MIB:.2f} MiB, "
        f"ratio={actual_peak / reference_peak:.3f}"
    )

    assert actual_peak < reference_peak * 0.7, (
        f"reentrant peak memory did not decrease enough: actual={actual_peak / _MIB:.2f} MiB, "
        f"reference={reference_peak / _MIB:.2f} MiB"
    )


def test_reentrant_checkpoint_host_dispatch_performance() -> None:
    """Hook-free forward dispatch should be faster than non-reentrant checkpoint."""
    device = _get_device()
    value = torch.linspace(-1.0, 1.0, steps=64, device=device).requires_grad_(True)
    non_reentrant = ckpt_wrapper(_DispatchWorkload())
    reentrant = reentrant_checkpoint_wrapper(_DispatchWorkload())

    non_reentrant_seconds = _measure_dispatch_seconds(non_reentrant, value, iterations=20)
    reentrant_seconds = _measure_dispatch_seconds(reentrant, value, iterations=20)
    print(
        "Torch checkpoint host dispatch: "
        f"reentrant={reentrant_seconds:.6f}s, non_reentrant={non_reentrant_seconds:.6f}s, "
        f"speedup={non_reentrant_seconds / reentrant_seconds:.3f}x"
    )

    assert reentrant_seconds < non_reentrant_seconds, (
        f"hook-free host dispatch was not faster: reentrant={reentrant_seconds:.6f}s, "
        f"non_reentrant={non_reentrant_seconds:.6f}s"
    )


def test_reentrant_checkpoint_device_memory_no_leak() -> None:
    """Repeated backward steps should not retain device graph allocations."""
    device = _require_accelerator()
    device_api = _device_module(device)
    module = reentrant_checkpoint_wrapper(_Workload({"middle": 0}, depth=3, exclude_middle=True)).to(device)
    samples = []
    for step in range(20):
        _run_step(module, device, (64, 64))
        _synchronize(device)
        if step >= 5:
            gc.collect()
            samples.append(device_api.memory_allocated())

    growth = max(samples) - min(samples)
    print(f"Torch steady-state device allocation range: {growth / _MIB:.2f} MiB")
    assert max(samples) - min(samples) <= 4 * _MIB, (
        f"device allocations grew across steady-state steps: {[value / _MIB for value in samples]}"
    )


def test_reentrant_checkpoint_host_memory_no_leak() -> None:
    """Repeated checkpoint graphs should not cause sustained process RSS growth."""
    device = _get_device()
    module = reentrant_checkpoint_wrapper(_Workload({"middle": 0}, depth=2, exclude_middle=True)).to(device)
    process = psutil.Process()
    for _ in range(10):
        _run_step(module, device, (32, 32))
    _synchronize(device)
    gc.collect()
    baseline_rss = process.memory_info().rss

    for _ in range(40):
        _run_step(module, device, (32, 32))
    _synchronize(device)
    gc.collect()
    growth = process.memory_info().rss - baseline_rss

    print(f"Torch steady-state host RSS growth: {growth / _MIB:.2f} MiB")
    assert growth <= 16 * _MIB, f"host RSS grew by {growth / _MIB:.2f} MiB"

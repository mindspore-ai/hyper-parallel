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
"""MindSpore correctness, performance, and memory checks for reentrant exclude."""
import gc
import math
import statistics
import time

import numpy as np
import psutil
import pytest
import mindspore as ms
from mindspore import nn, ops, Tensor
from mindspore.graph.api import _pynative_executor

from hyper_parallel.platform.mindspore.activation_checkpoint.checkpoint_wrapper import ckpt_wrapper
from hyper_parallel.platform.mindspore.activation_checkpoint.reentrant_checkpoint import (
    reentrant_checkpoint_exclude_wrapper,
    reentrant_checkpoint_wrapper,
)
from hyper_parallel.platform.mindspore.autograd_compat import enable_mindspore_backward_compat


_MIB = 1024 * 1024

enable_mindspore_backward_compat()
ms.set_context(mode=ms.PYNATIVE_MODE)


def _enable_compat_grad_recording() -> None:
    """Restore grad flags cleared by the PyNative executor reset."""
    _pynative_executor.set_enable_grad(True)
    _pynative_executor.set_grad_flag(True)


@pytest.fixture(autouse=True)
def _clear_pynative_state():
    """Isolate PyNative graphs and allocator state between test cases."""
    _pynative_executor.clear_res()
    _enable_compat_grad_recording()
    yield
    _pynative_executor.clear_res()
    gc.collect()
    ms.runtime.empty_cache()


def _require_accelerator() -> None:
    """Skip hardware-only checks when MindSpore targets CPU."""
    if ms.get_context("device_target") == "CPU":
        pytest.skip("device-memory validation requires an NPU or GPU")


class _Middle(nn.Cell):
    """Small parameterized nonlinear region that can be excluded."""

    def __init__(self, calls: dict) -> None:
        """Initialize a deterministic scale and shared call counter."""
        super().__init__()
        self.calls = calls
        self.scale = ms.Parameter(Tensor(0.9, ms.float32), name="scale")

    def construct(self, value: Tensor) -> Tensor:
        """Apply the excluded nonlinear operation.

        Args:
            value: Input activation.
        """
        self.calls["middle"] += 1
        return ops.tanh(value * self.scale)


class _Workload(nn.Cell):
    """Pointwise workload with an excluded region between two op sequences."""

    def __init__(self, calls: dict, depth: int, exclude_middle: bool) -> None:
        """Initialize the workload."""
        super().__init__()
        middle = _Middle(calls)
        self.middle = reentrant_checkpoint_exclude_wrapper(middle) if exclude_middle else middle
        self.depth = depth

    def construct(self, value: Tensor) -> Tensor:
        """Return a scalar loss with enough saved activations for memory checks.

        Args:
            value: Input activation.
        """
        for _ in range(self.depth):
            value = ops.sin(value * 1.01) + value * 0.01
        value = self.middle(value)
        for _ in range(self.depth):
            value = ops.cos(value * 0.99) + value * 0.01
        return (value * value).mean()


class _DispatchWorkload(nn.Cell):
    """Small-tensor workload dominated by eager host dispatch."""

    def __init__(self, depth: int = 40) -> None:
        """Store the number of pointwise operator pairs."""
        super().__init__()
        self.depth = depth

    def construct(self, value: Tensor) -> Tensor:
        """Dispatch a fixed sequence of lightweight operators.

        Args:
            value: Input activation.
        """
        for _ in range(self.depth):
            value = ops.sin(value) + value * 0.01
        return value


def _make_input(shape: tuple[int, ...]) -> Tensor:
    """Create one deterministic differentiable Tensor."""
    value = Tensor(np.linspace(-1.0, 1.0, num=math.prod(shape), dtype=np.float32).reshape(shape))
    value.requires_grad = True
    return value


def _clear_parameter_grads(module: nn.Cell) -> None:
    """Drop compatibility gradients before the next step."""
    for parameter in module.trainable_params():
        parameter.grad = None


def _run_step(module: nn.Cell, shape: tuple[int, ...]) -> None:
    """Run one compatibility-backward step and release graph roots."""
    _clear_parameter_grads(module)
    value = _make_input(shape)
    loss = module(value)
    loss.backward()
    del loss
    del value


def _forward_peak_bytes(module: nn.Cell, shape: tuple[int, ...]) -> int:
    """Measure device allocations added by one forward graph."""
    gc.collect()
    ms.runtime.empty_cache()
    value = _make_input(shape)
    ms.runtime.synchronize()
    baseline = ms.runtime.memory_allocated()
    ms.runtime.reset_peak_memory_stats()
    output = module(value)
    ms.runtime.synchronize()
    peak_delta = ms.runtime.max_memory_allocated() - baseline
    del output
    del value
    _pynative_executor.clear_res()
    _enable_compat_grad_recording()
    gc.collect()
    ms.runtime.empty_cache()
    return peak_delta


def _measure_dispatch_seconds(module: nn.Cell, value: Tensor, iterations: int) -> float:
    """Return median host submission time for forward-only iterations."""
    for _ in range(5):
        output = module(value)
        del output
    ms.runtime.synchronize()
    samples = []
    for _ in range(3):
        start = time.perf_counter()
        for _ in range(iterations):
            output = module(value)
            del output
        samples.append(time.perf_counter() - start)
        ms.runtime.synchronize()
        gc.collect()
    return statistics.median(samples)


def test_reentrant_exclude_result_correctness() -> None:
    """Compatibility backward should match eager values and all gradients."""
    reference_calls = {"middle": 0}
    actual_calls = {"middle": 0}
    reference = _Workload(reference_calls, depth=3, exclude_middle=False)
    actual = reentrant_checkpoint_wrapper(_Workload(actual_calls, depth=3, exclude_middle=True))
    reference_input = _make_input((16, 64))
    actual_input = _make_input((16, 64))

    reference_loss = reference(reference_input)
    reference_loss.backward()
    actual_loss = actual(actual_input)
    actual_loss.backward()

    np.testing.assert_allclose(actual_loss.asnumpy(), reference_loss.asnumpy(), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(actual_input.grad.asnumpy(), reference_input.grad.asnumpy(), atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(
        actual.middle.scale.grad.asnumpy(), reference.middle.scale.grad.asnumpy(), atol=1e-6, rtol=1e-5
    )
    assert reference_calls == {"middle": 1}
    assert actual_calls == {"middle": 1}


def test_reentrant_checkpoint_device_peak_memory() -> None:
    """Reentrant forward should retain materially less device activation memory."""
    _require_accelerator()
    reference = _Workload({"middle": 0}, depth=8, exclude_middle=False)
    actual = reentrant_checkpoint_wrapper(_Workload({"middle": 0}, depth=8, exclude_middle=True))
    shape = (1024, 1024)

    reference_peak = _forward_peak_bytes(reference, shape)
    actual_peak = _forward_peak_bytes(actual, shape)
    print(
        "MindSpore reentrant forward peak: "
        f"actual={actual_peak / _MIB:.2f} MiB, reference={reference_peak / _MIB:.2f} MiB, "
        f"ratio={actual_peak / reference_peak:.3f}"
    )

    assert actual_peak < reference_peak * 0.7, (
        f"reentrant peak memory did not decrease enough: actual={actual_peak / _MIB:.2f} MiB, "
        f"reference={reference_peak / _MIB:.2f} MiB"
    )


def test_reentrant_checkpoint_host_dispatch_performance() -> None:
    """Hook-free forward dispatch should be faster than non-reentrant checkpoint."""
    value = _make_input((64,))
    non_reentrant = ckpt_wrapper(_DispatchWorkload())
    reentrant = reentrant_checkpoint_wrapper(_DispatchWorkload())

    non_reentrant_seconds = _measure_dispatch_seconds(non_reentrant, value, iterations=20)
    reentrant_seconds = _measure_dispatch_seconds(reentrant, value, iterations=20)
    print(
        "MindSpore checkpoint host dispatch: "
        f"reentrant={reentrant_seconds:.6f}s, non_reentrant={non_reentrant_seconds:.6f}s, "
        f"speedup={non_reentrant_seconds / reentrant_seconds:.3f}x"
    )

    assert reentrant_seconds < non_reentrant_seconds, (
        f"hook-free host dispatch was not faster: reentrant={reentrant_seconds:.6f}s, "
        f"non_reentrant={non_reentrant_seconds:.6f}s"
    )


def test_reentrant_checkpoint_device_memory_no_leak() -> None:
    """Repeated compatibility backward should not retain device allocations."""
    _require_accelerator()
    module = reentrant_checkpoint_wrapper(_Workload({"middle": 0}, depth=3, exclude_middle=True))
    samples = []
    for step in range(20):
        _run_step(module, (64, 64))
        ms.runtime.synchronize()
        if step >= 5:
            gc.collect()
            samples.append(ms.runtime.memory_allocated())

    growth = max(samples) - min(samples)
    print(f"MindSpore steady-state device allocation range: {growth / _MIB:.2f} MiB")
    assert max(samples) - min(samples) <= 4 * _MIB, (
        f"device allocations grew across steady-state steps: {[value / _MIB for value in samples]}"
    )


def test_reentrant_checkpoint_host_memory_no_leak() -> None:
    """Repeated compatibility-backward graphs should not cause sustained RSS growth."""
    module = reentrant_checkpoint_wrapper(_Workload({"middle": 0}, depth=2, exclude_middle=True))
    process = psutil.Process()
    for _ in range(10):
        _run_step(module, (32, 32))
    ms.runtime.synchronize()
    gc.collect()
    baseline_rss = process.memory_info().rss

    for _ in range(40):
        _run_step(module, (32, 32))
    ms.runtime.synchronize()
    gc.collect()
    growth = process.memory_info().rss - baseline_rss

    print(f"MindSpore steady-state host RSS growth: {growth / _MIB:.2f} MiB")
    assert growth <= 32 * _MIB, f"host RSS grew by {growth / _MIB:.2f} MiB"

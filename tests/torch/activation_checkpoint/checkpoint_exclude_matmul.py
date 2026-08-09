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
"""Validate PyTorch checkpoint exclusion memory, precision, and performance."""
import gc
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict

import torch

from hyper_parallel.core.activation_checkpoint import checkpoint_exclude_wrapper, checkpoint_wrapper


_TOKEN_NUM = 16384
_HIDDEN_SIZE = 2048
_LAYER_NUM = 20
_RESULT_MARKER = "__TORCH_RMSNORM_MATMUL_RESULT__"
_EXCLUDE_MODULE = importlib.import_module(
    "hyper_parallel.platform.torch.activation_checkpoint.checkpoint_exclude_wrapper"
)


class _CountedMatmul(torch.nn.Module):
    """MatMul that records whether checkpoint replay executes it."""

    def __init__(self, calls: Dict[str, int]) -> None:
        """Initialize a deterministic square projection."""
        super().__init__()
        self.calls = calls
        self.weight = torch.nn.Parameter(torch.empty(_HIDDEN_SIZE, _HIDDEN_SIZE, dtype=torch.bfloat16))
        with torch.no_grad():
            torch.manual_seed(2026)
            self.weight.normal_(0.0, 0.02)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply the projection."""
        self.calls["matmul"] += 1
        return tensor @ self.weight


class _RmsNormMatmulBlock(torch.nn.Module):
    """Apply RMSNorm, an excluded MatMul, and SiLU."""

    def __init__(self, calls: Dict[str, int]) -> None:
        """Initialize one checkpointed layer."""
        super().__init__()
        self.norm_weight = torch.nn.Parameter(torch.ones(_HIDDEN_SIZE, dtype=torch.bfloat16))
        self.matmul = checkpoint_exclude_wrapper(_CountedMatmul(calls))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Run the layer."""
        normalized = torch.nn.functional.rms_norm(
            tensor,
            (_HIDDEN_SIZE,),
            self.norm_weight,
            eps=1e-6,
        )
        return torch.nn.functional.silu(self.matmul(normalized))


class _RmsNormMatmulNet(torch.nn.Module):
    """Stack individually checkpointed layers."""

    def __init__(self, calls: Dict[str, int]) -> None:
        """Create one checkpoint for each layer."""
        super().__init__()
        self.layers = torch.nn.ModuleList([
            checkpoint_wrapper(_RmsNormMatmulBlock(calls))
            for _ in range(_LAYER_NUM)
        ])

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Run all layers and return a scalar loss."""
        for layer in self.layers:
            tensor = layer(tensor)
        return tensor.float().square().sum()


class _SaveInputSquare(torch.nn.Module):
    """SAVE operator that needs its input but not its output for backward."""

    calls = 0

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return a square whose backward saves the input tensor."""
        _SaveInputSquare.calls += 1
        return tensor.square()


class _SaveOutputExp(torch.nn.Module):
    """SAVE operator whose backward needs only its own output."""

    calls = 0

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return an exponential whose backward saves only its output."""
        _SaveOutputExp.calls += 1
        return tensor.exp()


class _FirstSaveActivation(torch.nn.Module):
    """First SAVE operator that saves its recomputed input and output activation."""

    calls = 0

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Save the square input and the ReLU output for backward."""
        _FirstSaveActivation.calls += 1
        return torch.nn.functional.relu(tensor.square())


class _SecondSaveAdd(torch.nn.Module):
    """Second SAVE operator whose output is not a backward activation."""

    calls = 0

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Add a constant without saving the input or output for backward."""
        _SecondSaveAdd.calls += 1
        return tensor + 0.125


class _ThirdSaveScale(torch.nn.Module):
    """Third SAVE operator whose output is not a backward activation."""

    calls = 0

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Multiply by a fixed scalar without saving input or output."""
        _ThirdSaveScale.calls += 1
        return tensor * 1.5


class _FourthSaveAdd(torch.nn.Module):
    """Fourth SAVE operator that produces the retained final output."""

    calls = 0

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Add a fixed constant without saving input or output."""
        _FourthSaveAdd.calls += 1
        return tensor + 0.25


class _SaveChainNet(torch.nn.Module):
    """Chain four SAVE operators with two non-activation intermediate outputs."""

    def __init__(self, save_intermediate_outputs: bool) -> None:
        """Configure whether the first three SAVE outputs remain cached."""
        super().__init__()
        self.first = checkpoint_exclude_wrapper(
            _FirstSaveActivation(),
            save_output=save_intermediate_outputs,
        )
        self.second = checkpoint_exclude_wrapper(
            _SecondSaveAdd(),
            save_output=save_intermediate_outputs,
        )
        self.third = checkpoint_exclude_wrapper(
            _ThirdSaveScale(),
            save_output=save_intermediate_outputs,
        )
        self.fourth = checkpoint_exclude_wrapper(_FourthSaveAdd())

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return a scalar loss through four adjacent SAVE regions."""
        return self.fourth(self.third(self.second(self.first(tensor * 2)))).float().sum()


class _LongStabilityNet(torch.nn.Module):
    """Exercise both checkpoint boundaries with a small, repeatable workload."""

    def __init__(self) -> None:
        """Create a RECOMPUTE→SAVE(False)→SAVE(True) chain."""
        super().__init__()
        self.first = checkpoint_exclude_wrapper(_SaveInputSquare(), save_output=False)
        self.second = checkpoint_exclude_wrapper(_SaveOutputExp())

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Run the checkpoint boundary chain and return a scalar."""
        return self.second(self.first(tensor * 2)).float().sum()


def _synchronize() -> None:
    """Synchronize the current NPU."""
    torch.npu.synchronize()


def _clear_grads(module: torch.nn.Module) -> None:
    """Release parameter gradients without touching parameters."""
    for parameter in module.parameters():
        parameter.grad = None


def _run_exclude() -> Dict[str, Any]:
    """Run one optimized or legacy exclusion step."""
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(2027)
    torch.npu.manual_seed(2027)
    calls = {"matmul": 0}
    net = _RmsNormMatmulNet(calls).npu()
    input_data = torch.randn(_TOKEN_NUM, _HIDDEN_SIZE, device="npu", dtype=torch.bfloat16).mul_(0.5)

    warmup_input = input_data.detach().clone().requires_grad_()
    warmup_loss = net(warmup_input)
    warmup_loss.backward()
    _synchronize()
    _clear_grads(net)
    del warmup_input, warmup_loss
    calls["matmul"] = 0
    torch.npu.empty_cache()

    baseline_bytes = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    tensor = input_data.detach().clone().requires_grad_()
    loss = net(tensor)
    _synchronize()
    forward_bytes = torch.npu.memory_allocated() - baseline_bytes
    loss.backward()
    _synchronize()

    return {
        "loss": float(loss.detach().cpu()),
        "matmul_calls": calls["matmul"],
        "forward_bytes": int(forward_bytes),
        "peak_bytes": int(torch.npu.max_memory_allocated() - baseline_bytes),
    }


def _run_save_chain(save_intermediate_outputs: bool) -> Dict[str, Any]:
    """Measure four SAVE regions with or without their intermediate outputs."""
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(2028)
    torch.npu.manual_seed(2028)
    net = checkpoint_wrapper(_SaveChainNet(save_intermediate_outputs)).npu()
    input_data = torch.randn(_TOKEN_NUM, _HIDDEN_SIZE, device="npu", dtype=torch.bfloat16).mul_(0.01)

    warmup_input = input_data.detach().clone().requires_grad_()
    warmup_loss = net(warmup_input)
    warmup_loss.backward()
    _synchronize()
    del warmup_input, warmup_loss
    _FirstSaveActivation.calls = 0
    _SecondSaveAdd.calls = 0
    _ThirdSaveScale.calls = 0
    _FourthSaveAdd.calls = 0
    torch.npu.empty_cache()

    baseline_bytes = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    tensor = input_data.detach().clone().requires_grad_()
    loss = net(tensor)
    _synchronize()
    forward_bytes = torch.npu.memory_allocated() - baseline_bytes
    loss.backward()
    _synchronize()

    return {
        "loss": float(loss.detach().cpu()),
        "input_grad": float(tensor.grad[0, 0].detach().cpu()),
        "first_calls": _FirstSaveActivation.calls,
        "second_calls": _SecondSaveAdd.calls,
        "third_calls": _ThirdSaveScale.calls,
        "fourth_calls": _FourthSaveAdd.calls,
        "forward_bytes": int(forward_bytes),
    }


def _disable_input_rematerialization(invocation_id: object, args: Any, kwargs: Any) -> tuple:
    """Reproduce legacy exclusion, which retains saved input storage."""
    del invocation_id, args, kwargs
    return [], []


def _run_host_performance_comparison() -> Dict[str, Any]:
    """Compare optimized and legacy host-visible step time on one warmed network."""
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(2030)
    torch.npu.manual_seed(2030)
    net = _RmsNormMatmulNet({"matmul": 0}).npu()
    input_data = torch.randn(_TOKEN_NUM, _HIDDEN_SIZE, device="npu", dtype=torch.bfloat16).mul_(0.5)
    original_mark = _EXCLUDE_MODULE._mark_recompute_inputs  # pylint: disable=protected-access
    timings = {"current": [], "legacy": []}

    def set_mode(mode: str) -> None:
        """Switch only the input rematerialization control-plane path."""
        mark_inputs = original_mark if mode == "current" else _disable_input_rematerialization
        _EXCLUDE_MODULE._mark_recompute_inputs = mark_inputs  # pylint: disable=protected-access

    try:
        for mode in ("legacy", "current"):
            set_mode(mode)
            tensor = input_data.detach().clone().requires_grad_()
            loss = net(tensor)
            loss.backward()
            _synchronize()
            _clear_grads(net)
        order = ("legacy", "current", "current", "legacy", "legacy", "current")
        for mode in order:
            set_mode(mode)
            tensor = input_data.detach().clone().requires_grad_()
            _synchronize()
            start = time.perf_counter()
            loss = net(tensor)
            loss.backward()
            _synchronize()
            timings[mode].append((time.perf_counter() - start) * 1000)
            _clear_grads(net)
            del tensor, loss
    finally:
        set_mode("current")

    result = {}
    for mode in ("current", "legacy"):
        values = sorted(timings[mode])
        result[mode] = {
            "total_ms": values[len(values) // 2],
            "samples": len(values),
        }
    result["ratio"] = result["current"]["total_ms"] / result["legacy"]["total_ms"]
    return result


def _current_rss_kib() -> int:
    """Read the process's current resident set size from procfs."""
    resident_pages = int(Path(f"/proc/{os.getpid()}/statm").read_text(encoding="utf-8").split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE") // 1024


def _run_long_stability(steps: int = 100000) -> Dict[str, Any]:
    """Track host and device memory while repeatedly exercising both boundaries."""
    torch.manual_seed(2031)
    torch.npu.manual_seed(2031)
    net = checkpoint_wrapper(_LongStabilityNet()).npu()
    input_data = torch.randn(32, 128, device="npu", dtype=torch.float32).mul_(0.01)
    sample_steps = {0, 1000, 10000, 25000, 50000, steps}
    samples = []

    def run_step() -> None:
        """Run one complete forward/backward and release graph roots."""
        tensor = input_data.detach().clone().requires_grad_()
        loss = net(tensor)
        loss.backward()

    for _ in range(100):
        run_step()
    _synchronize()
    gc.collect()
    torch.npu.empty_cache()

    for step in range(steps + 1):
        if step in sample_steps:
            _synchronize()
            gc.collect()
            trigger = _EXCLUDE_MODULE._get_recompute_trigger()  # pylint: disable=protected-access
            samples.append({
                "step": step,
                "rss_kib": _current_rss_kib(),
                "allocated_bytes": int(torch.npu.memory_allocated()),
                "reserved_bytes": int(torch.npu.memory_reserved()),
                "trigger_refcount": sys.getrefcount(trigger),
                "trigger_grad_none": trigger.grad is None,
            })
        if step != steps:
            run_step()
        if step % 100 == 0:
            _synchronize()

    return {
        "steps": steps,
        "samples": samples,
        "first_calls": _SaveInputSquare.calls,
        "second_calls": _SaveOutputExp.calls,
    }


def _run_mode(mode: str) -> Dict[str, Any]:
    """Run optimized, legacy, or performance checkpoint exclusion."""
    if mode == "exclude":
        return _run_exclude()
    if mode == "save_chain":
        return _run_save_chain(save_intermediate_outputs=False)
    if mode == "legacy_save_chain":
        return _run_save_chain(save_intermediate_outputs=True)
    if mode == "host_performance_comparison":
        return _run_host_performance_comparison()
    if mode == "long_stability":
        return _run_long_stability()
    if mode != "legacy_exclude":
        raise ValueError(f"Unsupported checkpoint exclusion mode: {mode}")

    original_mark = _EXCLUDE_MODULE._mark_recompute_inputs  # pylint: disable=protected-access
    _EXCLUDE_MODULE._mark_recompute_inputs = _disable_input_rematerialization  # pylint: disable=protected-access
    try:
        return _run_exclude()
    finally:
        _EXCLUDE_MODULE._mark_recompute_inputs = original_mark  # pylint: disable=protected-access


def _run_mode_in_subprocess(mode: str) -> Dict[str, Any]:
    """Run one mode in an isolated NPU allocator process."""
    project_root = Path(__file__).resolve().parents[3]
    command = [
        sys.executable,
        "-c",
        (
            "import json; "
            "from tests.torch.activation_checkpoint.checkpoint_exclude_matmul import _run_mode; "
            f"print({_RESULT_MARKER!r} + json.dumps(_run_mode({mode!r})))"
        ),
    ]
    completed = subprocess.run(
        command,
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=1200,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"RMSNorm/MatMul subprocess exited with code {completed.returncode}.\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    for line in reversed(completed.stdout.splitlines()):
        if _RESULT_MARKER in line:
            return json.loads(line.split(_RESULT_MARKER, maxsplit=1)[1])
    raise RuntimeError(
        "RMSNorm/MatMul subprocess did not produce a result marker.\n"
        f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
    )


def test_rmsnorm_matmul_checkpoint_exclude_memory() -> None:
    """Optimized exclusion should retain less memory than legacy exclusion."""
    excluded = _run_mode_in_subprocess("exclude")
    legacy = _run_mode_in_subprocess("legacy_exclude")

    assert excluded["loss"] == legacy["loss"]
    assert excluded["matmul_calls"] == _LAYER_NUM
    assert legacy["matmul_calls"] == _LAYER_NUM

    expected_gap = _LAYER_NUM * _TOKEN_NUM * _HIDDEN_SIZE * 2
    tolerance = expected_gap // 4
    forward_gap = legacy["forward_bytes"] - excluded["forward_bytes"]
    peak_gap = legacy["peak_bytes"] - excluded["peak_bytes"]
    assert abs(forward_gap - expected_gap) <= tolerance
    assert abs(peak_gap - expected_gap) <= tolerance

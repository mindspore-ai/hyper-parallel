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
from typing import Any, Dict, Optional
from unittest.mock import patch

import torch

from hyper_parallel.core.activation_memory import (
    PinnedMemoryPool,
    SwapManager,
    checkpoint_exclude_wrapper,
    checkpoint_wrapper,
)


_TOKEN_NUM = 16384
_HIDDEN_SIZE = 2048
_LAYER_NUM = 20
_SWAP_TOKEN_NUM = 256
_SWAP_HIDDEN_SIZE = 256
_SWAP_LAYER_NUM = 4
_SWAP_POOL_BYTES = 64 * 1024**2
_RESULT_MARKER = "__TORCH_RMSNORM_MATMUL_RESULT__"
_EXCLUDE_MODULE = importlib.import_module(
    "hyper_parallel.core.activation_memory.checkpoint_exclude"
)


class _CountedMatmul(torch.nn.Module):
    """Project a half-width view and record whether replay executes it."""

    def __init__(self, calls: Dict[str, int]) -> None:
        """Initialize a deterministic square projection."""
        super().__init__()
        self.calls = calls
        self.weight = torch.nn.Parameter(torch.empty(_HIDDEN_SIZE // 2, _HIDDEN_SIZE, dtype=torch.bfloat16))
        with torch.no_grad():
            torch.manual_seed(2026)
            self.weight.normal_(0.0, 0.02)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply the projection."""
        self.calls["matmul"] += 1
        return tensor @ self.weight


class _RmsNormMatmulBlock(torch.nn.Module):
    """Apply RMSNorm, a sliced view, an excluded MatMul, and SiLU."""

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
        viewed = normalized.view(_TOKEN_NUM, _HIDDEN_SIZE)[:, : _HIDDEN_SIZE // 2]
        return torch.nn.functional.silu(self.matmul(viewed))


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
    """Chain four SAVE operators whose outputs are retained for replay."""

    def __init__(self) -> None:
        """Wrap every operator with the default output-cache behavior."""
        super().__init__()
        self.first = checkpoint_exclude_wrapper(_FirstSaveActivation())
        self.second = checkpoint_exclude_wrapper(_SecondSaveAdd())
        self.third = checkpoint_exclude_wrapper(_ThirdSaveScale())
        self.fourth = checkpoint_exclude_wrapper(_FourthSaveAdd())

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return a scalar loss through four adjacent SAVE regions."""
        return self.fourth(self.third(self.second(self.first(tensor * 2)))).float().sum()


class _SwapSaveBlock(torch.nn.Module):
    """Produce replayable input saves plus independent internal activations."""

    def __init__(self, calls: list[int], index: int, randomized: bool = False) -> None:
        """Store the shared execution counters and this block's index."""
        super().__init__()
        self.calls = calls
        self.index = index
        self.randomized = randomized

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Save the input for replay and two internal tensors for backward."""
        self.calls[self.index] += 1
        replay_input = tensor.square()
        if self.randomized:
            replay_input = torch.nn.functional.dropout(replay_input, p=0.25, training=True)
        hidden = replay_input * 0.5 + 0.5
        return hidden.square() + replay_input.square()


class _SwapSaveStage(torch.nn.Module):
    """Own one checkpoint-excluded region inside an independently replayed stage."""

    def __init__(
        self,
        calls: list[int],
        index: int,
        *,
        swap: bool,
        group_swap: bool,
        cpu_pool: Optional[PinnedMemoryPool],
        reentrant: bool = False,
    ) -> None:
        """Wrap one activation-producing block with checkpoint exclusion."""
        super().__init__()
        self.reentrant = reentrant
        self.excluded = checkpoint_exclude_wrapper(
            _SwapSaveBlock(calls, index, randomized=reentrant),
            swap=swap,
            group_swap=group_swap,
            cpu_pool=cpu_pool,
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Run the excluded block inside this stage's checkpoint invocation."""
        result = self.excluded(tensor)
        return result * torch.rand_like(result) if self.reentrant else result


class _SwapSaveChainNet(torch.nn.Module):
    """Chain independently checkpointed stages under a real swap prefetch schedule."""

    def __init__(
        self,
        calls: list[int],
        *,
        swap: bool,
        group_swap: bool = False,
        cpu_pool: Optional[PinnedMemoryPool] = None,
        reentrant: bool = False,
    ) -> None:
        """Create four excluded blocks and wire their swap groups when requested."""
        super().__init__()
        self.layers = torch.nn.ModuleList([
            checkpoint_wrapper(
                _SwapSaveStage(
                    calls,
                    index,
                    swap=swap,
                    group_swap=group_swap,
                    cpu_pool=cpu_pool,
                    reentrant=reentrant,
                ),
                use_reentrant=reentrant,
            )
            for index in range(_SWAP_LAYER_NUM)
        ])
        if swap:
            for current_layer, next_layer in zip(self.layers[:-1], self.layers[1:]):
                SwapManager().set_forward_prefetch_layer(current_layer, next_layer)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Run every excluded block and return a scalar loss."""
        for layer in self.layers:
            tensor = layer(tensor)
        return tensor.float().sum()


class _LongStabilityNet(torch.nn.Module):
    """Exercise both checkpoint boundaries with a small, repeatable workload."""

    def __init__(self) -> None:
        """Create a RECOMPUTE followed by two output-caching SAVE regions."""
        super().__init__()
        self.first = checkpoint_exclude_wrapper(_SaveInputSquare())
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


def _save_chain_eager_reference(tensor: torch.Tensor) -> torch.Tensor:
    """Compute the SAVE chain without checkpoint exclusion for parity checks."""
    result = torch.nn.functional.relu((tensor * 2).square())
    result = result + 0.125
    result = result * 1.5
    result = result + 0.25
    return result.float().sum()


def _run_save_chain() -> Dict[str, Any]:
    """Validate four adjacent SAVE regions with retained replay outputs."""
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(2028)
    torch.npu.manual_seed(2028)
    net = checkpoint_wrapper(_SaveChainNet()).npu()
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

    tensor = input_data.detach().clone().requires_grad_()
    loss = net(tensor)
    _synchronize()
    loss.backward()
    _synchronize()

    reference_tensor = input_data.detach().clone().requires_grad_()
    reference_loss = _save_chain_eager_reference(reference_tensor)
    reference_loss.backward()
    _synchronize()

    return {
        "loss": float(loss.detach().cpu()),
        "input_grad": float(tensor.grad[0, 0].detach().cpu()),
        "reference_loss": float(reference_loss.detach().cpu()),
        "reference_input_grad": float(reference_tensor.grad[0, 0].detach().cpu()),
        "first_calls": _FirstSaveActivation.calls,
        "second_calls": _SecondSaveAdd.calls,
        "third_calls": _ThirdSaveScale.calls,
        "fourth_calls": _FourthSaveAdd.calls,
    }


def _swap_group_metrics(layers: torch.nn.ModuleList) -> Dict[str, int]:
    """Count registered, offloaded, and group-managed tensors before backward."""
    manager = SwapManager()
    registered_storage_count = 0
    offloaded_tensor_count = 0
    group_managed_tensor_count = 0
    for layer in layers[:-1]:
        group_name = layer._swap_group_name  # pylint: disable=protected-access
        group = manager._groups[group_name]  # pylint: disable=protected-access
        registered_storage_count += len(group._storages)  # pylint: disable=protected-access
        for storage in group._storages:  # pylint: disable=protected-access
            for swap_tensor in storage.iter_swap_tensors():
                if swap_tensor.state == "host":
                    offloaded_tensor_count += 1
                if swap_tensor._group_managed:  # pylint: disable=protected-access
                    group_managed_tensor_count += 1
    return {
        "registered_storage_count": registered_storage_count,
        "offloaded_tensor_count": offloaded_tensor_count,
        "group_managed_tensor_count": group_managed_tensor_count,
    }


def _remaining_swap_storage_count(layers: torch.nn.ModuleList) -> int:
    """Return swap storages still retained after backward completes."""
    manager = SwapManager()
    count = 0
    for layer in layers:
        group_name = getattr(layer, "_swap_group_name", None)
        if group_name is None:
            continue
        group = manager._groups.get(group_name)  # pylint: disable=protected-access
        if group is not None:
            count += len(group._storages)  # pylint: disable=protected-access
    return count


def _run_swap_save_chain(*, swap: bool, group_swap: bool, use_cpu_pool: bool,
                         reentrant: bool = False) -> Dict[str, Any]:
    """Run one checkpoint-exclusion swap mode and report correctness and lifecycle metrics."""
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(2029)
    torch.npu.manual_seed(2029)
    cpu_pool = PinnedMemoryPool(max_host_bytes=_SWAP_POOL_BYTES) if use_cpu_pool else None
    calls = [0] * _SWAP_LAYER_NUM
    chain = _SwapSaveChainNet(
        calls,
        swap=swap,
        group_swap=group_swap,
        cpu_pool=cpu_pool,
        reentrant=reentrant,
    )
    net = chain.npu()
    tensor = torch.randn(
        _SWAP_TOKEN_NUM,
        _SWAP_HIDDEN_SIZE,
        device="npu",
        dtype=torch.float32,
    ).mul_(0.1).requires_grad_()

    saved = {"recipes": 0, "internal": 0, "swapped_internal": 0}
    original_pack = _EXCLUDE_MODULE._pack_saved_tensor  # pylint: disable=protected-access

    def record_pack(value: torch.Tensor) -> Any:
        """Inspect the real saved-tensor hook without retaining its activation."""
        packed = original_pack(value)
        if isinstance(packed, _EXCLUDE_MODULE._RecomputedInputHandle):  # pylint: disable=protected-access
            saved["recipes"] += 1
        elif isinstance(packed, _EXCLUDE_MODULE._SavedTensorData):  # pylint: disable=protected-access
            saved["internal"] += 1
            if packed.swap_hooks is not None:
                saved["swapped_internal"] += 1
        return packed

    with patch.object(_EXCLUDE_MODULE, "_pack_saved_tensor", side_effect=record_pack):
        loss = net(tensor)
    _synchronize()
    forward_bytes = int(torch.npu.memory_allocated())
    swap_metrics = _swap_group_metrics(chain.layers) if swap else {
        "registered_storage_count": 0,
        "offloaded_tensor_count": 0,
        "group_managed_tensor_count": 0,
    }
    loss.backward()
    _synchronize()

    return {
        "loss": float(loss.detach().cpu()),
        "input_grad": float(tensor.grad[0, 0].detach().cpu()),
        "calls": calls,
        "pool_allocated_bytes": 0 if cpu_pool is None else cpu_pool.total_allocated,
        "pool_capacity_bytes": 0 if cpu_pool is None else cpu_pool.max_host_bytes,
        "remaining_storage_count": _remaining_swap_storage_count(chain.layers),
        "forward_bytes": forward_bytes,
        "rng_state": torch.npu.get_rng_state().cpu().tolist(),
        **saved,
        **swap_metrics,
    }


def _disable_input_rematerialization(args: Any, kwargs: Any) -> tuple:
    """Reproduce legacy exclusion, which retains saved input storage."""
    del args, kwargs
    return ()


def _run_host_performance_comparison() -> Dict[str, Any]:
    """Compare optimized and legacy host-visible step time on one warmed network."""
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(2030)
    torch.npu.manual_seed(2030)
    net = _RmsNormMatmulNet({"matmul": 0}).npu()
    input_data = torch.randn(_TOKEN_NUM, _HIDDEN_SIZE, device="npu", dtype=torch.bfloat16).mul_(0.5)
    original_mark = _EXCLUDE_MODULE._capture_recompute_inputs  # pylint: disable=protected-access
    timings = {"current": [], "legacy": []}

    def set_mode(mode: str) -> None:
        """Switch only the input rematerialization control-plane path."""
        mark_inputs = original_mark if mode == "current" else _disable_input_rematerialization
        _EXCLUDE_MODULE._capture_recompute_inputs = mark_inputs  # pylint: disable=protected-access

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
        return _run_save_chain()
    if mode == "swap_save_chain_baseline":
        return _run_swap_save_chain(swap=False, group_swap=False, use_cpu_pool=False)
    if mode == "swap_save_chain":
        return _run_swap_save_chain(swap=True, group_swap=False, use_cpu_pool=False)
    if mode == "group_swap_pool_save_chain":
        return _run_swap_save_chain(swap=True, group_swap=True, use_cpu_pool=True)
    if mode == "reentrant_save_chain_baseline":
        return _run_swap_save_chain(swap=False, group_swap=False, use_cpu_pool=False, reentrant=True)
    if mode == "reentrant_swap_save_chain":
        return _run_swap_save_chain(swap=True, group_swap=False, use_cpu_pool=False, reentrant=True)
    if mode == "reentrant_group_swap_pool_save_chain":
        return _run_swap_save_chain(swap=True, group_swap=True, use_cpu_pool=True, reentrant=True)
    if mode == "host_performance_comparison":
        return _run_host_performance_comparison()
    if mode == "long_stability":
        return _run_long_stability()
    if mode != "legacy_exclude":
        raise ValueError(f"Unsupported checkpoint exclusion mode: {mode}")

    original_mark = _EXCLUDE_MODULE._capture_recompute_inputs  # pylint: disable=protected-access
    _EXCLUDE_MODULE._capture_recompute_inputs = _disable_input_rematerialization  # pylint: disable=protected-access
    try:
        return _run_exclude()
    finally:
        _EXCLUDE_MODULE._capture_recompute_inputs = original_mark  # pylint: disable=protected-access


def _run_mode_in_subprocess(mode: str) -> Dict[str, Any]:
    """Run one mode in an isolated NPU allocator process."""
    project_root = Path(__file__).resolve().parents[3]
    command = [
        sys.executable,
        "-c",
        (
            "import json; "
            "from tests.torch.activation_memory.checkpoint_exclude import _run_mode; "
            f"print({_RESULT_MARKER!r} + json.dumps(_run_mode({mode!r})))"
        ),
    ]
    completed = subprocess.run(
        command,
        cwd=project_root,
        env=os.environ.copy(),
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


def test_adjacent_save_regions_execute_once_and_replay_cached_outputs() -> None:
    """Each excluded region should execute once and replay its cached output."""
    result = _run_mode_in_subprocess("save_chain")

    for region in ("first", "second", "third", "fourth"):
        actual_calls = result[f"{region}_calls"]
        assert actual_calls == 1, f"{region} region should execute once, got {actual_calls} calls"

    loss_tolerance = max(1e-4, abs(result["reference_loss"]) * 1e-6)
    loss_error = abs(result["loss"] - result["reference_loss"])
    assert loss_error <= loss_tolerance, (
        f"Cached SAVE outputs changed the loss: reference={result['reference_loss']}, "
        f"actual={result['loss']}, error={loss_error}, tolerance={loss_tolerance}"
    )

    grad_tolerance = max(1e-5, abs(result["reference_input_grad"]) * 1e-5)
    grad_error = abs(result["input_grad"] - result["reference_input_grad"])
    assert grad_error <= grad_tolerance, (
        f"Cached SAVE outputs changed the input gradient: "
        f"reference={result['reference_input_grad']}, actual={result['input_grad']}, "
        f"error={grad_error}, tolerance={grad_tolerance}"
    )


def _assert_swap_save_chain_matches_baseline(baseline: dict, actual: dict) -> None:
    """Validate numerical parity, exclusion behavior, and swap storage cleanup."""
    loss_tolerance = max(1e-4, abs(baseline["loss"]) * 1e-6)
    grad_tolerance = max(1e-5, abs(baseline["input_grad"]) * 1e-5)
    assert abs(actual["loss"] - baseline["loss"]) <= loss_tolerance
    assert abs(actual["input_grad"] - baseline["input_grad"]) <= grad_tolerance
    assert baseline["calls"] == [1, 1, 1, 1]
    assert actual["calls"] == [1, 1, 1, 1]
    assert actual["registered_storage_count"] > 0
    assert actual["offloaded_tensor_count"] > 0
    assert actual["remaining_storage_count"] == 0


def test_checkpoint_exclude_save_chain_swap() -> None:
    """checkpoint_exclude_wrapper(swap=True) should offload and restore SAVE activations."""
    baseline = _run_mode_in_subprocess("swap_save_chain_baseline")
    swapped = _run_mode_in_subprocess("swap_save_chain")

    _assert_swap_save_chain_matches_baseline(baseline, swapped)
    assert swapped["group_managed_tensor_count"] == 0
    assert swapped["pool_allocated_bytes"] == 0


def test_checkpoint_exclude_save_chain_group_swap_cpu_pool() -> None:
    """Group swap should fuse SAVE tensors and allocate their host buffers from the explicit pool."""
    baseline = _run_mode_in_subprocess("swap_save_chain_baseline")
    pooled = _run_mode_in_subprocess("group_swap_pool_save_chain")

    _assert_swap_save_chain_matches_baseline(baseline, pooled)
    assert pooled["group_managed_tensor_count"] > 0
    assert 0 < pooled["pool_allocated_bytes"] <= pooled["pool_capacity_bytes"]


def test_reentrant_checkpoint_exclude_swap() -> None:
    """Reentrant replay rebuilds input saves, swaps internal saves and clears prefetch groups."""
    baseline = _run_mode_in_subprocess("reentrant_save_chain_baseline")
    for mode in ("reentrant_swap_save_chain", "reentrant_group_swap_pool_save_chain"):
        swapped = _run_mode_in_subprocess(mode)
        _assert_swap_save_chain_matches_baseline(baseline, swapped)
        assert swapped["rng_state"] == baseline["rng_state"]
        assert baseline["recipes"] > 0 and swapped["recipes"] > 0
        assert baseline["internal"] > 0 and swapped["internal"] > 0
        assert baseline["swapped_internal"] == 0
        assert swapped["swapped_internal"] > 0
        if mode == "reentrant_group_swap_pool_save_chain":
            assert swapped["group_managed_tensor_count"] > 0
            assert 0 < swapped["pool_allocated_bytes"] <= swapped["pool_capacity_bytes"]

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
"""Validate checkpoint exclusion memory with stacked RMSNorm, MatMul, and SiLU."""
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict

import mindspore as ms
from mindspore import nn, Tensor
import numpy as np

from hyper_parallel.core.activation_checkpoint import checkpoint_exclude_wrapper, checkpoint_wrapper
from hyper_parallel.platform.mindspore.autograd_compat import enable_mindspore_backward_compat


enable_mindspore_backward_compat()


_TOKEN_NUM = 16384
_HIDDEN_SIZE = 2048
_LAYER_NUM = 20
_RESULT_MARKER = "__RMSNORM_MATMUL_RESULT__"
_EXCLUDE_MODULE = importlib.import_module(
    "hyper_parallel.platform.mindspore.activation_checkpoint.checkpoint_exclude_wrapper"
)


class _CountedMatmul(nn.Cell):
    """MatMul that records whether checkpoint replay executes it."""

    def __init__(self, calls: Dict[str, int]) -> None:
        """Initialize a deterministic square projection."""
        super().__init__()
        self.calls = calls
        rng = np.random.default_rng(2026)
        weight = rng.normal(0.0, 0.02, (_HIDDEN_SIZE, _HIDDEN_SIZE)).astype(np.float32)
        self.weight = ms.Parameter(Tensor(weight, ms.bfloat16), name="matmul_weight")

    def construct(self, tensor: Tensor) -> Tensor:
        """Apply the projection."""
        self.calls["matmul"] += 1
        return ms.ops.matmul(tensor, self.weight)


class _RmsNormMatmulBlock(nn.Cell):
    """Apply RMSNorm, an excluded MatMul, and SiLU."""

    def __init__(self, calls: Dict[str, int]) -> None:
        """Initialize one checkpointed layer."""
        super().__init__()
        self.rms_norm = ms.ops.RmsNorm(epsilon=1e-6)
        self.norm_weight = ms.Parameter(ms.ops.ones((_HIDDEN_SIZE,), ms.bfloat16), name="norm_weight")
        self.matmul = checkpoint_exclude_wrapper(_CountedMatmul(calls))

    def construct(self, tensor: Tensor) -> Tensor:
        """Run the layer."""
        normalized = self.rms_norm(tensor, self.norm_weight)[0]
        return ms.ops.silu(self.matmul(normalized))


class _RmsNormMatmulNet(nn.Cell):
    """Stack individually checkpointed layers."""

    def __init__(self, calls: Dict[str, int]) -> None:
        """Create one checkpoint for each layer."""
        super().__init__()
        self.layers = nn.CellList([
            checkpoint_wrapper(_RmsNormMatmulBlock(calls))
            for _ in range(_LAYER_NUM)
        ])

    def construct(self, tensor: Tensor) -> Tensor:
        """Run all layers and return a scalar loss."""
        for layer in self.layers:
            tensor = layer(tensor)
        return ms.ops.sum(ms.ops.square(tensor))


class _SaveInputSquare(nn.Cell):
    """SAVE operator that needs its input but not its output for backward."""

    calls = 0

    def construct(self, tensor: Tensor) -> Tensor:
        """Return a square whose backward saves the input tensor."""
        _SaveInputSquare.calls += 1
        return tensor * tensor


class _SaveOutputExp(nn.Cell):
    """SAVE operator whose backward needs only its own output."""

    calls = 0

    def construct(self, tensor: Tensor) -> Tensor:
        """Return an exponential whose backward saves only its output."""
        _SaveOutputExp.calls += 1
        return ms.ops.exp(tensor)


class _FirstSaveActivation(nn.Cell):
    """First SAVE operator that saves its recomputed input and output activation."""

    calls = 0

    def construct(self, tensor: Tensor) -> Tensor:
        """Save the square input and the ReLU output for backward."""
        _FirstSaveActivation.calls += 1
        return ms.ops.relu(ms.ops.square(tensor))


class _SecondSaveAdd(nn.Cell):
    """Second SAVE operator whose output is not a backward activation."""

    calls = 0

    def construct(self, tensor: Tensor) -> Tensor:
        """Add a constant without saving the input or output for backward."""
        _SecondSaveAdd.calls += 1
        return tensor + 0.125


class _ThirdSaveScale(nn.Cell):
    """Third SAVE operator whose output is not a backward activation."""

    calls = 0

    def construct(self, tensor: Tensor) -> Tensor:
        """Multiply by a fixed scalar without saving input or output."""
        _ThirdSaveScale.calls += 1
        return tensor * 1.5


class _FourthSaveAdd(nn.Cell):
    """Fourth SAVE operator that produces the retained final output."""

    calls = 0

    def construct(self, tensor: Tensor) -> Tensor:
        """Add a fixed constant without saving input or output."""
        _FourthSaveAdd.calls += 1
        return tensor + 0.25


class _SaveChainNet(nn.Cell):
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

    def construct(self, tensor: Tensor) -> Tensor:
        """Return a scalar loss through four adjacent SAVE regions."""
        return self.fourth(self.third(self.second(self.first(tensor * 2)))).sum()


def _run_exclude() -> Dict[str, Any]:
    """Run one optimized or legacy exclusion step."""
    ms.set_deterministic(True)
    ms.set_context(mode=ms.PYNATIVE_MODE)
    calls = {"matmul": 0}
    net = _RmsNormMatmulNet(calls)
    rng = np.random.default_rng(2027)
    input_data = rng.normal(0.0, 0.5, (_TOKEN_NUM, _HIDDEN_SIZE)).astype(np.float32)

    warmup_input = Tensor(input_data, ms.bfloat16)
    warmup_input.requires_grad = True
    warmup_loss = net(warmup_input)
    warmup_loss.backward()
    ms.runtime.synchronize()
    for parameter in net.trainable_params():
        parameter.grad = None
    del warmup_input, warmup_loss
    calls["matmul"] = 0
    ms.runtime.empty_cache()

    baseline_bytes = ms.runtime.memory_allocated()
    ms.runtime.reset_peak_memory_stats()
    tensor = Tensor(input_data, ms.bfloat16)
    tensor.requires_grad = True
    loss = net(tensor)
    ms.runtime.synchronize()
    forward_bytes = ms.runtime.memory_allocated() - baseline_bytes
    loss.backward()
    ms.runtime.synchronize()

    return {
        "loss": float(loss.asnumpy()),
        "matmul_calls": calls["matmul"],
        "forward_bytes": int(forward_bytes),
        "peak_bytes": int(ms.runtime.max_memory_allocated() - baseline_bytes),
    }


def _run_host_performance_comparison() -> Dict[str, Any]:
    """Compare SAVE-fusion host overhead on one warmed 20-layer network."""
    ms.set_deterministic(True)
    ms.set_context(mode=ms.PYNATIVE_MODE)
    net = _RmsNormMatmulNet({"matmul": 0})
    rng = np.random.default_rng(2030)
    input_data = rng.normal(0.0, 0.5, (_TOKEN_NUM, _HIDDEN_SIZE)).astype(np.float32)
    original_mark = _EXCLUDE_MODULE._mark_recompute_inputs  # pylint: disable=protected-access
    original_finalize = _EXCLUDE_MODULE._finalize_save_outputs  # pylint: disable=protected-access
    timings = {
        "current": {"forward": [], "backward": [], "loss": []},
        "baseline": {"forward": [], "backward": [], "loss": []},
    }

    def set_mode(mode: str) -> None:
        """Switch only the two Python paths introduced by SAVE fusion."""
        if mode == "current":
            mark_inputs = original_mark
            finalize_outputs = original_finalize
        else:
            mark_inputs = _mark_recompute_inputs_before_save_fusion
            finalize_outputs = _finalize_outputs_before_save_fusion
        _EXCLUDE_MODULE._mark_recompute_inputs = mark_inputs  # pylint: disable=protected-access
        _EXCLUDE_MODULE._finalize_save_outputs = finalize_outputs  # pylint: disable=protected-access

    try:
        set_mode("current")
        warmup_input = Tensor(input_data, ms.bfloat16)
        warmup_input.requires_grad = True
        warmup_loss = net(warmup_input)
        warmup_loss.backward()
        ms.runtime.synchronize()
        del warmup_input, warmup_loss
        for parameter in net.trainable_params():
            parameter.grad = None

        order = (
            "baseline", "current", "current", "baseline", "current",
            "baseline", "baseline", "current", "baseline", "current",
        )
        for mode in order:
            set_mode(mode)
            tensor = Tensor(input_data, ms.bfloat16)
            tensor.requires_grad = True
            ms.runtime.synchronize()
            start = time.perf_counter()
            loss = net(tensor)
            forward_end = time.perf_counter()
            loss.backward()
            backward_end = time.perf_counter()
            ms.runtime.synchronize()
            timings[mode]["forward"].append((forward_end - start) * 1000)
            timings[mode]["backward"].append((backward_end - forward_end) * 1000)
            timings[mode]["loss"].append(float(loss.asnumpy()))
            for parameter in net.trainable_params():
                parameter.grad = None
            del tensor, loss
    finally:
        set_mode("current")

    result = {}
    for mode in ("current", "baseline"):
        forward_values = sorted(timings[mode]["forward"])
        backward_values = sorted(timings[mode]["backward"])
        median_index = len(forward_values) // 2
        forward_median = forward_values[median_index]
        backward_median = backward_values[median_index]
        result[mode] = {
            "loss": timings[mode]["loss"][0],
            "forward_host_ms": forward_median,
            "backward_host_ms": backward_median,
            "total_host_ms": forward_median + backward_median,
            "samples": len(forward_values),
        }
    return result


def _run_save_chain(save_intermediate_outputs: bool) -> Dict[str, Any]:
    """Measure four SAVE regions with or without their intermediate outputs."""
    ms.set_deterministic(True)
    ms.set_context(mode=ms.PYNATIVE_MODE)
    net = checkpoint_wrapper(_SaveChainNet(save_intermediate_outputs))
    rng = np.random.default_rng(2028)
    input_data = rng.normal(0.0, 0.01, (_TOKEN_NUM, _HIDDEN_SIZE)).astype(np.float32)

    warmup_input = Tensor(input_data, ms.bfloat16)
    warmup_input.requires_grad = True
    warmup_loss = net(warmup_input)
    warmup_loss.backward()
    ms.runtime.synchronize()
    del warmup_input, warmup_loss
    _FirstSaveActivation.calls = 0
    _SecondSaveAdd.calls = 0
    _ThirdSaveScale.calls = 0
    _FourthSaveAdd.calls = 0
    ms.runtime.empty_cache()

    baseline_bytes = ms.runtime.memory_allocated()
    ms.runtime.reset_peak_memory_stats()
    tensor = Tensor(input_data, ms.bfloat16)
    tensor.requires_grad = True
    loss = net(tensor)
    ms.runtime.synchronize()
    forward_bytes = ms.runtime.memory_allocated() - baseline_bytes
    loss.backward()
    ms.runtime.synchronize()

    return {
        "loss": float(loss.asnumpy()),
        "input_grad": float(tensor.grad[0, 0].asnumpy()),
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


def _mark_recompute_inputs_before_save_fusion(
    invocation_id: object,
    args: Any,
    kwargs: Any,
) -> tuple:
    """Reproduce input marking before SAVE-output provenance was introduced."""
    del invocation_id
    bindings = []
    previous_handles = []
    seen_tensor_ids = set()
    try:
        for path, tensor in _EXCLUDE_MODULE._collect_tensor_inputs(args, kwargs):  # pylint: disable=protected-access
            if isinstance(tensor, ms.Parameter) or id(tensor) in seen_tensor_ids:
                continue
            handle = _EXCLUDE_MODULE._RecomputedInputHandle()  # pylint: disable=protected-access
            key = _EXCLUDE_MODULE._RECOMPUTE_INPUT_HANDLE_KEY  # pylint: disable=protected-access
            previous = tensor._get_user_data(key)  # pylint: disable=protected-access
            previous_handles.append((tensor, previous))
            tensor._set_user_data(key, handle)  # pylint: disable=protected-access
            binding = _EXCLUDE_MODULE._InputBinding(path, handle)  # pylint: disable=protected-access
            bindings.append(binding)
            seen_tensor_ids.add(id(tensor))
    except BaseException:
        _EXCLUDE_MODULE._restore_recompute_inputs(previous_handles)  # pylint: disable=protected-access
        raise
    return bindings, previous_handles


def _finalize_outputs_before_save_fusion(
    output: Any,
    add_recompute_boundary: bool,
    invocation_id: object,
    tensor_leaf_count: Any = None,
) -> Any:
    """Apply boundaries without SAVE provenance as before fusion."""
    del invocation_id
    if isinstance(output, ms.Tensor):
        if tensor_leaf_count is not None:
            tensor_leaf_count[0] += 1
        if add_recompute_boundary:
            trigger = _EXCLUDE_MODULE._get_recompute_trigger()  # pylint: disable=protected-access
            return _EXCLUDE_MODULE._RecomputeBoundary.apply(output, trigger)  # pylint: disable=protected-access
        return output
    if isinstance(output, list):
        return [
            _finalize_outputs_before_save_fusion(item, add_recompute_boundary, None, tensor_leaf_count)
            for item in output
        ]
    if isinstance(output, tuple):
        items = [
            _finalize_outputs_before_save_fusion(item, add_recompute_boundary, None, tensor_leaf_count)
            for item in output
        ]
        if hasattr(output, "_fields"):
            return type(output)(*items)
        return tuple(items)
    if isinstance(output, dict):
        return type(output)(
            (key, _finalize_outputs_before_save_fusion(value, add_recompute_boundary, None, tensor_leaf_count))
            for key, value in output.items()
        )
    return output


def _run_mode(mode: str) -> Dict[str, Any]:
    """Run optimized or legacy checkpoint exclusion."""
    if mode == "exclude":
        return _run_exclude()
    if mode == "save_chain":
        return _run_save_chain(save_intermediate_outputs=False)
    if mode == "legacy_save_chain":
        return _run_save_chain(save_intermediate_outputs=True)
    if mode == "host_performance_comparison":
        return _run_host_performance_comparison()
    if mode != "legacy_exclude":
        raise ValueError(f"Unsupported checkpoint exclusion mode: {mode}")

    original_mark = _EXCLUDE_MODULE._mark_recompute_inputs  # pylint: disable=protected-access
    _EXCLUDE_MODULE._mark_recompute_inputs = _disable_input_rematerialization  # pylint: disable=protected-access
    try:
        return _run_exclude()
    finally:
        _EXCLUDE_MODULE._mark_recompute_inputs = original_mark  # pylint: disable=protected-access


def _run_mode_in_subprocess(mode: str) -> Dict[str, Any]:
    """Run one mode in an isolated allocator process."""
    project_root = Path(__file__).resolve().parents[4]
    command = [
        sys.executable,
        "-c",
        (
            "import json; "
            "from tests.mindspore.st.activation_checkpoint.checkpoint_exclude_matmul import _run_mode; "
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

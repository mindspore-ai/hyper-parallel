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


def _disable_input_rematerialization(args: Any, kwargs: Any) -> tuple:
    """Reproduce legacy exclusion, which retains saved input storage."""
    del args, kwargs
    return [], []


def _run_mode(mode: str) -> Dict[str, Any]:
    """Run optimized or legacy checkpoint exclusion."""
    if mode == "exclude":
        return _run_exclude()
    if mode != "legacy_exclude":
        raise ValueError(f"Unsupported RMSNorm/MatMul mode: {mode}")

    exclude_module = importlib.import_module(
        "hyper_parallel.platform.mindspore.activation_checkpoint.checkpoint_exclude_wrapper"
    )
    original_mark = exclude_module._mark_recompute_inputs  # pylint: disable=protected-access
    exclude_module._mark_recompute_inputs = _disable_input_rematerialization  # pylint: disable=protected-access
    try:
        return _run_exclude()
    finally:
        exclude_module._mark_recompute_inputs = original_mark  # pylint: disable=protected-access


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

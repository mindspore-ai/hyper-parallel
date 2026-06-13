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
"""MindSpore ST: comm_fusion=False HSDP backward RS/AR layer overlap (PR #823).

Guards the per-layer post-backward pipeline (AllReduceParamGroup + 4-step overlap)
on an 8-card HSDP mesh. Functional case checks numerics and that fused async
all-reduce is issued during backward; performance case times the backward phase
and asserts a conservative CI ceiling.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import List, Tuple

import mindspore as ms
import numpy as np
from mindspore import Tensor, nn
from mindspore.communication import get_rank, init

from hyper_parallel import SkipDTensorDispatch, init_device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.fully_shard.api import HSDPModule, fully_shard
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from hyper_parallel.platform.mindspore.autograd_compat import enable_mindspore_backward_compat
from hyper_parallel.platform.mindspore.fully_shard.param_group import AllReduceParamGroup

ms.set_seed(42)
ms.set_deterministic(True)
enable_mindspore_backward_compat()

_NUM_LAYERS = 8
_HIDDEN = 128
_BATCH = 4
_WARMUP_STEPS = 3
_MEASURE_STEPS = 5
# Conservative CI ceiling for backward-only timing on the small ST model (ms).
_BACKWARD_MEDIAN_CEILING_MS = 10000.0
# 8-layer HSDP overlap should issue fused AR at least once per backward step.
_MIN_FUSED_AR_ISSUES = 1


class _MLPLayer(nn.Cell):
    """Single block wrapped as one fully_shard unit."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Dense(hidden, hidden, weight_init="normal", bias_init="zeros")
        self.relu = nn.ReLU()
        self.fc2 = nn.Dense(hidden, hidden, weight_init="normal", bias_init="zeros")

    def construct(self, hidden_states: Tensor) -> Tensor:
        return self.fc2(self.relu(self.fc1(hidden_states)))


class _StackedMLP(nn.Cell):
    """Multi-layer model for layer-wise overlap."""

    def __init__(self, num_layers: int, hidden: int) -> None:
        super().__init__()
        self.layers = nn.CellList([_MLPLayer(hidden) for _ in range(num_layers)])

    def construct(self, hidden_states: Tensor) -> Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states.sum()


def _ms_sync() -> None:
    ms.runtime.synchronize()


def _build_hsdp_mesh():
    return init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("replicate", "shard"),
    )


def _setup_layer_prefetch(model: _StackedMLP) -> None:
    """Configure one-hop forward/backward prefetch between wrapped layers."""
    layers = [layer for layer in model.layers if isinstance(layer, HSDPModule)]
    if len(layers) < 2:
        return
    for idx, layer in enumerate(layers):
        fwd_targets = layers[idx + 1 : min(len(layers), idx + 2)]
        if fwd_targets:
            layer.set_modules_to_forward_prefetch(fwd_targets)
    rev_layers = list(reversed(layers))
    for idx, layer in enumerate(rev_layers):
        bwd_targets = rev_layers[idx + 1 : min(len(rev_layers), idx + 2)]
        if bwd_targets:
            layer.set_modules_to_backward_prefetch(bwd_targets)


def _build_sharded_model(mesh) -> _StackedMLP:
    """Build a per-layer fully_shard model on ``mesh`` with comm_fusion=False."""
    mp_policy = MixedPrecisionPolicy(
        param_dtype=ms.float32,
        reduce_dtype=ms.float32,
        output_dtype=ms.float32,
        cast_forward_inputs=False,
    )
    ms.set_seed(42)
    with SkipDTensorDispatch():
        model = _StackedMLP(_NUM_LAYERS, _HIDDEN)
    shard_kw = {"mesh": mesh, "mp_policy": mp_policy, "comm_fusion": False}
    for layer in model.layers:
        fully_shard(layer, **shard_kw)
    fully_shard(model, **shard_kw)
    model.set_reduce_op_type("avg")
    _setup_layer_prefetch(model)
    return model


def _assert_comm_fusion_false(model: _StackedMLP) -> None:
    modules = [model, *model.layers]
    for mod in modules:
        scheduler = mod.hsdp_scheduler
        state = scheduler.hsdp_state
        assert state.config.comm_fusion is False, (
            f"{type(mod).__name__}: expected comm_fusion=False, got {state.config.comm_fusion}"
        )
        assert getattr(state, "param_group", None) is None, (
            f"{type(mod).__name__}: comm_fusion=False must not allocate fusion param_group"
        )


def _fixed_inputs(num_steps: int) -> List[Tensor]:
    rng = np.random.default_rng(84)
    return [
        Tensor(rng.standard_normal((_BATCH, _HIDDEN)).astype(np.float32))
        for _ in range(num_steps)
    ]


def _collect_first_layer_grad(model: _StackedMLP) -> np.ndarray:
    grad = model.layers[0].fc1.weight.grad
    assert grad is not None, "expected gradient on first layer weight"
    local = grad.to_local() if isinstance(grad, DTensor) else grad
    return local.asnumpy().copy()


def _run_train_steps(model: _StackedMLP, inputs: List[Tensor]) -> np.ndarray:
    last_grad = None
    with SkipDTensorDispatch():
        for data in inputs:
            model.zero_grad()
            loss = model(data)
            loss.backward()
            last_grad = _collect_first_layer_grad(model)
    assert last_grad is not None
    return last_grad


@contextmanager
def _count_fused_allreduce_issues():
    """Count AllReduceParamGroup.issue_async_allreduce calls during backward."""
    counter = {"count": 0}
    original = AllReduceParamGroup.issue_async_allreduce

    def wrapped(group_self) -> None:
        counter["count"] += 1
        return original(group_self)

    AllReduceParamGroup.issue_async_allreduce = wrapped
    try:
        yield counter
    finally:
        AllReduceParamGroup.issue_async_allreduce = original


def _median(values: List[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _time_backward_steps(
    model: _StackedMLP,
    inputs: List[Tensor],
) -> Tuple[List[float], List[float]]:
    """Return per-step forward and backward latencies in milliseconds (NPU-synced)."""
    forward_ms: List[float] = []
    backward_ms: List[float] = []
    with SkipDTensorDispatch():
        for data in inputs:
            model.zero_grad()
            _ms_sync()
            t0 = time.perf_counter()
            loss = model(data)
            _ms_sync()
            forward_ms.append((time.perf_counter() - t0) * 1000.0)

            _ms_sync()
            t1 = time.perf_counter()
            loss.backward()
            _ms_sync()
            backward_ms.append((time.perf_counter() - t1) * 1000.0)
    return forward_ms, backward_ms


def test_ms_hsdp_backward_overlap_functional():
    """
    Feature: comm_fusion=False HSDP backward RS/AR layer overlap (MindSpore).
    Description: 8-card HSDP mesh (2x4), per-layer fully_shard with prefetch,
        comm_fusion=False. Verify overlap path issues fused async all-reduce during
        backward and that repeated deterministic runs produce identical reduced grads.
    Expectation: Run success.
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    init()
    rank_id = get_rank()
    mesh = _build_hsdp_mesh()

    model_a = _build_sharded_model(mesh)
    model_b = _build_sharded_model(mesh)
    _assert_comm_fusion_false(model_a)
    _assert_comm_fusion_false(model_b)

    inputs = _fixed_inputs(3)
    with _count_fused_allreduce_issues() as counter:
        grad_a = _run_train_steps(model_a, inputs)
    assert counter["count"] >= _MIN_FUSED_AR_ISSUES, (
        "Expected fused async all-reduce during backward overlap pipeline, "
        f"got {counter['count']} issue_async_allreduce calls"
    )

    grad_b = _run_train_steps(model_b, inputs)
    assert np.allclose(grad_a, grad_b, rtol=1e-5, atol=1e-5), (
        "Deterministic overlap backward must produce identical reduced grads"
    )

    if rank_id == 0:
        print(
            "ms HSDP backward overlap functional passed: "
            f"fused_ar_issues={counter['count']}"
        )


def test_ms_hsdp_backward_overlap_performance():
    """
    Feature: comm_fusion=False HSDP backward overlap performance guard (MindSpore).
    Description: Same 8-card HSDP per-layer setup; warmup then measure backward
        latency over several steps with NPU sync. Assert overlap path is exercised
        and backward median stays below a conservative CI ceiling.
    Expectation: Run success.
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    init()
    rank_id = get_rank()
    mesh = _build_hsdp_mesh()
    model = _build_sharded_model(mesh)
    _assert_comm_fusion_false(model)

    warmup_inputs = _fixed_inputs(_WARMUP_STEPS)
    _run_train_steps(model, warmup_inputs)

    measure_inputs = _fixed_inputs(_MEASURE_STEPS)
    with _count_fused_allreduce_issues() as counter:
        forward_ms, backward_ms = _time_backward_steps(model, measure_inputs)

    assert counter["count"] >= _MIN_FUSED_AR_ISSUES * _MEASURE_STEPS, (
        "Expected fused async all-reduce on each measured backward step, "
        f"got {counter['count']} calls over {_MEASURE_STEPS} steps"
    )

    backward_median = _median(backward_ms)
    forward_median = _median(forward_ms)
    assert backward_median <= _BACKWARD_MEDIAN_CEILING_MS, (
        f"Backward median {backward_median:.2f}ms exceeds ceiling "
        f"{_BACKWARD_MEDIAN_CEILING_MS:.2f}ms"
    )

    if rank_id == 0:
        print(
            "ms HSDP backward overlap performance passed: "
            f"backward_median={backward_median:.2f}ms "
            f"forward_median={forward_median:.2f}ms "
            f"fused_ar_issues={counter['count']}"
        )

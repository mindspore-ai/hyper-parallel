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
"""MindSpore ST: HSDP AVG gradient scaling correctness (comm_fusion=False vs True).

Validates that the AllReduceParamGroup manual AVG division path
(``comm_fusion=False``, fused AR uses SUM + ``/ replicate_world_size``)
matches the native HCCL AVG path (``comm_fusion=True``).

Also checks AVG vs SUM ratio on the sharded gradient (expect ``world_size``).
Rank 0 optionally dumps the no-fusion AVG grad for cross-platform comparison.
"""
from __future__ import annotations

import os
from typing import Dict

import mindspore as ms
import numpy as np
from mindspore import Parameter, Tensor, nn
from mindspore.communication import get_group_size, get_rank, init

from hyper_parallel import SkipDTensorDispatch, init_device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.fully_shard.api import HSDPModule, fully_shard
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from hyper_parallel.platform.mindspore.autograd_compat import enable_mindspore_backward_compat

ms.set_seed(42)
ms.set_deterministic(True)
enable_mindspore_backward_compat()

_NUM_LAYERS = 2
_HIDDEN = 64
_BATCH = 4
_RTOL = 1e-4
_ATOL = 1e-5
_GRAD_DUMP_ENV = "HP_HSDP_AVG_GRAD_DUMP_PATH"


class _MLPLayer(nn.Cell):
    """Single block wrapped as one fully_shard unit."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Dense(hidden, hidden, has_bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Dense(hidden, hidden, has_bias=True)

    def construct(self, hidden_states: Tensor) -> Tensor:
        return self.fc2(self.relu(self.fc1(hidden_states)))


class _StackedMLP(nn.Cell):
    """Small multi-layer model for HSDP AVG grad checks."""

    def __init__(self, num_layers: int, hidden: int) -> None:
        super().__init__()
        self.layers = nn.CellList([_MLPLayer(hidden) for _ in range(num_layers)])

    def construct(self, hidden_states: Tensor) -> Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states.sum()


def _build_hsdp_mesh():
    return init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("replicate", "shard"),
    )


def _mp_policy() -> MixedPrecisionPolicy:
    return MixedPrecisionPolicy(
        param_dtype=ms.float32,
        reduce_dtype=ms.float32,
        output_dtype=ms.float32,
        cast_forward_inputs=False,
    )


def _init_deterministic_weights(model: _StackedMLP) -> None:
    """Assign fixed weights so cross-platform dumps are comparable."""
    rng = np.random.default_rng(1234)
    with SkipDTensorDispatch():
        for param in model.get_parameters():
            if not isinstance(param, Parameter):
                continue
            shape = param.shape
            values = rng.standard_normal(shape).astype(np.float32) * 0.05
            param.set_data(Tensor(values))


def _set_reduce_op_all(model: _StackedMLP, reduce_op: str) -> None:
    """Apply reduce_op to every wrapped fully_shard unit (root + layers)."""
    modules = [model, *model.layers]
    for module in modules:
        if isinstance(module, HSDPModule):
            module.set_reduce_op_type(reduce_op)


def _build_sharded_model(mesh, *, comm_fusion: bool, reduce_op: str) -> _StackedMLP:
    """Build per-layer fully_shard model with the requested comm/reduce settings."""
    with SkipDTensorDispatch():
        model = _StackedMLP(_NUM_LAYERS, _HIDDEN)
    _init_deterministic_weights(model)
    shard_kw = {"mesh": mesh, "mp_policy": _mp_policy(), "comm_fusion": comm_fusion}
    for layer in model.layers:
        fully_shard(layer, **shard_kw)
    fully_shard(model, **shard_kw)
    _set_reduce_op_all(model, reduce_op)
    return model


def _per_rank_input() -> Tensor:
    """Different micro-batch per rank to exercise real reduce semantics."""
    rank_id = get_rank()
    rng = np.random.default_rng(84 + rank_id)
    return Tensor(rng.standard_normal((_BATCH, _HIDDEN)).astype(np.float32))


def _to_local_numpy(grad) -> np.ndarray:
    local = grad.to_local() if isinstance(grad, DTensor) else grad
    return local.asnumpy().copy()


def _collect_named_grads(model: _StackedMLP) -> Dict[str, np.ndarray]:
    """Collect local-shard grads for the first layer parameters."""
    layer = model.layers[0]
    out: Dict[str, np.ndarray] = {}
    for name, param in (
        ("fc1.weight", layer.fc1.weight),
        ("fc1.bias", layer.fc1.bias),
        ("fc2.weight", layer.fc2.weight),
        ("fc2.bias", layer.fc2.bias),
    ):
        grad = param.grad
        assert grad is not None, f"missing grad for {name}"
        out[name] = _to_local_numpy(grad)
    return out


def _run_one_backward(model: _StackedMLP, data: Tensor) -> Dict[str, np.ndarray]:
    with SkipDTensorDispatch():
        model.zero_grad()
        loss = model(data)
        loss.backward()
    return _collect_named_grads(model)


def _assert_grad_dicts_close(
    left: Dict[str, np.ndarray],
    right: Dict[str, np.ndarray],
    *,
    label: str,
) -> None:
    assert left.keys() == right.keys(), f"{label}: grad key mismatch"
    for key in left:
        assert np.allclose(left[key], right[key], rtol=_RTOL, atol=_ATOL), (
            f"{label}: grad mismatch on {key}, "
            f"max_abs_diff={np.max(np.abs(left[key] - right[key]))}"
        )


def _assert_avg_sum_ratio(
    grad_avg: np.ndarray,
    grad_sum: np.ndarray,
    *,
    world_size: int,
    key: str,
) -> None:
    """SUM reduce should scale linearly against AVG by the HSDP world size."""
    mask = np.abs(grad_avg) > 1e-6
    if not np.any(mask):
        return
    ratio = grad_sum[mask] / grad_avg[mask]
    expected = float(world_size)
    assert np.allclose(ratio, expected, rtol=1e-3, atol=1e-3), (
        f"AVG vs SUM ratio mismatch on {key}: expected {expected}, "
        f"median={np.median(ratio)}, max_dev={np.max(np.abs(ratio - expected))}"
    )


def test_ms_hsdp_avg_grad_scale_parity():
    """
    Feature: HSDP AVG gradient scaling with comm_fusion=False (MindSpore).
    Description: 8-card HSDP mesh (2x4), per-layer fully_shard. Compare
        comm_fusion=False (AllReduceParamGroup SUM+manual AVG) against
        comm_fusion=True (native HCCL AVG), and verify AVG vs SUM ratio.
    Expectation: Run success.
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    init()
    rank_id = get_rank()
    world_size = get_group_size()
    mesh = _build_hsdp_mesh()
    data = _per_rank_input()

    model_no_fusion_avg = _build_sharded_model(mesh, comm_fusion=False, reduce_op="avg")
    model_fusion_avg = _build_sharded_model(mesh, comm_fusion=True, reduce_op="avg")
    model_no_fusion_sum = _build_sharded_model(mesh, comm_fusion=False, reduce_op="sum")

    grad_no_fusion_avg = _run_one_backward(model_no_fusion_avg, data)
    grad_fusion_avg = _run_one_backward(model_fusion_avg, data)
    grad_no_fusion_sum = _run_one_backward(model_no_fusion_sum, data)

    _assert_grad_dicts_close(
        grad_no_fusion_avg,
        grad_fusion_avg,
        label="comm_fusion=False vs True (AVG)",
    )

    for key, grad_avg in grad_no_fusion_avg.items():
        _assert_avg_sum_ratio(
            grad_avg,
            grad_no_fusion_sum[key],
            world_size=world_size,
            key=key,
        )

    dump_path = os.environ.get(_GRAD_DUMP_ENV)
    if dump_path and rank_id == 0:
        np.savez(dump_path, **grad_no_fusion_avg)
        print(f"rank 0 dumped no-fusion AVG grads to {dump_path}")

    if rank_id == 0:
        print(
            "ms HSDP AVG grad scale parity passed: "
            f"world_size={world_size}, keys={sorted(grad_no_fusion_avg.keys())}"
        )

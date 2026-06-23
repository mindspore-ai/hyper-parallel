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
"""Torch ST: HSDP AVG gradient scaling correctness (comm_fusion=False vs True).

Mirrors the MindSpore case in ``tests/mindspore/st/fully_shard/_test_hsdp_avg_grad_scale_parity.py``.
"""
from __future__ import annotations

import os
from typing import Dict

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

import numpy as np
import torch
import torch_npu  # pylint: disable=unused-import
from torch import nn

from hyper_parallel import SkipDTensorDispatch, init_device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.fully_shard.api import HSDPModule, fully_shard
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from tests.torch.utils import init_dist, to_device

_NUM_LAYERS = 2
_HIDDEN = 64
_BATCH = 4
_RTOL = 1e-4
_ATOL = 1e-5
_GRAD_DUMP_ENV = "HP_HSDP_AVG_GRAD_DUMP_PATH"


class _MLPLayer(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, hidden, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(hidden_states)))


class _StackedMLP(nn.Module):
    def __init__(self, num_layers: int, hidden: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_MLPLayer(hidden) for _ in range(num_layers)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
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
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        output_dtype=torch.float32,
        cast_forward_inputs=False,
    )


def _init_deterministic_weights(model: _StackedMLP, device_type: str) -> None:
    rng = np.random.default_rng(1234)
    with torch.no_grad():
        for param in model.parameters():
            values = rng.standard_normal(tuple(param.shape)).astype(np.float32) * 0.05
            param.copy_(to_device(torch.from_numpy(values), device_type))


def _set_reduce_op_all(model: _StackedMLP, reduce_op: str) -> None:
    modules = [model, *model.layers]
    for module in modules:
        if isinstance(module, HSDPModule):
            module.set_reduce_op_type(reduce_op)


def _build_sharded_model(mesh, *, comm_fusion: bool, reduce_op: str, device_type: str) -> _StackedMLP:
    model = _StackedMLP(_NUM_LAYERS, _HIDDEN)
    _init_deterministic_weights(model, device_type)
    model = to_device(model, device_type)
    shard_kw = {"mesh": mesh, "mp_policy": _mp_policy(), "comm_fusion": comm_fusion}
    for layer in model.layers:
        fully_shard(layer, **shard_kw)
    fully_shard(model, **shard_kw)
    _set_reduce_op_all(model, reduce_op)
    return model


def _per_rank_input(device_type: str, rank_id: int) -> torch.Tensor:
    rng = np.random.default_rng(84 + rank_id)
    data = torch.from_numpy(rng.standard_normal((_BATCH, _HIDDEN)).astype(np.float32))
    return to_device(data, device_type)


def _to_local_numpy(grad: torch.Tensor) -> np.ndarray:
    local = grad.to_local() if isinstance(grad, DTensor) else grad
    return local.detach().cpu().numpy().copy()


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


def _run_one_backward(model: _StackedMLP, data: torch.Tensor) -> Dict[str, np.ndarray]:
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


def test_torch_hsdp_avg_grad_scale_parity():
    """
    Feature: HSDP AVG gradient scaling with comm_fusion=False (Torch).
    Description: Same 8-card HSDP setup as the MindSpore ST; compare fusion paths
        and AVG vs SUM ratio.
    Expectation: Run success.
    """
    rank_id, _ = init_dist()
    world_size = torch.distributed.get_world_size()
    device_type = "npu"
    mesh = _build_hsdp_mesh()
    data = _per_rank_input(device_type, rank_id)

    model_no_fusion_avg = _build_sharded_model(
        mesh, comm_fusion=False, reduce_op="avg", device_type=device_type
    )
    model_fusion_avg = _build_sharded_model(
        mesh, comm_fusion=True, reduce_op="avg", device_type=device_type
    )
    model_no_fusion_sum = _build_sharded_model(
        mesh, comm_fusion=False, reduce_op="sum", device_type=device_type
    )

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
            "torch HSDP AVG grad scale parity passed: "
            f"world_size={world_size}, keys={sorted(grad_no_fusion_avg.keys())}"
        )

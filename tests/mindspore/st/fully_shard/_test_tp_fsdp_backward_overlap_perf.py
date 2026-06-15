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
"""MindSpore ST: TP/EP + FSDP backward overlap performance diagnostics.

Measures backward latency for composite parallel layouts under ``comm_fusion=False``
and counts whether ``reduce_scatter_grad(..., output_buffer=...)`` hits the extra
``copy_without_bumping_version`` path when FSDP does not issue a real reduce-scatter.
"""
from __future__ import annotations

# pylint: disable=protected-access

import os

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import mindspore as ms
import numpy as np
from mindspore import Tensor
from mindspore.communication import get_rank, get_group_size, init

from hyper_parallel import DTensor, SkipDTensorDispatch, init_device_mesh
from hyper_parallel.core.fully_shard.api import HSDPModule, fully_shard
from hyper_parallel.core.fully_shard.utils import FSDPMeshInfo, MixedPrecisionPolicy
from hyper_parallel.platform.mindspore.autograd_compat import enable_mindspore_backward_compat
from hyper_parallel.platform.mindspore.fully_shard import _version_utils
from hyper_parallel.platform.mindspore.fully_shard import param as fsdp_param_module
from hyper_parallel.platform.mindspore.fully_shard import state as fsdp_state_module
from hyper_parallel.platform.mindspore.fully_shard.param import MindSporeHSDPParamV2
from hyper_parallel.platform.mindspore.fully_shard.param_group import AllReduceParamGroup

ms.set_seed(42)
enable_mindspore_backward_compat()

_NUM_LAYERS = 8
_HIDDEN = 128
_BATCH = 8
_INPUT = 128
_OUTPUT = 128
_WARMUP_STEPS = 3
_MEASURE_STEPS = 5
_INIT_SEED = 74
_INPUT_SEED = 72
_LABEL_SEED = 73
_BACKWARD_CEILING_PURE_HSDP_MS = 10000.0
_BACKWARD_CEILING_COMPOSITE_MS = 15000.0
_MAX_SLOWDOWN_VS_PURE_HSDP = 3.0


@dataclass
class _BackwardPerfStats:
    """Collected backward perf and overlap-path diagnostics."""

    backward_median_ms: float
    forward_median_ms: float
    rs_with_output_buffer: int
    rs_output_buffer_copy_path: int
    copy_without_bump_calls: int
    direct_compat_ar: int
    fused_ar_issues: int


class _MLPLayer(ms.nn.Cell):
    """Single block wrapped as one fully_shard unit."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.fc1 = ms.nn.Dense(hidden, hidden, weight_init="normal", bias_init="zeros")
        self.relu = ms.nn.ReLU()
        self.fc2 = ms.nn.Dense(hidden, hidden, weight_init="normal", bias_init="zeros")

    def construct(self, hidden_states: Tensor) -> Tensor:
        return self.fc2(self.relu(self.fc1(hidden_states)))


class _StackedMLP(ms.nn.Cell):
    """Multi-layer model for layer-wise overlap."""

    def __init__(self, num_layers: int, hidden: int) -> None:
        super().__init__()
        self.layers = ms.nn.CellList([_MLPLayer(hidden) for _ in range(num_layers)])

    def construct(self, hidden_states: Tensor) -> Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states.sum()


def _ms_sync() -> None:
    ms.runtime.synchronize()


def _median(values: List[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _build_pure_hsdp_mesh():
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


def _build_pure_hsdp_model(mesh) -> _StackedMLP:
    """Wrap a stacked MLP with per-layer fully_shard on the given HSDP mesh."""
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
    for idx, layer in enumerate(model.layers):
        fully_shard(layer, **shard_kw)
        model.layers[idx] = layer
    fully_shard(model, **shard_kw)
    model.set_reduce_op_type("avg")
    _setup_layer_prefetch(model)
    return model


def _predict_output_buffer_copy_path(hsdp_param: MindSporeHSDPParamV2) -> bool:
    """Return True when reduce_scatter_grad would copy into output_buffer instead of RS."""
    shard_group_info = getattr(hsdp_param, "sharded_group_info", None)
    shard_group = shard_group_info.group if shard_group_info is not None else None
    shard_group_size = shard_group_info.rank_size if shard_group_info is not None else 1
    if shard_group is None and isinstance(hsdp_param.mesh_info, FSDPMeshInfo):
        shard_group = hsdp_param.mesh_info.shard_process_group
        shard_group_size = hsdp_param.shard_world_size
    if not hsdp_param.is_sharded:
        return True
    return shard_group is None or shard_group_size <= 1


@contextmanager
def _instrument_overlap_paths():
    """Count overlap-path activity during backward."""
    counter: Dict[str, int] = {
        "rs_with_output_buffer": 0,
        "rs_output_buffer_copy_path": 0,
        "copy_without_bump_calls": 0,
        "direct_compat_ar": 0,
        "fused_ar_issues": 0,
    }
    orig_copy = _version_utils.copy_without_bumping_version
    orig_param_copy = fsdp_param_module.copy_without_bumping_version
    orig_rs = MindSporeHSDPParamV2.reduce_scatter_grad
    orig_direct = fsdp_state_module.MindSporeHSDPStateV2._queue_direct_compat_all_reduce
    orig_fused_ar = AllReduceParamGroup.issue_async_allreduce

    def counting_copy(dst, src) -> None:
        counter["copy_without_bump_calls"] += 1
        return orig_copy(dst, src)

    def wrapping_rs(self, *args, **kwargs):
        output_buffer = kwargs.get("output_buffer")
        if output_buffer is not None:
            counter["rs_with_output_buffer"] += 1
            if _predict_output_buffer_copy_path(self):
                counter["rs_output_buffer_copy_path"] += 1
        return orig_rs(self, *args, **kwargs)

    def wrapping_direct(state_self, hsdp_param) -> None:
        counter["direct_compat_ar"] += 1
        return orig_direct(state_self, hsdp_param)

    def wrapping_fused_ar(group_self) -> None:
        counter["fused_ar_issues"] += 1
        return orig_fused_ar(group_self)

    # reduce_scatter_grad imports copy_without_bumping_version into param's namespace;
    # patching only _version_utils would miss output_buffer copy-path calls.
    _version_utils.copy_without_bumping_version = counting_copy
    fsdp_param_module.copy_without_bumping_version = counting_copy
    MindSporeHSDPParamV2.reduce_scatter_grad = wrapping_rs
    fsdp_state_module.MindSporeHSDPStateV2._queue_direct_compat_all_reduce = wrapping_direct
    AllReduceParamGroup.issue_async_allreduce = wrapping_fused_ar
    try:
        yield counter
    finally:
        _version_utils.copy_without_bumping_version = orig_copy
        fsdp_param_module.copy_without_bumping_version = orig_param_copy
        MindSporeHSDPParamV2.reduce_scatter_grad = orig_rs
        fsdp_state_module.MindSporeHSDPStateV2._queue_direct_compat_all_reduce = orig_direct
        AllReduceParamGroup.issue_async_allreduce = orig_fused_ar


def _fixed_plain_inputs(num_steps: int) -> List[Tensor]:
    rng = np.random.default_rng(_INPUT_SEED)
    return [
        Tensor(rng.standard_normal((_BATCH, _HIDDEN)).astype(np.float32))
        for _ in range(num_steps)
    ]


def _run_plain_backward_steps(model: _StackedMLP, inputs: List[Tensor]) -> None:
    with SkipDTensorDispatch():
        for data in inputs:
            model.zero_grad()
            loss = model(data)
            loss.backward()


def _time_plain_backward_steps(
    model: _StackedMLP,
    inputs: List[Tensor],
) -> Tuple[List[float], List[float]]:
    """Time forward and backward for each input micro-batch."""
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


def _measure_plain_hsdp(model: _StackedMLP) -> _BackwardPerfStats:
    """Warm up and time plain HSDP backward steps while collecting overlap-path counters."""
    warmup = _fixed_plain_inputs(_WARMUP_STEPS)
    measure = _fixed_plain_inputs(_MEASURE_STEPS)
    _run_plain_backward_steps(model, warmup)
    with _instrument_overlap_paths() as counter:
        forward_ms, backward_ms = _time_plain_backward_steps(model, measure)
    stats = _BackwardPerfStats(
        backward_median_ms=_median(backward_ms),
        forward_median_ms=_median(forward_ms),
        rs_with_output_buffer=counter["rs_with_output_buffer"],
        rs_output_buffer_copy_path=counter["rs_output_buffer_copy_path"],
        copy_without_bump_calls=counter["copy_without_bump_calls"],
        direct_compat_ar=counter["direct_compat_ar"],
        fused_ar_issues=counter["fused_ar_issues"],
    )
    _validate_overlap_counters(stats)
    return stats


def _import_tp_fsdp_e2e():
    """Import TP + fully_shard e2e helpers lazily to avoid duplicate global MS setup."""
    from tests.mindspore.st.fully_shard import _tp_fully_shard_e2e as tp_e2e  # pylint: disable=import-outside-toplevel

    return tp_e2e


def _build_tp_fsdp_training_bundle(
    *,
    fsdp_mesh,
    tp_mesh,
    input_size: int,
    output_size: int,
    batch_size: int,
    reduce_op: str,
) -> Tuple[object, DTensor, DTensor, Callable[[], None], Callable[[], Tuple[float, float]]]:
    """Build one TP-sharded + fully_shard model and one training micro-batch."""
    tp_e2e = _import_tp_fsdp_e2e()
    state_dict_np = tp_e2e._build_model_state_dict(input_size, output_size, _INIT_SEED)
    model = tp_e2e.TPFullyShardNet(state_dict_np)
    model, x_placements = tp_e2e._wrap_tp_fully_shard_model(model, tp_mesh, mesh=fsdp_mesh)
    model.set_reduce_op_type(reduce_op)

    input_rng = np.random.default_rng(_INPUT_SEED)
    label_rng = np.random.default_rng(_LABEL_SEED)
    input_data = Tensor(input_rng.standard_normal((batch_size, input_size)).astype(np.float32))
    label_data = Tensor(label_rng.standard_normal((batch_size, output_size)).astype(np.float32))

    dp_size = fsdp_mesh.size()
    dp_idx = tp_e2e._flatten_coordinate(fsdp_mesh.mesh_shape, fsdp_mesh.get_coordinate())
    local_x, local_y = tp_e2e._get_local_batch_slice(input_data, label_data, dp_size, dp_idx)
    dist_x = DTensor.from_local(local_x, tp_mesh, x_placements)
    dist_y = DTensor.from_local(local_y, tp_mesh, x_placements)
    grad_scale = Tensor(1.0 / tp_mesh.size(), ms.float32)

    def forward_fn(data, label):
        y_pred = model(data)
        y_shard = label.redistribute(tp_mesh, y_pred.placements)
        loss = tp_e2e.mse_loss_sum(y_pred, y_shard)
        if isinstance(loss, DTensor):
            loss = loss.reduce_partial()
        return loss

    def run_step() -> None:
        model.zero_grad()
        loss = forward_fn(dist_x, dist_y)
        loss.backward(grad_scale)

    def time_one_step() -> Tuple[float, float]:
        model.zero_grad()
        _ms_sync()
        t0 = time.perf_counter()
        loss = forward_fn(dist_x, dist_y)
        _ms_sync()
        forward_ms = (time.perf_counter() - t0) * 1000.0
        _ms_sync()
        t1 = time.perf_counter()
        loss.backward(grad_scale)
        _ms_sync()
        backward_ms = (time.perf_counter() - t1) * 1000.0
        return forward_ms, backward_ms

    return model, dist_x, dist_y, run_step, time_one_step


def _measure_tp_fsdp_bundle(
    run_step: Callable[[], None],
    time_one_step: Callable[[], Tuple[float, float]],
) -> _BackwardPerfStats:
    """Warm up and time TP+FSDP training steps while collecting overlap-path counters."""
    for _ in range(_WARMUP_STEPS):
        run_step()
    forward_ms: List[float] = []
    backward_ms: List[float] = []
    with _instrument_overlap_paths() as counter:
        for _ in range(_MEASURE_STEPS):
            fwd_ms, bwd_ms = time_one_step()
            forward_ms.append(fwd_ms)
            backward_ms.append(bwd_ms)
    stats = _BackwardPerfStats(
        backward_median_ms=_median(backward_ms),
        forward_median_ms=_median(forward_ms),
        rs_with_output_buffer=counter["rs_with_output_buffer"],
        rs_output_buffer_copy_path=counter["rs_output_buffer_copy_path"],
        copy_without_bump_calls=counter["copy_without_bump_calls"],
        direct_compat_ar=counter["direct_compat_ar"],
        fused_ar_issues=counter["fused_ar_issues"],
    )
    _validate_overlap_counters(stats)
    return stats


def _validate_overlap_counters(stats: _BackwardPerfStats) -> None:
    """Ensure copy-path instrumentation observes reduce_scatter output_buffer copies."""
    if stats.rs_output_buffer_copy_path > 0:
        assert stats.copy_without_bump_calls >= stats.rs_output_buffer_copy_path, (
            "copy_without_bump_calls should include output_buffer copy-path invocations; "
            f"got copy_without_bump_calls={stats.copy_without_bump_calls}, "
            f"rs_output_buffer_copy_path={stats.rs_output_buffer_copy_path}"
        )


def _print_stats(label: str, stats: _BackwardPerfStats) -> None:
    print(
        f"{label}: backward_median={stats.backward_median_ms:.2f}ms "
        f"rs_with_output_buffer={stats.rs_with_output_buffer} "
        f"rs_output_buffer_copy_path={stats.rs_output_buffer_copy_path} "
        f"copy_without_bump_calls={stats.copy_without_bump_calls} "
        f"direct_compat_ar={stats.direct_compat_ar} "
        f"fused_ar_issues={stats.fused_ar_issues}"
    )


def test_ms_pure_hsdp_backward_overlap_perf_baseline():
    """
    Feature: pure HSDP backward overlap performance baseline (MindSpore).
    Description: Per-layer fully_shard on HSDP mesh (2x4) with comm_fusion=False.
    Expectation: Run success.
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    init()
    rank_id = get_rank()
    if get_group_size() != 8:
        if rank_id == 0:
            print(f"Skip pure HSDP baseline: expected world_size=8, got {get_group_size()}")
        return

    model = _build_pure_hsdp_model(_build_pure_hsdp_mesh())
    stats = _measure_plain_hsdp(model)
    assert stats.backward_median_ms <= _BACKWARD_CEILING_PURE_HSDP_MS
    if rank_id == 0:
        _print_stats("pure_hsdp_baseline", stats)


def test_ms_tp_fsdp_2d_backward_overlap_perf():
    """
    Feature: TP + FSDP (2D mesh) backward overlap performance (MindSpore).
    Description: TP-sharded TPFullyShardNet with fully_shard on DP mesh (4x2), comm_fusion=False.
    Expectation: Run success.
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    init()
    rank_id = get_rank()
    world_size = get_group_size()
    tp_e2e = _import_tp_fsdp_e2e()
    root_mesh, dp_mesh, tp_mesh, _ = tp_e2e._build_mesh(world_size, tp_size=2)
    if root_mesh is None:
        if rank_id == 0:
            print(f"Skip TP+FSDP 2D perf: world_size={world_size} cannot form (dp, tp)=(4, 2)")
        return

    _, _, _, run_step, time_one_step = _build_tp_fsdp_training_bundle(
        fsdp_mesh=dp_mesh,
        tp_mesh=tp_mesh,
        input_size=_INPUT,
        output_size=_OUTPUT,
        batch_size=_BATCH,
        reduce_op="sum",
    )
    stats = _measure_tp_fsdp_bundle(run_step, time_one_step)
    assert stats.backward_median_ms <= _BACKWARD_CEILING_COMPOSITE_MS
    if rank_id == 0:
        _print_stats("tp_fsdp_2d", stats)


def test_ms_hsdp_tp_3d_backward_overlap_perf():
    """
    Feature: HSDP + TP (3D mesh) backward overlap performance (MindSpore).
    Description: TP-sharded model with fully_shard on (dp, fsdp) submesh (2x2x2), comm_fusion=False.
    Expectation: Run success.
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    init()
    rank_id = get_rank()
    world_size = get_group_size()
    tp_e2e = _import_tp_fsdp_e2e()
    root_mesh, hsdp_mesh, tp_mesh, _, _ = tp_e2e._build_hsdp_tp_mesh(world_size)
    if root_mesh is None:
        if rank_id == 0:
            print(f"Skip HSDP+TP 3D perf: world_size={world_size} cannot form (2, 2, 2)")
        return

    _, _, _, run_step, time_one_step = _build_tp_fsdp_training_bundle(
        fsdp_mesh=hsdp_mesh,
        tp_mesh=tp_mesh,
        input_size=_INPUT,
        output_size=_OUTPUT,
        batch_size=_BATCH,
        reduce_op="avg",
    )
    stats = _measure_tp_fsdp_bundle(run_step, time_one_step)
    assert stats.backward_median_ms <= _BACKWARD_CEILING_COMPOSITE_MS
    if rank_id == 0:
        _print_stats("hsdp_tp_3d", stats)


def test_ms_dp_tp_ep_3d_backward_overlap_perf():
    """
    Feature: DP + TP + EP + FSDP backward overlap performance (MindSpore).
    Description: TP-sharded model with fully_shard on DP mesh under (dp, tp, ep)=(2, 2, 2).
    Expectation: Run success.
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    init()
    rank_id = get_rank()
    world_size = get_group_size()
    tp_e2e = _import_tp_fsdp_e2e()
    root_mesh, dp_mesh, tp_ep_mesh, _, _ = tp_e2e._build_3d_mesh(world_size)
    if root_mesh is None:
        if rank_id == 0:
            print(f"Skip DP+TP+EP perf: world_size={world_size} cannot form (2, 2, 2)")
        return

    _, _, _, run_step, time_one_step = _build_tp_fsdp_training_bundle(
        fsdp_mesh=dp_mesh,
        tp_mesh=tp_ep_mesh,
        input_size=_INPUT,
        output_size=_OUTPUT,
        batch_size=_BATCH,
        reduce_op="sum",
    )
    stats = _measure_tp_fsdp_bundle(run_step, time_one_step)
    assert stats.backward_median_ms <= _BACKWARD_CEILING_COMPOSITE_MS
    if rank_id == 0:
        _print_stats("dp_tp_ep_3d", stats)


def test_ms_hsdp_tp_vs_pure_hsdp_backward_perf():
    """
    Feature: TP + FSDP backward overlap slowdown guard (MindSpore).
    Description: Compare pure HSDP per-layer baseline against HSDP+TP (2x2x2) composite
        backward latency under comm_fusion=False, and report output_buffer copy-path counters.
    Expectation: Run success.
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    init()
    rank_id = get_rank()
    world_size = get_group_size()
    if world_size != 8:
        if rank_id == 0:
            print(f"Skip slowdown compare: expected world_size=8, got {world_size}")
        return

    pure_stats = _measure_plain_hsdp(_build_pure_hsdp_model(_build_pure_hsdp_mesh()))
    tp_e2e = _import_tp_fsdp_e2e()
    root_mesh, hsdp_mesh, tp_mesh, _, _ = tp_e2e._build_hsdp_tp_mesh(world_size)
    assert root_mesh is not None
    _, _, _, run_step, time_one_step = _build_tp_fsdp_training_bundle(
        fsdp_mesh=hsdp_mesh,
        tp_mesh=tp_mesh,
        input_size=_INPUT,
        output_size=_OUTPUT,
        batch_size=_BATCH,
        reduce_op="avg",
    )
    composite_stats = _measure_tp_fsdp_bundle(run_step, time_one_step)

    assert pure_stats.backward_median_ms <= _BACKWARD_CEILING_PURE_HSDP_MS
    assert composite_stats.backward_median_ms <= _BACKWARD_CEILING_COMPOSITE_MS
    slowdown = composite_stats.backward_median_ms / max(pure_stats.backward_median_ms, 1e-6)
    assert slowdown <= _MAX_SLOWDOWN_VS_PURE_HSDP, (
        f"HSDP+TP backward is {slowdown:.2f}x slower than pure HSDP "
        f"({composite_stats.backward_median_ms:.2f}ms vs {pure_stats.backward_median_ms:.2f}ms)"
    )

    if rank_id == 0:
        _print_stats("pure_hsdp", pure_stats)
        _print_stats("hsdp_tp_3d", composite_stats)
        print(
            "ms HSDP+TP vs pure HSDP backward perf passed: "
            f"slowdown={slowdown:.2f}x "
            f"output_buffer_copy_path={composite_stats.rs_output_buffer_copy_path} "
            f"direct_compat_ar={composite_stats.direct_compat_ar}"
        )

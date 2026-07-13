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
"""Distributed integration tests for DFunction on the MindSpore backend.

Tests cover forward numerical correctness for:
- Basic element-wise add (DTensor dispatch, layout, single vs multi-card parity)
- Custom ReLU activation
- Column-parallel Linear (preprocess → _dispatch_layout_infer path)
- Row-parallel Linear with bias (get_expand_impl non-None path)
"""
import numpy as np
import mindspore as ms
import mindspore.communication.management as D
from mindspore import Tensor
from mindspore import mint

from hyper_parallel import init_device_mesh, DFunction
from hyper_parallel.core.dtensor.dtensor import _build_layout, distribute_tensor
from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_ops import DistributedOp


np.random.seed(42)
_X_NP = np.random.randn(8, 16).astype(np.float32)
_Y_NP = np.random.randn(8, 16).astype(np.float32)
_W_COL_NP = np.random.randn(32, 16).astype(np.float32)
_W_ROW_NP = np.random.randn(32, 16).astype(np.float32)
_BIAS_NP = np.random.randn(32).astype(np.float32)


def setup_module():
    """Initialise the distributed backend once for all tests in this file."""
    ms.set_device("Ascend")
    D.init()


def _local_to_global(dtensor):
    """Gather a DTensor to the full global tensor via all-Replicate redistribution."""
    layout = dtensor.layout
    replicate_placements = tuple(Replicate() for _ in layout.placements)
    return dtensor.redistribute(layout.mesh, replicate_placements).to_local()


# ---------------------------------------------------------------------------
# Op 1: Element-wise add
# ---------------------------------------------------------------------------

class _ElemWiseDistOp(DistributedOp):
    """Element-wise op: output layout equals the first input layout."""

    def __init__(self):
        super().__init__("MsDFuncElemWise")

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        x, y = args[0], args[1]
        local_args = (x.to_local(), y.to_local())
        local_kwargs = {}
        cache_values = [x.layout, y.layout]
        return local_args, local_kwargs, cache_values

    def infer_layout(self, cache_values: list) -> tuple:  # pylint: disable=W0221
        x_layout = cache_values[0]
        self._check_partial_inputs([x_layout])
        return ((x_layout,), None)


_ElemWiseDistOp()


class _ElemWiseFunc(DFunction):
    """Element-wise add: output = x + y."""

    _op_name = "MsDFuncElemWise"

    @staticmethod
    def forward(ctx, x, y):  # pylint: disable=W0221
        ctx.save_for_backward(x, y)
        return x + y

    @staticmethod
    def backward(ctx, grad):  # pylint: disable=W0221
        return grad, grad


# ---------------------------------------------------------------------------
# Op 2: Custom ReLU activation
# ---------------------------------------------------------------------------

class _ReLUDistOp(DistributedOp):
    """ReLU: output layout equals input layout."""

    def __init__(self):
        super().__init__("MsDFuncReLU")

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        x = args[0]
        local_args = (x.to_local(),)
        local_kwargs = {}
        cache_values = [x.layout]
        return local_args, local_kwargs, cache_values

    def infer_layout(self, cache_values: list) -> tuple:  # pylint: disable=W0221
        x_layout = cache_values[0]
        self._check_partial_inputs([x_layout])
        return ((x_layout,), None)


_ReLUDistOp()


class _ReLUFunc(DFunction):
    """Custom ReLU: output = max(x, 0)."""

    _op_name = "MsDFuncReLU"

    @staticmethod
    def forward(ctx, x):  # pylint: disable=W0221
        ctx.save_for_backward(x)
        return mint.clamp(x, min=0)

    @staticmethod
    def backward(ctx, grad):  # pylint: disable=W0221
        x, = ctx.saved_tensors
        return grad * (x > 0).astype(grad.dtype)


# ---------------------------------------------------------------------------
# Op 3: Column-parallel Linear (preprocess → _dispatch_layout_infer, get_expand_impl=None)
# ---------------------------------------------------------------------------

def _linear_output_layout(x_layout: Layout, w_layout: Layout) -> Layout:
    """Derive F.linear output layout from x[.., in] and w[out, in]."""
    x_map = x_layout.alias_tensor_map
    w_map = w_layout.alias_tensor_map
    if x_map[-1] != w_map[-1]:
        raise ValueError(
            f"Contracting dimensions must have the same layout, "
            f"got x_map[-1]={x_map[-1]}, w_map[-1]={w_map[-1]}"
        )
    out_map = x_map[:-1] + (w_map[0],)
    base = Layout(
        mesh_shape=x_layout.mesh_shape,
        alias_name=x_layout.alias_name,
        rank_list=x_layout.rank_list,
    )
    out_layout = base(*out_map)
    contract = x_map[-1]
    if contract != "None":
        if isinstance(contract, tuple):
            for axis in contract:
                out_layout.set_partial_by_dev_axis(axis, "sum")
        else:
            out_layout.set_partial_by_dev_axis(contract, "sum")
    return out_layout


class _LinColDistOp(DistributedOp):
    """Column-parallel Linear using the preprocess → _dispatch_layout_infer path."""

    def __init__(self):
        super().__init__("MsDFuncLinCol")

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        x, w = args[0], args[1]
        local_args = (x.to_local(), w.to_local())
        local_kwargs = {}
        cache_values = [x.layout, w.layout]
        return local_args, local_kwargs, cache_values

    def infer_layout(self, cache_values: list) -> tuple:  # pylint: disable=W0221
        x_layout, w_layout = cache_values[0], cache_values[1]
        self._check_partial_inputs([x_layout, w_layout])
        out_layout = _linear_output_layout(x_layout, w_layout)
        return ((out_layout,), None)

    def get_expand_impl(self, func, infer_result, cache_values) -> None:  # pylint: disable=W0221,W0613
        return None


_LinColDistOp()


class _LinColFunc(DFunction):
    """Column-parallel linear: output = x @ w.T (no bias)."""

    _op_name = "MsDFuncLinCol"

    @staticmethod
    def forward(ctx, x, w):  # pylint: disable=W0221
        ctx.save_for_backward(x, w)
        return mint.nn.functional.linear(x, w)

    @staticmethod
    def backward(ctx, grad):  # pylint: disable=W0221
        x, w = ctx.saved_tensors
        grad_x = grad @ w
        grad_w = grad.t() @ x
        return grad_x, grad_w


# ---------------------------------------------------------------------------
# Op 4: Row-parallel Linear with bias scaling (get_expand_impl non-None)
# ---------------------------------------------------------------------------

class _LinRowDistOp(DistributedOp):
    """Row-parallel Linear with bias scaling via get_expand_impl."""

    def __init__(self):
        super().__init__("MsDFuncLinRow")

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        x, w, bias = args[0], args[1], args[2]
        local_args = (x.to_local(), w.to_local(), bias)
        cache_values = [x.layout, w.layout, True]
        return local_args, {}, cache_values

    def infer_layout(self, cache_values: list) -> tuple:  # pylint: disable=W0221
        x_layout, w_layout = cache_values[0], cache_values[1]
        self._check_partial_inputs([x_layout, w_layout])
        out_layout = _linear_output_layout(x_layout, w_layout)
        return ((out_layout,), None)

    def get_expand_impl(self, func, infer_result, cache_values):  # pylint: disable=W0221,W0613
        x_layout = cache_values[0]
        bias_present = cache_values[2]
        x_map = x_layout.alias_tensor_map
        contract = x_map[-1]
        if contract == "None" or not bias_present:
            return None
        if isinstance(contract, tuple):
            scaling_factor = 1
            for axis in contract:
                scaling_factor *= x_layout.mesh.get_device_num_along_axis(axis)
        else:
            scaling_factor = x_layout.mesh.get_device_num_along_axis(contract)

        def expand_impl(x, w, bias):
            return func(x, w, bias / scaling_factor)

        return expand_impl


_LinRowDistOp()


class _LinRowFunc(DFunction):
    """Row-parallel linear: output = x @ w.T + bias."""

    _op_name = "MsDFuncLinRow"

    @staticmethod
    def forward(ctx, x, w, bias):  # pylint: disable=W0221
        ctx.save_for_backward(x, w, bias)
        return mint.nn.functional.linear(x, w, bias)

    @staticmethod
    def backward(ctx, grad):  # pylint: disable=W0221
        x, w, _ = ctx.saved_tensors
        grad_x = grad @ w
        grad_w = grad.t() @ x
        grad_bias = grad.sum(0)
        return grad_x, grad_w, grad_bias


# ---------------------------------------------------------------------------
# Tests — group 1: DTensor forward output type and numerical correctness
# ---------------------------------------------------------------------------

def test_dtensor_forward_output_is_dtensor():
    """
    Feature: DFunction with DTensor inputs (MindSpore)
    Description:
        - Apply _ElemWiseFunc to two DTensors sharded on dim 0.
        - Verify the output is a DTensor with the expected placement.
    Expectation: Output is a DTensor matching the input placement.
    """
    from hyper_parallel.core.dtensor.dtensor import DTensor  # pylint: disable=C0415

    x = Tensor(_X_NP)
    y = Tensor(_Y_NP)
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))

    x_dist = distribute_tensor(x, mesh, (Shard(0), Replicate()))
    y_dist = distribute_tensor(y, mesh, (Shard(0), Replicate()))

    result = _ElemWiseFunc.apply(x_dist, y_dist)

    assert isinstance(result, DTensor), (
        f"Expected DTensor output, got {type(result)}"
    )
    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
    assert result.layout == expected_layout, (
        f"Output layout mismatch: expected={expected_layout}, got={result.layout}"
    )


def test_dtensor_forward_numerical_correctness():
    """
    Feature: DFunction with DTensor inputs — numerical correctness (MindSpore)
    Description:
        - Compute x + y standalone and via DFunction with DTensors.
        - Gather and compare.
    Expectation: Numerically equivalent results.
    """
    x = Tensor(_X_NP)
    y = Tensor(_Y_NP)
    standalone = x + y

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_dist = distribute_tensor(x, mesh, (Shard(0), Replicate()))
    y_dist = distribute_tensor(y, mesh, (Shard(0), Replicate()))

    result = _ElemWiseFunc.apply(x_dist, y_dist)
    gathered = _local_to_global(result)

    assert np.allclose(standalone.asnumpy(), gathered.asnumpy(), rtol=1e-5, atol=1e-5), (
        f"Numerical mismatch: max_diff="
        f"{np.max(np.abs(standalone.asnumpy() - gathered.asnumpy()))}"
    )


# ---------------------------------------------------------------------------
# Tests — group 2: single-card vs multi-card parity + ReLU
# ---------------------------------------------------------------------------

def test_single_vs_multi_card_forward_parity():
    """
    Feature: DFunction single-card vs multi-card forward parity (MindSpore)
    Description:
        - Run _ElemWiseFunc on plain tensors (single-card path).
        - Run on DTensors with Shard(0) on DP.
        - Gather and compare.
    Expectation: Results are numerically identical.
    """
    x = Tensor(_X_NP)
    y = Tensor(_Y_NP)

    standalone = _ElemWiseFunc.apply(x, y)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_dist = distribute_tensor(x, mesh, (Shard(0), Replicate()))
    y_dist = distribute_tensor(y, mesh, (Shard(0), Replicate()))
    result_dist = _ElemWiseFunc.apply(x_dist, y_dist)
    gathered = _local_to_global(result_dist)

    assert np.allclose(standalone.asnumpy(), gathered.asnumpy(), rtol=1e-5, atol=1e-5), (
        f"Forward parity failed: max_diff="
        f"{np.max(np.abs(standalone.asnumpy() - gathered.asnumpy()))}"
    )


def test_relu_forward_correctness():
    """
    Feature: Custom ReLU DFunction forward correctness (MindSpore)
    Description:
        - Apply _ReLUFunc to a DTensor with positive and negative values.
        - Gather and compare with standalone mint.clamp(x, min=0).
    Expectation: Numerically equivalent results.
    """
    x_np = np.random.randn(8, 16).astype(np.float32)
    x = Tensor(x_np)
    standalone = mint.clamp(x, min=0)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_dist = distribute_tensor(x, mesh, (Shard(0), Replicate()))
    result_dist = _ReLUFunc.apply(x_dist)
    gathered = _local_to_global(result_dist)

    assert np.allclose(standalone.asnumpy(), gathered.asnumpy(), rtol=1e-5, atol=1e-5), (
        f"ReLU forward parity failed: max_diff="
        f"{np.max(np.abs(standalone.asnumpy() - gathered.asnumpy()))}"
    )


# ---------------------------------------------------------------------------
# Tests — group 3: column-parallel Linear (preprocess dispatch path)
# ---------------------------------------------------------------------------

def test_linear_colwise_dispatch_new():
    """
    Feature: Column-parallel Linear via DFunction (preprocess path, MindSpore)
    Description:
        - x [8, 16] Shard(0) on DP, Replicate on TP.
        - w [32, 16] Replicate.
        - _LinColFunc uses preprocess → _dispatch_layout_infer; get_expand_impl returns None.
    Expectation: Output matches mint.nn.functional.linear(x_full, w_full) gathered.
    """
    x = Tensor(_X_NP)
    w = Tensor(_W_COL_NP)
    standalone = mint.nn.functional.linear(x, w)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_dist = distribute_tensor(x, mesh, (Shard(0), Replicate()))
    w_dist = distribute_tensor(w, mesh, (Replicate(), Replicate()))

    result = _LinColFunc.apply(x_dist, w_dist)

    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
    assert result.layout == expected_layout, (
        f"ColLinear layout mismatch: expected={expected_layout}, got={result.layout}"
    )

    gathered = _local_to_global(result)
    assert np.allclose(standalone.asnumpy(), gathered.asnumpy(), rtol=1e-4, atol=1e-4), (
        f"ColLinear forward mismatch: max_diff="
        f"{np.max(np.abs(standalone.asnumpy() - gathered.asnumpy()))}"
    )


# ---------------------------------------------------------------------------
# Tests — group 4: row-parallel Linear with get_expand_impl
# ---------------------------------------------------------------------------

def test_linear_rowwise_get_expand_impl():
    """
    Feature: Row-parallel Linear with bias scaling via get_expand_impl (MindSpore)
    Description:
        - x [8, 16] and w [32, 16] both Shard(1) on TP — contracting dim is sharded.
        - bias [32] plain tensor.
        - get_expand_impl scales bias by 1/tp_size; output is partial.
        - After reduce_partial(), result matches mint.nn.functional.linear(x, w, bias).
    Expectation: Forward output (after reduction) numerically matches standalone.
    """
    x = Tensor(_X_NP)
    w = Tensor(_W_ROW_NP)
    bias = Tensor(_BIAS_NP)
    standalone = mint.nn.functional.linear(x, w, bias)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_dist = distribute_tensor(x, mesh, (Replicate(), Shard(1)))
    w_dist = distribute_tensor(w, mesh, (Replicate(), Shard(1)))

    result = _LinRowFunc.apply(x_dist, w_dist, bias)
    reduced = result.reduce_partial()
    gathered = _local_to_global(reduced)

    assert np.allclose(standalone.asnumpy(), gathered.asnumpy(), rtol=1e-4, atol=1e-4), (
        f"RowLinear forward mismatch: max_diff="
        f"{np.max(np.abs(standalone.asnumpy() - gathered.asnumpy()))}"
    )

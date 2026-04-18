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
"""Distributed integration tests for DFunction on the PyTorch backend.

Tests cover:
- Basic DTensor dispatch (forward output type, numerical correctness, layout cache)
- Backward gradient correctness through the distributed path
- Single-card vs multi-card forward/backward numerical parity
- Custom ReLU activation (simple single-input op)
- Custom column-parallel Linear (preprocess → _dispatch_new, get_expand_impl=None)
- Custom row-parallel Linear with bias (preprocess + get_expand_impl non-None)
"""
import numpy as np
import torch
import torch.nn.functional as F

from hyper_parallel import init_device_mesh, DFunction
from hyper_parallel.core.dtensor.dtensor import _build_layout, distribute_tensor
from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_ops import DistributedOp
from tests.torch.utils import init_dist
from tests.torch.shard.utils import global_to_local, local_to_global

np.random.seed(42)
_X_NP = np.random.randn(8, 16).astype(np.float32)
_Y_NP = np.random.randn(8, 16).astype(np.float32)
_W_COL_NP = np.random.randn(32, 16).astype(np.float32)   # weight for col-parallel linear
_W_ROW_NP = np.random.randn(32, 16).astype(np.float32)   # weight for row-parallel linear
_BIAS_NP = np.random.randn(32).astype(np.float32)


# ---------------------------------------------------------------------------
# Op 1: Element-wise add  (basic DTensor dispatch tests)
# ---------------------------------------------------------------------------

class _ElemWiseDistOp(DistributedOp):
    """Element-wise op: output layout equals the first input layout."""

    def __init__(self):
        super().__init__("DFuncElemWise")

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:  # pylint: disable=W0613
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

    _op_name = "DFuncElemWise"

    @staticmethod
    def forward(ctx, x, y):  # pylint: disable=W0221
        ctx.save_for_backward(x, y)
        return x + y

    @staticmethod
    def backward(ctx, grad):  # pylint: disable=W0221,W0613
        return grad, grad


# ---------------------------------------------------------------------------
# Op 2: Custom ReLU activation  (single-input, simple backward)
# ---------------------------------------------------------------------------

class _ReLUDistOp(DistributedOp):
    """ReLU: output layout equals input layout."""

    def __init__(self):
        super().__init__("DFuncReLU")

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:  # pylint: disable=W0613
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

    _op_name = "DFuncReLU"

    @staticmethod
    def forward(ctx, x):  # pylint: disable=W0221
        ctx.save_for_backward(x)
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad):  # pylint: disable=W0221
        x, = ctx.saved_tensors
        return grad * (x > 0).to(grad.dtype)


# ---------------------------------------------------------------------------
# Op 3: Column-parallel Linear  (preprocess → _dispatch_new, no expand_impl)
#
# x  [B, in]   Shard(0) on DP, Replicate on TP  → x_map  = ("dp",  "None")
# w  [out, in] Replicate on DP, Replicate on TP  → w_map  = ("None","None")
# out[B, out]  Shard(0) on DP, Replicate on TP   → output = ("dp",  "None")
#
# Contracting dim (in) is not sharded → no partial → get_expand_impl returns None.
# ---------------------------------------------------------------------------

def _linear_output_layout(x_layout: Layout, w_layout: Layout) -> Layout:
    """Derive F.linear output layout from x[.., in] and w[out, in] layouts.

    F.linear computes x @ w.T so the contracting dimension is the last dim of
    both x and w.  If that dimension is sharded the output carries a partial
    (sum) state; otherwise the output is fully materialised.
    """
    x_map = x_layout.alias_tensor_map
    w_map = w_layout.alias_tensor_map
    if x_map[-1] != w_map[-1]:
        raise ValueError(
            f"Contracting dimensions must have the same layout, "
            f"got x_map[-1]={x_map[-1]}, w_map[-1]={w_map[-1]}"
        )
    # Output: batch dims of x  +  output dim of w
    out_map = x_map[:-1] + (w_map[0],)
    base = Layout(
        mesh_shape=x_layout.mesh_shape,
        alias_name=x_layout.alias_name,
        rank_list=x_layout.rank_list,
    )
    out_layout = base(*out_map)
    # Mark partial if contracting dim is sharded
    contract = x_map[-1]
    if contract != "None":
        if isinstance(contract, tuple):
            for axis in contract:
                out_layout.set_partial_by_dev_axis(axis, "sum")
        else:
            out_layout.set_partial_by_dev_axis(contract, "sum")
    return out_layout


class _LinColDistOp(DistributedOp):
    """Column-parallel Linear: w output dim is sharded, contracting dim is not.

    Uses the ``preprocess`` → ``_dispatch_new`` path.  ``get_expand_impl`` returns
    None because the contracting dimension is not sharded (no bias scaling needed).
    """

    def __init__(self):
        super().__init__("DFuncLinCol")

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:  # pylint: disable=W0613
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
    """Column-parallel linear: output = F.linear(x, w) (no bias)."""

    _op_name = "DFuncLinCol"

    @staticmethod
    def forward(ctx, x, w):  # pylint: disable=W0221
        ctx.save_for_backward(x, w)
        return F.linear(x, w)

    @staticmethod
    def backward(ctx, grad):  # pylint: disable=W0221
        x, w = ctx.saved_tensors
        grad_x = grad @ w
        grad_w = grad.t() @ x
        return grad_x, grad_w


# ---------------------------------------------------------------------------
# Op 4: Row-parallel Linear with bias  (preprocess + get_expand_impl non-None)
#
# x  [B, in]   Replicate on DP, Shard(1) on TP  → x_map  = ("None", "tp")
# w  [out, in] Replicate on DP, Shard(1) on TP  → w_map  = ("None", "tp")
# bias [out]   plain tensor (not DTensor)
#
# Contracting dim (in) is "tp"-sharded → partial output on tp axis.
# bias appears in the partial sum tp times, so it must be pre-scaled by 1/tp.
# get_expand_impl returns expand_impl that calls func(x, w, bias/tp_size).
# ---------------------------------------------------------------------------

class _LinRowDistOp(DistributedOp):
    """Row-parallel Linear with bias scaling via get_expand_impl."""

    def __init__(self):
        super().__init__("DFuncLinRow")

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:  # pylint: disable=W0613
        x, w, bias = args[0], args[1], args[2]
        local_args = (x.to_local(), w.to_local(), bias)
        local_kwargs = {}
        cache_values = [x.layout, w.layout, True]
        return local_args, local_kwargs, cache_values

    def infer_layout(self, cache_values: list) -> tuple:  # pylint: disable=W0221
        x_layout, w_layout = cache_values[0], cache_values[1]
        self._check_partial_inputs([x_layout, w_layout])
        out_layout = _linear_output_layout(x_layout, w_layout)
        return ((out_layout,), None)

    def get_expand_impl(self, func, infer_result, cache_values):  # pylint: disable=W0221,W0613,C0116
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
    """Row-parallel linear: output = F.linear(x, w, bias)."""

    _op_name = "DFuncLinRow"

    @staticmethod
    def forward(ctx, x, w, bias):  # pylint: disable=W0221
        ctx.save_for_backward(x, w, bias)
        return F.linear(x, w, bias)

    @staticmethod
    def backward(ctx, grad):  # pylint: disable=W0221,W0612
        x, w, _ = ctx.saved_tensors
        grad_x = grad @ w
        grad_w = grad.t() @ x
        grad_bias = grad.sum(0)
        return grad_x, grad_w, grad_bias


# ---------------------------------------------------------------------------
# Tests — group 1: DTensor forward correctness
# ---------------------------------------------------------------------------

def test_dtensor_forward_output_is_dtensor():
    """
    Feature: DFunction with DTensor inputs
    Description:
        - Apply _ElemWiseFunc to two DTensors sharded on dim 0.
        - Verify the output is a DTensor with the expected placement.
    Expectation: Output is a DTensor matching the input placement.
    """
    from hyper_parallel.core.dtensor.dtensor import DTensor  # pylint: disable=C0415

    init_dist()

    x = torch.from_numpy(_X_NP).npu()
    y = torch.from_numpy(_Y_NP).npu()
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
    Feature: DFunction with DTensor inputs — numerical correctness
    Description:
        - Compute x + y standalone and via DFunction with DTensors.
        - Gather and compare.
    Expectation: Numerically equivalent results.
    """
    init_dist()

    x = torch.from_numpy(_X_NP).npu()
    y = torch.from_numpy(_Y_NP).npu()
    standalone = x + y

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_dist = distribute_tensor(x, mesh, (Shard(0), Replicate()))
    y_dist = distribute_tensor(y, mesh, (Shard(0), Replicate()))

    result = _ElemWiseFunc.apply(x_dist, y_dist)
    gathered = local_to_global(result)

    assert torch.allclose(standalone, gathered, rtol=1e-5, atol=1e-5), (
        f"Numerical mismatch: max_diff={torch.max(torch.abs(standalone - gathered)).item()}"
    )


# ---------------------------------------------------------------------------
# Tests — group 2: layout cache and backward gradient
# ---------------------------------------------------------------------------

def test_layout_cache_hit():
    """
    Feature: DFunction layout caching
    Description:
        - Call apply twice with DTensors of the same layout.
        - infer_layout should be called only once (cache hit on second call).
    Expectation: infer_layout call count does not exceed 1.
    """
    from hyper_parallel.core.shard.ops.parallel_ops_register import get_distributed_op  # pylint: disable=C0415

    init_dist()

    x = torch.from_numpy(_X_NP).npu()
    y = torch.from_numpy(_Y_NP).npu()
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_dist = distribute_tensor(x, mesh, (Shard(0), Replicate()))
    y_dist = distribute_tensor(y, mesh, (Shard(0), Replicate()))

    dist_op = get_distributed_op("DFuncElemWise")
    original_infer = dist_op.infer_layout
    call_count = [0]

    def counting_infer(cache_values):  # pylint: disable=W0613
        call_count[0] += 1
        return original_infer(cache_values)

    dist_op.infer_layout = counting_infer
    try:
        _ElemWiseFunc.apply(x_dist, y_dist)
        _ElemWiseFunc.apply(x_dist, y_dist)
    finally:
        dist_op.infer_layout = original_infer

    assert call_count[0] <= 1, (
        f"infer_layout should be called at most once due to caching, "
        f"but was called {call_count[0]} times"
    )


def test_dtensor_backward_gradient_correctness():
    """
    Feature: DFunction backward with DTensor inputs
    Description:
        - Compute loss via DFunction on distributed tensors.
        - Verify that local tensor gradients are numerically correct.
    Expectation: Gradient values match expected (all-ones from sum loss).
    """
    init_dist()

    x_np = np.ones((8, 16), dtype=np.float32)
    y_np = np.zeros((8, 16), dtype=np.float32)
    x = torch.from_numpy(x_np).npu().requires_grad_(True)
    y = torch.from_numpy(y_np).npu().requires_grad_(True)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_dist = distribute_tensor(x, mesh, (Shard(0), Replicate()))
    y_dist = distribute_tensor(y, mesh, (Shard(0), Replicate()))

    x_local = x_dist.to_local()
    x_local.retain_grad()

    result = _ElemWiseFunc.apply(x_dist, y_dist)
    result.to_local().sum().backward()

    assert x_local.grad is not None, "x local grad is None after backward"
    assert torch.allclose(x_local.grad, torch.ones_like(x_local)), (
        f"x grad mismatch: expected ones, got {x_local.grad}"
    )


# ---------------------------------------------------------------------------
# Tests — group 3: single-card vs multi-card numerical parity
# ---------------------------------------------------------------------------

def test_single_vs_multi_card_forward_parity():
    """
    Feature: DFunction single-card vs multi-card forward parity
    Description:
        - Run _ElemWiseFunc on plain tensors (single-card path).
        - Run _ElemWiseFunc on DTensors with Shard(0) on DP (multi-card path).
        - Gather the distributed result and compare with the single-card result.
    Expectation: Results are numerically identical.
    """
    init_dist()

    x = torch.from_numpy(_X_NP).npu()
    y = torch.from_numpy(_Y_NP).npu()

    # Single-card reference
    standalone = _ElemWiseFunc.apply(x, y)

    # Multi-card via distributed path
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_dist = distribute_tensor(x, mesh, (Shard(0), Replicate()))
    y_dist = distribute_tensor(y, mesh, (Shard(0), Replicate()))
    result_dist = _ElemWiseFunc.apply(x_dist, y_dist)
    gathered = local_to_global(result_dist)

    assert torch.allclose(standalone, gathered, rtol=1e-5, atol=1e-5), (
        f"Forward parity failed: max_diff={torch.max(torch.abs(standalone - gathered)).item()}"
    )


def test_single_vs_multi_card_backward_parity():
    """
    Feature: DFunction single-card vs multi-card backward parity
    Description:
        - Run backward on both paths with identical all-ones input tensors.
        - Verify that each rank's local gradient equals the corresponding slice of
          the single-card gradient.
    Expectation: local_grad == standalone_grad[slice_for_this_rank].
    """
    init_dist()

    x_np = np.ones((8, 16), dtype=np.float32)
    y_np = np.zeros((8, 16), dtype=np.float32)

    # Single-card reference
    x_full = torch.from_numpy(x_np).npu().requires_grad_(True)
    y_full = torch.from_numpy(y_np).npu().requires_grad_(True)
    standalone_out = _ElemWiseFunc.apply(x_full, y_full)
    standalone_out.sum().backward()
    standalone_x_grad = x_full.grad.detach()  # shape [8, 16], all ones

    # Multi-card path
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_dist = distribute_tensor(
        torch.from_numpy(x_np).npu().requires_grad_(True), mesh, (Shard(0), Replicate())
    )
    y_dist = distribute_tensor(
        torch.from_numpy(y_np).npu().requires_grad_(True), mesh, (Shard(0), Replicate())
    )

    x_local = x_dist.to_local()
    x_local.retain_grad()

    result_dist = _ElemWiseFunc.apply(x_dist, y_dist)
    result_dist.to_local().sum().backward()

    local_grad = x_local.grad
    assert local_grad is not None, "x local grad is None after backward"

    # Expected local grad: the slice of the full gradient for this rank
    expected_local_grad = global_to_local(standalone_x_grad, x_dist.layout).to_local()
    assert torch.allclose(local_grad, expected_local_grad, rtol=1e-5, atol=1e-5), (
        f"Backward parity failed: max_diff="
        f"{torch.max(torch.abs(local_grad - expected_local_grad)).item()}"
    )


# ---------------------------------------------------------------------------
# Tests — group 4: custom ReLU + column-parallel linear (preprocess path)
# ---------------------------------------------------------------------------

def test_relu_forward_backward():
    """
    Feature: Custom ReLU DFunction forward and backward
    Description:
        - Apply _ReLUFunc to a DTensor with negative and positive values.
        - Verify output is correct (zeros out negatives).
        - Verify gradient is 1 where input > 0 and 0 elsewhere.
    Expectation: Forward and backward numerically correct.
    """
    init_dist()

    x_np = np.random.randn(8, 16).astype(np.float32)
    x = torch.from_numpy(x_np).npu().requires_grad_(True)

    # Single-card reference
    standalone = x.clamp(min=0)
    standalone.sum().backward()
    standalone_x_grad = x.grad.detach()
    x.grad = None

    # Multi-card
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_dist = distribute_tensor(
        torch.from_numpy(x_np).npu().requires_grad_(True), mesh, (Shard(0), Replicate())
    )

    x_local = x_dist.to_local()
    x_local.retain_grad()

    result_dist = _ReLUFunc.apply(x_dist)
    result_dist.to_local().sum().backward()

    # Forward parity
    gathered = local_to_global(result_dist)
    assert torch.allclose(standalone.detach(), gathered, rtol=1e-5, atol=1e-5), (
        f"ReLU forward parity failed: max_diff="
        f"{torch.max(torch.abs(standalone.detach() - gathered)).item()}"
    )

    # Backward parity
    local_grad = x_local.grad
    assert local_grad is not None, "x local grad is None after backward"
    expected_local_grad = global_to_local(standalone_x_grad, x_dist.layout).to_local()
    assert torch.allclose(local_grad, expected_local_grad, rtol=1e-5, atol=1e-5), (
        f"ReLU backward parity failed: max_diff="
        f"{torch.max(torch.abs(local_grad - expected_local_grad)).item()}"
    )


def test_linear_colwise_dispatch_new():
    """
    Feature: Custom column-parallel Linear via DFunction (preprocess dispatch path)
    Description:
        - x [8, 16] Shard(0) on DP, Replicate on TP.
        - w [32, 16] Replicate on both dims.
        - _LinColFunc.apply takes the preprocess → _dispatch_new path.
        - get_expand_impl returns None (contracting not sharded).
    Expectation: Output matches F.linear(x_full, w_full) gathered; backward runs.
    """
    init_dist()

    x = torch.from_numpy(_X_NP).npu()
    w = torch.from_numpy(_W_COL_NP).npu()
    standalone = F.linear(x, w)  # [8, 32]

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_dist = distribute_tensor(x.requires_grad_(True), mesh, (Shard(0), Replicate()))
    w_dist = distribute_tensor(w.requires_grad_(True), mesh, (Replicate(), Replicate()))

    x_local = x_dist.to_local()
    x_local.retain_grad()

    result = _LinColFunc.apply(x_dist, w_dist)

    # Layout check: output should be Shard(0) on DP, Replicate on TP
    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
    assert result.layout == expected_layout, (
        f"ColLinear output layout mismatch: expected={expected_layout}, got={result.layout}"
    )

    # Forward numerical correctness
    gathered = local_to_global(result)
    assert torch.allclose(standalone, gathered, rtol=1e-4, atol=1e-4), (
        f"ColLinear forward mismatch: max_diff="
        f"{torch.max(torch.abs(standalone - gathered)).item()}"
    )

    # Backward runs without error; check local grad shape
    result.to_local().sum().backward()
    assert x_local.grad is not None, "x local grad is None after backward"
    assert x_local.grad.shape == x_local.shape, (
        f"Unexpected x grad shape: {x_local.grad.shape}"
    )


# ---------------------------------------------------------------------------
# Tests — group 5: row-parallel linear with get_expand_impl
# ---------------------------------------------------------------------------

def test_linear_rowwise_get_expand_impl():
    """
    Feature: Custom row-parallel Linear with bias scaling (get_expand_impl non-None)
    Description:
        - x [8, 16] Replicate on DP, Shard(1) on TP → contracting dim is tp-sharded.
        - w [32, 16] Replicate on DP, Shard(1) on TP.
        - bias [32] plain tensor.
        - get_expand_impl is exercised; it scales bias by 1/tp_size.
        - After reduce_partial(), the output should match F.linear(x_full, w_full, bias).
    Expectation: Forward output (after reduction) numerically matches standalone.
    """
    init_dist()

    x = torch.from_numpy(_X_NP).npu()
    w = torch.from_numpy(_W_ROW_NP).npu()
    bias = torch.from_numpy(_BIAS_NP).npu()
    standalone = F.linear(x, w, bias)  # [8, 32]

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    # Shard contracting (in) dimension across TP
    x_dist = distribute_tensor(x, mesh, (Replicate(), Shard(1)))
    w_dist = distribute_tensor(w, mesh, (Replicate(), Shard(1)))

    # bias is a plain tensor (not DTensor); contracting dim is sharded →
    # get_expand_impl returns expand_impl that pre-scales bias by 1/tp_size.
    result = _LinRowFunc.apply(x_dist, w_dist, bias)

    # result is a partial DTensor; reduce_partial() triggers the allreduce
    reduced = result.reduce_partial()
    gathered = local_to_global(reduced)

    assert torch.allclose(standalone, gathered, rtol=1e-4, atol=1e-4), (
        f"RowLinear forward mismatch: max_diff="
        f"{torch.max(torch.abs(standalone - gathered)).item()}"
    )

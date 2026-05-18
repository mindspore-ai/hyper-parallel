# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Tests for forward_and_gradfn."""
# pylint: disable=import-error,no-name-in-module

import numpy as np
import pytest

from mindspore import Tensor, Parameter, nn, ops, mint
from mindspore import value_and_grad
from hyper_parallel.platform.mindspore.pipeline_parallel.backward import forward_and_gradfn
from tests.common.mark_utils import arg_mark

def _assert_allclose(tensors_a, tensors_b, atol=1e-6, rtol=1e-6):
    assert len(tensors_a) == len(tensors_b)
    for a, b in zip(tensors_a, tensors_b):
        assert np.allclose(a.asnumpy(), b.asnumpy(), atol=atol, rtol=rtol)


def _to_list(grads):
    if grads is None:
        return []
    if isinstance(grads, (list, tuple)):
        return list(grads)
    return [grads]


def test_forward_output_consistency():
    """
    Feature: forward_and_gradfn forward output.
    Description: Forward output should align with value_and_grad baseline.
    Expectation: Outputs match.
    """
    class Net(nn.Cell):
        def construct(self, x, y):
            return ops.relu(ops.matmul(x, y))

    x = Tensor(np.array([[1.0, 2.0]], np.float32))
    y = Tensor(np.array([[3.0], [4.0]], np.float32))

    net = Net()
    forward_baseline, _ = value_and_grad(net)(x, y)
    forward_out, grad_fn = forward_and_gradfn(net, x, y, grad_position=(0, 1))

    assert np.allclose(forward_out.asnumpy(), forward_baseline.asnumpy(), atol=1e-6, rtol=1e-6)
    assert grad_fn is not None


def test_dx_only_inputs():
    """
    Feature: forward_and_gradfn compute_input_grad.
    Description: dx should match value_and_grad when only inputs are differentiated.
    Expectation: Gradients match baseline.
    """
    class Net(nn.Cell):
        def construct(self, a, b, c):
            return (a * b) + ops.sin(c)

    a = Tensor(np.array([1.2], np.float32))
    b = Tensor(np.array([0.7], np.float32))
    c = Tensor(np.array([0.3], np.float32))

    _, grad_fn = forward_and_gradfn(Net(), a, b, c, grad_position=(0, 2))
    dx = grad_fn.compute_input_grad()

    _, grad_baseline = value_and_grad(Net(), grad_position=(0, 2))(a, b, c)
    dx_list = _to_list(dx)
    baseline_list = _to_list(grad_baseline)
    _assert_allclose(dx_list, baseline_list)


def test_call_returns_dx_and_dw_together():
    """
    Feature: GradFunction.__call__ returns dx and dw.
    Description: __call__ should produce both input and weight gradients.
    Expectation: Gradients match value_and_grad baseline.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.w1 = Parameter(Tensor(np.array([[1.0, -1.0]], np.float32)), name="w1")
            self.w2 = Parameter(Tensor(np.array([[2.0], [0.5]], np.float32)), name="w2")

        def construct(self, x):
            hidden = ops.matmul(x, self.w2)
            out = ops.matmul(hidden, self.w1)
            return out

    x = Tensor(np.array([[0.6, -0.8]], np.float32))
    net = Net()
    forward_output, grad_fn = forward_and_gradfn(net, x, weights=(net.w1, net.w2), grad_position=0)
    grads = grad_fn()

    forward_value, grads_baseline = value_and_grad(net, weights=(net.w1, net.w2))(x)
    assert isinstance(grads_baseline, tuple) and len(grads_baseline) == 2
    dx_expected = _to_list(grads_baseline[0])
    dw_expected = _to_list(grads_baseline[1])

    assert isinstance(grads, tuple) and len(grads) == 2
    dx_actual = _to_list(grads[0])
    dw_actual = _to_list(grads[1])
    assert np.allclose(forward_output, forward_value.asnumpy(), atol=1e-6, rtol=1e-6)
    _assert_allclose(dx_actual, dx_expected)
    _assert_allclose(dw_actual, dw_expected)


def test_dx_dw_split_pipeline_consistency():
    """
    Feature: dx/dw split path.
    Description: compute_input_grad then compute_weight_grad should match baseline.
    Expectation: dx and dw match value_and_grad.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.w = Parameter(Tensor(np.array([[1.5, -0.5], [0.2, 1.0]], np.float32)), name="w")

        def construct(self, x):
            y = ops.relu(ops.matmul(x, self.w))
            return y

    x = Tensor(np.array([[1.0, 2.0]], np.float32))
    net = Net()
    _, grad_fn = forward_and_gradfn(net, x, weights=(net.w,), grad_position=0)

    dx = grad_fn.compute_input_grad()
    dw = grad_fn.compute_weight_grad()

    dx_expected = [Tensor(np.array([[1.0, 1.2]], np.float32))]
    dw_expected = [
        Tensor(np.array([[1.0, 1.0], [2.0, 2.0]], np.float32))
    ]

    _assert_allclose(_to_list(dx), dx_expected)
    _assert_allclose(_to_list(dw), dw_expected)


def test_weight_grad_keep_graph_allows_reuse():
    """
    Feature: compute_weight_grad keep_graph reuse.
    Description: keep_graph=True allows multiple compute_weight_grad calls after compute_input_grad.
                 _accumulate_grads uses in-place addition on param._grad to save memory, so the
                 tensor returned by an earlier call shares storage with param._grad and will reflect
                 subsequent accumulations.  The contract being tested is:
                   1. Each call's return value (collected at call time) matches the per-call gradient.
                   2. After N calls, param._grad equals N * per-call gradient (accumulated total).
    Expectation: dw_second matches baseline; net.w._grad equals 2x baseline after two calls.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.w = Parameter(Tensor(np.array([[1.5, -0.5], [0.2, 1.0]], np.float32)), name="w")

        def construct(self, x):
            return ops.relu(ops.matmul(x, self.w))

    x = Tensor(np.array([[1.0, 2.0]], np.float32))
    net = Net()
    _, grad_fn = forward_and_gradfn(net, x, weights=(net.w,), grad_position=0)

    _ = grad_fn.compute_input_grad()
    grad_fn.compute_weight_grad(keep_graph=True)   # first call: accumulates into net.w._grad
    dw_second = grad_fn.compute_weight_grad(keep_graph=True)  # second call: fresh return value

    _, grads_baseline = value_and_grad(net, weights=(net.w,), grad_position=0)(x)
    dw_expected = _to_list(grads_baseline[1])

    # The return value of the second call is freshly computed and equals the per-call gradient.
    _assert_allclose(_to_list(dw_second), dw_expected)
    # After two calls, param._grad holds the accumulated total (2x per-call gradient).
    _assert_allclose([net.w._grad], [2.0 * dw_expected[0]], atol=1e-5, rtol=1e-5)  # pylint: disable=protected-access


def test_multi_output_intermediate_slot_grad():
    """
    Feature: multi-output intermediate gradients.
    Description: Ensure slot-wise gradients are passed correctly for split outputs.
    Expectation: dw matches baseline.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.w = Parameter(Tensor(np.array([[1.0], [2.0], [0.5], [1.5]], np.float32)), name="w")

        def construct(self, x):
            z = ops.matmul(self.w, x)
            split_out = mint.split(z, 2)
            return ops.add(split_out[0], split_out[1])

    x = Tensor(np.array([[3.0]], np.float32))
    net = Net()
    _, grad_fn = forward_and_gradfn(net, x, weights=(net.w,))

    dx = grad_fn.compute_input_grad()
    dw = grad_fn.compute_weight_grad()

    _, grads_baseline = value_and_grad(net, weights=(net.w,))(x)
    dx_expected = _to_list(grads_baseline[0])
    dw_expected = _to_list(grads_baseline[1])

    _assert_allclose(_to_list(dx), dx_expected)
    _assert_allclose(_to_list(dw), dw_expected)


def test_shared_intermediate_weight_group_merge_dw():
    """
    Feature: weight grouping merge.
    Description: Shared intermediates should merge groups; dw matches baseline.
    Expectation: Gradients match value_and_grad.
    """
    class Net(nn.Cell):
        """Test network with shared intermediate weight group."""

        def __init__(self):
            super().__init__()
            self.w1 = Parameter(Tensor(np.array([[1.0], [0.5]], np.float32)), name="w1")
            self.w2 = Parameter(Tensor(np.array([[0.3], [0.7]], np.float32)), name="w2")

        def construct(self, x):
            shared = ops.tanh(x)
            out1 = ops.matmul(shared, self.w1)
            out2 = ops.matmul(shared, self.w2)
            return out1 + out2

    x = Tensor(np.array([[0.4, -0.6]], np.float32))
    net = Net()
    _, grad_fn = forward_and_gradfn(net, x, weights=(net.w1, net.w2), grad_position=0)

    dx = grad_fn.compute_input_grad()
    dw = grad_fn.compute_weight_grad()

    _, grads_baseline = value_and_grad(net, weights=(net.w1, net.w2))(x)
    dx_expected = _to_list(grads_baseline[0])
    dw_expected = _to_list(grads_baseline[1])

    _assert_allclose(_to_list(dx), dx_expected)
    _assert_allclose(_to_list(dw), dw_expected)


def test_complex_multilayer_network_dx_dw_accuracy():
    """
    Feature: complex multilayer network gradient accuracy.
    Description: Multilayer network with multiple weights, residual connections, and branches.
    Expectation: All input and weight gradients match value_and_grad baseline.
    """
    class ComplexNet(nn.Cell):
        """Complex multilayer test network for dx/dw accuracy."""

        def __init__(self):
            super().__init__()
            self.w1 = Parameter(Tensor(np.array([
                [0.5, -0.3, 0.2, 0.1, -0.4],
                [0.2, 0.8, -0.1, 0.6, 0.3],
                [0.1, 0.4, 0.6, -0.2, 0.5],
                [-0.3, 0.7, 0.2, 0.4, -0.1]
            ], np.float32)), name="w1")
            self.b1 = Parameter(Tensor(np.array([[0.1, -0.2, 0.15, 0.05, 0.08]], np.float32)), name="b1")
            self.w_branch = Parameter(Tensor(np.array([
                [0.6, -0.4, 0.2],
                [0.3, 0.5, -0.1],
                [-0.2, 0.8, 0.4],
                [0.1, -0.3, 0.7],
                [0.5, 0.2, -0.6]
            ], np.float32)), name="w_branch")
            self.w2 = Parameter(Tensor(np.array([
                [1.2, 0.4, -0.2],
                [-0.5, 0.7, 0.3],
                [0.3, -0.6, 0.9],
                [0.4, 0.2, -0.5],
                [-0.1, 0.3, 0.6]
            ], np.float32)), name="w2")
            self.b2 = Parameter(Tensor(np.array([[0.3, -0.1, 0.05]], np.float32)), name="b2")
            self.w3 = Parameter(Tensor(np.array([
                [0.7, -0.2],
                [0.9, 0.4],
                [0.2, 0.5]
            ], np.float32)), name="w3")
            self.b3 = Parameter(Tensor(np.array([[0.2, -0.05]], np.float32)), name="b3")
            self.w_residual = Parameter(Tensor(np.array([
                [0.4, -0.1],
                [0.5, 0.2],
                [0.6, -0.3],
                [-0.2, 0.4]
            ], np.float32)), name="w_residual")
            self.scale = Parameter(Tensor(np.array([[1.1, 0.9]], np.float32)), name="scale")

        def construct(self, x1, x2):
            concat_input = ops.concat((x1, x2), axis=1)
            layer1 = ops.relu(ops.matmul(concat_input, self.w1) + self.b1)
            branch1 = ops.matmul(layer1, self.w_branch)
            branch2 = ops.tanh(ops.matmul(layer1, self.w2) + self.b2)
            merged = branch1 + branch2
            layer3 = ops.relu(ops.matmul(merged, self.w3) + self.b3)
            residual = ops.matmul(concat_input, self.w_residual)
            output = (layer3 + residual) * self.scale
            return output

    x1 = Tensor(np.array([[0.8, 1.2]], np.float32))
    x2 = Tensor(np.array([[1.5, -0.3]], np.float32))
    net = ComplexNet()

    _, grad_fn = forward_and_gradfn(
        net, x1, x2,
        weights=(net.w1, net.b1, net.w2, net.b2, net.w3, net.b3, net.w_branch, net.w_residual, net.scale),
        grad_position=(0, 1)
    )

    dx = grad_fn.compute_input_grad()
    dw = grad_fn.compute_weight_grad()

    _, grads_baseline = value_and_grad(
        net,
        weights=(net.w1, net.b1, net.w2, net.b2, net.w3, net.b3, net.w_branch, net.w_residual, net.scale),
        grad_position=(0, 1)
    )(x1, x2)

    assert isinstance(grads_baseline, tuple) and len(grads_baseline) == 2
    dx_expected = _to_list(grads_baseline[0])
    dw_expected = _to_list(grads_baseline[1])

    _assert_allclose(_to_list(dx), dx_expected)
    _assert_allclose(_to_list(dw), dw_expected)


def test_weight_subgraph_meets_input_later_dx_dw_accuracy():
    """
    Feature: weight-only subgraph before meeting input.
    Description: Multiple weights form a multi-op subgraph first, then join with input later.
    Expectation: dx and dw match value_and_grad baseline.
    """
    class Net(nn.Cell):
        """Test network where weight subgraph meets input later."""

        def __init__(self):
            super().__init__()
            self.w1 = Parameter(Tensor(np.array([
                [0.3, -0.1, 0.5],
                [0.2, 0.4, -0.3],
                [-0.6, 0.7, 0.1],
                [0.9, -0.2, 0.8],
            ], np.float32)), name="w1")  # (4, 3)
            self.w2 = Parameter(Tensor(np.array([
                [0.6, -0.4],
                [0.1, 0.7],
                [-0.2, 0.3],
            ], np.float32)), name="w2")  # (3, 2)
            self.b = Parameter(Tensor(np.array([
                [0.05, -0.02],
                [0.01, 0.03],
                [-0.04, 0.02],
                [0.02, -0.01],
            ], np.float32)), name="b")  # (4, 2)
            self.w3 = Parameter(Tensor(np.array([
                [0.2, -0.3, 0.5, 0.1, -0.4],
                [0.7, 0.6, -0.2, 0.3, 0.8],
            ], np.float32)), name="w3")  # (2, 5)
            self.reduce_sum = ops.ReduceSum(keep_dims=False)

        def construct(self, x):
            weight_only_0 = ops.matmul(self.w1, self.w2)  # (4, 2)
            weight_only_1 = ops.relu(weight_only_0 + self.b)  # (4, 2)
            weight_only_2 = ops.tanh(ops.matmul(weight_only_1, self.w3))  # (4, 5)
            joined = ops.matmul(x, weight_only_2)  # (1, 5)
            return self.reduce_sum(joined)

    x = Tensor(np.array([[0.8, -0.6, 1.1, 0.3]], np.float32))  # (1, 4)
    net = Net()

    _, grad_fn = forward_and_gradfn(net, x, weights=(net.w1, net.w2, net.b, net.w3), grad_position=0)
    dx = grad_fn.compute_input_grad()
    dw = grad_fn.compute_weight_grad()

    _, grads_baseline = value_and_grad(net, weights=(net.w1, net.w2, net.b, net.w3), grad_position=0)(x)
    dx_expected = _to_list(grads_baseline[0])
    dw_expected = _to_list(grads_baseline[1])

    _assert_allclose(_to_list(dx), dx_expected)
    _assert_allclose(_to_list(dw), dw_expected)


def test_shared_weight_used_multiple_places_dx_dw_accuracy():
    """
    Feature: shared weight used in multiple places.
    Description: One weight participates in multiple ops/branches; its gradient should accumulate correctly.
    Expectation: dx and dw match value_and_grad baseline.
    """
    class Net(nn.Cell):
        """Test network with shared weight used in multiple places."""

        def __init__(self):
            super().__init__()
            self.w_shared = Parameter(Tensor(np.array([[0.4, -0.2],
                                                       [0.7, 0.3]], np.float32)), name="w_shared")  # (2, 2)
            self.w_a = Parameter(Tensor(np.array([[0.5],
                                                  [-0.6]], np.float32)), name="w_a")  # (2, 1)
            self.w_b = Parameter(Tensor(np.array([[0.2],
                                                  [0.9]], np.float32)), name="w_b")  # (2, 1)
            self.bias = Parameter(Tensor(np.array([[0.05, -0.03]], np.float32)), name="bias")  # (1, 2)
            self.reduce_sum = ops.ReduceSum(keep_dims=False)

        def construct(self, x):
            h0 = ops.matmul(x, self.w_shared) + self.bias  # (1, 2)
            h1 = ops.tanh(h0)
            h2 = ops.matmul(h1, self.w_shared)  # (1, 2)  shared weight used again
            branch_a = ops.matmul(h2, self.w_a)  # (1, 1)
            branch_b = ops.matmul(h2, self.w_b)  # (1, 1)
            out = branch_a + ops.sin(branch_b) + self.reduce_sum(h2)
            return out

    x = Tensor(np.array([[1.1, -0.7]], np.float32))  # (1, 2)
    net = Net()
    weights = (net.w_shared, net.w_a, net.w_b, net.bias)

    _, grad_fn = forward_and_gradfn(net, x, weights=weights, grad_position=0)
    dx = grad_fn.compute_input_grad()
    dw = grad_fn.compute_weight_grad()

    _, grads_baseline = value_and_grad(net, weights=weights, grad_position=0)(x)
    dx_expected = _to_list(grads_baseline[0])
    dw_expected = _to_list(grads_baseline[1])

    _assert_allclose(_to_list(dx), dx_expected)
    _assert_allclose(_to_list(dw), dw_expected)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_dw_only_weights():
    """
    Feature: forward_and_gradfn compute_weight_grad.
    Description: Calling compute_weight_grad by itself should raise ValueError.
    Expectation: Raises ValueError (must call compute_input_grad first).
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.w = Parameter(Tensor(np.array([[2.0]], np.float32)), name="w")

        def construct(self, x):
            return ops.matmul(x, self.w)

    x = Tensor(np.array([[3.0]], np.float32))
    net = Net()
    _, grad_fn = forward_and_gradfn(net, x, weights=(net.w,), grad_position=None)
    err_msg = "Before calling compute_weight_grad, you need first call compute_input_grad!"
    with pytest.raises(RuntimeError, match=err_msg):
        _ = grad_fn.compute_weight_grad()


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_exceptions_and_edge_cases():
    """
    Feature: robustness checks.
    Description: Validate error paths and empty inputs/weights.
    Expectation: Proper exceptions or empty gradients.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.w = Parameter(Tensor(np.array([[1.0]], np.float32)), name="w")

        def construct(self, x):
            return ops.matmul(x, self.w)

    x = Tensor(np.array([[1.0]], np.float32))
    net = Net()

    with pytest.raises(ValueError):
        forward_and_gradfn(net, x, weights=None, grad_position=None)

    with pytest.raises(ValueError):
        forward_and_gradfn(net, x, weights=(), grad_position=None)

    _, grad_fn = forward_and_gradfn(net, x, weights=(net.w,), grad_position=None)
    dw = grad_fn()
    assert dw is not None

    _, grad_fn2 = forward_and_gradfn(net, x, weights=None, grad_position=0)
    dx = grad_fn2.compute_input_grad()
    assert dx is not None


def test_tuple_input_grad_position_all():
    """
    Feature: forward_and_gradfn with tuple input.
    Description: When input contains tuple of tensors and grad_position=-1,
                 gradients should be returned with the same tuple structure.
    Expectation: Gradients match theoretical values and preserve tuple structure.
    """
    # f(x, (a, b)) = x * a + x * b = x * (a + b)
    # df/dx = a + b, df/da = x, df/db = x
    def fn(x, tuple_input):
        a, b = tuple_input
        return x * a + x * b

    x = Tensor(np.array([2.0], np.float32))
    a = Tensor(np.array([3.0], np.float32))
    b = Tensor(np.array([4.0], np.float32))

    _, grad_fn = forward_and_gradfn(fn, x, (a, b), grad_position=-1)
    grads = grad_fn()

    # Verify structure: should return (dx, (da, db))
    assert isinstance(grads, tuple) and len(grads) == 2
    dx = grads[0]
    tuple_grads = grads[1]
    assert isinstance(tuple_grads, tuple) and len(tuple_grads) == 2

    # Verify values: df/dx = a + b = 7, df/da = x = 2, df/db = x = 2
    expected_dx = np.array([7.0], np.float32)
    expected_da = np.array([2.0], np.float32)
    expected_db = np.array([2.0], np.float32)

    assert np.allclose(dx.asnumpy(), expected_dx, atol=1e-6)
    assert np.allclose(tuple_grads[0].asnumpy(), expected_da, atol=1e-6)
    assert np.allclose(tuple_grads[1].asnumpy(), expected_db, atol=1e-6)


def test_compute_input_grad_raises_for_none_grad_position():
    """
    Feature: compute_input_grad enforces explicit grad_position.
    Description: When grad_fn is created with grad_position=None (weights provided so
                 _validate_grad_config passes), calling compute_input_grad must raise
                 ValueError because the dx/dw split requires an explicit grad_position
                 to identify which tensors are inputs.
    Expectation: ValueError is raised with a message mentioning grad_position.
    """
    class LinearCell(nn.Cell):
        def __init__(self):
            super().__init__()
            self.weight = Parameter(Tensor(np.array([2.0], np.float32)))

        def construct(self, x):
            return x * self.weight

    net = LinearCell()
    x = Tensor(np.array([1.0], np.float32))
    weights = tuple(net.trainable_params())
    _, grad_fn = forward_and_gradfn(net, x, weights=weights, grad_position=None)

    try:
        grad_fn.compute_input_grad()
        assert False, "Expected ValueError was not raised"
    except ValueError as exc:
        assert "grad_position" in str(exc), (
            f"ValueError message should mention grad_position, got: {exc}"
        )


def test_call_accumulates_weight_grads():
    """
    Feature: GradFunction.__call__ accumulates weight gradients into param._grad.
    Description: When grad_fn is called directly (not via compute_input_grad/compute_weight_grad
                 split), weight gradients must be written into param._grad so the optimizer
                 can read them — matching the behaviour of compute_weight_grad.
                 With grad_position=None, no input tensors are tracked (input_size=0),
                 so x._grad stays None regardless of the input accumulation logic in __call__.
    Expectation: param._grad is set after grad_fn() returns; result matches reference.
    """
    class LinearCell(nn.Cell):
        def __init__(self):
            super().__init__()
            self.weight = Parameter(Tensor(np.array([2.0], np.float32)))

        def construct(self, x):
            return x * self.weight

    net = LinearCell()
    # f(x) = x * w = 1 * 2 = 2;  df/dw = x = 1;  df/dx = w = 2
    x = Tensor(np.array([1.0], np.float32))
    weights = tuple(net.trainable_params())

    # grad_position=None: only weight grads computed; x is not tracked as input tensor
    _, grad_fn = forward_and_gradfn(net, x, weights=weights, grad_position=None)
    _ = grad_fn()  # caller discards return value (pipeline first-stage pattern)

    w = weights[0]
    assert w._grad is not None, "Weight grad must be accumulated into param._grad by __call__"  # pylint: disable=protected-access
    assert np.allclose(w._grad.asnumpy(), np.array([1.0], np.float32), atol=1e-5), (  # pylint: disable=protected-access
        f"Weight grad should be 1.0 (= input x), got {w._grad.asnumpy()}"  # pylint: disable=protected-access
    )
    # grad_position=None → input_size=0 → x is not collected as input tensor → x._grad stays None
    assert x._grad is None, (  # pylint: disable=protected-access
        f"x._grad should be None when grad_position=None (x not tracked); got {x._grad}"  # pylint: disable=protected-access
    )


def test_call_accumulates_input_grads():
    """
    Feature: GradFunction.__call__ accumulates input gradients into tensor._grad.
    Description: When grad_fn is called directly with a non-None grad_position,
                 input gradients must be written into tensor._grad in addition to
                 being returned — matching the behaviour of compute_input_grad.
    Expectation: x._grad is set after grad_fn() returns; value matches df/dx.
    """
    class LinearCell(nn.Cell):
        def __init__(self):
            super().__init__()
            self.weight = Parameter(Tensor(np.array([2.0], np.float32)))

        def construct(self, x):
            return x * self.weight

    net = LinearCell()
    # f(x) = x * w = 1 * 2 = 2;  df/dx = w = 2;  df/dw = x = 1
    x = Tensor(np.array([1.0], np.float32))
    weights = tuple(net.trainable_params())

    _, grad_fn = forward_and_gradfn(net, x, weights=weights, grad_position=0)
    _ = grad_fn()

    w = weights[0]
    assert w._grad is not None, "Weight grad must be accumulated into param._grad by __call__"  # pylint: disable=protected-access
    assert np.allclose(w._grad.asnumpy(), np.array([1.0], np.float32), atol=1e-5), (  # pylint: disable=protected-access
        f"Weight grad should be 1.0 (= input x), got {w._grad.asnumpy()}"  # pylint: disable=protected-access
    )
    assert x._grad is not None, "Input grad must be accumulated into x._grad when grad_position=0"  # pylint: disable=protected-access
    assert np.allclose(x._grad.asnumpy(), np.array([2.0], np.float32), atol=1e-5), (  # pylint: disable=protected-access
        f"Input grad should be 2.0 (= weight value), got {x._grad.asnumpy()}"  # pylint: disable=protected-access
    )


def test_frozen_weight_gets_no_gradient():
    """
    Feature: Gradient freezing — excluded weights receive no gradient.
    Description: Only parameters passed to forward_and_gradfn's `weights` argument
                 participate in gradient computation. A frozen parameter (excluded from
                 `weights`) must have _grad=None after backward, even when it is used
                 in the forward computation.
    Expectation: trainable weight accumulates correct gradient; frozen weight keeps _grad=None.
    """
    class TwoWeightNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.trainable = Parameter(Tensor(np.array([2.0], np.float32)), name="trainable")
            self.frozen = Parameter(Tensor(np.array([3.0], np.float32)), name="frozen")

        def construct(self, x):
            # f(x) = x * trainable + x * frozen
            return x * self.trainable + x * self.frozen

    net = TwoWeightNet()
    # f(1) = 1*2 + 1*3 = 5;  df/d(trainable) = 1;  df/d(frozen) = 1 (but frozen excluded)
    x = Tensor(np.array([1.0], np.float32))

    _, grad_fn = forward_and_gradfn(net, x, weights=(net.trainable,), grad_position=None)
    _ = grad_fn()

    # pylint: disable=protected-access
    assert net.trainable._grad is not None, (
        f"trainable weight must have gradient after backward, got {net.trainable._grad}"
    )
    assert np.allclose(net.trainable._grad.asnumpy(), np.array([1.0], np.float32), atol=1e-5), (
        f"trainable._grad should be 1.0 (= input x), got {net.trainable._grad.asnumpy()}"
    )
    assert net.frozen._grad is None, (
        f"frozen weight must have no gradient (excluded from weights), got {net.frozen._grad}"
    )
    # pylint: enable=protected-access


def test_requires_grad_false_weight_raises():
    """
    Feature: requires_grad=False weight passed directly to forward_and_gradfn weights.
    Description: run_backward cannot locate a gradient node for a parameter with
                 requires_grad=False. Passing such a parameter inside the weights
                 argument raises RuntimeError. The correct usage is to pass only
                 trainable_params() (which already filters by requires_grad=True).
    Expectation: RuntimeError is raised when grad_fn() is called.
    """
    class TwoWeightNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.trainable = Parameter(Tensor(np.array([2.0], np.float32)), name="trainable")
            self.frozen = Parameter(Tensor(np.array([3.0], np.float32)), name="frozen",
                                    requires_grad=False)

        def construct(self, x):
            return x * self.trainable + x * self.frozen

    net = TwoWeightNet()
    x = Tensor(np.array([1.0], np.float32))
    all_params = tuple(net.get_parameters())  # includes the requires_grad=False param

    _, grad_fn = forward_and_gradfn(net, x, weights=all_params, grad_position=None)
    with pytest.raises(RuntimeError):
        grad_fn()


def test_dict_input_grad_position_all():
    """
    Feature: forward_and_gradfn with dict input.
    Description: When input contains dict of tensors and grad_position=-1,
                 gradients should be returned with the same dict structure.
    Expectation: Gradients match theoretical values and preserve dict structure.
    """
    # f(x, {"w1": w1, "w2": w2}) = x * w1 + x^2 * w2
    # df/dx = w1 + 2*x*w2, df/dw1 = x, df/dw2 = x^2
    def fn(x, dict_input):
        w1 = dict_input["w1"]
        w2 = dict_input["w2"]
        return x * w1 + x * x * w2

    x = Tensor(np.array([3.0], np.float32))
    w1 = Tensor(np.array([2.0], np.float32))
    w2 = Tensor(np.array([1.0], np.float32))

    _, grad_fn = forward_and_gradfn(fn, x, {"w1": w1, "w2": w2}, grad_position=-1)
    grads = grad_fn()

    # Verify structure: should return (dx, {"w1": dw1, "w2": dw2})
    assert isinstance(grads, tuple) and len(grads) == 2
    dx = grads[0]
    dict_grads = grads[1]
    assert isinstance(dict_grads, dict)
    assert "w1" in dict_grads and "w2" in dict_grads

    # Verify values: df/dx = w1 + 2*x*w2 = 2 + 2*3*1 = 8
    # df/dw1 = x = 3, df/dw2 = x^2 = 9
    expected_dx = np.array([8.0], np.float32)
    expected_dw1 = np.array([3.0], np.float32)
    expected_dw2 = np.array([9.0], np.float32)

    assert np.allclose(dx[0].asnumpy(), expected_dx, atol=1e-6)
    assert np.allclose(dict_grads["w1"].asnumpy(), expected_dw1, atol=1e-6)
    assert np.allclose(dict_grads["w2"].asnumpy(), expected_dw2, atol=1e-6)


def test_kwargs_tensor_grad_position_all():
    """
    Feature: forward_and_gradfn with tensor in kwargs.
    Description: When kwargs contains tensor and grad_position=-1,
                 gradients for kwargs tensors should also be computed.
    Expectation: Gradients match theoretical values for both args and kwargs.
    """
    # f(x, scale=s) = x^2 * s
    # df/dx = 2*x*s, df/ds = x^2
    def fn(x, scale=None):
        return x * x * scale

    x = Tensor(np.array([4.0], np.float32))
    scale = Tensor(np.array([0.5], np.float32))

    _, grad_fn = forward_and_gradfn(fn, x, grad_position=-1, scale=scale)
    grads = grad_fn()

    # Verify structure: should return (dx,), (dscale,) or similar
    # Based on the implementation, it returns (args_grads, kwargs_grads, weights_grads)
    # When no weights, it returns (args_grads, kwargs_grads)
    assert isinstance(grads, tuple)

    # First element is dx (single tensor since only one arg)
    dx = grads[0]
    # Second element is kwargs grads (dict with 'scale')
    kwargs_grads = grads[1]

    # Verify values: df/dx = 2*x*s = 2*4*0.5 = 4
    # df/ds = x^2 = 16
    expected_dx = np.array([4.0], np.float32)
    expected_ds = np.array([16.0], np.float32)

    assert np.allclose(dx[0].asnumpy(), expected_dx, atol=1e-6)
    assert np.allclose(kwargs_grads["scale"].asnumpy(), expected_ds, atol=1e-6)


def test_tuple_input_compute_input_grad():
    """
    Feature: compute_input_grad with tuple input.
    Description: compute_input_grad should return gradients preserving tuple structure.
    Expectation: Gradients match theoretical values with correct structure.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.w = Parameter(Tensor(np.array([2.0], np.float32)), name="w")

        def construct(self, x, tuple_input):
            # f = w * (x + a + b)
            a, b = tuple_input
            return self.w * (x + a + b)

    x = Tensor(np.array([1.0], np.float32))
    a = Tensor(np.array([2.0], np.float32))
    b = Tensor(np.array([3.0], np.float32))
    net = Net()

    _, grad_fn = forward_and_gradfn(net, x, (a, b), weights=(net.w,), grad_position=-1)
    dx_grads = grad_fn.compute_input_grad()
    dw = grad_fn.compute_weight_grad()

    # Verify structure: dx_grads should be (dx, (da, db))
    assert isinstance(dx_grads, tuple) and len(dx_grads) == 2
    dx = dx_grads[0]
    tuple_grads = dx_grads[1]
    assert isinstance(tuple_grads, tuple) and len(tuple_grads) == 2

    # Verify values: df/dx = w = 2, df/da = w = 2, df/db = w = 2
    # df/dw = x + a + b = 6
    expected_grad = np.array([2.0], np.float32)
    expected_dw = np.array([6.0], np.float32)

    assert np.allclose(dx.asnumpy(), expected_grad, atol=1e-6)
    assert np.allclose(tuple_grads[0].asnumpy(), expected_grad, atol=1e-6)
    assert np.allclose(tuple_grads[1].asnumpy(), expected_grad, atol=1e-6)
    assert np.allclose(dw[0].asnumpy(), expected_dw, atol=1e-6)


def test_dict_output_with_sens():
    """
    Feature: forward_and_gradfn with dict output.
    Description: When output is a dict of tensors, verify gradient accuracy
                 with and without sens parameter.
    Expectation: Gradients match theoretical values for both sens=None and sens=dict cases.
    """
    class Net(nn.Cell):
        def construct(self, x):
            out1 = x * x
            out2 = x * x * x
            return {"out1": out1, "out2": out2}

    x = Tensor(np.array([2.0], np.float32))
    net = Net()

    _, grad_fn1 = forward_and_gradfn(net, x, grad_position=0)
    dx1 = grad_fn1.compute_input_grad(sens=None)

    expected_dx1 = np.array([16.0], np.float32)
    assert np.allclose(dx1.asnumpy(), expected_dx1, atol=1e-6)

    sens_dict = {
        "out1": Tensor(np.array([1.0], np.float32)),
        "out2": Tensor(np.array([2.0], np.float32))
    }
    _, grad_fn2 = forward_and_gradfn(net, x, grad_position=0)
    dx2 = grad_fn2.compute_input_grad(sens=sens_dict)

    expected_dx2 = np.array([28.0], np.float32)
    assert np.allclose(dx2.asnumpy(), expected_dx2, atol=1e-6)

    _, grad_fn3 = forward_and_gradfn(net, x, grad_position=0)
    grads3 = grad_fn3(sens=None)
    assert np.allclose(grads3.asnumpy(), expected_dx1, atol=1e-6)

    _, grad_fn4 = forward_and_gradfn(net, x, grad_position=0)
    grads4 = grad_fn4(sens=sens_dict)
    assert np.allclose(grads4.asnumpy(), expected_dx2, atol=1e-6)


def test_dict_output_with_weights():
    """
    Feature: forward_and_gradfn with dict output and weights.
    Description: When output is dict and weights are present, verify dx/dw split accuracy.
    Expectation: Both input and weight gradients match theoretical values.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.w = Parameter(Tensor(np.array([3.0], np.float32)), name="w")

        def construct(self, x):
            out1 = self.w * x * x
            out2 = self.w * x * x * x
            return {"out1": out1, "out2": out2}

    x = Tensor(np.array([2.0], np.float32))
    net = Net()

    _, grad_fn = forward_and_gradfn(net, x, weights=(net.w,), grad_position=0)
    dx = grad_fn.compute_input_grad(sens=None)
    dw = grad_fn.compute_weight_grad()

    expected_dx = np.array([48.0], np.float32)
    expected_dw = np.array([12.0], np.float32)

    assert np.allclose(dx.asnumpy(), expected_dx, atol=1e-6)
    assert np.allclose(dw[0].asnumpy(), expected_dw, atol=1e-6)

    sens_dict = {
        "out1": Tensor(np.array([0.5], np.float32)),
        "out2": Tensor(np.array([1.5], np.float32))
    }
    _, grad_fn2 = forward_and_gradfn(net, x, weights=(net.w,), grad_position=0)
    dx2 = grad_fn2.compute_input_grad(sens=sens_dict)
    dw2 = grad_fn2.compute_weight_grad()

    expected_dx2 = np.array([60.0], np.float32)
    expected_dw2 = np.array([14.0], np.float32)

    assert np.allclose(dx2.asnumpy(), expected_dx2, atol=1e-6)
    assert np.allclose(dw2[0].asnumpy(), expected_dw2, atol=1e-6)


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_parallel_grad_cases_suit():
    """
    Feature: Run all non-exception cases together.
    Description: Aggregate suite for forward_and_gradfn scenarios.
    Expectation: All cases pass.
    """
    test_forward_output_consistency()
    test_dx_only_inputs()
    test_call_returns_dx_and_dw_together()
    test_dx_dw_split_pipeline_consistency()
    test_weight_grad_keep_graph_allows_reuse()
    test_multi_output_intermediate_slot_grad()
    test_shared_intermediate_weight_group_merge_dw()
    test_complex_multilayer_network_dx_dw_accuracy()
    test_weight_subgraph_meets_input_later_dx_dw_accuracy()
    test_shared_weight_used_multiple_places_dx_dw_accuracy()
    test_tuple_input_grad_position_all()
    test_compute_input_grad_raises_for_none_grad_position()
    test_call_accumulates_weight_grads()
    test_call_accumulates_input_grads()
    test_frozen_weight_gets_no_gradient()
    test_requires_grad_false_weight_raises()
    test_dict_input_grad_position_all()
    test_kwargs_tensor_grad_position_all()
    test_tuple_input_compute_input_grad()
    test_dict_output_with_sens()
    test_dict_output_with_weights()

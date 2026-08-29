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
"""forward_and_gradfn grad storage impl (lazy-loaded by test_forward_and_gradfn_grad_storage)."""
# pylint: disable=wrong-import-position
import os

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

import numpy as np
import mindspore as ms
from mindspore import Tensor, Parameter, nn, ops

from hyper_parallel.platform.mindspore.autograd_compat import enable_mindspore_backward_compat
from hyper_parallel.platform.mindspore.pipeline_parallel.backward import forward_and_gradfn

def _assert_param_grad_is_none(param):
    assert param.grad is None
    assert getattr(param, "_grad", None) is None


def _assert_param_grad_allclose(param, expected):
    assert param.grad is not None
    np.testing.assert_allclose(param.grad.asnumpy(), np.array(expected, dtype=np.float32))


class Net(nn.Cell):
    """Minimal network for checking returned grads vs stored grads."""

    def __init__(self):
        super().__init__()
        self.w = Parameter(Tensor(np.array([[2.0]], np.float32)), name="w")

    def construct(self, x):
        return ops.matmul(x, self.w)


def test_forward_and_gradfn_populates_parameter_grad():
    """
    Feature: forward_and_gradfn grad storage behavior.
    Description: forward_and_gradfn should return gradients and write weight gradients into Parameter.grad.
    Expectation: Returned dx/dw are correct and Parameter.grad is populated.
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    enable_mindspore_backward_compat()

    x = Tensor(np.array([[3.0]], np.float32))

    net = Net()
    _assert_param_grad_is_none(net.w)

    _, grad_fn = forward_and_gradfn(net, x, weights=(net.w,), grad_position=0)
    dx, dw = grad_fn()

    np.testing.assert_allclose(dx[0].asnumpy(), np.array([[2.0]], dtype=np.float32))
    np.testing.assert_allclose(dw[0].asnumpy(), np.array([[3.0]], dtype=np.float32))
    _assert_param_grad_allclose(net.w, [[3.0]])

    net = Net()
    _assert_param_grad_is_none(net.w)

    _, grad_fn = forward_and_gradfn(net, x, weights=(net.w,), grad_position=0)
    dx = grad_fn.compute_input_grad()
    dw = grad_fn.compute_weight_grad()

    np.testing.assert_allclose(dx.asnumpy(), np.array([[2.0]], dtype=np.float32))
    np.testing.assert_allclose(dw[0].asnumpy(), np.array([[3.0]], dtype=np.float32))
    _assert_param_grad_allclose(net.w, [[3.0]])


def test_forward_and_gradfn_parameter_grad_accumulates():
    """
    Feature: forward_and_gradfn grad accumulation.
    Description: Repeated forward_and_gradfn backward passes should accumulate into Parameter.grad by default.
    Expectation: Parameter.grad doubles after two identical backward passes.
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    enable_mindspore_backward_compat()

    x = Tensor(np.array([[3.0]], np.float32))

    net = Net()
    _assert_param_grad_is_none(net.w)

    _, grad_fn = forward_and_gradfn(net, x, weights=(net.w,), grad_position=0)
    _ = grad_fn()
    _assert_param_grad_allclose(net.w, [[3.0]])

    _, grad_fn = forward_and_gradfn(net, x, weights=(net.w,), grad_position=0)
    _ = grad_fn()
    _assert_param_grad_allclose(net.w, [[6.0]])

    net = Net()
    _assert_param_grad_is_none(net.w)

    _, grad_fn = forward_and_gradfn(net, x, weights=(net.w,), grad_position=0)
    _ = grad_fn.compute_input_grad()
    _ = grad_fn.compute_weight_grad()
    _assert_param_grad_allclose(net.w, [[3.0]])

    _, grad_fn = forward_and_gradfn(net, x, weights=(net.w,), grad_position=0)
    _ = grad_fn.compute_input_grad()
    _ = grad_fn.compute_weight_grad()
    _assert_param_grad_allclose(net.w, [[6.0]])


def test_accumulate_grad_falls_back_for_shared_non_leaf_weight():
    """
    Feature: Pipeline parameter gradient accumulation.
    Description: Distinct forward graphs share a transformed non-leaf weight, as in TP pipeline stages.
    Expectation: Input gradients stay per-graph while the transformed weight gradient accumulates.
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    enable_mindspore_backward_compat()

    base_weight = Parameter(Tensor(np.array([[2.0]], np.float32)), name="base_weight")
    weight = ops.mul(base_weight, Tensor(np.array([[1.0]], np.float32)))
    assert not weight.is_leaf
    _assert_param_grad_is_none(weight)

    def _fn(x: Tensor) -> Tensor:
        return ops.matmul(x, weight)

    for expected_weight_grad in ([[3.0]], [[6.0]]):
        x = Tensor(np.array([[3.0]], np.float32))
        _, grad_fn = forward_and_gradfn(_fn, x, weights=(weight,), grad_position=0)
        input_grad = grad_fn.accumulate_grad()

        np.testing.assert_allclose(input_grad.asnumpy(), np.array([[2.0]], dtype=np.float32))
        _assert_param_grad_allclose(weight, expected_weight_grad)

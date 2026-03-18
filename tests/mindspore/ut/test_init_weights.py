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
"""Tests for init_empty_weights / init_on_device (MindSpore)."""

import os
os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

# pylint: disable=C0413
import pytest
import numpy as np
import mindspore as ms
from mindspore import nn, mint
from mindspore.common.initializer import initializer
from hyper_parallel.core.dtensor.init_weights import init_empty_weights, init_on_device

HAS_NPU = ms.device_context.ascend.is_available()


class Net(nn.Cell):
    """A minimal model with hand-crafted parameter, buffer and standard layer."""
    def __init__(self):
        super().__init__()
        self.weight = ms.Parameter(initializer("normal", [8, 4], ms.float32))
        self.register_buffer("scale", mint.randn(4))
        self.linear = mint.nn.Linear(4, 2)

    def construct(self, x):
        return self.linear(x @ self.weight) * self.scale[:2]


def _build_ref_state():
    """Build a reference model and return (model, state_dict)."""
    ms.manual_seed(0)
    ref = Net()
    return ref, {
        k: ms.Parameter(v.clone(), requires_grad=False) if isinstance(v, ms.Tensor) else v.clone()
        for k, v in ref.state_dict().items()
    }


def _assert_device(model, device):
    for p in model.get_parameters():
        assert p.device == device


# ---------------------------------------------------------------------------
# init_empty_weights
# ---------------------------------------------------------------------------

class TestInitEmptyWeights:
    def test_shapes_and_requires_grad(self):
        with init_empty_weights():
            model = Net()
        assert model.weight.shape == (8, 4)
        assert model.linear.weight.shape == (2, 4)
        assert model.linear.bias.shape == (2,)
        assert model.weight.requires_grad
        assert model.linear.weight.requires_grad


# ---------------------------------------------------------------------------
# init_on_device — CPU
# ---------------------------------------------------------------------------

class TestInitOnDeviceCPU:
    def test_params_on_cpu(self):
        with init_on_device("CPU"):
            model = Net()
        _assert_device(model, "CPU")

    def test_include_buffers(self):
        with init_on_device("CPU", include_buffers=True):
            model = Net()
        _assert_device(model, "CPU")


# ---------------------------------------------------------------------------
# init_on_device — NPU
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_NPU, reason="NPU not available")
class TestInitOnDeviceNPU:
    def test_params_on_npu(self):
        with init_on_device("Ascend"):
            model = Net()
        _assert_device(model, "Ascend:0")

    def test_include_buffers(self):
        with init_on_device("Ascend", include_buffers=True):
            model = Net()
        _assert_device(model, "Ascend:0")


# ---------------------------------------------------------------------------
# Load weights & training after init_empty_weights
# ---------------------------------------------------------------------------

class TestLoadAndTrain:
    """Tests for loading weights and training after init_empty_weights."""
    def test_load_param_into_net(self):
        """init_empty_weights → load → weights must match."""
        _, state = _build_ref_state()

        with init_empty_weights():
            model = Net()
        ms.load_param_into_net(model, state)

        for p in model.get_parameters():
            if p.name in state:
                np.testing.assert_array_almost_equal(
                    p.asnumpy(), state[p.name].asnumpy()
                )

    def test_forward_after_load(self):
        """Loaded model must produce the same output as the reference."""
        ref, state = _build_ref_state()

        with init_empty_weights():
            model = Net()
        ms.load_param_into_net(model, state)
        model.scale.assign_value(state["scale"].data)

        x = mint.randn(2, 8)
        np.testing.assert_array_almost_equal(
            model(x).asnumpy(), ref(x).asnumpy(), decimal=5
        )

    def test_backward_after_load(self):
        """Gradients must be computed correctly after loading weights."""
        _, state = _build_ref_state()

        with init_empty_weights():
            model = Net()
        ms.load_param_into_net(model, state)

        def forward_fn(x):
            return model(x).sum()
        x = mint.randn(2, 8)
        _, grads = ms.value_and_grad(forward_fn, grad_position=None,
                                     weights=model.trainable_params())(x)
        for g in grads:
            assert g is not None
            assert g.shape != (0,)

    def test_train_step(self):
        """One full SGD step must update parameters without error."""
        _, state = _build_ref_state()

        with init_empty_weights():
            model = Net()
        ms.load_param_into_net(model, state)

        optimizer = mint.optim.SGD(model.trainable_params(), lr=0.01)

        def forward_fn(x):
            return model(x).sum()

        x = mint.randn(2, 8)
        before = {p.name: p.asnumpy().copy() for p in model.trainable_params()}
        _, grads = ms.value_and_grad(forward_fn, grad_position=None,
                                     weights=model.trainable_params())(x)
        optimizer(grads)

        for p in model.trainable_params():
            assert not np.array_equal(p.asnumpy(), before[p.name]), \
                f"{p.name} should have been updated by optimizer"

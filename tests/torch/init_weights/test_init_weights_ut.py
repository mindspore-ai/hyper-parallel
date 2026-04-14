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
"""Unit tests for init_on_device (PyTorch)."""

import os
os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

# pylint: disable=C0413
import torch
from torch import nn

from hyper_parallel.platform.torch.init_weights import init_on_device


class _CustomParameter(nn.Parameter):
    """Parameter subclass with a narrow __new__ signature (like accelerate)."""

    def __new__(cls, data, requires_grad=True):
        # pylint: disable=E1123  # torch.Tensor.__new__ accepts requires_grad at runtime
        return super().__new__(cls, data, requires_grad=requires_grad)

    def detach(self):
        return _CustomParameter(self.data, requires_grad=False)


class _NetWithCustomParam(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = _CustomParameter(torch.tensor([1.0]), requires_grad=True)


class _SimpleNet(nn.Module):
    def __init__(self, hidden=4):
        super().__init__()
        self.linear = nn.Linear(hidden, hidden)
        self.register_buffer("scale", torch.ones(hidden))


class TestInitOnDeviceBasic:
    """Basic placement semantics of init_on_device."""

    def test_params_moved_to_meta(self):
        """Parameters should be on meta after init_on_device(meta)."""
        with init_on_device(torch.device("meta")):
            model = _SimpleNet()
        assert model.linear.weight.is_meta, (
            f"linear.weight expected is_meta=True, got device={model.linear.weight.device}"
        )
        assert model.linear.bias.is_meta, (
            f"linear.bias expected is_meta=True, got device={model.linear.bias.device}"
        )

    def test_buffers_moved_when_included(self):
        """Buffers should move to target device only when include_buffers=True."""
        with init_on_device(torch.device("meta"), include_buffers=True):
            model = _SimpleNet()
        assert model.scale.is_meta, (
            f"buffer 'scale' expected is_meta=True, got device={model.scale.device}"
        )

    def test_buffers_untouched_by_default(self):
        """Buffers should stay on the default device when include_buffers=False."""
        with init_on_device(torch.device("meta")):
            model = _SimpleNet()
        assert not model.scale.is_meta, (
            f"buffer 'scale' should stay on CPU, got device={model.scale.device}"
        )

    def test_same_device_no_reconstruction(self):
        """If param is already on target device, keep the exact same object."""
        cpu = torch.device("cpu")
        with init_on_device(cpu):
            model = _SimpleNet()
        assert model.linear.weight.device == cpu, (
            f"expected device=cpu, got {model.linear.weight.device}"
        )


class TestDictPollutionRegression:
    """Regression tests: the old implementation wrote `requires_grad` into
    `orig_param.__dict__`, which (1) polluted the user's original parameter
    dict and (2) desynced Python-level `__dict__['requires_grad']` from the
    C++-level `requires_grad` descriptor after later `requires_grad_()` calls.
    """

    def test_new_param_dict_not_polluted(self):
        """The new parameter's __dict__ must not contain 'requires_grad'."""
        with init_on_device(torch.device("meta")):
            model = _NetWithCustomParam()
        assert "requires_grad" not in model.p.__dict__, (
            f"new param __dict__ should not carry requires_grad, got {model.p.__dict__}"
        )

    def test_requires_grad_descriptor_consistent_after_toggle(self):
        """After requires_grad_(False), __dict__ must not hold a stale True."""
        with init_on_device(torch.device("meta")):
            model = _NetWithCustomParam()
        model.p.requires_grad_(False)
        assert model.p.requires_grad is False, (
            f"C++-level requires_grad expected False, got {model.p.requires_grad}"
        )
        assert "requires_grad" not in model.p.__dict__, (
            f"__dict__ must not shadow C++ requires_grad, got {model.p.__dict__}"
        )

    def test_preserves_requires_grad_true(self):
        """requires_grad=True on the source must carry to the new param."""
        with init_on_device(torch.device("meta")):
            model = _NetWithCustomParam()
        assert model.p.requires_grad is True, (
            f"requires_grad expected True, got {model.p.requires_grad}"
        )

    def test_preserves_requires_grad_false(self):
        """requires_grad=False on the source must carry to the new param."""

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.p = _CustomParameter(torch.tensor([1.0]), requires_grad=False)

        with init_on_device(torch.device("meta")):
            model = Net()
        assert model.p.requires_grad is False, (
            f"requires_grad expected False, got {model.p.requires_grad}"
        )

    def test_custom_subclass_type_preserved(self):
        """The registered parameter must remain the user's subclass type."""
        with init_on_device(torch.device("meta")):
            model = _NetWithCustomParam()
        assert isinstance(model.p, _CustomParameter), (
            f"expected _CustomParameter, got {type(model.p).__name__}"
        )

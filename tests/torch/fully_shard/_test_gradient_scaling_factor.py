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
"""Distributed test cases for HSDPModule.set_gradient_scaling_factor."""
# pylint: disable=W0611,C0413,C0412
import os

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
import numpy as np
import pytest
import torch
import torch_npu
from hyper_parallel import DTensor, init_device_mesh, SkipDTensorDispatch
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from tests.torch.common_net import SimpleModel
from tests.torch.utils import init_dist


def _build_model_and_input(comm_fusion: bool):
    """Build a freshly-initialized fully_shard model on a 1-D mesh."""
    mesh = init_device_mesh(device_type="npu", mesh_shape=(4,), mesh_dim_names=("dp",))
    model = SimpleModel().npu()
    model = fully_shard(
        model,
        mesh=mesh,
        reshard_after_forward=True,
        mp_policy=MixedPrecisionPolicy(),
        comm_fusion=comm_fusion,
    )
    # Use sum reduce so every rank sees the same gradient regardless of repeat.
    model.set_reduce_op_type("sum")
    data = torch.ones(8, 8, device="npu")
    return model, data


def _run_one_step_grad(model, data):
    """Run one forward/backward and return the local-shard gradient as numpy."""
    with SkipDTensorDispatch():
        loss = model(data)
        loss.backward(torch.ones_like(loss))
        grad = model.weight.grad
        assert isinstance(grad, DTensor), f"Expected DTensor grad, got {type(grad)}"
        local_grad = grad.to_local().detach().clone().cpu().numpy()
    model.reset_iter_state()
    return local_grad


def _assert_scaled(baseline, scaled, factor):
    """Compare scaled grad to baseline * factor."""
    expected = baseline * float(factor)
    assert np.allclose(scaled, expected, rtol=1e-5, atol=1e-5), (
        f"gradient_scaling_factor={factor} did not scale grad as expected: "
        f"baseline={baseline}, scaled={scaled}, expected={expected}"
    )


def _get_gradient_scaling_factor(model):
    """Read the factor back from whichever layer the setter wrote it to.

    Mirrors HSDPState.set_gradient_scaling_factor: comm_fusion stores it on the
    param_group, otherwise per managed param. Kept local to the test since the
    factor is write-only on the public API.
    """
    state = model.hsdp_scheduler.hsdp_state
    if state.param_group is not None:
        return state.param_group.gradient_scaling_factor
    for hsdp_param in state.hsdp_params:
        return hsdp_param.gradient_scaling_factor
    return None


def _run_scaling_check(comm_fusion: bool):
    """Run one baseline step and three scaled steps, then validate."""
    init_dist()

    # Baseline: factor unset (None).
    model, data = _build_model_and_input(comm_fusion)
    assert _get_gradient_scaling_factor(model) is None, (
        "gradient_scaling_factor should default to None, got "
        f"{_get_gradient_scaling_factor(model)}"
    )
    baseline = _run_one_step_grad(model, data)

    # Float factor.
    model, data = _build_model_and_input(comm_fusion)
    model.set_gradient_scaling_factor(2.0)
    assert _get_gradient_scaling_factor(model) == 2.0, (
        "set_gradient_scaling_factor(2.0) did not propagate, got "
        f"{_get_gradient_scaling_factor(model)}"
    )
    scaled_float = _run_one_step_grad(model, data)
    _assert_scaled(baseline, scaled_float, 2.0)

    # Int factor.
    model, data = _build_model_and_input(comm_fusion)
    model.set_gradient_scaling_factor(3)
    scaled_int = _run_one_step_grad(model, data)
    _assert_scaled(baseline, scaled_int, 3)

    # Tensor factor (1-element). Place on NPU so the in-place mul_ stays on device.
    model, data = _build_model_and_input(comm_fusion)
    factor_tensor = torch.tensor(0.5, device="npu", dtype=torch.float32)
    model.set_gradient_scaling_factor(factor_tensor)
    scaled_tensor = _run_one_step_grad(model, data)
    _assert_scaled(baseline, scaled_tensor, 0.5)

    # Setting back to None disables scaling and grad returns to baseline.
    model, data = _build_model_and_input(comm_fusion)
    model.set_gradient_scaling_factor(2.0)
    model.set_gradient_scaling_factor(None)
    restored = _run_one_step_grad(model, data)
    assert np.allclose(restored, baseline, rtol=1e-5, atol=1e-5), (
        f"Resetting factor to None should restore baseline grad, "
        f"baseline={baseline}, restored={restored}"
    )


@pytest.mark.parametrize(
    "comm_fusion",
    [False, True],
    ids=["per_param", "param_group"],
)
def test_gradient_scaling_factor(comm_fusion):
    """
    Feature: HSDPModule.set_gradient_scaling_factor
    Description: Verify the factor on per-parameter and parameter-group paths.
    Expectation: Scaled grad equals baseline * factor for float, int and tensor.
    """
    _run_scaling_check(comm_fusion)


def test_gradient_scaling_factor_validation():
    """
    Feature: HSDPModule.set_gradient_scaling_factor input validation
    Description: Reject bool, str, multi-element tensors and other invalid types.
    Expectation: ValueError raised; valid types accepted; None resets the factor.
    """
    init_dist()
    model, _ = _build_model_and_input(comm_fusion=False)

    invalid_inputs = [
        True,
        "0.5",
        [1.0],
        torch.tensor([1.0, 2.0], device="npu"),
    ]
    for bad in invalid_inputs:
        try:
            model.set_gradient_scaling_factor(bad)
        except ValueError:
            continue
        raise AssertionError(
            f"set_gradient_scaling_factor({bad!r}) should have raised ValueError"
        )

    # Valid types must not raise.
    for good in (None, 1.0, 2, torch.tensor(0.25, device="npu")):
        model.set_gradient_scaling_factor(good)
    model.reset_iter_state()

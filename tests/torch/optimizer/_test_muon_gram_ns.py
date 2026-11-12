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
"""CUDA workers for optional Gram NS integration, resume and distributed parity."""

import copy
import os

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor

from hyper_parallel.components.functional import make_gram_newton_schulz
from hyper_parallel.core.optimizer import get_hyper_optimizer
from hyper_parallel.core.optimizer.dtensor_compat import detect_dtensor_backend
from hyper_parallel.core.optimizer.muon import _NS_ASYM5_COEFFS, zeropower_via_newtonschulz5

_EPSILON = 1e-10


def _device() -> torch.device:
    """Select the scheduler-visible logical device for this worker."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device)[0] not in (9, 10):
        pytest.skip("requires Hopper or Blackwell")
    pytest.importorskip("gram_newton_schulz")
    return device


def _callback():
    """Use the same coefficient/epsilon configuration as the native Muon baseline."""
    return make_gram_newton_schulz(_NS_ASYM5_COEFFS, epsilon=_EPSILON, use_kernels=None)


def _optimizer(params):
    """Build the same Gram-backed optimizer through the factory for plain and sharded parameters."""
    model = torch.nn.ParameterList(params)
    return get_hyper_optimizer(
        model, list(model.parameters()), [], muon_kwargs={"lr": 0.01, "momentum": 0.9},
        muon_ns_backend="gram_quack",
    ).optimizer


def test_gram_adapter_and_numerics() -> None:
    """Check adapter transparency and record differences against both NS references."""
    device = _device()
    # The dependency is optional and is loaded only inside a GPU worker.
    package = pytest.importorskip("gram_newton_schulz")
    callback = _callback()
    upstream = package.GramNewtonSchulz(
        ns_epsilon=_EPSILON, ns_coefficients=list(_NS_ASYM5_COEFFS), ns_use_kernels=True, compile_kwargs=None,
    )
    standard = package.StandardNewtonSchulz(
        ns_epsilon=_EPSILON, ns_coefficients=list(_NS_ASYM5_COEFFS), ns_use_kernels=True, compile_kwargs=None,
    )
    torch.manual_seed(2026)
    for shape in ((64, 128), (256, 1024), (1024, 256), (256, 256), (3, 256, 512)):
        inputs = torch.randn(shape, device=device, dtype=torch.bfloat16)
        original = inputs.clone()
        actual = callback(inputs, steps=5)
        torch.testing.assert_close(actual, upstream(inputs), rtol=0, atol=0)
        torch.testing.assert_close(inputs, original, rtol=0, atol=0)
        assert actual.shape == inputs.shape, f"expected={inputs.shape}, got={actual.shape}"
        assert actual.dtype == inputs.dtype, f"expected={inputs.dtype}, got={actual.dtype}"
        assert actual.device == inputs.device, f"expected={inputs.device}, got={actual.device}"
        assert torch.isfinite(actual).all(), f"expected finite result for shape={shape}, got nonfinite values"
        for name, reference in (
            ("native_bf16", zeropower_via_newtonschulz5(inputs, steps=5)),
            ("upstream_standard_fp16", standard(inputs)),
        ):
            relative = ((actual.float() - reference.float()).norm() / reference.float().norm()).item()
            print({"shape": shape, "reference": name, "relative_l2": relative}, flush=True)
    for inputs in (
        torch.zeros(256, 512, device=device, dtype=torch.bfloat16),
        (torch.randn(256, 1, device=device) @ torch.randn(1, 512, device=device)).bfloat16(),
    ):
        actual = callback(inputs, steps=5)
        assert torch.isfinite(actual).all(), f"expected finite result, got nonfinite values for {inputs.shape}"
    torch.testing.assert_close(callback(torch.zeros_like(inputs), steps=5), torch.zeros_like(inputs))


def test_gram_muon_and_resume() -> None:
    """The adapter must match direct upstream integration and resume exactly."""
    device = _device()
    torch.manual_seed(17)
    initial = torch.randn(256, 512, device=device) * 0.02
    parameters = [torch.nn.Parameter(initial.clone()) for _ in range(2)]
    plain = _optimizer([parameters[0]])
    direct = _optimizer([parameters[1]])
    package = pytest.importorskip("gram_newton_schulz")
    upstream = package.GramNewtonSchulz(
        ns_epsilon=_EPSILON, ns_coefficients=list(_NS_ASYM5_COEFFS), ns_use_kernels=True, compile_kwargs=None,
    )
    direct.zeropower_fn = lambda inputs, steps: upstream(inputs)
    for _ in range(4):
        gradient = torch.randn_like(initial) * 0.01
        for parameter, optimizer in zip(parameters, (plain, direct)):
            parameter.grad = gradient.clone()
            optimizer.step()
        torch.testing.assert_close(parameters[0], parameters[1], rtol=0, atol=0)
    checkpoint = copy.deepcopy(plain.state_dict())
    resumed_parameter = torch.nn.Parameter(parameters[0].detach().clone())
    resumed = _optimizer([resumed_parameter])
    resumed.load_state_dict(checkpoint)
    for _ in range(3):
        gradient = torch.randn_like(initial) * 0.01
        parameters[0].grad = gradient.clone()
        resumed_parameter.grad = gradient.clone()
        plain.step()
        resumed.step()
        torch.testing.assert_close(parameters[0], resumed_parameter, rtol=0, atol=0)


def test_gram_muon_distributed() -> None:
    """Shard either matrix axis or replicate the matrix and compare complete updates."""
    device = _device()
    dist.init_process_group("nccl")
    try:
        mesh = init_device_mesh("cuda", (dist.get_world_size(),))
        for placement in (Shard(0), Shard(1), Replicate()):
            torch.manual_seed(23)
            initial = torch.randn(256, 512, device=device) * 0.02
            reference = torch.nn.Parameter(initial.clone())
            sharded = torch.nn.Parameter(distribute_tensor(initial.clone(), mesh, [placement]))
            detect_dtensor_backend([], [{"params": [sharded]}])
            plain_optimizer = _optimizer([reference])
            sharded_optimizer = _optimizer([sharded])
            for _ in range(4):
                gradient = torch.randn_like(initial) * 0.01
                reference.grad = gradient.clone()
                sharded.grad = distribute_tensor(gradient.clone(), mesh, [placement])
                plain_optimizer.step()
                sharded_optimizer.step()
                torch.testing.assert_close(sharded.full_tensor(), reference, rtol=1e-3, atol=1e-5)
    finally:
        dist.destroy_process_group()

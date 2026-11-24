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
"""Real NPU checkpoint regressions for process-wide operator selection."""

import importlib
import importlib.util
import os

import pytest
import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from hyper_parallel.components.ops import ops, swiglu
from hyper_parallel.core.activation_memory.api import checkpoint


@pytest.fixture(name="npu_device")
def _npu_device():
    """Select a real NPU without changing the checkpoint default device type."""
    if importlib.util.find_spec("torch_npu") is None:
        pytest.skip("torch_npu is unavailable")
    importlib.import_module("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU hardware is unavailable")
    device = torch.device("npu", int(os.environ.get("LOCAL_RANK", "0")))
    torch.npu.set_device(device)
    return device


@pytest.fixture(autouse=True)
def _restore_backends():
    """Restore process-wide selections after all forward and backward work."""
    previous = ops.get_backends()
    try:
        yield
    finally:
        ops.set_backends(previous)


@pytest.mark.parametrize("backend", ["native", "torch_npu"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("checkpoint_fn", [checkpoint, torch_checkpoint], ids=["component", "torch"])
@pytest.mark.parametrize("reentrant", [True, False], ids=["reentrant", "non_reentrant"])
def test_checkpoint_backend_selection(npu_device, backend, dtype, checkpoint_fn, reentrant) -> None:
    """Checkpointing must preserve actual NPU selection, outputs and input gradients."""
    ops.set_backends({"swiglu": backend})
    values = torch.linspace(-3, 3, 128).reshape(4, 32).to(device=npu_device, dtype=dtype)
    reference_input = values.detach().clone().requires_grad_()
    expected = swiglu(reference_input)
    upstream = torch.linspace(-1, 1, expected.numel()).reshape(expected.shape).to(device=npu_device, dtype=dtype)
    expected.backward(upstream)

    actual_input = values.detach().clone().requires_grad_()
    actual = checkpoint_fn(swiglu, actual_input, use_reentrant=reentrant)
    actual.backward(upstream)
    torch.npu.synchronize()

    assert actual.device == npu_device, f"Expected output device {npu_device}, got {actual.device}"
    assert actual_input.grad is not None, f"Expected input gradient on {npu_device}, got {actual_input.grad}"
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_input.grad, reference_input.grad, rtol=0, atol=0)
    expected_usage = (("swiglu", "standard", backend),)
    assert ops.used_backends() == expected_usage, f"Expected {expected_usage}, got {ops.used_backends()}"

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
"""A5 Dense training through the scheme-selected Linear Family."""

import copy
import json
from typing import Any

import pytest
import torch
from torch import nn

from hyper_parallel.components.quantization.functional.linear_strategy_factory import build_linear_strategy
from hyper_parallel.components.quantization.modules.linear import LowPrecisionLinear
from hyper_parallel.components.quantization.ops.npu_mxfp8 import MXFP8NpuOps
from hyper_parallel.components.quantization.quantizers.mxfp8 import MXFP8Quantizer
from tests.common.linear_family import linear_config
from tests.common.qat_family import install_family


class CountingDenseOps(MXFP8NpuOps):
    """Observe actual MXFP8 operands while preserving device computation."""

    def __init__(self, backend: Any) -> None:
        """Initialize the normal backend and a call counter."""
        super().__init__(backend)
        self.calls = 0

    def quant_matmul(self, x1: torch.Tensor, x2: torch.Tensor, scale: torch.Tensor,
                     **kwargs: Any) -> torch.Tensor:
        """Count genuine float8 matrix multiplies, including backward.

        Args:
            x1: Quantized left operand.
            x2: Quantized right operand.
            scale: Right operand scales.
            **kwargs: Unchanged device operation arguments.
        """
        assert x1.dtype == torch.float8_e4m3fn and x2.dtype == torch.float8_e4m3fn
        self.calls += 1
        return super().quant_matmul(x1, x2, scale, **kwargs)


@pytest.mark.parametrize("quantized", [False, True])
def test_linear_family(quantized: bool) -> None:
    """Compare ten optimizer steps against the existing Dense implementation.

    Args:
        quantized: Select MXFP8 rather than baseline computation.
    """
    backend = pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("A real NPU is required")
    torch.npu.set_device(0)
    torch.manual_seed(971)
    source = nn.Linear(128, 256, dtype=torch.bfloat16, device="npu")
    reference = copy.deepcopy(source)
    model = install_family(nn.Sequential(source), linear_config(quantized=quantized))
    ops = CountingDenseOps(backend)
    if quantized:
        strategy = build_linear_strategy(None, in_features=reference.in_features, out_features=reference.out_features)
        reference = LowPrecisionLinear.from_linear(reference, strategy=strategy)
        model[0].strategy.quantizer = MXFP8Quantizer(npu_ops=ops)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    inputs = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)
    initial = model[0].weight.detach().clone()
    losses = []
    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        actual_input = inputs.detach().clone().requires_grad_()
        expected_input = inputs.detach().clone().requires_grad_()
        actual, expected = model(actual_input), reference(expected_input)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        loss, expected_loss = actual.float().square().mean(), expected.float().square().mean()
        loss.backward()
        expected_loss.backward()
        torch.testing.assert_close(actual_input.grad, expected_input.grad, rtol=0, atol=0)
        for weight, expected_weight in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(weight.grad, expected_weight.grad, rtol=0, atol=0)
        optimizer.step()
        reference_optimizer.step()
        for weight, expected_weight in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(weight, expected_weight, rtol=0, atol=0)
        assert torch.isfinite(loss)
        losses.append(loss.item())
    assert not torch.equal(initial, model[0].weight)
    assert ops.calls == (30 if quantized else 0)
    print(json.dumps({"linear_quantized": quantized, "steps": 10, "matmul_calls": ops.calls,
                      "losses": losses, "reference": "exact outputs, input/weight gradients and updates"}), flush=True)

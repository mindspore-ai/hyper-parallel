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
"""Real A5 training with V4.1 clamped experts and the existing quantization strategies."""

import json
from typing import Any

import pytest
import torch
from torch import nn

from hyper_parallel.components.quantization.functional.base_gmm_func import _GroupedLinearFunction
from hyper_parallel.components.quantization.functional.strategy_factory import build_low_precision_strategy
from tests.common.qat_family import deepseek_experts_model, deepseek_qat_config, install_family


@pytest.mark.parametrize("fake,weight_format", [(False, "mxfp8"), (False, "mxfp4"), (True, "mxfp4")])
def test_deepseek_qat_family(fake: bool, weight_format: str) -> None:
    """Compare local training against native activation and the unmodified strategy.

    Args:
        fake: Select fake W4A8 rather than native quantization.
        weight_format: Named scheme's weight format; activation format is MXFP8.
    """
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("A real NPU is required")
    torch.npu.set_device(0)
    torch.manual_seed(971)
    config = deepseek_qat_config(fake=fake, weight_format=weight_format)
    model = deepseek_experts_model().to(device="npu", dtype=torch.bfloat16)
    source = model.experts
    reference_weights = nn.ParameterList([nn.Parameter(p.detach().clone()) for p in source.parameters()])
    install_family(model, config)
    expert = model.experts
    actual_weights = (expert.gate_up_proj, expert.down_proj)
    policy = config.training.low_precision.resolve_dtype_scheme("experts")
    strategy = build_low_precision_strategy(policy, tile_shapes=tuple(tuple(p.shape) for p in reference_weights))
    assert type(expert.grouped_linear) is type(strategy)
    calls = []
    original = expert.grouped_linear.grouped_matmul

    def observed(*args: Any, **kwargs: Any) -> torch.Tensor:
        """Record real strategy dispatch while preserving device execution."""
        calls.append(kwargs["layout"])
        return original(*args, **kwargs)

    expert.grouped_linear.grouped_matmul = observed
    optimizer = torch.optim.SGD(expert.parameters(), lr=1e-4)
    reference_optimizer = torch.optim.SGD(reference_weights.parameters(), lr=1e-4)
    groups = torch.tensor([16, 0, 16, 32], device="npu", dtype=torch.int64)
    # Large projections exercise clipping; ordinary initialization alone rarely reaches limit=10.
    inputs = torch.randn(64, 128, device="npu", dtype=torch.bfloat16) * 64
    initial = [p.detach().clone() for p in actual_weights]
    losses = []
    for step in range(10):
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        actual_input = inputs.detach().clone().requires_grad_()
        expected_input = inputs.detach().clone().requires_grad_()
        actual = expert.forward_expert_major(actual_input, groups)
        projected = _GroupedLinearFunction.apply(expected_input, reference_weights[0], groups, strategy, 1)
        if step == 0:
            gate, up = projected.chunk(2, dim=-1)
            assert (gate > source.limit).any()
            assert (up > source.limit).any() and (up < -source.limit).any()
        # Use the native activation hook so the reference keeps identical quantization groups and order.
        activated = source._apply_gate(projected)  # pylint: disable=protected-access
        expected = _GroupedLinearFunction.apply(activated, reference_weights[1], groups, strategy, 1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        loss, expected_loss = actual.float().square().mean(), expected.float().square().mean()
        loss.backward()
        expected_loss.backward()
        torch.testing.assert_close(actual_input.grad, expected_input.grad, rtol=0, atol=0)
        for actual_weight, expected_weight in zip(actual_weights, reference_weights):
            torch.testing.assert_close(actual_weight.grad, expected_weight.grad.transpose(-2, -1), rtol=0, atol=0)
        optimizer.step()
        reference_optimizer.step()
        for actual_weight, expected_weight in zip(actual_weights, reference_weights):
            torch.testing.assert_close(actual_weight, expected_weight.transpose(-2, -1), rtol=0, atol=0)
        assert torch.isfinite(loss)
        losses.append(loss.item())
    assert all(not torch.equal(before, after) for before, after in zip(initial, actual_weights))
    assert len(calls) == 60
    assert {"NN", "NT", "TN"}.issubset(calls)
    print(json.dumps({"deepseek_clamp": source.limit, "fake": fake, "weight_format": weight_format,
                      "strategy": type(strategy).__name__, "steps": 10, "gmm_calls": len(calls),
                      "losses": losses, "reference": "exact outputs, input/weight gradients and updates"}), flush=True)

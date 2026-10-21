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
"""A5 comparison of Family-selected QAT against the unmodified strategy path."""

import copy
import json
from typing import Any

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from hyper_parallel.components.quantization.functional.base_gmm_func import _GroupedLinearFunction
from hyper_parallel.components.quantization.functional.strategy_factory import build_low_precision_strategy
from hyper_parallel.components.quantization.quantizers.mxfp8 import MXFP8Quantizer
from tests.common.grouped_experts_family import TinyMoE, family_config, replace_experts
from tests.torch.expert_parallel._test_family_mxfp8_local import CountingMXOps
from tests.common.qat_family import install_family, qat_config


def _run(fake, block_size, weight_format):
    """Compare ten local expert training steps against the existing strategy."""
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("A real NPU is required")
    torch.npu.set_device(0)
    torch.manual_seed(971)
    config = qat_config(fake=fake, block_size=block_size, weight_format=weight_format)
    model = TinyMoE(dtype=torch.bfloat16).to("npu")
    install_family(model, config)
    expert = model.mlp.experts
    actual_weights = (expert.gate_up_proj, expert.down_proj)
    reference_weights = nn.ParameterList([
        nn.Parameter(p.detach().transpose(-2, -1).contiguous()) for p in actual_weights
    ])
    policy = config.training.low_precision.resolve_dtype_scheme("experts")
    strategy = build_low_precision_strategy(policy, tile_shapes=tuple(tuple(p.shape) for p in reference_weights))
    assert type(expert.grouped_linear) is type(strategy)
    calls = []
    original = expert.grouped_linear.grouped_matmul

    def observed(*args: Any, **kwargs: Any) -> torch.Tensor:
        """Record actual strategy dispatch without replacing the device operation."""
        calls.append(kwargs["layout"])
        return original(*args, **kwargs)

    expert.grouped_linear.grouped_matmul = observed
    optimizer = torch.optim.SGD(expert.parameters(), lr=0.01)
    reference_optimizer = torch.optim.SGD(reference_weights.parameters(), lr=0.01)
    groups = torch.tensor([16, 0, 16, 32], device="npu", dtype=torch.int64)
    inputs = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)
    initial = [p.detach().clone() for p in actual_weights]
    losses = []
    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        actual_input = inputs.detach().clone().requires_grad_()
        expected_input = inputs.detach().clone().requires_grad_()
        actual = expert.forward_expert_major(actual_input, groups)
        projected = _GroupedLinearFunction.apply(expected_input, reference_weights[0], groups, strategy, 1)
        gate, up = projected.chunk(2, -1)
        expected = _GroupedLinearFunction.apply(F.silu(gate) * up, reference_weights[1], groups, strategy, 1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        loss = actual.float().square().sum()
        expected_loss = expected.float().square().sum()
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
    assert all(not torch.equal(a, b) for a, b in zip(initial, actual_weights))
    assert {"NN", "NT", "TN"}.issubset(calls)
    print(json.dumps({"fake": fake, "block_size": block_size, "weight_format": weight_format,
                      "strategy": type(strategy).__name__, "steps": 10, "gmm_calls": len(calls),
                      "losses": losses, "reference": "exact outputs, input/weight gradients and updates"}), flush=True)


@pytest.mark.parametrize("fake,block_size,weight_format", [
    (False, 32, "mxfp8"), (False, 32, "mxfp4"), (True, 32, "mxfp4"),
    (False, 128, "mxfp4"), (True, 128, "mxfp4"),
])
def test_qat_family(fake: bool, block_size: int, weight_format: str) -> None:
    """Compare identical groups and reduction order through both real device paths.

    Args:
        fake: Select fake rather than native quantization.
        block_size: Number of elements sharing a quantization scale.
        weight_format: Weight format selected by the catalog entry.
    """
    _run(fake, block_size, weight_format)


def test_scheme_fused_mxfp8() -> None:
    """The named W8 scheme and fusion flag preserve the existing routed fused path."""
    backend = pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("A real NPU is required")
    torch.npu.set_device(0)
    torch.manual_seed(793)
    model = TinyMoE(dtype=torch.bfloat16).to("npu")
    reference = copy.deepcopy(model)
    config = qat_config(weight_format="mxfp8")
    config.plan_overrides[0].replace_module = config.plan_overrides[0].replace_module.replace(fused=True)
    install_family(model, config)
    replace_experts(reference, family_config(use_mxfp8=True, use_fused_mxfp8=True))
    ops = CountingMXOps(backend)
    model.mlp.experts.quantizer = MXFP8Quantizer(npu_ops=ops)
    indices = torch.tensor([[i % 4, (i + 1) % 4] for i in range(64)], device="npu")
    scores = torch.tensor([[.25, .75]] * 64, device="npu", dtype=torch.bfloat16)
    for current in (model, reference):
        current.mlp.gate.indices, current.mlp.gate.scores = indices, scores
    inputs = torch.randn(1, 64, 128, device="npu", dtype=torch.bfloat16)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    initial = model.mlp.experts.gate_up_proj.detach().clone()
    losses = []
    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        actual_input = inputs.detach().clone().requires_grad_()
        expected_input = inputs.detach().clone().requires_grad_()
        output, expected = model(actual_input), reference(expected_input)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        loss, reference_loss = output.float().square().sum(), expected.float().square().sum()
        loss.backward()
        reference_loss.backward()
        torch.testing.assert_close(actual_input.grad, expected_input.grad, rtol=0, atol=0)
        for weight, expected_weight in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(weight.grad, expected_weight.grad, rtol=0, atol=0)
        optimizer.step()
        reference_optimizer.step()
        for weight, expected_weight in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(weight, expected_weight, rtol=0, atol=0)
        assert torch.isfinite(loss)
        losses.append(loss.item())
    assert not torch.equal(initial, model.mlp.experts.gate_up_proj)
    assert ops.fused_forward == ops.fused_backward == 10
    assert len(ops.calls) == 60
    print(json.dumps({"scheme_fused_mxfp8": True, "steps": 10, "fusion_calls": [10, 10],
                      "gmm_calls": len(ops.calls), "losses": losses,
                      "reference": "exact outputs, input/weight gradients and updates"}), flush=True)

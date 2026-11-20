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
"""Single-NPU worker for local family construction and native execution parity."""
import os
from copy import deepcopy
from itertools import product
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch_npu  # pylint: disable=unused-import
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4MLP
from hyper_multicore.modules.mega_mhc.module import MegaMhc as NativeMegaMhc
from hyper_multicore.modules.mega_gate import MegaGate as SourceMegaGate

from examples.torch.module_family.config import (
    MegaMhcConfig, RMSNormConfig, SwiGLUMLPConfig, GateConfig, MegaGateConfig,
)
from hyper_parallel.components.modules import MegaMhc, RMSNorm, SwiGLUMLP, OffsetRMSNorm, SharedExpert
from hyper_parallel.components.modules.gate import Gate
from tests.common.mhc_reference import torch_mega_mhc


def test_local_families_npu() -> None:
    """Check native RMSNorm and clamped SwiGLU outputs and gradients against explicit math."""
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    device = torch.device("npu", torch.npu.current_device())
    for dtype in (torch.float32, torch.bfloat16):
        norm = RMSNorm(config=RMSNormConfig(128), use_fused_ops=True).to(device=device, dtype=dtype)
        x = torch.randn(4, 128, device=device, dtype=dtype, requires_grad=True)
        reference = x.detach().clone().requires_grad_()
        expected = (reference.float() / (reference.float().square().mean(-1, keepdim=True) + 1e-5).sqrt()).to(dtype)
        actual = norm(x)
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
        actual.sum().backward()
        expected.sum().backward()
        torch.testing.assert_close(x.grad, reference.grad, rtol=2e-2, atol=2e-2)
        mlp = SwiGLUMLP(config=SwiGLUMLPConfig(128, 256, swiglu_limit=0.3)).to(device=device, dtype=dtype)
        value = torch.randn(4, 128, device=device, dtype=dtype, requires_grad=True)
        gate, up = mlp.linear_fc1(value).float().chunk(2, -1)
        expected_mlp = mlp.linear_fc2((torch.nn.functional.silu(gate.clamp(max=0.3))
                                    * up.clamp(-0.3, 0.3)).to(dtype))
        result = mlp(value)
        torch.testing.assert_close(result, expected_mlp)
        result.float().square().sum().backward()
        if not all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                   for parameter in mlp.parameters()):
            raise AssertionError("clamped MLP must produce finite parameter gradients")
    torch.npu.synchronize()


def test_mega_mhc_family_npu() -> None:
    """The tracked fused boundary preserves existing multicore outputs, input and parameter gradients."""
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    device = torch.device("npu", torch.npu.current_device())
    tracked = MegaMhc(config=MegaMhcConfig(128), backend="fused").to(device)
    native = NativeMegaMhc(128, device=device)
    native.load_state_dict(tracked.state_dict())
    state = (torch.randn(64, 128, device=device, dtype=torch.bfloat16),
             torch.randn(64, 4, 128, device=device, dtype=torch.bfloat16),
             torch.rand(64, 4, device=device), torch.rand(64, 4, device=device),
             torch.rand(64, 4, 4, device=device).softmax(-1))
    left = tuple(value.requires_grad_() for value in state)
    right = tuple(value.detach().clone().requires_grad_() for value in state)
    actual, expected = tracked(*left), native(*right)
    for result, target in zip(actual, expected):
        torch.testing.assert_close(result, target, rtol=3e-3, atol=3e-3)
    gradients = tuple(torch.randn_like(value) * 0.1 for value in actual)
    torch.autograd.backward(actual, gradients)
    torch.autograd.backward(expected, gradients)
    for result, target in zip((*left, *tracked.parameters()), (*right, *native.parameters())):
        torch.testing.assert_close(result.grad, target.grad, rtol=3e-3, atol=3e-3)
    torch.npu.synchronize()


def test_ordinary_mega_mhc_family_npu() -> None:
    """Ordinary shifted boundary matches the pre-existing independent Torch golden on NPU."""
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    device = torch.device("npu", torch.npu.current_device())
    member = MegaMhc(config=MegaMhcConfig(128)).to(device)
    reference = NativeMegaMhc(128, device=device)
    reference.load_state_dict(member.state_dict())
    state = (torch.randn(64, 128, device=device, dtype=torch.bfloat16),
             torch.randn(64, 4, 128, device=device, dtype=torch.bfloat16),
             torch.rand(64, 4, device=device), torch.rand(64, 4, device=device),
             torch.rand(64, 4, 4, device=device).softmax(-1))
    left = tuple(value.requires_grad_() for value in state)
    right = tuple(value.detach().clone().requires_grad_() for value in state)
    actual = member(*left)
    expected = torch_mega_mhc(*right, reference.phi, reference.alpha, reference.bias, reference.norm_weight)
    for result, target in zip(actual, expected):
        torch.testing.assert_close(result, target)
    gradients = tuple(torch.randn_like(value) * 0.1 for value in actual)
    torch.autograd.backward(actual, gradients)
    torch.autograd.backward(expected, gradients)
    for result, target in zip((*left, *member.parameters()), (*right, *reference.parameters())):
        torch.testing.assert_close(result.grad, target.grad, rtol=3e-3, atol=3e-3)
    torch.npu.synchronize()


def test_source_contracts_npu() -> None:
    """Validate actual HF cast order, low-precision offset and V4.1 FP32 shared activation."""
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    torch.manual_seed(20261117)
    device = torch.device("npu", torch.npu.current_device())
    for source_type, factory in ((Qwen3RMSNorm, RMSNorm), (Qwen3_5RMSNorm, OffsetRMSNorm)):
        for backend, weight_dtype in product((False, True), (torch.float32, torch.bfloat16)):
            source = source_type(128).to(device=device, dtype=weight_dtype)
            torch.nn.init.uniform_(source.weight, 0.1, 0.9)
            member = factory(module=deepcopy(source), use_fused_ops=backend)
            left = torch.randn(4, 128, device=device, dtype=torch.bfloat16).requires_grad_()
            right = left.detach().clone().requires_grad_()
            expected = source(left)
            actual = member(right)
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
            if actual.dtype != expected.dtype:
                raise AssertionError("source output dtype promotion must be preserved")
            upstream = torch.randn_like(expected)
            expected.backward(upstream)
            actual.backward(upstream)
            torch.testing.assert_close(right.grad, left.grad, rtol=2e-2, atol=2e-2)
            torch.testing.assert_close(member.weight.grad, source.weight.grad, rtol=2e-2, atol=2e-2)
    for cast_order in ("before_weight", "after_weight"):
        for weight_dtype in (torch.float32, torch.bfloat16):
            config = SimpleNamespace(hidden_size=128, eps=1e-6)
            eager = RMSNorm(config=config, cast_order=cast_order,
                            fp32_weight=weight_dtype == torch.float32).to(device=device, dtype=weight_dtype)
            torch.nn.init.uniform_(eager.weight, 0.2, 1.8)
            npu = RMSNorm(config=config, cast_order=cast_order, fp32_weight=weight_dtype == torch.float32,
                          use_fused_ops=True).to(device=device, dtype=weight_dtype)
            npu.load_state_dict(eager.state_dict())
            left = torch.randn(4, 128, device=device, dtype=torch.bfloat16).requires_grad_()
            right = left.detach().clone().requires_grad_()
            expected = eager(left)
            actual = npu(right)
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
            if actual.dtype != expected.dtype:
                raise AssertionError("NPU must preserve the declared scale/cast output dtype")
            upstream = torch.randn_like(expected)
            expected.backward(upstream)
            actual.backward(upstream)
            torch.testing.assert_close(right.grad, left.grad, rtol=2e-2, atol=2e-2)
            torch.testing.assert_close(npu.weight.grad, eager.weight.grad, rtol=2e-2, atol=2e-2)
    for limit in (0.0, 0.3):
        config = SimpleNamespace(hidden_size=128, intermediate_size=256,
                                 hidden_act="silu", mlp_bias=False, swiglu_limit=0.3)
        source = DeepseekV4MLP(config).to(device=device, dtype=torch.bfloat16)
        member = SharedExpert(module=deepcopy(source), preserve_source_layout=True,
                              activation_dtype="float32", swiglu_limit=limit)
        left = torch.randn(4, 128, device=device, dtype=torch.bfloat16).requires_grad_()
        right = left.detach().clone().requires_grad_()
        gate, up = source.gate_proj(left).float(), source.up_proj(left).float()
        if limit > 0:
            gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
        expected = source.down_proj((torch.nn.functional.silu(gate) * up).to(left.dtype))
        actual = member(right)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        upstream = torch.randn_like(expected)
        expected.backward(upstream)
        actual.backward(upstream)
        torch.testing.assert_close(right.grad, left.grad, rtol=2e-2, atol=2e-2)
        for name, parameter in source.named_parameters():
            torch.testing.assert_close(member.get_parameter(name).grad, parameter.grad, rtol=2e-2, atol=2e-2)
    torch.npu.synchronize()


def test_router_contracts_npu() -> None:
    """Compare ordinary routing math and all gradients with an independent device-side expression."""
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    torch.manual_seed(20261117)
    device = torch.device("npu", torch.npu.current_device())
    for scoring in ("sigmoid", "softmax", "sqrtsoftplus"):
        for dtype in (torch.float32, torch.bfloat16):
            member = Gate(config=GateConfig(
                128, 8, 2, scoring_func=scoring, routed_scaling_factor=1.7, sorted_indices=True,
            )).to(device=device, dtype=dtype)
            weight = member.weight.detach().clone().requires_grad_()
            left = torch.randn(2, 4, 128, device=device, dtype=dtype).requires_grad_()
            right = left.detach().clone().requires_grad_()
            bias = torch.linspace(-0.3, 0.3, 8, device=device)
            logits = right.reshape(-1, 128).float() @ weight.float().T
            if scoring == "sigmoid":
                scores = logits.sigmoid()
            elif scoring == "softmax":
                scores = logits.softmax(-1)
            else:
                scores = torch.nn.functional.softplus(logits).sqrt()  # pylint: disable=not-callable
            indices = (scores + bias).topk(2, -1).indices
            expected_weights = scores.gather(1, indices)
            expected_weights = expected_weights / (expected_weights.sum(-1, keepdim=True) + 1e-20) * 1.7
            actual_logits, actual_weights, actual_indices = member(left, bias)
            torch.testing.assert_close(actual_logits, logits, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(actual_weights, expected_weights, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(actual_indices, indices)
            upstream = torch.randn_like(expected_weights)
            (actual_logits.square().sum() * 0.01 + (actual_weights * upstream).sum()).backward()
            (logits.square().sum() * 0.01 + (expected_weights * upstream).sum()).backward()
            torch.testing.assert_close(left.grad, right.grad, rtol=2e-2, atol=2e-2)
            torch.testing.assert_close(member.weight.grad, weight.grad, rtol=2e-2, atol=2e-2)
    torch.npu.synchronize()


def test_mega_gate_family_npu() -> None:
    """Exercise real fused Route/RouteGrad against both the original module and its Torch reference."""
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    device = torch.device("npu", torch.npu.current_device())
    torch.manual_seed(73)
    for dtype, top_k, vision in product((torch.float32, torch.bfloat16), (1, 6), (False, True)):
        config = MegaGateConfig(128, 16, top_k, routed_scaling_factor=1.5, vision_enabled=vision)
        candidate = Gate(config=config, implementation="mega").to(device=device, dtype=dtype)
        original = SourceMegaGate(hidden_size=128, num_experts=16, top_k=top_k,
                                  scoring_func="sqrtsoftplus", routed_scaling_factor=1.5,
                                  vision_enabled=vision).to(device=device, dtype=dtype)
        with torch.no_grad():
            candidate.bias.copy_(torch.linspace(-0.2, 0.2, 16, device=device))
            if vision:
                candidate.bias_vl.copy_(-candidate.bias)
        original.load_state_dict(candidate.state_dict())
        reference = deepcopy(original)
        hidden = torch.randn(1, 257, 128, device=device, dtype=dtype)
        mask = (torch.arange(257, device=device).remainder(4) == 0).view(1, 257) if vision else None
        runs = []
        for module in (reference, original, candidate):
            inputs = hidden.detach().clone().requires_grad_()
            if module is reference:
                result = module._torch_forward(inputs, mask)  # pylint: disable=protected-access
            else:
                with patch.object(module, "_torch_forward", side_effect=AssertionError("native route required")):
                    result = module(inputs, mask)
            logits, weights, indices = result
            loss = logits.square().mean() * 0.01 + (weights * (indices.float() + 1) / 16).square().mean()
            loss.backward()
            order = indices.argsort(dim=-1)
            runs.append((logits.detach(), weights.gather(1, order).detach(), indices.gather(1, order),
                         inputs.grad.detach(), module.weight.grad.detach()))
            assert module.bias.grad is None
            if vision:
                assert module.bias_vl.grad is None
        for actual in runs[1:]:
            for index, (value, expected) in enumerate(zip(actual, runs[0])):
                torch.testing.assert_close(value, expected, rtol=1e-2 if index >= 3 else 1e-5,
                                           atol=1e-4 if index >= 3 else 1e-6)
    torch.npu.synchronize()

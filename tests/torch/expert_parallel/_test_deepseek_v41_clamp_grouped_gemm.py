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
"""Ascend worker for DeepSeek-V4.1 clamped grouped-GEMM parity."""

import copy
import os
from types import SimpleNamespace

import torch
import torch_npu  # pylint: disable=unused-import
from torch.nn import functional
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Experts

from hyper_parallel.models.deepseek_v41.adapter.distributed.moe_engram_expert_parallel import (
    _deepseek_v41_grouped_expert_forward,
)


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    """Compare NPU tensors with the tolerance used by grouped-GEMM ST."""
    torch.testing.assert_close(
        actual.detach().float().cpu(),
        expected.detach().float().cpu(),
        rtol=1.0e-2,
        atol=1.0e-2,
        msg=label,
    )


def _check_grouped_gemm_parity(device: torch.device, limit: float) -> None:
    """Check native optional clamping with real grouped projections and gradients."""
    config = SimpleNamespace(
        num_local_experts=4,
        hidden_size=64,
        intermediate_size=128,
        hidden_act="silu",
        swiglu_limit=limit,
    )
    torch.manual_seed(91)
    experts = DeepseekV4Experts(config)
    for parameter in experts.parameters():
        torch.nn.init.normal_(parameter)
    reference = copy.deepcopy(experts)
    experts = experts.to(device=device, dtype=torch.bfloat16)
    reference = reference.to(device=device, dtype=torch.bfloat16)

    counts = torch.tensor([3, 0, 2, 1], device=device, dtype=torch.int64)
    inputs = torch.randn(6, config.hidden_size, device=device, dtype=torch.bfloat16, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_()
    actual = _deepseek_v41_grouped_expert_forward(experts, inputs, counts)

    expected_parts = []
    group_start = 0
    for expert_id, token_count in enumerate(counts.cpu().tolist()):
        group_end = group_start + token_count
        gate, up = functional.linear(  # pylint: disable=not-callable
            reference_inputs[group_start:group_end],
            reference.gate_up_proj[expert_id],
        ).chunk(2, dim=-1)
        gate, up = gate.float(), up.float()
        if limit > 0:
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
        intermediate = functional.silu(gate) * up
        expected_parts.append(
            functional.linear(  # pylint: disable=not-callable
                intermediate.to(reference_inputs.dtype),
                reference.down_proj[expert_id],
            )
        )
        group_start = group_end
    expected = torch.cat(expected_parts, dim=0)
    upstream = torch.randn_like(actual)
    (actual * upstream).sum().backward()
    (expected * upstream).sum().backward()

    _assert_close(actual, expected, "DeepSeek-V4.1 clamped grouped-GEMM output")
    _assert_close(inputs.grad, reference_inputs.grad, "DeepSeek-V4.1 clamped grouped-GEMM input gradient")
    for (actual_name, actual_parameter), (expected_name, expected_parameter) in zip(
            experts.named_parameters(), reference.named_parameters(),
    ):
        if actual_name != expected_name:
            raise AssertionError(f"Parameter order mismatch: {actual_name} != {expected_name}")
        _assert_close(
            actual_parameter.grad,
            expected_parameter.grad,
            f"DeepSeek-V4.1 clamped grouped-GEMM {actual_name} gradient",
        )


def test_clamped_grouped_gemm_forward_backward() -> None:
    """Real grouped projections preserve native positive and disabled clamp limits."""
    device = torch.device("npu", int(os.environ.get("LOCAL_RANK", "0")))
    torch.npu.set_device(device)
    for limit in (-1.0, 0.0, 1.5, 10.0):
        _check_grouped_gemm_parity(device, limit)

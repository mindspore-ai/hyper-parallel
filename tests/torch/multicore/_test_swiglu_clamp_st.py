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
"""Numerical NPU regression for MegaMoe SwiGLU clamp semantics."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu

from hyper_parallel.components.functional.grouped_matmul import grouped_matmul
from hyper_parallel.core.multicore import MegaMoeExperts
from hyper_parallel.core.multicore.modules.mega_moe import function as mega_function
from hyper_parallel.core.multicore.modules.mega_moe.function import execute_mega_moe
from hyper_parallel.core.multicore.modules.mega_moe.route import (
    PreparedTopKRoute,
    prepare_topk_route,
)
from tests.torch.multicore import _test_mega_moe as baseline


_CLAMP_LIMIT = 10.0


def _shape() -> baseline.MoeShape:
    """Return a one-rank shape that exercises the activation kernel."""
    return baseline.MoeShape(
        local_num_tokens=128,
        hidden_size=512,
        intermediate_size=128,
        num_experts=2,
        top_k=2,
        ep_size=1,
    )


def _boundary_inputs(shape: baseline.MoeShape) -> tuple[torch.Tensor, torch.Tensor]:
    """Create BF16 rows at, inside, outside, and adjacent to both limits."""
    cases = torch.tensor(
        [
            (10.0, 10.0),
            (10.0, -10.0),
            (9.9375, 0.5),
            (10.0625, 0.5),
            (1.0, 9.9375),
            (1.0, 10.0),
            (1.0, 10.0625),
            (1.0, -9.9375),
            (1.0, -10.0),
            (1.0, -10.0625),
            (-10.0, 0.5),
        ],
        dtype=torch.bfloat16,
        device=baseline.DEVICE,
    )
    row_indices = torch.arange(shape.local_num_tokens, device=baseline.DEVICE)
    selected = cases[row_indices.remainder(cases.shape[0])]
    hidden = torch.zeros(
        (shape.local_num_tokens, shape.hidden_size),
        dtype=torch.bfloat16,
        device=baseline.DEVICE,
    )
    hidden[:, 0] = selected[:, 0]
    hidden[:, 1] = selected[:, 1]
    return hidden, torch.ones_like(hidden)


def _initialize_weights(
    common: torch.nn.Module,
    mega: MegaMoeExperts,
    shape: baseline.MoeShape,
) -> None:
    """Make gate/up projections expose the boundary inputs exactly."""
    # pylint: disable=protected-access
    with torch.no_grad():
        gate_weight = baseline._local_tensor(common.w1)
        up_weight = baseline._local_tensor(common.w3)
        down_weight = baseline._local_tensor(common.w2)
        gate_weight.zero_()
        up_weight.zero_()
        down_weight.fill_(1.0 / shape.intermediate_size)
        gate_weight[:, :, 0] = 1.0
        up_weight[:, :, 1] = 1.0
    baseline._copy_common_weights_to_mega(common, mega)


def _new_layers(
    shape: baseline.MoeShape,
    swiglu_limit: float | None,
) -> tuple[torch.nn.Module, MegaMoeExperts]:
    """Create reference and MegaMoe layers with identical deterministic weights."""
    common = baseline.new_common_moe(shape)
    mega = MegaMoeExperts(
        local_num_tokens=shape.local_num_tokens,
        hidden_size=shape.hidden_size,
        intermediate_size=shape.intermediate_size,
        num_experts=shape.num_experts,
        top_k=shape.top_k,
        swiglu_limit=swiglu_limit,
        ep_size=shape.ep_size,
        ep_group=dist.group.WORLD,
    ).to(device=baseline.DEVICE, dtype=torch.bfloat16)
    _initialize_weights(common, mega, shape)
    return common, mega


def _run_reference(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    grad_output: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Run grouped matmuls around an autograd-visible PyTorch clamp reference."""
    hidden = hidden_states.detach().clone()
    route_weights = topk_weights.detach().clone().requires_grad_(True)
    gate_up = torch.nn.Parameter(gate_up_weight.detach().clone())
    down = torch.nn.Parameter(down_weight.detach().clone())
    routed, mapping = torch_npu.npu_moe_token_permute(hidden, topk_ids)
    group_list = tokens_per_expert.cumsum(dim=0).to(torch.int64).contiguous()
    gate_up_output = grouped_matmul(routed, gate_up, group_list=group_list)
    gate, up = gate_up_output.chunk(2, dim=-1)
    activation = (
        F.silu(gate.float().clamp(max=_CLAMP_LIMIT))
        * up.float().clamp(min=-_CLAMP_LIMIT, max=_CLAMP_LIMIT)
    ).to(hidden.dtype)
    expert_output = grouped_matmul(activation, down, group_list=group_list)
    output = torch_npu.npu_moe_token_unpermute(
        expert_output,
        mapping,
        probs=route_weights.float().contiguous(),
    )
    output.backward(grad_output)
    torch.npu.synchronize(baseline.DEVICE)
    return {
        "output": output.detach(),
        "route_weight_grad": route_weights.grad.detach(),
        "gate_up_weight_grad": gate_up.grad.detach(),
        "down_weight_grad": down.grad.detach(),
    }


def _run_mega(
    layer: MegaMoeExperts,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    grad_output: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Run MegaMoe forward/backward and return detached numerical evidence."""
    layer.zero_grad(set_to_none=True)
    hidden = hidden_states.detach().clone()
    route_weights = topk_weights.detach().clone().requires_grad_(True)
    output = layer(
        hidden,
        topk_ids,
        route_weights,
        tokens_per_expert=tokens_per_expert,
    )
    output.backward(grad_output)
    torch.npu.synchronize(baseline.DEVICE)
    return {
        "output": output.detach(),
        "route_weight_grad": route_weights.grad.detach(),
        "gate_up_weight_grad": layer.gate_up_weight.grad.detach(),
        "down_weight_grad": layer.down_weight.grad.detach(),
    }


def _run_grouped_backward(
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    grad_output: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Run the reference expert path without requesting input gradients."""
    layer.zero_grad(set_to_none=True)
    route_weights = topk_weights.detach().clone().requires_grad_(True)
    output = baseline.forward_layer(
        layer,
        hidden_states.detach(),
        topk_ids,
        route_weights,
        tokens_per_expert=tokens_per_expert,
    )
    output.backward(grad_output)
    torch.npu.synchronize(baseline.DEVICE)
    gate_up_grad, down_grad = baseline.expert_weight_gradients(layer)
    return {
        "output": output.detach(),
        "route_weight_grad": route_weights.grad.detach(),
        "gate_up_weight_grad": gate_up_grad.detach(),
        "down_weight_grad": down_grad.detach(),
    }


def _run_routed_reference(
    layer: torch.nn.Module,
    routed_tokens: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    grad_output: torch.Tensor,
    swiglu_limit: float | None,
) -> torch.Tensor:
    """Run grouped matmuls with optional clamping on already routed rows."""
    # pylint: disable=protected-access
    layer.zero_grad(set_to_none=True)
    routed = routed_tokens.detach().clone().requires_grad_(True)
    group_list = tokens_per_expert.cumsum(dim=0).to(torch.int64).contiguous()
    gate = grouped_matmul(
        routed,
        baseline._local_tensor(layer.w1).transpose(1, 2),
        group_list=group_list,
    )
    up = grouped_matmul(
        routed,
        baseline._local_tensor(layer.w3).transpose(1, 2),
        group_list=group_list,
    )
    if swiglu_limit is None:
        activation = (F.silu(gate.float()) * up.float()).to(routed.dtype)
    else:
        activation = (
            F.silu(gate.float().clamp(max=swiglu_limit))
            * up.float().clamp(min=-swiglu_limit, max=swiglu_limit)
        ).to(routed.dtype)
    output = grouped_matmul(
        activation,
        baseline._local_tensor(layer.w2).transpose(1, 2),
        group_list=group_list,
    )
    input_grad = torch.autograd.grad(output, routed, grad_output)[0]
    torch.npu.synchronize(baseline.DEVICE)
    return input_grad.detach()


def _run_routed_mega(
    layer: MegaMoeExperts,
    routed_tokens: torch.Tensor,
    route: PreparedTopKRoute,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    """Run the MegaMoe expert kernel on already permuted rows."""
    layer.zero_grad(set_to_none=True)
    routed = routed_tokens.detach().clone().requires_grad_(True)
    resources = layer._get_execution_resources(routed)  # pylint: disable=protected-access
    output = execute_mega_moe(
        routed,
        layer.gate_up_weight,
        layer.down_weight,
        route.metadata,
        resources.plan,
        resources.workspace,
    )
    input_grad = torch.autograd.grad(output, routed, grad_output)[0]
    torch.npu.synchronize(baseline.DEVICE)
    return input_grad.detach()


def _assert_numerical_results(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
) -> None:
    """Compare clamped outputs and gradients with the PyTorch reference."""
    for name, actual_tensor in actual.items():
        baseline.assert_close(name, actual_tensor, expected[name])


def test_swiglu_clamp_numerics() -> None:
    """Validate clamped boundary gradients and the unclamped compatibility path."""
    # pylint: disable=protected-access
    shape = _shape()
    hidden, grad_output = _boundary_inputs(shape)
    topk_ids, topk_weights, tokens_per_expert = baseline._make_fixed_route(
        shape,
        "balanced",
    )

    common, unclamped = _new_layers(shape, None)
    try:
        expected = _run_grouped_backward(
            common,
            hidden,
            topk_ids,
            topk_weights,
            tokens_per_expert,
            grad_output,
        )
        actual = _run_mega(
            unclamped,
            hidden,
            topk_ids,
            topk_weights,
            tokens_per_expert,
            grad_output,
        )
        _assert_numerical_results(actual, expected)
    finally:
        unclamped.close()

    _, clamped = _new_layers(shape, _CLAMP_LIMIT)
    try:
        actual = _run_mega(
            clamped,
            hidden,
            topk_ids,
            topk_weights,
            tokens_per_expert,
            grad_output,
        )
        expected = _run_reference(
            hidden,
            topk_ids,
            topk_weights,
            tokens_per_expert,
            grad_output,
            clamped.gate_up_weight,
            clamped.down_weight,
        )
        _assert_numerical_results(actual, expected)
    finally:
        clamped.close()


def test_swiglu_clamp_input_gradient() -> None:
    """Compare routed input gradients with and without the clamp option."""
    # pylint: disable=protected-access
    shape = _shape()
    hidden, _ = _boundary_inputs(shape)
    topk_ids, topk_weights, tokens_per_expert = baseline._make_fixed_route(
        shape,
        "balanced",
    )
    for swiglu_limit in (None, _CLAMP_LIMIT):
        common, mega = _new_layers(shape, swiglu_limit)
        try:
            resources = mega._get_execution_resources(hidden)  # pylint: disable=protected-access
            route = prepare_topk_route(
                hidden,
                topk_ids,
                topk_weights,
                resources.spec,
                tokens_per_expert,
                workspace=resources.workspace,
            )
            routed_grad = torch.ones_like(route.routed_tokens)
            expected = _run_routed_reference(
                common,
                route.routed_tokens,
                route.tokens_per_expert,
                routed_grad,
                swiglu_limit,
            )
            actual = _run_routed_mega(
                mega,
                route.routed_tokens,
                route,
                routed_grad,
            )
            baseline.assert_close("input_grad", actual, expected)
        finally:
            mega.close()


def test_swiglu_clamp_ub_scratch() -> None:
    """Keep scalar Select scratch from overwriting nonzero SwiGLU gradients."""
    original_launch = mega_function._launch_backward_kernel  # pylint: disable=protected-access
    observed_calls = 0
    comparisons = []

    def check_backward(plan: Any, saved: Any, grad_output: Any, execution: Any) -> None:
        nonlocal observed_calls
        original_launch(plan, saved, grad_output, execution)
        torch.npu.synchronize(baseline.DEVICE)
        rows = int(saved.group_list[-1].item())
        gate, up = saved.up_proj[:rows].detach().cpu().double().chunk(2, dim=-1)
        derivative = execution.intermediates.act_grad[:rows].detach().cpu().double()
        sigmoid = torch.sigmoid(gate.clamp(max=_CLAMP_LIMIT))
        gate_grad = derivative * up.clamp(-_CLAMP_LIMIT, _CLAMP_LIMIT) * sigmoid
        gate_grad *= 1 + gate.clamp(max=_CLAMP_LIMIT) * (1 - sigmoid)
        gate_grad *= gate <= _CLAMP_LIMIT
        up_grad = derivative * gate.clamp(max=_CLAMP_LIMIT) * sigmoid
        up_grad *= (up >= -_CLAMP_LIMIT) & (up <= _CLAMP_LIMIT)
        expected = torch.cat((gate_grad, up_grad), dim=-1).bfloat16().float()
        actual = execution.intermediates.swiglu_grad[:rows].detach().cpu().float()
        comparisons.append((actual, expected))
        observed_calls += 1

    for token_count in (128, 4096):
        shape = baseline.MoeShape(
            local_num_tokens=token_count,
            hidden_size=512,
            intermediate_size=2304,
            num_experts=2,
            top_k=2,
            ep_size=1,
        )
        hidden = torch.zeros((token_count, shape.hidden_size), dtype=torch.bfloat16, device=baseline.DEVICE)
        hidden[:, 0] = 1.5
        hidden[:, 1] = -2.75
        grad_output = torch.full_like(hidden, -0.001)
        topk_ids, topk_weights, counts = baseline._make_fixed_route(  # pylint: disable=protected-access
            shape, "balanced"
        )
        _, mega = _new_layers(shape, _CLAMP_LIMIT)
        try:
            with patch.object(mega_function, "_launch_backward_kernel", check_backward):
                _run_mega(mega, hidden, topk_ids, topk_weights, counts, grad_output)
        finally:
            mega.close()
        # Assert after backward releases its workspace so failures preserve normal cleanup.
        actual, expected = comparisons[-1]
        torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.0)
    assert observed_calls == 2, f"Expected 2 inspected backward calls, got {observed_calls}"

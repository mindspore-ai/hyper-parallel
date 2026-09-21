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
"""Isolate per-rank BF16 dW rounding from expert-owner gradient aggregation.

This diagnostic intentionally uses private precision harness helpers. It changes
only reference copies, not the native operator or production HF block.
"""
# pylint: disable=protected-access

from __future__ import annotations

import copy
import json
import math
import os

import torch
import torch.distributed as dist
from torch.nn import functional
import torch_npu

from hyper_parallel.core.multicore.examples.mega_moe import deepseek_v41_precision as precision
from hyper_parallel.core.multicore.examples.mega_moe.deepseek_v41_operator_trace import _cpu, _metric


def _fused_experts(module, hidden, indices, weights):
    flat = hidden.reshape(-1, hidden.shape[-1])
    final = torch.zeros_like(flat)
    intermediates = {}
    for expert in range(module.num_experts):
        slots, tokens = torch.where(indices.T == expert)
        if tokens.numel() == 0:
            continue
        x = flat[tokens]
        # Torch exposes linear through a builtin binding that pylint cannot infer.
        # pylint: disable-next=not-callable
        gate_up = functional.linear(x, module.gate_up_proj[expert])
        activation = torch_npu.npu_swiglu(gate_up)
        # pylint: disable-next=not-callable
        down = functional.linear(activation, module.down_proj[expert])
        for value in (gate_up, activation, down):
            value.retain_grad()
        final.index_add_(0, tokens, (down * weights[tokens, slots, None]).to(final.dtype))
        intermediates[expert] = (x, gate_up, activation, down)
    return final.reshape_as(hidden), intermediates


def _fp32_partials(module, intermediates):
    gradients = {"gate_up_proj": torch.zeros_like(module.gate_up_proj, device="cpu", dtype=torch.float32),
                 "down_proj": torch.zeros_like(module.down_proj, device="cpu", dtype=torch.float32)}
    for expert, (x, gate_up, activation, down) in intermediates.items():
        gradients["gate_up_proj"][expert] = _cpu(gate_up.grad).T @ _cpu(x)
        gradients["down_proj"][expert] = _cpu(down.grad).T @ _cpu(activation)
    return gradients


def _step(source, candidate, args, generator, device, step):
    rank = dist.get_rank()
    precision._synchronize_parameters(source, candidate, rank)
    fused_source = copy.deepcopy(source)
    source.zero_grad(set_to_none=True)
    candidate.zero_grad(set_to_none=True)
    fused_source.zero_grad(set_to_none=True)
    inputs = torch.randn(1, args.tokens, args.hidden_size, generator=generator).to(device, torch.bfloat16)
    inputs.requires_grad_()
    other = inputs.detach().clone().requires_grad_()
    dy = torch.randn(inputs.shape, generator=generator).to(device) / math.sqrt(args.tokens)
    expected, actual, expected_route, actual_route = precision._forward_pair(
        source, candidate, inputs, other, args, None)
    (expected.float() * dy).sum().backward()
    (actual.float() * dy).sum().backward()
    fused_inputs = inputs.detach().clone().requires_grad_()
    fused_output, intermediates = _fused_experts(
        fused_source.experts, fused_inputs, expected_route["indices"], expected_route["weights"].detach())
    fused_output = fused_output + fused_source.shared_experts(fused_inputs)
    (fused_output.float() * dy).sum().backward()
    partials = _fp32_partials(fused_source.experts, intermediates)
    result = {"step": step, "route_ids_equal": torch.equal(actual_route["indices"], expected_route["indices"]),
              "gradients": {}}
    for name, partial in partials.items():
        hf_grad = getattr(source.experts, name).grad
        fused_grad = getattr(fused_source.experts, name).grad
        partial_rounding = _metric(_cpu(fused_grad), partial.bfloat16().float())
        # Three reductions use the same group and differ only in which numerical
        # boundary is rounded: unfused local BF16, fused local BF16, FP32 partials.
        dist.all_reduce(hf_grad)
        dist.all_reduce(fused_grad)
        accumulated = partial.to(device)
        dist.all_reduce(accumulated)
        local_experts = candidate.experts.local_experts
        expert_slice = slice(rank * local_experts, (rank + 1) * local_experts)
        native_name = name.replace("gate_up_proj", "gate_up_weight").replace("down_proj", "down_weight")
        native_grad = _cpu(getattr(candidate.experts, native_name).grad.transpose(1, 2))
        result["gradients"][name] = {
            "local_fused_dw_vs_fp32_rounded": partial_rounding,
            "native_vs_original_bf16_reduce": _metric(native_grad, _cpu(hf_grad[expert_slice])),
            "native_vs_fused_bf16_reduce": _metric(native_grad, _cpu(fused_grad[expert_slice])),
            "native_vs_fused_fp32_reduce": _metric(native_grad, _cpu(accumulated[expert_slice])),
            "native_vs_fused_fp32_reduce_rounded": _metric(
                native_grad, _cpu(accumulated[expert_slice].bfloat16())),
        }
    torch.optim.SGD(source.parameters(), lr=0.01).step()
    return result


def main() -> None:
    """Compare synchronized WORLD=EP expert gradients across reduction boundaries."""
    args = precision._arguments(None)
    args.reference = "hf_replicated"
    if args.vision:
        raise ValueError("Reduction diagnosis currently uses text routing only")
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group("hccl")
    if args.ep_size not in (None, dist.get_world_size()):
        raise ValueError("Reduction diagnosis requires WORLD=EP; subgroups are not supported")
    if args.num_experts % dist.get_world_size():
        raise ValueError("num-experts must be divisible by WORLD=EP")
    device = torch.device("npu", torch.npu.current_device())
    source = precision._source_block(args, device)
    candidate = precision.DeepseekV41MegaMoe(copy.deepcopy(source), local_num_tokens=args.tokens,
                                           ep_group=dist.group.WORLD, dispatch_mode=args.dispatch_mode)
    try:
        generator = torch.Generator().manual_seed(4000 + dist.get_rank())
        steps = [_step(source, candidate, args, generator, device, step) for step in range(args.steps)]
        ranks = [None] * dist.get_world_size()
        dist.all_gather_object(ranks, {"rank": dist.get_rank(), "steps": steps})
        if dist.get_rank() == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            report = {"environment": precision._environment(), "config": vars(args) | {"output": str(args.output)},
                      "route": args.route,
                      "dispatch_mode": args.dispatch_mode, "world_size": dist.get_world_size(), "ranks": ranks,
                      "scope": "Same-state intervention: fusion versus per-rank dW rounding before EP reduction; "
                               "FP32 partials use actual fused HF BF16 intermediates, not an end-to-end oracle"}
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"Reduction trace written to {args.output}", flush=True)
    finally:
        candidate.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

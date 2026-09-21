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
"""Capture actual EP1 native intermediates and replay individual MoE operators.

Private launch hooks are intentional here: the public adapter does not expose
kernel scratch. Copies synchronize before scratch reuse; never use this probe
for timing. The native kernels, weights and activation policy are unchanged.
"""

# pylint: disable=protected-access

from __future__ import annotations

import copy
import json
import math
import os
from types import MethodType
from typing import Any
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.nn import functional
import torch_npu

from hyper_parallel.core.multicore.examples.mega_moe import deepseek_v41_precision as precision
from hyper_parallel.core.multicore.examples.mega_moe.deepseek_v41_accuracy import fp32_accuracy
from hyper_parallel.core.multicore.modules.mega_moe import function as native


def _cpu(tensor):
    return tensor.detach().to(device="cpu", dtype=torch.float32).clone()


def _metric(actual, expected):
    result = fp32_accuracy(actual, expected)
    result["equal"] = torch.equal(actual, expected)
    result["different_elements"] = int((actual != expected).sum())
    return result


class _Capture:
    """Copy scratch at the launch boundary before ownership or aliases change."""

    def __init__(self) -> None:
        """Retain the unpatched launchers and initialize this step's snapshots."""
        self.values = {}
        self.forward = native._launch_forward_kernel
        self.backward = native._launch_backward_kernel

    def capture_forward(self, *args: Any) -> None:
        """Run the unchanged kernel and snapshot its received rows and outputs."""
        self.forward(*args)
        _, metadata, _, weight1, weight2, dispatch, _, intermediates, *_ = args
        count = int(metadata.group_list[-1])
        self.values.update(x=_cpu(dispatch[:count]), w1=_cpu(weight1), w2=_cpu(weight2),
                           ends=metadata.group_list.cpu().tolist())
        for name, value in zip(("gate_up", "activation", "down"), intermediates):
            self.values[name] = _cpu(value[:count])

    def capture_backward(self, *args: Any) -> None:
        """Copy weighted dY before launch, then gradients before scratch reuse."""
        self.values["dy"] = _cpu(args[2][:self.values["x"].shape[0]])
        self.backward(*args)
        for name, value in zip(("dw1", "dw2", "dactivation", "dgate_up", "dx"), args[5]):
            count = value.shape[0] if name.startswith("dw") else self.values["x"].shape[0]
            self.values[name] = _cpu(value[:count])


def _activation_gradient(gate_up, upstream):
    gate, up = gate_up.chunk(2, dim=-1)
    sigmoid = gate.sigmoid()
    return torch.cat((upstream * up * sigmoid * (1 + gate * (1 - sigmoid)),
                      upstream * functional.silu(gate)), dim=-1)


def _local_references(values):
    """FP32 operators using each actual native operator's own rounded inputs."""
    references = {name: [] for name in ("gate_up", "activation", "down", "dactivation", "dgate_up", "dx")}
    dw1, dw2 = [], []
    start = 0
    for expert, end in enumerate(values["ends"]):
        x, gate_up, activation, dy, da, dg = (
            values[name][start:end] for name in ("x", "gate_up", "activation", "dy", "dactivation", "dgate_up")
        )
        w1, w2 = values["w1"][expert], values["w2"][expert]
        gate, up = gate_up.chunk(2, dim=-1)
        for name, value in zip(references, (
                x @ w1, functional.silu(gate) * up, activation @ w2,
                dy @ w2.T, _activation_gradient(gate_up, da), dg @ w1.T)):
            references[name].append(value)
        dw1.append(x.T @ dg)
        dw2.append(activation.T @ dy)
        start = end
    return {**{name: torch.cat(parts) for name, parts in references.items()},
            "dw1": torch.stack(dw1), "dw2": torch.stack(dw2)}


def _replay(values, device, *, fused):
    """Replay the expert chain on NPU with identical X, weights and weighted dY."""
    pieces = {name: [] for name in ("gate_up", "activation", "down", "dactivation", "dgate_up", "dx", "dw1", "dw2")}
    start = 0
    for expert, end in enumerate(values["ends"]):
        x = values["x"][start:end].to(device, torch.bfloat16).requires_grad_()
        w1 = values["w1"][expert].to(device, torch.bfloat16).requires_grad_()
        w2 = values["w2"][expert].to(device, torch.bfloat16).requires_grad_()
        gate_up = x @ w1
        gate, up = gate_up.chunk(2, dim=-1)
        activation = torch_npu.npu_swiglu(gate_up) if fused else functional.silu(gate) * up
        down = activation @ w2
        gate_up.retain_grad()
        activation.retain_grad()
        down.backward(values["dy"][start:end].to(device, torch.bfloat16))
        tensors = (gate_up, activation, down, activation.grad, gate_up.grad, x.grad, w1.grad, w2.grad)
        for name, value in zip(pieces, tensors):
            pieces[name].append(_cpu(value))
        start = end
    return {name: torch.stack(parts) if name.startswith("dw") else torch.cat(parts) for name, parts in pieces.items()}


def _rounding_checks(values, device):
    """Hold native gate/up and incoming activation gradient fixed for SwiGLU."""
    gate_up = values["gate_up"].to(device, torch.bfloat16).requires_grad_()
    gate, up = gate_up.chunk(2, dim=-1)
    activation = functional.silu(gate) * up
    activation.backward(values["dactivation"].to(device, torch.bfloat16))
    fused_input = gate_up.detach().clone().requires_grad_()
    fused = torch_npu.npu_swiglu(fused_input)
    fused.backward(values["dactivation"].to(device, torch.bfloat16))
    return {"hf_forward": _metric(values["activation"], _cpu(activation)),
            "fused_forward": _metric(values["activation"], _cpu(fused)),
            "hf_backward": _metric(values["dgate_up"], _cpu(gate_up.grad)),
            "fused_backward": _metric(values["dgate_up"], _cpu(fused_input.grad))}


def _routing_checks(values, inputs, dy, route, routed_output):
    """Check EP1 transport and route multiplication using captured down output."""
    flat = _cpu(inputs).reshape(-1, inputs.shape[-1])
    lookup = {tuple(row): index for index, row in enumerate(flat.tolist())}
    token_ids = torch.tensor([lookup[tuple(row)] for row in values["x"].tolist()])
    weights, indices = _cpu(route["weights"]), route["indices"].cpu()
    ordered_weights, start = [], 0
    for expert, end in enumerate(values["ends"]):
        tokens = token_ids[start:end]
        slots = (indices[tokens] == expert).to(torch.int64).argmax(-1)
        ordered_weights.append(weights[tokens, slots])
        start = end
    probs = torch.cat(ordered_weights)[:, None]
    expected_dy = _cpu(dy.to(torch.bfloat16)).reshape_as(flat)[token_ids] * probs
    combined = torch.zeros_like(flat).index_add(0, token_ids, values["down"] * probs)
    return {"dispatch_x_exact": torch.equal(values["x"], flat[token_ids]),
            "weighted_dy": _metric(values["dy"], expected_dy),
            "weighted_dy_rounded": _metric(values["dy"], expected_dy.bfloat16().float()),
            "weighted_combine": _metric(_cpu(routed_output).reshape_as(flat), combined),
            "weighted_combine_rounded": _metric(_cpu(routed_output).reshape_as(flat), combined.bfloat16().float())}


def _fused_gate(_module, gate_up):
    return torch_npu.npu_swiglu(gate_up)


def _step(source, candidate, args, generator, device, step):
    precision._synchronize_parameters(source, candidate, 0)
    fused_source = copy.deepcopy(source)
    fused_source.experts._apply_gate = MethodType(_fused_gate, fused_source.experts)
    inputs = torch.randn(1, args.tokens, args.hidden_size, generator=generator).to(device, torch.bfloat16)
    inputs.requires_grad_()
    other = inputs.detach().clone().requires_grad_()
    dy = torch.randn(inputs.shape, generator=generator).to(device) / math.sqrt(args.tokens)
    source.zero_grad(set_to_none=True)
    candidate.zero_grad(set_to_none=True)
    capture = _Capture()
    routed = {}
    hook = candidate.experts.register_forward_hook(lambda _m, _i, out: routed.update(output=out.detach().clone()))
    with patch.object(native, "_launch_forward_kernel", capture.capture_forward), patch.object(
            native, "_launch_backward_kernel", capture.capture_backward):
        expected, actual, expected_route, actual_route = precision._forward_pair(
            source, candidate, inputs, other, args, None)
        (expected.float() * dy).sum().backward()
        (actual.float() * dy).sum().backward()
    hook.remove()
    fused_inputs = inputs.detach().clone().requires_grad_()
    fused_route_weights = expected_route["weights"].detach().clone().requires_grad_()
    fused_output = fused_source.experts(fused_inputs.reshape(-1, args.hidden_size),
                                       expected_route["indices"], fused_route_weights).reshape_as(inputs)
    fused_output = fused_output + fused_source.shared_experts(fused_inputs)
    (fused_output.float() * dy).sum().backward()
    values = capture.values
    local = _local_references(values)
    hf_replay, fused_replay = _replay(values, device, fused=False), _replay(values, device, fused=True)
    result = {"step": step,
              "native_local_fp32": {name: _metric(values[name], ref) for name, ref in local.items()},
              "native_local_fp32_rounded": {name: _metric(values[name], ref.bfloat16().float())
                                            for name, ref in local.items()},
              "native_vs_hf_replay": {name: _metric(values[name], ref) for name, ref in hf_replay.items()},
              "native_vs_fused_replay": {name: _metric(values[name], ref) for name, ref in fused_replay.items()},
              "same_input_swiglu": _rounding_checks(values, device),
              "routing": _routing_checks(values, inputs, dy, actual_route, routed["output"]),
              "full_block_intervention": {}, "shared_gradients": {}}
    for name in ("gate_up_proj", "down_proj"):
        native_name = name.replace("gate_up_proj", "gate_up_weight").replace("down_proj", "down_weight")
        gradient = _cpu(getattr(candidate.experts, native_name).grad.transpose(1, 2))
        result["full_block_intervention"][name] = {
            "original_hf": _metric(gradient, _cpu(getattr(source.experts, name).grad)),
            "fused_hf": _metric(gradient, _cpu(getattr(fused_source.experts, name).grad))}
    for name, parameter in source.shared_experts.named_parameters():
        result["shared_gradients"][name] = _metric(
            _cpu(dict(candidate.shared_experts.named_parameters())[name].grad), _cpu(parameter.grad))
    # Preserve raw tensors for additional causal replays without rerunning NPU.
    torch.save(values, args.output.with_suffix(f".step{step}.pt"))
    torch.optim.SGD(source.parameters(), lr=0.01).step()
    return result


def main() -> None:
    """Run a synchronized EP1 push/pull diagnostic and write stage evidence."""
    args = precision._arguments(None)
    args.reference = "hf_replicated"
    if args.vision:
        raise ValueError("This EP1 operator probe currently uses text routing only")
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group("hccl")
    if dist.get_world_size() != 1 or args.ep_size not in (None, 1):
        raise ValueError("Operator scratch tracing currently requires WORLD=EP=1")
    device = torch.device("npu", torch.npu.current_device())
    source = precision._source_block(args, device)
    candidate = precision.DeepseekV41MegaMoe(copy.deepcopy(source), local_num_tokens=args.tokens,
                                           ep_group=dist.group.WORLD, dispatch_mode=args.dispatch_mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        generator = torch.Generator().manual_seed(4000)
        results = [_step(source, candidate, args, generator, device, step) for step in range(args.steps)]
        report = {"environment": precision._environment(), "config": vars(args) | {"output": str(args.output)},
                  "route": args.route,
                  "dispatch_mode": args.dispatch_mode, "steps": results,
                  "scope": "EP1 synchronized weights; actual native scratch; operator-local FP32 references; "
                           "NPU HF and fused SwiGLU replays; no performance claims"}
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Operator trace written to {args.output}", flush=True)
    finally:
        candidate.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

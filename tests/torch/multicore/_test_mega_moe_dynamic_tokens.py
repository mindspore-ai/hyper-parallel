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

"""Unequal and empty-token MegaMoe validation against an independent CPU FP32 oracle."""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu  # pylint: disable=unused-import
from torch.nn import functional
from torch.utils.checkpoint import checkpoint

from hyper_parallel.core.multicore import MegaMoeExperts
from tests.torch.multicore._evidence_helpers import EXTENDED_LEVEL1
from tests.torch.multicore._mega_moe_utils import start_shmem_lifetime, write_evidence

_HIDDEN = 512
_INTERMEDIATE = 128
_CAPACITY = 4096


def _weights(experts: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate exactly representable weights shared by the device and CPU oracle."""
    generator = torch.Generator().manual_seed(9023)
    first = torch.randn(experts, _HIDDEN, 2 * _INTERMEDIATE, generator=generator) / _HIDDEN**0.5
    second = torch.randn(experts, _INTERMEDIATE, _HIDDEN, generator=generator) / _INTERMEDIATE**0.5
    return first.bfloat16(), second.bfloat16()


def _inputs(lengths: list[int], experts: int, top_k: int, hot: bool = False) -> tuple:
    """Make deterministic original-token inputs without synthetic routes."""
    states, indices, probabilities, upstream = [], [], [], []
    for rank, length in enumerate(lengths):
        generator = torch.Generator().manual_seed(11000 + rank + length)
        states.append((torch.randn(length, _HIDDEN, generator=generator) * 0.5).bfloat16())
        ids = torch.arange(length * top_k).reshape(length, top_k)
        indices.append((ids % top_k if hot else (ids + rank * top_k) % experts).int())
        probabilities.append(torch.softmax(torch.randn(length, top_k, generator=generator), dim=-1))
        upstream.append((torch.randn(length, _HIDDEN, generator=generator) * 0.1).bfloat16())
    return states, indices, probabilities, upstream


def _fp32_oracle(weights: tuple, inputs: tuple) -> tuple:
    """Differentiate explicit per-expert FP32 matmuls over all real source tokens."""
    states, indices, probabilities, upstream = inputs
    x = torch.cat(states).float().requires_grad_()
    ids = torch.cat(indices).long()
    probs = torch.cat(probabilities).requires_grad_()
    w1, w2 = (value.float().requires_grad_() for value in weights)
    output = x * 0 + probs.sum() * 0
    if x.shape[0] == 0:
        output = output + (w1.sum() + w2.sum()) * 0
    for expert in range(w1.shape[0]):
        rows, slots = torch.where(ids == expert)
        if rows.numel() == 0:
            continue
        gate, up = (x[rows] @ w1[expert]).chunk(2, dim=-1)
        values = (functional.silu(gate) * up) @ w2[expert]
        output = output.index_add(0, rows, values * probs[rows, slots, None])
    (output * torch.cat(upstream).float()).sum().backward()
    return output.detach(), x.grad, probs.grad, w1.grad, w2.grad


def _rounded_oracle(weights: tuple, inputs: tuple) -> tuple:
    """Use independent FP32 matmuls with explicit kernel-boundary BF16 rounding."""
    states, indices, probabilities, upstream = inputs
    x, ids, probs, dy = (torch.cat(values) for values in (states, indices, probabilities, upstream))
    w1, w2 = weights
    output = torch.zeros_like(x, dtype=torch.float32)
    dx = torch.zeros_like(x, dtype=torch.float32)
    dp = torch.zeros_like(probs)
    dw1, dw2 = torch.zeros_like(w1), torch.zeros_like(w2)
    for expert in range(w1.shape[0]):
        rows, slots = torch.where(ids == expert)
        if rows.numel() == 0:
            continue
        source = x[rows].float()
        gate, up = (source @ w1[expert].float()).bfloat16().float().chunk(2, dim=-1)
        sigmoid = torch.sigmoid(gate)
        activation = (gate / (torch.exp(-gate) + 1) * up).bfloat16().float()
        value = (activation @ w2[expert].float()).bfloat16().float()
        output.index_add_(0, rows, value * probs[rows, slots, None])
        dp[rows, slots] = (value * dy[rows].float()).sum(-1)
        grad_value = (dy[rows].float() * probs[rows, slots, None]).bfloat16().float()
        dw2[expert] = (activation.T @ grad_value).bfloat16()
        grad_activation = (grad_value @ w2[expert].float().T).bfloat16().float()
        grad_gate = grad_activation * up * sigmoid * (1 + gate * (1 - sigmoid))
        grad_up = grad_activation * gate * sigmoid
        grad_projection = torch.cat((grad_gate, grad_up), dim=-1).bfloat16().float()
        dw1[expert] = (source.T @ grad_projection).bfloat16()
        dx.index_add_(0, rows, (grad_projection @ w1[expert].float().T).bfloat16().float())
    return output.bfloat16(), dx.bfloat16(), dp, dw1, dw2


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Apply the unchanged common-MoE BF16 threshold to the rounded reference."""
    actual = actual.detach().float().cpu()
    expected = expected.float()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3, msg=lambda message: f"{name}: {message}")
    return (actual - expected).abs().max().item() if actual.numel() else 0.0


def _forward(mega: MegaMoeExperts, inputs: tuple, rank: int, device: torch.device,
             input_grad: bool = True) -> tuple:
    """Preserve real shapes and leaf gradients at the model-facing API."""
    states, indices, probabilities, upstream = inputs
    x = states[rank].to(device).requires_grad_(input_grad)
    probs = probabilities[rank].to(device).requires_grad_()
    output = mega(x, indices[rank].to(device), probs)
    return x, probs, output, upstream[rank].to(device)


def _check_gradients(mega: MegaMoeExperts, invocation: tuple, expected: tuple,
                     lengths: list[int], rank: int) -> dict:
    """Compare local outputs/dX/router gradients and owner-aggregated expert gradients."""
    x, probs, output, upstream = invocation
    start = sum(lengths[:rank])
    stop = start + lengths[rank]
    local_experts = mega.local_experts
    expert_slice = slice(rank * local_experts, (rank + 1) * local_experts)
    errors = {"output": _assert_close("output", output, expected[0][start:stop])}
    (output * upstream).sum().backward()
    errors["input_grad"] = _assert_close("dX", x.grad, expected[1][start:stop])
    errors["route_grad"] = _assert_close("dTopK", probs.grad, expected[2][start:stop])
    errors["w1_grad"] = _assert_close("dW1", mega.gate_up_weight.grad, expected[3][expert_slice])
    errors["w2_grad"] = _assert_close("dW2", mega.down_weight.grad, expected[4][expert_slice])
    return errors


def _cases(world: int) -> list[tuple[list[int], bool]]:
    """Retain boundary, empty, skewed and revisited inputs; restore extra uniform lengths on demand."""
    sizes = (1, 127, 128, 129, 255, 256, 257, 385, 513, 641, 1, 4095) if EXTENDED_LEVEL1 else (
        1, 127, 128, 129, 255, 256, 257, 1,
    )
    cases = [([value] * world, False) for value in sizes]
    cases.extend([([224, 240, 1, 127, 128, 129, 0, 257][:world], False),
                  ([0] + [17] * (world - 1), True), ([0] * world, False)])
    if EXTENDED_LEVEL1:
        cases.append(([4096] + [1] * (world - 1), False))
    return cases + [([4096, 1, 0, 129, 0, 240, 4095, 128][:world], False)]


def _deferred(mega: MegaMoeExperts, weights: tuple, rank: int, device: torch.device) -> None:
    """Five pending forwards retain evicted plans through both backward orders."""
    world = dist.get_world_size()
    lengths = [[size + rank_index for rank_index in range(world)] for size in (1, 129, 257, 385, 513)]
    batches = [_inputs(item, mega.num_experts, mega.top_k) for item in lengths]
    references = [_rounded_oracle(weights, batch) for batch in batches]
    for order in (tuple(reversed(range(5))), tuple(range(5))):
        invocations = [_forward(mega, batch, rank, device) for batch in batches]
        try:
            mega.close()
        except RuntimeError as error:
            if "pending backward" not in str(error):
                raise
        else:
            raise AssertionError("close accepted outstanding backward graphs")
        for index in order:
            mega.zero_grad(set_to_none=True)
            _check_gradients(mega, invocations[index], references[index], lengths[index], rank)


def _pending_backward_across_restart(weights: tuple, rank: int, world: int, device: torch.device) -> None:
    """Complete a pending backward after a hot-route forward restarts its workspace."""
    if world == 1 or os.environ.get("HP_DYNAMIC_DISPATCH", "push") != "push":
        return
    experts, top_k = world * 6, min(world * 6, 6)
    module = MegaMoeExperts(
        hidden_size=_HIDDEN,
        intermediate_size=_INTERMEDIATE,
        num_experts=experts,
        top_k=top_k,
        ep_size=world,
        ep_group=dist.group.WORLD,
        initial_capacity_factor=1.0,
        dispatch_mode="push",
    ).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        module.gate_up_weight.copy_(weights[0][rank * 6:(rank + 1) * 6])
        module.down_weight.copy_(weights[1][rank * 6:(rank + 1) * 6])

    pending_lengths = [1] * world
    pending_inputs = _inputs(pending_lengths, experts, top_k)
    pending_expected = _rounded_oracle(weights, pending_inputs)
    pending_invocation = _forward(module, pending_inputs, rank, device)
    resources = module._resource_group.resources  # pylint: disable=protected-access
    epoch_before = resources.heap_manager.epoch

    growth_lengths = [129] * world
    growth_inputs = _inputs(growth_lengths, experts, top_k, hot=True)
    growth_invocation = _forward(module, growth_inputs, rank, device)
    if resources.heap_manager.epoch <= epoch_before:
        raise AssertionError("hot-route forward did not restart the SHMEM Heap")

    _, _, growth_output, growth_upstream = growth_invocation
    (growth_output * growth_upstream).sum().backward()
    module.zero_grad(set_to_none=True)
    _check_gradients(module, pending_invocation, pending_expected, pending_lengths, rank)
    module.close()


def _snapshot(mega: MegaMoeExperts, invocation: tuple) -> tuple:
    """Own outputs and every gradient before another invocation reuses resources."""
    x, probs, output, upstream = invocation
    (output * upstream).sum().backward()
    return tuple(tensor.detach().cpu().clone() for tensor in
                 (output, x.grad, probs.grad, mega.gate_up_weight.grad, mega.down_weight.grad))


def _padded_inputs(inputs: tuple) -> tuple:
    """Construct the old fixed-capacity baseline, including its zero-weight dummy routes."""
    batches = [[], [], [], []]
    for x, ids, probs, dy in zip(*inputs):
        missing = _CAPACITY - x.shape[0]
        batches[0].append(functional.pad(x, (0, 0, 0, missing)))
        batches[1].append(torch.cat((ids, torch.arange(ids.shape[1]).int().repeat(missing, 1))))
        batches[2].append(functional.pad(probs, (0, 0, 0, missing)))
        batches[3].append(functional.pad(dy, (0, 0, 0, missing)))
    return tuple(batches)


def _fixed_comparison(fixed: MegaMoeExperts, weights: tuple, inputs: tuple, actual: tuple,
                      lengths: list[int], rank: int, device: torch.device) -> dict:
    """Reuse the validated dynamic result for padding parity and raw FP32 diagnostics."""
    fixed.zero_grad(set_to_none=True)
    padded = _snapshot(fixed, _forward(fixed, _padded_inputs(inputs), rank, device))
    baseline = tuple(value[:lengths[rank]] if index < 3 else value for index, value in enumerate(padded))
    for index, (value, expected) in enumerate(zip(actual, baseline)):
        _assert_close(f"fixed path tensor {index}", value, expected)
    oracle = _fp32_oracle(weights, inputs)
    start, stop = sum(lengths[:rank]), sum(lengths[:rank + 1])
    oracle = tuple(value[start:stop] if index < 3 else value[rank * 6:(rank + 1) * 6]
                   for index, value in enumerate(oracle))
    report = {"lengths": lengths, "fp32": []}
    for value, old, expected in zip(actual, baseline, oracle):
        delta = (value.float() - expected).abs()
        old_delta = (old.float() - expected).abs()
        report["fp32"].append({
            "dynamic_bad_elements": int((delta > 2e-3 + 2e-2 * expected.abs()).sum()),
            "fixed_bad_elements": int((old_delta > 2e-3 + 2e-2 * expected.abs()).sum()),
            "dynamic_relative_l2": float(delta.norm() / expected.norm().clamp_min(1e-30)),
            "fixed_relative_l2": float(old_delta.norm() / expected.norm().clamp_min(1e-30)),
        })
    return report


def _lifecycle_forward(layers: list[MegaMoeExperts], rank: int, device: torch.device,
                       replay: bool) -> tuple:
    """Submit the same shared-layer forwards with optional checkpoint replay."""
    world = dist.get_world_size()
    streams = [torch.npu.Stream() for _ in layers]
    default_stream = torch.npu.current_stream()
    outputs, leaves = [], []
    for size in (1, 129):
        inputs = _inputs([size + index for index in range(world)], layers[0].num_experts, layers[0].top_k)
        x = inputs[0][rank].to(device).requires_grad_()
        ids = inputs[1][rank].to(device)
        probs = [inputs[2][rank].to(device).requires_grad_() for _ in layers]
        dy = inputs[3][rank].to(device)
        value, previous = x, default_stream
        for index, layer in enumerate(layers):
            with torch.npu.stream(streams[index]):
                streams[index].wait_stream(previous)
                if replay:
                    value = checkpoint(layer, value, ids, probs[index], use_reentrant=False)
                else:
                    value = layer(value, ids, probs[index])
            previous = streams[index]
        default_stream.wait_stream(previous)
        outputs.append((value, dy))
        leaves.extend([x, *probs])
    return outputs, leaves


def _lifecycle(layers: list[MegaMoeExperts], rank: int, device: torch.device) -> None:
    """Compare shared-layer checkpoint and cross-stream gradients and an SGD update."""
    references = None
    for replay in (False, True):
        for layer in layers:
            layer.zero_grad(set_to_none=True)
        outputs, leaves = _lifecycle_forward(layers, rank, device, replay)
        for value, dy in reversed(outputs):
            (value * dy).sum().backward()
        torch.npu.synchronize()
        tensors = [value for value, _ in outputs] + [leaf.grad for leaf in leaves]
        parameters = [parameter for layer in layers for parameter in layer.parameters()]
        tensors.extend(parameter.grad for parameter in parameters)
        observed = [value.detach().cpu().clone() for value in tensors]
        if references is None:
            references = observed
        else:
            for index, (actual, expected) in enumerate(zip(observed, references)):
                _assert_close(f"checkpoint tensor {index}", actual, expected)
            expected_updates = [parameter.detach().cpu().add(gradient, alpha=-1e-3)
                                for parameter, gradient in zip(parameters, references[-len(parameters):])]
            torch.optim.SGD(parameters, lr=1e-3).step()
            for actual, expected in zip(parameters, expected_updates):
                _assert_close("SGD update", actual, expected)


def _configuration_failures(world: int, rank: int, device: torch.device) -> None:
    """Configuration mismatches must fail on every rank before SHMEM."""
    if world == 1:
        return
    for different_split in (False, True):
        options = ({"dispatch_split": 128 if rank == 0 else 256} if different_split
                   else {"swiglu_limit": None if rank == 0 else 10.0})
        module = MegaMoeExperts(**options, hidden_size=_HIDDEN, intermediate_size=_INTERMEDIATE,
                               num_experts=world * 6, top_k=6, ep_size=world,
                               ep_group=dist.group.WORLD,
                               dispatch_mode=os.environ.get("HP_DYNAMIC_DISPATCH", "push"),
                               ).to(device=device, dtype=torch.bfloat16)
        lengths = [128] * world
        try:
            _forward(module, _inputs(lengths, world * 6, 6), rank, device)
        except (RuntimeError, ValueError) as error:
            messages = ("specifications differ", "configurations differ", "resource manifests differ")
            if not any(message in str(error) for message in messages):
                raise
        else:
            raise AssertionError("accepted inconsistent resource configurations")
        module.close()


def _negative_routes(mega: MegaMoeExperts, hotspot: MegaMoeExperts, rank: int, device: torch.device) -> None:
    """Grow lossless hotspot capacity and reject inconsistent backward participation."""
    world = dist.get_world_size()
    if world == 1:
        return
    invocation = _forward(hotspot, _inputs([128] * world, mega.num_experts, mega.top_k, hot=True), rank, device)
    invocation[2].sum().backward()
    hotspot.zero_grad(set_to_none=True)
    inputs = _inputs([1] * world, mega.num_experts, mega.top_k)
    x = inputs[0][rank].to(device).requires_grad_(rank != 0)
    try:
        mega(x, inputs[1][rank].to(device), inputs[2][rank].to(device))
    except RuntimeError as error:
        if "autograd participation" not in str(error):
            raise
    else:
        raise AssertionError("accepted inconsistent backward participation")


def _frozen_gradients(mega: MegaMoeExperts, weights: tuple, rank: int, device: torch.device) -> None:
    """Exercise each consistent frozen-input/expert policy, including a zero-token rank."""
    world = dist.get_world_size()
    lengths = [17] if world == 1 else [0] + [17] * (world - 1)
    inputs = _inputs(lengths, mega.num_experts, mega.top_k)
    expected = _rounded_oracle(weights, inputs)
    start, stop = sum(lengths[:rank]), sum(lengths[:rank + 1])
    for input_grad, expert_grad in ((False, True), (True, False), (False, False)):
        mega.zero_grad(set_to_none=True)
        mega.requires_grad_(expert_grad)
        x, probs, output, dy = _forward(mega, inputs, rank, device, input_grad)
        _assert_close("frozen output", output, expected[0][start:stop])
        (output * dy).sum().backward()
        _assert_close("frozen dTopK", probs.grad, expected[2][start:stop])
        if input_grad:
            _assert_close("frozen dX", x.grad, expected[1][start:stop])
        elif x.grad is not None:
            raise AssertionError("frozen input received a gradient")
        for index, parameter in enumerate((mega.gate_up_weight, mega.down_weight)):
            if expert_grad:
                _assert_close("frozen dW", parameter.grad, expected[index + 3][rank * 6:(rank + 1) * 6])
            elif parameter.grad is not None:
                raise AssertionError("frozen expert received a gradient")
    mega.requires_grad_(True)


def _layer(mode: str, weights: tuple, rank: int, world: int, device: torch.device) -> MegaMoeExperts:
    """Construct an automatic-token module without predicting input lengths."""
    layer = MegaMoeExperts(hidden_size=_HIDDEN, intermediate_size=_INTERMEDIATE,
                           num_experts=world * 6, top_k=6, ep_size=world,
                           ep_group=dist.group.WORLD, dispatch_mode=mode).to(
                               device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        layer.gate_up_weight.copy_(weights[0][rank * 6:(rank + 1) * 6])
        layer.down_weight.copy_(weights[1][rank * 6:(rank + 1) * 6])
    return layer


def _state(layer: MegaMoeExperts) -> dict:
    """Read physical allocation sizes without synchronizing tensor contents."""
    resources = layer._resource_group.resources  # pylint: disable=protected-access
    workspace = resources.workspace
    return {"routed_rows": workspace.routed_slots, "receive_rows": workspace.expert_capacity,
            "heap_bytes": resources.heap_manager.heap_bytes, "epoch": resources.heap_manager.epoch}


def _verify(layer: MegaMoeExperts, lengths: list[int], weights: tuple, rank: int,
            device: torch.device, hot: bool = False) -> dict:
    """Check the complete local output and gradient contract against the independent oracle."""
    layer.zero_grad(set_to_none=True)
    inputs = _inputs(lengths, layer.num_experts, layer.top_k, hot=hot)
    expected = _rounded_oracle(weights, inputs)
    errors = _check_gradients(layer, _forward(layer, inputs, rank, device), expected, lengths, rank)
    return {"lengths": lengths, "hot": hot, "max_abs_error": errors, "capacity": _state(layer)}


def _pending_growth(layer: MegaMoeExperts, weights: tuple, rank: int, world: int,
                    device: torch.device) -> dict:
    """Grow source storage while an earlier forward still needs backward."""
    lengths = [1] * world
    inputs = _inputs(lengths, layer.num_experts, layer.top_k)
    pending = _forward(layer, inputs, rank, device)
    before = _state(layer)
    result = _verify(layer, [257] * world, weights, rank, device, hot=True)
    after = _state(layer)
    if after["epoch"] <= before["epoch"] or after["routed_rows"] <= before["routed_rows"]:
        raise AssertionError("larger input did not grow the source allocation")
    layer.zero_grad(set_to_none=True)
    _check_gradients(layer, pending, _rounded_oracle(weights, inputs), lengths, rank)
    return result


def _pending_bucket_growth(layer: MegaMoeExperts, weights: tuple, rank: int, world: int,
                           device: torch.device) -> dict:
    """Replace a partial bucket while backward still retains its previous plan."""
    lengths = [257] * world
    inputs = _inputs(lengths, layer.num_experts, layer.top_k)
    pending = _forward(layer, inputs, rank, device)
    resources = layer._resource_group.resources  # pylint: disable=protected-access
    retained = resources._plans[512]  # pylint: disable=protected-access
    coverage = retained.spec.plan_tokens
    result = _verify(layer, [385] * world, weights, rank, device)
    expanded = resources._plans[512]  # pylint: disable=protected-access
    if expanded is retained or retained.spec.plan_tokens != coverage or expanded.spec.plan_tokens < 512:
        raise AssertionError(f"partial bucket was not replaced immutably: old={coverage}, "
                             f"new={expanded.spec.plan_tokens}")
    layer.zero_grad(set_to_none=True)
    _check_gradients(layer, pending, _rounded_oracle(weights, inputs), lengths, rank)
    return result


def _bucket_reuse(layer: MegaMoeExperts, weights: tuple, rank: int, world: int,
                  device: torch.device) -> dict:
    """Cycle through more than four buckets without rebuilding stable plans or storage."""
    lengths = (0, 1, 129, 257, 513, 1025, 2049, 4097)
    for tokens in lengths:
        _verify(layer, [tokens] * world, weights, rank, device)
    resources = layer._resource_group.resources  # pylint: disable=protected-access
    plans = dict(resources._plans)  # pylint: disable=protected-access
    epoch = _state(layer)["epoch"]
    if len(plans) < 7:
        raise AssertionError(f"wide token cycle did not retain seven buckets: cached={list(plans)}")
    errors = []
    for tokens in lengths:
        errors.append(_verify(layer, [tokens] * world, weights, rank, device)["max_abs_error"])
        if _state(layer)["epoch"] != epoch:
            raise AssertionError(f"warm token cycle restarted the heap: expected_epoch={epoch}")
        if any(resources._plans.get(bucket) is not plan for bucket, plan in plans.items()):  # pylint: disable=protected-access
            raise AssertionError(f"warm token cycle replaced cached plans: buckets={list(plans)}")
    return {"buckets": sorted(plans), "epoch": epoch, "max_abs_errors": errors}


def _other_workspace_growth(layer: MegaMoeExperts, weights: tuple, rank: int, world: int,
                            device: torch.device) -> None:
    """Rebuild a root containing both transports without invalidating pending graphs."""
    lengths = [17] * world
    inputs = _inputs(lengths, layer.num_experts, layer.top_k)
    pending = _forward(layer, inputs, rank, device)
    other = _layer("pull" if layer.dispatch_mode == "push" else "push", weights, rank, world, device)
    _verify(other, [769] * world, weights, rank, device)
    layer.zero_grad(set_to_none=True)
    _check_gradients(layer, pending, _rounded_oracle(weights, inputs), lengths, rank)
    other.close()


def _run_token_matrix(mega: MegaMoeExperts, fixed: MegaMoeExperts, weights: tuple,
                      rank: int, world: int, device: torch.device) -> tuple:
    """Check real-shape gradients, monotonic capacity and the padded FP32 diagnostic baseline."""
    results = []
    fixed_reports = []
    reference_lengths = [[224, 240, 1, 127, 128, 129, 0, 257][:world]]
    if EXTENDED_LEVEL1:
        reference_lengths += [[127] * world, [4096] + [1] * (world - 1)]
    cases = _cases(world) + [([129 + index for index in range(world)], False),
                            ([769] + [0] * (world - 1), True), ([4097] + [1] * (world - 1), False)]
    for lengths, hot in cases:
        inputs = _inputs(lengths, mega.num_experts, mega.top_k, hot)
        expected = _rounded_oracle(weights, inputs)
        mega.zero_grad(set_to_none=True)
        previous = _state(mega)
        resources = mega._resource_group.resources  # pylint: disable=protected-access
        workspace = resources.workspace
        if workspace.expert_buffer is not None:
            workspace.wait_for_reuse()
            workspace.expert_buffer.fill_(float("nan"))
            workspace.routed_buffer.fill_(float("nan"))
        invocation = _forward(mega, inputs, rank, device)
        errors = _check_gradients(mega, invocation, expected, lengths, rank)
        current = _state(mega)
        required = max(lengths) * mega.top_k
        if current["routed_rows"] < max(required, previous["routed_rows"]):
            raise AssertionError("source capacity is insufficient or shrank")
        if (required <= previous["routed_rows"] and not hot and max(lengths) <= 1
                and current["epoch"] != previous["epoch"]):
            raise AssertionError("small input unnecessarily restarted the heap")
        if lengths in reference_lengths:
            x, probs, output, _ = invocation
            actual = tuple(value.detach().cpu().clone() for value in
                           (output, x.grad, probs.grad, mega.gate_up_weight.grad, mega.down_weight.grad))
            fixed_reports.append(_fixed_comparison(fixed, weights, inputs, actual, lengths, rank, device))
        results.append({"lengths": lengths, "hot": hot, "max_abs_error": errors, "capacity": current})
        print(json.dumps({"rank": rank, **results[-1]}), flush=True)
    return results, fixed_reports


def test_dynamic_tokens() -> None:
    """Validate automatic lengths, growth, numerical oracles and retained graphs in one EP worker."""
    torch.set_num_threads(2)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("npu", local_rank)
    mode = os.environ.get("HP_DYNAMIC_DISPATCH", "push")
    _configuration_failures(world, rank, device)
    start_shmem_lifetime()
    weights = _weights(world * 6)
    _pending_backward_across_restart(weights, rank, world, device)
    mega = _layer(mode, weights, rank, world, device)
    peer = _layer(mode, weights, rank, world, device)
    fixed = _layer(mode, weights, rank, world, device)
    hotspot = _layer(mode, weights, rank, world, device)
    MegaMoeExperts.share_execution_resources([mega, peer])
    results = [_verify(mega, [0] * world, weights, rank, device)]
    if _state(mega)["routed_rows"] != 128 * mega.top_k:
        raise AssertionError("empty first input reserved more than the bootstrap capacity")
    results.append(_pending_growth(mega, weights, rank, world, device))
    results.append(_pending_bucket_growth(mega, weights, rank, world, device))
    _other_workspace_growth(mega, weights, rank, world, device)
    _negative_routes(mega, hotspot, rank, device)
    results.append(_verify(hotspot, [17] * world, weights, rank, device))
    matrix_results, fixed_reports = _run_token_matrix(mega, fixed, weights, rank, world, device)
    results.extend(matrix_results)
    results.append(_verify(peer, [127] * world, weights, rank, device))
    if mega._resource_group.resources is not peer._resource_group.resources:  # pylint: disable=protected-access
        raise AssertionError("shared second layer acquired independent execution resources")
    _frozen_gradients(mega, weights, rank, device)
    _lifecycle([mega, peer], rank, device)
    mega.load_state_dict(fixed.state_dict())
    peer.close()
    _deferred(mega, weights, rank, device)
    bucket_reuse = _bucket_reuse(mega, weights, rank, world, device)
    final_state = _state(mega)
    for layer in (mega, fixed, hotspot):
        layer.close()
    result = {"status": "passed", "results": results, "deferred_backward": "passed",
              "fixed_reference": fixed_reports,
              "checkpoint_shared_stream_sgd": "passed", "frozen_policies": "passed",
              "source_growth_pending_backward": "passed", "mixed_transport_restart": "passed",
              "same_bucket_growth_pending_backward": "passed", "geometric_plan_reuse": bucket_reuse,
              "final_capacity": final_state, "torch": torch.__version__, "torch_npu": torch_npu.__version__,
              "dispatch_mode": mode}
    if os.getenv("HP_MEGA_MOE_LEVEL1_RESULT"):
        write_evidence(result)
    elif os.getenv("HP_DYNAMIC_RESULT_DIR"):
        path = Path(os.environ["HP_DYNAMIC_RESULT_DIR"])
        path.mkdir(parents=True, exist_ok=True)
        (path / f"ep{world}-rank{rank}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    test_dynamic_tokens()

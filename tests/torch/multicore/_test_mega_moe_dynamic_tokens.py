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
from torch.nn import functional
from torch.utils.checkpoint import checkpoint
import torch_npu  # pylint: disable=unused-import

from hyper_parallel.core.multicore import MegaMoeExperts
from tests.torch.multicore._mega_moe_utils import start_shmem_lifetime


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
    """Exercise boundary tiles, cache misses, emptiness and maximal skew."""
    cases = [([value] * world, False) for value in (1, 127, 128, 129, 255, 256, 257, 385, 513, 641, 1, 4095)]
    ragged = [224, 240, 1, 127, 128, 129, 0, 257]
    cases.extend([(ragged[:world], False), ([0] + [17] * (world - 1), True),
                  ([0] * world, False), ([4096] + [1] * (world - 1), False),
                  ([4096, 1, 0, 129, 0, 240, 4095, 128][:world], False)])
    return cases


def _deferred(mega: MegaMoeExperts, weights: tuple, rank: int, device: torch.device) -> None:
    """Five pending forwards retain evicted plans through both backward orders."""
    world = dist.get_world_size()
    for order in (tuple(reversed(range(5))), tuple(range(5))):
        lengths = [[size + rank_index for rank_index in range(world)] for size in (1, 129, 257, 385, 513)]
        batches = [_inputs(item, mega.num_experts, mega.top_k) for item in lengths]
        references = [_rounded_oracle(weights, batch) for batch in batches]
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


def _fixed_comparison(mega: MegaMoeExperts, fixed: MegaMoeExperts, weights: tuple,
                      rank: int, device: torch.device) -> list:
    """Compare real-token execution to the existing padded path and report raw FP32 error."""
    reports = []
    world = dist.get_world_size()
    for lengths in ([127] * world, [224, 240, 1, 127, 128, 129, 0, 257][:world],
                    [4096] + [1] * (world - 1)):
        inputs = _inputs(lengths, mega.num_experts, mega.top_k)
        mega.zero_grad(set_to_none=True)
        fixed.zero_grad(set_to_none=True)
        actual = _snapshot(mega, _forward(mega, inputs, rank, device))
        padded = _snapshot(fixed, _forward(fixed, _padded_inputs(inputs), rank, device))
        baseline = tuple(value[:lengths[rank]] if index < 3 else value
                         for index, value in enumerate(padded))
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
        reports.append(report)
    return reports


def _lifecycle(layers: list[MegaMoeExperts], rank: int, device: torch.device) -> None:
    """Compare shared-layer checkpoint and cross-stream gradients and an SGD update."""
    world = dist.get_world_size()
    references = None
    for replay in (False, True):
        for layer in layers:
            layer.zero_grad(set_to_none=True)
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
    """Configuration mismatches must fail on both static and dynamic ranks before SHMEM."""
    if world == 1:
        return
    for mixed_modes in (False, True):
        options = ({"local_num_tokens": 128} if mixed_modes and rank == 0
                   else {"max_local_num_tokens": 128 if rank == 0 else 127})
        module = MegaMoeExperts(**options, hidden_size=_HIDDEN, intermediate_size=_INTERMEDIATE,
                               num_experts=world * 6, top_k=6, ep_size=world,
                               ep_group=dist.group.WORLD,
                               dispatch_mode=os.environ.get("HP_DYNAMIC_DISPATCH", "push"),
                               ).to(device=device, dtype=torch.bfloat16)
        lengths = [128] * world if mixed_modes else [1] * world
        # The dynamic peer's over-limit input still reaches the configuration check first.
        try:
            _forward(module, _inputs(lengths, world * 6, 6), rank, device)
        except (RuntimeError, ValueError) as error:
            messages = ("specifications differ", "configurations differ", "resource manifests differ")
            if not any(message in str(error) for message in messages):
                raise
        else:
            raise AssertionError("accepted inconsistent resource configurations")
        module.close()


def _negative_routes(mega: MegaMoeExperts, bounded: MegaMoeExperts, rank: int, device: torch.device) -> None:
    """Grow lossless hotspot capacity and reject inconsistent backward participation."""
    world = dist.get_world_size()
    if world == 1:
        return
    invocation = _forward(bounded, _inputs([128] * world, mega.num_experts, mega.top_k, hot=True), rank, device)
    invocation[2].sum().backward()
    bounded.zero_grad(set_to_none=True)
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


def test_dynamic_tokens() -> None:
    """Run the dynamic-length EP contract and full-gradient precision matrix."""
    torch.set_num_threads(2)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("npu", local_rank)
    _configuration_failures(world, rank, device)
    start_shmem_lifetime()
    experts, top_k = world * 6, min(world * 6, 6)
    weights = _weights(experts)
    mega = MegaMoeExperts(max_local_num_tokens=_CAPACITY, hidden_size=_HIDDEN,
                          intermediate_size=_INTERMEDIATE, num_experts=experts,
                          top_k=top_k, ep_size=world,
                          ep_group=dist.group.WORLD,
                               dispatch_mode=os.environ.get("HP_DYNAMIC_DISPATCH", "push"),
                               ).to(device=device, dtype=torch.bfloat16)
    fixed = MegaMoeExperts(local_num_tokens=_CAPACITY, hidden_size=_HIDDEN,
                           intermediate_size=_INTERMEDIATE, num_experts=experts,
                           top_k=top_k, ep_size=world,
                          ep_group=dist.group.WORLD,
                               dispatch_mode=os.environ.get("HP_DYNAMIC_DISPATCH", "push"),
                               ).to(device=device, dtype=torch.bfloat16)
    peer = MegaMoeExperts(max_local_num_tokens=_CAPACITY, hidden_size=_HIDDEN,
                          intermediate_size=_INTERMEDIATE, num_experts=experts,
                          top_k=top_k, ep_size=world,
                          ep_group=dist.group.WORLD,
                               dispatch_mode=os.environ.get("HP_DYNAMIC_DISPATCH", "push"),
                               ).to(device=device, dtype=torch.bfloat16)
    bounded = MegaMoeExperts(max_local_num_tokens=128, hidden_size=_HIDDEN,
                            intermediate_size=_INTERMEDIATE, num_experts=experts,
                            top_k=top_k, ep_size=world,
                            ep_group=dist.group.WORLD,
                               dispatch_mode=os.environ.get("HP_DYNAMIC_DISPATCH", "push"),
                               ).to(device=device, dtype=torch.bfloat16)
    MegaMoeExperts.share_execution_resources([mega, peer])
    with torch.no_grad():
        mega.gate_up_weight.copy_(weights[0][rank * 6:(rank + 1) * 6])
        mega.down_weight.copy_(weights[1][rank * 6:(rank + 1) * 6])
    fixed.load_state_dict(mega.state_dict())
    peer.load_state_dict(mega.state_dict())
    bounded.load_state_dict(mega.state_dict())
    _negative_routes(mega, bounded, rank, device)
    bounded_inputs = _inputs([17] * world, experts, top_k)
    _check_gradients(bounded, _forward(bounded, bounded_inputs, rank, device),
                     _rounded_oracle(weights, bounded_inputs), [17] * world, rank)
    results = []
    for lengths, hot in _cases(world):
        inputs = _inputs(lengths, experts, top_k, hot)
        expected = _rounded_oracle(weights, inputs)
        mega.zero_grad(set_to_none=True)
        # Poison private buffers to detect stale data across dynamic-token calls.
        resources = mega._resource_group.resources  # pylint: disable=protected-access
        if resources is not None:
            workspace = resources.workspace
            if workspace.expert_buffer is not None:
                workspace.wait_for_reuse()
                workspace.expert_buffer.fill_(float("nan"))
                workspace.routed_buffer.fill_(float("nan"))
        invocation = _forward(mega, inputs, rank, device)
        errors = _check_gradients(mega, invocation, expected, lengths, rank)
        results.append({"lengths": lengths, "hot": hot, "max_abs_error": errors})
        print(json.dumps({"rank": rank, **results[-1]}), flush=True)
    _frozen_gradients(mega, weights, rank, device)
    fixed_reports = _fixed_comparison(mega, fixed, weights, rank, device)
    _lifecycle([mega, peer], rank, device)
    mega.load_state_dict(fixed.state_dict())
    # Retire the other member so closing the last owner checks pending graphs.
    peer.close()
    _deferred(mega, weights, rank, device)
    too_long = [4097] + [0] * (world - 1)
    try:
        _forward(mega, _inputs(too_long, experts, top_k), rank, device)
    except RuntimeError as error:
        if "token capacity overflow" not in str(error):
            raise
    else:
        raise AssertionError("accepted an over-capacity source")
    mega.close()
    peer.close()
    fixed.close()
    bounded.close()
    result_dir = os.getenv("HP_DYNAMIC_RESULT_DIR")
    if result_dir:
        path = Path(result_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"ep{world}-rank{rank}.json").write_text(json.dumps(
            {"status": "passed", "results": results, "deferred_backward": "passed",
             "coordinated_overflow": "passed", "fixed_reference": fixed_reports,
             "checkpoint_shared_stream_sgd": "passed", "frozen_policies": "passed",
             "torch": torch.__version__, "torch_npu": torch_npu.__version__,
             "dispatch_mode": os.environ.get("HP_DYNAMIC_DISPATCH", "push")},
            indent=2) + "\n", encoding="utf-8")
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    test_dynamic_tokens()

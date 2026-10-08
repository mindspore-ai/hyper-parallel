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

"""Compare declarative DSV4.1 experts with direct fixed-capacity MegaMoe at limit 10."""

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional
import torch_npu  # pylint: disable=unused-import

from hyper_parallel.core.multicore import MegaMoeExperts
from hyper_parallel.models.deepseek_v41.adapter.distributed.megamoe import DeepseekV41MegaMoeExperts
from tests.torch.multicore._mega_moe_utils import start_shmem_lifetime

_HIDDEN = 512
_INTERMEDIATE = 128


def _weights(experts: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate exactly representable weights shared by the device and CPU oracle."""
    generator = torch.Generator().manual_seed(9023)
    first = (torch.randn(experts, _HIDDEN, 2 * _INTERMEDIATE, generator=generator) / _HIDDEN**0.5)
    second = (torch.randn(experts, _INTERMEDIATE, _HIDDEN, generator=generator) / _INTERMEDIATE**0.5)
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


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Apply the unchanged common-MoE BF16 threshold to the rounded reference."""
    actual = actual.detach().float().cpu()
    expected = expected.float()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3, msg=lambda message: f"{name}: {message}")
    return (actual - expected).abs().max().item() if actual.numel() else 0.0


def _oracle(weights: tuple, inputs: tuple) -> tuple:
    """Compute independent FP32 clipped experts; report rounding error separately."""
    states, indices, probabilities, upstream = inputs
    x = torch.cat(states).float().requires_grad_()
    ids = torch.cat(indices).long()
    probs = torch.cat(probabilities).requires_grad_()
    first, second = (value.float().requires_grad_() for value in weights)
    output = x * 0 + (first.sum() + second.sum() + probs.sum()) * 0
    clipped = 0
    for expert in range(first.shape[0]):
        rows, slots = torch.where(ids == expert)
        gate, up = (x[rows] @ first[expert]).chunk(2, -1)
        clipped += int(((gate > 10) | (up.abs() > 10)).sum())
        activation = functional.silu(gate.clamp(max=10)) * up.clamp(-10, 10)
        output = output.index_add(0, rows, (activation @ second[expert]) * probs[rows, slots, None])
    (output * torch.cat(upstream).float()).sum().backward()
    return (output.detach(), x.grad, probs.grad, first.grad, second.grad), clipped


def _snapshot(module: nn.Module, inputs: tuple, rank: int, *, padded: bool = False) -> tuple:
    """Own output and every gradient before the next workspace reuse."""
    states, indices, probabilities, upstream = inputs
    x = states[rank].npu().requires_grad_()
    ids = indices[rank].npu()
    probs = probabilities[rank].npu().requires_grad_()
    module.zero_grad(set_to_none=True)
    if padded:
        missing = 256 - x.shape[0]
        dummy_ids = torch.arange(missing * 6, device=x.device, dtype=ids.dtype).remainder(48).reshape(missing, 6)
        output = module(functional.pad(x, (0, 0, 0, missing)), torch.cat((ids, dummy_ids)),
                        functional.pad(probs, (0, 0, 0, missing)))[:x.shape[0]]
        first, second = module.gate_up_weight, module.down_weight
    else:
        output = module(x, ids, probs)
        first, second = module.gate_up_proj, module.down_proj
    (output.float() * upstream[rank].npu().float()).sum().backward()
    return tuple(value.detach().cpu().clone() for value in (output, x.grad, probs.grad, first.grad, second.grad))


def test_deepseek_v41_megamoe() -> None:
    """Check unequal and empty source ranks, saturated activation, and external-weight gradients."""
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    if world != 8:
        raise ValueError("This DSV4.1 crop requires EP8 and E48")
    start_shmem_lifetime()
    weights = _weights(48)
    source = nn.Module()
    source.num_experts, source.hidden_dim, source.intermediate_dim = 48, 512, 128
    source.limit, source.act_fn = 10.0, nn.SiLU()
    source.gate_up_proj = nn.Parameter(weights[0].transpose(1, 2).contiguous())
    source.down_proj = nn.Parameter(weights[1].transpose(1, 2).contiguous())
    adapted = DeepseekV41MegaMoeExperts(module=source, local_num_tokens=256, expert_capacity_factor=None)
    experts = slice(rank * 6, (rank + 1) * 6)
    adapted.gate_up_proj = nn.Parameter(adapted.gate_up_proj[experts].detach().npu())
    adapted.down_proj = nn.Parameter(adapted.down_proj[experts].detach().npu())
    adapted.configure(dist.group.WORLD, world, 6)
    fixed = MegaMoeExperts(local_num_tokens=256, hidden_size=512, intermediate_size=128,
                           num_experts=48, top_k=6, swiglu_limit=10.0, ep_size=world,
                           ep_group=dist.group.WORLD, expert_capacity_factor=None).npu().bfloat16()
    with torch.no_grad():
        fixed.gate_up_weight.copy_(weights[0][experts])
        fixed.down_weight.copy_(weights[1][experts])
    reports = []
    cases = ([224, 240] * 4, [0, 17, 17, 17, 17, 17, 17, 17], [0] * 8, [256] * 8)
    for lengths in cases:
        inputs = _inputs(lengths, 48, 6)
        # Saturation must occur on real routes so disabling the clamp cannot pass unnoticed.
        inputs = ([value * 16 for value in inputs[0]], *inputs[1:])
        actual = _snapshot(adapted, inputs, rank)
        baseline = _snapshot(fixed, inputs, rank, padded=True)
        errors = [_assert_close(f"padded reference tensor {index}", value, expected)
                  for index, (value, expected) in enumerate(zip(actual, baseline))]
        reference, clipped = _oracle(weights, inputs)
        if sum(lengths):
            assert clipped > 0
        rows = slice(sum(lengths[:rank]), sum(lengths[:rank + 1]))
        fp32 = []
        for index, (value, old, expected) in enumerate(zip(actual, baseline, reference)):
            expected = expected[rows if index < 3 else experts]
            scale = expected.norm().clamp_min(1e-30)
            fp32.append({"adapted_relative_l2": float((value.float() - expected).norm() / scale),
                         "fixed_relative_l2": float((old.float() - expected).norm() / scale)})
        reports.append({"lengths": lengths, "clipped_elements": clipped,
                        "fixed_comparison_max_abs": errors, "fp32_diagnostic": fp32})
    adapted.close_runtime()
    fixed.close()
    result = {"rank": rank, "limit": 10, "ep": world, "experts": 48,
              "passed": True, "cases": reports, "rtol": 2e-2, "atol": 2e-3}
    if os.getenv("HP_DSV41_MEGAMOE_RESULT_DIR"):
        output = Path(os.environ["HP_DSV41_MEGAMOE_RESULT_DIR"])
        output.mkdir(parents=True, exist_ok=True)
        (output / f"block-rank{rank}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    test_deepseek_v41_megamoe()

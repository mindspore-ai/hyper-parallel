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
"""Independent NPU PIK correctness and graph acceptance; no model dependencies."""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_func
import torch_npu

from rl.unified_execution.components.row_parallel import PikLinear
from rl.unified_execution.ops.pik import PikPlan, pik_linear, prepare_pik
from rl.unified_execution.ops.pik.reduction import fixed_tree


def reference(x: torch.Tensor, weight: torch.Tensor, leaves: int,
              bias: torch.Tensor | None = None) -> torch.Tensor:
    """Independent sequential submissions and explicit adjacent-pair FP32 tree."""
    flat = x.reshape(-1, x.shape[-1])
    width = weight.shape[1] // leaves
    nodes = [torch.bmm(flat[:, i * width:(i + 1) * width].unsqueeze(0),
                       weight[:, i * width:(i + 1) * width].T.unsqueeze(0))[0].float()
             for i in range(leaves)]
    while len(nodes) > 1:
        nodes = [nodes[i] + nodes[i + 1] for i in range(0, len(nodes), 2)]
    result = nodes[0].to(torch.bfloat16)
    if bias is not None:
        result = result + bias.to(torch.bfloat16)
    return result.reshape(*x.shape[:-1], weight.shape[0])


def _assert_bits(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Check raw BF16/FP32 representations, including signed zeros."""
    integer = torch.int16 if actual.dtype == torch.bfloat16 else torch.int32
    torch.testing.assert_close(actual.contiguous().view(integer), expected.contiguous().view(integer), rtol=0, atol=0)


def _run_case(shape, n, leaves, layout, transport, sp, axis, rank, world):
    """Compare PIK outputs and gradients for one device configuration."""
    torch.manual_seed(271)
    full_x = torch.randn(*shape, device="npu", dtype=torch.bfloat16)
    full_w = torch.randn(n, shape[-1], device="npu", dtype=torch.bfloat16)
    bias = torch.randn(n, device="npu", dtype=torch.float32)
    if layout == "packed":
        bias = bias.bfloat16()
    width = shape[-1] // world
    # Slice views intentionally exercise noncontiguous local input and weight storage.
    x = full_x[..., rank * width:(rank + 1) * width].detach().requires_grad_()
    weight = full_w[:, rank * width:(rank + 1) * width].detach().requires_grad_()
    bias.requires_grad_()
    ctx = prepare_pik(shape[-1], n, plan=PikPlan(leaves, layout), group=dist.group.WORLD,
                      sequence_parallel=sp, sequence_dim=axis, transport=transport)
    with torch.no_grad():
        expected = reference(full_x, full_w, leaves, bias)
        if sp:
            expected = expected.chunk(world, axis)[rank]
    actual = pik_linear(x, weight, bias, context=ctx)
    _assert_bits(actual, expected)
    torch.manual_seed(31)
    dy = torch.randn(*shape[:-1], n, device="npu", dtype=torch.bfloat16)
    local_dy = dy.chunk(world, axis)[rank] if sp else dy
    (actual.float() * local_dy.float()).sum().backward()
    expected_dx = (dy.reshape(-1, n) @ weight.detach()).reshape(x.shape)
    expected_dw = dy.reshape(-1, n).T @ x.detach().reshape(-1, width)
    expected_db = dy.reshape(-1, n).sum(0).to(bias.dtype)
    for actual_grad, expected_grad in ((x.grad, expected_dx), (weight.grad, expected_dw), (bias.grad, expected_db)):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    result = {"shape": list(shape), "n": n, "g": leaves, "layout": layout,
              "transport": transport, "sp": sp, "axis": axis, "bias_dtype": str(bias.dtype),
              "forward_exact": True,
              "conventional_gradients_exact": True}
    return result


def _cross_batch(rank, world):
    """Check fixed PIK arithmetic across different token batch sizes."""
    torch.manual_seed(991)
    x = torch.randn(129, 512, device="npu", dtype=torch.bfloat16)
    weight = torch.randn(64, 512, device="npu", dtype=torch.bfloat16)
    single = prepare_pik(512, 64)
    parallel = prepare_pik(512, 64, group=dist.group.WORLD)
    width = 512 // world
    local_weight = weight[:, rank * width:(rank + 1) * width]
    count = 0
    with torch.no_grad():
        canonical = pik_linear(x, weight, context=single)
        for rows in (1, 3, 16, 65, 129):
            for start in sorted({0, 129 - rows}):
                local_x = x[start:start + rows, rank * width:(rank + 1) * width]
                actual = pik_linear(local_x, local_weight, context=parallel)
                _assert_bits(actual, canonical[start:start + rows])
                count += 1
    if world > 1:
        for bad_n in (64 + rank, -1 if rank == 0 else 64):
            try:
                prepare_pik(512, bad_n, group=dist.group.WORLD)
            except ValueError:
                pass
            else:
                raise AssertionError(f"TP{world} accepted inconsistent contracts on rank {rank}")
        try:
            prepare_pik(512, 64, group=dist.group.WORLD, policy={"unknown": True} if rank == 0 else {})
        except ValueError:
            pass
        else:
            raise AssertionError(f"TP{world} did not propagate invalid policy from rank zero")
    return count


def _row_adapter_backward(rank: int, world: int) -> dict:
    """Compare SP/non-SP model adapter gradients with native row-linear autograd."""
    results = {}
    for sp in (False, True):
        torch.manual_seed(901)
        module = torch.nn.Linear(512, 64, bias=False, device="npu", dtype=torch.bfloat16)
        module.weight = torch.nn.Parameter(module.weight[:, rank * (512 // world):(rank + 1) * (512 // world)])
        row = PikLinear(module, PikPlan(8), parallel=True, fill_uninitialized_memory=False)
        row.bind_pik(dist.group.WORLD, sequence_parallel=sp)
        inputs = torch.randn(1, world * 3, 512 // world, device="npu", dtype=torch.bfloat16, requires_grad=True)
        dy = torch.randn(1, world * 3, 64, device="npu", dtype=torch.bfloat16)
        reference_inputs = inputs.detach().clone().requires_grad_()
        reference_weight = row.weight.detach().clone().requires_grad_()
        native_output = torch.nn.functional.linear(reference_inputs, reference_weight)
        if not sp:
            dy = dy + rank
            native_output = dist_func.all_reduce(native_output, group=dist.group.WORLD)
        expected = torch.autograd.grad(native_output, (reference_inputs, reference_weight), grad_outputs=dy)
        row(inputs).backward(dy.chunk(world, dim=1)[rank].contiguous() if sp else dy)
        for actual, wanted in zip((inputs.grad, row.weight.grad), expected, strict=True):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
        results["sp" if sp else "non_sp"] = {"input_gradient_exact": True, "weight_gradient_exact": True}
    return results


def test_pik_npu() -> None:
    """Exercise fixed G across TP, layouts, gradients, eager/compile/device graph."""
    torch_npu.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group("hccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    rows = []
    try:
        for leaves in (8, 16):
            for layout in ("strided", "packed"):
                for transport in ("all_gather", "all_to_all"):
                    rows.append(_run_case((world * 3, 512), 64, leaves, layout, transport,
                                          False, 0, rank, world))
                    rows.append(_run_case((2, world * 3, 512), 64, leaves, layout, transport,
                                          True, 1, rank, world))
        for global_k in (4096, 9728):
            rows.append(_run_case((17, global_k), 2560, 8, "strided", "all_gather",
                                  False, 0, rank, world))
        # A cancellation example distinguishes a balanced tree from a left fold.
        values = torch.tensor([1e8, 1, -1e8, 1, 2, 3, 4, 5], device="npu", dtype=torch.float32)
        actual = fixed_tree(values[:, None, None])
        torch.testing.assert_close(actual, torch.tensor([[14.0]], device="npu"), rtol=0, atol=0)
        width = 512 // world
        x = torch.randn(world * 2, width, device="npu", dtype=torch.bfloat16)
        w = torch.randn(64, width, device="npu", dtype=torch.bfloat16)
        context = prepare_pik(512, 64, group=dist.group.WORLD, sequence_parallel=True, transport="all_to_all")

        def run(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            """Trace the production PIK including the custom tree and transport."""
            return pik_linear(inputs, weights, context=context)

        compiled = torch.compile(run, backend="aot_eager", fullgraph=True)
        with torch.no_grad():
            expected = run(x, w)
            _assert_bits(compiled(x, w), expected)
            for _ in range(3):
                run(x, w)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                captured = run(x, w)
            for scale in (0.25, -0.5, 2.0):
                x.mul_(scale)
                w.mul_(-0.75)
                graph.replay()
                torch.npu.synchronize()
                _assert_bits(captured, run(x, w))
        # A compiled backward must preserve the custom first-order gradient boundary.
        x.requires_grad_()
        w.requires_grad_()
        compiled(x, w).float().sum().backward()
        torch.testing.assert_close(x.grad, torch.ones(world * 2, 64, device="npu", dtype=torch.bfloat16) @ w.detach(),
                                   rtol=0, atol=0)
        cross_batch_checks = _cross_batch(rank, world)
        row_gradients = _row_adapter_backward(rank, world)
        result = {"rank": rank, "tp": world, "checks": rows, "cross_batch_checks": cross_batch_checks,
                  "compiled_forward": True,
                  "compiled_backward": True, "graph_replays": 3, "row_adapter_gradients": row_gradients}
        directory = Path(os.environ.get("RL_PIK_OUTPUT", "/tmp/hyper-rl-pik"))
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({"rank": rank, "tp": world, "cases": len(rows), "passed": True}), flush=True)
    finally:
        dist.destroy_process_group()

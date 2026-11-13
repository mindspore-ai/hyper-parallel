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
"""Real HCCL regression for HP version-compatible functional collectives."""
from contextlib import nullcontext
from itertools import product
import json
import os

import torch
import torch.distributed as dist
import torch_npu

from hyper_parallel.core.utils.communication import (
    differentiable_all_gather_concat,
    differentiable_all_reduce,
    differentiable_reduce_scatter,
)


def _collective(operation, axis, group, size):
    """Build a static callable so Dynamo traces the actual HP wrapper."""
    def run(value: torch.Tensor) -> torch.Tensor:
        """Execute a single HP collective without bypassing its compile branch."""
        if operation == "gather":
            return differentiable_all_gather_concat(value, group, size, axis)
        reduce_op = dist.ReduceOp.SUM if operation == "sum_enum" else operation
        return differentiable_reduce_scatter(value, size, axis, reduce_op, group)

    return run


def test_compiled_communication():
    """Check FP32/BF16, three axes, SUM/AVG, changed data and both gradients."""
    torch_npu.npu.set_device(int(os.environ["LOCAL_RANK"]))
    torch_npu.npu.set_deterministic_level(3)
    dist.init_process_group("hccl")
    results = []
    try:
        rank, size = dist.get_rank(), dist.get_world_size()
        for operation, axis in product(("gather", "sum", "avg", "sum_enum"), (0, 1, 2)):
            eager = _collective(operation, axis, dist.group.WORLD, size)
            compiled = torch.compile(eager, backend="aot_eager", fullgraph=True)
            for dtype in (torch.float32, torch.bfloat16):
                for seed in (17, 43):
                    torch.manual_seed(seed + rank)
                    base = torch.randn(4, 6, 8, dtype=dtype, device="npu")
                    for mode, context in (("grad", nullcontext), ("no_grad", torch.no_grad),
                                          ("inference", torch.inference_mode)):
                        outputs, gradients = [], []
                        with context():
                            for function in (eager, compiled):
                                value = base.detach().clone().requires_grad_()
                                output = function(value)
                                if mode == "grad":
                                    # Nonlinear loss exposes wrong rank/dimension gradients.
                                    output.float().square().sum().backward()
                                    gradients.append(value.grad.detach())
                                outputs.append(output.detach())
                            torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)
                            if mode == "grad":
                                torch.testing.assert_close(gradients[1], gradients[0], rtol=0, atol=0)
                        results.append({"op": operation, "axis": axis, "dtype": str(dtype),
                                        "seed": seed, "mode": mode, "forward_exact": True,
                                        "backward_exact": True if mode == "grad" else None})
            torch.compiler.reset()
        torch_npu.npu.synchronize()
        print(json.dumps({"rank": rank, "checks": results}), flush=True)
    finally:
        dist.destroy_process_group()


def test_compiled_all_reduce_inference():
    """Check inference aliases and retain eager SUM gradients on actual HCCL."""
    torch_npu.npu.set_device(int(os.environ["LOCAL_RANK"]))
    torch_npu.npu.set_deterministic_level(3)
    dist.init_process_group("hccl")
    results = []
    try:
        rank, size = dist.get_rank(), dist.get_world_size()
        for dtype in (torch.float32, torch.bfloat16):
            operations = ["sum", "avg", "max", "min", "all", "mean"]
            # The current HCCL functional PRODUCT kernel does not accept BF16.
            if dtype == torch.float32:
                operations.append("prod")
            for operation in operations:
                def eager(value: torch.Tensor, operation=operation) -> torch.Tensor:
                    """Trace the public wrapper with a fixed string operation."""
                    return differentiable_all_reduce(value, operation, dist.group.WORLD)

                compiled = torch.compile(eager, backend="aot_eager", fullgraph=True)
                for offset in (0, 1):
                    base = torch.arange(24, device="npu", dtype=dtype).reshape(4, 6) / 8 + rank + offset
                    for context in (torch.no_grad, torch.inference_mode):
                        with context():
                            torch.testing.assert_close(compiled(base), eager(base), rtol=0, atol=0)
                        results.append({"op": operation, "dtype": str(dtype), "offset": offset,
                                        "mode": context.__name__, "forward_exact": True})
                torch.compiler.reset()
            value = torch.ones(4, 6, device="npu", dtype=dtype, requires_grad=True)
            output = differentiable_all_reduce(value, "sum", dist.group.WORLD)
            (output.float().sum() * (rank + 1)).backward()
            expected = torch.full_like(value, size * (size + 1) / 2)
            torch.testing.assert_close(value.grad, expected, rtol=0, atol=0)
            results.append({"dtype": str(dtype), "eager_backward_exact": True})
        torch_npu.npu.synchronize()
        print(json.dumps({"rank": rank, "checks": results}), flush=True)
    finally:
        dist.destroy_process_group()

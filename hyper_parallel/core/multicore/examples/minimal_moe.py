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
"""Run automatic-capacity MegaMoe alone and migrate materialized SwiGLU expert weights."""

from __future__ import annotations

import os
import socket
from contextlib import nullcontext
from typing import Any

import torch

from hyper_parallel.components.modules.moe import GroupedExperts
from hyper_parallel.core.multicore import MegaMoeExperts
from hyper_parallel.core.multicore.diagnostics import execution_monitor
from hyper_parallel.core.multicore.examples._minimal_common import (
    compare,
    device_for,
    parser,
    report,
)


def _reference(source: GroupedExperts, hidden: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Evaluate selected experts independently with FP32 Torch matmuls."""
    flat = hidden.reshape(-1, hidden.shape[-1])
    output = torch.zeros_like(flat, dtype=torch.float32)
    for expert in range(source.num_experts):
        rows, slots = torch.where(ids == expert)
        gate = flat[rows].float() @ source.w1[expert].float().transpose(0, 1)
        up = flat[rows].float() @ source.w3[expert].float().transpose(0, 1)
        values = (torch.nn.functional.silu(gate) * up) @ source.w2[expert].float().transpose(0, 1)
        output = output.index_add(0, rows, values * weights[rows, slots, None])
    return output.to(hidden.dtype).reshape_as(hidden)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Compare one local expert module without requiring Gate, mHC or an EP8 network.

    Args:
        argv: Optional command-line argument list; None reads process arguments.
    """
    options = parser(__doc__)
    options.add_argument("--dispatch-mode", choices=("push", "pull"), default="push",
                         help="advanced communication policy; omit for the standard example")
    options.add_argument("--diagnostic-output", help="write a Host execution report for this forward/backward sample")
    args = options.parse_args(argv)
    if args.device != "npu":
        raise ValueError("MegaMoeExperts requires NPU execution; use --device npu")
    device = device_for(args.device)
    with socket.socket() as endpoint:
        endpoint.bind(("127.0.0.1", 0))
        os.environ.setdefault("HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT", f"tcp://127.0.0.1:{endpoint.getsockname()[1]}")
    torch.manual_seed(37)
    source = GroupedExperts(args.hidden_size, 128, 4).to(device=device, dtype=torch.bfloat16)
    experts = MegaMoeExperts.from_module(source, top_k=2, dispatch_mode=args.dispatch_mode)
    hidden = (torch.randn(1, args.tokens, args.hidden_size, device=device, dtype=torch.bfloat16) * 0.2).requires_grad_()
    ids = (torch.arange(args.tokens * 2, device=device).reshape(-1, 2) % 4).int()
    weights = torch.randn(args.tokens, 2, device=device).softmax(-1).requires_grad_()
    reference_hidden = hidden.detach().clone().requires_grad_()
    reference_weights = weights.detach().clone().requires_grad_()
    session = (execution_monitor(experts, output_dir=args.diagnostic_output)
               if args.diagnostic_output else nullcontext())
    with session as monitor:
        actual = experts(hidden, topk_ids=ids, topk_weights=weights)
        expected = _reference(source, reference_hidden, ids, reference_weights)
        probe = torch.randn_like(actual) * 0.1
        observed = torch.autograd.grad((actual * probe).sum(),
                                      (hidden, weights, experts.gate_up_weight, experts.down_weight))
        reference = torch.autograd.grad((expected * probe).sum(),
                                       (reference_hidden, reference_weights, source.w1, source.w3, source.w2))
        if monitor is not None:
            monitor.step()
    pairs = (("output", actual, expected), ("input_grad", observed[0], reference[0]),
             ("route_grad", observed[1], reference[1]),
             ("gate_grad", observed[2][..., :128].transpose(1, 2), reference[2]),
             ("up_grad", observed[2][..., 128:].transpose(1, 2), reference[3]),
             ("down_grad", observed[3].transpose(1, 2), reference[4]))
    errors = {name: compare(value, golden, name) for name, value, golden in pairs}
    experts.close()
    return report("moe", errors, input_shape=list(hidden.shape), dispatch_mode=args.dispatch_mode,
                  execution_reports=str(monitor.directory) if monitor is not None else None)


if __name__ == "__main__":
    main()

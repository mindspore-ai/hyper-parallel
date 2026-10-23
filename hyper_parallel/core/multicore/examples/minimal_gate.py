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
"""Run MegaGate alone, preserving an existing router and checking outputs and gradients."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from hyper_parallel.core.multicore import MegaGate
from hyper_parallel.core.multicore.examples._minimal_common import (
    compare,
    device_for,
    parser,
    report,
)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Compare a migrated router with explicit ordinary projection, scoring and TopK.

    Args:
        argv: Optional command-line argument list; None reads process arguments.
    """
    options = parser(__doc__)
    options.add_argument("--scoring-func", choices=("sqrtsoftplus", "softmax", "sigmoid"), default="sqrtsoftplus")
    args = options.parse_args(argv)
    device = device_for(args.device)
    torch.manual_seed(31)
    source = torch.nn.Module()
    source.weight = torch.nn.Parameter(torch.randn(16, args.hidden_size, device=device) * 0.02)
    source.bias = torch.nn.Parameter(torch.linspace(-0.05, 0.05, 16, device=device), requires_grad=False)
    source.top_k, source.scoring_func, source.routed_scaling_factor = 4, args.scoring_func, 1.0
    dtype = torch.bfloat16 if args.device == "npu" else torch.float32
    gate = MegaGate.from_module(source).to(device=device, dtype=dtype)
    hidden = torch.randn(1, args.tokens, args.hidden_size, device=device, dtype=gate.weight.dtype, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_()
    actual = gate(hidden)
    logits = F.linear(reference_hidden.flatten(0, 1).float(), source.weight.float())
    if source.scoring_func == "sqrtsoftplus":
        scores = F.softplus(logits).sqrt()
    elif source.scoring_func == "softmax":
        scores = logits.softmax(-1)
    else:
        scores = logits.sigmoid()
    ids = (scores + source.bias).topk(source.top_k, dim=-1, sorted=False).indices
    weights = scores.gather(1, ids)
    weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
    torch.testing.assert_close(actual.ids, ids, rtol=0, atol=0)
    probe = torch.randn_like(weights)
    actual_grad = torch.autograd.grad((actual.weights * probe).sum(), (hidden, gate.weight))
    reference_grad = torch.autograd.grad((weights * probe).sum(), (reference_hidden, source.weight))
    errors = {"logits": compare(actual.logits, logits, "logits"),
              "weights": compare(actual.weights, weights, "weights"),
              "input_grad": compare(actual_grad[0], reference_grad[0], "input gradient"),
              "weight_grad": compare(actual_grad[1], reference_grad[1], "projection gradient")}
    return report("gate", errors, backend=gate.backend_for(hidden)._asdict(), input_shape=list(hidden.shape))


if __name__ == "__main__":
    main()

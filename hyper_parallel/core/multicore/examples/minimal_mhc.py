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
"""Run a four-stream mHC boundary alone with named state and existing-parameter migration."""

from __future__ import annotations

from typing import Any

import torch

from hyper_parallel.core.multicore import MegaMhc, MhcState
from hyper_parallel.core.multicore.examples._minimal_common import (
    compare,
    device_for,
    parser,
    report,
)
from hyper_parallel.core.multicore.examples.mhc import MhcBoundary


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Compare five boundary outputs and all nine input/parameter gradients with ordinary operations.

    Args:
        argv: Optional command-line argument list; None reads process arguments.
    """
    args = parser(__doc__).parse_args(argv)
    if args.device != "npu":
        raise ValueError("MegaMhc native execution requires NPU; use --device npu")
    device = device_for(args.device)
    requirements = MegaMhc.execution_requirements(device)
    if args.tokens < requirements["minimum_training_tokens"]:
        minimum = requirements["minimum_training_tokens"]
        raise ValueError(f"this device requires at least {minimum} local training tokens")
    torch.manual_seed(41)
    original = MhcBoundary(args.hidden_size, 1e-6, use_mega_mhc=False)
    boundary = MegaMhc.from_module(original).to(device=device, dtype=torch.bfloat16)
    hidden = torch.randn(1, args.tokens, args.hidden_size, device=device, dtype=torch.bfloat16)
    initial = MhcState.initialize(hidden)
    state = MhcState(*(tensor.detach().clone().requires_grad_() for tensor in initial))
    reference_state = MhcState(*(tensor.detach().clone().requires_grad_() for tensor in initial))
    actual = boundary(state)
    expected = original(reference_state)
    probes = tuple(torch.randn_like(tensor) / args.tokens for tensor in actual)
    observed = torch.autograd.grad(sum((value * probe).sum() for value, probe in zip(actual, probes)),
                                  (*state, *boundary.parameters()))
    golden = torch.autograd.grad(sum((value * probe).sum() for value, probe in zip(expected, probes)),
                                (*reference_state, *original.parameters()))
    errors = {name: compare(value, reference, name)
              for name, value, reference in zip(actual._fields, actual, expected)}
    errors.update({f"gradient_{index}": compare(value, reference, f"gradient {index}")
                   for index, (value, reference) in enumerate(zip(observed, golden))})
    final = actual.next_state(actual.hidden_states.detach()).finish()
    return report("mhc", errors, final_shape=list(final.shape),
                  minimum_training_tokens=requirements["minimum_training_tokens"])


if __name__ == "__main__":
    main()

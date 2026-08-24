# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""testing.grad_equiv: M_D.15a dual-mode gradient equivalence utilities (05 §5.5 revised).

The in-house DTensor is forward-only (05 §1.0): the backward of both
production (FSDP/source_shard_info bypass) and validate (local autograd, direct
output) follows the **local tensor path** — there is no "DTensor backward"
control group. Dual-mode gradient equivalence therefore compares gradients
parameter by parameter directly:

- TP-Shard parameters: gradients in both modes are naturally local shards,
  equal rank by rank (no sync needed);
- TP-Replicate parameters: gradients in both modes are likewise Partial
  contributions, equal rank by rank; before comparing against the
  single-card reference gradient they must first go through the
  source_shard_info bypass all-reduce (this module provides a simulation; the
  real FSDP2 fork path is part of the M_M.2a joint integration).
"""

from typing import Any, Optional

import torch
import torch.distributed as dist


def run_one_step(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int,
) -> tuple[torch.Tensor, dict[str, Optional[torch.Tensor]]]:
    """Single forward+backward step, returns {param_fqn: grad}."""
    model.zero_grad()
    logits = model(input_ids)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, vocab_size).float(), labels.reshape(-1))
    loss.backward()
    return loss, {
        name: (param.grad.clone() if param.grad is not None else None)
        for name, param in model.named_parameters()
    }


def assert_grad_equivalence(
    prod_grads: dict[str, Optional[torch.Tensor]],
    val_grads: dict[str, Optional[torch.Tensor]],
    *,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> None:
    """Dual-mode per-parameter assert_close on gradients (parameters missing on both sides are skipped)."""
    for name, gp in prod_grads.items():
        gv = val_grads.get(name)
        if gp is None and gv is None:
            continue
        assert gp is not None, f"{name}: production missing gradient"
        assert gv is not None, f"{name}: validate missing gradient"
        torch.testing.assert_close(gp, gv, rtol=rtol, atol=atol)


def simulate_tp_replicate_grad_sync(grad: torch.Tensor, tp_group: Any) -> torch.Tensor:
    """Simulate the source_shard_info bypass: TP all-reduce of TP-Replicate parameter gradients.

    The real path is implemented by the FSDP2 fork's all_reduce_grad
    (M_M.2a joint integration); this is used for gradient equivalence
    validation during the independent-development phase.
    """
    synced = grad.clone()
    dist.all_reduce(synced, group=tp_group)
    return synced

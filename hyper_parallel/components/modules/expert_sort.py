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
"""Expert-major ordering for MoE routing, without an integer sort.

``ArgSort`` has no AI Core kernel for int32 or int64 on this stack: torch_npu
emits a runtime warning and the kernel runs on AI CPU, which it says to avoid
by casting to float32. Routing sorts one key per routed slot on every MoE
layer of every step, so the fallback is paid per layer per step.

float32 holds every integer below ``2**24`` exactly. Encoding the slot index
into the key as ``expert * slots + slot`` makes it strictly increasing in
``(expert, slot)``, so sorting the float32 key ascending reproduces the stable
expert-major permutation exactly rather than leaving ties arbitrary. The
encoding needs ``num_experts * slots`` below the same limit; past it, the
integer sort is used unchanged.
"""
from __future__ import annotations

import torch  # pylint: disable=forbidden-backend-import

# float32 represents every integer up to 2**24 exactly (see
# ``distributed/expert_parallel/experts._SORT_KEY_FP32_LIMIT``, which states the
# same bound for the dispatch sort).
SORT_KEY_FP32_LIMIT = 1 << 24


def expert_major_permutation(flat_expert: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Return the stable permutation that groups routed slots by expert.

    The result is the permutation ``argsort(flat_expert, stable=True)`` would
    give: slots ordered by expert, and by slot index inside one expert.

    Args:
        flat_expert: Expert index of every routed slot, one dimension.
        num_experts: Size of the expert index space, an exclusive upper bound
            on ``flat_expert``, known statically so no device sync is needed.

    Returns:
        An int64 permutation of ``range(flat_expert.numel())``.
    """
    slots = flat_expert.numel()
    if 0 < num_experts and num_experts * slots <= SORT_KEY_FP32_LIMIT:
        slot = torch.arange(slots, device=flat_expert.device, dtype=torch.float32)
        key = torch.addcmul(slot, flat_expert.to(torch.float32),
                            torch.full((), float(slots), device=flat_expert.device,
                                       dtype=torch.float32))
        return key.argsort()
    return flat_expert.argsort(stable=True)


__all__ = ["SORT_KEY_FP32_LIMIT", "expert_major_permutation"]

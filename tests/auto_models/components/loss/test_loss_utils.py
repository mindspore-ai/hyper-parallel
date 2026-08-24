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
"""Tests for Trainer loss normalization over mixed parallel meshes."""

from typing import Any
from unittest.mock import patch

import torch

from hyper_parallel.auto_models.components.loss.loss_utils import mean_global_loss


class _FakeSubmesh:
    def __init__(self, group: Any) -> None:
        self._group = group

    def get_group(self) -> Any:
        return self._group


class _FakeMeshContext:
    def __init__(self, group: Any) -> None:
        self.dp_cp_mesh = _FakeSubmesh(group)
        self.dp_size = 2
        self.cp_size = 2
        self.sequence_parallel = False


def test_mean_global_loss_reduces_tokens_over_dp_cp_only() -> None:
    """TP replicas must not be counted as distinct data-parallel tokens."""
    dp_cp_group = object()
    mesh = _FakeMeshContext(dp_cp_group)
    micro_tokens = {"foundation_tokens": torch.tensor(32)}
    step_tokens = {"foundation_tokens": torch.tensor(256)}

    with patch(
        "hyper_parallel.auto_models.components.loss.loss_utils.all_reduce",
        side_effect=[512, 384],
    ) as mock_all_reduce:
        local_loss = torch.tensor(6.0, requires_grad=True)
        result = mean_global_loss(
            local_loss,
            micro_tokens,
            step_tokens,
            device_mesh=mesh,
        )

    assert mock_all_reduce.call_args_list == [
        ((256,), {"op": "sum", "group": dp_cp_group}),
        ((192.0,), {"op": "sum", "group": dp_cp_group}),
    ], (
        f"Unexpected DP+CP reductions: got={mock_all_reduce.call_args_list}"
    )
    assert torch.allclose(result["foundation_loss"], torch.tensor(0.75))
    result["foundation_loss"].backward()
    assert torch.allclose(local_loss.grad, torch.tensor(0.25)), (
        f"DP*CP gradient scale mismatch: expected={torch.tensor(0.25)}, "
        f"got={local_loss.grad}"
    )

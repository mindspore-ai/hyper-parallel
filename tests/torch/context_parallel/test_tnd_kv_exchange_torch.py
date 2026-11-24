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
"""Torch autograd coverage of the platform-neutral zero-communication path."""
import torch

from tests.common.mark_utils import arg_mark

from hyper_parallel.core.context_parallel.tnd_kv_exchange import (
    block_views, exchange_tnd_kv, plan_tnd_kv_exchange, reduce_tnd_kv_gradients,
)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_local_only_bank_supports_shared_consumers_and_explicit_kl() -> None:
    """Shared views accumulate without communication or a full-sequence buffer."""
    plan = plan_tnd_kv_exchange(tuple(range(4, 33, 4)), ((2, 7), (1, 5), (0, 3), (4, 6)))[0]
    local = torch.arange(24, dtype=torch.float32).reshape(8, 3).requires_grad_()
    bank = exchange_tnd_kv(local, plan, None)
    views = block_views(bank, plan)
    first, second = views[0], views[1]
    (first.sum() + 3 * second.sum() + 2 * bank.sum()).backward()
    torch.testing.assert_close(local.grad, torch.cat((torch.full((4, 3), 3.), torch.full((4, 3), 5.))))
    explicit = reduce_tnd_kv_gradients((torch.ones_like(first), 3*torch.ones_like(second)), plan, None)
    torch.testing.assert_close(explicit, torch.cat((torch.ones(4, 3), torch.full((4, 3), 3.))))

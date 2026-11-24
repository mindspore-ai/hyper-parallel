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
"""Four-rank HCCL tests for causal-document KV transport and its adjoint."""
import os

import numpy as np
import mindspore as ms
from mindspore.communication import init, get_rank, get_group_size
from mindspore.communication.management import GlobalComm

from hyper_parallel.platform.mindspore.autograd_compat import enable_mindspore_backward_compat
from hyper_parallel.core.context_parallel.tnd_kv_exchange import (
    exchange_tnd_kv, plan_tnd_kv_exchange, reduce_tnd_kv_gradients,
)


def test_tnd_kv_partial_exchange() -> None:
    """Unequal messages, idle ranks and duplicate consumers preserve owner gradients."""
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend",
                   device_id=int(os.environ.get("DEVICE_ID", os.environ.get("LOCAL_RANK", "0"))))
    init()
    enable_mindspore_backward_compat()
    rank = get_rank()
    if get_group_size() != 4:
        raise ValueError("This routing regression requires exactly four ranks.")
    pairs = ((2, 7), (1, 5), (0, 3), (4, 6))
    owned = np.concatenate([np.arange(chunk * 4, (chunk + 1) * 4) for chunk in pairs[rank]])
    natural = np.arange(96, dtype=np.float32).reshape(32, 3)
    for dtype in (ms.float32, ms.bfloat16):
        local = ms.Tensor(natural[owned], dtype)
        for ends in (tuple(range(4, 33, 4)), (2, 6, 17, 32, 32), (12, 16, 20, 24, 28, 32)):
            plans = plan_tnd_kv_exchange(ends, pairs)
            plan = plans[rank]
            bank = exchange_tnd_kv(local, plan, GlobalComm.WORLD_COMM_GROUP)
            wanted = np.concatenate([np.arange(start, stop) for start, stop in plan.intervals])
            np.testing.assert_array_equal(bank.float().asnumpy(), natural[wanted])

            def objective(value: ms.Tensor) -> ms.Tensor:
                """Accumulate two consumers through the same communicated bank."""
                selected = exchange_tnd_kv(value, plan, GlobalComm.WORLD_COMM_GROUP)
                return (selected * (rank + 1)).sum() + (selected * 2).sum()

            gradient = ms.grad(objective)(local)
            expected = np.zeros_like(natural)
            for peer, route in enumerate(plans):
                for start, stop in route.intervals:
                    expected[start:stop] += peer + 3
            np.testing.assert_array_equal(gradient.float().asnumpy(), expected[owned])
            explicit = tuple(ms.Tensor(np.full((block.key_interval[1] - block.key_interval[0], 3),
                                               rank + 1, np.float32), dtype) for block in plan.blocks)
            actual = reduce_tnd_kv_gradients(explicit, plan, GlobalComm.WORLD_COMM_GROUP)
            expected[:] = 0
            for peer, route in enumerate(plans):
                for block in route.blocks:
                    expected[slice(*block.key_interval)] += peer + 1
            np.testing.assert_array_equal(actual.float().asnumpy(), expected[owned])

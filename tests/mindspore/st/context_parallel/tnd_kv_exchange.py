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
from typing import Any
from unittest.mock import patch

import numpy as np
import mindspore as ms
from mindspore.communication import init, get_rank, get_group_size
from mindspore.communication.management import GlobalComm

from hyper_parallel.platform.mindspore.autograd_compat import enable_mindspore_backward_compat
from hyper_parallel.platform.mindspore.platform import inner_comm_all_to_all_v_op
from hyper_parallel.core.context_parallel.tnd_kv_exchange import (
    exchange_tnd_kv, plan_tnd_kv_exchange, reduce_tnd_kv_gradients,
)


def _assert_directions(counter: Any, directions: tuple) -> None:
    """Count actual backend primitive entries, including any nested autograd replay."""
    assert counter.call_count == len(directions), (counter.call_count, len(directions))
    for call, (sends, receives) in zip(counter.call_args_list, directions):
        assert tuple(call.args[2]) == sends
        assert tuple(call.args[3]) == receives
    counter.reset_mock()


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
            with patch("hyper_parallel.platform.mindspore.platform.inner_comm_all_to_all_v_op",
                       wraps=inner_comm_all_to_all_v_op) as counter:
                bank = exchange_tnd_kv(local, plan, GlobalComm.WORLD_COMM_GROUP)
                active = int(plan.global_remote_rows > 0)
                sends = tuple(3*count for count in plan.send_rows)
                receives = tuple(3*count for count in plan.recv_rows)
                if ends == (12, 16, 20, 24, 28, 32):
                    assert plan.global_remote_rows > 0
                    assert sum(plans[3].send_rows) == sum(plans[3].recv_rows) == 0
                wanted = np.concatenate([np.arange(start, stop) for start, stop in plan.intervals])
                np.testing.assert_array_equal(bank.float().asnumpy(), natural[wanted])
                _assert_directions(counter, ((sends, receives),) if active else ())

                def objective(value: ms.Tensor) -> ms.Tensor:
                    """Accumulate two consumers through the same communicated bank."""
                    selected = exchange_tnd_kv(value, plan, GlobalComm.WORLD_COMM_GROUP)
                    return (selected * (rank + 1)).sum() + (selected * 2).sum()

                gradient = ms.grad(objective)(local).float().asnumpy()
                _assert_directions(counter, ((sends, receives), (receives, sends)) if active else ())
                expected = np.zeros_like(natural)
                for peer, route in enumerate(plans):
                    for start, stop in route.intervals:
                        expected[start:stop] += peer + 3
                np.testing.assert_array_equal(gradient, expected[owned])
                explicit = tuple(ms.Tensor(np.full((block.key_interval[1] - block.key_interval[0], 3),
                                                   rank + 1, np.float32), dtype) for block in plan.blocks)
                actual = reduce_tnd_kv_gradients(explicit, plan, GlobalComm.WORLD_COMM_GROUP)
                expected[:] = 0
                for peer, route in enumerate(plans):
                    for block in route.blocks:
                        expected[slice(*block.key_interval)] += peer + 1
                np.testing.assert_array_equal(actual.float().asnumpy(), expected[owned])
                _assert_directions(counter, ((receives, sends),) if active else ())
                print(f"KV_COLLECTIVES rank={rank} dtype={dtype} global_remote={plan.global_remote_rows} "
                      f"idle={sum(plan.send_rows) == sum(plan.recv_rows) == 0} "
                      f"forward={active} ms_grad_total={2*active} explicit_reverse={active}", flush=True)

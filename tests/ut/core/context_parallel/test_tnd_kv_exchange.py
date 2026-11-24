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
"""Independent ownership and causal-document coverage tests for partial exchange."""
import random

import pytest

from tests.common.mark_utils import arg_mark

from hyper_parallel.core.context_parallel.tnd_kv_exchange import plan_tnd_kv_exchange
from hyper_parallel.core.shard.ops.tnd_kv_plan import plan_tnd_kv


@pytest.mark.parametrize('size', [1, 2, 4, 8])
@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_routes_send_exact_causal_document_rows(size: int) -> None:
    """Enumerate tokens independently, including non-involutive dynamic ownership."""
    rng = random.Random(size)
    full_len = 64
    width = full_len // (2 * size)
    for _ in range(20):
        ends = tuple(sorted(rng.sample(range(1, full_len), 9))) + (full_len, full_len)
        chunks = list(range(2 * size))
        rng.shuffle(chunks)
        pairs = tuple(tuple(chunks[2 * rank:2 * rank + 2]) for rank in range(size))
        plans = plan_tnd_kv_exchange(ends, pairs)
        total_rows = 0
        for rank, plan in enumerate(plans):
            wanted = set()
            for chunk in pairs[rank]:
                for query in range(chunk * width, (chunk + 1) * width):
                    left = max([0] + [end for end in ends if end <= query])
                    wanted.update(range(left, query + 1))
            bank = [token for start, stop in plan.intervals for token in range(start, stop)]
            assert bank == sorted(wanted)
            received = {}
            for src, routes in enumerate(plan.routes_in):
                owned = [token for chunk in pairs[src] for token in range(chunk * width, (chunk + 1) * width)]
                assert routes == plans[src].routes_out[rank]
                for source, target, count in routes:
                    for index in range(count):
                        assert target + index not in received
                        received[target + index] = owned[source + index]
                        total_rows += src != rank
            assert [received[index] for index in range(plan.bank_rows)] == bank
            assert plan.send_rows[rank] == plan.recv_rows[rank] == 0
            assert plan.remote_rows <= plan.local_rows * (size - 1)
        assert total_rows == plans[0].global_remote_rows


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_local_documents_eliminate_all_remote_traffic() -> None:
    """A full allgather is unnecessary when every selected chunk owns complete documents."""
    pairs = ((2, 7), (1, 5), (0, 3), (4, 6))
    plans = plan_tnd_kv_exchange(tuple(range(4, 33, 4)), pairs)
    assert all(plan.global_remote_rows == 0 for plan in plans)
    assert all(plan.bank_rows == plan.local_rows for plan in plans)


@pytest.mark.parametrize('ends,pairs', [((8,), ((0, 0),)), ((9,), ((0, 1),)),
                                       ((4, 8), ((True, 1),)), ((8, 4), ((0, 1),))])
@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_invalid_routes_fail_before_communication(ends: tuple, pairs: tuple) -> None:
    """Reject inconsistent global geometry before entering any collective."""
    with pytest.raises(ValueError):
        plan_tnd_kv_exchange(ends, pairs)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_shared_planner_preserves_document_local_causal_coordinates() -> None:
    """Non-folded intervals keep the visible document prefixes and query offsets."""
    blocks = plan_tnd_kv((0, 4, 8, 20, 20), 20, ((3, 11), (12, 15)))
    assert blocks[0].key_interval == (0, 11)
    assert blocks[0].query_ends == (1, 5, 8)
    assert blocks[0].key_ends == (4, 8, 11)
    assert blocks[1].key_interval == (8, 15)
    assert blocks[1].query_ends == (3,)
    assert blocks[1].key_ends == (7,)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_sequence_inputs_normalize_before_cache() -> None:
    """Mutable application metadata and immutable equivalents select the same plan."""
    expected = plan_tnd_kv_exchange((4, 8), ((0, 1),))
    assert plan_tnd_kv_exchange([4, 8], [[0, 1]]) == expected


@pytest.mark.parametrize('ends,pairs', [([8, []], ((0, 1),)), ((8,), [0, 1]), ((8,), 3),
                                       ((8,), None), ([[8]], ((0, 1),)), ('8', ((0, 1),)),
                                       ((8,), ('01',)), ((8,), ((False, 1),))])
@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_malformed_sequence_structure_is_a_value_error(ends: object, pairs: object) -> None:
    """Structural failures are reported by the API rather than leaking cache TypeErrors."""
    with pytest.raises(ValueError):
        plan_tnd_kv_exchange(ends, pairs)

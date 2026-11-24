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
from dataclasses import replace
from typing import Any

import pytest
import torch

from tests.common.mark_utils import arg_mark

from hyper_parallel.core.context_parallel import tnd_kv_exchange as kv
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


@pytest.mark.parametrize('ends', [tuple(range(4, 33, 4)), (2, 6, 17, 32, 32),
                                  (12, 16, 20, 24, 28, 32), (32,)])
@pytest.mark.parametrize('explicit', [False, True])
@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_remote_transport_matches_independent_cpu_collective(
        monkeypatch: Any, ends: tuple, explicit: bool) -> None:
    """Exercise actual pack/unpack and adjoint through a transport-only collective stub."""
    pairs = ((2, 7), (1, 5), (0, 3), (4, 6))
    plans = plan_tnd_kv_exchange(ends, pairs)
    owners = [[token for chunk in pair for token in range(chunk*4, (chunk+1)*4)] for pair in pairs]
    wanted = []
    block_tokens = []
    for pair in pairs:
        blocks = []
        for chunk in pair:
            visible = set()
            for query in range(chunk*4, (chunk+1)*4):
                left = max([0] + [end for end in ends if end <= query])
                visible.update(range(left, query+1))
            blocks.append(visible)
        block_tokens.append(blocks)
        wanted.append(set.union(*blocks))
    natural = torch.arange(96, dtype=torch.float32).reshape(32, 3)/7
    grad_by_receiver = []
    for peer in range(4):
        if explicit:
            grad = torch.zeros_like(natural)
            for slot, tokens in enumerate(block_tokens[peer]):
                grad[sorted(tokens)] += (peer+1)*(slot+1)
        else:
            grad = 2*natural*(peer+1)**2+2
        grad_by_receiver.append(grad)

    for rank, plan in enumerate(plans):
        calls = []

        def exchange(flat: Any, sends: list, receives: list, group: Any) -> Any:
            """Emulate global all-to-all ordering using token enumeration, never production routes."""
            del group
            reverse = explicit or bool(calls)
            expected_send, receive_data, send_counts, receive_counts = [], [], [], []
            for peer in range(4):
                sent = ([token for token in owners[peer] if token in wanted[rank]] if reverse else
                        [token for token in owners[rank] if token in wanted[peer]]) if peer != rank else []
                received = ([token for token in owners[rank] if token in wanted[peer]] if reverse else
                            [token for token in owners[peer] if token in wanted[rank]]) if peer != rank else []
                send_counts.append(3*len(sent))
                receive_counts.append(3*len(received))
                expected_send.append((grad_by_receiver[rank] if reverse else natural)[sent].reshape(-1))
                receive_data.append((grad_by_receiver[peer] if reverse else natural)[received].reshape(-1))
            assert flat.ndim == 1
            assert list(sends) == send_counts
            assert list(receives) == receive_counts
            torch.testing.assert_close(flat, torch.cat(expected_send))
            calls.append(reverse)
            return torch.cat(receive_data)

        monkeypatch.setattr(kv.platform, 'differentiable_all_to_all_single_async', exchange)
        if explicit:
            gradients = tuple(torch.full((len(tokens), 3), (rank+1)*(slot+1), dtype=torch.float32, requires_grad=True)
                              for slot, tokens in enumerate(block_tokens[rank]))
            actual = reduce_tnd_kv_gradients(gradients, plan, None)
            assert not actual.requires_grad
        else:
            local = natural[owners[rank]].clone().requires_grad_()
            bank = exchange_tnd_kv(local, plan, None)
            torch.testing.assert_close(bank, natural[sorted(wanted[rank])])
            ((bank*(rank+1)).square().sum()+2*bank.sum()).backward()
            actual = local.grad
        expected = sum(grad_by_receiver[peer]*torch.tensor(
            [token in wanted[peer] for token in range(32)]).unsqueeze(1) for peer in range(4))
        torch.testing.assert_close(actual, expected[owners[rank]])
        assert len(calls) == (int(plan.global_remote_rows > 0)*(1 if explicit else 2))


@pytest.mark.parametrize('explicit', [False, True])
@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_missing_union_coverage_is_rejected(monkeypatch: Any, explicit: bool) -> None:
    """A malformed public route must fail before missing blocks or gradients disappear."""
    plan = plan_tnd_kv_exchange((8,), ((0, 1),))[0]
    plan = replace(plan, intervals=())

    def forbidden(*_args: Any) -> Any:
        """No collective should be issued for invalid metadata."""
        raise AssertionError('invalid union reached transport')

    monkeypatch.setattr(kv.platform, 'differentiable_all_to_all_single_async', forbidden)
    with pytest.raises(ValueError, match='union'):
        if explicit:
            reduce_tnd_kv_gradients((torch.ones(4, 3), torch.ones(8, 3)), plan, None)
        else:
            block_views(torch.empty(0, 3), plan)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_empty_gradient_plan_and_invalid_query_are_rejected() -> None:
    """Empty or inconsistent block metadata must not leak IndexError or unused query coordinates."""
    plan = plan_tnd_kv_exchange((8,), ((0, 1),))[0]
    with pytest.raises(ValueError):
        reduce_tnd_kv_gradients((), replace(plan, blocks=()), None)
    bad = replace(plan.blocks[0], query_interval=(9, 10))
    with pytest.raises(ValueError, match='query'):
        block_views(torch.ones(plan.bank_rows, 3), replace(plan, blocks=(bad, plan.blocks[1])))

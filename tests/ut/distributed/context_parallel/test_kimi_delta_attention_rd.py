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
"""CPU checks for RD configuration, reset plans and summary consumers."""
from dataclasses import FrozenInstanceError
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from hyper_parallel.components.functional.linear_attention_packed import PackedSequenceInfo
from hyper_parallel.distributed.context_parallel import kimi_delta_attention_rd as rd
from tests.common.mark_utils import arg_mark


def _make_protocol(rank, size, ranks=None):
    """Mock only the transport device; keep the executor and exchange logic real."""
    exchange = object.__new__(rd.RoundExchange)
    exchange.group, exchange.rank, exchange.size = None, rank, size
    exchange.peers, exchange.device = ranks or tuple(range(size)), torch.device("cpu")
    with patch.object(rd, "RoundExchange", return_value=exchange):
        return rd.KDAStateRecursiveDoubling(None, rank, exchange.peers)


def _finish_reference(product, previous, save_m=True):
    """CPU arithmetic for executor tests; NPU system tests cover the real epilogue."""
    width = product.shape[-2]
    product[..., :width].add_(previous[..., :width])
    return product[..., width:].clone() if save_m else None


def _finish_state_reference(product, previous, save_m):
    """Expose the final S without loading optional device kernels in unit tests."""
    matrix = _finish_reference(product, previous, save_m)
    return matrix, product[..., :product.shape[-2]].clone()


def _load_summary_functional():
    """Load the actual producer helper without importing optional Triton kernels."""
    source = Path(rd.__file__).resolve().parents[2] / "components/functional/kimi_delta_attention.py"
    package = "hyper_parallel.components.functional"
    spec = importlib.util.spec_from_file_location(f"{package}._kda_summary_cpu_regression", source)
    module = importlib.util.module_from_spec(spec)
    backend = SimpleNamespace(**{name: Mock() for name in (
        "apply_kda_state_gradient_summary", "apply_kda_state_summary",
        "kda_state_gradient_summary_from_prepared", "kda_state_summary_forward_from_prepared",
    )})
    with patch.dict(sys.modules, {f"{package}.kimi_delta_attention_state_summary": backend}):
        spec.loader.exec_module(module)
    return module


class TestCachedDoubling(unittest.TestCase):
    """Check reset plans, cache ownership and transport contracts without hardware."""

    def setUp(self) -> None:
        """Replace device epilogues with small test-only references."""
        for name, reference in (("_finish_round", _finish_reference), ("_finish_to_state", _finish_state_reference)):
            patcher = patch.object(rd, name, side_effect=reference)
            patcher.start()
            self.addCleanup(patcher.stop)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_invalid_inputs_fail_before_exchange(self):
        """Feature: RD input validation.

        Description: Pass incompatible peers, shape, dtype and invocation caches.
        Expectation: Invalid calls fail before any peer exchange.
        """
        with patch.object(rd.dist, "get_backend", return_value="gloo"), \
                patch.object(rd.dist, "get_process_group_ranks", return_value=[0, 2, 4]):
            with self.assertRaisesRegex(ValueError, "HCCL"):
                rd.RoundExchange(None, 2, (0, 2, 4))
            with self.assertRaisesRegex(ValueError, "group order"):
                rd.RoundExchange(None, 0, (2, 0, 4))
        protocol = _make_protocol(2, 3, (0, 2, 4))
        state = torch.zeros(1, 2, 128, 128)
        with patch.object(protocol, "exchange") as exchange:
            for matrix in (state.double(), state[..., :64], state.transpose(-1, -2)):
                with self.assertRaises(ValueError):
                    protocol.forward(state, matrix)
            for cache in ((), (state,), [state, state], (state, state.double()),
                          (state, state.transpose(-1, -2))):
                with self.assertRaises(ValueError):
                    protocol.backward(state, cache)
            exchange.assert_not_called()
        with self.assertRaisesRegex(ValueError, "outside"):
            protocol.exchange(state, 3, None)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_dense_rounds_keep_peer_order_and_transfer_lifetimes(self):
        """Feature: Rank-local dense RD planning.

        Description: Plan CP1, CP5 and CP8 endpoints and intermediate ranks.
        Expectation: Peers, stage order and the final transfer consumer stay unchanged.
        """
        cases = (
            (1, 0, ()),
            (5, 0, ((0, 1, None, False), (1, 2, None, False), (2, 4, None, False))),
            (5, 3, ((0, 4, 2, True), (1, None, 1, False))),
            (5, 4, ((0, None, 3, True), (1, None, 2, True), (2, None, 0, False))),
            (8, 1, ((0, 2, 0, False), (1, 3, None, False), (2, 5, None, False))),
        )
        for size, rank, expected in cases:
            with self.subTest(size=size, rank=rank):
                rounds = _make_protocol(rank, size).rounds
                actual = tuple((step.stage, step.destination, step.source, step.save_matrix) for step in rounds)
                self.assertEqual(actual, expected)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_mixed_reset_flags_and_invocation_binding(self):
        """Feature: Reset-aware RD invocation metadata.

        Description: Bind mixed and independent CP5 layouts with internal document resets.
        Expectation: Producer flags and immutable peer plans match document connectivity.
        """
        offsets = (0, 5, 8, 20)
        expected = ((True, False), (False, False), (True, False), (True, True), (False, False))
        shifts = ((1, None), (None, 0), (3, None), (4, 2), (None, 3))
        audit = rd.build_reset_plan(offsets, 4, 5)
        calls = []
        for rank, flags in enumerate(expected):
            protocol = _make_protocol(rank, 5)
            call = protocol.bind_sequence(PackedSequenceInfo(offsets, rank * 4, (rank + 1) * 4))
            self.assertEqual(call.plan.local_rank, rank)
            self.assertIs(call.plan.execution_rounds(rank), call.plan.local_rounds)
            self.assertEqual(call.plan.local_rounds, audit.execution_rounds(rank))
            self.assertEqual(call.plan.shift_peers(rank), shifts[rank])
            self.assertEqual(call.plan.is_transfer_active(rank), (False, False, False, True, True)[rank])
            self.assertEqual((call.needs_state_summary, call.needs_transition_summary), flags)
            self.assertEqual(call.plan.cache_count(rank) > 0, flags[1])
            calls.append(call)
        self.assertEqual(audit.local_rounds, ())
        call_a = calls[3]
        protocol = call_a.protocol
        call_b = protocol.bind_sequence(PackedSequenceInfo((0, 4, 8, 12, 16, 20), 12, 16))
        self.assertEqual(tuple(step.stage for step in call_a.plan.execution_rounds(2)), (0,))
        send = call_a.plan.execution_rounds(2)[0]
        self.assertEqual((send.stage, send.destination, send.source, send.save_matrix), (0, 3, None, False))
        self.assertIs(call_a.plan.execution_rounds(3), call_a.plan.local_rounds)
        step = call_a.plan.execution_rounds(3)[0]
        self.assertEqual((step.stage, step.destination, step.source, step.save_matrix), (0, None, 2, False))
        with self.assertRaises(FrozenInstanceError):
            step.source = 0
        self.assertEqual(call_b.plan.local_rounds, ())
        self.assertEqual(call_b.plan.shift_peers(3), (None, None))
        with self.assertRaises(FrozenInstanceError):
            call_a.plan.size = 6

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_empty_plan_omits_packing_and_rounds(self):
        """Feature: RD execution for an empty reset plan.

        Description: Run forward and backward with independent documents on CP1, CP2 and CP5.
        Expectation: No packed tensor or RD round is created, and both state boundaries are zero.
        """
        state = torch.ones(1, 1, 128, 128)
        for size in (1, 2, 5):
            with self.subTest(size=size):
                rank = size // 2
                protocol = _make_protocol(rank, size)
                info = PackedSequenceInfo(tuple(range(0, size * 4 + 1, 4)), rank * 4, (rank + 1) * 4)
                call = protocol.bind_sequence(info)
                with patch.object(rd.torch, "cat") as pack, \
                        patch.object(protocol, "exchange", return_value=None) as exchange:
                    incoming, cache = call.forward(state, state)
                    gradient = call.backward(state, cache)
                pack.assert_not_called()
                self.assertEqual(cache, ())
                self.assertEqual(exchange.call_count, 2)
                for item in exchange.call_args_list:
                    self.assertEqual(item.args[1:], (None, None))
                torch.testing.assert_close(incoming, torch.zeros_like(state), rtol=0, atol=0)
                torch.testing.assert_close(gradient, torch.zeros_like(state), rtol=0, atol=0)
                sparse_call = protocol.bind_sequence(info, state.shape)
                with patch.object(rd.torch, "cat") as pack, \
                        patch.object(rd.torch, "zeros_like") as zero, \
                        patch.object(rd.torch, "empty") as allocate, \
                        patch.object(rd.torch, "empty_like") as allocate_like:
                    incoming, cache = sparse_call.forward(None, None)
                    gradient = sparse_call.backward(None, cache)
                self.assertIsNone(incoming)
                self.assertIsNone(gradient)
                self.assertEqual(cache, ())
                for allocation in (pack, zero, allocate, allocate_like):
                    allocation.assert_not_called()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_early_receive_keeps_latest_state_for_later_sends(self):
        """Feature: RD state lifetime after an early final receive.

        Description: Rank 1 of CP8 receives once and then sends through later packed and narrow rounds.
        Expectation: The final narrow round and exclusive shift send its latest composed state.
        """
        protocol = _make_protocol(1, 8)
        call = protocol.bind_sequence(PackedSequenceInfo((0, 32), 4, 8))
        state = torch.full((1, 1, 128, 128), 2.0)
        matrix = (2 * torch.eye(128)).reshape(1, 1, 128, 128)
        preceding = torch.full_like(state, 3.0)
        received = torch.cat((preceding, torch.zeros_like(matrix)), dim=-1)
        with patch.object(protocol, "exchange", side_effect=(received, None, None, preceding)) as exchange:
            incoming, cache = call.forward(state, matrix)
        self.assertEqual(len(cache), 1)
        self.assertIs(cache[0], matrix)
        self.assertEqual([item.args[1:] for item in exchange.call_args_list],
                         [(2, 0), (3, None), (5, None), (2, 0)])
        for item in exchange.call_args_list[2:]:
            torch.testing.assert_close(item.args[0], torch.full_like(state, 8.0), rtol=0, atol=0)
        torch.testing.assert_close(incoming, preceding, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_reset_stops_transfer_copies_after_last_receive(self):
        """Feature: Reset-aware transfer copy pruning.

        Description: Run CP8 rank 5 with only its first RD receive active.
        Expectation: Only the original M is cached, no next M is copied, and the shift keeps the latest S.
        """
        protocol = _make_protocol(5, 8)
        call = protocol.bind_sequence(PackedSequenceInfo((0, 5, 8, 13, 16, 25, 32), 20, 24))
        state = torch.full((1, 1, 128, 128), 2.0)
        matrix = (2 * torch.eye(128)).reshape(1, 1, 128, 128)
        preceding = torch.full_like(state, 3.0)
        received = torch.cat((preceding, torch.zeros_like(matrix)), dim=-1)
        with patch.object(protocol, "exchange", side_effect=(received, preceding)) as exchange, \
                patch.object(rd, "_finish_round", wraps=rd._finish_round) as finish:
            _, cache = call.forward(state, matrix)
        self.assertEqual(tuple(step.stage for step in call.plan.local_rounds), (0,))
        self.assertEqual(len(cache), 1)
        self.assertIs(cache[0], matrix)
        self.assertFalse(finish.call_args.kwargs["save_m"])
        torch.testing.assert_close(exchange.call_args.args[0], torch.full_like(state, 8.0), rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_batch_plan_reuse_preserves_layout_and_invocation_identity(self):
        """Feature: Immutable packed plans shared across layers.

        Description: Bind a batch twice, then change head shape, state layout and documents.
        Expectation: Only the CPU schedule is shared; invocation shape and reset semantics stay independent.
        """
        rd.build_reset_plan.cache_clear()
        self.addCleanup(rd.build_reset_plan.cache_clear)
        info = PackedSequenceInfo((0, 5, 17, 32), 4, 8)
        first = _make_protocol(1, 8).bind_sequence(info, (1, 12, 128, 128))
        second = _make_protocol(1, 8).bind_sequence(info, (1, 6, 128, 128))
        self.assertIsNot(first, second)
        self.assertIs(first.plan, second.plan)
        self.assertEqual(first.state_shape, (1, 12, 128, 128))
        self.assertEqual(second.state_shape, (1, 6, 128, 128))
        expanded = info.ulysses(2, 1)
        different_layout = _make_protocol(0, 4).bind_sequence(expanded)
        self.assertIsNot(first.plan, different_layout.plan)
        self.assertEqual(different_layout.plan.local_length, 8)
        other = _make_protocol(1, 8).bind_sequence(PackedSequenceInfo((0, 32), 4, 8))
        self.assertFalse(first.plan.is_transfer_active(1))
        self.assertTrue(other.plan.is_transfer_active(1))
        with self.assertRaisesRegex(ValueError, "equal state-rank intervals"):
            _make_protocol(1, 4).bind_sequence(info)
        with self.assertRaises(FrozenInstanceError):
            first.plan.local_rank = 0
        for stop in range(1, 41):
            rd.build_reset_plan(PackedSequenceInfo((0, stop), 0, stop).documents, stop, 1, 0)
        self.assertIsNot(_make_protocol(1, 8).bind_sequence(info).plan, first.plan)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_producer_skips_dead_s_and_m_but_retains_head_gradient(self):
        """Feature: Independent forward summary and backward gradient consumers.

        Description: Mock S/M kernels at the first, middle, internally reset and last state ranks.
        Expectation: Only live summaries are produced, while the last rank still sends its head G.
        """
        functional = _load_summary_functional()
        kernel = functional.kda_state_summary_forward_from_prepared
        prepared = torch.ones(1, 4, 1, 128, dtype=torch.bfloat16)
        summary = torch.ones(1, 1, 128, 128, dtype=torch.float32)
        cases = ((0, (0, 20), False), (1, (0, 20), True), (2, (0, 9, 20), False))
        for rank, offsets, needs_matrix in cases:
            with self.subTest(rank=rank, offsets=offsets):
                info = PackedSequenceInfo(offsets, rank * 4, (rank + 1) * 4)
                call = _make_protocol(rank, 5).bind_sequence(info)
                kernel.reset_mock()
                kernel.return_value = (summary, summary if needs_matrix else None)
                state, matrix = functional._prepare_packed_kda_summary(
                    prepared, prepared, prepared, prepared, info, call, info.continues_next, 64)
                kernel.assert_called_once()
                self.assertEqual(kernel.call_args.kwargs["output_transition"], needs_matrix)
                self.assertEqual(kernel.call_args.args[0].shape[1], info.length - info.tail.start)
                self.assertIs(state, summary)
                torch.testing.assert_close(matrix, summary if needs_matrix else torch.zeros_like(summary))
        info = PackedSequenceInfo((0, 20), 16, 20)
        protocol = _make_protocol(4, 5)
        call = protocol.bind_sequence(info)
        kernel.reset_mock()
        state, matrix = functional._prepare_packed_kda_summary(
            prepared, prepared, prepared, prepared, info, call, False, 64)
        kernel.assert_not_called()
        self.assertTrue(info.continues_previous)
        torch.testing.assert_close(state, torch.zeros_like(summary), rtol=0, atol=0)
        torch.testing.assert_close(matrix, torch.zeros_like(summary), rtol=0, atol=0)
        with patch.object(protocol, "exchange", return_value=None) as exchange:
            call.backward(summary, ())
        exchange.assert_called_once_with(summary, 3, None)
        sparse_call = protocol.bind_sequence(info, summary.shape)
        with patch.object(torch.Tensor, "new_zeros") as zero:
            sparse_state, sparse_matrix = functional._prepare_packed_kda_summary(
                prepared, prepared, prepared, prepared, info, sparse_call, False, 64)
        self.assertIsNone(sparse_state)
        self.assertIsNone(sparse_matrix)
        kernel.assert_not_called()
        zero.assert_not_called()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_receive_shape_is_independent_of_optional_send(self):
        """Feature: Receive-only RD transport.

        Description: Receive with no payload and with a differently sized send tensor.
        Expectation: Only the requested receive is allocated and every operation is awaited.
        """
        exchange = _make_protocol(1, 3).exchange
        shape = (1, 2, 128, 128)
        payload = torch.ones(1, 2, 128, 256)
        receive_handle, send_handle = Mock(), Mock()
        with patch.object(rd.torch, "empty", wraps=torch.empty) as allocate, \
                patch.object(rd.dist, "P2POp") as operation, \
                patch.object(rd.dist, "batch_isend_irecv",
                             side_effect=([receive_handle], [send_handle, receive_handle])) as batch:
            receive_only = exchange(None, None, 0, receive_shape=shape)
            simultaneous = exchange(payload, 2, 0, receive_shape=shape)
            self.assertIsNone(exchange(None, None, None))
        self.assertEqual(receive_only.shape, shape)
        self.assertEqual(receive_only.dtype, torch.float32)
        self.assertEqual(simultaneous.shape, shape)
        self.assertEqual(allocate.call_count, 2)
        self.assertEqual(batch.call_count, 2)
        self.assertEqual(operation.call_count, 3)
        self.assertEqual(receive_handle.wait.call_count, 2)
        send_handle.wait.assert_called_once_with()
        with self.assertRaisesRegex(ValueError, "send requires"):
            exchange(None, 0, None)
        with self.assertRaisesRegex(ValueError, "requires a shape"):
            exchange(None, None, 0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_optional_boundaries_keep_independent_head_and_tail_consumers(self):
        """Feature: Independent optional H and D boundaries.

        Description: Execute both ends of a continuing CP2 document with only live summaries.
        Expectation: The first rank omits G/H, and the final rank omits S/M/D but still sends G.
        """
        summary = torch.ones(1, 2, 128, 128)
        for rank in (0, 1):
            protocol = _make_protocol(rank, 2)
            info = PackedSequenceInfo((0, 8), rank * 4, (rank + 1) * 4)
            call = protocol.bind_sequence(info, summary.shape)
            with patch.object(protocol, "exchange", side_effect=(None, summary) if rank == 0 else (summary, None)) \
                    as exchange, patch.object(rd.torch, "zeros_like") as zero:
                incoming, cache = call.forward(summary, summary) if rank == 0 else call.forward(None, None)
                gradient = call.backward(None if rank == 0 else summary, cache)
            zero.assert_not_called()
            self.assertEqual(cache, ())
            self.assertIs(incoming, None if rank == 0 else summary)
            self.assertIs(gradient, summary if rank == 0 else None)
            receives = exchange.call_args_list[1 - rank]
            self.assertIsNone(receives.args[0])
            self.assertEqual(receives.kwargs["receive_shape"], summary.shape)
            with patch.object(protocol, "exchange") as exchange:
                if rank == 0:
                    with self.assertRaisesRegex(ValueError, "exit-state consumer"):
                        call.forward(None, None)
                else:
                    with self.assertRaisesRegex(ValueError, "head-gradient consumer"):
                        call.backward(None, ())
                exchange.assert_not_called()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_bound_shape_validates_supplied_summaries_and_cache(self):
        """Feature: Optional RD input validation.

        Description: Supply bad tensor contracts and missing live summaries with a bound shape.
        Expectation: Metadata, supplied tensors and caches are checked before any exchange.
        """
        protocol = _make_protocol(1, 3)
        info = PackedSequenceInfo((0, 12), 4, 8)
        state = torch.ones(1, 2, 128, 128)
        call = protocol.bind_sequence(info, state.shape)
        for shape in ((1, 2, 128), (1, 2, 64, 128), (0, 2, 128, 128)):
            with self.assertRaisesRegex(ValueError, "invocation shape"):
                protocol.bind_sequence(info, shape)
        invalid = (state[:, :1], state.double(), state.transpose(-1, -2), state.to("meta"))
        with patch.object(protocol, "exchange") as exchange:
            with self.assertRaises(ValueError):
                call.forward(state, None)
            for tensor in invalid:
                with self.assertRaises(ValueError):
                    call.forward(tensor, tensor)
                with self.assertRaises(ValueError):
                    call.backward(tensor, (state,))
                with self.assertRaises(ValueError):
                    call.backward(state, (tensor,))
            for cache in ((), [state], (None,)):
                with self.assertRaises(ValueError):
                    call.backward(state, cache)
            dead = protocol.bind_sequence(PackedSequenceInfo((0, 4, 8, 12), 4, 8), state.shape)
            with self.assertRaises(ValueError):
                dead.forward(state[:, :1], state[:, :1])
            with self.assertRaises(ValueError):
                dead.backward(state[:, :1], ())
            with self.assertRaises(ValueError):
                dead.backward(None, (state,))
            exchange.assert_not_called()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_bound_shapes_survive_reverse_backward_of_two_live_calls(self):
        """Feature: Invocation-owned shape and cache lifetimes.

        Description: Bind different head counts and layouts on one executor, then reverse backward order.
        Expectation: Each call retains its own immutable shape, reset plan and forward cache.
        """
        protocol = _make_protocol(1, 3)
        state = torch.ones(1, 1, 128, 128)
        matrix = torch.eye(128).reshape(1, 1, 128, 128)
        original_matrix = matrix.clone()
        shape = [1, 1, 128, 128]
        first = protocol.bind_sequence(PackedSequenceInfo((0, 12), 4, 8), shape)
        shape[1] = 2
        second = protocol.bind_sequence(PackedSequenceInfo((0, 4, 8, 12), 4, 8), shape)
        preceding = torch.cat((state, matrix), dim=-1)
        with patch.object(protocol, "exchange", side_effect=(preceding, state, None)):
            incoming, cache = first.forward(state, matrix)
            empty_incoming, empty_cache = second.forward(None, None)
        self.assertIs(incoming, state)
        self.assertIsNone(empty_incoming)
        self.assertEqual(len(cache), 1)
        self.assertIs(cache[0], matrix)
        with patch.object(protocol, "exchange", side_effect=(None, state.clone(), None)):
            self.assertIsNone(second.backward(None, empty_cache))
            gradient = first.backward(state, cache)
        torch.testing.assert_close(gradient, state, rtol=0, atol=0)
        torch.testing.assert_close(matrix, original_matrix, rtol=0, atol=0)
        self.assertEqual(first.state_shape, (1, 1, 128, 128))
        self.assertEqual(second.state_shape, (1, 2, 128, 128))
        with self.assertRaises(FrozenInstanceError):
            first.state_shape = second.state_shape

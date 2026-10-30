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
"""CPU contracts for Actor DP row balancing and complete target transport."""
from dataclasses import dataclass, fields, replace
from unittest import TestCase
from unittest.mock import Mock, patch

import torch
from torch.nn import functional

from rl.roles.policy.dp_balance import balanced_row_indices, balance_experience, _cpu_metadata
from tests.ut.rl.policy.test_dynamic_batch import make_actor, make_experience


class TestActorBalance(TestCase):
    """Keep sample identities, global update sets, and strict batch limits intact."""

    def test_balancing_keeps_each_global_minibatch_and_partial_tail(self):
        """Every rank retains equal counts without mixing consecutive optimizer steps."""
        lengths = [[20, 18, 16, 12, 9], [3, 4, 5, 6, 7], [15, 14, 13, 8, 2]]
        orders = balanced_row_indices(lengths, 2)
        self.assertEqual(sorted(sum(orders, [])), list(range(15)))
        self.assertEqual([len(order) for order in orders], [5, 5, 5])
        for start in (0, 2, 4):
            actual = {index for order in orders for index in order[start:start + 2]}
            expected = {rank * 5 + row for rank in range(3) for row in range(start, min(start + 2, 5))}
            self.assertEqual(actual, expected)
        self.assertEqual(orders, balanced_row_indices(lengths, 2))

    def test_all_tensor_fields_and_identities_follow_the_same_permutation(self):
        """Mock transport supplies remote rows with a different padding width."""
        experiences = [make_experience([8, 7, 6, 5]), make_experience([3, 4, 3, 4])]
        experiences = [replace(exp, returns=exp.advantages + 1, values=exp.advantages - 1,
                               bootstrap_values=exp.rewards + 2, rollout_log_probs=exp.old_log_probs,
                               teacher_log_probs=exp.old_log_probs, teacher_keys=tuple(f"{rank}/{i}" for i in range(4)),
                               responses=tuple(f"{rank}/{i}" for i in range(4)))
                       for rank, exp in enumerate(experiences)]
        lengths = [[8, 7, 6, 5], [3, 4, 3, 4]]
        orders = balanced_row_indices(lengths, 2)
        tensor_fields = [field.name for field in fields(experiences[0])
                         if isinstance(getattr(experiences[0], field.name), torch.Tensor)]
        for rank, original in enumerate(experiences):
            received = [i for source in range(2) for i in orders[rank] if i // 4 == source]
            field_names = iter(tensor_fields)

            def _gather(output, local, group=None):
                if group is None:
                    output[:] = [local, local]
                    return
                self.assertEqual(group, "dp")
                output[:] = [(lengths[i], exp.sequences.shape[1], local[2], None)
                             for i, exp in enumerate(experiences)]

            def _exchange(output, outgoing, *, field_names=field_names, rank=rank, received=received, **kwargs):
                name = next(field_names)
                values = []
                for source in range(2):
                    value = getattr(experiences[source], name)
                    if value.ndim == 2:
                        value = functional.pad(value, (0, 8 - experiences[source].sequences.shape[1]))
                    values.append(value)
                expected_send = [i % 4 for order in orders for i in order if i // 4 == rank]
                torch.testing.assert_close(outgoing, values[rank][expected_send])
                self.assertEqual(kwargs["input_split_sizes"],
                                 [sum(i // 4 == rank for i in order) for order in orders])
                output.copy_(torch.cat(values)[received])

            def _metadata(experience, send_rows, group, *, rank=rank):
                del experience
                self.assertEqual(group, "dp")
                self.assertEqual(sum(map(len, send_rows)), 4)
                return [{name: tuple(getattr(experiences[source], name)[i % 4]
                                     for i in orders[rank] if i // 4 == source)
                         for name in ("trajectories", "responses", "teacher_keys")}
                        for source in range(2)]

            with patch("rl.roles.policy.dp_balance.dist.get_world_size", return_value=2), patch(
                "rl.roles.policy.dp_balance.dist.get_rank", return_value=rank,
            ), patch("rl.roles.policy.dp_balance.dist.all_gather_object", side_effect=_gather), patch(
                "rl.roles.policy.dp_balance.dist.all_to_all_single", side_effect=_exchange,
            ), patch("rl.roles.policy.dp_balance._exchange_metadata", side_effect=_metadata):
                actual = balance_experience(original, lengths[rank], 2, "dp")
            self.assertEqual(actual.responses, tuple(f"{i // 4}/{i % 4}" for i in orders[rank]))
            self.assertEqual(actual.teacher_keys, actual.responses)
            for position, index in enumerate(orders[rank]):
                source, row = divmod(index, 4)
                self.assertIs(actual.trajectories[position], experiences[source].trajectories[row])
                for name in tensor_fields:
                    expected = getattr(experiences[source], name)[row]
                    observed = getattr(actual, name)[position]
                    if expected.ndim:
                        observed = observed[:expected.numel()]
                    torch.testing.assert_close(observed, expected)
            self.assertEqual(original.sequences.shape[1], max(lengths[rank]))

    def test_metadata_tensor_views_are_compact_detached_and_independent(self):
        """A row view must not serialize its whole original batch storage."""
        @dataclass(frozen=True)
        class Record:
            """Nested metadata with a storage-sharing tensor view."""
            values: torch.Tensor
            metadata: dict

        backing = torch.arange(1024, dtype=torch.float32, requires_grad=True)
        original = Record(backing[10:15], {"views": [backing[20:23]]})
        copied = _cpu_metadata(original)
        self.assertEqual(copied.values.untyped_storage().nbytes(), 5 * 4)
        self.assertEqual(copied.metadata["views"][0].untyped_storage().nbytes(), 3 * 4)
        self.assertFalse(copied.values.requires_grad)
        copied.values.zero_()
        torch.testing.assert_close(original.values, torch.arange(10, 15, dtype=torch.float32))

    def test_balance_precedes_anchors_and_update_never_redistributes(self):
        """The explicit pre-anchor entry accepts targets without old/reference scores."""
        actor = make_actor(16)
        actor._balance_dp_tokens = True
        actor._dp_size = 2
        actor._dp_group_info = Mock(group="dp")
        raw = replace(make_experience([5, 8, 3, 7]), old_log_probs=None, reference_log_probs=None)
        with patch("rl.roles.policy.actor.synchronized_call", side_effect=lambda _label, fn: fn()), patch(
            "rl.roles.policy.actor.balance_experience", return_value=raw,
        ) as exchange:
            self.assertIs(actor.balance_rollout(raw), raw)
            exchange.assert_called_once_with(raw, [5, 8, 3, 7], 4, "dp")
        with self.assertRaisesRegex(ValueError, "advantages"):
            actor._balance_lengths(replace(raw, advantages=None))
        actor._dp_size = 1
        late_exchange = patch("rl.roles.policy.actor.balance_experience",
                              side_effect=AssertionError("late redistribution"))
        with late_exchange, patch(
            "rl.roles.policy.actor.hsdp_sync_stream",
        ), patch("rl.roles.policy.actor.clip_grad_norm_", torch.nn.utils.clip_grad_norm_):
            actor.update(make_experience([5, 8, 3, 7]))

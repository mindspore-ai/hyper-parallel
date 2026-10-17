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
"""Tests for the expert-major permutation used by MoE routing.

Integer ``argsort`` has no AI Core kernel, so the permutation is produced by
sorting a float32 key that encodes the slot index. The property that makes the
substitution safe is that the result equals ``argsort(stable=True)`` exactly,
not merely that it groups slots by expert, so that is what these pin.
"""
import torch

from hyper_parallel.components.modules.expert_sort import (
    SORT_KEY_FP32_LIMIT,
    expert_major_permutation,
)


def _stable_reference(flat_expert: torch.Tensor) -> torch.Tensor:
    """The integer stable sort the helper replaces."""
    return flat_expert.argsort(stable=True)


def test_matches_the_stable_integer_sort():
    """The permutation is the stable one, element for element."""
    torch.manual_seed(13)
    flat_expert = torch.randint(0, 16, (4096,))
    assert torch.equal(expert_major_permutation(flat_expert, 16),
                       _stable_reference(flat_expert))


def test_matches_across_expert_counts_and_slot_counts():
    """The encoding holds over the shapes routing actually produces."""
    torch.manual_seed(5)
    for num_experts in (1, 2, 8, 16, 64, 256):
        for slots in (1, 7, 64, 1024):
            flat_expert = torch.randint(0, num_experts, (slots,))
            got = expert_major_permutation(flat_expert, num_experts)
            assert torch.equal(got, _stable_reference(flat_expert)), (num_experts, slots)


def test_ties_keep_slot_order():
    """Slots routed to one expert stay in increasing slot order."""
    flat_expert = torch.tensor([2, 0, 2, 1, 0, 2])
    order = expert_major_permutation(flat_expert, 3)
    assert order.tolist() == [1, 4, 3, 0, 2, 5]


def test_groups_every_slot_by_expert():
    """The permuted expert ids are non-decreasing."""
    torch.manual_seed(7)
    flat_expert = torch.randint(0, 8, (512,))
    permuted = flat_expert[expert_major_permutation(flat_expert, 8)]
    assert torch.all(permuted[1:] >= permuted[:-1])


def test_permutation_dtype_and_contents():
    """An int64 permutation of every slot index, as the callers index with."""
    flat_expert = torch.randint(0, 4, (33,))
    order = expert_major_permutation(flat_expert, 4)
    assert order.dtype == torch.int64
    assert sorted(order.tolist()) == list(range(33))


def test_falls_back_past_the_float32_limit():
    """Too many experts times slots and the integer sort is used unchanged."""
    slots = 64
    num_experts = SORT_KEY_FP32_LIMIT  # product far past the limit
    flat_expert = torch.randint(0, 8, (slots,))
    got = expert_major_permutation(flat_expert, num_experts)
    assert torch.equal(got, _stable_reference(flat_expert))


def test_guard_boundary_is_inclusive_and_exact():
    """At exactly the limit the encoded key is still represented exactly."""
    slots = 1 << 12
    num_experts = SORT_KEY_FP32_LIMIT // slots
    assert num_experts * slots == SORT_KEY_FP32_LIMIT
    torch.manual_seed(11)
    flat_expert = torch.randint(0, num_experts, (slots,))
    assert torch.equal(expert_major_permutation(flat_expert, num_experts),
                       _stable_reference(flat_expert))


def test_single_expert_and_single_slot():
    """Degenerate shapes stay identity-like rather than raising."""
    assert expert_major_permutation(torch.zeros(5, dtype=torch.int64), 1).tolist() == [0, 1, 2, 3, 4]
    assert expert_major_permutation(torch.tensor([3]), 4).tolist() == [0]


def test_empty_input():
    """No routed slots yields an empty permutation."""
    order = expert_major_permutation(torch.empty(0, dtype=torch.int64), 8)
    assert order.numel() == 0


def test_non_positive_expert_count_falls_back():
    """An unusable bound must not be encoded; the integer sort still answers."""
    flat_expert = torch.randint(0, 4, (16,))
    assert torch.equal(expert_major_permutation(flat_expert, 0),
                       _stable_reference(flat_expert))

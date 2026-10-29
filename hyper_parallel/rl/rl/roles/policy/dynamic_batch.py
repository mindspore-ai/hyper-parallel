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
"""Token-budget scheduling over an existing RL optimizer mini-batch."""

from dataclasses import fields, replace

import torch

from rl.dataset.contracts import ExperienceBatch
from hyper_parallel.data.batching.packing import FirstFitPackingSelector


def plan_token_batches(lengths: list[int], token_budget: int) -> list[list[int]]:
    """Reuse shared selection, then bound rectangular model inputs including padding.

    Args:
        lengths: Complete prompt plus response lengths for one local mini-batch.
        token_budget: Padded input token limit for grouping rows. A complete row
            longer than this limit runs alone without truncation.

    Returns:
        Groups of row indices, covering every input exactly once.
    """
    if token_budget <= 0 or not lengths or any(length < 2 for length in lengths):
        raise ValueError("Dynamic token budget must be positive and sequences must contain at least two tokens")
    remaining = sorted(range(len(lengths)), key=lambda index: (-lengths[index], index))
    selector = FirstFitPackingSelector()
    batches = []
    while remaining:
        if lengths[remaining[0]] > token_budget:
            batches.append([remaining.pop(0)])
            continue
        selected = selector.select_samples_to_pack(
            [{"packing_length": lengths[index]} for index in remaining], token_budget,
        )
        # Keep overflow candidates for later groups instead of creating tiny padding-induced batches.
        # The longest selected row is first; rectangular allocation bounds the usable row count.
        selected = selected[:token_budget // lengths[remaining[0]]]
        batches.append([remaining[position] for position in selected])
        selected_set = set(selected)
        remaining = [index for position, index in enumerate(remaining) if position not in selected_set]
    return batches


def equalize_batch_count(batches: list[list[int]], count: int) -> list[list[int]]:
    """Split groups to match the DP maximum without duplicating or dropping rows."""
    if not len(batches) <= count <= sum(map(len, batches)):
        raise ValueError("Cannot equalize dynamic micro-batches without empty batches")
    batches = [list(group) for group in batches]
    while len(batches) < count:
        index = max(range(len(batches)), key=lambda position: len(batches[position]))
        group = batches[index]
        midpoint = len(group) // 2
        batches[index:index + 1] = [group[:midpoint], group[midpoint:]]
    return batches


def select_experience_rows(experience: ExperienceBatch, indices: list[int]) -> ExperienceBatch:
    """Keep all row-aligned targets and identities together; leave the source untouched."""
    selection = torch.tensor(indices, dtype=torch.long, device=experience.sequences.device)
    selected = {}
    for field in fields(experience):
        value = getattr(experience, field.name)
        if isinstance(value, torch.Tensor):
            selected[field.name] = value.index_select(0, selection.to(value.device))
        elif field.name in ("trajectories", "responses", "teacher_keys", "router_traces") and value:
            selected[field.name] = tuple(value[index] for index in indices)
    return replace(experience, **selected)

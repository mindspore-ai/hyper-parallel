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
"""Sort, retain and export recompute plans at fixed parallel degrees.

What a plan costs here is the cost model's own estimate of the step under
it, the same estimate the search ranks strategies with. A measured run can
disagree with it, so the sequence reads as what the cost model expects each
further recomputed layer to cost, not as a measured or guaranteed figure.
"""

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hyper_parallel.auto_parallel.sapp_nd.nd.dimensions import Dimensions
from hyper_parallel.auto_parallel.sapp_nd.recompute.candidate import MODES, RecomputeChoice, to_records, trainer_plan

# The configurations a plot shows when the run asked for no particular number.
DEFAULT_TOP = 20

# The columns of the CSV a retained sequence is written to.
_FIXED_COLUMNS = (
    "changed_layer", "full_layers", "memory_mb", "score", "recompute_mode", "recompute_layers",
    "layer_recompute", "stage_memory_mb", "stage_savings", "plan_scope",
)


def sort_recompute_results(results: Sequence[RecomputeChoice]) -> List[RecomputeChoice]:
    """Return recompute plans fastest first by the cost model's estimate, peak memory breaking ties.

    Args:
        results: Scored plans to order, which are not modified.

    Returns:
        The plans in ascending score order.

    Raises:
        ValueError: A plan has no finite score.
    """
    if any(result.score is None or not math.isfinite(result.score) for result in results):
        raise ValueError("recompute results must have finite performance scores before sorting")
    return sorted(results, key=lambda result: (result.score, result.memory))


def _layer_settings(choice: RecomputeChoice) -> Dict[int, Tuple[Any, str, Any]]:
    """Expand a named plan per layer, independently of how its consecutive ranges are grouped."""
    defined: Dict[int, Tuple[Any, str, Any]] = {}
    for item in choice.ranges:
        if item.first < 0 or item.count <= 0 or item.mode not in MODES:
            raise ValueError("recompute potentials require valid ranges with a named mode per layer")
        for index in range(item.first, item.first + item.count):
            if index in defined:
                raise ValueError("recompute layer ranges must not overlap")
            defined[index] = (item.kind, item.mode, item.option)
    return defined


def _one_more_full(
    defined: Mapping[int, Tuple[Any, str, Any]], proposed: Mapping[int, Tuple[Any, str, Any]]
) -> bool:
    """Whether *proposed* runs one more layer of one kind full and leaves every other count alone."""
    if any(proposed[index][0] != defined[index][0] or proposed[index][2].link_bandwidth for index in defined):
        return False
    before = Counter((kind, mode) for kind, mode, _ in defined.values())
    after = Counter((kind, mode) for kind, mode, _ in proposed.values())
    removed, added = before - after, after - before
    if sum(removed.values()) != 1 or sum(added.values()) != 1:
        return False
    old_kind, old_mode = next(iter(removed))
    new_kind, new_mode = next(iter(added))
    return old_kind == new_kind and old_mode != "full" and new_mode == "full"


def keep_recompute_potential(
    current: RecomputeChoice,
    kept: List[RecomputeChoice],
    results: Sequence[RecomputeChoice],
) -> Optional[RecomputeChoice]:
    """Append and return the fastest unused plan that recomputes one more layer of one kind.

    Args:
        current: The plan whose degrees, kinds and other mode counts are kept.
        kept: The plans retained so far; a qualifying plan is appended in place.
        results: The scored alternatives, which need not be sorted.

    Returns:
        The appended plan, or ``None`` when no unused qualifying plan is left.

    Raises:
        ValueError: A plan has no finite score, or no valid plan per layer.
    """
    defined = _layer_settings(current)
    previous = [(choice.dimensions, _layer_settings(choice)) for choice in kept]
    for result in sort_recompute_results(results):
        if result.dimensions != current.dimensions:
            continue
        proposed = _layer_settings(result)
        if proposed.keys() != defined.keys() or not _one_more_full(defined, proposed):
            continue
        if any(dimensions == result.dimensions and layers == proposed for dimensions, layers in previous):
            continue
        kept.append(result)
        return result
    return None


def recompute_next_plot_groups(
    scored_space: Sequence[tuple],
    results: Mapping[Dimensions, Sequence[RecomputeChoice]],
    top_num: Optional[int] = None,
) -> List[Tuple[Dimensions, List[RecomputeChoice]]]:
    """Select the plans below each next strategy's score, and the next best beyond it.

    Args:
        scored_space: The parallel configurations in the search's order.
        results: The scored alternatives of each configuration, its own plan
            among them.
        top_num: How many configurations to show; :data:`DEFAULT_TOP` when
            omitted.

    Returns:
        The groups in the search's order, each sorted by score. A group holds
        every plan the cost model scores below the next strategy, and the
        first plan at or above it. It keeps at least two plans where there
        are two, so a group shows an alternative even when the scores tie.
        The last group takes the mean size of the groups before it, rounded
        half up, at least two and at most what it has; with one
        configuration alone it takes up to two plans.
    """
    top = scored_space[:max(0, DEFAULT_TOP if top_num is None else top_num)]
    if not top:
        return []
    groups = []
    for entry, following in zip(top, top[1:]):
        config = entry[0]
        choices = sort_recompute_results(results.get(config, ()))
        count = sum(choice.score < following[2] for choice in choices)
        groups.append((config, choices[:max(2, count + 1)]))
    count = math.floor(sum(len(choices) for _, choices in groups) / len(groups) + 0.5) if groups else 2
    config = top[-1][0]
    groups.append((config, sort_recompute_results(results.get(config, ()))[:max(2, count)]))
    return groups


def recompute_plot_data(
    groups: Sequence[Tuple[Dimensions, Sequence[RecomputeChoice]]],
) -> List[Tuple[Dimensions, List[dict]]]:
    """Prepare the selected plans as chart records, without changing their configuration ranks.

    Args:
        groups: The selected plans, grouped by fixed parallel configuration.

    Returns:
        The groups with each plan's score, its parts, its peak memory, the
        count of fully recomputed layers and the plan the trainer reads.
    """
    prepared = []
    for config, choices in groups:
        records = []
        for choice in choices:
            mode, layers = trainer_plan(choice) if choice.ranges or choice.mode is not None else (None, None)
            records.append({
                "score": choice.score,
                "parts": choice.score_parts,
                "memory": choice.memory,
                "mode": mode,
                "layers": layers,
                "full_layers": _full_layers(choice),
            })
        prepared.append((config, records))
    return prepared


def _full_layers(choice: RecomputeChoice) -> Optional[int]:
    """How many layers the plan fully recomputes; ``None`` for a plan that is not stated per layer."""
    if not choice.ranges:
        return None
    return sum(item.count for item in choice.ranges if item.mode == "full")


def _plan_scope(choice: RecomputeChoice) -> str:
    """How wide a plan is stated: per layer, for the whole model, with offload, or not at all."""
    if any(item.option.link_bandwidth for item in choice.ranges):
        return "offload"
    if choice.ranges:
        return "per_layer"
    return "whole_model" if choice.mode is not None else "unavailable"


def _csv_row(
    choice: RecomputeChoice,
    rank: Any,
    step: int,
    previous: Mapping[int, Tuple[Any, str, Any]],
) -> Dict[str, Any]:
    """One row of the sequence: the plan, what it changed against the step before, and its estimate."""
    defined = _layer_settings(choice)
    changed = [index for index, setting in defined.items() if step and setting != previous[index]]
    mode, layers = trainer_plan(choice) if choice.ranges or choice.mode is not None else ("", {})
    row = dict(choice.dimensions)
    row.update({
        "TOP": rank,
        "step": step,
        "changed_layer": ",".join(map(str, changed)),
        "full_layers": _full_layers(choice) if choice.ranges else "",
        "memory_mb": choice.memory,
        "score": choice.score,
        "recompute_mode": mode,
        "recompute_layers": json.dumps(layers),
        "layer_recompute": json.dumps(to_records(choice)),
        "stage_memory_mb": json.dumps(choice.stage_memory),
        "stage_savings": json.dumps(choice.stage_savings),
        "plan_scope": _plan_scope(choice),
    })
    return row


def write_recompute_csv(potentials: Mapping[Dimensions, Sequence[RecomputeChoice]], path: str) -> None:
    """Replace a CSV with each fixed strategy's retained recompute sequence.

    The score of every row is the cost model's estimate of the step under
    that plan, so a difference between two rows is what the cost model
    expects the extra recomputed layer to cost.

    Args:
        potentials: The retained plans in sequence order, grouped by
            parallel configuration.
        path: The CSV to write, whose parent directory is created where it
            is missing.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    degrees = list(dict.fromkeys(
        name for choices in potentials.values() for choice in choices for name, _ in choice.dimensions
    ))
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["TOP", "step"] + degrees + list(_FIXED_COLUMNS))
        writer.writeheader()
        for config, choices in potentials.items():
            previous: Mapping[int, Tuple[Any, str, Any]] = {}
            for step, choice in enumerate(choices):
                writer.writerow(_csv_row(choice, config.rank, step, previous))
                previous = _layer_settings(choice)

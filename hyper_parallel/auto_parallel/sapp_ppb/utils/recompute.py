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
"""Recomputation taxonomy and conversion helpers between internal dicts and the YAML schema."""
from enum import IntEnum
from typing import Any, Dict, List, Optional

from hyper_parallel.auto_parallel.sapp_ppb.utils.logger import logger

TYPE = IntEnum("RecomputeType", ["NONE", "SLCT", "COMM", "BOTH", "FULL"], start=0)
OFFSET = "offset"

DEFAULT_COEF = {
    TYPE.NONE: 0,
    TYPE.SLCT: 0.04,
    TYPE.COMM: 0.125,
    TYPE.BOTH: 0.165,
    TYPE.FULL: 0.5,
}

YAML_NAME = {
    TYPE.NONE: "",
    TYPE.COMM: "select_comm_recompute",
    TYPE.SLCT: "select_recompute",
    TYPE.BOTH: "both_comm_select",
    TYPE.FULL: "recompute",
}

JSON_MEMORY_NAME = {
    TYPE.NONE: "memory_activation",
    TYPE.COMM: "memory_select_comm",
    TYPE.BOTH: "memory_both_comm_select",
    TYPE.SLCT: "memory_select_rec",
    TYPE.FULL: "memory_recompute",
}

JSON_MEMORY_NAME_ALIGNED = {
    TYPE.NONE: "memory_activation ",
    TYPE.COMM: "memory_select_comm",
    TYPE.BOTH: "memory_both_comm_select",
    TYPE.SLCT: "memory_select_rec ",
    TYPE.FULL: "memory_recompute  ",
}


JSON_TIME_NAME = {
    TYPE.NONE: "backward_time",
    TYPE.COMM: "select_comm_time",
    TYPE.BOTH: "both_comm_select_time",
    TYPE.SLCT: "select_rec_time",
    TYPE.FULL: "recompute_time ",
}

JSON_COEF_NAME = {
    TYPE.NONE: "backward_coef",
    TYPE.SLCT: "select_rec_coef",
    TYPE.BOTH: "both_comm_select_coef",
    TYPE.COMM: "select_comm_coef",
    TYPE.FULL: "recompute_coef",
}


def sums(rec_dict: Dict[TYPE, int]) -> int:
    """Return the sum of layer counts across every :class:`TYPE` key in ``rec_dict``."""
    x = 0
    for r in TYPE:
        x += rec_dict[r]
    return x


def zero_if_none_var(v: Any, i: int, s: int) -> int:
    """Read ``int(v[i][s].varValue)`` guarding against ``None`` at any step."""
    if v is not None and v[i][s].varValue is not None:
        return int(v[i][s].varValue)
    return 0


def zero_if_none(v: Any, i: int, s: int) -> int:
    """Read ``int(v[i][s])`` guarding against ``None`` at any step."""
    if v is not None and v[i][s] is not None:
        return int(v[i][s])
    return 0


def yaml_from_internal(vpp: int, pp: int,
                       lp_variables: Dict[TYPE, Any],
                       nass: List[List[int]]) -> Dict[str, List[List[int]]]:
    """Convert solver variables into the MindFormers YAML schema.

    Args:
        vpp: Number of virtual pipeline (VPP) chunks.
        pp: Number of physical pipeline stages.
        lp_variables: Solver variables keyed by :class:`TYPE`.
        nass: Naive layer assignments per ``(vpp_chunk, stage)``.

    Returns:
        A mapping from YAML field name to a 2-D list of ``(vpp, pp)`` integers.
    """
    slct_is = 0
    comm_is = 0
    both_is = 0
    full_is = 0

    yaml_out: Dict[str, List[List[int]]] = {
        OFFSET: [],
        YAML_NAME[TYPE.FULL]: [],
        YAML_NAME[TYPE.SLCT]: [],
        YAML_NAME[TYPE.COMM]: [],
    }
    logger.debug("pp = %s, vpp = %s", pp, vpp)
    for i in range(vpp):
        for _, v in yaml_out.items():
            v.append([])
        for s in range(pp):
            gass_i_s = 0
            for r in TYPE:
                gass_i_s += zero_if_none_var(lp_variables[r], i, s)
            slct_is = zero_if_none_var(lp_variables[TYPE.SLCT], i, s)
            comm_is = zero_if_none_var(lp_variables[TYPE.COMM], i, s)
            both_is = zero_if_none_var(lp_variables[TYPE.BOTH], i, s)
            full_is = zero_if_none_var(lp_variables[TYPE.FULL], i, s)
            yaml_out[OFFSET][i].append(gass_i_s - nass[i][s])
            yaml_out[YAML_NAME[TYPE.FULL]][i].append(full_is)
            yaml_out[YAML_NAME[TYPE.SLCT]][i].append(slct_is + both_is + full_is)
            yaml_out[YAML_NAME[TYPE.COMM]][i].append(comm_is + both_is + full_is)

    logger.debug("yaml = %s", yaml_out)
    return yaml_out


def internal_from_yaml(vpp: int, pp: int,
                       yaml_in: Dict[str, Any],
                       nass: List[List[int]]) -> Dict[TYPE, List[List[int]]]:
    """Convert a MindFormers YAML schema back into per-type layer counts.

    Args:
        vpp: Number of virtual pipeline chunks.
        pp: Number of physical pipeline stages.
        yaml_in: YAML mapping with ``offset`` and per-recomputation-type fields.
        nass: Naive layer assignments per ``(vpp_chunk, stage)``.

    Returns:
        A mapping from :class:`TYPE` to a 2-D list of ``(vpp, pp)`` integers.
    """
    slct_is = 0
    comm_is = 0
    full_is = 0
    layer_per_recompute: Dict[TYPE, List[List[int]]] = {r: [] for r in TYPE}
    if yaml_in[OFFSET] == 0:
        yaml_in[OFFSET] = [[0] * pp for _ in range(vpp)]

    for rec in [TYPE.SLCT, TYPE.COMM, TYPE.FULL]:
        if (
                YAML_NAME[rec] not in yaml_in
                or yaml_in[YAML_NAME[rec]] is False
                or yaml_in[YAML_NAME[rec]] == 0
        ):
            yaml_in[YAML_NAME[rec]] = [[0] * pp for _ in range(vpp)]
        if yaml_in[YAML_NAME[rec]] is True:
            yaml_in[YAML_NAME[rec]] = [
                [a + b for a, b in zip(list1, list2)]
                for list1, list2 in zip(nass, yaml_in[OFFSET])
            ]

    for i in range(vpp):
        for _, v in layer_per_recompute.items():
            v.append([])
        for s in range(pp):
            slct_is = zero_if_none(yaml_in[YAML_NAME[TYPE.SLCT]], i, s)
            comm_is = zero_if_none(yaml_in[YAML_NAME[TYPE.COMM]], i, s)
            full_is = zero_if_none(yaml_in[YAML_NAME[TYPE.FULL]], i, s)
            layer_per_recompute[TYPE.FULL][i].append(full_is)
            layer_per_recompute[TYPE.BOTH][i].append(
                max(min(slct_is - full_is, comm_is - full_is), 0)
            )
            layer_per_recompute[TYPE.SLCT][i].append(
                max(slct_is - full_is - layer_per_recompute[TYPE.BOTH][i][s], 0)
            )
            layer_per_recompute[TYPE.COMM][i].append(
                max(comm_is - full_is - layer_per_recompute[TYPE.BOTH][i][s], 0)
            )
            layer_per_recompute[TYPE.NONE][i].append(
                (
                    yaml_in[OFFSET][i][s]
                    + nass[i][s]
                    - layer_per_recompute[TYPE.FULL][i][s]
                    - layer_per_recompute[TYPE.BOTH][i][s]
                    - layer_per_recompute[TYPE.SLCT][i][s]
                    - layer_per_recompute[TYPE.COMM][i][s]
                )
            )

    logger.debug("layer_per_recompute = %s", layer_per_recompute)
    return layer_per_recompute


def to_list(rec_dict: Dict[TYPE, Any]) -> List[Any]:
    """Return the values of ``rec_dict`` in :class:`TYPE` enum order."""
    return list(rec_dict.values())


def right_extend(ll: List[List[int]], n: int) -> List[List[int]]:
    """Return ``ll`` extended by appending each of ``range(n)`` to every sub-list.

    Args:
        ll: List of partially built index vectors.
        n: Number of values (``0..n-1``) to append.

    Returns:
        A new list where each input sub-list appears ``n`` times, each with one of the new values.
    """
    all_l: List[List[int]] = []
    for i in range(n):
        for sublist in ll:
            all_l += [sublist + [i]]
    return all_l


def make_all_indexes_local(used_rec: Dict[TYPE, bool], num_of_interleave: int,
                           all_indexes: List[List[int]], r: TYPE) -> List[List[int]]:
    """Recursive helper behind :func:`make_all_indexes`.

    Args:
        used_rec: Which recomputation types are currently considered.
        num_of_interleave: Interleave (VPP) degree.
        all_indexes: Accumulated partial assignments.
        r: The current :class:`TYPE` being processed.

    Returns:
        The completed list of index vectors once the last :class:`TYPE` is reached.
    """
    if r >= len(TYPE) - 1:
        if used_rec[r]:
            all_indexes = right_extend(all_indexes, num_of_interleave)
        return all_indexes
    if used_rec[r]:
        return make_all_indexes_local(
            used_rec,
            num_of_interleave,
            right_extend(all_indexes, num_of_interleave),
            TYPE(r + 1),
        )
    return make_all_indexes_local(used_rec, num_of_interleave, all_indexes, TYPE(r + 1))


def make_all_indexes(used_rec: Dict[TYPE, bool], num_of_interleave: int) -> List[List[int]]:
    """Enumerate all per-recomputation-type assignments across ``num_of_interleave`` chunks."""
    return make_all_indexes_local(used_rec, num_of_interleave, [[]], TYPE.NONE)


def recomputes_from_indexes(used_rec: Dict[TYPE, bool],
                            indexes: List[List[int]]) -> List[Dict[TYPE, Optional[int]]]:
    """Decode index vectors produced by :func:`make_all_indexes` into per-type dictionaries."""
    recomputes: List[Dict[TYPE, Optional[int]]] = []
    for idx in indexes:
        recompute: Dict[TYPE, Optional[int]] = {r: None for r in TYPE}
        for r in TYPE:
            if used_rec[r]:
                recompute[r] = idx[0]
                idx.pop(0)
        recomputes.append(recompute)
    return recomputes


def average(rec_list: List[Dict[TYPE, Optional[float]]]) -> Dict[TYPE, Optional[float]]:
    """Return the per-type mean of a list of per-type recomputation dicts.

    Args:
        rec_list: Mapping from :class:`TYPE` to a numeric value or ``None``.

    Returns:
        A new dict holding the arithmetic mean for each :class:`TYPE` (``None`` propagates).
    """
    num = len(rec_list)
    if num == 0:
        return rec_list
    rec_1 = rec_list.pop(0)
    for rec_i in rec_list:
        for r in TYPE:
            if rec_1[r] is not None and rec_i[r] is not None:
                rec_1[r] = rec_1[r] + rec_i[r]
            elif not (rec_1[r] is None and rec_i[r] is None):
                logger.warning(
                    "WARNING: Recomputation %s is not taken into consideration by all body layers",
                    r.name,
                )
    for r in TYPE:
        if rec_1[r] is not None:
            rec_1[r] = rec_1[r] / num
    return rec_1


def assign_used(values: List[int], unused_rec: List[TYPE]) -> Dict[TYPE, Optional[int]]:
    """Associate each value with its recomputation type, skipping ``unused_rec`` entries."""
    assignment: Dict[TYPE, Optional[int]] = {r: None for r in TYPE}
    value_idx = 0
    for r in TYPE:
        if r not in unused_rec:
            assignment[r] = values[value_idx]
            value_idx += 1
    return assignment


def get_used_list(recompute_considered: Dict[TYPE, bool]) -> List[TYPE]:
    """Return recomputation types flagged as enabled in ``recompute_considered``."""
    used_rec: List[TYPE] = []
    for rec in TYPE:
        if recompute_considered[rec]:
            used_rec.append(rec)
    return used_rec


def get_unused_list(recompute_considered: Dict[TYPE, bool]) -> List[TYPE]:
    """Return recomputation types flagged as disabled (or missing) in ``recompute_considered``."""
    unused_rec: List[TYPE] = []
    for rec in TYPE:
        if rec not in recompute_considered or not recompute_considered[rec]:
            unused_rec.append(rec)
    return unused_rec


def least_recomputed(recompute_considered: Dict[TYPE, bool]) -> TYPE:
    """Return the lowest-index enabled recomputation :class:`TYPE`."""
    rec = TYPE.NONE
    for r in TYPE:
        if recompute_considered[r]:
            rec = r
            break
    return rec


def most_recomputed(recompute_considered: Dict[TYPE, bool]) -> TYPE:
    """Return the highest-index enabled recomputation :class:`TYPE`."""
    rec = TYPE.FULL
    for r in TYPE:
        if recompute_considered[r]:
            rec = r
    return rec

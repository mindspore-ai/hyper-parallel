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
"""Candidate strategy generation and constraint filtering.

When an SAPP-ND ``Parallelize`` instance is available the search-space
enumeration delegates to ``run_generation_to_ordering()``; otherwise an
analytical fallback enumerates DP/TP/PP/CP/MBN combinations.
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional

from hyper_parallel.auto_parallel.dense_tuner.config import (
    ConstraintConfig,
    HardwareConfig,
    ModelConfig,
    SearchSpaceConfig,
)
from hyper_parallel.auto_parallel.dense_tuner.strategy import (
    ParallelStrategy,
)

logger = logging.getLogger(__name__)


def _divisors(n: int, min_val: int = 1, max_val: int = 0) -> List[int]:
    """Return sorted divisors of n within [min_val, max_val].

    Args:
        n: Number to find divisors of.
        min_val: Minimum divisor value (inclusive).
        max_val: Maximum divisor value (inclusive). 0 means n.

    Returns:
        Sorted list of divisors.
    """
    if max_val <= 0:
        max_val = n
    result = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            if min_val <= i <= max_val:
                result.append(i)
            j = n // i
            if j != i and min_val <= j <= max_val:
                result.append(j)
    return sorted(result)


def _is_power_of_2(n: int) -> bool:
    """Check if n is a power of 2.

    Args:
        n: Number to check.

    Returns:
        True if n is a power of 2.
    """
    return n > 0 and (n & (n - 1)) == 0


def _enumerate_tp_values(
    num_devices: int,
    model: ModelConfig,
    search_space: SearchSpaceConfig,
) -> List[int]:
    """Enumerate valid TP values.

    TP must be a power of 2 and divide num_devices.
    TP must not exceed num_kv_heads.

    Args:
        num_devices: Total number of devices.
        model: Model configuration.
        search_space: Search space configuration.

    Returns:
        List of valid TP values.
    """
    if search_space.tp_range:
        return [tp for tp in search_space.tp_range if _is_power_of_2(tp) and tp <= model.effective_num_kv_heads]
    kv_heads = model.effective_num_kv_heads
    candidates = [d for d in _divisors(num_devices) if _is_power_of_2(d)]
    return [tp for tp in candidates if tp <= kv_heads]


def _enumerate_pp_values(
    num_devices: int,
    model: ModelConfig,
    search_space: SearchSpaceConfig,
) -> List[int]:
    """Enumerate valid PP values.

    PP must divide num_devices and must not exceed num_layers.

    Args:
        num_devices: Total number of devices.
        model: Model configuration.
        search_space: Search space configuration.

    Returns:
        List of valid PP values.
    """
    if search_space.pp_range:
        max_pp = model.num_layers if model.num_layers > 0 else num_devices
        return [pp for pp in search_space.pp_range if 0 < pp <= max_pp and num_devices % pp == 0]
    max_pp = model.num_layers if model.num_layers > 0 else num_devices
    return [pp for pp in _divisors(num_devices) if pp <= max_pp]


def _enumerate_cp_values(
    num_devices: int,
    search_space: SearchSpaceConfig,
) -> List[int]:
    """Enumerate valid CP values.

    CP must divide num_devices.

    Args:
        num_devices: Total number of devices.
        search_space: Search space configuration.

    Returns:
        List of valid CP values.
    """
    if not search_space.enable_cp:
        return [1]
    if search_space.cp_range:
        return [cp for cp in search_space.cp_range if cp > 0 and num_devices % cp == 0]
    return _divisors(num_devices)


def _enumerate_mbn_values(
    gbs: int,
    dp: int,
    pp: int,
    search_space: SearchSpaceConfig,
) -> List[int]:
    """Enumerate valid micro-batch number values.

    MBN = GBS / (DP * MBS), where MBS >= 1.
    MBN must be >= PP when PP > 1.

    Args:
        gbs: Global batch size.
        dp: Data parallelism degree.
        pp: Pipeline parallelism degree.
        search_space: Search space configuration.

    Returns:
        List of valid MBN values.
    """
    if gbs <= 0 or dp <= 0:
        return [1]
    effective_gbs = gbs // dp
    if effective_gbs <= 0:
        return []
    candidates = _divisors(effective_gbs)
    if pp > 1:
        candidates = [mbn for mbn in candidates if mbn >= pp]
    if search_space.micro_batch_num_range:
        return [mbn for mbn in search_space.micro_batch_num_range if mbn in candidates]
    return candidates


def _build_sapp_dimensions(search_space: SearchSpaceConfig) -> Optional[List[Any]]:
    """Build SAPP-ND Dimension list from SearchSpaceConfig.

    Args:
        search_space: Search space configuration.

    Returns:
        List of Dimension objects for SAPP-ND, or None for default ALL_DIMS.
    """
    # pylint: disable=import-outside-toplevel
    from hyper_parallel.auto_parallel.sapp_nd.nd.dimensions import (
        ALL_DIMS,
        CP,
        PP,
        VPP,
    )

    if not search_space.enable_pp:
        return [d for d in ALL_DIMS if d is not PP and d is not VPP]
    if not search_space.enable_cp:
        return [d for d in ALL_DIMS if d is not CP]
    return None


def _patch_evaluator_set_config(parallelize: Any) -> None:
    """Patch EvaluatorV2.set_config to also sync _overhead_obj._ccfg.

    SAPP-ND's ``estimate_peak()`` deep-copies the ccfg, processes it,
    then restores it and syncs ``_overhead_obj._ccfg`` — but only
    *after* the call returns.  When ``memory_estim()`` calls
    ``set_config()`` before the next ``estimate_peak()``, the
    overhead object's ccfg reference becomes stale, causing an
    ``IndexError`` in ``_bwd_overhead.py`` when VPP > 1.

    This patch ensures ``set_config`` also updates
    ``_overhead_obj._ccfg`` so the overhead module always sees the
    current configuration.

    Args:
        parallelize: A ``Parallelize`` instance whose internal
            ``ParallelizeLayer.mem_eval`` will be patched.
    """
    evaluator = parallelize.instance.mem_eval
    original_set_config = evaluator.set_config

    def _synced_set_config(config: Any) -> None:
        original_set_config(config)
        evaluator._overhead_obj._ccfg = config

    evaluator.set_config = _synced_set_config


def _matches_search_ranges(
    strategy: ParallelStrategy,
    search_space: SearchSpaceConfig,
) -> bool:
    """Check whether *strategy* matches range constraints in *search_space*.

    SAPP-ND's ``Parallelize`` does not respect per-dimension ranges
    (``dp_range``, ``tp_range``, etc.), so this function filters
    candidates after generation.

    Args:
        strategy: A candidate strategy.
        search_space: Search space configuration with range constraints.

    Returns:
        True if the strategy matches all specified ranges.
    """
    if search_space.dp_range and strategy.dp_degree not in search_space.dp_range:
        return False
    if search_space.tp_range and strategy.tp_degree not in search_space.tp_range:
        return False
    if search_space.pp_range and strategy.pp_degree not in search_space.pp_range:
        return False
    if search_space.cp_range and strategy.cp_degree not in search_space.cp_range:
        return False
    if search_space.micro_batch_num_range and strategy.micro_batch_num not in search_space.micro_batch_num_range:
        return False
    return True


def generate_candidates_sapp(
    parallelize: Any,
    search_space: SearchSpaceConfig,
    constraints: ConstraintConfig,  # pylint: disable=unused-argument
) -> List[ParallelStrategy]:
    """Generate candidates using SAPP-ND ``Parallelize`` entry point.

    Uses the ``Parallelize`` instance for search-space enumeration,
    memory estimation (via ``generate_search_space()``) and
    performance scoring (via ``order_search_space()``), then converts
    the scored space into ``ParallelStrategy`` objects.

    When ``search_space.enable_fsdp`` is True, each returned candidate
    with ``dp_degree > 1`` is additionally duplicated with
    ``fsdp_enabled=True``.

    Args:
        parallelize: A ``Parallelize`` instance (from
            ``TunerConfig.create_sapp_parallelize()``).
        search_space: Search space configuration.
        constraints: Constraint configuration (used for GBS in the
            scored-space conversion).

    Returns:
        List of ParallelStrategy objects with memory and performance
        estimates from SAPP-ND.
    """
    _patch_evaluator_set_config(parallelize)
    parallelize.instance.enable_debug = False
    ccfg = parallelize.config.ccfg
    if "vp" not in ccfg.__dict__ or ccfg.__dict__.get("vp", 0) < 1:
        ccfg.vp = 1
    parallelize.bound_space()
    space = parallelize.generate_search_space(
        folder="", threads_num=None
    )
    scored_space, _ = parallelize.order_search_space(
        space, threads_num=None, cache_file=None
    )

    perf_scores: dict = {}
    for item in scored_space:
        dims = item[0]
        score = item[2]
        perf_scores[dims] = score

    strategies = []
    for dims, peak_mem in space:
        s = ParallelStrategy.from_sapp_dimensions(dims)
        s.memory_cost.total = peak_mem
        s.extra["sapp_dims"] = dims
        score = perf_scores.get(dims)
        if score is not None:
            s.extra["sapp_perf_score"] = score
        if _matches_search_ranges(s, search_space):
            strategies.append(s)

        if search_space.enable_fsdp and s.dp_degree > 1:
            fsdp_s = ParallelStrategy.from_sapp_dimensions(dims)
            fsdp_s.fsdp_enabled = True
            fsdp_s.extra["sapp_dims"] = dims
            if score is not None:
                fsdp_s.extra["sapp_perf_score"] = score
            if _matches_search_ranges(fsdp_s, search_space):
                strategies.append(fsdp_s)

    logger.info(
        "SAPP-ND generated %d candidate strategies via Parallelize "
        "(fsdp=%s, scored=%d)",
        len(strategies),
        search_space.enable_fsdp,
        len(perf_scores),
    )
    return strategies


def collect_nd_visualization_output(output_dir: str) -> List[str]:
    """Collect SAPP-ND search-space visualization files into *output_dir*.

    ND's ``ParallelizeLayer`` writes ``output/results.pdf`` and
    ``output/debug_*.csv`` when verbosity >= 2.  This function moves
    those files into the given output directory.

    Args:
        output_dir: Destination directory for ND visualization files.

    Returns:
        List of collected file paths.
    """
    # pylint: disable=import-outside-toplevel
    import glob
    import shutil

    collected: List[str] = []
    os.makedirs(output_dir, exist_ok=True)

    nd_output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "sapp_nd", "nd", "output",
    )
    if not os.path.isdir(nd_output_dir):
        return collected

    for pattern in ["results.pdf", "debug_*.csv"]:
        for src in sorted(glob.glob(os.path.join(nd_output_dir, pattern))):
            dst = os.path.join(output_dir, os.path.basename(src))
            shutil.copy2(src, dst)
            collected.append(dst)

    return collected


def generate_candidates(
    hardware: HardwareConfig,
    model: ModelConfig,
    search_space: SearchSpaceConfig,
    constraints: ConstraintConfig,
) -> List[ParallelStrategy]:
    """Generate all candidate parallel strategies from the search space.

    Enumerates combinations of DP, TP, PP, CP, and micro-batch number,
    subject to divisibility constraints.  When ``search_space.enable_fsdp``
    is True, each candidate with ``dp_degree > 1`` is additionally
    duplicated with ``fsdp_enabled=True``.

    Args:
        hardware: Hardware configuration.
        model: Model configuration.
        search_space: Search space configuration.
        constraints: Constraint configuration.

    Returns:
        List of candidate ParallelStrategy objects (unfiltered).
    """
    n = hardware.num_devices
    gbs = constraints.global_batch_size

    tp_values = _enumerate_tp_values(n, model, search_space)
    pp_values = _enumerate_pp_values(n, model, search_space) if search_space.enable_pp else [1]
    cp_values = _enumerate_cp_values(n, search_space)
    dp_values = search_space.dp_range if search_space.dp_range else []

    candidates = []

    for tp in tp_values:
        for pp in pp_values:
            if n % (tp * pp) != 0:
                continue
            remaining = n // (tp * pp)
            for cp in cp_values:
                if remaining % cp != 0:
                    continue
                dp = remaining // cp
                if dp < 1:
                    continue
                if dp_values and dp not in dp_values:
                    continue

                mbn_values = _enumerate_mbn_values(gbs, dp, pp, search_space)
                if not mbn_values and not search_space.micro_batch_num_range:
                    mbn_values = [1]

                for mbn in mbn_values:
                    strategy = ParallelStrategy(
                        dp_degree=dp,
                        tp_degree=tp,
                        pp_degree=pp,
                        cp_degree=cp,
                        micro_batch_num=mbn,
                    )
                    candidates.append(strategy)

                    if search_space.enable_fsdp and dp > 1:
                        fsdp_strategy = ParallelStrategy(
                            dp_degree=dp,
                            tp_degree=tp,
                            pp_degree=pp,
                            cp_degree=cp,
                            micro_batch_num=mbn,
                            fsdp_enabled=True,
                        )
                        candidates.append(fsdp_strategy)

    logger.info(
        "Generated %d candidate strategies from search space "
        "(tp=%s, pp=%s, cp=%s, fsdp=%s, n=%d)",
        len(candidates),
        tp_values,
        pp_values,
        cp_values,
        search_space.enable_fsdp,
        n,
    )
    return candidates


def filter_divisibility(
    strategies: List[ParallelStrategy],
    model: ModelConfig,
    hardware: HardwareConfig,
    constraints: ConstraintConfig,
) -> List[ParallelStrategy]:
    """Filter strategies that violate divisibility constraints.

    Checks:
    - TP must be a power of 2 and <= num_kv_heads.
    - PP must divide num_layers (for equal partition).
    - DP * TP * PP * CP must equal num_devices.
    - micro_batch_num >= pp_degree when pp_degree > 1.
    - global_batch_size must be divisible by (dp * micro_batch_num).
    - seq_length must be divisible by cp_degree.

    Args:
        strategies: Candidate strategies.
        model: Model configuration.
        hardware: Hardware configuration.
        constraints: Constraint configuration.

    Returns:
        Filtered list of strategies.
    """
    n = hardware.num_devices
    kv_heads = model.effective_num_kv_heads
    filtered = []

    for s in strategies:
        if not _is_power_of_2(s.tp_degree):
            s.mark_infeasible(
                f"tp_degree={s.tp_degree} is not a power of 2"
            )
            filtered.append(s)
            continue

        if s.tp_degree > kv_heads:
            s.mark_infeasible(
                f"tp_degree={s.tp_degree} > num_kv_heads={kv_heads}"
            )
            filtered.append(s)
            continue

        if s.pp_degree > 1 and model.num_layers % s.pp_degree != 0:
            s.mark_infeasible(
                f"num_layers={model.num_layers} not divisible by "
                f"pp_degree={s.pp_degree}"
            )
            filtered.append(s)
            continue

        total = s.dp_degree * s.tp_degree * s.pp_degree * s.cp_degree
        if total != n:
            s.mark_infeasible(
                f"dp*tp*pp*cp={total} != num_devices={n}"
            )
            filtered.append(s)
            continue

        if s.pp_degree > 1 and s.micro_batch_num < s.pp_degree:
            s.mark_infeasible(
                f"micro_batch_num={s.micro_batch_num} < pp_degree={s.pp_degree}"
            )
            filtered.append(s)
            continue

        if (
            constraints.global_batch_size > 0
            and s.dp_degree > 0
            and s.micro_batch_num > 0
            and constraints.global_batch_size % (s.dp_degree * s.micro_batch_num) != 0
        ):
            s.mark_infeasible(
                f"gbs={constraints.global_batch_size} not divisible by "
                f"dp*mbn={s.dp_degree * s.micro_batch_num}"
            )
            filtered.append(s)
            continue

        if s.cp_degree > 1 and model.seq_length % s.cp_degree != 0:
            s.mark_infeasible(
                f"seq_length={model.seq_length} not divisible by "
                f"cp_degree={s.cp_degree}"
            )
            filtered.append(s)
            continue

        if s.fsdp_enabled and s.dp_degree <= 1:
            s.mark_infeasible(
                f"fsdp_enabled=True requires dp_degree>1, got dp={s.dp_degree}"
            )
            filtered.append(s)
            continue

        filtered.append(s)

    feasible_count = sum(1 for s in filtered if s.is_feasible)
    infeasible_count = len(filtered) - feasible_count
    logger.info(
        "Divisibility filter: %d feasible, %d infeasible out of %d",
        feasible_count,
        infeasible_count,
        len(strategies),
    )
    return filtered


def filter_memory_constraint(
    strategies: List[ParallelStrategy],
    constraints: ConstraintConfig,
) -> List[ParallelStrategy]:
    """Filter strategies that exceed memory limit.

    Args:
        strategies: Strategies with memory_cost populated.
        constraints: Constraint configuration with memory_limit_mb.

    Returns:
        Filtered list of strategies.
    """
    if constraints.memory_limit_mb <= 0:
        return strategies

    for s in strategies:
        if not s.is_feasible:
            continue
        if s.memory_cost.total > constraints.memory_limit_mb:
            s.mark_infeasible(
                f"memory={s.memory_cost.total:.0f}MB > "
                f"limit={constraints.memory_limit_mb:.0f}MB"
            )
    return strategies


def filter_min_dp(
    strategies: List[ParallelStrategy],
    constraints: ConstraintConfig,
) -> List[ParallelStrategy]:
    """Filter strategies with DP degree below minimum.

    Args:
        strategies: Candidate strategies.
        constraints: Constraint configuration with min_dp_degree.

    Returns:
        Filtered list of strategies.
    """
    for s in strategies:
        if not s.is_feasible:
            continue
        if s.dp_degree < constraints.min_dp_degree:
            s.mark_infeasible(
                f"dp_degree={s.dp_degree} < min_dp_degree={constraints.min_dp_degree}"
            )
    return strategies


def filter_step_time_constraint(
    strategies: List[ParallelStrategy],
    constraints: ConstraintConfig,
) -> List[ParallelStrategy]:
    """Filter strategies that exceed maximum step time.

    Args:
        strategies: Strategies with performance_cost populated.
        constraints: Constraint configuration with max_step_time_ms.

    Returns:
        Filtered list of strategies.
    """
    if constraints.max_step_time_ms <= 0:
        return strategies

    for s in strategies:
        if not s.is_feasible:
            continue
        if s.performance_cost.total > constraints.max_step_time_ms:
            s.mark_infeasible(
                f"step_time={s.performance_cost.total:.1f}ms > "
                f"max_step_time={constraints.max_step_time_ms:.1f}ms"
            )
    return strategies

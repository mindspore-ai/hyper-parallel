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
"""Memory and performance estimation for Dense LLM strategies.

When a SAPP-ND ``Parallelize`` instance is available the estimation
delegates to its ``mem_eval`` (EvaluatorV2) and the parser-backed
ccfg for memory and performance estimation.  Otherwise an analytical
fallback is used.
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional

from hyper_parallel.auto_parallel.dense_tuner.config import (
    HardwareConfig,
    ModelConfig,
)
from hyper_parallel.auto_parallel.dense_tuner.strategy import (
    MemoryCost,
    ParallelStrategy,
    PerformanceCost,
    StageInfo,
)

logger = logging.getLogger(__name__)


def _apply_strategy_to_ccfg(
    ccfg: Any,
    strategy: ParallelStrategy,
) -> None:
    """Write parallel dimensions from *strategy* into *ccfg*.

    The ccfg **must** have a parser (as produced by ``Parallelize``).
    Delegates to ``ccfg.set_strategy()`` which re-derives shard / comm
    fields automatically.

    Because changing PP/VPP without also adapting ``offset`` and
    ``full_rec`` triggers an ``is_consistent_pp_config`` failure
    inside ``set_strategy``, this function computes a consistent
    offset/recompute via ``BalancingAdapter`` before calling
    ``set_strategy``.

    Args:
        ccfg: A ``CostModelConfig`` with a parser set.
        strategy: Source parallel dimensions.
    """
    # pylint: disable=import-outside-toplevel
    from hyper_parallel.auto_parallel.sapp_nd.nd.balancing_adapter import (
        BalancingAdapter,
    )

    ccfg.has_grad_shard = bool(strategy.fsdp_enabled)
    new_pp = strategy.pp_degree
    new_vpp = strategy.vpp_degree
    n_lay = ccfg.n_lay
    if getattr(ccfg, "emb_out_in_offset", False):
        n_lay += 2
    if getattr(ccfg, "is_mtp_in_offset", False):
        n_lay += getattr(ccfg, "n_mtp", 0)
    adapter = BalancingAdapter(
        layers=n_lay,
        offset=ccfg.offset,
        recompute=ccfg.full_rec,
        manual_ppb=False,
    )
    new_offset = adapter.treat_offset(new_pp, new_vpp)
    new_recompute = adapter.treat_recompute(new_pp, new_vpp)
    ccfg.set_strategy(
        dp=strategy.dp_degree,
        mp=strategy.tp_degree,
        pp=strategy.pp_degree,
        cp=strategy.cp_degree,
        ep=strategy.ep_degree,
        vpp=strategy.vpp_degree,
        op=strategy.op_degree,
        mb=strategy.micro_batch_num,
        mbs=strategy.micro_batch_size,
        offset=new_offset,
        full_rec=new_recompute,
    )


def estimate_memory(
    strategy: ParallelStrategy,
    model: ModelConfig,
    global_batch_size: int,
    ccfg: Optional[Any] = None,
    plot: bool = False,
) -> MemoryCost:
    """Estimate per-device memory cost for a parallel strategy.

    When *ccfg* is provided the estimation delegates to SAPP-ND's
    ``EvaluatorV2.estimate_peak()``; otherwise an analytical fallback
    is used.

    Args:
        strategy: The parallel strategy to estimate.
        model: Model configuration.
        global_batch_size: Global batch size.
        ccfg: Optional pre-built ``CostModelConfig``. When supplied,
            ``EvaluatorV2`` is used for the estimation.
        plot: Whether to generate per-stage memory plots (SAPP-ND only).

    Returns:
        MemoryCost with breakdown in MB.
    """
    if ccfg is not None:
        return _estimate_memory_sapp(strategy, ccfg, global_batch_size, plot=plot)
    return _estimate_memory_analytical(strategy, model, global_batch_size)


def _estimate_memory_sapp(
    strategy: ParallelStrategy,
    ccfg: Any,
    global_batch_size: int,  # pylint: disable=unused-argument
    plot: bool = False,
) -> MemoryCost:
    """Estimate memory using SAPP-ND ``EvaluatorV2`` with parser-backed ccfg.

    When the strategy already has ``memory_cost.total > 0`` (populated
    by ``generate_candidates_sapp``), the breakdown is obtained via
    ``estimate_peak_insight()`` without re-running ``estimate_peak()``.

    Args:
        strategy: The parallel strategy to estimate.
        ccfg: A ``CostModelConfig`` with a parser set (from ``Parallelize``).
        global_batch_size: Global batch size.
        plot: Whether to generate per-stage memory plots.

    Returns:
        MemoryCost with breakdown in MB.
    """
    # pylint: disable=import-outside-toplevel
    from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.estimate_v2 import (
        EvaluatorV2,
    )

    peak_mb = strategy.memory_cost.total
    need_full = peak_mb <= 0 or plot or strategy.fsdp_enabled

    _apply_strategy_to_ccfg(ccfg, strategy)
    evaluator = EvaluatorV2(None, ccfg=ccfg, log_level=0)

    if need_full:
        peak_mb = evaluator.estimate_peak(plot=plot)

    insights = evaluator.estimate_peak_insight()

    model_states_mb = 0.0
    activations_mb = 0.0
    gradients_mb = 0.0
    optimizer_states_mb = 0.0
    comm_mb = 0.0

    if insights:
        first = insights[0] if isinstance(insights, list) else insights
        model_states_mb = float(first.get("ModelParameters", 0))
        optimizer_states_mb = float(first.get("OptimizerStates", 0))
        gradients_mb = float(first.get("AccumulGradients", 0))
        activations_mb = float(first.get("Dynamic", 0))
        comm_mb = float(first.get("AllGather Comm", 0)) + float(first.get("All2All Comm", 0))

    return MemoryCost(
        model_states=model_states_mb,
        activations=activations_mb,
        gradients=gradients_mb,
        optimizer_states=optimizer_states_mb,
        communication=comm_mb,
        total=peak_mb,
    )


def _estimate_memory_analytical(
    strategy: ParallelStrategy,
    model: ModelConfig,
    global_batch_size: int,
) -> MemoryCost:
    """Analytical fallback for memory estimation.

    Args:
        strategy: The parallel strategy to estimate.
        model: Model configuration.
        global_batch_size: Global batch size.

    Returns:
        MemoryCost with breakdown in MB.
    """
    dp = strategy.dp_degree
    tp = strategy.tp_degree
    pp = strategy.pp_degree
    cp = strategy.cp_degree
    mbn = strategy.micro_batch_num
    bytes_p = model.precision_bytes

    h = model.hidden_size
    hff = model.effective_intermediate_size
    v = model.vocab_size
    n_lay = model.num_layers
    s = model.seq_length
    a = model.num_heads
    kv = model.effective_num_kv_heads

    mbs = max(1, global_batch_size // (dp * mbn)) if dp > 0 and mbn > 0 else 1

    param_per_layer = (
        2 * (1 + kv / a) * h * h
        + 3 * h * hff
        + 2 * 2 * h
    )
    total_params = v * h + n_lay * param_per_layer + (0 if model.tie_word_embeddings else v * h)

    params_per_device = total_params / (tp * pp)
    model_states_mb = params_per_device * bytes_p / (1024 * 1024)

    if strategy.fsdp_enabled and dp > 1:
        gradients_mb = params_per_device * bytes_p / dp / (1024 * 1024)
        optimizer_states_mb = params_per_device * (12 if bytes_p < 4 else 8) / dp / (1024 * 1024)
    else:
        gradients_mb = params_per_device * bytes_p / (1024 * 1024)
        optimizer_states_mb = params_per_device * (12 if bytes_p < 4 else 8) / (1024 * 1024)

    act_per_layer = (
        2 * mbs * s * h * bytes_p
        + 4 * mbs * s * s * (a / tp)
        + mbs * s * hff * bytes_p
    ) / cp
    total_act = n_lay * act_per_layer / pp
    activations_mb = total_act / (1024 * 1024)

    comm_mb = 0.0
    if tp > 1:
        comm_mb += 2 * mbs * s * h * bytes_p / (1024 * 1024)
    if cp > 1:
        comm_mb += 2 * mbs * s * h * bytes_p / (1024 * 1024)

    total = model_states_mb + gradients_mb + optimizer_states_mb + activations_mb + comm_mb

    return MemoryCost(
        model_states=model_states_mb,
        activations=activations_mb,
        gradients=gradients_mb,
        optimizer_states=optimizer_states_mb,
        communication=comm_mb,
        total=total,
    )


def estimate_performance(
    strategy: ParallelStrategy,
    model: ModelConfig,
    hardware: HardwareConfig,
    global_batch_size: int,
    ccfg: Optional[Any] = None,
) -> PerformanceCost:
    """Estimate step time performance for a parallel strategy.

    When *ccfg* is provided the estimation delegates to SAPP-ND's
    ``estimate_performance``; otherwise an analytical fallback
    is used.

    Args:
        strategy: The parallel strategy to estimate.
        model: Model configuration.
        hardware: Hardware configuration.
        global_batch_size: Global batch size.
        ccfg: Optional pre-built ``CostModelConfig``.

    Returns:
        PerformanceCost with breakdown in ms.
    """
    if ccfg is not None:
        return _estimate_performance_sapp(strategy, ccfg, hardware, global_batch_size)
    return _estimate_performance_analytical(strategy, model, hardware, global_batch_size)


def _estimate_performance_sapp(
    strategy: ParallelStrategy,
    ccfg: Any,
    hardware: HardwareConfig,
    global_batch_size: int,  # pylint: disable=unused-argument
) -> PerformanceCost:
    """Estimate performance using SAPP-ND performance score.

    When the strategy carries a ``sapp_perf_score`` (populated by
    ``generate_candidates_sapp`` via ``order_search_space``), that
    score is used directly as the compute cost.  Otherwise the score
    is obtained by calling ``sapp_estimate_performance`` on the
    parser-backed ccfg.

    The SAPP-ND score is a *cost* (lower is better), not a wall-clock
    time in ms.  For ranking purposes it is used as-is; the pipeline
    bubble fraction is added on top when PP > 1.

    Args:
        strategy: The parallel strategy to estimate.
        ccfg: A ``CostModelConfig`` with a parser set (from ``Parallelize``).
        hardware: Hardware configuration (used for device type).
        global_batch_size: Global batch size.

    Returns:
        PerformanceCost with breakdown in cost-score units.
    """
    sapp_score = strategy.extra.get("sapp_perf_score")
    if sapp_score is None:
        # pylint: disable=import-outside-toplevel
        from hyper_parallel.auto_parallel.sapp_nd.nd.common import hardware
        from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.estimate import (
            estimate_performance as sapp_estimate_performance,
        )

        _apply_strategy_to_ccfg(ccfg, strategy)
        device_type = hardware.to_sapp_machine().device
        sapp_score = sapp_estimate_performance(ccfg, device_type=device_type)

    pp = strategy.pp_degree
    mbn = strategy.micro_batch_num

    compute_ms = sapp_score
    comm_ms = 0.0
    pipeline_bubble_ms = 0.0
    if pp > 1 and mbn > 0:
        pipeline_bubble_ms = compute_ms * (pp - 1) / mbn
    recompute_ms = 0.0
    total = compute_ms + comm_ms + pipeline_bubble_ms + recompute_ms

    return PerformanceCost(
        compute=compute_ms,
        communication=comm_ms,
        pipeline_bubble=pipeline_bubble_ms,
        recompute=recompute_ms,
        total=total,
    )


def _estimate_performance_analytical(
    strategy: ParallelStrategy,
    model: ModelConfig,
    hardware: HardwareConfig,
    global_batch_size: int,
) -> PerformanceCost:
    """Analytical fallback for performance estimation.

    Args:
        strategy: The parallel strategy to estimate.
        model: Model configuration.
        hardware: Hardware configuration.
        global_batch_size: Global batch size.

    Returns:
        PerformanceCost with breakdown in ms.
    """
    dp = strategy.dp_degree
    tp = strategy.tp_degree
    pp = strategy.pp_degree
    cp = strategy.cp_degree
    mbn = strategy.micro_batch_num
    bytes_p = model.precision_bytes

    h = model.hidden_size
    hff = model.effective_intermediate_size
    n_lay = model.num_layers
    s = model.seq_length
    a = model.num_heads
    kv = model.effective_num_kv_heads

    mbs = max(1, global_batch_size // (dp * mbn)) if dp > 0 and mbn > 0 else 1

    flops_per_layer = (
        4 * (1 + kv / a) * h * h * mbs * s
        + 2 * 3 * h * hff * mbs * s
        + 4 * mbs * s * s * h / tp
    ) / cp
    total_flops = n_lay * flops_per_layer / pp

    tflops = hardware.tflops_per_device
    if tflops <= 0:
        tflops = 300.0
    compute_ms = total_flops / (tflops * 1e9) * 1000

    comm_ms = 0.0
    if tp > 1:
        msg_size = 2 * mbs * s * h * bytes_p
        alpha = 0.005
        bw = hardware.intra_node_bandwidth_gb * 1e9
        comm_ms += 2 * (alpha + msg_size / bw) * 1000

    if cp > 1:
        msg_size = 2 * mbs * s * h * bytes_p
        alpha = 0.005
        bw = hardware.intra_node_bandwidth_gb * 1e9
        comm_ms += 2 * (alpha + msg_size / bw) * 1000

    if dp > 1:
        grad_size = bytes_p * model.num_params_billion * 1e9 / tp / pp
        alpha = 0.01 if dp <= hardware.devices_per_node else 0.05
        bw = (hardware.intra_node_bandwidth_gb if dp <= hardware.devices_per_node
              else hardware.inter_node_bandwidth_gb)
        bw_bytes = bw * 1e9
        comm_ms += (alpha + grad_size / bw_bytes) * 1000

    pipeline_bubble_ms = 0.0
    if pp > 1 and mbn > 0:
        pipeline_bubble_ms = compute_ms * (pp - 1) / mbn

    recompute_ms = 0.0

    total = compute_ms + comm_ms + pipeline_bubble_ms + recompute_ms

    return PerformanceCost(
        compute=compute_ms,
        communication=comm_ms,
        pipeline_bubble=pipeline_bubble_ms,
        recompute=recompute_ms,
        total=total,
    )


def compute_stage_info(
    strategy: ParallelStrategy,
    model: ModelConfig,
    global_batch_size: int,  # pylint: disable=unused-argument
) -> List[StageInfo]:
    """Compute per-stage information for pipeline parallelism.

    Args:
        strategy: The parallel strategy (must have pp_degree > 1).
        model: Model configuration.
        global_batch_size: Global batch size.

    Returns:
        List of StageInfo, one per pipeline stage.
    """
    pp = strategy.pp_degree
    n_lay = model.num_layers

    if pp <= 1:
        return [StageInfo(
            stage_id=0,
            layer_range=(0, n_lay),
            memory_mb=strategy.memory_cost.total,
            time_ms=strategy.performance_cost.total,
            num_layers=n_lay,
        )]

    offset = strategy.layer_offset
    layers_per_stage = n_lay // pp

    stages = []
    start = 0
    for i in range(pp):
        if offset and i < len(offset):
            num_layers = layers_per_stage + int(offset[i])
        else:
            num_layers = layers_per_stage
        end = start + num_layers
        stage_mem = strategy.memory_cost.total / pp
        stage_time = strategy.performance_cost.compute / pp
        stages.append(StageInfo(
            stage_id=i,
            layer_range=(start, end),
            memory_mb=stage_mem,
            time_ms=stage_time,
            num_layers=num_layers,
        ))
        start = end

    return stages


def generate_nd_memory_plots(
    strategy: ParallelStrategy,
    ccfg: Any,
    output_dir: str,
) -> List[str]:
    """Generate per-stage memory estimation plots via SAPP-ND.

    Calls ``EvaluatorV2.estimate_peak(plot=True)`` which writes PNG
    files into ``plots/`` relative to the working directory, then
    moves them into *output_dir*.

    The ccfg must have a parser (from ``Parallelize``).

    Args:
        strategy: The parallel strategy to plot.
        ccfg: A ``CostModelConfig`` with a parser set.
        output_dir: Directory to save plot files.

    Returns:
        List of output file paths.
    """
    # pylint: disable=import-outside-toplevel
    import glob
    import shutil

    from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.estimate_v2 import (
        EvaluatorV2,
    )

    _apply_strategy_to_ccfg(ccfg, strategy)
    evaluator = EvaluatorV2(None, ccfg=ccfg, log_level=0)
    evaluator.estimate_peak(plot=True)

    plot_files: List[str] = []
    os.makedirs(output_dir, exist_ok=True)
    for src in sorted(glob.glob("plots/MemPlot_*.png")):
        dst = os.path.join(output_dir, os.path.basename(src))
        shutil.move(src, dst)
        plot_files.append(dst)

    return plot_files

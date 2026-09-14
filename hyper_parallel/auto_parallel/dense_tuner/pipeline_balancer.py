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
"""Pipeline parallelism balancing via SAPP-PPB.

Wraps ``SappPipeline`` / ``choose_interleave`` to compute per-stage
layer offsets and recompute policies for pipeline-parallel strategies.
The ccfg must come from a ``Parallelize`` instance so that it has a
parser for ``set_strategy()`` to work correctly.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from hyper_parallel.auto_parallel.dense_tuner.estimator import (
    _apply_strategy_to_ccfg,
)
from hyper_parallel.auto_parallel.dense_tuner.strategy import (
    ParallelStrategy,
    StageInfo,
)

logger = logging.getLogger(__name__)

# Persistent memoization for the (expensive) SAPP-PPB ILP solver.  The solver
# is deterministic given identical inputs, and denser-model search often asks
# for the same (strategy-specific) pipeline balance repeatedly.  The cache is
# keyed strictly on solver inputs and stores only solver outputs, so the
# per-strategy ``StageInfo`` reconstruction (which depends on strategy config
# fields such as ``layers_per_stage``) is always recomputed on each call.
_PPB_SOLVER_CACHE: Dict[Tuple, Tuple] = {}


def _layers_cache_key(layers: List[Any]) -> Tuple:
    """Return a hashable representation of a ``Layer`` list.

    Only the fields the SAPP-PPB solver consumes are included, mapped to
    plain hashable primitives (enums to names, per-recompute dicts to ordered
    tuples) so the result can be used as a dict key.
    """
    # pylint: disable=import-outside-toplevel
    import hyper_parallel.auto_parallel.sapp_ppb.utils.recompute as Recompute

    key = []
    for layer in layers:
        key.append((
            layer.name_,
            layer.model_name_,
            layer.type_.name if layer.type_ is not None else "UNKNOWN",
            layer.nb_layer_,
            layer.time_,
            layer.forward_time_,
            layer.memory_parameter_,
            tuple(
                layer.memory_activation_rec_.get(r)
                for r in Recompute.TYPE
            ),
            tuple(layer.backward_time_rec_.get(r) for r in Recompute.TYPE),
            tuple(layer.backward_coef_rec_.get(r) for r in Recompute.TYPE),
            tuple(
                bool(layer.recompute_considered_.get(r))
                for r in Recompute.TYPE
            ),
        ))
    return tuple(key)


def _build_layers_from_strategy(
    strategy: ParallelStrategy,  # pylint: disable=unused-argument
    ccfg: Any,
    profiling_json_path: Optional[str] = None,
) -> List[Any]:
    """Build SAPP-PPB ``Layer`` list from a strategy and ccfg.

    When *profiling_json_path* is provided, loads the layer description
    from that JSON file (``sapp_ppb/layers/`` format) instead of
    calling ``EvaluatorV2.estimate_layer_memory``.  This allows users
    to supply their own profiling data.

    Args:
        strategy: A pipeline-parallel strategy.
        ccfg: A ``CostModelConfig`` with strategy already set.
        profiling_json_path: Optional path to a user profiling JSON
            file in ``sapp_ppb/layers/`` format.  When given,
            ``generate_layers_list()`` is used directly.

    Returns:
        List of ``Layer`` objects suitable for ``SappPipeline``.

    Raises:
        FileNotFoundError: If *profiling_json_path* does not exist.
        ValueError: If the JSON file is missing ``layers_description``.
    """
    # pylint: disable=import-outside-toplevel
    if profiling_json_path:
        from hyper_parallel.auto_parallel.sapp_ppb.utils.layer import (
            generate_layers_list,
        )
        folder = os.path.dirname(os.path.abspath(profiling_json_path))
        base = os.path.splitext(os.path.basename(profiling_json_path))[0]
        return generate_layers_list(folder, base)

    from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.estimate_v2 import (
        EvaluatorV2,
    )

    evaluator = EvaluatorV2(None, ccfg=ccfg, log_level=0)
    ppb_input = evaluator.estimate_layer_memory()

    return _layers_from_ppb_input(ppb_input, ccfg)


def _layers_from_ppb_input(ppb_input: Dict, ccfg: Any) -> List[Any]:
    """Convert PPB input dict to SAPP-PPB Layer objects.

    Reads the ``layers_description`` key (ppb_format=1) produced by
    ``EvaluatorV2.estimate_layer_memory()`` and maps fields using the
    same ``Recompute.JSON_MEMORY_NAME`` / ``JSON_TIME_NAME`` /
    ``JSON_COEF_NAME`` dictionaries that
    ``generate_layers_list()`` (``sapp_ppb/utils/layer.py``) uses.

    Args:
        ppb_input: Output of ``EvaluatorV2.estimate_layer_memory()``.
        ccfg: CostModelConfig for model name.

    Returns:
        List of ``Layer`` objects.
    """
    # pylint: disable=import-outside-toplevel
    from hyper_parallel.auto_parallel.sapp_ppb import Layer
    import hyper_parallel.auto_parallel.sapp_ppb.utils.recompute as Recompute

    layers = []
    model_name = getattr(ccfg, "model_name", "dense-llm")

    layer_list = ppb_input.get("layers_description", [])
    if not layer_list:
        layer_list = ppb_input.get("layers_description_new", [])
    if not layer_list:
        return layers

    for layer_dict in layer_list:
        ltype_str = layer_dict.get("type", "BODY")
        ltype = {
            "HEAD": Layer.type_enum.HEAD,
            "BODY": Layer.type_enum.BODY,
            "TAIL": Layer.type_enum.TAIL,
        }.get(ltype_str.upper(), Layer.type_enum.BODY)

        name = layer_dict.get("name", ltype_str)
        nb_layer = layer_dict.get("nb_layer", 1)
        time = layer_dict.get("time", 0.0)
        forward_time = layer_dict.get("forward_time", 0.0)
        memory_parameter = layer_dict.get("memory_parameter", 0.0)

        memory_activation_rec = {}
        backward_time_rec = {}
        backward_coef_rec = {}
        for rec_type in Recompute.TYPE:
            mem_key = Recompute.JSON_MEMORY_NAME.get(rec_type)
            time_key = Recompute.JSON_TIME_NAME.get(rec_type)
            coef_key = Recompute.JSON_COEF_NAME.get(rec_type)
            memory_activation_rec[rec_type] = (
                layer_dict.get(mem_key, 0.0) if mem_key else 0.0
            )
            backward_time_rec[rec_type] = (
                layer_dict.get(time_key, 0.0) if time_key else 0.0
            )
            backward_coef_rec[rec_type] = (
                layer_dict.get(coef_key, 0.0) if coef_key else 0.0
            )

        layer = Layer(
            model_name=model_name,
            name=name,
            ltype=ltype,
            nb_layer=nb_layer,
            time=time,
            forward_time=forward_time,
            backward_time_rec=backward_time_rec,
            backward_coef_rec=backward_coef_rec,
            memory_parameter=memory_parameter,
            memory_activation_rec=memory_activation_rec,
        )
        layers.append(layer)

    return layers


def _solve_pipeline(
    model_name: str,
    num_stages: int,
    num_micro_batch: int,
    max_memory: int,
    layers: List[Any],
    num_interleave: Optional[int] = None,
) -> Any:
    """Construct and solve a ``SappPipeline``, returning the solved instance.

    Args:
        model_name: Model identifier passed to ``SappPipeline``.
        num_stages: Number of physical pipeline stages.
        num_micro_batch: Micro-batches scheduled per iteration.
        max_memory: Per-device memory budget in MB.
        layers: Ordered ``Layer`` descriptors.
        num_interleave: Virtual-pipeline chunk count.  ``None`` keeps the
            ``SappPipeline`` default (used by the fallback path).

    Returns:
        The constructed and solved ``SappPipeline`` instance.
    """
    # pylint: disable=import-outside-toplevel
    from hyper_parallel.auto_parallel.sapp_ppb import SappPipeline

    kwargs = {
        "model_name": model_name,
        "num_of_stage": num_stages,
        "num_of_micro_batch": num_micro_batch,
        "max_memory": max_memory,
        "layers": layers,
    }
    if num_interleave is not None:
        kwargs["num_of_interleave"] = num_interleave
    pipeline = SappPipeline(**kwargs)
    pipeline.construct_problem()
    pipeline.solve_problem()
    return pipeline


def _build_stage_info(
    strategy: ParallelStrategy,
    mem_act: Any,
    mem_param: Any,
    fw_times: Any,
    rec_times: Any,
) -> List[StageInfo]:
    """Build the per-stage ``StageInfo`` list from PPB solver outputs.

    Args:
        strategy: A pipeline-parallel strategy.
        mem_act: Per-stage activation memory from the solver.
        mem_param: Per-stage parameter memory from the solver.
        fw_times: Per-stage forward times from the solver.
        rec_times: Per-stage recompute times from the solver.

    Returns:
        Ordered list of ``StageInfo`` records, one per stage.
    """
    # pylint: disable=import-outside-toplevel
    from hyper_parallel.auto_parallel.sapp_ppb import flatten

    mem_act_flat = flatten(mem_act)
    mem_param_flat = flatten(mem_param)
    fw_flat = flatten(fw_times)
    rec_flat = flatten(rec_times)

    n_stages = strategy.pp_degree
    start = 0
    new_stage_info = []
    for i in range(min(n_stages, len(mem_act_flat))):
        n_lay = strategy.extra.get("layers_per_stage", strategy.extra.get("num_layers", 0) // n_stages)
        if isinstance(n_lay, list) and i < len(n_lay):
            nl = n_lay[i]
        else:
            nl = strategy.memory_cost.total / n_stages if n_stages > 0 else 0
        end = start + int(nl) if isinstance(nl, (int, float)) else start
        new_stage_info.append(StageInfo(
            stage_id=i,
            layer_range=(start, end),
            memory_mb=float(mem_act_flat[i] + mem_param_flat[i]) if i < len(mem_act_flat) else 0.0,
            time_ms=float(fw_flat[i] + rec_flat[i]) if i < len(fw_flat) else 0.0,
            num_layers=int(nl) if isinstance(nl, (int, float)) else 0,
        ))
        start = end

    return new_stage_info


def _per_stage_layer_counts(
    pipeline: Any,
    layers: List[Any],
    num_stages: int,
    num_interleave: int,
) -> Tuple[List[int], List[int]]:
    """Extract per-physical-stage layer and recompute counts from a solved pipeline.

    Reads the solver's per-``(recompute_type, vpp_chunk, stage)`` layer-count
    variables (including ``TYPE.NONE``) and aggregates them across VPP chunks
    so each physical pipeline stage reports its total assigned layers and the
    number of layers subject to recomputation.  Only BODY layers carry these
    variables; ``HEAD``/``TAIL`` and internal solver variables are skipped.

    Args:
        pipeline: A solved ``SappPipeline`` instance.
        layers: The ``Layer`` list fed to the solver (BODY entries matched by name).
        num_stages: Number of physical pipeline stages.
        num_interleave: Number of VPP chunks in the solved configuration.

    Returns:
        Tuple of ``(stage_layers, stage_recompute)`` lists, each of length
        ``num_stages``, holding absolute per-stage layer counts and per-stage
        recomputed-layer counts respectively.
    """
    # pylint: disable=import-outside-toplevel
    import hyper_parallel.auto_parallel.sapp_ppb.utils.recompute as Recompute
    from hyper_parallel.auto_parallel.sapp_ppb import Layer

    body_names = {
        layer.name_ for layer in layers
        if layer.type_ == Layer.type_enum.BODY
    }
    variables = pipeline.problem_.variables_

    stage_layers = [0] * num_stages
    stage_recompute = [0] * num_stages
    for layer_name in body_names:
        rec_map = variables.get(layer_name)
        if rec_map is None:
            continue
        for rec in Recompute.TYPE:
            counts = rec_map[rec]
            for i in range(num_interleave):
                for s in range(num_stages):
                    cnt = Recompute.zero_if_none_var(counts, i, s)
                    stage_layers[s] += cnt
                    if rec != Recompute.TYPE.NONE:
                        stage_recompute[s] += cnt

    return stage_layers, stage_recompute


def _solve_balanced_pipeline(
    model_name: str,
    num_stages: int,
    num_micro_batch: int,
    max_memory: int,
    layers: List[Any],
) -> Tuple[int, Any, Any, Any, Any, Any, List[int], List[int]]:
    """Run the SAPP-PPB solver for the given inputs, memoizing the result.

    The underlying ILP solver is deterministic for identical inputs.  To
    avoid repeatedly re-solving (and spawning a CBC subprocess per solve)
    for the same pipeline configuration during a dense-model search, the
    solver outputs are cached keyed on their exact inputs.

    Returns:
        Tuple of ``(best_inter, best_dist, mem_act, mem_param, fw_times,
        rec_times, stage_layers, stage_recompute)`` where the memory/time
        lists correspond to the chosen interleave and the final two lists
        hold per-physical-stage layer and recompute counts.
    """
    key = (
        model_name,
        num_stages,
        num_micro_batch,
        max_memory,
        _layers_cache_key(layers),
    )

    cached = _PPB_SOLVER_CACHE.get(key)
    if cached is not None:
        return cached

    # pylint: disable=import-outside-toplevel
    from hyper_parallel.auto_parallel.sapp_ppb import choose_interleave

    best_inter = 1
    try:
        best_inter, _, best_dist = choose_interleave(
            model_name=model_name,
            number_of_stage=num_stages,
            number_of_micro_batch=num_micro_batch,
            max_memory=max_memory,
            layers=layers,
        )
    except Exception:
        pipeline = _solve_pipeline(
            model_name=model_name,
            num_stages=num_stages,
            num_micro_batch=num_micro_batch,
            max_memory=max_memory,
            layers=layers,
        )
        best_dist = pipeline.get_result()

    pipeline = _solve_pipeline(
        model_name=model_name,
        num_stages=num_stages,
        num_micro_batch=num_micro_batch,
        max_memory=max_memory,
        layers=layers,
        num_interleave=best_inter,
    )

    result = (
        best_inter,
        best_dist,
        pipeline.get_memory_activation(),
        pipeline.get_memory_parameter(),
        pipeline.get_fw_time(),
        pipeline.get_recompute_time(),
        *_per_stage_layer_counts(
            pipeline, layers, num_stages, best_inter
        ),
    )
    _PPB_SOLVER_CACHE[key] = result
    return result


def _stage_offsets(stage_layers: List[int], num_stages: int) -> List[int]:
    """Convert per-stage layer counts into offsets relative to equal partition.

    Each entry measures how many layers a physical stage deviates from the
    equal ``total_layers // num_stages`` split — the same interpretation
    ``compute_stage_info`` expects from ``ParallelStrategy.layer_offset``.

    Args:
        stage_layers: Per-stage absolute layer counts (VPP chunks flattened).
        num_stages: Number of physical pipeline stages.

    Returns:
        Per-stage offset relative to the equal partition.
    """
    if not stage_layers:
        return []
    total = sum(stage_layers)
    naive = total // num_stages if num_stages > 0 else 0
    return [count - naive for count in stage_layers]


def balance_pipeline(
    strategy: ParallelStrategy,
    ccfg: Any,
    memory_limit_mb: Optional[float] = None,
    profiling_json_path: Optional[str] = None,
) -> ParallelStrategy:
    """Balance pipeline stages using SAPP-PPB.

    Computes per-stage layer distribution and recompute policy for
    a pipeline-parallel strategy.  When PP > 1 the strategy's
    ``layer_offset``, ``layer_recompute``, and ``stage_info`` fields
    are populated from the PPB solver output.

    Args:
        strategy: A pipeline-parallel strategy (pp_degree > 1).
        ccfg: A ``CostModelConfig`` with model hyperparameters.
        memory_limit_mb: Per-device memory limit in MB.
        profiling_json_path: Optional path to a user profiling JSON
            file in ``sapp_ppb/layers/`` format.

    Returns:
        The same strategy with PPB results filled in.

    Raises:
        RuntimeError: If PPB layer build or solver fails (no silent
            fallback).
    """
    if strategy.pp_degree <= 1:
        return strategy

    _apply_strategy_to_ccfg(ccfg, strategy)

    layers = _build_layers_from_strategy(
        strategy, ccfg, profiling_json_path=profiling_json_path,
    )

    if not layers:
        raise RuntimeError(
            f"PPB produced no layers for strategy {strategy.key}"
        )

    max_memory = int(memory_limit_mb) if memory_limit_mb else int(strategy.memory_cost.total)

    model_name = strategy.extra.get("model_name", "dense-llm")

    best_inter, best_dist, mem_act, mem_param, fw_times, rec_times, stage_layers, stage_recompute = (
        _solve_balanced_pipeline(
            model_name=model_name,
            num_stages=strategy.pp_degree,
            num_micro_batch=strategy.micro_batch_num,
            max_memory=max_memory,
            layers=layers,
        )
    )

    strategy.vpp_degree = best_inter
    strategy.extra["ppb_distribution"] = best_dist
    strategy.layer_offset = _stage_offsets(stage_layers, strategy.pp_degree)
    strategy.layer_recompute = stage_recompute

    new_stage_info = _build_stage_info(
        strategy,
        mem_act,
        mem_param,
        fw_times,
        rec_times,
    )

    if new_stage_info:
        strategy.stage_info = new_stage_info

    return strategy


def balance_strategies(
    strategies: List[ParallelStrategy],
    ccfg: Any,
    memory_limit_mb: Optional[float] = None,
    profiling_json_path: Optional[str] = None,
) -> List[ParallelStrategy]:
    """Apply pipeline balancing to a list of PP strategies.

    Args:
        strategies: Strategies with pp_degree > 1.
        ccfg: A ``CostModelConfig`` with model hyperparameters.
        memory_limit_mb: Per-device memory limit in MB.
        profiling_json_path: Optional path to a user profiling JSON
            file in ``sapp_ppb/layers/`` format.

    Returns:
        Same list with PP strategies balanced.
    """
    for s in strategies:
        if s.is_feasible and s.pp_degree > 1:
            balance_pipeline(
                s, ccfg, memory_limit_mb,
                profiling_json_path=profiling_json_path,
            )
    return strategies


def generate_ppb_pipeline_plot(
    strategy: ParallelStrategy,
    ccfg: Any,
    output_dir: str,
    memory_limit_mb: Optional[float] = None,
) -> Optional[str]:
    """Generate PPB pipeline simulation timeline plot for a strategy.

    After balancing, constructs a ``SappPipeline`` and calls its
    ``simulate()`` method to produce an SVG timeline chart.

    Args:
        strategy: A pipeline-parallel strategy (pp_degree > 1).
        ccfg: A ``CostModelConfig`` with model hyperparameters.
        output_dir: Directory to save the plot file.
        memory_limit_mb: Per-device memory limit in MB.

    Returns:
        Path to the generated plot file, or None on failure.
    """
    if strategy.pp_degree <= 1:
        return None

    # pylint: disable=import-outside-toplevel
    from hyper_parallel.auto_parallel.sapp_ppb import SappPipeline

    _apply_strategy_to_ccfg(ccfg, strategy)

    try:
        layers = _build_layers_from_strategy(strategy, ccfg)
    except Exception as exc:
        logger.warning("PPB layer build failed for %s plot: %s", strategy.key, exc)
        return None

    if not layers:
        return None

    max_memory = int(memory_limit_mb) if memory_limit_mb else int(strategy.memory_cost.total)
    vpp = strategy.vpp_degree if strategy.vpp_degree > 0 else 1

    try:
        pipeline = SappPipeline(
            model_name="dense-llm",
            num_of_stage=strategy.pp_degree,
            num_of_micro_batch=strategy.micro_batch_num,
            max_memory=max_memory,
            layers=layers,
            num_of_interleave=vpp,
        )
        pipeline.construct_problem()
        pipeline.solve_problem()

        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, f"ppb_pipeline_{strategy.key}.svg")
        pipeline.simulate(show=False, file_name=plot_path)
        if os.path.exists(plot_path):
            return plot_path
    except Exception as exc:
        logger.warning("PPB pipeline plot failed for %s: %s", strategy.key, exc)

    return None

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
"""Dense LLM automatic parallel strategy tuner — main entry point.

Orchestrates the end-to-end pipeline using SAPP-ND ``Parallelize``
for search-space enumeration and memory/performance estimation,
and SAPP-PPB for pipeline stage balancing.
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional

from hyper_parallel.auto_parallel.dense_tuner.candidate_generator import (
    collect_nd_visualization_output,
    filter_memory_constraint,
    filter_min_dp,
    filter_step_time_constraint,
    generate_candidates_sapp,
)
from hyper_parallel.auto_parallel.dense_tuner.config import (
    ConstraintConfig,
    HardwareConfig,
    ModelConfig,
    TunerConfig,
)
from hyper_parallel.auto_parallel.dense_tuner.estimator import (
    compute_stage_info,
    estimate_memory,
    estimate_performance,
    generate_nd_memory_plots,
)
from hyper_parallel.auto_parallel.dense_tuner.pipeline_balancer import (
    balance_strategies,
    generate_ppb_pipeline_plot,
)
from hyper_parallel.auto_parallel.dense_tuner.result import (
    PPBResult,
    StrategySummary,
    TunerResult,
)
from hyper_parallel.auto_parallel.dense_tuner.strategy import (
    ParallelStrategy,
)
from hyper_parallel.auto_parallel.dense_tuner.visualizer import StrategyVisualizer

logger = logging.getLogger(__name__)


class DenseTuner:
    """Dense LLM automatic parallel strategy tuner.

    Orchestrates the end-to-end pipeline:
    1. Create SAPP-ND ``Parallelize`` instance from user configuration.
    2. Generate candidate strategies via ``Parallelize.run_generation_to_ordering()``.
    3. Apply divisibility and device constraints.
    4. Estimate memory and performance for each candidate.
    5. Apply memory and other feasibility constraints.
    6. Balance pipeline stages with SAPP-PPB (for PP strategies).
    7. Sort feasible strategies.
    8. Output top-k strategies with breakdown and visualization.

    Args:
        config: TunerConfig with all sub-configurations.
    """

    def __init__(self, config: TunerConfig) -> None:
        """Initialize the tuner.

        Args:
            config: Tuner config with all sub-configurations and output directory.
        """
        self.config = config
        self._visualizer = StrategyVisualizer(output_dir=config.output_dir)
        self._parallelize: Optional[Any] = None
        self._ccfg: Optional[Any] = None

    def _init_sapp(self) -> None:
        """Lazily initialise SAPP-ND Parallelize instance and extract ccfg."""
        if self._parallelize is not None:
            return
        self._parallelize = self.config.create_sapp_parallelize()
        self._ccfg = self._parallelize.config.ccfg

    def tune(self) -> TunerResult:
        """Run the full tuning pipeline and return results.

        Returns:
            TunerResult with top-k strategies, summaries, and breakdowns.
        """
        cfg = self.config
        log_lines: List[str] = []

        log_lines.append(
            f"Model: {cfg.model.model_name}, "
            f"Devices: {cfg.hardware.num_devices}x{cfg.hardware.device_type}, "
            f"FSDP: {cfg.search_space.enable_fsdp}"
        )

        self._init_sapp()

        candidates = generate_candidates_sapp(
            self._parallelize, cfg.search_space, cfg.constraints
        )
        log_lines.append(
            f"Generated {len(candidates)} candidates via SAPP-ND Parallelize"
        )

        candidates = filter_min_dp(candidates, cfg.constraints)

        self._estimate_all(candidates, cfg.model, cfg.hardware, cfg.constraints)

        candidates = filter_memory_constraint(candidates, cfg.constraints)
        candidates = filter_step_time_constraint(candidates, cfg.constraints)

        feasible = [s for s in candidates if s.is_feasible]
        infeasible = [s for s in candidates if not s.is_feasible]

        log_lines.append(f"Feasible: {len(feasible)}, Infeasible: {len(infeasible)}")

        sorted_feasible = self._sort_strategies(feasible, cfg.sort_by)

        # Run the expensive SAPP-PPB solver only on the best ``ppb_top_k``
        # strategies (sorted by the chosen criterion) rather than on the whole
        # candidate set, keeping the pipeline balancing cost bounded.
        ppb_candidates = [
            s for s in sorted_feasible[: cfg.ppb_top_k] if s.pp_degree > 1
        ]
        if ppb_candidates:
            prof_path = cfg.profiling_json_path or None
            balance_strategies(
                ppb_candidates, self._ccfg, cfg.constraints.memory_limit_mb or None,
                profiling_json_path=prof_path,
            )

        top_k = sorted_feasible[: cfg.top_k]

        log_lines.append(f"Top-{len(top_k)} strategies (sorted by {cfg.sort_by}):")
        for i, s in enumerate(top_k):
            log_lines.append(
                f"  #{i + 1} {s.key}: "
                f"step_time={s.performance_cost.total:.1f}ms, "
                f"memory={s.memory_cost.total:.0f}MB"
            )

        memory_breakdown = [s.memory_cost.to_dict() for s in top_k]
        performance_breakdown = [s.performance_cost.to_dict() for s in top_k]
        stage_summary = [[dict(st.__dict__.items()) for st in s.stage_info] for s in top_k]
        ppb_results = [self._build_ppb_result(s) for s in top_k]

        self._cleanup_yaml()

        return TunerResult(
            top_k_strategies=top_k,
            all_candidates=candidates,
            all_candidate_summary=StrategySummary.from_strategies(candidates),
            infeasible_summary=StrategySummary.from_strategies(infeasible),
            memory_breakdown=memory_breakdown,
            performance_breakdown=performance_breakdown,
            stage_summary=stage_summary,
            ppb_results=ppb_results,
            search_log="\n".join(log_lines),
        )

    def tune_and_visualize(self, output_dir: Optional[str] = None) -> TunerResult:
        """Run tuning and generate visualization charts.

        In addition to the dense-tuner's own charts, collects SAPP-ND
        memory estimation plots and SAPP-PPB pipeline timeline plots
        into the output directory.

        Args:
            output_dir: Override output directory for visualizations.

        Returns:
            TunerResult with top-k strategies.
        """
        result = self.tune()

        nd_plot_files: List[str] = []
        ppb_plot_files: List[str] = []
        out = output_dir or self.config.output_dir

        nd_plot_files.extend(
            collect_nd_visualization_output(out)
        )
        for s in result.top_k_strategies:
            if s.pp_degree > 1:
                plot_path = generate_ppb_pipeline_plot(
                    s, self._ccfg, out,
                    memory_limit_mb=self.config.constraints.memory_limit_mb or None,
                )
                if plot_path:
                    ppb_plot_files.append(plot_path)
            if not nd_plot_files and not ppb_plot_files:
                try:
                    nd_plot_files.extend(
                        generate_nd_memory_plots(s, self._ccfg, out)
                    )
                except Exception as exc:
                    logger.warning("ND memory plot failed: %s", exc)

        chart_files = self._visualizer.visualize(
            result.top_k_strategies,
            result.all_candidates,
            output_dir=out,
            nd_plot_files=nd_plot_files,
            ppb_plot_files=ppb_plot_files,
        )
        if chart_files:
            result.search_log += f"\nVisualization files: {chart_files}"
        return result

    def _estimate_all(
        self,
        strategies: List[ParallelStrategy],
        model: ModelConfig,
        hardware: HardwareConfig,
        constraints: ConstraintConfig,
    ) -> None:
        """Estimate memory and performance for all strategies.

        Args:
            strategies: List of candidate strategies to estimate.
            model: Model configuration.
            hardware: Hardware configuration.
            constraints: Constraint configuration.
        """
        gbs = constraints.global_batch_size
        if gbs <= 0:
            gbs = self._infer_gbs(strategies, hardware)

        for s in strategies:
            if not s.is_feasible:
                continue
            try:
                s.memory_cost = estimate_memory(s, model, gbs, ccfg=self._ccfg)
                s.performance_cost = estimate_performance(s, model, hardware, gbs, ccfg=self._ccfg)
                s.stage_info = compute_stage_info(s, model, gbs)
            except Exception as exc:
                s.mark_not_supported(f"estimation failed: {exc}")
                logger.warning("Estimation failed for %s: %s", s.key, exc)

    def _infer_gbs(
        self,
        strategies: List[ParallelStrategy],
        hardware: HardwareConfig,
    ) -> int:
        """Infer a default global batch size.

        Uses the first feasible strategy to compute a reasonable GBS.

        Args:
            strategies: Candidate strategies.
            hardware: Hardware configuration.

        Returns:
            Inferred global batch size.
        """
        for s in strategies:
            if s.is_feasible:
                return s.dp_degree * s.micro_batch_num * max(1, hardware.num_devices // s.total_devices)
        return hardware.num_devices

    def _cleanup_yaml(self) -> None:
        """Remove the temporary YAML file created by ``to_sapp_yaml()``."""
        yaml_path = getattr(self._parallelize, "_yaml_path", None)
        if yaml_path and os.path.exists(yaml_path):
            try:
                os.unlink(yaml_path)
            except OSError:
                pass

    @staticmethod
    def _build_ppb_result(strategy: ParallelStrategy) -> Optional[PPBResult]:
        """Extract PPB result from a strategy's extra metadata.

        Args:
            strategy: A strategy that may have PPB results in extra.

        Returns:
            PPBResult if the strategy has PPB data, else None.
        """
        if strategy.pp_degree <= 1:
            return None
        ppb_dist = strategy.extra.get("ppb_distribution")
        if ppb_dist is None and strategy.vpp_degree <= 1 and not strategy.stage_info:
            return None
        return PPBResult(
            vpp_degree=strategy.vpp_degree,
            stage_offsets=strategy.layer_offset,
            recompute_policy=strategy.layer_recompute,
            ppb_distribution=ppb_dist,
        )

    @staticmethod
    def _sort_strategies(
        strategies: List[ParallelStrategy],
        sort_by: str,
    ) -> List[ParallelStrategy]:
        """Sort feasible strategies by the given criterion.

        Args:
            strategies: Feasible strategies.
            sort_by: Sort criterion ("step_time", "memory", "dp_first").

        Returns:
            Sorted list of strategies.
        """
        if sort_by == "step_time":
            return sorted(strategies, key=lambda s: s.performance_cost.total)
        if sort_by == "memory":
            return sorted(strategies, key=lambda s: s.memory_cost.total)
        if sort_by == "dp_first":
            return sorted(strategies, key=lambda s: (-s.dp_degree, s.performance_cost.total))
        return sorted(strategies, key=lambda s: s.performance_cost.total)

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
"""Visualization for Dense LLM strategy tuning results."""

from __future__ import annotations

import logging
import os
import re
import shutil
from typing import Dict, List, Optional

from hyper_parallel.auto_parallel.dense_tuner.strategy import (
    ParallelStrategy,
    StrategyStatus,
)

logger = logging.getLogger(__name__)


class StrategyVisualizer:
    """Generate visualization charts for strategy tuning results.

    Supports text-based charts and optional matplotlib-based charts
    for step time comparison, memory comparison, stage distribution,
    and filter reason statistics.
    """

    def __init__(self, output_dir: str = "./tuner_output") -> None:
        """Initialize the visualizer.

        Args:
            output_dir: Directory to save visualization output files.
        """
        self.output_dir = output_dir
        self._matplotlib_available: Optional[bool] = None

    def _check_matplotlib(self) -> bool:
        """Check if matplotlib is available.

        Returns:
            True if matplotlib is available.
        """
        if self._matplotlib_available is None:
            try:
                # pylint: disable=import-outside-toplevel
                import matplotlib
                matplotlib.use("Agg")
                self._matplotlib_available = True
            except ImportError:
                self._matplotlib_available = False
                logger.warning(
                    "matplotlib not available; falling back to text charts"
                )
        return self._matplotlib_available

    def visualize(
        self,
        top_k: List[ParallelStrategy],
        all_candidates: List[ParallelStrategy],
        output_dir: Optional[str] = None,
        nd_plot_files: Optional[List[str]] = None,
        ppb_plot_files: Optional[List[str]] = None,
    ) -> List[str]:
        """Generate all visualization charts.

        Args:
            top_k: Top-k feasible strategies.
            all_candidates: All candidate strategies.
            output_dir: Override output directory.
            nd_plot_files: Optional ND memory estimation plot paths to
                include in the output.
            ppb_plot_files: Optional PPB pipeline simulation plot paths
                to include in the output.

        Returns:
            List of output file paths.
        """
        out = output_dir or self.output_dir
        os.makedirs(out, exist_ok=True)
        files = []

        files.append(self._step_time_chart(top_k, out))
        files.append(self._memory_chart(top_k, out))
        files.append(self._stage_distribution_chart(top_k, out))
        files.append(self._feasibility_chart(all_candidates, out))
        files.append(self._filter_reasons_chart(all_candidates, out))
        files.append(self._dimension_trend_chart(all_candidates, out))

        for src in (nd_plot_files or []):
            if src and os.path.exists(src):
                dst = os.path.join(out, os.path.basename(src))
                if os.path.abspath(src) != os.path.abspath(dst):
                    shutil.copy2(src, dst)
                files.append(dst)

        for src in (ppb_plot_files or []):
            if src and os.path.exists(src):
                dst = os.path.join(out, os.path.basename(src))
                if os.path.abspath(src) != os.path.abspath(dst):
                    shutil.copy2(src, dst)
                files.append(dst)

        return [f for f in files if f]

    def _step_time_chart(
        self,
        strategies: List[ParallelStrategy],
        out: str,
    ) -> Optional[str]:
        """Generate step time comparison chart for top-k strategies.

        Args:
            strategies: Top-k strategies.
            out: Output directory.

        Returns:
            Output file path, or None if no strategies.
        """
        if not strategies:
            return None

        labels = [s.key for s in strategies]
        step_times = [s.performance_cost.total for s in strategies]
        path = os.path.join(out, "step_time_comparison.txt")

        with open(path, "w", encoding="utf-8") as f:
            f.write("Step Time Comparison (ms)\n")
            f.write("=" * 60 + "\n")
            max_val = max(step_times) if step_times else 1
            for label, val in zip(labels, step_times):
                bar_len = int(40 * val / max_val) if max_val > 0 else 0
                f.write(f"{label:40s} | {'#' * bar_len} {val:.1f}\n")

        if self._check_matplotlib():
            try:
                # pylint: disable=import-outside-toplevel
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, max(4, len(strategies) * 0.5)))
                y_pos = range(len(strategies))
                ax.barh(y_pos, step_times, align="center")
                ax.set_yticks(list(y_pos))
                ax.set_yticklabels(labels)
                ax.set_xlabel("Estimated Step Time (ms)")
                ax.set_title("Top-k Strategy Step Time Comparison")
                fig.tight_layout()
                img_path = os.path.join(out, "step_time_comparison.png")
                fig.savefig(img_path, dpi=150)
                plt.close(fig)
                return img_path
            except Exception as exc:
                logger.warning("matplotlib chart failed: %s", exc)

        return path

    def _memory_chart(
        self,
        strategies: List[ParallelStrategy],
        out: str,
    ) -> Optional[str]:
        """Generate memory comparison chart for top-k strategies.

        Args:
            strategies: Top-k strategies.
            out: Output directory.

        Returns:
            Output file path, or None if no strategies.
        """
        if not strategies:
            return None

        labels = [s.key for s in strategies]
        memories = [s.memory_cost.total for s in strategies]
        path = os.path.join(out, "memory_comparison.txt")

        with open(path, "w", encoding="utf-8") as f:
            f.write("Per-Device Memory Comparison (MB)\n")
            f.write("=" * 60 + "\n")
            max_val = max(memories) if memories else 1
            for label, val in zip(labels, memories):
                bar_len = int(40 * val / max_val) if max_val > 0 else 0
                f.write(f"{label:40s} | {'#' * bar_len} {val:.0f}\n")

        if self._check_matplotlib():
            try:
                # pylint: disable=import-outside-toplevel
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, max(4, len(strategies) * 0.5)))
                y_pos = range(len(strategies))
                ax.barh(y_pos, memories, align="center")
                ax.set_yticks(list(y_pos))
                ax.set_yticklabels(labels)
                ax.set_xlabel("Per-Device Memory (MB)")
                ax.set_title("Top-k Strategy Memory Comparison")
                fig.tight_layout()
                img_path = os.path.join(out, "memory_comparison.png")
                fig.savefig(img_path, dpi=150)
                plt.close(fig)
                return img_path
            except Exception as exc:
                logger.warning("matplotlib chart failed: %s", exc)

        return path

    def _stage_distribution_chart(
        self,
        strategies: List[ParallelStrategy],
        out: str,
    ) -> Optional[str]:
        """Generate PP stage distribution chart.

        Args:
            strategies: Top-k strategies (with stage_info populated).
            out: Output directory.

        Returns:
            Output file path, or None if no PP strategies.
        """
        pp_strategies = [s for s in strategies if s.pp_degree > 1 and s.stage_info]
        if not pp_strategies:
            return None

        path = os.path.join(out, "stage_distribution.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("Pipeline Stage Distribution\n")
            f.write("=" * 60 + "\n")
            for s in pp_strategies:
                vpp_info = f", vpp={s.vpp_degree}" if s.vpp_degree > 1 else ""
                f.write(f"\nStrategy: {s.key}{vpp_info}\n")
                for stage in s.stage_info:
                    f.write(
                        f"  Stage {stage.stage_id}: "
                        f"layers={stage.layer_range}, "
                        f"memory={stage.memory_mb:.0f}MB, "
                        f"time={stage.time_ms:.1f}ms\n"
                    )

        if self._check_matplotlib():
            try:
                # pylint: disable=import-outside-toplevel
                import matplotlib.pyplot as plt
                # pylint: disable=import-outside-toplevel
                import numpy as np

                n_strats = len(pp_strategies)
                fig, axes = plt.subplots(1, n_strats, figsize=(6 * n_strats, 4))
                if n_strats == 1:
                    axes = [axes]
                for ax, s in zip(axes, pp_strategies):
                    stages = s.stage_info
                    stage_ids = [st.stage_id for st in stages]
                    mems = [st.memory_mb for st in stages]
                    times = [st.time_ms for st in stages]
                    x = np.arange(len(stage_ids))
                    width = 0.35
                    ax.bar(x - width / 2, mems, width, label="Memory (MB)")
                    ax2 = ax.twinx()
                    ax2.bar(x + width / 2, times, width, label="Time (ms)", color="orange")
                    ax.set_xlabel("Stage ID")
                    ax.set_ylabel("Memory (MB)")
                    ax2.set_ylabel("Time (ms)")
                    ax.set_title(s.key)
                    ax.set_xticks(x)
                    ax.set_xticklabels(stage_ids)
                    lines1, labels1 = ax.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
                fig.suptitle("PP Stage Memory/Time Distribution")
                fig.tight_layout()
                img_path = os.path.join(out, "stage_distribution.png")
                fig.savefig(img_path, dpi=150)
                plt.close(fig)
                return img_path
            except Exception as exc:
                logger.warning("matplotlib chart failed: %s", exc)

        return path

    def _feasibility_chart(
        self,
        all_candidates: List[ParallelStrategy],
        out: str,
    ) -> Optional[str]:
        """Generate feasible/infeasible distribution chart.

        Args:
            all_candidates: All candidate strategies.
            out: Output directory.

        Returns:
            Output file path.
        """
        feasible = sum(1 for s in all_candidates if s.status == StrategyStatus.FEASIBLE)
        infeasible = sum(1 for s in all_candidates if s.status == StrategyStatus.INFEASIBLE)
        not_supported = sum(1 for s in all_candidates if s.status == StrategyStatus.NOT_SUPPORTED)
        total = len(all_candidates)

        path = os.path.join(out, "feasibility_distribution.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("Strategy Feasibility Distribution\n")
            f.write("=" * 40 + "\n")
            f.write(f"Total candidates: {total}\n")
            if total > 0:
                f.write(f"Feasible:         {feasible} ({100 * feasible / total:.1f}%)\n")
                f.write(f"Infeasible:       {infeasible} ({100 * infeasible / total:.1f}%)\n")
                f.write(f"Not supported:    {not_supported} ({100 * not_supported / total:.1f}%)\n")
            else:
                f.write("Feasible:         0\n")
                f.write("Infeasible:       0\n")
                f.write("Not supported:    0\n")

        if self._check_matplotlib() and total > 0:
            try:
                # pylint: disable=import-outside-toplevel
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(6, 4))
                labels = ["Feasible", "Infeasible", "Not Supported"]
                sizes = [feasible, infeasible, not_supported]
                colors = ["#4CAF50", "#F44336", "#9E9E9E"]
                ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
                ax.set_title("Strategy Feasibility Distribution")
                img_path = os.path.join(out, "feasibility_distribution.png")
                fig.savefig(img_path, dpi=150)
                plt.close(fig)
                return img_path
            except Exception as exc:
                logger.warning("matplotlib chart failed: %s", exc)

        return path

    def _filter_reasons_chart(
        self,
        all_candidates: List[ParallelStrategy],
        out: str,
    ) -> Optional[str]:
        """Generate filter reasons statistics chart.

        Args:
            all_candidates: All candidate strategies.
            out: Output directory.

        Returns:
            Output file path.
        """
        reasons: Dict[str, int] = {}
        for s in all_candidates:
            if s.filter_reason:
                category = re.split(r"[=<>]", s.filter_reason)[0].strip()
                reasons[category] = reasons.get(category, 0) + 1

        path = os.path.join(out, "filter_reasons.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("Filter Reasons Statistics\n")
            f.write("=" * 40 + "\n")
            for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
                f.write(f"  {reason}: {count}\n")

        if self._check_matplotlib() and reasons:
            try:
                # pylint: disable=import-outside-toplevel
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8, max(4, len(reasons) * 0.5)))
                sorted_reasons = sorted(reasons.items(), key=lambda x: -x[1])
                labels = [r[0][:40] for r in sorted_reasons]
                counts = [r[1] for r in sorted_reasons]
                ax.barh(range(len(labels)), counts, align="center")
                ax.set_yticks(range(len(labels)))
                ax.set_yticklabels(labels)
                ax.set_xlabel("Count")
                ax.set_title("Filter Reasons Distribution")
                fig.tight_layout()
                img_path = os.path.join(out, "filter_reasons.png")
                fig.savefig(img_path, dpi=150)
                plt.close(fig)
                return img_path
            except Exception as exc:
                logger.warning("matplotlib chart failed: %s", exc)

        return path

    def _dimension_trend_chart(
        self,
        all_candidates: List[ParallelStrategy],
        out: str,
    ) -> Optional[str]:
        """Generate performance/memory trend charts across parallel dimensions.

        Args:
            all_candidates: All candidate strategies.
            out: Output directory.

        Returns:
            Output file path.
        """
        feasible = [s for s in all_candidates if s.is_feasible]
        if len(feasible) < 2:
            return None

        path = os.path.join(out, "dimension_trends.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("Performance/Memory Trends by Parallel Dimension\n")
            f.write("=" * 60 + "\n")
            for dim_name, dim_key in [
                ("DP", "dp_degree"),
                ("TP", "tp_degree"),
                ("PP", "pp_degree"),
                ("CP", "cp_degree"),
                ("EP", "ep_degree"),
                ("VPP", "vpp_degree"),
                ("OP", "op_degree"),
                ("MBS", "micro_batch_size"),
                ("FSDP", "fsdp_enabled"),
            ]:
                by_dim: Dict[int, List[ParallelStrategy]] = {}
                for s in feasible:
                    val = getattr(s, dim_key)
                    by_dim.setdefault(val, []).append(s)
                f.write(f"\n{dim_name}:\n")
                for val in sorted(by_dim.keys()):
                    group = by_dim[val]
                    avg_mem = sum(s.memory_cost.total for s in group) / len(group)
                    avg_time = sum(s.performance_cost.total for s in group) / len(group)
                    f.write(f"  {dim_name}={val}: avg_mem={avg_mem:.0f}MB, avg_time={avg_time:.1f}ms\n")

        if self._check_matplotlib():
            try:
                # pylint: disable=import-outside-toplevel
                import matplotlib.pyplot as plt

                dim_list = [
                    ("DP", "dp_degree"), ("TP", "tp_degree"),
                    ("PP", "pp_degree"), ("CP", "cp_degree"),
                    ("EP", "ep_degree"), ("VPP", "vpp_degree"),
                    ("OP", "op_degree"), ("MBS", "micro_batch_size"),
                    ("FSDP", "fsdp_enabled"),
                ]
                n_dims = len(dim_list)
                fig, axes = plt.subplots(n_dims, 2, figsize=(12, 3 * n_dims))
                for idx, (dim_name, dim_key) in enumerate(dim_list):
                    by_dim: Dict[int, List[ParallelStrategy]] = {}
                    for s in feasible:
                        val = getattr(s, dim_key)
                        by_dim.setdefault(val, []).append(s)
                    vals = sorted(by_dim.keys())
                    avg_mems = [sum(s.memory_cost.total for s in by_dim[v]) / len(by_dim[v]) for v in vals]
                    avg_times = [sum(s.performance_cost.total for s in by_dim[v]) / len(by_dim[v]) for v in vals]

                    axes[idx][0].plot(vals, avg_mems, "o-")
                    axes[idx][0].set_title(f"Memory vs {dim_name}")
                    axes[idx][0].set_xlabel(dim_name)
                    axes[idx][0].set_ylabel("Avg Memory (MB)")

                    axes[idx][1].plot(vals, avg_times, "o-", color="orange")
                    axes[idx][1].set_title(f"Step Time vs {dim_name}")
                    axes[idx][1].set_xlabel(dim_name)
                    axes[idx][1].set_ylabel("Avg Step Time (ms)")

                fig.suptitle("Performance & Memory Trends by Dimension")
                fig.tight_layout()
                img_path = os.path.join(out, "dimension_trends.png")
                fig.savefig(img_path, dpi=150)
                plt.close(fig)
                return img_path
            except Exception as exc:
                logger.warning("matplotlib chart failed: %s", exc)

        return path

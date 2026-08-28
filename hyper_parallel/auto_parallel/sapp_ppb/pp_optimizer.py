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
"""PP optimizer — YAML + JSON driven pipeline parallelism strategy optimizer."""

from __future__ import annotations

import logging
import math
from typing import Any

from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.layer_loader import (
    LayerBuilder,
)
from hyper_parallel.auto_parallel.sapp_ppb.pp_modeling.pp_structs import (
    PPStrategyResult,
)
from hyper_parallel.auto_parallel.sapp_ppb.pp_modeling.pp_balancer import (
    PPBalancer,
)
from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.yaml_parser import (
    parse_yaml_for_optimization,
)
from hyper_parallel.auto_parallel.sapp_ppb.pp_sim_adapter import PPSimulator


class PPOptimizer:
    """Unified PP strategy optimizer — YAML + JSON driven.

    Pipeline topology and ILP constraints (pp_degree, num_layer,
    micro_batch_num, num_of_interleave, memory_limit, etc.) come from
    a YAML file; layer descriptions come from a JSON file.

    Orchestration flow:

    1. Parse YAML config and build layers from JSON.
    2. Call :class:`PPBalancer` to solve the ILP and build the
       strategy result.
    3. If ``enable_simulation`` is ``True``, run :class:`PPSimulator`
       and merge simulation metrics into the result.

    Example:
        >>> optimizer = PPOptimizer()
        >>> result = optimizer.optimize(
        ...     yaml_path="init_demo.yaml",
        ...     json_path="demo.json",
        ... )
    """

    def optimize(
        self,
        yaml_path: str = "",
        json_path: str = "",
    ) -> PPStrategyResult:
        """Optimize PP strategy — YAML + JSON driven.

        Reads pipeline topology and ILP constraints from ``yaml_path``
        and layer descriptions from ``json_path``, then runs a single
        ILP optimisation, optionally runs the pipeline simulator, and
        returns the result.

        Args:
            yaml_path: Path to the YAML configuration file with pipeline
                topology and ILP constraints (``pipeline_num``,
                ``num_layer``, ``micro_batch_num``,
                ``num_of_interleave``, ``memory_limit``, etc.).
            json_path: Path to the JSON file with ``layers_description``
                defining the model layers (HEAD, BODY, TAIL).

        Returns:
            Optimised PP strategy result.

        Raises:
            ValueError: If ``yaml_path`` or ``json_path`` is empty.
            RuntimeError: If the ILP solution is infeasible.

        Example:
            >>> optimizer = PPOptimizer()
            >>> result = optimizer.optimize(
            ...     yaml_path="init_demo.yaml",
            ...     json_path="demo.json",
            ... )
        """
        if not yaml_path:
            raise ValueError(
                "PPOptimizer.optimize requires yaml_path. "
                "Please provide a valid YAML configuration path."
            )
        if not json_path:
            raise ValueError(
                "PPOptimizer.optimize requires json_path. "
                "Please provide a valid JSON profile path."
            )

        yaml_config = parse_yaml_for_optimization(yaml_path)

        builder = LayerBuilder(yaml_config, json_path)
        balancer = PPBalancer(builder)
        result = balancer.balance_with_ilp()

        if not result.is_feasible:
            details = result.infeasibility_details
            msg = f"PP optimization failed: {details.get('reason', 'unknown')}"
            if details.get("error"):
                msg += f" ({details['error']})"
            raise RuntimeError(msg)

        if yaml_config.enable_simulation:
            self._run_simulation(
                result,
                balancer.pipeline,
                yaml_config,
                builder.constant_memory,
                yaml_config.sim_comm_time,
            )

        if result.simulation_status == "failed":
            logging.getLogger(__name__).warning(
                "ILP succeeded but simulation failed: %s. "
                "Pipeline bubble and step time estimates are unavailable.",
                result.simulation_error,
            )
        elif result.simulation_status == "not_run":
            logging.getLogger(__name__).info(
                "Simulation skipped (enable_simulation=False). "
                "Pipeline bubble and step time estimates are not available."
            )

        return result

    @staticmethod
    def _run_simulation(
        result: PPStrategyResult,
        pipeline: Any,
        yaml_config: Any,
        constant_memory: int,
        sim_comm_time: float,
    ) -> None:
        """Run pipeline simulator and merge results into *result* in-place.

        Creates a :class:`PPSimulator`, runs the post-ILP simulation,
        and merges the simulator output fields into the provided
        ``result``.  On failure, marks ``simulation_status`` as
        ``"failed"`` with a descriptive error.

        Args:
            result: Strategy result from :meth:`PPBalancer.balance_with_ilp`
                (modified in-place with simulation metrics).
            pipeline: Solved :class:`SappPipeline` instance.
            yaml_config: YAML configuration with pipeline topology.
            constant_memory: Constant memory per stage (MB).
            sim_comm_time: P2P communication time between adjacent stages
                (ms) for the simulator.
        """
        pp_sim = PPSimulator(
            pipeline=pipeline,
            yaml_config=yaml_config,
            constant_memory=constant_memory,
        )

        sim_result = pp_sim.simulate_from_ilp(sim_comm_time=sim_comm_time)

        if sim_result is None:
            result.simulation_status = "failed"
            result.simulation_error = (
                "ILP simulation returned None (e.g. micro_batch_num < pp_degree)"
            )
            return

        real_bubble_val = sim_result.simulator_bubbles.get("real", 0.0)
        if not math.isfinite(sim_result.simulator_end_time) or not math.isfinite(real_bubble_val):
            result.simulation_status = "failed"
            result.simulation_error = (
                "Simulation produced non-finite results (e.g. total compute time is zero)"
            )
            return

        result.simulation_status = sim_result.simulation_status
        result.simulation_error = sim_result.simulation_error
        result.simulator_end_time = sim_result.simulator_end_time
        result.simulator_bubbles = sim_result.simulator_bubbles
        result.simulator_peak_memory = sim_result.simulator_peak_memory

        pipeline_bubble = sim_result.simulator_bubbles.get("real")
        if pipeline_bubble is None:
            logging.getLogger(__name__).warning(
                "Simulator bubbles dict missing 'real' key. "
                "Available keys: %s",
                list(sim_result.simulator_bubbles.keys()),
            )
        result.pipeline_bubble = pipeline_bubble

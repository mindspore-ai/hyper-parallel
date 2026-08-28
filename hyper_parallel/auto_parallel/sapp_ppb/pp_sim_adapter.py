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
"""PP simulator — pipeline parallelism schedule simulation."""

from __future__ import annotations

import logging
from typing import Any, Optional

from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.layer_loader import (
    SAPP_PPB_AVAILABLE,
)
from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.yaml_parser import (
    YamlOptimizationConfig,
)
from hyper_parallel.auto_parallel.sapp_ppb.pp_modeling.pp_structs import PPBOutput

logger = logging.getLogger(__name__)


class PPSimulator:
    """Pipeline parallelism schedule simulator.

    Wraps :class:`SappPipeline.simulate` and provides high-level methods
    for simulating ILP-optimized pipeline schedules.

    Args:
        pipeline: A solved :class:`SappPipeline` instance.
        yaml_config: YAML configuration with pipeline topology.
        constant_memory: Constant memory per stage (MB).

    Example:
        >>> sim = PPSimulator(pipeline, yaml_config, constant_mem)
        >>> result = sim.simulate_from_ilp(sim_comm_time=0.1)
    """

    def __init__(
        self,
        pipeline: Any,
        yaml_config: YamlOptimizationConfig,
        constant_memory: int,
    ) -> None:
        """Initialize PPSimulator.

        Args:
            pipeline: A solved :class:`SappPipeline` instance.
            yaml_config: YAML configuration with pipeline topology.
            constant_memory: Constant memory per stage (MB).
        """
        self._pipeline = pipeline
        self._yaml_config = yaml_config
        self._constant_memory = constant_memory

    def simulate_from_ilp(
        self,
        sim_comm_time: float = 0.0,
    ) -> Optional[PPBOutput]:
        """Run PipelineSimulator using ILP solver output for accurate step time.

        Delegates to :meth:`SappPipeline.simulate` which internally calls
        ``get_fw_time()``, ``get_recompute_time()``,
        ``get_memory_activation()``, and ``get_memory_parameter()``
        to build and run the :class:`PipelineSimulator`.  The resulting
        ``end_time`` reflects the true pipeline schedule (1F1B / VPP
        with warmup-steady-cooldown phases, P2P communication, bubble
        overlap) — far more accurate than
        ``max_stage_time × micro_batch_num``.

        Args:
            sim_comm_time: P2P communication time between adjacent stages
                (ms). Passed to the pipeline simulator only; does NOT
                affect ILP optimization. Default 0.0 (no communication
                delay).

        Returns:
            :class:`PPBOutput` with ``simulation_status="success"`` and
            populated ``simulator_end_time``, ``simulator_bubbles``,
            ``simulator_peak_memory``; or ``None`` if the pipeline has
            not been solved yet or the simulation cannot run.

        Example:
            >>> sim = PPSimulator(pipeline, yaml_config, constant_mem)
            >>> result = sim.simulate_from_ilp(sim_comm_time=0.1)
            >>> result.simulator_end_time
            1234.5
        """
        if not SAPP_PPB_AVAILABLE or self._pipeline is None:
            return None

        if self._yaml_config.micro_batch_num < self._yaml_config.pp_degree:
            logger.warning(
                "micro_batch_num (%d) < pp_degree (%d); simulator skipped.",
                self._yaml_config.micro_batch_num, self._yaml_config.pp_degree,
            )
            return None

        try:
            end_time = self._pipeline.simulate(
                show=False, comm_time=sim_comm_time,
            )

            if end_time is None or end_time <= 0:
                return None

            sim_instance = self._pipeline.simulator
            return PPBOutput(
                simulation_status="success",
                simulator_end_time=sim_instance.end_time,
                simulator_bubbles=dict(sim_instance.bubbles),
                simulator_peak_memory=list(sim_instance.peak_memory),
            )
        except ValueError as exc:
            logger.warning("Simulator failed with ValueError: %s", exc)
            return None
        except Exception as exc:
            from hyper_parallel.auto_parallel.sapp_ppb.simulator.causal_error import CausalCommError, CausalError  # pylint: disable=C0415
            if isinstance(exc, (CausalCommError, CausalError)):
                logger.warning("Simulator failed with %s: %s", type(exc).__name__, exc)
                return None
            raise

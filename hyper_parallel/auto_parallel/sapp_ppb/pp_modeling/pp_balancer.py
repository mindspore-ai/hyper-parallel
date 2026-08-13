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
"""PP balancer — ILP-based pipeline parallelism load balancing."""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.layer_loader import (
    SAPP_PPB_AVAILABLE,
    _get_pipeline_layer_class,
)
from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.yaml_parser import (
    YamlOptimizationConfig,
)
from hyper_parallel.auto_parallel.sapp_ppb.pp_modeling.pp_structs import (
    PPBOutput,
    PPStrategyResult,
)
from hyper_parallel.auto_parallel.sapp_ppb.utils.recompute import TYPE as RecomputeType

if SAPP_PPB_AVAILABLE:
    from hyper_parallel.auto_parallel.sapp_ppb.sapp.sapp_pipeline import SappPipeline  # pylint: disable=C0415
    from hyper_parallel.auto_parallel.sapp_ppb.utils import recompute as Recompute  # pylint: disable=C0415

logger = logging.getLogger(__name__)


class PPBalancer:
    """ILP-based pipeline parallelism load balancer.

    Receives a :class:`LayerBuilder` (which holds the converted
    sapp-ppb Layer objects and YAML/JSON configs), constructs and
    solves the ILP, extracts the balanced partition, and builds
    the strategy result.

    Args:
        layer_builder: A :class:`LayerBuilder` instance with validated
            configuration and constructed Layer objects.

    Example:
        >>> builder = LayerBuilder(yaml_config, json_path)
        >>> balancer = PPBalancer(builder)
        >>> output = balancer.balance_with_ilp(time_limit=60)
    """

    def __init__(
        self,
        layer_builder: Any,
    ) -> None:
        """Initialize PPBalancer.

        Args:
            layer_builder: A :class:`LayerBuilder` instance.

        Raises:
            ImportError: If sapp-ppb module is not available.
        """
        if not SAPP_PPB_AVAILABLE:
            raise ImportError(
                "sapp-ppb module is not available. "
                "Please ensure sapp-ppb is installed and accessible."
            )

        self._layer_builder = layer_builder
        self._pipeline: Optional[Any] = None
        self._is_successful: bool = False

    @property
    def yaml_config(self) -> YamlOptimizationConfig:
        """YAML configuration with pipeline topology."""
        return self._layer_builder.yaml_config

    @property
    def pipeline(self) -> Optional[Any]:
        """Solved SappPipeline instance (available after :meth:`balance_with_ilp`)."""
        return self._pipeline

    def _make_infeasible_output(
        self,
        reason: str,
        error: str = "",
        solver_status: Any = None,
    ) -> PPBOutput:
        """Create a PPBOutput indicating infeasibility.

        Args:
            reason: Human-readable infeasibility reason.
            error: Optional error message string.
            solver_status: Optional solver status code.

        Returns:
            PPBOutput with is_feasible=False and is_successful=False.
        """
        details: Dict[str, Any] = {"reason": reason}
        if error:
            details["error"] = error
        if solver_status is not None:
            details["solver_status"] = solver_status
        return PPBOutput(
            stage_partition=[],
            layer_offset={},
            is_feasible=False,
            infeasibility_details=details,
        )

    def _build_feasible_output(
        self,
        stage_partition: List[List[Tuple[int, RecomputeType]]],
        layer_offset: Optional[Dict[str, List[List[int]]]] = None,
        num_of_interleave: int = 1,
    ) -> PPBOutput:
        """Build a PPBOutput from the ILP solution.

        Args:
            stage_partition: Per-(virtual-)stage list of ``(layer_id, RecomputeType)`` tuples.
                Length is ``vpp * pp`` when VPP > 1, otherwise ``pp``.
            layer_offset: Per-group layer offset (key=group_name, value shape ``[vpp][pp]``).
            num_of_interleave: VPP interleaving factor (1 = no VPP).

        Returns:
            PPBOutput with is_feasible=True and is_successful=True.
        """
        if layer_offset is None:
            layer_offset = {}

        return PPBOutput(
            stage_partition=stage_partition,
            layer_offset=layer_offset,
            is_feasible=True,
            is_successful=True,
            infeasibility_details={},
            num_of_interleave=num_of_interleave,
        )

    @staticmethod
    def _pulp_has_feasible_solution(pulp_problem: Any) -> bool:
        """Check whether a PuLP problem has a feasible solution despite non-optimal status.

        When CBC times out (status=0 / Not Solved), it may have found a
        feasible incumbent without proving optimality.  PuLP exposes this
        via ``sol_status``: ``LpSolutionOptimal`` (1) or
        ``LpSolutionIntegerFeasible`` (2) indicate a genuine feasible
        solution was found.  ``LpSolutionNoSolutionFound`` (0) means no
        incumbent exists even though decision variables may have default
        values assigned.

        Args:
            pulp_problem: The PuLP LpProblem instance.

        Returns:
            True if ``sol_status`` indicates a feasible or optimal
            solution was found.
        """
        from pulp import (  # pylint: disable=C0415
            LpSolutionOptimal,
            LpSolutionIntegerFeasible,
        )
        try:
            return pulp_problem.sol_status in (LpSolutionOptimal, LpSolutionIntegerFeasible)
        except (AttributeError, TypeError):
            return False

    def _check_ilp_solve_status(self) -> Optional[PPBOutput]:
        """Check ILP solver status and determine feasibility.

        Distinguishes three categories:

        1. **Optimal** (status=1) → return ``None`` (proceed to extract).
        2. **Not Solved** (status=0) with feasible incumbent
           (``sol_status`` is ``LpSolutionOptimal`` or
           ``LpSolutionIntegerFeasible``) → return ``None`` (proceed to
           extract, ``_is_successful`` set to ``True``).
        3. **Infeasible / Undefined / Unbounded / Not Solved without
           incumbent** (``sol_status`` is ``LpSolutionNoSolutionFound``)
           → return infeasible PPBOutput.

        Returns:
            PPBOutput with ``is_feasible=False`` if no usable solution
            exists, otherwise ``None``.
        """
        from pulp import (  # pylint: disable=C0415
            LpStatusOptimal,
            LpStatusInfeasible,
            LpStatusUndefined,
            LpStatusNotSolved,
        )

        if not hasattr(self._pipeline, 'problem_'):
            return None
        pulp_problem = getattr(self._pipeline.problem_, 'problem_', None)
        if not pulp_problem or not hasattr(pulp_problem, 'status'):
            return None

        status = pulp_problem.status

        if status == LpStatusOptimal:
            self._is_successful = True
            return None

        if status == LpStatusInfeasible:
            return self._make_infeasible_output(
                "ILP solver returned infeasible status",
                solver_status=status,
            )

        if status == LpStatusUndefined:
            return self._make_infeasible_output(
                "ILP solver returned undefined status",
                solver_status=status,
            )

        if status == LpStatusNotSolved:
            if self._pulp_has_feasible_solution(pulp_problem):
                self._is_successful = True
                return None
            return self._make_infeasible_output(
                "ILP solver timed out with no feasible solution found",
                solver_status=status,
            )

        return self._make_infeasible_output(
            f"ILP solver returned unbounded status {status}",
            solver_status=status,
        )

    def balance_with_ilp(
        self,
        time_limit: int = 90,
        solver: str = "pulp",
    ) -> PPStrategyResult:
        """Run ILP-based load balancing using sapp-ppb.

        Solves the ILP and builds the :class:`PPStrategyResult`.
        Simulation is *not* run here — that responsibility belongs to
        :class:`PPOptimizer`.

        Pipeline topology parameters (``num_of_interleave``,
        ``vpp_less_memory``, ``optimization_level``) are read from
        the ``yaml_config`` stored in the :class:`LayerBuilder` passed
        at construction.

        Args:
            time_limit: Solver time limit in seconds.
            solver: Solver backend ("pulp" or "gurobi").

        Returns:
            PP strategy result with balanced partition (without
            simulation metrics; simulation is the caller's
            responsibility).

        Example:
            >>> builder = LayerBuilder(yaml_config, json_path)
            >>> balancer = PPBalancer(builder)
            >>> result = balancer.balance_with_ilp(time_limit=60)
        """
        if not SAPP_PPB_AVAILABLE:
            raise ImportError("sapp-ppb module is not available")

        if not self._layer_builder.memory_limit:
            raise ValueError(
                "memory_limit is required for ILP load balancing. "
                "Please specify memory_limit in the JSON profile."
            )

        num_of_interleave = self.yaml_config.num_of_interleave
        vpp_less_memory = self.yaml_config.vpp_less_memory
        optimization_level = self.yaml_config.optimization_level

        self._pipeline = SappPipeline(  # pylint: disable=E0606
            model_name="sapp_nd_model",
            num_of_stage=self.yaml_config.pp_degree,
            num_of_micro_batch=self.yaml_config.micro_batch_num,
            max_memory=self._layer_builder.memory_limit,
            layers=self._layer_builder.layers_sapp_ppb,
            num_of_interleave=num_of_interleave,
            vpp_less_memory=vpp_less_memory,
            constant_memory=self._layer_builder.constant_memory,
            optimization_level=optimization_level,
            use_backward_time=self._layer_builder.use_backward_time,
        )

        self._pipeline.construct_problem(solver=solver)

        self._pipeline.solve_problem(time_limit=time_limit)

        ppb_output = self._build_ilp_result()

        return self._build_strategy_result(ppb_output)

    def _build_strategy_result(
        self,
        ppb_output: PPBOutput,
    ) -> PPStrategyResult:
        """Build a :class:`PPStrategyResult` from the ILP output.

        Translates the raw ILP result into the strategy result that
        downstream consumers use.  Simulation metrics are left at
        their defaults (``simulation_status="not_run"``, etc.);
        :class:`PPOptimizer` is responsible for running the simulator
        and merging those fields.

        Args:
            ppb_output: Output from :meth:`_build_ilp_result`.

        Returns:
            Assembled strategy result (without simulation metrics).
        """
        return PPStrategyResult(
            pp_degree=self.yaml_config.pp_degree,
            micro_batch_num=self.yaml_config.micro_batch_num,
            vpp_less_memory=self.yaml_config.vpp_less_memory,
            pipeline_bubble=None,
            **asdict(ppb_output),
        )

    def _build_ilp_result(
        self,
    ) -> PPBOutput:
        """Extract results from solved ILP and build PPBOutput.

        Returns:
            PPB output with balanced partition.
        """
        infeasible_output = self._check_ilp_solve_status()
        if infeasible_output is not None:
            return infeasible_output

        result = self._pipeline.get_result()

        try:
            stage_partition = self._extract_stage_partition(result)
        except RuntimeError as e:
            return self._make_infeasible_output(
                "Failed to extract partition from ILP solution",
                error=str(e),
            )

        try:
            layer_offset = self._extract_layer_offset_from_ilp(stage_partition)
        except RuntimeError as e:
            return self._make_infeasible_output(
                "Failed to extract layer offset from ILP solution",
                error=str(e),
            )

        vpp = self._pipeline.num_of_interleave_

        return self._build_feasible_output(
            stage_partition,
            layer_offset=layer_offset,
            num_of_interleave=vpp,
        )

    def _extract_stage_partition(self, result: Dict[str, List[List[str]]]) -> List[List[Tuple[int, RecomputeType]]]:  # pylint: disable=unused-argument
        """Extract stage partition from sapp-ppb result.

        Each BODY group has its own entry in the ILP variables (keyed by
        ``layer.name_``).  This method extracts per-group
        per-interleave-per-stage per-recompute layer counts, maps them
        to contiguous layer-ID ranges (HEAD=0, then BODY groups in
        order, then TAIL=total_body+1), and attaches the per-layer
        ``RecomputeType`` from the ILP decision variables.

        When ``num_of_interleave == 1`` (no VPP), the output has
        ``pp_degree`` entries indexed by physical stage.

        When ``num_of_interleave > 1`` (VPP enabled), the output has
        ``vpp * pp_degree`` entries treated as **virtual stages**:
        virtual stage ``v * pp_degree + s`` corresponds to VPP chunk
        ``v`` on physical stage ``s``.  Layer IDs are assigned
        per-VPP-chunk: each chunk gets a contiguous range of body
        layer IDs (chunk 0 gets the first range, chunk 1 the next,
        etc.), preserving the true layer-to-chunk mapping.

        HEAD (layer_id 0) is placed in virtual stage 0 (chunk 0,
        stage 0).  TAIL (layer_id ``total_body + 1``) is placed in
        the last virtual stage (chunk ``vpp-1``, stage
        ``pp_degree-1``).

        HEAD and TAIL layers are always annotated as
        ``RecomputeType.NONE``.

        Args:
            result: sapp-ppb result dictionary (unused, kept for API compatibility).

        Returns:
            List of ``(layer_id, RecomputeType)`` tuples per virtual
            stage.  Length is ``vpp * pp_degree`` when VPP > 1,
            otherwise ``pp_degree``.
        """
        if self._pipeline is None or self._pipeline.problem_ is None:
            raise RuntimeError("Pipeline not constructed or solved yet")

        solver = self._pipeline.problem_
        pp = self.yaml_config.pp_degree
        vpp = self._pipeline.num_of_interleave_
        num_virtual_stages = vpp * pp

        body_group_names = self._get_body_group_names()
        total_body = self._total_body_layers()

        stage_partition: List[List[Tuple[int, RecomputeType]]] = [
            [] for _ in range(num_virtual_stages)
        ]

        if vpp <= 1:
            self._assign_layers_no_vpp(
                solver, body_group_names, pp, stage_partition,
                total_body,
            )
        else:
            self._assign_layers_with_vpp(
                solver, body_group_names, pp, vpp, stage_partition,
                total_body,
            )

        for vstage in stage_partition:
            vstage.sort(key=lambda entry: entry[0])

        return stage_partition

    def _assign_layers_no_vpp(
        self,
        solver: Any,
        body_group_names: List[str],
        pp: int,  # pylint: disable=unused-argument
        stage_partition: List[List[Tuple[int, RecomputeType]]],
        total_body: int,
    ) -> None:
        """Assign layer IDs when VPP is disabled (vpp <= 1).

        Iterates over BODY groups in order and assigns contiguous
        layer IDs to each physical stage based on the summed
        (across recompute types) ILP variable values.

        Args:
            solver: ILP solver object with ``variables_`` attribute.
            body_group_names: Ordered list of BODY group names.
            _pp: Pipeline parallel degree (unused, kept for API
                consistency with :meth:`_assign_layers_with_vpp`).
            stage_partition: ``[pp]`` list to populate.
            total_body: Total number of BODY layers.
        """
        current_layer_id = 1
        for group_name in body_group_names:
            if group_name not in solver.variables_:
                raise RuntimeError(
                    f"Group '{group_name}' not found in solver variables. "
                    f"Available groups: {list(solver.variables_.keys())}"
                )
            body_lay = self._get_body_layer_by_name(group_name)
            stage_rec_counts = self._extract_group_stage_recompute(
                solver, group_name, body_lay.recompute_considered_,
            )
            for stage_id, rec_counts in enumerate(stage_rec_counts):
                for rec_type, count in rec_counts:
                    for lid in range(current_layer_id, current_layer_id + count):
                        stage_partition[stage_id].append((lid, rec_type))
                    current_layer_id += count

        if current_layer_id != total_body + 1:
            raise RuntimeError(
                f"ILP layer count mismatch: extracted {current_layer_id - 1} "
                f"body layers, expected {total_body}"
            )

        stage_partition[0].insert(0, (0, RecomputeType.NONE))
        stage_partition[-1].append((total_body + 1, RecomputeType.NONE))

    def _assign_layers_with_vpp(
        self,
        solver: Any,
        body_group_names: List[str],
        pp: int,
        vpp: int,
        stage_partition: List[List[Tuple[int, RecomputeType]]],
        total_body: int,
    ) -> None:
        """Assign layer IDs when VPP is enabled (vpp > 1).

        Each VPP chunk receives a contiguous block of body layer IDs.
        Chunk 0 gets the first block, chunk 1 the next, and so on.
        Within each chunk, BODY groups are iterated in order; for each
        group the per-stage per-recompute counts are read from the ILP
        variables for that specific interleave index.

        HEAD (layer_id 0) is placed in virtual stage 0 (chunk 0,
        stage 0).  TAIL (layer_id ``total_body + 1``) is placed in
        the last virtual stage (chunk ``vpp-1``, stage ``pp-1``).

        Args:
            solver: ILP solver object with ``variables_`` attribute.
            body_group_names: Ordered list of BODY group names.
            pp: Pipeline parallel degree.
            vpp: VPP interleaving factor.
            stage_partition: ``[vpp * pp]`` list to populate.
            total_body: Total number of BODY layers.
        """
        current_layer_id = 1
        for inter in range(vpp):
            for group_name in body_group_names:
                if group_name not in solver.variables_:
                    raise RuntimeError(
                        f"Group '{group_name}' not found in solver variables. "
                        f"Available groups: {list(solver.variables_.keys())}"
                    )
                body_lay = self._get_body_layer_by_name(group_name)
                chunk_stage_rec = self._extract_chunk_stage_recompute(
                    solver, group_name, inter, body_lay.recompute_considered_,
                )
                for stage_id, rec_counts in enumerate(chunk_stage_rec):
                    for rec_type, count in rec_counts:
                        vstage = inter * pp + stage_id
                        for lid in range(current_layer_id, current_layer_id + count):
                            stage_partition[vstage].append((lid, rec_type))
                        current_layer_id += count

        if current_layer_id != total_body + 1:
            raise RuntimeError(
                f"ILP layer count mismatch: extracted {current_layer_id - 1} "
                f"body layers, expected {total_body}"
            )

        stage_partition[0].insert(0, (0, RecomputeType.NONE))
        stage_partition[vpp * pp - 1].append((total_body + 1, RecomputeType.NONE))

    def _extract_chunk_stage_recompute(
        self,
        solver: Any,
        group_name: str,
        interleave: int,
        layer_recompute_considered: Optional[Dict[Any, bool]] = None,
    ) -> List[List[Tuple[Any, int]]]:
        """Extract per-stage per-recompute layer counts for a single VPP chunk.

        Unlike :meth:`_extract_group_stage_recompute` which sums across
        all interleaves, this method reads only the specified interleave
        index from the ILP variables.

        Args:
            solver: ILP solver object with ``variables_`` attribute.
            group_name: Name of the layer group.
            interleave: The VPP chunk (interleave) index to extract.
            layer_recompute_considered: Optional per-layer recompute
                considered dict.

        Returns:
            Per-stage list of ``(recompute_type, count)`` pairs for the
            specified interleave.
        """
        recompute_considered = layer_recompute_considered
        pp = self.yaml_config.pp_degree
        stage_rec: List[List[Tuple[Any, int]]] = [
            [] for _ in range(pp)
        ]

        for stage_id in range(pp):
            for rec in Recompute.TYPE:  # pylint: disable=E0606
                if recompute_considered and not recompute_considered.get(rec, False):
                    continue
                try:
                    var_value = solver.variables_[group_name][rec][interleave][stage_id].varValue
                    if var_value is not None:
                        count = round(var_value)
                        if count > 0:
                            stage_rec[stage_id].append((rec, count))
                except (KeyError, AttributeError):
                    continue

        return stage_rec

    def _get_body_group_names(self) -> List[str]:
        """Return ordered list of BODY group names from layers.

        Returns:
            List of group name strings, in the order the BODY layers were
            appended to the layer list.
        """
        pipeline_layer = _get_pipeline_layer_class()
        return [
            lay.name_ for lay in self._layer_builder.layers_sapp_ppb
            if lay.type_ == pipeline_layer.type_enum.BODY
        ]

    def _total_body_layers(self) -> int:
        """Return total number of BODY layers across all groups.

        Returns:
            Sum of ``nb_layer_`` for all BODY layer objects.
        """
        pipeline_layer = _get_pipeline_layer_class()
        return sum(
            lay.nb_layer_ for lay in self._layer_builder.layers_sapp_ppb
            if lay.type_ == pipeline_layer.type_enum.BODY
        )

    def _get_body_layer_by_name(self, name: str) -> Any:
        """Find a BODY layer by name from layers.

        Args:
            name: The ``name_`` of the BODY layer to find.

        Returns:
            The matching :class:`Layer` object.

        Raises:
            RuntimeError: If no BODY layer with the given name exists.
        """
        pipeline_layer = _get_pipeline_layer_class()
        for lay in self._layer_builder.layers_sapp_ppb:
            if lay.type_ == pipeline_layer.type_enum.BODY and lay.name_ == name:
                return lay
        raise RuntimeError(f"BODY layer '{name}' not found in layers")

    def _extract_group_stage_recompute(
        self,
        solver: Any,
        group_name: str,
        layer_recompute_considered: Optional[Dict[Any, bool]] = None,
    ) -> List[List[Tuple[Any, int]]]:
        """Extract per-stage per-recompute layer counts for a group.

        For each stage, returns a list of ``(Recompute.TYPE, count)``
        pairs for recompute types that have non-zero layer counts in
        that stage (summed across all interleaves).

        Args:
            solver: ILP solver object with ``variables_`` attribute.
            group_name: Name of the layer group.
            layer_recompute_considered: Optional per-layer recompute
                considered dict.  When provided, used instead of the
                solver's global ``recompute_considered_`` so that each
                BODY group is filtered by its own supported types.

        Returns:
            Per-stage list of ``(recompute_type, count)`` pairs.
        """
        recompute_considered = layer_recompute_considered
        stage_rec: List[List[Tuple[Any, int]]] = [
            [] for _ in range(self.yaml_config.pp_degree)
        ]

        for stage_id in range(self.yaml_config.pp_degree):
            for rec in Recompute.TYPE:
                if recompute_considered and not recompute_considered.get(rec, False):
                    continue
                total_count = 0
                for inter in range(self._pipeline.num_of_interleave_):
                    try:
                        var_value = solver.variables_[group_name][rec][inter][stage_id].varValue
                        if var_value is not None:
                            total_count += round(var_value)
                    except (KeyError, AttributeError):
                        continue
                if total_count > 0:
                    stage_rec[stage_id].append((rec, total_count))

        return stage_rec

    def _extract_layer_offset_from_ilp(
        self,
        stage_partition: List[List[Tuple[int, RecomputeType]]],  # pylint: disable=unused-argument
    ) -> Dict[str, List[List[int]]]:
        """Extract per-group layer offset from ILP solution using sapp-ppb native semantics.

        For each BODY group, delegates to
        :func:`Recompute.yaml_from_internal` which computes the offset
        following the sapp-ppb convention::

            offset[group_name][i][s] = actual_group[i][s] - nass[i][s]

        where ``nass`` is the naive layer assignment (pure integer division
        ``nb_layer_ // (pp * vpp)``) per ``(interleave, stage)`` cell for
        that group, and ``actual_group[i][s]`` is the total body layers the
        ILP assigned to that cell for that group (summed across recompute
        types).

        Using naive (uncorrected) nass ensures round-trip consistency with
        :func:`Recompute.internal_from_yaml` and
        :meth:`SappPipeline.print_yaml_results`, which both use the same
        naive nass baseline.

        **Edge case —** ``nb_layer_ < pp * vpp``: the naive nass is 0 for
        every cell, so the offset equals the total ILP assignment for that
        cell.  This is consistent with the round-trip.

        Args:
            stage_partition: The extracted stage partition from ILP
                (unused; offset is computed directly from solver
                variables to preserve the VPP dimension).

        Returns:
            Per-group offset dict.  Key is the BODY group name; value
            is a list of shape ``[vpp][pp]``.  Values may be negative
            when the ILP deviates from the naive uniform baseline (see
            edge-case note above).
        """
        if self._pipeline is None or self._pipeline.problem_ is None:
            return {}

        solver = self._pipeline.problem_
        pp = self.yaml_config.pp_degree
        vpp = self._pipeline.num_of_interleave_

        pipeline_layer = _get_pipeline_layer_class()

        result: Dict[str, List[List[int]]] = {}

        for lay in self._layer_builder.layers_sapp_ppb:
            if lay.type_ != pipeline_layer.type_enum.BODY:
                continue

            group_name = lay.name_
            if group_name not in solver.variables_:
                raise RuntimeError(
                    f"Cannot extract layer offset from ILP: '{group_name}' not in solver variables"
                )

            raw_nass = (
                [[lay.nb_layer_ // (pp * vpp)] * pp for _ in range(vpp)]
                if (pp * vpp) > 0
                else [[0] * pp for _ in range(vpp)]
            )

            yaml_out = Recompute.yaml_from_internal(  # pylint: disable=E0606
                vpp, pp, solver.variables_[group_name], raw_nass,
            )
            result[group_name] = yaml_out[Recompute.OFFSET]  # pylint: disable=E0606

        return result

# Copyright 2025 Huawei Technologies Co., Ltd
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
"""
Mathematical model construction for Pipeline Parallel (PP) optimization.
Defines variables, constraints and objective function for PP layer distribution,
time scheduling and memory constraint optimization using OR-Tools solver.
"""

import sys
import numpy as np
from ortools.linear_solver import pywraplp
from fast_tuner.utils.logger import logger

from fast_tuner.pipeline_conductor import pp_util
from fast_tuner.pipeline_conductor.start_service import InitConfig, HIGHS_NAME
from fast_tuner.pipeline_conductor import micro


class Model:
    """
    Mathematical model for Pipeline Parallel (PP) optimization.

    Constructs optimization model to solve:
    - Layer distribution across pipeline stages/interleaves
    - Forward/backward time scheduling of micro-batches
    - Memory constraint compliance
    - Recomputation strategy (select/full recomp)

    Attributes:
        input: InitConfig object containing PP configuration
        memory: Memory constraint parameters from init config
        expert_input: Expert optimization parameters from init config
        x_type1/x_type2: Layer count variables for type1/type2 layers
        offset: Pipeline stage offset (not used in initial version)
        f_duration/b_duration: Forward/backward time duration variables
        indicator_type1/2: Boolean indicators for layer existence
        rs_type1/2: Selective recomputation layer count variables
        ra_type1/2: Full recomputation layer count variables
        rs_or_ra: Mutually exclusive flag for rs/ra (if not supported)
        mem: Memory consumption variables per stage/interleave
        forward_s/backward_s: Start time variables for forward/backward tasks
        distribution: Micro-batch distribution across pipeline parts
        sort_micro: SortMicro object for micro-batch scheduling order
        final_orders: Final micro-batch execution order per stage
        solver: OR-Tools solver instance (HIGHS by default)
        layer_upper: Upper bound for layer count variables
        b_duration_upper: Upper bound for backward duration
        mem_upper: Upper bound for memory variables
        time_upper: Upper bound for time variables
        min_time: Optimal objective value (minimum total time)
        gap: Solver optimality gap (not used)
        model_status: Model solve status (not used)
        solution_status: Solution status (not used)
        run_time: Solver run time (not used)
    """

    def __init__(self, model_input: InitConfig):
        """
        Initialize PP optimization model.

        Args:
            model_input: InitConfig object with PP config, memory params and expert settings
        """
        # init inputs
        self.input = model_input
        self.memory = model_input.memory
        self.expert_input = model_input.expert_input
        # Initialize layers and time
        self.x_type1 = np.empty((self.input.pipeline_stage, self.input.pp_interleave_num), dtype=object)
        self.x_type2 = np.empty((self.input.pipeline_stage, self.input.pp_interleave_num), dtype=object)
        self.offset = None
        self.f_duration = np.empty((self.input.pipeline_stage, self.input.pp_interleave_num), dtype=object)
        self.b_duration = np.empty((self.input.pipeline_stage, self.input.pp_interleave_num), dtype=object)
        self.indicator_type1 = np.empty((self.input.pipeline_stage, self.input.pp_interleave_num), dtype=object)
        self.indicator_type2 = np.empty((self.input.pipeline_stage, self.input.pp_interleave_num), dtype=object)
        # Initialize select re-computation, full re-computation and memory
        self.rs_type1 = np.empty((self.input.pipeline_stage, self.input.pp_interleave_num), dtype=object)
        self.rs_type2 = np.empty((self.input.pipeline_stage, self.input.pp_interleave_num), dtype=object)
        self.ra_type1 = np.empty((self.input.pipeline_stage, self.input.pp_interleave_num), dtype=object)
        self.ra_type2 = np.empty((self.input.pipeline_stage, self.input.pp_interleave_num), dtype=object)
        self.rs_or_ra = np.empty((self.input.pipeline_stage, self.input.pp_interleave_num), dtype=object)
        self.mem = np.empty((self.input.pipeline_stage, self.input.pp_interleave_num), dtype=object)
        # The start times of the forward and backward, and the distribution of micro batches across each part
        magic_number = 5
        if self.input.micro_batch_num // self.input.pipeline_stage >= magic_number + 1:
            dummy_mbn = (self.input.micro_batch_num % self.input.pipeline_stage
                         + magic_number * self.input.pipeline_stage)
            self.residue = self.input.micro_batch_num // self.input.pipeline_stage - magic_number
            temp_parts = magic_number
            self.input.parts = magic_number
        else:
            dummy_mbn = self.input.micro_batch_num
            self.residue = 0
            temp_parts = model_input.parts
            self.input.parts = temp_parts
        self.forward_s = np.empty(
            (self.input.pipeline_stage, self.input.pp_interleave_num, self.input.parts,
             dummy_mbn, self.input.seq_splits),
            dtype=object
        )
        self.backward_s = np.empty(
            (self.input.pipeline_stage, self.input.pp_interleave_num, self.input.parts,
             dummy_mbn, self.input.seq_splits),
            dtype=object
        )
        self.distribution = pp_util.construct_distribution(dummy_mbn, self.input.pipeline_stage)
        # Formulate the forward and backward task lists
        self.sort_micro = micro.SortMicro(
            temp_parts, model_input.pp_interleave_num, model_input.pipeline_stage,
            self.distribution, self.expert_input.low_mem, model_input.seq_splits
        )
        self.final_orders = self.sort_micro.final_orders
        # Initialize the solver
        self.solver = pywraplp.Solver.CreateSolver(model_input.expert_input.solver_name)
        if not self.solver:
            self.solver = pywraplp.Solver.CreateSolver(HIGHS_NAME)
        self.layer_upper = 20
        self.b_duration_upper = self.layer_upper * 3
        self.mem_upper = 60000
        self.time_upper = 99999
        self.min_time = None
        self.gap = None
        self.model_status = None
        self.solution_status = None
        self.run_time = None

    def define_variables(self):
        """
        Define optimization variables for PP model.

        Variables include:
        - stable_dur: Stable duration for micro-batch scheduling
        - x_type1/x_type2: Layer count per stage/interleave
        - indicator_type1/2: Boolean flags for layer existence
        - f_duration/b_duration: Forward/backward time per stage/interleave
        - rs_type1/2/ra_type1/2: Recomputation layer counts
        - rs_or_ra: Mutually exclusive flag for rs/ra
        - mem: Memory consumption per stage/interleave
        - forward_s/backward_s: Start time for forward/backward tasks
        """
        self.stable_dur = self.solver.NumVar(0, self.time_upper, 'stable_duration')
        for stage in range(self.input.pipeline_stage):
            for vpp in range(self.input.pp_interleave_num):
                self.x_type1[stage][vpp] = self.solver.IntVar(0, self.layer_upper, f'x_type1_{stage}_{vpp}')
                self.x_type2[stage][vpp] = self.solver.IntVar(0, self.layer_upper, f'x_type2_{stage}_{vpp}')
                self.indicator_type1[stage][vpp] = self.solver.BoolVar(f'indicator1_{stage}_{vpp}')
                self.indicator_type2[stage][vpp] = self.solver.BoolVar(f'indicator2_{stage}_{vpp}')
                self.f_duration[stage][vpp] = self.solver.NumVar(0, self.layer_upper, f'f_duration_{stage}_{vpp}')
                self.b_duration[stage][vpp] = self.solver.NumVar(0, self.b_duration_upper, f'b_duration_{stage}_{vpp}')
                if self.input.expert_input.is_select_recomp:
                    self.rs_type1[stage][vpp] = self.solver.IntVar(
                        0, self.layer_upper, f'rs_type1_{stage}_{vpp}')
                    self.rs_type2[stage][vpp] = self.solver.IntVar(
                        0, self.layer_upper, f'rs_type2_{stage}_{vpp}')
                else:
                    self.rs_type1[stage][vpp] = self.solver.IntVar(0, 0, f'rs_type1_{stage}_{vpp}')
                    self.rs_type2[stage][vpp] = self.solver.IntVar(0, 0, f'rs_type2_{stage}_{vpp}')
                if self.input.expert_input.is_full_recomp:
                    self.ra_type1[stage][vpp] = self.solver.IntVar(0, self.layer_upper, f'ra_type1_{stage}_{vpp}')
                    self.ra_type2[stage][vpp] = self.solver.IntVar(0, self.layer_upper, f'ra_type2_{stage}_{vpp}')
                else:
                    self.ra_type1[stage][vpp] = self.solver.IntVar(0, 0, f'ra_type1_{stage}_{vpp}')
                    self.ra_type2[stage][vpp] = self.solver.IntVar(0, 0, f'ra_type2_{stage}_{vpp}')
                if not self.expert_input.is_support_ra_with_rs:
                    self.rs_or_ra[stage][vpp] = self.solver.IntVar(0, 1, f'rs_or_ra_{stage}_{vpp}')
                self.mem[stage][vpp] = self.solver.NumVar(0, self.mem_upper, f'mem_{stage}_{vpp}')

        for stage in range(self.input.pipeline_stage):
            for vpp in range(self.input.pp_interleave_num):
                for part in range(self.input.parts):
                    for micro_id in range(self.distribution[part]):
                        for split in range(self.input.seq_splits):
                            self.forward_s[stage][vpp][part][micro_id][split] = (
                                self.solver.NumVar(0, self.time_upper,
                                                   f'forward_s_{stage}_{vpp}_{part}_{micro_id}_{split}'))
                            self.backward_s[stage][vpp][part][micro_id][split] = (
                                self.solver.NumVar(0, self.time_upper,
                                                   f'backward_s{stage}_{vpp}_{part}_{micro_id}_{split}'))
        logger.info('Number of variables = %s', self.solver.NumVariables())

    def define_constraint(self):
        """
        Define all constraints for PP optimization model.

        Includes:
        - Layer allocation constraints (type1/type2 layer distribution)
        - Forward/backward time constraints (duration calculation)
        - Same stage micro-batch constraints (execution order)
        - Same micro-batch stage constraints (pipeline order)
        - Memory constraints (per stage memory limit)
        - Layer count constraints (total layers match config)
        """
        # Layer allocation constraint in the order of type1 followed by type2
        self.layer_alloc_constraints()

        # Forward and backward calculation time constraints
        self.forward_backward_time_constraints()

        # Constraints between micro-batches in the same stage
        self.same_stage_micro_constraints()
        # Constraints between the stages of micro-batch
        self.same_micro_stage_constraints()

        # Memory constraints
        self.mem_constraints()
        # Layer Constraints
        self.layers_constraints()
        if self.input.parts > 1:
            for s in range(self.input.pipeline_stage):
                self.solver.Add(self.stable_dur >= self.forward_s[s][0][-1][0][0] - self.forward_s[s][0][-2][0][0])
        logger.info("Number of constraints = %s", self.solver.NumConstraints())

    def layer_alloc_constraints(self):
        """
        Define layer allocation constraints.

        Ensures:
        - Type1 layers are distributed in non-increasing order across stages/interleaves
        - Type2 layers are distributed in non-decreasing order across stages/interleaves
        - Indicator variables match layer existence (x_type > 0 → indicator = 1)
        """
        for vpp in range(self.input.pp_interleave_num):
            for stage in range(self.input.pipeline_stage):
                self.solver.Add(self.x_type1[stage][vpp] <= self.layer_upper * self.indicator_type1[stage][vpp])
                self.solver.Add(self.x_type1[stage][vpp] >= self.indicator_type1[stage][vpp])
                self.solver.Add(self.x_type2[stage][vpp] <= self.layer_upper * self.indicator_type2[stage][vpp])
                self.solver.Add(self.x_type2[stage][vpp] >= self.indicator_type2[stage][vpp])
                if stage < self.input.pipeline_stage - 1:
                    self.solver.Add(self.indicator_type1[stage][vpp] >= self.indicator_type1[stage + 1][vpp])
                    self.solver.Add(self.indicator_type2[stage][vpp] <= self.indicator_type2[stage + 1][vpp])
            if vpp < self.input.pp_interleave_num - 1:
                self.solver.Add(
                    self.indicator_type1[self.input.pipeline_stage - 1][vpp] >= self.indicator_type1[0][vpp + 1])
                self.solver.Add(
                    self.indicator_type2[self.input.pipeline_stage - 1][vpp] <= self.indicator_type2[0][vpp + 1])

    def forward_backward_time_constraints(self):
        """
        Define forward/backward time duration constraints.

        Calculates:
        - Forward duration: type2 layers + type1 layers × layer ratio (+ head loss for last stage/interleave)
        - Backward duration: forward duration × backward ratio + recomputation overhead
        """
        if self.input.expert_input.is_head_loss_input:
            head_loss = self.input.expert_input.head_loss
        else:
            try:
                head_loss = (
                    (self.input.vocab_size / 2 / self.input.hidden_size) /
                    (1 + 1 + 3 * self.input.intermediate_size / 2 / self.input.hidden_size +
                     self.input.seq_length / self.input.hidden_size) * 1.6
                )
            except TypeError:
                head_loss = 1
        for stage in range(self.input.pipeline_stage):
            for vpp in range(self.input.pp_interleave_num):
                if stage == self.input.pipeline_stage - 1 and vpp == self.input.pp_interleave_num - 1:
                    self.solver.Add(self.f_duration[stage][vpp] == self.x_type2[stage][vpp] +
                                    self.x_type1[stage][vpp] * self.expert_input.layer_ratio + head_loss)
                else:
                    self.solver.Add(self.f_duration[stage][vpp] == self.x_type2[stage][vpp] +
                                    self.x_type1[stage][vpp] * self.expert_input.layer_ratio)
                self.solver.Add(self.b_duration[stage][vpp] == self.expert_input.backward_ratio *
                                self.f_duration[stage][vpp] + self.expert_input.srRatio *
                                (self.rs_type2[stage][vpp] + self.rs_type1[stage][vpp] * self.expert_input.layer_ratio)
                                + self.expert_input.recompute_ratio * self.ra_type2[stage][vpp] +
                                self.ra_type1[stage][vpp] * self.expert_input.layer_ratio)

    def same_stage_micro_constraints(self):
        """
        Define same-stage micro-batch execution order constraints.

        Ensures micro-batches on the same stage execute in the predefined order,
        with forward/backward tasks not overlapping (start time ≥ previous task end time).
        """
        for stage in range(self.input.pipeline_stage):
            stage_order = self.final_orders[stage]
            for i in range(len(stage_order) - 1):
                p0, vpp0, state0, id0, split0 = (stage_order[i].part, stage_order[i].vpp, stage_order[i].state,
                                                 stage_order[i].micro_id, stage_order[i].split)
                p1, vpp1, state1, id1, split1 = (stage_order[i + 1].part, stage_order[i + 1].vpp,
                                                 stage_order[i + 1].state, stage_order[i + 1].micro_id,
                                                 stage_order[i].split)
                if state0 == 'f':
                    if state1 == 'f':
                        self.solver.Add(self.forward_s[stage][vpp0][p0][id0][split0] + self.f_duration[stage][vpp0] /
                                        self.input.seq_splits <= self.forward_s[stage][vpp1][p1][id1][split1])
                    else:
                        self.solver.Add(self.forward_s[stage][vpp0][p0][id0][split0] + self.f_duration[stage][vpp0] /
                                        self.input.seq_splits <= self.backward_s[stage][vpp1][p1][id1][split1])
                else:
                    if state1 == 'f':
                        self.solver.Add(self.backward_s[stage][vpp0][p0][id0][split0] + self.b_duration[stage][vpp0] /
                                        self.input.seq_splits <= self.forward_s[stage][vpp1][p1][id1][split1])
                    else:
                        self.solver.Add(self.backward_s[stage][vpp0][p0][id0][split0] + self.b_duration[stage][vpp0] /
                                        self.input.seq_splits <= self.backward_s[stage][vpp1][p1][id1][split1])

    def _forward_constraint(self, stage, vpp, part, micro_id, split):
        """
        Helper function for forward stage constraints (extracted to reduce nesting).

        Args:
            stage: Pipeline stage index
            vpp: PP interleave index
            part: Micro-batch part index
            micro_id: Micro-batch ID
            split: Sequence split index
        """
        if stage != self.input.pipeline_stage - 1:
            self.solver.Add(
                self.forward_s[stage][vpp][part][micro_id][split] + self.f_duration[stage][vpp] /
                self.input.seq_splits <= self.forward_s[stage + 1][vpp][part][micro_id][split])
        elif vpp != self.input.pp_interleave_num - 1:
            self.solver.Add(
                self.forward_s[stage][vpp][part][micro_id][split] + self.f_duration[stage][vpp] /
                self.input.seq_splits <= self.forward_s[0][vpp + 1][part][micro_id][split])
        else:
            self.solver.Add(
                self.forward_s[stage][vpp][part][micro_id][split] + self.f_duration[stage][vpp] /
                self.input.seq_splits <= self.backward_s[stage][vpp][part][micro_id][split])

    def _backward_constraint(self, stage, vpp, part, micro_id, split):
        """
        Helper function for backward stage constraints (extracted to reduce nesting).

        Args:
            stage: Pipeline stage index
            vpp: PP interleave index
            part: Micro-batch part index
            micro_id: Micro-batch ID
            split: Sequence split index
        """
        if stage != 0:
            self.solver.Add(
                self.backward_s[stage][vpp][part][micro_id][split] + self.b_duration[stage][vpp] /
                self.input.seq_splits <= self.backward_s[stage - 1][vpp][part][micro_id][split])
        else:
            if vpp != 0:
                self.solver.Add(
                    self.backward_s[stage][vpp][part][micro_id][split] + self.b_duration[stage][vpp] /
                    self.input.seq_splits <= self.backward_s[
                        self.input.pipeline_stage - 1][vpp - 1][part][micro_id][split])

    def same_micro_stage_constraints(self):
        """
        Define same-micro-batch cross-stage constraints (pipeline order).

        Ensures:
        - Forward tasks flow from stage N to N+1 (or interleave wrap-around)
        - Backward tasks flow from stage N to N-1 (or interleave wrap-around)
        """
        for part in range(self.input.parts):
            for micro_id in range(self.distribution[part]):
                for split in range(self.input.seq_splits):
                    for vpp in range(self.input.pp_interleave_num):
                        for stage in range(self.input.pipeline_stage):
                            self._forward_constraint(stage, vpp, part, micro_id, split)
                            self._backward_constraint(stage, vpp, part, micro_id, split)

    def layers_constraints(self):
        """
        Define total layer count constraints.

        Ensures:
        - Sum of type1/type2 layers matches config values
        - Recomputation layers (rs/ra) do not exceed total layers
        - RS/RA are mutually exclusive (if not supported)
        - Indicator variables sum to reasonable range
        """
        layers_type1 = 0
        layers_type2 = 0
        indicator_total = 0
        for stage in range(self.input.pipeline_stage):
            for vpp in range(self.input.pp_interleave_num):
                layers_type1 += self.x_type1[stage][vpp]
                layers_type2 += self.x_type2[stage][vpp]
                indicator_total += self.indicator_type1[stage][vpp] + self.indicator_type2[stage][vpp]
                self.solver.Add(self.x_type1[stage][vpp] >= self.ra_type1[stage][vpp] + self.rs_type1[stage][vpp])
                self.solver.Add(self.x_type2[stage][vpp] >= self.ra_type2[stage][vpp] + self.rs_type2[stage][vpp])
                if not self.expert_input.is_support_ra_with_rs:
                    self.solver.Add(self.rs_type1[stage][vpp] <= self.layer_upper * self.rs_or_ra[stage][vpp])
                    self.solver.Add(self.rs_type2[stage][vpp] <= self.layer_upper * self.rs_or_ra[stage][vpp])
                    self.solver.Add(self.ra_type1[stage][vpp] <= self.layer_upper * (1 - self.rs_or_ra[stage][vpp]))
                    self.solver.Add(self.ra_type2[stage][vpp] <= self.layer_upper * (1 - self.rs_or_ra[stage][vpp]))
        self.solver.Add(layers_type1 == self.input.num_layers_type1)
        self.solver.Add(layers_type2 == self.input.num_layers_type2)
        self.solver.Add(indicator_total >= self.input.pipeline_stage * self.input.pp_interleave_num)
        self.solver.Add(indicator_total <= self.input.pipeline_stage * self.input.pp_interleave_num + 1)

    def mem_constraints(self):
        """
        Define memory constraints per pipeline stage.

        Calculates:
        - Per stage/interleave memory consumption (activation + recomputation + layer memory)
        - Total memory consumption (layer memory + activation memory from task execution)
        - Ensures memory consumption ≤ stage-specific memory limits
        """
        for stage in range(self.input.pipeline_stage):
            for vpp in range(self.input.pp_interleave_num):
                self.solver.Add(self.mem[stage][vpp] == self.memory.act_mem *
                                (self.x_type2[stage][vpp] - self.rs_type2[stage][vpp] - self.ra_type2[stage][vpp]) +
                                self.memory.select_mem * self.rs_type2[stage][vpp] +
                                self.memory.re_comp_mem * self.ra_type2[stage][vpp] +
                                self.memory.act_mem0 * (self.x_type1[stage][vpp] - self.rs_type1[stage][vpp] -
                                                        self.ra_type1[stage][vpp]) + self.memory.select_mem0 *
                                self.rs_type1[stage][vpp] + self.memory.re_comp_mem0 * self.ra_type1[stage][vpp])
        for stage in range(self.input.pipeline_stage):
            total_layer_mem = 0
            for vpp in range(self.input.pp_interleave_num):
                total_layer_mem += (self.x_type1[stage][vpp] * self.memory.layer_mem012 + self.x_type2[stage][vpp] *
                                    self.memory.layer_mem)
            for sub in range(1, len(self.final_orders[stage]) + 1):
                consume_mem = total_layer_mem
                for i in range(sub):
                    _, vpp, state, _, _ = (self.final_orders[stage][i].part,
                                           self.final_orders[stage][i].vpp,
                                           self.final_orders[stage][i].state,
                                           self.final_orders[stage][i].micro_id,
                                           self.final_orders[stage][i].split)
                    # The computation time is equally divided,
                    # so the activated memory here is also processed with equal division.
                    if state == 'f':
                        consume_mem += self.mem[stage][vpp] / self.input.seq_splits
                    else:
                        consume_mem -= self.mem[stage][vpp] / self.input.seq_splits
                if stage == 0:
                    self.solver.Add(consume_mem <= self.memory.mem_lim_stage0)
                elif stage == self.input.pipeline_stage - 1:
                    self.solver.Add(consume_mem <= self.memory.mem_lim_last)
                else:
                    self.solver.Add(consume_mem <= self.memory.mem_lim_others)

    def define_obj(self):
        """
        Define optimization objective function.

        Minimizes total pipeline execution time:
        - Last backward task end time + residual micro-batch stable duration
        """
        self.solver.Minimize(self.backward_s[0][0][-1][self.distribution[-1] - 1][0] + self.b_duration[0][0] /
                             self.input.seq_splits + self.residue * self.stable_dur)

    def output_model(self, mps_file):
        """
        Export optimization model to MPS file (OR-Tools format).

        Args:
            mps_file: Path to output MPS file
        """
        with open(mps_file, 'w', encoding='utf-8') as file:
            mps_text = self.solver.ExportModelAsMpsFormat(False, False)
            file.write(mps_text)
        logger.info('Had write to file: %s', mps_file)

    def solve(self):
        """
        Solve the PP optimization model.

        Configures solver (time limit, output), runs solve, and stores optimal objective value.
        """
        self.solver.EnableOutput()
        logger.info('Solving with %s', self.solver.SolverVersion())
        if self.expert_input.time_limit != sys.maxsize:
            self.solver.SetTimeLimit(self.expert_input.time_limit)
        self.solver.Solve()
        self.min_time = self.solver.Objective().Value()
        logger.info('The objective value = %s', self.min_time)


if __name__ == '__main__':
    yaml_file = 'path/to/your_file.yaml'

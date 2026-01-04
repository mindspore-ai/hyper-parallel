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
Memory fitting module for pipeline parallel memory correction.
Provides linear least squares fitting to adjust memory parameters and check memory overflow,
ensuring each pipeline stage complies with memory limits.
"""

import numpy as np
from fast_tuner.utils.logger import logger
from fast_tuner.pipeline_conductor import solution
from fast_tuner.pipeline_conductor import dryrun
from fast_tuner.pipeline_conductor import pp_util


class FitMem:
    """
    Memory fitting class for pipeline parallel training.

    Core functions:
    - Perform linear least squares fitting to correct memory parameters
    - Check memory overflow for each pipeline stage via dryrun
    - Adjust memory limits based on overflow amount

    Attributes:
        cur_peak_mem: List of peak memory values for each pipeline stage
        cur_solution: Solution object containing pipeline parallel config and peak memory stats
        peaks: Peak number statistics of different memory types (act/recomp/select)
        init_config: Initial pipeline parallel configuration
        init_memory: Initial memory parameter config from init_config
    """
    cur_peak_mem = []

    def __init__(self, cur_solution: solution.Solution):
        """
        Initialize FitMem instance with solution object.

        Args:
            cur_solution: Solution object containing pipeline config, peak memory, layer distribution
        """
        self.cur_solution = cur_solution
        self.cur_peak_mem = cur_solution.check_peak_mem
        self.peaks = cur_solution.peak_num
        self.init_config = cur_solution.init_config
        self.init_memory = cur_solution.init_config.memory

    def linear_fit(self):
        """
        Perform linear least squares fitting to correct memory parameters.

        Fitting logic varies by recompilation config (select_recomp/full_recomp):
        - Determines fitting dimension based on pipeline stage and recomp config
        - Forms coefficient matrix and target array
        - Solves least squares problem and corrects memory parameters if fitting is valid
        """
        if self.init_config.expert_input.is_select_recomp and self.init_config.expert_input.is_full_recomp:
            length_x_lim = 11
        elif not self.init_config.expert_input.is_select_recomp and not self.init_config.expert_input.is_full_recomp:
            length_x_lim = 7
        else:
            length_x_lim = 9
        if self.init_config.pipeline_stage < length_x_lim:
            length_x = self.init_config.pipeline_stage
        else:
            length_x = length_x_lim
        if length_x < 4:
            logger.warning("can not fit for pipeline_stage = %s", self.init_config.pipeline_stage)
            return
        coe_a, array_b = self.form_coe_matrix_mem_array(length_x)
        np.set_printoptions(suppress=True, precision=3)  # retain 3 decimal places
        modified_mem, res, rank, _ = np.linalg.lstsq(coe_a, array_b, rcond=None)
        logger.info('the residual = %s', np.linalg.norm(res))
        logger.info('the rank = %s', rank)
        if np.linalg.norm(res) > 1e-3 or rank < self.init_config.pipeline_stage:
            logger.warning("The distribution can not correct the memory!")
        else:
            modified_mem = list(np.round(np.array(modified_mem), decimals=1))
            self.correct_mem(modified_mem)

    def correct_mem(self, modify_mem):
        """
        Correct initial memory parameters using fitted values from linear_fit.

        The number of fitted parameters varies by recompilation config:
        - Basic params (4): static_mem0/static_mem/lm_head_mem/act_mem
        - Extended params: layer_mem/re_comp_mem/layer_mem012/act_mem12/re_comp_mem12/select_mem etc.

        Args:
            modify_mem: List of fitted memory values (length 4-11 based on recomp config)
        """
        self.init_memory.static_mem0 = modify_mem[0]
        self.init_memory.static_mem = modify_mem[1]
        self.init_memory.lm_head_mem = modify_mem[2]
        self.init_memory.act_mem = modify_mem[3]

        if len(modify_mem) == 5:
            self.init_memory.layer_mem = modify_mem[4]
        if len(modify_mem) == 6:
            self.init_memory.layer_mem = modify_mem[4]
            self.init_memory.re_comp_mem = modify_mem[5]
        if len(modify_mem) == 8:
            self.init_memory.layer_mem = modify_mem[4]
            self.init_memory.re_comp_mem = modify_mem[5]
            self.init_memory.layer_mem012 = modify_mem[6]
            self.init_memory.act_mem12 = modify_mem[7]
            self.init_memory.act_mem0 = modify_mem[7]
        if self.init_config.expert_input.is_select_recomp:
            if len(modify_mem) == 9:
                self.init_memory.layer_mem = modify_mem[4]
                self.init_memory.re_comp_mem = modify_mem[5]
                self.init_memory.layer_mem012 = modify_mem[6]
                self.init_memory.act_mem12 = modify_mem[7]
                self.init_memory.re_comp_mem12 = modify_mem[8]
                self.init_memory.re_comp_mem0 = modify_mem[8]
            if len(modify_mem) == 10:
                self.init_memory.layer_mem = modify_mem[4]
                self.init_memory.re_comp_mem = modify_mem[5]
                self.init_memory.layer_mem012 = modify_mem[6]
                self.init_memory.act_mem12 = modify_mem[7]
                self.init_memory.re_comp_mem12 = modify_mem[8]
                self.init_memory.re_comp_mem0 = modify_mem[8]
                self.init_memory.select_mem = modify_mem[9]
            if len(modify_mem) > 10:
                self.init_memory.layer_mem = modify_mem[4]
                self.init_memory.re_comp_mem = modify_mem[5]
                self.init_memory.layer_mem012 = modify_mem[6]
                self.init_memory.act_mem12 = modify_mem[7]
                self.init_memory.re_comp_mem12 = modify_mem[8]
                self.init_memory.re_comp_mem0 = modify_mem[8]
                self.init_memory.select_mem = modify_mem[9]
                self.init_memory.select_mem12 = modify_mem[10]
                self.init_memory.select_mem0 = self.init_memory.select_mem12
        else:
            if len(modify_mem) >= 9:
                self.init_memory.layer_mem = modify_mem[4]
                self.init_memory.re_comp_mem = modify_mem[5]
                self.init_memory.layer_mem012 = modify_mem[6]
                self.init_memory.act_mem12 = modify_mem[7]
                self.init_memory.re_comp_mem12 = modify_mem[8]
                self.init_memory.re_comp_mem0 = modify_mem[8]
        self.init_memory.update_up_mem()
        logger.info('The correct memory information:')
        self.init_memory.print_mem()

    def form_coe_matrix_mem_array(self, length_x):
        """
        Form coefficient matrix (A) and target array (b) for least squares fitting.

        Constructs matrix based on pipeline stage and recompilation config,
        calculates target array by subtracting fixed memory terms from peak memory.

        Args:
            length_x: Column length of coefficient matrix (4/6/8/9/10/11 based on recomp config)

        Returns:
            Tuple of (coe_a, array_b):
                - coe_a: 2D numpy array (pipeline_stage × length_x) coefficient matrix
                - array_b: 1D numpy array (pipeline_stage) target array
        """
        coe_a = np.empty((self.init_config.pipeline_stage, length_x), float)
        array_b = np.empty(self.init_config.pipeline_stage, float)
        for stage in range(self.init_config.pipeline_stage):
            if stage == 0:
                coe_a[stage][0] = 1
                coe_a[stage][1] = 0
                coe_a[stage][2] = 0
            elif stage == self.init_config.pipeline_stage - 1:
                coe_a[stage][0] = 0
                coe_a[stage][1] = 0
                coe_a[stage][2] = 1
            else:
                coe_a[stage][0] = 0
                coe_a[stage][1] = 1
                coe_a[stage][2] = 0
            coe_a[stage][3] = self.cur_solution.peak_num.peak_num_act_type2[stage]
            if length_x == 4:
                array_b[stage] = (
                    self.cur_peak_mem[stage]
                    - self.cur_solution.layer2_dis_stage[stage] * self.init_memory.layer_mem
                    - self.peaks.peak_num_recompute_type2[stage] * self.init_memory.re_comp_mem
                    - self.cur_solution.layer1_dis_stage[stage] * self.init_memory.layer_mem012
                    - self.peaks.peak_num_act_type1[stage] * self.init_memory.act_mem12
                    - self.peaks.peak_num_recompute_type1[stage] * self.init_memory.re_comp_mem12
                    - self.peaks.peak_num_select_recom_type2[stage] * self.init_memory.select_mem
                    - self.peaks.peak_num_select_recom_type1 * self.init_memory.select_mem12
                )
                continue
            if length_x == 6:
                coe_a[stage][4] = self.cur_solution.layer2_dis_stage[stage]
                coe_a[stage][5] = self.peaks.peak_num_recompute_type2[stage]
                array_b[stage] = (
                    self.cur_peak_mem[stage]
                    - self.cur_solution.layer1_dis_stage[stage] * self.init_memory.layer_mem012
                    - self.peaks.peak_num_act_type1[stage] * self.init_memory.act_mem12
                    - self.peaks.peak_num_recompute_type1[stage] * self.init_memory.re_comp_mem12
                    - self.peaks.peak_num_select_recom_type2[stage] * self.init_memory.select_mem
                    - self.peaks.peak_num_select_recom_type1 * self.init_memory.select_mem12
                )
                continue
            if length_x == 8:
                coe_a[stage][4] = self.cur_solution.layer2_dis_stage[stage]
                coe_a[stage][5] = self.peaks.peak_num_recompute_type2[stage]
                coe_a[stage][6] = self.cur_solution.layer1_dis_stage[stage]
                coe_a[stage][7] = self.peaks.peak_num_act_type1[stage]
                array_b[stage] = (
                    self.cur_peak_mem[stage]
                    - self.peaks.peak_num_recompute_type1[stage] * self.init_memory.re_comp_mem12
                    - self.peaks.peak_num_select_recom_type2[stage] * self.init_memory.select_mem
                    - self.peaks.peak_num_select_recom_type1[stage] * self.init_memory.select_mem12
                )
                continue
            if length_x == 9:
                coe_a[stage][4] = self.cur_solution.layer2_dis_stage[stage]
                coe_a[stage][5] = self.peaks.peak_num_recompute_type2[stage]
                coe_a[stage][6] = self.cur_solution.layer1_dis_stage[stage]
                coe_a[stage][7] = self.peaks.peak_num_act_type1[stage]
                coe_a[stage][8] = self.peaks.peak_num_recompute_type1[stage]
                array_b[stage] = (
                    self.cur_peak_mem[stage]
                    - self.peaks.peak_num_select_recom_type2[stage] * self.init_memory.select_mem
                    - self.peaks.peak_num_select_recom_type1 * self.init_memory.select_mem12
                )
                continue
            if self.init_config.expert_input.is_select_recomp:
                if length_x == 10:
                    coe_a[stage][4] = self.cur_solution.layer2_dis_stage[stage]
                    coe_a[stage][5] = self.peaks.peak_num_recompute_type2[stage]
                    coe_a[stage][6] = self.cur_solution.layer1_dis_stage[stage]
                    coe_a[stage][7] = self.peaks.peak_num_act_type1[stage]
                    coe_a[stage][8] = self.peaks.peak_num_recompute_type1[stage]
                    coe_a[stage][9] = self.peaks.peak_num_select_recom_type2[stage]
                    array_b[stage] = (
                        self.cur_peak_mem[stage]
                        - self.peaks.peak_num_select_recom_type1 * self.init_memory.select_mem12
                    )
                    continue
                if length_x >= 11:
                    coe_a[stage][4] = self.cur_solution.layer2_dis_stage[stage]
                    coe_a[stage][5] = self.peaks.peak_num_recompute_type2[stage]
                    coe_a[stage][6] = self.cur_solution.layer1_dis_stage[stage]
                    coe_a[stage][7] = self.peaks.peak_num_act_type1[stage]
                    coe_a[stage][8] = self.peaks.peak_num_recompute_type1[stage]
                    coe_a[stage][9] = self.peaks.peak_num_select_recom_type2[stage]
                    coe_a[stage][10] = self.peaks.peak_num_select_recom_type1[stage]
                    array_b[stage] = self.cur_peak_mem[stage]
                    continue
            else:
                if length_x > 9:
                    coe_a[stage][4] = self.cur_solution.layer2_dis_stage[stage]
                    coe_a[stage][5] = self.peaks.peak_num_recompute_type2[stage]
                    coe_a[stage][6] = self.cur_solution.layer1_dis_stage[stage]
                    coe_a[stage][7] = self.peaks.peak_num_act_type1[stage]
                    coe_a[stage][8] = self.peaks.peak_num_recompute_type1[stage]
                    array_b[stage] = self.cur_peak_mem[stage]
                    continue
        return coe_a, array_b

    def is_over_mem(self):
        """
        Check if any pipeline stage exceeds memory limit via dryrun.

        If no overflow: write config file (yaml/shell) based on dryrun config.
        If overflow: return overflow amount and flag.

        Returns:
            Tuple of (over_mem, is_over):
                - over_mem: Dict {stage: overflow_amount}, 0 if no overflow
                - is_over: Boolean, True if any stage overflows, False otherwise
        """
        over_mem = self.get_over_mem(self.cur_solution)
        if all(over_mem[stage] == 0 for stage in range(len(over_mem))):
            if dryrun.DryRun.is_write_to_file:
                if dryrun.DryRun.config_file_type == 0:
                    recompute_config = pp_util.build_recompute_config(True, True, self.cur_solution.rs_dis.tolist(),
                                                                      self.cur_solution.ra_dis.tolist())
                    pp_util.write_config_to_yaml(recompute_config, self.cur_solution.offset.tolist(), self.init_config.
                                                 config_file)
                elif dryrun.DryRun.config_file_type == 1:
                    pp_util.write_config_to_shell(self.cur_solution.offset.tolist(), self.init_config.
                                                 config_file)
                else:
                    raise TypeError(dryrun.dryrun_config_error)
                logger.info('The result is available for training, config has write to %s!',
                            self.init_config.config_file)
            return over_mem, False

        return over_mem, True

    def reduce_mem_lim_for_fitting(self, over_mem, i):
        """
        Reduce memory limits for fitting based on overflow amount.

        Adjusts mem_lim_stage0/mem_lim_last/mem_lim_others by scaling overflow amount with iteration index.

        Args:
            over_mem: Dict {stage: overflow_amount}
            i: Iteration index (scaling factor for overflow reduction)
        """
        if 0 in over_mem:
            self.init_config.memory.mem_lim_stage0 -= over_mem[0] * (i + 1)
        last_stage = self.init_config.pipeline_stage - 1
        if last_stage in over_mem:
            self.init_config.memory.mem_lim_last -= over_mem[last_stage] * (i + 1)
        over_mem = {key: value for key, value in over_mem.items()
                    if key not in (0, last_stage)}
        if over_mem:
            self.init_config.memory.mem_lim_others -= max(over_mem.values()) * (i + 1)

    def set_peak_mem_by_dryrun(self, cur_solution: solution.Solution):
        """
        Set peak memory values by running dryrun and extracting memory info.

        Executes dryrun with recomp config and layer distribution,
        updates cur_solution.check_peak_mem and self.cur_peak_mem.

        Args:
            cur_solution: Solution object to update with dryrun peak memory
        """
        init_config = cur_solution.init_config
        expert_input = init_config.expert_input
        dry_run = dryrun.DryRun(expert_input.config_file, expert_input.ms_adapter_file,
                                expert_input.double_check_dryrun_filename)
        recompute_config = pp_util.build_recompute_config(True, True, cur_solution.rs_dis.
                                                          tolist(), cur_solution.ra_dis.tolist())
        total_number = init_config.num_layers_type1 + init_config.num_layers_type2
        dry_run.start_dryrun(recompute_config, cur_solution.offset.tolist(), total_number,
                             init_config.pp_interleave_num,
                             init_config.pipeline_stage, init_config.rank_size, init_config.num_layers_type1,
                             init_config.micro_batch_num)
        cur_solution.check_peak_mem = dry_run.extract_memory_info_act(init_config.pipeline_stage)
        self.cur_peak_mem = cur_solution.check_peak_mem
        logger.info('check peak_mem = %s', cur_solution.check_peak_mem)

    def get_over_mem(self, cur_solution: solution.Solution):
        """
        Calculate memory overflow amount for each pipeline stage.

        Calls set_peak_mem_by_dryrun to get latest peak memory,
        computes overflow as (peak_mem - mem_lim) for each stage.

        Args:
            cur_solution: Solution object with dryrun peak memory

        Returns:
            Dict {stage: overflow_amount}, 0 if peak_mem <= mem_lim
        """
        self.set_peak_mem_by_dryrun(cur_solution)
        over_mem = {}
        init_config = cur_solution.init_config
        for stage in range(init_config.pipeline_stage):
            if cur_solution.check_peak_mem[stage] - init_config.mem_lim > 0:
                logger.warning('The stage%s result is over memory limit! please check the offset!', stage)
                over_mem[stage] = cur_solution.check_peak_mem[stage] - init_config.mem_lim
            else:
                over_mem[stage] = 0
        return over_mem

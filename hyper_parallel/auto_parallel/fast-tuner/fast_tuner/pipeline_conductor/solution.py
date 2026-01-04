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
'''
This Module consists a couple of utility functions
for solution. Including parser, peak memory calculation
and validity check
'''

import itertools
import re
import numpy as np
from fast_tuner.pipeline_conductor.start_service import InitConfig
from fast_tuner.pipeline_conductor.math_model import Model
from fast_tuner.pipeline_conductor import micro
from fast_tuner.pipeline_conductor.start_service import ExpertInput
from fast_tuner.utils.logger import logger

class Solution:
    '''
    parse and record various properties of
    solutions, including a few simple analysis
    '''
    x_type1 = [int]
    x_type2 = [int]
    layer_dis = [int]
    layer1_dis_stage = [int]
    layer2_dis_stage = [int]
    offset = [int]
    indicator_type1 = [int]
    indicator_type2 = [int]
    rs_type1 = [int]
    rs_type2 = [int]
    rs_dis = [int]
    ra_type1 = [int]
    ra_type2 = [int]
    ra_dis = [int]
    forward_s = [float]
    backward_s = [float]
    peak_num = micro.PeakNum
    check_peak_mem = []
    object_value = float
    gap = float
    solution_status = str
    model_status = str
    run_time = float
    sol_file = ''

    def __init__(self, init_config: InitConfig):
        self.init_config = init_config
        self.x_type1 = np.empty((self.init_config.pp_interleave_num, self.init_config.pipeline_stage), int)
        self.x_type2 = np.empty((self.init_config.pp_interleave_num, self.init_config.pipeline_stage), int)
        self.offset = np.empty((self.init_config.pp_interleave_num, self.init_config.pipeline_stage), int)
        self.indicator_type1 = np.empty((self.init_config.pp_interleave_num, self.init_config.pipeline_stage), int)
        self.indicator_type2 = np.empty((self.init_config.pp_interleave_num, self.init_config.pipeline_stage), int)
        self.rs_type1 = np.empty((self.init_config.pp_interleave_num, self.init_config.pipeline_stage), int)
        self.rs_type2 = np.empty((self.init_config.pp_interleave_num, self.init_config.pipeline_stage), int)
        self.ra_type1 = np.empty((self.init_config.pp_interleave_num, self.init_config.pipeline_stage), int)
        self.ra_type2 = np.empty((self.init_config.pp_interleave_num, self.init_config.pipeline_stage), int)
        self.forward_s = np.empty((self.init_config.pipeline_stage, self.init_config.pp_interleave_num,
                                   self.init_config.parts, self.init_config.micro_batch_num,
                                   self.init_config.seq_splits), dtype=float)
        self.backward_s = np.empty((self.init_config.pipeline_stage, self.init_config.pp_interleave_num,
                                   self.init_config.parts, self.init_config.micro_batch_num,
                                   self.init_config.seq_splits), dtype=float)

    def set_solution(self, solved_model: Model, is_origin_solver, sol_file):
        '''
        parse the raw MIP solution into pipeline
        parallelism layer distributions
        '''
        self.sol_file = sol_file
        if is_origin_solver:
            for stage in range(self.init_config.pipeline_stage):
                for vpp in range(self.init_config.pp_interleave_num):
                    self.x_type1[vpp][stage] = solved_model.x_type1[stage][vpp].solution_value()
                    self.x_type2[vpp][stage] = solved_model.x_type2[stage][vpp].solution_value()
                    self.indicator_type1[vpp][stage] = solved_model.indicator_type1[stage][vpp].solution_value()
                    self.indicator_type2[vpp][stage] = solved_model.indicator_type2[stage][vpp].solution_value()
                    self.rs_type1[vpp][stage] = solved_model.rs_type1[stage][vpp].solution_value()
                    self.rs_type2[vpp][stage] = solved_model.rs_type2[stage][vpp].solution_value()
                    self.ra_type1[vpp][stage] = solved_model.ra_type1[stage][vpp].solution_value()
                    self.ra_type2[vpp][stage] = solved_model.ra_type2[stage][vpp].solution_value()
                    for part in range(self.init_config.parts):
                        for micro_ind, seq in itertools.product(range(solved_model.distribution[part]),
                                                                range(self.init_config.seq_splits)):
                            self.forward_s[stage][vpp][part][micro_ind][seq] = (
                                solved_model.forward_s[stage][vpp][part][micro_ind][seq].solution_value())
                            self.backward_s[stage][vpp][part][micro_ind][seq] = (
                                solved_model.backward_s[stage][vpp][part][micro_ind][seq].solution_value())
        else:
            with open(sol_file, 'r', encoding='utf-8') as file:
                for line in file:
                    content1 = re.search(r'^x_type1_(\d+)_(\d+)\s+([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)',
                                         line)
                    content2 = re.search(r'^x_type2_(\d+)_(\d+)\s+([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)',
                                         line)
                    content3 = re.search(r'^indicator1_(\d+)_(\d+)\s+([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)',
                                         line)
                    content4 = re.search(r'^indicator2_(\d+)_(\d+)\s+([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)',
                                         line)
                    content5 = re.search(r'^rs_type1_(\d+)_(\d+)\s+([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)',
                                         line)
                    content6 = re.search(r'^rs_type2_(\d+)_(\d+)\s+([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)',
                                         line)
                    content7 = re.search(r'^ra_type1_(\d+)_(\d+)\s+([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)',
                                         line)
                    content8 = re.search(r'^ra_type2_(\d+)_(\d+)\s+([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)',
                                         line)
                    content9 = re.search(
                        r'^forward_s_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)\s+([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)',
                        line)
                    content10 = re.search(
                        r'^backward_s(\d+)_(\d+)_(\d+)_(\d+)_(\d+)\s+([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)',
                        line)
                    if content1:
                        self.x_type1[int(content1.group(2))][int(content1.group(1))] = round(float(content1.group(3)))
                    if content2:
                        self.x_type2[int(content2.group(2))][int(content2.group(1))] = round(float(content2.group(3)))
                    if content3:
                        self.indicator_type1[int(content3.group(2))][int(content3.group(1))] = round(
                            float(content3.group(3)))
                    if content4:
                        self.indicator_type2[int(content4.group(2))][int(content4.group(1))] = round(
                            float(content4.group(3)))
                    if content5:
                        self.rs_type1[int(content5.group(2))][int(content5.group(1))] = round(float(content5.group(3)))
                    if content6:
                        self.rs_type2[int(content6.group(2))][int(content6.group(1))] = round(float(content6.group(3)))
                    if content7:
                        self.ra_type1[int(content7.group(2))][int(content7.group(1))] = round(float(content7.group(3)))
                    if content8:
                        self.ra_type2[int(content8.group(2))][int(content8.group(1))] = round(float(content8.group(3)))

                    if content9:
                        self.forward_s[int(content9.group(1))][int(content9.group(2))][int(content9.group(3))][
                            int(content9.group(4))][int(content9.group(5))] = float(content9.group(6))
                    if content10:
                        self.backward_s[int(content10.group(1))][int(content10.group(2))][int(content10.group(3))][
                            int(content10.group(4))][int(content10.group(5))] = float(content10.group(6))
        self.object_value = solved_model.min_time
        self.gap = solved_model.gap
        self.model_status = solved_model.model_status
        self.solution_status = solved_model.solution_status
        self.run_time = solved_model.run_time
        if self.solution_status != 'None':
            self.cal_peak_mem(solved_model)
            self.set_total_dis()
            self.check_time_list()

    def cal_peak_mem(self, solved_model: Model):
        self.peak_num = micro.PeakNum(solved_model.sort_micro)
        self.peak_num.set_peak_act_recompute_num(self.x_type2, self.rs_type2, self.ra_type2,
                                                 self.x_type1, self.rs_type1, self.ra_type1,
                                                 solved_model.memory)

    def set_total_dis(self):
        self.layer_dis = self.x_type1 + self.x_type2
        self.offset = (self.layer_dis - (self.init_config.num_layers_type1 + self.init_config.num_layers_type2) //
                       (self.init_config.pp_interleave_num * self.init_config.pipeline_stage))
        self.rs_dis = self.rs_type1 + self.rs_type2
        self.ra_dis = self.ra_type1 + self.ra_type2
        self.layer1_dis_stage = [sum(self.x_type1[v][s] for v in range(self.init_config.pp_interleave_num)) for s in
                                 range(self.init_config.pipeline_stage)]
        self.layer2_dis_stage = [sum(self.x_type2[v][s] for v in range(self.init_config.pp_interleave_num)) for s in
                                 range(self.init_config.pipeline_stage)]

    def check_time_list(self):
        '''
        inner functions for testing whether
        the starting times align with exec order
        '''
        s_time = [[] for _ in range(self.init_config.pipeline_stage)]
        for stage in range(self.init_config.pipeline_stage):
            for i in range(len(self.peak_num.sort_micro.final_orders[stage])):
                micro_batch = self.peak_num.sort_micro.final_orders[stage][i]
                if micro_batch.state == 'f':
                    s_time[stage].append(self.forward_s[stage][micro_batch.vpp][micro_batch.part][micro_batch.micro_id]
                                         [micro_batch.split])
                if micro_batch.state == 'b':
                    s_time[stage].append(self.backward_s[stage][micro_batch.vpp][micro_batch.part][micro_batch.micro_id]
                                         [micro_batch.split])
        # 各stage的micro0batch start time单调递增
        for stage in range(self.init_config.pipeline_stage):
            for i in range(len(self.peak_num.sort_micro.final_orders[stage]) - 1):
                if s_time[stage][i] > s_time[stage][i + 1]:
                    raise ValueError('Time sequence error!')

    def solution_print(self):
        '''print the information of the solution'''
        logger.info('layer distribution: ')
        logger.info(self.layer_dis.tolist())
        logger.info('layer of type1 distribution: ')
        logger.info(self.x_type1.tolist())
        logger.info('the indicator of layer1: ')
        logger.info(self.indicator_type1.tolist())

        logger.info('layer of type2 distribution: ')
        logger.info(self.x_type2.tolist())
        logger.info('the indicator of layer2: ')
        logger.info(self.indicator_type2.tolist())
        logger.info('the offset: ')
        logger.info(self.offset.tolist())

        logger.info('rs distribution: ')
        logger.info(self.rs_dis.tolist())
        logger.info('layer of type1 rs distribution: ')
        logger.info(self.rs_type1.tolist())
        logger.info('layer of type2 rs distribution: ')
        logger.info(self.rs_type2.tolist())

        logger.info('ra distribution: ')
        logger.info(self.ra_dis.tolist())
        logger.info('layer of type1 ra distribution: ')
        logger.info(self.ra_type1.tolist())
        logger.info('layer of type2 ra distribution: ')
        logger.info(self.ra_type2.tolist())

        logger.info('the peak memory:')
        logger.info(list(self.peak_num.max_mem.values()))
        logger.info('the number of micro batch when peak memory')
        logger.info(list(self.peak_num.micro_num_of_max_mem.values()))

        logger.info('the number of activations of the type of layer1')
        logger.info(list(self.peak_num.peak_num_act_type1.values()))
        logger.info('the number of select recomputes of the type of layer1')
        logger.info(list(self.peak_num.peak_num_select_recom_type1.values()))
        logger.info('the number of full recomputes of the type of layer1')
        logger.info(list(self.peak_num.peak_num_recompute_type1.values()))

        logger.info('the number of activations of the type of layer2')
        logger.info(list(self.peak_num.peak_num_act_type2.values()))
        logger.info('the number of select recomputes of the type of layer2')
        logger.info(list(self.peak_num.peak_num_select_recom_type2.values()))
        logger.info('the number of full recomputes of the type of layer2')
        logger.info(list(self.peak_num.peak_num_recompute_type2.values()))

def extract_solution_file(yaml_file, sol_file):
    expert_input = ExpertInput(yaml_file, '')
    expert_input.is_dryrun = False
    init_config = InitConfig(expert_input)
    origin_model = Model(init_config)
    file_solution = Solution(init_config)
    file_solution.set_solution(origin_model, False, sol_file)
    file_solution.solution_print()

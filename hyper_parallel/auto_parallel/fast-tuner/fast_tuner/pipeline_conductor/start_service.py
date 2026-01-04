# Copyright 2024 Huawei Technologies Co., Ltd
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
"""Pipeline conductor start service module.

This module provides classes for initializing and configuring pipeline parallelism
for deep learning models, supporting various model types and configuration formats.
"""
import os
import re
import numpy as np
import yaml

from fast_tuner.utils.logger import logger
from fast_tuner.pipeline_conductor import dryrun, pp_util
from fast_tuner.pipeline_conductor import micro
from fast_tuner.pipeline_conductor.memory import Memory

pipeline_output_file = 'pipeline_output'
init_dryrun_file = 'init_dryrun'
double_check_dryrun_filename = 'double_check_dryrun'
HIGHS_NAME = 'HIGHS'
default_low_mem = False
default_time_limit = 1e10
# 0：deepseek模型；1：boss模型
model_class = 0

# 专家输入：专家可根据环境变化更改
class ExpertInput:
    """Expert input configuration for pipeline tuning.

    This class stores expert-provided configuration parameters for tuning
    pipeline parallelism strategies.
    """
    config_file = ''
    ms_adapter_file = ''
    model_args = None

    solver_name = HIGHS_NAME
    llm_class = model_class
    time_limit = int(default_time_limit)

    is_dryrun = True
    is_double_check = False
    fit_level = 0

    low_mem = default_low_mem
    layer_ratio = 0.3
    backward_ratio = 2
    sr_ratio = 0.33
    is_support_ra_with_rs = False
    is_select_recomp = False
    is_full_recomp = True

    is_head_loss_input = True
    head_loss = 1.493
    recompute_ratio = 0.246

    output_file = pipeline_output_file
    output_file_dir = os.path.join(os.getcwd(), output_file)
    double_check_dryrun_filename = double_check_dryrun_filename

    def __init__(self, config_file, ms_adapter_file):
        """Initialize ExpertInput instance."""
        if not os.path.exists(self.output_file_dir):
            os.mkdir(self.output_file_dir)
        self.config_file = config_file
        self.ms_adapter_file = ms_adapter_file


class InitDryrun:
    """Initial dry run configuration class.

    This class handles the initial dry run setup for different model types.
    """
    layers_stage_vpp = []
    init_offset = []
    recompute_config = {}
    parts = int
    distribution = []
    x_type1 = []
    x_type2 = []
    rs_type1 = []
    rs_type2 = []
    ra_type1 = []
    ra_type2 = []
    dense_layers = None

    def __init__(self, data_parallel, model_parallel, expert_input_obj: ExpertInput):
        """Initialize InitDryrun instance."""
        if expert_input_obj.llm_class == 0:
            self.init_stages = 16
            self.init_layers = self.init_stages + 2
            self.init_micro = 32
            self.parts = self.init_micro // self.init_stages
            self.distribution = pp_util.construct_distribution(self.init_micro, self.init_stages)
            self.init_vpp = 1
            self.dense_layers = 9
            self.deepseek_build_offset()
            self.layers_stage_vpp = pp_util.get_layers_distribution(self.init_offset, self.init_layers,
                                                                    self.init_stages,
                                                                    self.init_vpp)
            self.recompute_config = self.construct_rec_config_for_deepseek()
        elif expert_input_obj.llm_class == 1:
            self.init_stages = 13
            self.init_layers = 72
            self.init_micro = self.init_stages * 3
            self.parts = self.init_micro // self.init_stages
            self.distribution = pp_util.construct_distribution(self.init_micro, self.init_stages)
            self.init_vpp = 1
            self.boss_build_offset()
            self.layers_stage_vpp = pp_util.get_layers_distribution(self.init_offset, self.init_layers,
                                                                    self.init_stages,
                                                                    self.init_vpp)
            self.x_type1 = np.zeros((self.init_vpp, self.init_stages), int)
            self.x_type2 = np.zeros((self.init_vpp, self.init_stages), int)
            for stage in range(0, 7):
                self.x_type1[0][stage] = self.layers_stage_vpp[stage][0]
            for stage in range(7, 13):
                self.x_type2[0][stage] = self.layers_stage_vpp[stage][0]
            self.recompute_config = self.construct_rec_config_for_boss()
        else:
            raise ValueError(f'can not support the model class number of {expert_input_obj.llm_class}!')

        self.rank_size = data_parallel * model_parallel * self.init_stages

        self.config_file = expert_input_obj.config_file
        self.ms_adapter_file = expert_input_obj.ms_adapter_file
        self.dryrun_output = init_dryrun_file

    def deepseek_build_offset(self):
        """Build offset configuration for DeepSeek model."""
        self.init_offset = np.zeros((self.init_vpp, self.init_stages), int).tolist()
        self.init_offset[0][7] = 1
        self.init_offset[0][14] = 1

    def boss_build_offset(self):
        """Build offset configuration for BOSS model."""
        self.init_offset = [[1, 2, -1, 1, 0, 2, -4, 2, -1, 1, 1, 2, 1]]

    def construct_rec_config_for_boss(self):
        """Construct recompute configuration for BOSS model."""
        is_select_recompute = True
        is_re_comp = True
        self.rs_type1 = np.zeros((self.init_vpp, self.init_stages), int)
        self.rs_type2 = np.zeros((self.init_vpp, self.init_stages), int)
        self.ra_type1 = np.zeros((self.init_vpp, self.init_stages), int)
        self.ra_type2 = np.zeros((self.init_vpp, self.init_stages), int)

        select_recompute_layers = [[0, 0, 0, 0, 1, 4, 0, 1, 1, 0, 2, 4, 0]]
        recompute_layers = [[6, 4, 2, 3, 0, 0, 0, 4, 2, 3, 0, 0, 6]]
        for stage in range(0, 7):
            self.rs_type1[0][stage] = select_recompute_layers[0][stage]
            self.ra_type1[0][stage] = recompute_layers[0][stage]
        for stage in range(7, 13):
            self.rs_type2[0][stage] = select_recompute_layers[0][stage]
            self.ra_type2[0][stage] = recompute_layers[0][stage]

        recompute_config = pp_util.build_recompute_config(is_select_recompute, is_re_comp, select_recompute_layers,
                                                          recompute_layers)
        return recompute_config

    def construct_rec_config_for_deepseek(self):
        """Construct recompute configuration for DeepSeek model."""
        is_select_recompute = True
        is_re_comp = True
        select_recompute_layers = np.zeros((self.init_vpp, self.init_stages), int).tolist()
        select_recompute_layers[0][3] = self.layers_stage_vpp[3][0]
        select_recompute_layers[0][4] = self.layers_stage_vpp[4][0]
        select_recompute_layers[0][10] = self.layers_stage_vpp[10][0]
        select_recompute_layers[0][11] = self.layers_stage_vpp[11][0]

        recompute_layers = np.zeros((self.init_vpp, self.init_stages), int).tolist()
        recompute_layers[0][0] = self.layers_stage_vpp[0][0]
        recompute_layers[0][1] = self.layers_stage_vpp[1][0]
        recompute_layers[0][2] = self.layers_stage_vpp[2][0]
        recompute_layers[0][7] = self.layers_stage_vpp[7][0]
        recompute_layers[0][8] = self.layers_stage_vpp[8][0]
        recompute_layers[0][9] = self.layers_stage_vpp[9][0]
        recompute_layers[0][14] = self.layers_stage_vpp[14][0]

        recompute_layers[0][15] = self.layers_stage_vpp[15][0]

        recompute_config = pp_util.build_recompute_config(is_select_recompute, is_re_comp, select_recompute_layers,
                                                          recompute_layers)
        return recompute_config

    def init_dryrun(self):
        """Execute initial dry run."""
        dry_run = dryrun.DryRun(self.config_file, self.ms_adapter_file, self.dryrun_output)
        dry_run.start_dryrun(self.recompute_config, self.init_offset, self.init_layers, self.init_vpp,
                             self.init_stages,
                             self.rank_size, self.dense_layers, self.init_micro)
        peak_mem = [dry_run.extract_memory_info(self.init_stages), dry_run.extract_memory_info_act(self.init_stages)]
        return peak_mem


class InitConfig:
    """Initial configuration class.

    This class handles the initialization of pipeline configuration from various
    configuration file formats.
    """
    pipeline_stage = int
    micro_batch_num = int
    model_parallel = int
    data_parallel = int
    expert_parallel = int
    rank_size = int
    num_layers_type1 = int
    num_layers_type2 = int
    pp_interleave_num = int(1)
    seq_length = int
    hidden_size = int
    intermediate_size = int
    vocab_size = int
    mem_lim = float
    seq_splits = int(1)
    parts = int
    mps_sol_filename = ''

    def __init__(self, expert_input_obj: ExpertInput):
        """Initialize InitConfig instance."""
        self.expert_input = expert_input_obj
        self.config_file = expert_input_obj.config_file
        self.ms_adapter_file = expert_input_obj.ms_adapter_file
        if dryrun.DryRun.config_file_type == 0:
            self.get_yaml_config()
        elif dryrun.DryRun.config_file_type == 1:
            self.get_shell_config()
        elif dryrun.DryRun.config_file_type == 2:
            self.get_toml_config()
        else:
            raise TypeError(dryrun.dryrun_config_error)
        self.rank_size = self.pipeline_stage * self.model_parallel * self.data_parallel
        self.parts = self.micro_batch_num // self.pipeline_stage
        if self.parts == 0:
            raise ValueError(f'stage = {self.pipeline_stage} is greater than micro batch = {self.micro_batch_num}!, '
                             f'please check the config file!')
        self.mps_sol_filename = (
            f'layers{self.num_layers_type1}_{self.num_layers_type2}_micro{self.micro_batch_num}_'
            f'dp{self.data_parallel}'
            f'_tp{self.model_parallel}_pp{self.pipeline_stage}_vp{self.pp_interleave_num}'
            f'_ep{self.expert_parallel}'
        )

        if self.pp_interleave_num <= 1:
            self.expert_input.low_mem = True

        self.memory = Memory(self.mem_lim)
        self.set_memory(expert_input_obj.is_dryrun)

    def set_memory(self, is_dryrun):
        """Set memory configuration."""
        memory_dir = os.path.join(self.expert_input.output_file_dir, 'memory_info')
        filename = (
            f'layers{self.num_layers_type1}_{self.num_layers_type2}_micro{self.micro_batch_num}_'
            f'dp{self.data_parallel}'
            f'_tp{self.model_parallel}_pp{self.pipeline_stage}_vp1'
            f'_ep{self.expert_parallel}.txt'
        )
        memory_file_name = os.path.join(memory_dir, filename)
        if is_dryrun:
            self.mem_calculator_by_dryrun()
            if not os.path.exists(memory_dir):
                os.mkdir(memory_dir)
            self.memory.write_memory_to_file(memory_file_name)
        elif os.path.exists(memory_file_name):
            logger.info(f'mem_lim = {self.mem_lim}')
            with open(memory_file_name, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                for line in lines:
                    key, value = line.split('=')
                    if hasattr(self.memory, key):
                        setattr(self.memory, key, float(value))
                    else:
                        logger.error(f'NameError: {key}')
            self.memory.mem_lim_stage0 = self.memory.mem_lim - self.memory.static_mem0
            self.memory.mem_lim_others = self.memory.mem_lim - self.memory.static_mem
            self.memory.mem_lim_last = self.memory.mem_lim - self.memory.lm_head_mem
            logger.info(f'Using the {filename} memory for vpp{self.pp_interleave_num}')
        else:
            logger.info(f'There is no memory file: {memory_file_name}! Using the default memory!')
        self.memory.print_mem()

    def get_yaml_config(self):
        """Parse YAML configuration file."""
        with open(self.config_file, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
            self.pipeline_stage = data['parallel_config']['pipeline_stage']
            self.micro_batch_num = data['parallel_config']['micro_batch_num']
            self.model_parallel = data['parallel_config']['model_parallel']
            self.data_parallel = data['parallel_config']['data_parallel']
            if 'expert_parallel' in data['parallel_config']:
                self.expert_parallel = data['parallel_config']['expert_parallel']
            else:
                self.expert_parallel = 1
            self.num_layers_type2 = data['model']['model_config']['num_layers']
            if 'mtp_depth' in data['model']['model_config']:
                self.num_layers_type2 += data['model']['model_config']['mtp_depth']
            if 'pp_interleave_num' in data['model']['model_config']:
                self.pp_interleave_num = data['model']['model_config']['pp_interleave_num']
            else:
                self.pp_interleave_num = 1
            self.num_layers_type1 = 0
            if 'moe_config' in data:
                if 'first_k_dense_replace' in data['moe_config']:
                    self.num_layers_type1 = data['moe_config']['first_k_dense_replace']
            if self.expert_input.llm_class == 1:
                self.num_layers_type1 = 36
            self.num_layers_type2 -= self.num_layers_type1
            if not self.expert_input.is_head_loss_input:
                self.seq_length = data['model']['model_config']['seq_length']
                self.hidden_size = data['model']['model_config']['hidden_size']
                self.intermediate_size = data['model']['model_config']['intermediate_size']
                self.vocab_size = data['model']['model_config']['vocab_size']
            self.mem_lim = int(re.search(r'(\d+)GB', data['context']['max_device_memory']).group(1)) * 1024.0

    def get_shell_config(self):
        """Parse shell configuration file."""
        configs, _ = pp_util.parse_shell(self.config_file)
        self.pipeline_stage = configs.get('PP')
        self.micro_batch_num = configs.get('GBS') // configs.get('MBS') // configs.get('DP')
        self.model_parallel = configs.get('TP')
        self.data_parallel = configs.get('DP')
        self.expert_parallel = configs.get('EP', 1)
        self.pp_interleave_num = configs.get('VPP', 1)
        self.num_layers_type1 = configs.get('FIRST_K_DENSE_REPLACE', 0)
        if self.expert_input.llm_class == 1:
            self.num_layers_type1 = 36
        self.num_layers_type2 = configs.get('NUM_LAYERS') - self.num_layers_type1
        if not self.expert_input.is_head_loss_input:
            self.seq_length = configs.get('SEQ_LEN')
            self.hidden_size = configs.get('HIDDEN_SIZE')
            self.intermediate_size = configs.get('FFN_HIDDEN_SIZE')
            if 'VOCAB_SIZE' in configs:
                self.vocab_size = configs.get('VOCAB_SIZE')
        self.mem_lim = configs.get('MAX_DEVICE_MEMORY', 58) * 1024.0

    def get_toml_config(self):
        """Parse TOML configuration file."""
        model_args = self.expert_input.model_args
        self.pipeline_stage = model_args.pp
        self.micro_batch_num = model_args.mbn
        self.model_parallel = model_args.tp

        self.data_parallel = model_args.dp
        self.expert_parallel = model_args.ep
        self.pp_interleave_num = 1

        self.num_layers_type1 = model_args.first_k_dense_replace
        self.num_layers_type2 = model_args.num_layers - self.num_layers_type1

        self.seq_length = model_args.seq_len
        self.hidden_size = model_args.hidden_size
        self.intermediate_size = model_args.ffn_hidden_size

        self.vocab_size = model_args.padded_vocab_size
        self.mem_lim = 58 * 1024.0

    def mem_calculator_by_dryrun(self):
        """Calculate memory using dry run."""
        self.set_cur_memory_info()
        self.memory.mem_lim_stage0 = self.mem_lim - self.memory.static_mem0
        self.memory.mem_lim_others = self.mem_lim - self.memory.static_mem
        self.memory.mem_lim_last = self.mem_lim - self.memory.lm_head_mem

    def set_cur_memory_info(self):
        """Set current memory information from dry run."""
        init_dryrun = InitDryrun(self.data_parallel, self.model_parallel, self.expert_input)
        is_input_mem = False
        if not is_input_mem:
            peak_mem = init_dryrun.init_dryrun()
            logger.info(f'The first initial dryrun: peak_mem={peak_mem}')
        else:
            logger.info('Using input peak memory for act_layer memory!')
            peak_mem = pp_util.get_init_input_peak_mem()
        if any(mem == 0 for mem in peak_mem[1]):
            raise ValueError(pp_util.dryrun_error)
        if self.expert_input.llm_class == 0:
            self.update_deepseek_memory(peak_mem, init_dryrun)
        elif self.expert_input.llm_class == 1:
            self.update_boss_memory(peak_mem, init_dryrun)
        else:
            raise ValueError(f'can not support the model class number of {self.expert_input.llm_class}!')

    def update_deepseek_memory(self, peak_mem, init_dryrun: InitDryrun):
        """Update memory configuration for DeepSeek model."""
        peaks = pp_util.get_peak_batch(init_dryrun.init_micro, init_dryrun.init_stages, init_dryrun.init_vpp,
                                       True, init_dryrun.init_offset, init_dryrun.init_layers)
        logger.info(f'peaks={peaks}')
        self.memory.select_mem12 = max((peak_mem[0][3] - peak_mem[0][4]) / (peaks[3] - peaks[4]), 0)
        self.memory.select_mem0 = self.memory.select_mem12
        self.memory.select_mem = max((peak_mem[0][10] - peak_mem[0][11]) / (peaks[10] - peaks[11]), 0)
        self.memory.re_comp_mem12 = max((peak_mem[0][1] - peak_mem[0][2]) / (peaks[1] - peaks[2]), 0)
        self.memory.re_comp_mem0 = self.memory.re_comp_mem12
        self.memory.re_comp_mem = max((peak_mem[0][8] - peak_mem[0][9]) / (peaks[8] - peaks[9]), 0)
        self.memory.act_mem12 = max((peak_mem[0][5] - peak_mem[0][6]) / (peaks[5] - peaks[6]), 0)
        self.memory.act_mem0 = self.memory.act_mem12
        self.memory.act_mem = (peak_mem[0][12] - peak_mem[0][13]) / (peaks[12] - peaks[13])
        # 更正re_comp_mem
        if dryrun.DryRun.config_file_type == 1:
            self.memory.re_comp_mem = self.memory.act_mem

        self.memory.layer_mem012 = (
            (peak_mem[0][7] - peak_mem[0][1])
            - self.memory.re_comp_mem12 * (peaks[7] - peaks[1])
        ) / (
            init_dryrun.layers_stage_vpp[7][0]
            - init_dryrun.layers_stage_vpp[1][0]
        )
        self.memory.layer_mem = (
            (peak_mem[0][14] - peak_mem[0][9])
            - self.memory.re_comp_mem * (peaks[14] - peaks[9])
        ) / (
            init_dryrun.layers_stage_vpp[14][0]
            - init_dryrun.layers_stage_vpp[9][0]
        )

        layers_stage = [
            sum(init_dryrun.layers_stage_vpp[stage][vpp] for vpp in range(init_dryrun.init_vpp))
            for stage in range(init_dryrun.init_stages)
        ]
        if init_dryrun.layers_stage_vpp[0][0] >= 3:
            self.memory.static_mem0 = (
                peak_mem[1][0]
                - self.memory.layer_mem012 * 3
                - self.memory.layer_mem * (layers_stage[0] - 3)
                - self.memory.re_comp_mem12 * peaks[0] * 3 / layers_stage[0]
                - self.memory.re_comp_mem * peaks[0] * (layers_stage[0] - 3) / layers_stage[0]
            )
        else:
            self.memory.static_mem0 = (
                peak_mem[1][0]
                - self.memory.layer_mem012 * init_dryrun.layers_stage_vpp[0][0]
                - self.memory.layer_mem * (layers_stage[0] - init_dryrun.layers_stage_vpp[0][0])
                - self.memory.re_comp_mem12 * peaks[0] * init_dryrun.layers_stage_vpp[0][0] / layers_stage[0]
                - self.memory.re_comp_mem * peaks[0] * (layers_stage[0] - init_dryrun.layers_stage_vpp[0][0])
                / layers_stage[0]
            )
        self.memory.static_mem = (
            peak_mem[1][-2]
            - self.memory.layer_mem * layers_stage[-2]
            - self.memory.re_comp_mem * peaks[-2]
        )
        self.memory.lm_head_mem = (
            peak_mem[1][-1]
            - self.memory.layer_mem * layers_stage[-1]
            - self.memory.re_comp_mem * peaks[-1]
        )

    def update_boss_memory(self, peak_mem, init_dryrun: InitDryrun):
        """Update memory configuration for BOSS model."""
        sort_micro = micro.SortMicro(init_dryrun.parts, init_dryrun.init_vpp, init_dryrun.init_stages,
                                     init_dryrun.distribution, True, 1)
        peaks = micro.PeakNum(sort_micro)
        init_mem = Memory(self.mem_lim)
        peaks.set_peak_act_recompute_num(init_dryrun.x_type2, init_dryrun.rs_type2, init_dryrun.ra_type2,
                                         init_dryrun.x_type1, init_dryrun.rs_type1, init_dryrun.ra_type1, init_mem)
        stage_range = [1, 6]
        coe_a1 = pp_util.build_coe_array(peaks.peak_num_act_type1, peaks.peak_num_select_recom_type1,
                                         peaks.peak_num_recompute_type1, init_dryrun.x_type1, stage_range)
        mem_b1 = [peak_mem[0][stage] for stage in range(stage_range[0], stage_range[-1])]
        mem_result, res, rank, _ = np.linalg.lstsq(coe_a1, mem_b1, rcond=None)
        mem_result = np.round(mem_result, decimals=1)
        logger.info('the type of layer1:')
        logger.info(f'the residual = {res}')
        logger.info(f'the normal of residual = {np.linalg.norm(res)}')
        logger.info(f'the rank = {rank}')
        static1 = mem_result[0]
        self.memory.act_mem12 = mem_result[1]
        self.memory.act_mem0 = mem_result[1]
        self.memory.layer_mem012 = mem_result[2]
        self.memory.re_comp_mem12 = mem_result[3]
        self.memory.re_comp_mem0 = mem_result[3]
        self.memory.select_mem12 = mem_result[4]
        self.memory.select_mem0 = mem_result[4]

        stage_range = [7, 12]
        coe_a2 = pp_util.build_coe_array(peaks.peak_num_act_type2, peaks.peak_num_select_recom_type2,
                                         peaks.peak_num_recompute_type2, init_dryrun.x_type2, stage_range)
        mem_b2 = [peak_mem[0][stage] for stage in range(stage_range[0], stage_range[-1])]
        mem_result, res, rank, _ = np.linalg.lstsq(coe_a2, mem_b2, rcond=None)
        mem_result = np.round(mem_result, decimals=1)
        logger.info('the type of layer2:')
        logger.info(f'the residual = {res}')
        logger.info(f'the normal of residual = {np.linalg.norm(res)}')
        logger.info(f'the rank = {rank}')

        static2 = mem_result[0]
        self.memory.act_mem = mem_result[1]
        self.memory.layer_mem = mem_result[2]
        self.memory.re_comp_mem = mem_result[3]
        self.memory.select_mem = mem_result[4]

        static_mem0 = (
            peak_mem[1][0]
            - self.memory.layer_mem012 * init_dryrun.x_type1[0][0]
            - self.memory.layer_mem * init_dryrun.x_type2[0][0]
            - self.memory.act_mem12 * peaks.peak_num_act_type1[0]
            - self.memory.act_mem * peaks.peak_num_act_type2[0]
            - self.memory.re_comp_mem12 * peaks.peak_num_recompute_type1[0]
            - self.memory.re_comp_mem * peaks.peak_num_recompute_type2[0]
            - self.memory.select_mem12 * peaks.peak_num_select_recom_type1[0]
            - self.memory.select_mem * peaks.peak_num_select_recom_type2[0]
        )
        logger.info(f'static1 = {static1}')
        logger.info(f'static2 = {static2}')
        static_mem = (static2 + static1) / 2
        lm_head_mem = (
            peak_mem[1][-1]
            - self.memory.layer_mem012 * init_dryrun.x_type1[0][-1]
            - self.memory.layer_mem * init_dryrun.x_type2[0][-1]
            - self.memory.act_mem12 * peaks.peak_num_act_type1[init_dryrun.init_stages - 1]
            - self.memory.act_mem * peaks.peak_num_act_type2[init_dryrun.init_stages - 1]
            - self.memory.re_comp_mem12 * peaks.peak_num_recompute_type1[init_dryrun.init_stages - 1]
            - self.memory.re_comp_mem * peaks.peak_num_recompute_type2[init_dryrun.init_stages - 1]
            - self.memory.select_mem12 * peaks.peak_num_select_recom_type1[init_dryrun.init_stages - 1]
            - self.memory.select_mem * peaks.peak_num_select_recom_type2[init_dryrun.init_stages - 1]
        )
        self.memory.static_mem0 = np.round(static_mem0, decimals=1)
        self.memory.static_mem = np.round(static_mem, decimals=1)
        self.memory.lm_head_mem = np.round(lm_head_mem, decimals=1)


if __name__ == '__main__':
    expert_input = ExpertInput(config_file='C:\\working\\768_4k.yaml',
                               ms_adapter_file='~/mindformer.py')
    expert_input.is_dryrun = False
    model_input = InitConfig(expert_input)
    print(model_input)
    
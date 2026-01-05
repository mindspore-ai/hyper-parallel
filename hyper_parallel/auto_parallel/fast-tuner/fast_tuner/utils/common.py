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
"""common use functions"""
import os
import json

from pathlib import Path
import shutil
import yaml
import toml
import numpy as np

from fast_tuner.pipeline_conductor.pp_util import configs_to_shell, parse_shell
from fast_tuner.utils.logger import logger


def cal_model_layers_num(input_args):
    if input_args.model.model_config.mtp_depth is None:
        input_args.model.model_config.mtp_depth = 0
    return input_args.model.model_config.num_layers + input_args.model.model_config.mtp_depth

GENERAL_TOML = 'DP{}_TP{}_PP{}_EP{}_CP{}_FSDP{}_pretrain.toml'
MOE_PATTERN_YAML = 'DP{}_TP{}_PP{}_EP{}_pretrain.yaml'
LLAMA_PATTERN_YAML = 'DP{}_TP{}_PP{}_pretrain.yaml'
MOE_PATTERN_SHELL = 'DP{}_TP{}_PP{}_EP{}_pretrain.sh'
LLAMA_PATTERN_SHELL = 'DP{}_TP{}_PP{}_pretrain.sh'
MODULE_PATTERN_REG_YAML = r'DP(\d+)_TP(\d+)_PP(\d+)_EP(\d+)_pretrain.yaml'
MOE_PATTERN_REG_SHELL = r'DP(\d+)_TP(\d+)_PP(\d+)_EP(\d+)_pretrain.sh'
LLAMA_PATTERN_REG_SHELL = r'DP(\d+)_TP(\d+)_PP(\d+)_pretrain.sh'

def initial_offset(pipeline_stage, num_layers):
    '''
    set offset in a memory friendly way such that
    we can estimate the minimal memory requirement
    '''

    if pipeline_stage == 1 or num_layers % pipeline_stage == 0:
        offset = 0
        return offset

    pp_interleave_num = 1
    offset = np.zeros((pp_interleave_num, pipeline_stage), dtype=int).tolist()
    remainder = num_layers % (pp_interleave_num * pipeline_stage)
    for vpp in range(pp_interleave_num):
        offset[0][0] += 1
        remainder-=1
        for stage in range(pipeline_stage):
            if remainder == 0:
                break
            offset[vpp][pipeline_stage - stage - 2] += 1
            remainder -= 1
    return offset[0]

def offset_for_dualpipe(pipeline_stage, num_layers):
    '''
    set offset in a memory friendly way such that
    we can estimate the minimal memory requirement
    '''
    pp_interleave_num = 2
    offset = np.zeros((pp_interleave_num, pipeline_stage), dtype=int).tolist()
    new_layers = num_layers - 2
    remainder = new_layers % pipeline_stage
    base = new_layers // pipeline_stage
    origin_base = new_layers // pipeline_stage

    for stage in range(pipeline_stage):
        # 获取当前stage的层数
        if remainder == 0:
            cur_layer = base
        else:
            cur_layer = base + 1
            remainder -= 1
        # 给vpp分配层
        vpp1_layer = cur_layer // pp_interleave_num
        if pipeline_stage - stage - 1 == 0:
            offset[0][pipeline_stage - stage - 1] = vpp1_layer + 2 - origin_base
        else:
            offset[0][pipeline_stage - stage - 1] = vpp1_layer - origin_base
        vpp2_layer = cur_layer - vpp1_layer
        offset[1][stage] = vpp2_layer - origin_base
    return offset

def cal_world_size(input_args):
    '''compute the worldsize from config'''
    world_size = input_args.dp * input_args.tp * input_args.pp * input_args.cp
    return world_size

def generate_dryrun_yaml(destination_file, config, para):
    """
    :param para: 用户输入参数
    :param destination_file: 修改并行参数的配置yaml文件
    :param config: [dp, tp, pp, ep]
    :return:
    """
    # 复制YAML文件
    shutil.copy2(para.YAML_PATH, destination_file)

    # 读取复制后的YAML文件, 修改config_para字段的值
    with open(destination_file, 'r', encoding='utf-8') as file:
        yaml_data = yaml.safe_load(file)
        yaml_data['parallel_config']['data_parallel'] = config[0]
        yaml_data['parallel_config']['model_parallel'] = config[1]
        yaml_data['parallel_config']['pipeline_stage'] = config[2]
        yaml_data['parallel_config']['expert_parallel'] = config[3]
        yaml_data['parallel_config']['micro_batch_num'] = para.GBS // config[0]
        yaml_data['model']['model_config']['offset'] = config[4]
        if yaml_data['parallel'].get('dataset_strategy') is not None:
            strategy_size = len(yaml_data['parallel']['dataset_strategy'])
            yaml_data['parallel']['dataset_strategy'] = [[config[0], 1] for _ in range(strategy_size)]
        # 重计算设置为true
        yaml_data['recompute_config']['recompute'] = True

        # todo: 适配tnd
        yaml_data['train_dataset']['data_loader']['dataset_dir'] = para.DATASET


    # 将修改后的数据写回YAML文件
    with open(destination_file, 'w', encoding='utf-8') as file:
        yaml.dump(yaml_data, file, default_flow_style=False, allow_unicode=True)

    logger.info(f"The dryrun YAML file copied and modified, new file is: {destination_file}")

def generate_dryrun_shell(destination_file, config, para):
    """

        :param para: 用户输入参数
        :param destination_file: 修改并行参数的配置shell文件
        :param config: [dp, tp, pp, ep, offset] or [dp, tp, pp]
        :return:
    """
    configs, unparses = parse_shell(para.SHELL_PATH)
    configs['DP'] = config[0]
    configs['TP'] = config[1]
    configs['PP'] = config[2]
    configs_to_shell(destination_file, configs, unparses)
    logger.info(f'The dryrun SHELL file copied and modified, new file is {destination_file}')

def generate_profile_yaml(destination_file, config, para):
    """
    :param para: 用户输入参数

    :param destination_file:
    :param config: [dp, tp, pp, ep, offset, num_layers]
    :return:
    """
    # 复制YAML文件
    shutil.copy2(para.YAML_PATH, destination_file)

    # 读取复制后的YAML文件, 修改config_para字段的值
    with open(destination_file, 'r', encoding='utf-8') as file:
        yaml_data = yaml.safe_load(file)
        yaml_data['parallel_config']['data_parallel'] = config[0]
        yaml_data['parallel_config']['model_parallel'] = config[1]
        yaml_data['parallel_config']['pipeline_stage'] = config[2]
        yaml_data['parallel_config']['expert_parallel'] = config[3]
        yaml_data['parallel_config']['micro_batch_num'] = para.GBS // config[0]
        yaml_data['model']['model_config']['offset'] = config[4]
        yaml_data['recompute_config']['recompute'] = config[5]
        yaml_data['model']['model_config']['num_layers'] = config[6]

        yaml_data['profile'] = True
        yaml_data['profile_output'] = os.path.join(Path(destination_file).parent, "output")
        yaml_data['init_start_profile'] = True
        yaml_data['profile_communication'] = True
        yaml_data['profile_memory'] = True
        yaml_data['op_time'] = True
        yaml_data['profile_level'] = 1
        yaml_data['profile_start_step'] = 4
        yaml_data['profile_stop_step'] = 6

    # 将修改后的数据写回YAML文件
    with open(destination_file, 'w', encoding='utf-8') as file:
        yaml.dump(yaml_data, file, default_flow_style=False, allow_unicode=True)

    logger.info(f"The profile YAML file copied and modified, config_para is: {config}")

def generate_profile_shell(destination_file, config, para):
    """
    :param para: 用户输入参数

    :param destination_file:
    :param config: [dp, tp, pp, ep, offset, num_layers]
    :return:
    """
    configs, unparses = parse_shell(para.SHELL_PATH)
    configs['DP'] = config[0]
    configs['TP'] = config[1]
    configs['PP'] = config[2]
    if '_EP' in destination_file:
        configs['EP'] = config[3]
        configs['FIRST_K_DENSE_RAPLACE'] = 3
    else:
        configs['EP'] = 0  # 或者这里不需要写入EP
        # TODO 应该需要读NPUS_PER_NODE，当前硬编码为8
    profile_need_rank_num = configs['DP'] * configs['TP'] * configs['PP']
    if profile_need_rank_num < para.RANK_NUM:
        configs['NNODES'] = profile_need_rank_num // 8
        configs['NPUS_PER_NODE'] = 8
        if configs['NNODES'] == 0:
            configs['NNODES'] = 1
            configs['NPUS_PER_NODE'] = profile_need_rank_num
    else:
        configs['NNODES'] = para.RANK_NUM // 8
        configs['NPUS_PER_NODE'] = 8
    configs['NUM_LAYERS'] = config[-1]
    configs['MINDSPEED_PATH'] = para.MINDSPEED_PATH
    # 生成要分析的进程编号列表
    step = profile_need_rank_num // config[2]
    profile_ranks = ' '.join(map(str, range(0, profile_need_rank_num, step)))
    if '_EP' in destination_file:
        output_dir = os.path.join(para.OUTPUT_PATH, "profile_result",
                                  f"DP{config[0]}_TP{config[1]}_PP{config[2]}_EP{config[3]}_profile")
    else:
        output_dir = os.path.join(para.OUTPUT_PATH, "profile_result",
                                  f"DP{config[0]}_TP{config[1]}_PP{config[2]}_profile")
    # 使用f-string构建参数字符串，提高可读性
    profile_args = (
        "--profile "
        "--profile-step-start 5 "
        "--profile-step-end 7 "
        "--profile-level level1 "
        "--profile-with-stack "
        "--profile-with-cpu "
        f"--profile-ranks {profile_ranks} "  # 确保参数之间有空格
        f"--profile-save-path {output_dir}"
    )
    # todo: 需要适配原有的args內容
    if configs['EP'] == 0:
        mem_args = (
            "--use-distributed-optimizer "
            "--recompute-method block "
            "--recompute-granularity full "
            "--recompute-num-layers 1 "
            "--num-layer-list 4,3 "
        )
    else:
        mem_args = (
            "--use-distributed-optimizer "
            "--recompute-method block "
            "--recompute-granularity full "
            "--recompute-num-layers 1 "
            "--num-layer-list 4,3 "
            "--first-k-dense-replace 3 "
        )

    # 存入配置字典
    configs['PROFILE_ARGS'] = profile_args
    configs['MEM_ARGS'] = mem_args
    configs_to_shell(destination_file, configs, unparses)
    insert_profile_args_final(destination_file)
    logger.info(f'The profile SHELL file copied and modified, new file is {destination_file}')
    return output_dir

def generate_profile_toml(destination_file, config, para):
    '''generate toml file for profile'''
    def read_toml(file_path):
        """读取 TOML 文件并返回字典"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return toml.load(f)

    def write_toml(data, file_path):
        """将字典写入 TOML 文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            toml.dump(data, f)

    dp, tp, pp, ep, cp, op = config[:6]
    template_data = read_toml(para.TOML_PATH)
    # [profiling]
    template_data['profiling']['enable_profiling'] = True
    template_data['profiling']['profile_freq'] = 20
    output_trace_dir = os.path.join(os.path.abspath(para.OUTPUT_PATH), "profile_result",
                                  f"DP{dp}_TP{tp}_PP{pp}_EP{ep}_CP{cp}_OP{op}_trace")
    template_data['profiling']['save_traces_folder'] = output_trace_dir
    # [model]
    template_data['model']['flavor'] = 'tune'
    # [training]
    template_data['training']['steps'] = 20
    # [parallelism]
    template_data['parallelism']['data_parallel_replicate_degree'] = dp // op
    template_data['parallelism']['data_parallel_shard_degree'] = op
    template_data['parallelism']['tensor_parallel_degree'] = tp
    template_data['parallelism']['context_parallel_degree'] = cp
    template_data['parallelism']['expert_parallel_degree'] = ep
    template_data['parallelism']['pipeline_parallel_degree'] = pp
    write_toml(template_data, destination_file)
    return output_trace_dir



def insert_profile_args_final(shell_file_path, profile_args="$PROFILE_ARGS \\"):
    '''modify shell files for profile'''
    with open(shell_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 标记是否在 torchrun 命令块内
    in_torchrun_block = False
    modified_lines = []

    for line in lines:
        stripped = line.strip()

        # 检测命令块开始
        if "$MEM_ARGS" in stripped:
            in_torchrun_block = True
            modified_lines.append(line)
            modified_lines.append(f"    {profile_args}\n")
            continue

        # 检测命令块结束（最后一个参数行，以 \ 结尾）
        if in_torchrun_block and stripped.endswith("\\"):
            modified_lines.append(line)
            continue

        # 命令块结束后的行
        if in_torchrun_block:
            in_torchrun_block = False

        # 普通行
        modified_lines.append(line)

    # 写入修改后的内容
    with open(shell_file_path, 'w', encoding='utf-8') as f:
        f.writelines(modified_lines)

    logger.info(f"insert PROFILE_ARGS to {shell_file_path}")


# 定义一个映射，将yaml_task和对应的生成函数关联起来
TASK_FUNCTION_MAP = {
    "dryrun_yaml": generate_dryrun_yaml,
    "dryrun_shell": generate_dryrun_shell,
    "profile_yaml": generate_profile_yaml,
    "profile_shell": generate_profile_shell,
    "profile_toml": generate_profile_toml,
}

def generate_files(candidate_configs, des_file_directory, file_task, para, input_args):
    '''generate config files w.r.t your training library'''
    # 如果目录不存在，则创建它
    if not os.path.exists(des_file_directory):
        os.makedirs(des_file_directory)

    # 检查file_task是否有效
    if file_task not in TASK_FUNCTION_MAP:
        logger.error(f"Invalid file_task value: {file_task}. Please use one of {list(TASK_FUNCTION_MAP.keys())}.")
        return None

    generate_function = TASK_FUNCTION_MAP[file_task]
    if para.SHELL_PATH:
        pattern = LLAMA_PATTERN_SHELL if input_args.expert_num is None else MOE_PATTERN_SHELL
    elif para.YAML_PATH:
        pattern = LLAMA_PATTERN_YAML if input_args.expert_num is None else MOE_PATTERN_YAML
    else:
        pattern = GENERAL_TOML

    file_path_list = []
    output_dir_list = []
    for config in candidate_configs:
        # 生成输出文件路径,包括文件名
        destination_file = (des_file_directory + pattern).format(*config)
        file_path_list.append(destination_file)
        # 只有file_task=profile_shell时才会返回output_dir
        output_dir = generate_function(destination_file, config, para)
        output_dir_list.append(output_dir)
    return file_path_list, output_dir_list


def is_dualpipe_open(input_args):
    if input_args.mf_args is None:
        return False
    parallel_cfg = input_args.mf_args.parallel
    use_zero_bubble_v = (
            hasattr(parallel_cfg, 'pipeline_config') and
            hasattr(parallel_cfg.pipeline_config, 'pipeline_scheduler') and
            parallel_cfg.pipeline_config.pipeline_scheduler == 'zero_bubble_v'
    )
    return use_zero_bubble_v

def check_dryrun_parallel_number(parallel_num):
    if parallel_num > 16:
        raise Exception(f"The parallel number {parallel_num} is too large.")

def parse_args_from_json(args):
    with open(args.config, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for key, value in data.items():
        if key not in vars(args):
            logger.error(f"The json contains invalid parameters: {key}")
        else:
            setattr(args, key, value)

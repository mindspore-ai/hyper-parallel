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
"""dryrun operation"""
import os
import subprocess
import re
import json
from multiprocessing import Pool
from fast_tuner.utils.common import cal_world_size, is_dualpipe_open, \
    LLAMA_PATTERN_REG_SHELL, MOE_PATTERN_REG_SHELL, MODULE_PATTERN_REG_YAML
from fast_tuner.utils.logger import logger


def env_update(rank_size, env_variable_json):
    # 设置环境变量
    with open(env_variable_json, 'r', encoding='utf-8') as f:
        env_vars = json.load(f)
    env_vars['RANK_SIZE'] = str(rank_size)
    os.environ.update(env_vars)

def create_target_dir(file, target_directory):
    output_dir = os.path.splitext(file)[0]
    target_dir = os.path.join(target_directory, f"dryrun_{output_dir}")
    return target_dir

def execute_command(para, config_file_path, target_dir, rank_id, tp):
    '''execute dryrun command'''
    if para.YAML_PATH:
        command = ['python', os.path.join(para.MINDFORMERS_DIR, 'run_mindformer.py'), '--config', config_file_path,
               '--register_path', para.REGISTER_PATH]
    else:
        command = ['bash', config_file_path]
    try:
        # 构建日志文件路径
        os.environ['RANK_ID'] = str(rank_id)
        os.environ['MODEL_PARALLEL'] = str(tp)
        os.makedirs(target_dir, exist_ok=True)
        log_file_path = os.path.join(target_dir, 'dryrun_new.log')
        with open(log_file_path, 'w', encoding='utf-8') as log_file:
            logger.info(f"Command: {' '.join(command)} rank {rank_id} start run")
            subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, check=False)
    except Exception as e:
        logger.error(f"The command execution failed.: {e}")

def calculate_rank_id(input_args, match, layer_num, rank_size):
    if is_dualpipe_open(input_args):
        return rank_size - 1
    pp = int(match.group(3))
    x = pp - layer_num % pp
    rank_id = x * rank_size // pp
    if rank_id == rank_size:
        rank_id -= 1
    return rank_id

def read_dryrun_info(root_dir):
    '''read the peak mem from dryrun log'''
    logger.info(f"Reading dryrun info from {root_dir}")
    result_list = []
    # 预编译正则表达式，提高匹配性能
    pattern = re.compile(r'DP(\d+)_TP(\d+)_PP(\d+)_EP(\d+)')

    for entry in os.scandir(root_dir):
        if not entry.is_dir():
            continue

        log_file_path = os.path.join(entry.path, 'dryrun_new.log')
        if not os.path.isfile(log_file_path):
            continue

        with open(log_file_path, 'r', encoding='utf-8') as log_file:
            for line in log_file:
                if "Used peak memory usage (without fragments):" in line:
                    peak_value = line.split(': ')[-1].strip()
                    peak_value = int(peak_value.rstrip('M'))
                    break
            else:
                # 如果没有找到 peak 值，跳过当前目录
                continue

        match = pattern.search(entry.name)
        if match:
            dp_num, tp_num, pp_num, ep_num = map(int, match.groups())
            result_list.append([dp_num, tp_num, pp_num, ep_num, peak_value])

    logger.info(f"read dryrun info done, result size {len(result_list)} format [dp, tp, pp, ep, peak_value]")
    logger.info('\n'.join(str(item) for item in result_list))
    return result_list

def get_file_pattern(root_dir, input_args, para):
    logger.info(f"Reading file pattern from {root_dir}")
    if para.SHELL_PATH:
        if input_args.expert_num is None:
            pattern = LLAMA_PATTERN_REG_SHELL
        else:
            pattern = MOE_PATTERN_REG_SHELL
    else:
        pattern = MODULE_PATTERN_REG_YAML
    return pattern

def launch_dryrun(input_args, dryrun_file_dir, dryrun_data_dir, para):
    '''dryrun launcher'''
    logger.info('start dryrun')
    rank_size = cal_world_size(input_args)
    env_update(rank_size, para.ENV_JSON)
    tasks = []
    layer_num = input_args.num_layers

    # 遍历指定目录下的所有文件
    for root, _, files in os.walk(dryrun_file_dir):
        for file in files:
            pattern = get_file_pattern(root, input_args, para)
            match = re.match(pattern, file)
            if not match:
                continue
            rank_id = calculate_rank_id(input_args, match, layer_num, rank_size)
            tp = int(match.group(2))
            target_dir = create_target_dir(file, dryrun_data_dir)
            yaml_file_path = os.path.join(root, file)
            tasks.append((para, yaml_file_path, target_dir, rank_id, tp))

    with Pool(processes=para.DRYRUN_LIM) as pool:
        pool.starmap(execute_command, tasks)
    logger.info("all dryrun done")

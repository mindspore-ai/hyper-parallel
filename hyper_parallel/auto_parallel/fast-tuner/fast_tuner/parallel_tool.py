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
"""entry file"""

import argparse
import time
import sys
import types

from fast_tuner.ndsearch.para_for_nd_search import ParaForNd
from fast_tuner.ndsearch.memory_model import filter_oom
from fast_tuner.ndsearch.build_initial_spaces import build_initial_spaces
from fast_tuner.ndsearch.expert_filter_configs import expert_filter_configs

from fast_tuner.utils.common import check_dryrun_parallel_number
from fast_tuner.utils.input_param import InputParam
from fast_tuner.utils.logger import logger
from fast_tuner.utils.common import parse_args_from_json
from fast_tuner.utils.ppc_input import ParallelInput
from fast_tuner.utils.profiling.profile_launch import ProfileLaunch
from fast_tuner.utils.profiling.profile_parser import ProfileParser, ProfileMemParser
from fast_tuner.utils.profiling.profile_prepare import profile_prepare
# 这里不应该有的
from fast_tuner.pipeline_conductor import pp_util
from fast_tuner.pipeline_conductor.pipeline_parallel import pipeline_proc


__all__ = ['taylor_search_tool']

def taylor_search_tool(para):
    """
    A function for find out optimal ND parallel configuration.

    Args:
        param para: 用户输入自定义的参数

    Returns:
        parallel config fill back to yaml file: [dp, cp, tp, ...] and candidate csv file.
    """
    start_time = time.time()
    para.print_params()

    input_args = ParaForNd(para)

    initial_configs = build_initial_spaces(input_args, para)
    logger.info(f"Initial Search space size: {len(initial_configs)}")
    expert_prune_search_space = expert_filter_configs(initial_configs, input_args, para.GBS)
    if len(expert_prune_search_space) == 0:
        logger.info("expert_prune_search_space is empty. Please check your expert rules.")
        sys.exit()
    logger.info(f"Expert Prune Search space size: {len(expert_prune_search_space)}")

    mem_prune_space = filter_oom(expert_prune_search_space, input_args, para)
    logger.info('%s', '\n'.join(str(item) for item in mem_prune_space))

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"program before profiling cost time: {elapsed_time} s")

    # profile_configs: list[dp, tp, pp, 0, 0, layers_num, toml/yaml/shell path, profile_result_dir]
    # profile_file_dir: profile shell/toml file dir
    profile_configs, profile_file_dir = profile_prepare(mem_prune_space, para, input_args)
    if para.ALG_PHASE == 1:
        logger.info(f"ALG_PHASE: {para.ALG_PHASE}, no need to profile and solve pipeline")
        return
    # 自动执行profile
    profile_launch = ProfileLaunch(profile_configs, para)
    profile_launch.profile_launch(profile_file_dir)
    # 自动profile解析
    profile_parser = ProfileParser(input_args, para)
    profile_parser.parse_batch_profile_result(profile_configs)

    # 内存解析
    profile_mem_parser = ProfileMemParser(input_args, para)
    profile_mem_parser.mem_parser(profile_configs)

    # 流水线求解 todo: 想办法把candidate_configs传进去，不用csv读取
    pipeline_input = ParallelInput(para, profile_file_dir)
    pipeline_proc(pipeline_input)

def main():
    logger.info('start to run parallel tool')
    parser = argparse.ArgumentParser(description='Run taylor_search_tool with user input parameters')
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['YAML_PATH']}", default='',
                        help='Path to the YAML file')
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['SHELL_PATH']}", default='',
                        help='Path to the SHELL type config file')
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['TOML_PATH']}", default='',
                        help='Path to the TOML file')
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['MINDFORMERS_DIR']}",
                        default='', help='Directory of mindformers')
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['MINDSPEED_PATH']}", default='',
                        help='Path to the MindSpeed file')
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['TORCHTITAN_PATH']}", default='',
                        help='Path to the Torchtitan file')
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['DRYRUN_DATA_DIR']}", default='',
                        help='Directory of dryrun data')
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['PROFILE_DATA_DIR']}",
                        default='', help='Directory of profile data')
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['NNODES']}", type=int,
                        default=1, help='The number of nodes')
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['NPUS_PER_NODE']}", type=int,
                        default=8, help='The npus per node')
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['DRYRUN_LIM']}", type=int,
                        default=2, help='The maximum number of dryrun at once')
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['RANK_NUM']}", type=int,
                        default=64, help='Number of available device number')
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['SOLVER_NAME']}", type=str,
                        default='HIGHS', help='Name of the solver')
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['DATASET']}", type=str,
                        default='', help='Directory of dataset')
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['MAX_EXPERT_PARALLEL']}", type=int,
                        default=64, help='Max number of expert parallel')
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['OUTPUT_PATH']}", type=str,
                        default='./output/', help='Directory of output info')
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['ENV_JSON']}", type=str,
                        default='', help='Environment variable config json file')
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['GBS']}", type=int,
                        default=1024, help='Global batch size')
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['CONFIG']}", type=str,
                        help='Path to the JSON config file')
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['SELECT_RECOMPUTE']}", type=bool, default=True,
                        help='Whether search select recompute')
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['ALG_PHASE']}", type=int, default=0,
                        help='Phase of parallel strategy search algorithm')
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['PARSER_RESULT']}", type=str, default=None,
                        help='Profiling parser result file')
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['DRYRUN']}", type=pp_util.str2bool, default=False,
                        help='Is auto dryrun')
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['CHECK']}", type=pp_util.str2bool, default=False,
                        help="Is double check")
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['STRATEGY']}", type=pp_util.str2dict,
                        default={'DP':True, 'TP':True, 'EP':True, 'FSDP':True, 'PP':False, 'CP':False},
                        help="Which parallel strategies are enabled")

    args = parser.parse_args()
    # for test
    args.config = './config/setup_config/args_for_parallel_tool_titan.json'
    # for test
    if args.config:
        parse_args_from_json(args)
    args.strategy = types.SimpleNamespace(**args.strategy)
    check_dryrun_parallel_number(args.dryrun_lim)

    para = InputParam(args)
    taylor_search_tool(para)

if __name__ == '__main__':
    main()

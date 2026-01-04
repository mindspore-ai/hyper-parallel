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
"""entry file for pipeline"""

import argparse

from fast_tuner.utils.common import check_dryrun_parallel_number
from fast_tuner.utils.ppc_input import ParallelInput
from fast_tuner.utils.logger import logger
from fast_tuner.pipeline_conductor.pipeline_parallel import pipeline_proc
from fast_tuner.pipeline_conductor import pp_util


if __name__ == '__main__':
    logger.info('start to run pipeline tool')
    # 用户输入profiling结果，候选配置等信息，流水线工具给出配置cost排序
    parser = argparse.ArgumentParser(description='Run taylor pipeline_search_tool with user input parameters')
    parser.add_argument('--files_dir', type=str, default='./output/dryrun_yaml/',
                        help='Path to the YAML or SHELL file directory')
    parser.add_argument('--yaml_path', type=str, default=None,
                        help='Path of training config (.yaml)')
    parser.add_argument('--shell_path', type=str, default=None,
                        help="Path of training config (.sh)")
    parser.add_argument('--mindformers_dir', type=str, default=None,
                        help='Directory of mindformers')
    parser.add_argument('--mindspeed_path', type=str, default=None,
                        help="Absolute path of posttrain_gpt (.py)")
    parser.add_argument('--profile_data_dir', type=str, default='./profile_data/',
                        help='Directory of profile data')
    parser.add_argument('--solver_name', type=str, default='HIGHS',
                        help='Name of the solver')
    parser.add_argument('--parser_result', type=str, default=None,
                        help='Profiling parser result file')
    parser.add_argument('--nd_path', type=str, default='./config/nd_result.csv',
                        help='Path to nd result file')
    parser.add_argument('--env_json', type=str, required=True,
                        default='./config/boss_env_config.json', help="Path of environment config (.json)")
    parser.add_argument('--register_path', type=str,default='research/jiutian',
                        help="Path of register")
    parser.add_argument('--parallel_num',  type=pp_util.str2int, default=16,
                        help="The number of dryrun at once")
    parser.add_argument('--dryrun', type=pp_util.str2bool, default=True,
                        help="Is auto dryrun")
    parser.add_argument('--check', type=pp_util.str2bool, default=True,
                        help="Is double check")
    parser.add_argument('--output_path', type=str,
                        default='./output/',
                        help='Directory of output info')

    args = parser.parse_args()
    check_dryrun_parallel_number(args.parallel_num)
    pipeline_input = ParallelInput(args)
    pipeline_proc(pipeline_input)

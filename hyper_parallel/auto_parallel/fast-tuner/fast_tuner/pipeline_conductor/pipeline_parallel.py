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
"""pp solver entry file"""
import sys
import os.path
import argparse
from fast_tuner.utils.logger import logger
from fast_tuner.pipeline_conductor import math_model, pp_util
from fast_tuner.pipeline_conductor.start_service import InitConfig, ExpertInput, HIGHS_NAME, pipeline_output_file
from fast_tuner.pipeline_conductor import solution
from fast_tuner.pipeline_conductor import fitting
from fast_tuner.utils.ppc_input import ParallelInput
from fast_tuner.pipeline_conductor.result_csv import ResultCsv
from fast_tuner.pipeline_conductor.dryrun import DryRun, dryrun_config_error

from fast_tuner.utils.common import check_dryrun_parallel_number

mps_dir_name = 'model_mps'
sol_dir_name = 'sol_output'


def pp_calculator(exp_input: ExpertInput) -> solution.Solution:
    """
    Adjust memory limits and re-solves problems
    if double-check is requested and memory constraints are exceeded
    """
    init_config = InitConfig(exp_input)
    cur_solution = solve_problem(init_config)
    if cur_solution.solution_status == 'None':
        return cur_solution
    if exp_input.is_double_check:
        fit_mem = fitting.FitMem(cur_solution)
        over_mem, is_over_mem = fit_mem.is_over_mem()
        if not is_over_mem:
            return cur_solution
        if exp_input.fit_level == 0:
            i = 0
            while i < 5 and is_over_mem:
                fit_mem.reduce_mem_lim_for_fitting(over_mem, i)
                logger.info(f'Correct the memory at the {i + 1} time!')
                init_config.memory.print_mem()
                cur_solution = solve_problem(init_config)
                if cur_solution.solution_status == 'None':
                    logger.info('dryrun_check result is over memory limit')
                    return cur_solution
                fit_mem = fitting.FitMem(cur_solution)
                over_mem, is_over_mem = fit_mem.is_over_mem()
                i += 1
        else:
            fit_mem.linear_fit()
            cur_solution = solve_problem(init_config)
    return cur_solution


def pipeline_proc(pipeline_input: ParallelInput):
    """
    Processes multiple candidate configurations in parallel
    and writes results to CSV file for each candidate
    """
    if len(pipeline_input.candidate_configs) == 0:
        raise ValueError('There is no candidate configs!')
    is_low_mem = pipeline_input.is_lowmem
    solver_name = pipeline_input.solver_name
    adapter_file = pipeline_input.ms_adapter_file
    DryRun.env_config_json = pipeline_input.env_config_json
    DryRun.dryrun_lim = pipeline_input.dryrun_lim
    ExpertInput.is_dryrun = pipeline_input.dryrun
    ExpertInput.is_double_check = pipeline_input.check
    result_csv = ResultCsv(pipeline_input.output_path, pipeline_output_file)
    num_all = len(pipeline_input.candidate_configs)
    num_cur = 0
    for candidate in pipeline_input.candidate_configs:
        result_csv.config_to_csv(candidate, is_low_mem, solver_name)
    for candidate in pipeline_input.candidate_configs:
        candidate_input = ExpertInput(candidate.config_path, adapter_file)
        candidate_input.model_args = candidate.model_args
        candidate_input.low_mem = is_low_mem
        candidate_input.solver_name = solver_name
        candidate_input.layer_ratio = candidate.profiling_info.dmratio
        candidate_input.backward_ratio = candidate.profiling_info.bfratio
        candidate_input.head_loss = candidate.profiling_info.hratio
        candidate_input.recompute_ratio = candidate.profiling_info.re_grow_ration
        num_cur += 1
        logger.info(f'--------------------- Testing {num_cur}/{num_all}:{candidate.config_path} ---------------------')
        try:
            cur_solution = pp_calculator(candidate_input)
            result_csv.result_to_csv(cur_solution)
        except Exception as e:
            logger.error(f'{candidate.config_path} error: {e}. Continue to next one')


def solve_problem(init_config: InitConfig):
    """
    Builds and solves an optimization model
    for the given configurations parameters
    """
    origin_model = math_model.Model(init_config)
    origin_model.define_variables()
    origin_model.define_constraint()
    origin_model.define_obj()
    cur_solution = solution.Solution(init_config)

    # output mps
    mps_dir = os.path.join(init_config.expert_input.output_file_dir, mps_dir_name)
    if not os.path.exists(mps_dir):
        os.mkdir(mps_dir)
    mps_file = os.path.join(mps_dir, init_config.mps_sol_filename + '.mps')
    origin_model.output_model(mps_file)

    # solve
    sol_dir = os.path.join(init_config.expert_input.output_file_dir, sol_dir_name)
    if not os.path.exists(sol_dir):
        os.mkdir(sol_dir)
    sol_file = os.path.join(sol_dir, init_config.mps_sol_filename + '.sol')
    if not os.path.exists(mps_file):
        logger.error('build model error!')
    if init_config.expert_input.solver_name == HIGHS_NAME:
        is_origin_solver = False
        pp_util.highs_solve_mps(mps_file, sol_file, origin_model, init_config.expert_input.time_limit)
    elif init_config.expert_input.solver_name == 'QIUQI':
        is_origin_solver = False
        solver_file = '/home/zhugelu/MIXSolver/bin/MIXSolver'  # 更改为本地的求解器地址
        pp_util.qiuqi_solver_mps(solver_file, mps_file, sol_file, origin_model)
    else:
        is_origin_solver = True
        origin_model.solve()

    cur_solution.set_solution(origin_model, is_origin_solver, sol_file)
    if cur_solution.solution_status != 'None':
        cur_solution.solution_print()
    return cur_solution


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='TAYLOR AutoBalancing', description=(
            'Balance layers onto pipeline stages, '
            + 'considering recomputation and interleaving'),
                                     epilog='')
    # 大模型类别
    parser.add_argument('-llm', '--llm_class', type=int, default=0,
                        help="0-deepseek,1-boss")
    # Training Yaml configuration
    parser.add_argument('-yaml', '--train_yaml', type=str, default=None,
                        help="Path of training config (.yaml)")
    parser.add_argument('-shell', '--train_shell', type=str, default=None,
                        help="Path of training config (.sh)")
    # mindformers location
    parser.add_argument('-mindformers', '--mindformers_loc', type=str, default=None,
                        help="Absolute path of run_mindformers (.py)")
    parser.add_argument('-mindspeed', '--mindspeed_loc', type=str, default=None,
                        help="Absolute path of posttrain_gpt (.py)")
    # solver name
    parser.add_argument('-solver', '--solver_name', type=str, default=HIGHS_NAME,
                        help="The solver name")
    # layer ratio
    parser.add_argument('-layer_ratio', '---layer_ratio', type=float, default=0.33,
                        help="Time ratio of calculating of dense to moe")
    # backward ratio
    parser.add_argument('-b_ratio', '---backward_ratio', type=float, default=2.0,
                        help="Time ratio of calculating of backward to forward")
    # head_loss
    parser.add_argument('-head_loss', '---head_loss', type=float, default=1.5,
                        help="Time of the last layer is added")
    # recompute_ratio
    parser.add_argument('-ra_ratio', '---recompute_ratio', type=float, default=1,
                        help="Time of the last layer is added")
    # Search time
    parser.add_argument('-t', '--time_limit', type=int, default=sys.maxsize,
                        help="Limitation on searching time")
    # 是否自动Dryrun
    parser.add_argument('-dryrun', '--dryrun', type=pp_util.str2bool, default=True,
                        help="Is auto dryrun")
    # 是否自动check
    parser.add_argument('-check', '--check', type=pp_util.str2bool, default=True,
                        help="IS double check")
    parser.add_argument('-is_write', '--is_write', type=pp_util.str2bool, default=True,
                        help="IS write solution to config file")
    # fit level，0：超内存时直接减少内存上限求解；1：超内存时线性回归拟合内存信息求解
    parser.add_argument('-fit', '--fit_level', type=int, default=0,
                        help="Fit memory when the result is over the limit: 0-reduce the memory limit;"
                             " 1 or >1-fit the memory info")
    # 是否提取solution信息
    parser.add_argument('-extract', '--extract', type=pp_util.str2bool, default=False,
                        help="Extract solution file separately")
    parser.add_argument('-solution', '--solution', default=None, help="The solution file")
    # env_config_json
    parser.add_argument('-env', '--env_config_json', type=str, required=True,
                        default='./config/boss_env_config.json', help="Path of environment config (.json)")
    parser.add_argument('-register', '--register_path', type=str, default='research/jiutian',
                        help="Path of register")
    parser.add_argument('-dryrun_lim', '--dryrun_lim',  type=pp_util.str2int, default=16,
                        help="The number of dryrun at once")

    args = parser.parse_args()
    check_dryrun_parallel_number(args.dryrun_lim)
    if args.train_yaml and args.mindformers_loc:
        config_file = args.train_yaml
        ms_adapter_file = args.mindformers_loc
        DryRun.config_file_type = 0
        ExpertInput.is_full_recomp = True
    elif args.train_shell and args.mindspeed_loc:
        config_file = args.train_shell
        ms_adapter_file = args.mindspeed_loc
        DryRun.config_file_type = 1
        ExpertInput.is_full_recomp = False
    else:
        raise TypeError(dryrun_config_error)

    if args.extract:
        solution.extract_solution_file(args.train_yaml, args.solution)
        sys.exit(0)

    expert_input = ExpertInput(config_file=config_file, ms_adapter_file=ms_adapter_file)
    expert_input.solver_name = args.solver_name
    expert_input.llm_class = int(args.llm_class)
    expert_input.time_limit = int(args.time_limit)
    if args.time_limit < sys.maxsize:
        logger.warning('You have configured the time limit parameter! The solution may be not optimal!')
    expert_input.is_dryrun = args.dryrun
    expert_input.is_double_check = args.check
    expert_input.fit_level = args.fit_level
    expert_input.layer_ratio = args.layer_ratio
    expert_input.backward_ratio = args.backward_ratio
    expert_input.head_loss = args.head_loss
    expert_input.recompute_ratio = args.recompute_ratio

    DryRun.env_config_json = args.env_config_json
    DryRun.register_path = args.register_path
    DryRun.dryrun_lim = args.dryrun_lim
    DryRun.is_write_to_file = args.is_write

    pp_calculator(expert_input)

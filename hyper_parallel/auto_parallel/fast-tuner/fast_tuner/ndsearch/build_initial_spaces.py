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
"""build initial space for nd"""

import itertools
import math

def cal_factor(num):
    factors = []
    for i in range(1, num + 1):
        if num % i == 0:
            factors.append(i)
    return factors

def find_integers_less_than_ceil(num_layers, pp):
    if pp == 1:
        # 不开pp时, vpp为1
        return [1]
    ceil_value = math.ceil(num_layers / pp)
    result = list(range(1, ceil_value))
    return result

def find_ep(expert_num):
    if expert_num is None:
        return [1]
    ep_list = []
    for i in range(1, expert_num + 1):
        if i == 1 or i % 2 == 0:
            ep_list.append(i)
    return ep_list

def build_initial_spaces(input_args, para):
    """
    generate initial space for nd
    """
    # part1. 求[dp, tp, cp, pp]的取值范围
    # 找到 world_size 的所有因子
    factors = cal_factor(input_args.world_size)

    # 找出所有可能的四个正整数乘积等于world_size
    part1_combinations = []
    for combination in itertools.combinations_with_replacement(factors, 4):
        product = 1
        for num in combination:
            product *= num
        if product == input_args.world_size:
            perms = list(itertools.permutations(combination))
            unique_configs = list(set(perms))
            part1_combinations.extend(unique_configs)
            part1_combinations = filter_specify_strategy(para, part1_combinations)

    # part2. [ep, op, vp, mbs]
    # ep是dp*tp的约数
    # vp是 < ceil(总层数/pp)的任意整数
    # mbs 取值(1， 2， 4)中
    part2_combinations = []
    num_layers = input_args.num_layers
    for world_size_config in part1_combinations:
        op_options = get_op_options(world_size_config)
        vp_options = find_integers_less_than_ceil(num_layers, world_size_config[3])
        mbs_options = [1, 2, 4]
        # todo: mindformers 和 mindspeed 对ep的限制不同，当前是mindformers的，待修改，把这个挪到专家剪枝
        ep_options = find_ep(input_args.expert_num)
        result = list(itertools.product(ep_options, op_options, vp_options, mbs_options))

        dp = world_size_config[0]
        for tmp_result in result:
            op = tmp_result[1]
            # 格式 [(dp, tp, cp, pp), (ep, op, vp, mbs)]
            if not para.STRATEGY.FSDP:
                if op != dp:
                    # 这里不搜FSDP的含义是: FSDP取剩余的空间
                    tmp_result = (tmp_result[0], -1, tmp_result[2], tmp_result[3])
            part2_combinations.append([world_size_config, tmp_result])


    if not para.STRATEGY.EP:
        part2_combinations = [config for config in part2_combinations if config[1][0] == 1]
    # part3. sp只有开关与否 格式 [[(dp, tp, cp, pp), (ep, op, vp, mbs)], sp]
    final_combinations = []
    for part2_config in part2_combinations:
        final_combinations.append([part2_config, False])
        if part2_config[0][1] != 1:
            final_combinations.append([part2_config, True])
    return final_combinations


def filter_specify_strategy(para, part1_combinations):
    if not para.STRATEGY.DP:
        part1_combinations = [config for config in part1_combinations if config[0] == 1]
    if not para.STRATEGY.TP:
        part1_combinations = [config for config in part1_combinations if config[1] == 1]
    if not para.STRATEGY.CP:
        part1_combinations = [config for config in part1_combinations if config[2] == 1]
    if not para.STRATEGY.PP:
        part1_combinations = [config for config in part1_combinations if config[3] == 1]
    return part1_combinations


# 优化器并行可以是dp的任意因子
def get_op_options(world_size_config):
    dp = world_size_config[0]
    op_options = cal_factor(dp)
    return op_options

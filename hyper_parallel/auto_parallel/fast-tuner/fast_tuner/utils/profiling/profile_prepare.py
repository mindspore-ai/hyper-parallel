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
"""profile prepare work"""
import os
from fast_tuner.utils.logger import logger
from fast_tuner.utils.common import generate_files


def find_file_by_name(directory, filename):
    """
    Docstring for find_file_by_name
    
    :param directory: Description
    :param filename: Description
    """
    for root, _, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)
    return None

def decide_pp_dp(config, available_devices):
    """

    :param config: [config, available_devices]
    :param available_devices: 可用来做profiling的卡数
    :return: [dp, tp, pp, ep]
    """
    dp, tp, origin_pp, ep, cp, op = config[:6]
    world_size = dp * tp * origin_pp * cp
    device_multiple = world_size //available_devices
    min_pp = 2   # pp最小取2

    # 检查设备数量是足够的, dp * tp >= ep
    if max(min_pp * tp * cp, min_pp * ep) > available_devices:
        print(f'not enough devices for config: dp: {dp}, tp: {tp}, pp: 2, ep: {ep}')

    # 裁剪pp
    if device_multiple >= origin_pp // min_pp:
        pp = 2
    else:
        if device_multiple % 3 == 0:
            if origin_pp % 3 == 0:
                pp = origin_pp // 3
            else:
                pp = origin_pp // (device_multiple // 3)
        else:
            pp = origin_pp // device_multiple

    # 调整dp以满足设备数量限制
    cur_multiple = pp * dp * tp * cp // available_devices
    if cur_multiple > 1:
        dp //= cur_multiple
    return [dp, tp, pp, ep, cp, op]

def decide_dp(config, available_devices):
    """
    :param config: [config, available_devices]
    :param available_devices: 可用来做profiling的卡数
    :return: [dp, tp, pp, ep]
    """
    dp, tp, _, ep = config[:4]
    pp = 2  # pp取2

    # 检查设备数量是足够的, dp * tp >= ep
    if max(pp * tp, pp * ep) > available_devices:
        print(f'not enough devices for config: dp: {dp}, tp: {tp}, pp: {pp}, ep: {ep}')
        return None

    # 调整dp以满足设备数量限制
    cur_multiple = pp * dp * tp // available_devices
    if cur_multiple > 1:
        dp //= cur_multiple
        if dp % ep != 0:
            print(
                f'dp_size ({dp}) is not divisible by ep_size({ep}), for config: dp: {dp}, tp: {tp}, pp: {pp}, ep: {ep}')
            return None
    return [dp, tp, pp, ep]

def decide_dp_for_titan(config, available_devices):
    """
    Docstring for decide_dp_for_titan
    
    :param config: Description
    :param available_devices: Description
    """
    dp, tp, pp, ep, cp, op = config[:6]
    world_size = dp * tp * pp * cp
    pp = 1
    if max(pp * tp, pp * ep) > available_devices:
        logger.info(f'not enough devices for config: dp: {dp}, tp: {tp}, pp: {pp}, ep: {ep}, cp: {cp}, fsdp: {op}')
        return None
    if world_size <= 8:
        if world_size != available_devices:
            logger.info(f'Currently, we assume world size {world_size}'
                        f'equals the number of available devices {available_devices} within a single node')
            return None
        return config[:6]
    # 调整dp以满足设备数量限制
    cur_multiple = pp * dp * cp * tp // available_devices
    if cur_multiple > 1:
        if op // cur_multiple == 0:
            logger.info(f'prune cause of fsdp: ddp: {dp//op}, tp: {tp}, pp: {pp}, ep: {ep}, cp: {cp}, fsdp: {op}')
            return None
        op //= cur_multiple
        dp //= cur_multiple
    return [dp, tp, pp, ep, cp, op]

def decide_pp_dp_llama(config, available_devices):
    """

    :param config: [config, available_devices]
    :param available_devices: 可用来做profiling的卡数
    :return: [dp, tp, pp]
    """
    dp, tp, origin_pp = config[:3]
    world_size = dp * tp * origin_pp
    device_multiple = world_size // available_devices
    min_pp = 1   # pp最小取2

    # 检查设备数量是足够的, dp * tp >= ep
    if min_pp * tp > available_devices:
        print(f'not enough devices for config: dp: {dp}, tp: {tp}, pp: 2')
        return None

    # 裁剪pp
    if device_multiple >= origin_pp // min_pp:
        pp = min_pp
    else:
        if device_multiple % 3 == 0:
            if origin_pp % 3 == 0:
                pp = origin_pp // 3
            else:
                pp = origin_pp // (device_multiple // 3)
        else:
            pp = origin_pp // device_multiple

    # 调整dp以满足设备数量限制
    cur_multiple = pp * dp * tp // available_devices
    if cur_multiple > 1:
        dp //= cur_multiple
    return [dp, tp, pp]

def trans_config_satisfy_rank_num(mem_prune_space, para, input_args):
    """

    :param mem_prune_space: [dp,tp,pp,ep,cp,op]
    :param para: user config para for tool
    :param input_args: model args and train args from config file
    :return: list[[dp, tp, pp, ep, cp, op, offset, num_layers]]
    """
    profile_configs = []
    rank_num = min(para.RANK_NUM, input_args.world_size)
    for config in mem_prune_space:
        trans_config = budget_profile_config_generator(config, para, rank_num)
        if trans_config is not None and trans_config not in profile_configs:
            profile_configs.append(trans_config)
    logger.info(f"profile configs len: {len(profile_configs)} by {rank_num} devices")
    # todo: 待添加注释，说明筛选逻辑
    if len(profile_configs) > 10:
        # pp <= 16
        profile_configs = [cand for cand in profile_configs if cand[2] <= 16]
        # 按tp , cp, 越小越好,  ep从大到小排
        profile_configs = sorted(profile_configs, key=lambda x: (x[1], x[4], -x[3]))
        profile_configs = profile_configs[:10]
    print(*(config for config in profile_configs), sep='\n')
    return profile_configs

def budget_profile_config_generator(config, para, available_devices):
    """
    根据可用卡数，将config转换成当前卡资源可满足的并行配置做profile,先裁剪pp, pp最小为2，然后再缩减dp
    一次profiling同时获取不开重计算和完全重计算的信息

    :param config: [dp, tp, pp, ep, cp, op, evaluate_peak_mem]
    :param para: user config para for tool
    :param available_devices:
    :return: [dp, tp, pp, ep, offset, recompute, num_layers], min(pp)=2, 层数(num_layers+MTP)
    """
    if para.YAML_PATH:
        if len(config) < 4:
            basic_config = decide_pp_dp_llama(config, available_devices)
        else:
            basic_config = decide_pp_dp(config, available_devices)
        pp = basic_config[2]
        if len(config) < 4:
            offset = 0
            recompute = 0
            num_layers = 32
        else:
            if pp == 2:
                offset = [1, 0]
                recompute = [6, 0]
                num_layers = 11
            elif pp == 4:
                offset = [1, 1, 1, 0]
                recompute = [3, 3, 0, 0]
                num_layers = 11
            elif pp == 8:
                offset = [2, 0, 0, 0, 0, 1, 1, 1]
                recompute = [3, 1, 1, 1, 1, 2, 0, 2]
                num_layers = 13
            else:
                print(f'pp {pp} not supported')
                return None

        config_num_layers = num_layers if para.SHELL_PATH else (num_layers - 1)
        basic_config.extend([offset, recompute, config_num_layers])
        return basic_config
    if para.SHELL_PATH:
        if len(config) < 4:
            # TODO llama系模型待适配
            basic_config = decide_pp_dp_llama(config, available_devices)
        else:
            basic_config = decide_dp(config, available_devices)
        if basic_config is None:
            return None
        num_layer_list = [4, 3]
        recompute_num_layers = 1
        num_layers = 7
        basic_config.extend([num_layer_list, recompute_num_layers, num_layers])
        return basic_config
    if para.TOML_PATH:
        basic_config = decide_dp_for_titan(config, available_devices)
        return basic_config

    print('Error: No config file path!')
    return None

def taylor_pp_adaptor(profile_info):
    layer_ratio = (profile_info['dense_fw']+profile_info['dense_bw'])/(profile_info['moe_fw']+profile_info['moe_bw'])
    backward_ratio = profile_info['moe_bw']/profile_info['moe_fw']
    return layer_ratio, backward_ratio

def sapp_adaptor(profile_info):
    body_dense = (profile_info['dense_fw']+profile_info['dense_bw'])/3
    body_moe = (profile_info['moe_fw']+profile_info['moe_bw'])/3
    tail = profile_info['head']/3
    return profile_info['embed'], body_dense, body_moe, tail

def profile_prepare(mem_prune_space, para, input_args):
    """
    prepare for profiling

    :param para: user input args for tool
    :param mem_prune_space: [dp, tp, pp, ep, cp, op, evaluate_peak_mem] or [dp, tp, pp]
    :param input_args: model args and train args from config file
    :return: ordered [dp, tp, pp, ep, cost]
    """
    # 配置转换可在现有卡资源下profile的配置
    profile_configs = trans_config_satisfy_rank_num(mem_prune_space, para, input_args)
    profile_dir = 'profile_yaml' if para.YAML_PATH else 'profile_shell' if para.SHELL_PATH else 'profile_toml'
    profile_file_dir = os.path.abspath(para.OUTPUT_PATH)  + os.sep + profile_dir + os.sep
    file_task = "profile_yaml" if para.YAML_PATH else "profile_shell" if para.SHELL_PATH else 'profile_toml'
    file_path_list, output_dir_list = generate_files(profile_configs, profile_file_dir, file_task, para, input_args)

    result = []
    for config, shell_file_path, profile_dir in zip(profile_configs, file_path_list, output_dir_list):
        # 创建子列表的副本并添加元素（避免修改原列表）
        new_config = config.copy()
        new_config.append(shell_file_path)
        new_config.append(profile_dir)
        result.append(new_config)
    return result, profile_file_dir

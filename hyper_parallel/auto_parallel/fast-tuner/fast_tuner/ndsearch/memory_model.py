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
"""nd phase memory model"""

import re
import os
import math
import csv
from enum import Enum
from collections import defaultdict
from fast_tuner.utils.logger import logger
from fast_tuner.utils.common import generate_files, initial_offset, offset_for_dualpipe, is_dualpipe_open
from fast_tuner.utils.dryrun_manage import launch_dryrun, read_dryrun_info


def memory_simple_prune(config, profile_info):
    memory_max = config.memory_max
    tmp_layer = config.num_layer // config.pp_size
    profile_memory = config.pp_size * tmp_layer * profile_info.act_mem_full_recomp \
        + profile_info.embedding_mem + tmp_layer * profile_info.static_layer_MoE
    coe_mem = 0.2
    if profile_memory * (1 + coe_mem) > memory_max:
        return False
    return True

def trans_format(dryrun_info_init, test_ep):
    """
    trans dryrun_info_init to [dp, tp, pp, ep, peak_mem_ep1, peak_mem_ep2]
    """
    config_mem_dict = defaultdict(lambda: [0, 0])
    for config in dryrun_info_init:
        key = tuple(config[:3])
        peak_mem = config[4]
        ep = config[3]
        if ep == test_ep[0]:
            index = 0
        elif ep == test_ep[1]:
            index = 1
        else:
            logger.error(f"error ep {ep} is not supported")
            continue
        config_mem_dict[key][index] = peak_mem

    return [list(key) + value for key, value in config_mem_dict.items()]

def grey_box_memory_prune(mindformers_args, dryrun_info_init, test_ep, max_expert_parallel):
    """
    ep灰盒剪枝

    :param max_expert_parallel: ep最大值
    :param test_ep:
    :param dryrun_info_init: [dp, tp, pp, ep, peak_value]
    :param mindformers_args:
    :return: [dp, tp, pp, ep, evaluate_peak_mem]
    """
    dryrun_info = trans_format(dryrun_info_init, test_ep)
    ep1, ep2 = test_ep
    logger.info(f"dryrun_info len: {len(dryrun_info)} format [dp, tp, pp, peak_mem_ep{ep1}, peak_mem_ep{ep2}]")
    logger.info("\n".join(str(config) for config in dryrun_info))

    try:
        max_mem = int(re.search(r'\d+', mindformers_args.context.max_device_memory).group()) * 1024
    except (AttributeError, ValueError):
        max_mem = 58 * 1024

    memory_aware_configs = []
    logger.info("format: dp_tp_pp_ep_evaluateMem")
    ep_power = find_power_of_two(max_expert_parallel)
    for dp, tp, pp, peak_ep, peak_ep_double in dryrun_info:
        # 线性拟合ep会影响到的内存和ep不会影响到的内存
        ep_memory = (peak_ep - peak_ep_double) * test_ep[1] #所有专家的内存
        base_memory = peak_ep - ep_memory / test_ep[0]
        # 确定ep最大能开多大,最大为6, ep最大64
        ep_upperbound = 0
        for i in range(ep_power+1):
            if (dp*tp) % (2**i) == 0:
                ep_upperbound += 1
        # 输出满足内存上限的ep，如果ep64都不够就不返回
        for j in range(ep_upperbound):
            ep = 2 ** j
            evaluate_mem = base_memory + ep_memory / ep
            logger.info(f"{dp}_{tp}_{pp}_{ep}_{evaluate_mem}")
            if evaluate_mem <= max_mem:
                memory_aware_configs.append([dp, tp, pp, ep, evaluate_mem])
    return memory_aware_configs

def find_power_of_two(m):
    if m <= 0:
        return None
    power = math.log2(m)
    if power.is_integer():
        return int(power)
    return None

def filter_oom(search_space, input_args, para):
    """
    filter evaluate oom configs
    """
    # todo: 是否dryurn返回值不同，需判断这里是否需要处理
    if para.DRYRUN:
        # 生成要做dryrun的配置
        care_part_configs = select_dry_config(search_space, input_args)
        test_ep = (8, 16)
        dry_config = generate_dry_config(care_part_configs, input_args, test_ep)
        dryrun_exe_switch = bool(para.DRYRUN_DATA_DIR)
        if dryrun_exe_switch:
            logger.info("need auto dryrun process")
            file_task = "dryrun_yaml" if para.YAML_PATH else "dryrun_shell"
            dryrun_file_dir = os.path.abspath(para.OUTPUT_PATH) + os.sep + file_task + os.sep
            generate_files(dry_config, dryrun_file_dir, file_task, para, input_args)
            dryrun_data_dir = os.path.join(os.path.abspath(para.OUTPUT_PATH), "dryrun_output")
            if input_args.mf_args:
                # 基于mindformers(mindspore)才做dryrun
                launch_dryrun(input_args, dryrun_file_dir, dryrun_data_dir, para)
        else:
            dryrun_data_dir = para.DRYRUN_DATA_DIR

        if input_args.mf_args:
            dryrun_info = read_dryrun_info(dryrun_data_dir)
            candidate_configs = grey_box_memory_prune(input_args, dryrun_info, test_ep, para.MAX_EXPERT_PARALLEL)
        else:
            candidate_configs = dry_config
        generate_csv(para.OUTPUT_PATH, candidate_configs, input_args)
        return candidate_configs

    candidate_configs = [] #the format is [dp, tp, pp, ep, cp, op, evaluate_mem]
    for config in search_space:
        op_disable = False
        dp, tp, cp, pp = config[0][0]
        ep, op = config[0][1][:2]
        if op < 0:
            op_disable = True
            op = dp
        input_args.op, input_args.tp, input_args.pp, input_args.ep = op, tp, pp, ep
        dense_size, moe_size, vocab_size = compute_weight_and_optimizer_memory(input_args)
        try:
            max_mem = int(re.search(r'\d+', input_args.context.max_device_memory).group()) * 1024
        except (AttributeError, ValueError):
            max_mem = 58 * 1024
        if input_args.pp == 1:
            evaluate_memory = (2 * vocab_size+moe_size * (input_args.num_layers - input_args.first_k_dense_replace)
                               + dense_size * input_args.first_k_dense_replace)
        elif input_args.first_k_dense_replace > input_args.num_layers // input_args.pp:
            estimated_first = vocab_size + dense_size * (input_args.num_layers // input_args.pp)
            estimated_general = moe_size * math.ceil((input_args.num_layers+2)/input_args.pp)
            evaluate_memory = max(estimated_first, estimated_general)
        else:
            estimated_first = (vocab_size + dense_size * input_args.first_k_dense_replace
                 + moe_size * (input_args.num_layers // input_args.pp - input_args.first_k_dense_replace))
            estimated_general = moe_size * math.ceil((input_args.num_layers + 2) / input_args.pp)
            evaluate_memory = max(estimated_first, estimated_general)
        if op_disable: op = -1
        if evaluate_memory <= max_mem:
            if [dp, tp, pp, ep, cp, op, evaluate_memory] not in candidate_configs:
                candidate_configs.append([dp, tp, pp, ep, cp, op, evaluate_memory])
        else:
            logger.info(f"mem over limit: evaluate mem {evaluate_memory}"
                        f"config dp {dp} tp {tp} pp {pp} ep {ep} cp {cp} op {op}")
    logger.info(f"Prune Search space size: {len(candidate_configs)},"
                f"format: [dp, tp, pp, ep, cp, op/fsdp, evaluate_peak_mem]")
    return candidate_configs

def select_dry_config(valid_configs, input_args):
    """

    :param valid_configs: [[(dp, tp, cp, pp), (ep, op, vp, mbs)], sp]
    :param input_args: 配置文件参数
    :return: [[(dp, tp, cp, pp), (ep, op, vp, mbs)], sp]
             其中vp=1, mbs=1, sp=true  每个(dp, tp, cp, pp)，对应ep最大的配置列表
    """
    first = valid_configs[0][0][0]
    max_ep = valid_configs[0][0][1][0]
    op_with_ep = valid_configs[0][0][1][1]
    ans = []
    for config in valid_configs:
        current_first = config[0][0]
        current_ep = config[0][1][0]
        current_op = config[0][1][1]
        pp = config[0][0][3]
        num_layers = input_args.num_layers

        if is_dualpipe_open(input_args) and pp * 2 >= num_layers:
            continue

        if current_first == first:
            if current_ep > max_ep:
                max_ep = current_ep
                op_with_ep = current_op
        else:
            ans.append([[first, [max_ep, op_with_ep, 1, 1]], True])
            first = current_first
            max_ep = current_ep
            op_with_ep = current_op
    # 添加最后一组数据
    ans.append([[first, [max_ep, op_with_ep, 1, 1]], True])
    logger.info(f"Dryrun candidate config size: {len(ans)}")
    return ans

def generate_csv(output_path, dryrun_config, input_args):
    """
    generate nd result to csv file
    """
    # 表头
    if input_args.expert_num is not None:
        headers = ['dp', 'tp', 'pp', 'ep', 'evaluate_mem']
    else:
        headers = ['dp', 'tp', 'pp', 'evaluate_mem']

    # 写入 CSV 文件
    try:
        csv_path = os.path.join(os.path.abspath(output_path), "nd_candidate_config.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # 写入表头
            writer.writerow(headers)
            # 写入数据
            writer.writerows(dryrun_config)
        logger.info("CSV file generate succ.")
    except Exception as e:
        logger.info(f"write CSV file fail: {e}")

class CareTpye(Enum):
    UNKNOWN = 0
    WITH_EXPERT_MF = 1
    NO_EXPERT = 2
    WITH_EXPERT_NO_MF = 3

def generate_dry_config(care_part_configs, input_args, test_ep):
    """

    :param test_ep: 要做dryrun的ep
    :param care_part_configs: [[(dp, tp, cp, pp), (ep, op, vp, mbs)], sp]
    :param input_args: 模型及环境等信息
    :return: [dp, tp, pp, ep, offset] 或 [dp, tp, pp]
    """
    dry_run_config = []
    layers_num = input_args.num_layers

    if input_args.expert_num is not None and input_args.mf_args is not None:
        care_type = CareTpye.WITH_EXPERT_MF
        logger.info('dryrun configs format: dp_tp_pp_ep_offset')
    elif input_args.expert_num is None:
        care_type = CareTpye.NO_EXPERT
        logger.info('dryrun configs format: dp_tp_pp')
    else:
        care_type = CareTpye.WITH_EXPERT_NO_MF
        logger.info('dryrun configs format: dp_tp_pp_ep')

    for config in care_part_configs:
        dp, tp, _, pp = config[0][0]
        ep = config[0][1][0]
        # 若为deepseek模型
        if care_type == CareTpye.WITH_EXPERT_MF:
            for ep in test_ep:
                # mindformers的约束
                if input_args.mf_args is not None and dp * tp % ep != 0:
                    continue

                use_zero_bubble_v = is_dualpipe_open(input_args)

                offset_calculator = offset_for_dualpipe if use_zero_bubble_v else initial_offset
                new_config = [dp, tp, pp, ep, offset_calculator(pp, layers_num)]
                dry_run_config.append(new_config)
        elif care_type == CareTpye.NO_EXPERT:
            new_config = [dp, tp, pp]
            dry_run_config.append(new_config)
        else:
            new_config = [dp, tp, pp, ep]
            dry_run_config.append(new_config)


    logger.info(f"Dryrun config size: {len(dry_run_config)}")
    for config in dry_run_config:
        config_str = "_".join(str(x) for x in config)
        logger.info(config_str)
    return dry_run_config

NUM_BYTES_IN_MEGA = 1024 * 1024

def compute_weight_and_optimizer_memory(input_args):
    """

    :param input_args: input training parameters
    :return: [dense_layer mem, moe_layer mem, embedding_layer mem]/MB
    """
    #Attention projection size
    if input_args.kv_channels:
        query_projection_to_hidden_size_ratio =(
                input_args.kv_channels * input_args.num_attention_heads / input_args.hidden_size)
    else: query_projection_to_hidden_size_ratio = 1
    # Group Query Attention.
    if not input_args.group_query_attention:
        input_args.num_query_groups = input_args.num_attention_heads
    # MoE.
    num_experts = 1 if input_args.expert_num is None else input_args.expert_num
    gated_linear_multiplier = 3 / 2 if input_args.swiglu else 1

    if input_args.expert_num is not None:
        moe_ffn_hidden_size = input_args.moe_intermediate_size
        shared_expert_ffn_hidden_size = (
            input_args.moe_intermediate_size
            if input_args.moe_shared_expert_intermediate_size is None
            else input_args.moe_shared_expert_intermediate_size
        )
    else:
        moe_ffn_hidden_size = 0
        shared_expert_ffn_hidden_size = 0

    if input_args.multi_latent_attention:
        if input_args.qk_head_dim:
            qk_head_dim = input_args.qk_head_dim
        elif input_args.qk_nope_head_dim:
            qk_head_dim = input_args.qk_nope_head_dim
        else:
            qk_head_dim = 0
            print('qk head dim not specified')
        if input_args.qk_pos_emb_head_dim:
            qk_pos_emb_head_dim = input_args.qk_pos_emb_head_dim
        elif input_args.qk_pos_rope_head_dim:
            qk_pos_emb_head_dim = input_args.qk_pos_rope_head_dim
        else:
            qk_pos_emb_head_dim = 0
            print('qk pos head dim not specified')
        assert not input_args.group_query_attention
        if input_args.q_lora_rank is None:
            q_term = input_args.hidden_size * input_args.num_attention_heads * (qk_head_dim + qk_pos_emb_head_dim)
        else:
            ## q lora + rope + q norm
            q_term = input_args.q_lora_rank * (
                    input_args.hidden_size + input_args.num_attention_heads * (qk_head_dim + qk_pos_emb_head_dim) + 1)

        self_attn_term = (
                q_term
                ## kv lora + rope + kv norm
                + input_args.kv_lora_rank
                * (input_args.hidden_size + input_args.num_attention_heads * (qk_head_dim + input_args.v_head_dim) + 1)
                + input_args.hidden_size * qk_pos_emb_head_dim
                ## o proj
                + (input_args.num_attention_heads * input_args.v_head_dim) * input_args.hidden_size
        )
    else:
        self_attn_term = (
                2
                * input_args.hidden_size
                * input_args.hidden_size
                * (
                # Attention.
                (
                        (1 + (input_args.num_query_groups / input_args.num_attention_heads))
                        * query_projection_to_hidden_size_ratio
                )
            )
        )

    num_parameters_in_dense_ffn = (
            2
            * input_args.hidden_size
            * (
            # Dense MoE MLP.
            (input_args.ffn_hidden_size * gated_linear_multiplier)
            # Transformer layer norms.
            + 2
        )
    )

    num_parameters_in_moe_ffn_without_routed_expert = (
            2
            * input_args.hidden_size
            * (
                # MoE MLP.
                # Shared MoE MLP.
                + (shared_expert_ffn_hidden_size * gated_linear_multiplier)
                # Transformer layer norms.
                + 2
            )
    )

    num_parameters_of_routed_expert = (2 * input_args.hidden_size * moe_ffn_hidden_size
                                       * num_experts * gated_linear_multiplier / input_args.ep)

    embedding_size = input_args.hidden_size * input_args.padded_vocab_size
    final_layernorm = 2 * input_args.hidden_size

    num_bytes_per_parameter = (
        18 if not input_args.use_distributed_optimizer else 6 + (12 / input_args.op)
    )

    if not input_args.use_distributed_optimizer:
        num_bytes_per_parameter_routed_expert = 18
    elif input_args.op < input_args.ep:
        num_bytes_per_parameter_routed_expert = 18
    else:
        num_bytes_per_parameter_routed_expert = 6 + (12 / input_args.op * input_args.ep)

    embedding_layer = embedding_size / input_args.tp * num_bytes_per_parameter
    dense_layer = (num_parameters_in_dense_ffn + self_attn_term) / input_args.tp * num_bytes_per_parameter
    moe_layer = (self_attn_term + num_parameters_in_moe_ffn_without_routed_expert) * num_bytes_per_parameter + \
                num_parameters_of_routed_expert * num_bytes_per_parameter_routed_expert
    moe_layer /= input_args.tp
    final_layernorm /= input_args.tp

    if input_args.pp == 1:
        return [dense_layer/NUM_BYTES_IN_MEGA, moe_layer/NUM_BYTES_IN_MEGA,
                (embedding_layer * 2 + final_layernorm)/NUM_BYTES_IN_MEGA]
    return [dense_layer/NUM_BYTES_IN_MEGA, moe_layer/NUM_BYTES_IN_MEGA, embedding_layer/NUM_BYTES_IN_MEGA]

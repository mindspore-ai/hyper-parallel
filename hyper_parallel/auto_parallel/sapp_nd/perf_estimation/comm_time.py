# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""Experimental : Comm time"""
from copy import deepcopy
from hyper_parallel.auto_parallel.sapp_nd.nd.logger import perf_logger as logger
import hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware as Hard
import hyper_parallel.auto_parallel.sapp_nd.nd.dimensions as Dim
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType
from hyper_parallel.auto_parallel.sapp_nd.nd.debug import PerfParts
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.comm import EvalLayerComm
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation._context import NodeEval, Context
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.head import EvalHead
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.tail import EvalTail
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.body import EvalBody
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.layer_block import (
    EvalAttn,
    EvalFFn,
    EvalNorm,
)
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.utils_classes import NetworkLevel, PerformanceType
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.getters import (
    get_layer_custom_configs,
    get_table_quantity,
)

COUNT_OPTIMIZER = False


def fill_dp_table(cfg, tables):
    """DP"""
    table_dp = {}
    table_dp["n_attMM"] = cfg.h * cfg.h / cfg.t
    table_dp["n_ffMM"] = cfg.h * cfg.hff / cfg.t
    table_dp["n_normOp"] = 2 * cfg.h / cfg.sp

    if COUNT_OPTIMIZER:
        table_dp["n_attParamCast"] = (
            11 * cfg.h * cfg.h / (cfg.d if cfg.has_op else 1)
        )
        table_dp["n_ffParamCast"] = (
            11 * cfg.h * cfg.hff / (cfg.d if cfg.has_op else 1)
        )
    for op in table_dp:
        table_dp[op] *= cfg.bytes_norm if op == "n_normOp" else cfg.bytes_p

    table_exp_dp = deepcopy(table_dp)
    table_exp_dp["n_ffMM"] = (
        2
        * (cfg.n_exp + cfg.n_shared_exp)
        * cfg.h
        * cfg.hff_exp
        / cfg.t
        * cfg.bytes_p
    )
    tables[Dim.DP] = table_dp
    tables["exp_dp"] = table_exp_dp


def fill_tp_table(cfg, tables):
    """TP"""
    table_tp = {}
    high_tp_bias = 11 / 16 if cfg.t >= 8 else 1  # Fix this
    table_tp["n_gather"] = cfg.b * cfg.s * cfg.h * high_tp_bias

    for op in table_tp:
        table_tp[op] *= cfg.bytes_compute

    table_exp_tp = deepcopy(table_tp)
    table_exp_tp["n_gather"] = (
        cfg.b * cfg.s * cfg.h * 1.5 * (cfg.ep / cfg.d) * cfg.bytes_compute
    )
    tables["tp"] = table_tp
    tables["exp_tp"] = table_exp_tp


def fill_ep_table(cfg, tables, device_type):
    """EP"""
    intra_devices = device_type.intra_node_num()
    table_ep = {}
    inter_node_bias_ep = 1
    table_ep["n_ffMM"] = (
        4
        * cfg.n_chosen_exp
        * cfg.b
        * cfg.s
        * cfg.h
        * (max(4, cfg.os_max_shard) / cfg.t)
        * cfg.cap_fact
        * (
            cfg.os_max_shard / min(intra_devices, cfg.ep)
            + (
                inter_node_bias_ep
                * cfg.os_max_shard
                / (cfg.ep / intra_devices)
                if cfg.ep > intra_devices
                else 0
            )
        )
    )

    for op in table_ep:
        table_ep[op] *= cfg.bytes_compute
    tables[Dim.EP] = table_ep


def dp_ratio(cfg, device_type):
    """formula"""
    return (
        0
        if cfg.comm_d_non_exp == 0
        else 1
        - True  # overlap_dp, Completely overlap standard DP comm
        + (
            1 / 16
            if cfg.n_exp == 1
            else 1 / max(1, cfg.ep / device_type.intra_node_num()) / 1.25
        )  # overlap_op, Bias in overlapping OP comm (todo:make it dynamic too)
        * (cfg.comm_d_non_exp - 1)
        * cfg.os_max_shard
        / cfg.d
    )


def comm_embed_ouput(cfg):
    """ "formula"""
    comm_embed = cfg.bytes_compute * cfg.h * cfg.v / cfg.shard_embed
    comm_output = cfg.h * cfg.v / cfg.t
    return comm_embed, comm_output


def estimate_op_bulk_comm(*args, **kwargs):
    """FW + BW"""
    param = {
        "cfg": args[0],
        "ccfg": args[1],
        "stages": args[2],
        "device_type": args[3],
        "with_recomp": kwargs.get(
            "with_recomp", args[4] if len(args) > 4 else False
        ),
        "debugger": kwargs.get("debugger", args[5] if len(args) > 5 else None),
    }

    param["tables"] = {}
    fill_dp_table(param["cfg"], param["tables"])

    param['dp_ratio'] = dp_ratio(param['cfg'], param['device_type'])

    param["comm_embed"], param["comm_output"] = comm_embed_ouput(param["cfg"])

    if param["cfg"].dc_kv != 0:  # Deepseek
        param["comm_output"] += param["cfg"].h * (
            2 * param["cfg"].h + param["cfg"].v
        )
        param["comm_output"] *= param["cfg"].n_mtp

    param["comm_output"] *= param["cfg"].bytes_p

    fill_tp_table(param["cfg"], param["tables"])
    fill_ep_table(param["cfg"], param["tables"], param["device_type"])

    lccfgs = get_layer_custom_configs(param["cfg"])
    logger.info(lccfgs)
    param["layer_count"] = 0
    param["idx_lccfg"] = 0
    comms = {Dim.DP: [], Dim.TP: [], Dim.EP: []}
    # ignores comm recomp, to improve
    for stage in param["stages"]:
        comm = {Dim.DP: 0.0, Dim.TP: 0.0, Dim.EP: 0.0}
        for chunk in stage:
            for layer in chunk:
                param["layer_count"], param["idx_lccfg"] = (
                    estimate_op_bulk_comm_layer(
                        param,
                        lccfgs,
                        layer=layer,
                        layer_count=param["layer_count"],
                        idx_lccfg=param["idx_lccfg"],
                    )
                )
        if param["ccfg"].ttype == PerformanceType.TIME:
            for dim, ov in zip([Dim.DP, Dim.TP, Dim.DP], [0.0, 0.0, 0.0]):
                comm[dim] = estimate_comm_score(
                    param["cfg"],
                    comm[dim],
                    dim,
                    overlap=ov,
                    device=param["device_type"],
                )

        comm[Dim.DP] *= param["dp_ratio"]
        comm[Dim.TP] *= param["cfg"].comm_t
        comm[Dim.EP] *= param["cfg"].comm_ep

        if param["device_type"].name == "A3":
            logger.info("A3 ratio")
            comm[Dim.TP] /= 3

        comms[Dim.DP].append(comm[Dim.DP])
        comms[Dim.TP].append(comm[Dim.TP])
        comms[Dim.EP].append(comm[Dim.EP])

    if param["debugger"] and param["debugger"].is_enabled():
        logger.info("DP_COMM = %s", comms[Dim.DP])
        logger.info("MP_COMM = %s", comms[Dim.TP])
        logger.info("EP_COMM = %s", comms[Dim.EP])
        param["debugger"].info[PerfParts.DP_COMM] = comms[Dim.DP]
        param["debugger"].info[PerfParts.MP_COMM] = comms[Dim.TP]
        param["debugger"].info[PerfParts.EP_COMM] = comms[Dim.EP]

    res = []
    for i, c in enumerate(comms[Dim.TP]):
        res.append(comms[Dim.DP][i] + c + comms[Dim.EP][i])

    return res


def estimate_op_bulk_comm_layer(cfg, lccfgs, **kwargs):
    """for estimate_op_bulk_comm"""
    if kwargs["layer"] == LayerType.EMBEDDING_LAYER:
        kwargs["comm"][Dim.DP] += kwargs["param"]["comm_embed"]
        return kwargs["layer_count"]

    if kwargs["layer"] == LayerType.OUTPUT_LAYER:
        kwargs["comm"][Dim.DP] += kwargs["param"]["comm_output"]
        if cfg.dc_kv != 0:  # Deepseek
            lccfg = lccfgs[kwargs["idx_lccfg"]][0]
            kwargs["comm"][Dim.TP] += cfg.n_mtp * get_table_quantity(
                lccfg,
                kwargs["param"]["tables"]["exp_tp"],
                LayerType.NOT_REC_LAYER,
                kwargs["param"]["with_recomp"],
            )
        return kwargs["layer_count"]

    if (
        kwargs["idx_lccfg"] + 1 < len(lccfgs)
        and lccfgs[kwargs["idx_lccfg"]][1] == kwargs["layer_count"]
    ):
        kwargs["layer_count"] = 0
        kwargs["idx_lccfg"] += 1

    lccfg = lccfgs[kwargs["idx_lccfg"]][0]
    is_moe_layer = lccfg.n_exp > 1

    if is_moe_layer:
        kwargs["comm"][Dim.DP] += get_table_quantity(
            lccfg,
            kwargs["param"]["tables"]["exp_dp"],
            kwargs["layer"],
            kwargs["param"]["with_recomp"],
        )
        kwargs["comm"][Dim.TP] += get_table_quantity(
            lccfg,
            kwargs["param"]["tables"]["exp_tp"],
            kwargs["layer"],
            kwargs["param"]["with_recomp"],
        )
        kwargs["comm"][Dim.EP] += get_table_quantity(
            lccfg,
            kwargs["param"]["tables"][Dim.EP],
            kwargs["layer"],
            kwargs["param"]["with_recomp"],
        )
    else:
        kwargs["comm"][Dim.DP] += get_table_quantity(
            lccfg,
            kwargs["param"]["tables"][Dim.DP],
            kwargs["layer"],
            kwargs["param"]["with_recomp"],
        )
        kwargs["comm"][Dim.TP] += get_table_quantity(
            lccfg,
            kwargs["param"]["tables"]["tp"],
            kwargs["layer"],
            kwargs["param"]["with_recomp"],
        )

    kwargs["layer_count"] += 1
    return kwargs["layer_count"], kwargs["idx_lccfg"]


def prepare_context():
    """context object"""
    ctx = Context()
    ctx.attn_num_p = EvalAttn.num_params_attn
    ctx.ffn_num_p = EvalFFn.num_params_ffn
    ctx.norm_num_p = EvalNorm.num_params_norm

    ctx.node_eval[LayerType.EMBEDDING_LAYER] = NodeEval(
        EvalHead.num_params_embed, None, None
    )
    ctx.node_eval[LayerType.OUTPUT_LAYER] = NodeEval(
        EvalTail.num_params_output, None, None
    )
    ctx.node_eval[LayerType.NOT_REC_LAYER] = NodeEval(
        EvalBody.num_params_layer, None, None
    )
    ctx.enable_accu_log = False
    return ctx


def estimate_from_mem_comm(*args, **kwargs):
    """For memory estimation"""

    param = {
        "cfg": args[0],
        "ccfg": args[1],
        "stages": args[2],
        "device_type": args[3],
    }
    param["debugger"] = kwargs.get(
        "debugger", args[5] if len(args) > 5 else None
    )
    param["ctx"] = prepare_context()

    # For layer type
    param["flatten"] = sum(
        [[f[1]] * f[0] for f in param["cfg"].layer_custom_config], []
    )
    comms = {Dim.DP: [], Dim.TP: [], Dim.EP: [], Dim.CP: []}
    for stage in param["stages"]:
        comm = {Dim.DP: 0.0, Dim.TP: 0.0, Dim.EP: 0.0, Dim.CP: 0.0}
        for chunk in stage:
            for layer in chunk:
                param["ctx"].current_node = layer
                if (
                    layer
                    not in [LayerType.EMBEDDING_LAYER, LayerType.OUTPUT_LAYER]
                    and param["flatten"]
                ):
                    custom_fun = param["flatten"].pop(0)
                    if custom_fun:
                        custom_fun(param["cfg"])
                    logger.info("is layer moe ? %s", param["cfg"].n_exp > 1)
                    param["ctx"].current_node = LayerType.NOT_REC_LAYER
                    logger.info("param ctx %s", param["ctx"])
                    comm[Dim.DP] += EvalLayerComm.dp_comm_layer(param["cfg"], param["ctx"])

                comm[Dim.TP] += EvalLayerComm.tp_comm_layer(
                    param["cfg"], param["ctx"], 1
                )  # / 4 #* (param["cfg"].t - 1)
                comm[Dim.EP] += EvalLayerComm.ep_comm_layer(
                    param["cfg"], param["ctx"], 1
                )  # * param["cfg"].ep
                comm[Dim.CP] += EvalLayerComm.cp_comm_layer(
                    param["cfg"], param["ctx"]
                )
                # min(device_type.level_bound_number[0], param["cfg"].ep)
                # comm_cp += EvalLayerComm.cp_comm_layer
                # (param["cfg"], param["ctx"])



        if param["ccfg"].ttype == PerformanceType.TIME:
            for dim, ov in zip([Dim.DP, Dim.TP, Dim.DP], [0.9, 0, 0.0]):
                comm[dim] = estimate_comm_score(
                    param["cfg"],
                    comm[dim],
                    dim,
                    overlap=ov,
                    device=param["device_type"],
                )

        # if (
        #     not param["cfg"].gmm
        #     and param["cfg"].ep > param["device_type"].level_bound_number[0]
        # ):
        #     comm[Dim.EP] *= 8

        # if comm[Dim.EP] > 0:
        # comm[Dim.EP] *= 2
        # comm[Dim.TP] /= 2

        # comm[Dim.DP] /= 100

        # comm[Dim.EP] += comm[Dim.EP] * param["cfg"].ep / 100
        # comm[Dim.TP] += comm[Dim.TP] * param["cfg"].t / 100

        dev_per_node = param["device_type"].level_bound_number[0]
        comm[Dim.TP] *= max(1, param["cfg"].t // dev_per_node)
        comm[Dim.EP] *= max(1, param["cfg"].ep // dev_per_node)
        comm[Dim.CP] *= max(1, param["cfg"].cp // dev_per_node)

        comm[Dim.TP] /= 2 # TO REMOVE: FOR RUIWEN TEST ONLY
        comm[Dim.DP] = 0 # /= 10 # TO REMOVE: FOR RUIWEN TEST ONLY

        if param["device_type"].name == "A3":
            logger.info("A3 ratio")
            comm[Dim.DP] /= 2
            comm[Dim.TP] /= 2
            comm[Dim.EP] /= 2
            comm[Dim.CP] /= 2

        comms[Dim.DP].append(comm[Dim.DP])
        comms[Dim.TP].append(comm[Dim.TP])
        comms[Dim.EP].append(comm[Dim.EP])
        comms[Dim.CP].append(comm[Dim.CP])

    if param["debugger"] and param["debugger"].is_enabled():
        logger.info("DP_COMM = %s", comms[Dim.DP])
        logger.info("MP_COMM = %s", comms[Dim.TP])
        logger.info("EP_COMM = %s", comms[Dim.EP])
        logger.info("CP_COMM = %s", comms[Dim.CP])
        param["debugger"].info[PerfParts.DP_COMM] = comms[Dim.DP]
        param["debugger"].info[PerfParts.MP_COMM] = comms[Dim.TP]
        param["debugger"].info[PerfParts.EP_COMM] = comms[Dim.EP]
        param["debugger"].info[PerfParts.CP_COMM] = comms[Dim.CP]

    res = []
    for i, c in enumerate(comms[Dim.TP]):
        res += [c + comms[Dim.DP][i] + comms[Dim.EP][i] + comms[Dim.CP][i]]

    return res


def estimate_comm(*args, **kwargs):
    """wrapper"""
    cfg, ccfg, stages, device_type = args[0], args[1], args[2], args[3]
    with_recomp = kwargs.get(
        "with_recomp", args[4] if len(args) > 4 else False
    )
    debugger = kwargs.get("debugger", args[5] if len(args) > 5 else None)
    # return estimate_op_bulk_comm(cfg, ccfg, stages,
    # device_type=device_type, with_recomp=with_recomp,
    # debugger=debugger)
    return estimate_from_mem_comm(
        cfg,
        ccfg,
        stages,
        device_type,
        with_recomp=with_recomp,
        debugger=debugger,
    )


def level_efficiency(level):
    """to improve for Ascend A2"""
    if level == NetworkLevel.NODE:
        return 0.7
    if level == NetworkLevel.CLUSTER:
        return 0.9
    raise ValueError


def level_bandwidth(level):
    """to improve for Ascend A2"""
    if level == NetworkLevel.NODE:
        return 300
    if level == NetworkLevel.CLUSTER:
        return 25
    raise ValueError


def level_latency(level):
    """to improve for Ascend A2"""
    if level == NetworkLevel.NODE:
        return 0.00001
    if level == NetworkLevel.CLUSTER:
        return 0.00002
    raise ValueError


def comm_throughput(level):
    """formula"""
    eff = level_efficiency(level)
    bw = level_bandwidth(level)
    return bw * eff


def estimate_comm_size_time(_, comm_size, level):
    """formula"""
    th = comm_throughput(level)
    lat = level_latency(level)
    return lat + comm_size / th


def estimate_comm_score(
    cfg, comm_volume, dim, overlap=0.0, device=Hard.device_map["A2"]
):
    """score assignment"""
    assignment = device.level_assign(dp=cfg.d, tp=cfg.t, pp=cfg.p)
    score = 0
    for level in range(device.levels):
        # intra_comm = comm_volume * (1-overlap)
        # * (assignment[dim][0]-1) / device.intra_node_bw
        score += (
            comm_volume
            * (1 - overlap)
            * (
                (assignment[dim][level] - 1)
                * device.devices_below_level(level)
                / device.level_bandwidth[level]
            )
        )
    return score

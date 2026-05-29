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
"""performance estimation"""
import json
from copy import deepcopy
import numpy as np

from hyper_parallel.auto_parallel.sapp_nd.nd.logger import perf_logger as logger
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType
from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_config import CostModelConfig
from hyper_parallel.auto_parallel.sapp_nd.nd.common.arch_hooks import check_and_apply_custom_hook
import hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware as Hard
from hyper_parallel.auto_parallel.sapp_nd.nd.debug import PerfParts, RealParts, estimation_in_real_parts

from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.utils_classes import (
    RatioType,
    PerformanceType,
    P2PCommType,
    RecType,
    CustomConfig,
)
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.comm_time import estimate_comm
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.getters import (
    get_layer_custom_configs,
    get_table_quantity,
)


GENERALIZE_PIPELINE_CALCULATION = False
MANUAL_P2P_RATIO = 0.002
BACKWARD_RATIO = 2

def op_table(cfg):
    """op compute load formulas"""
    table = {}
    # cfg.s /= cfg.cp
    table["n_attMM"] = (
        3 * (1 + cfg.n_kv / cfg.a) * cfg.b * cfg.s * cfg.h * cfg.h
    )
    table["n_ffMM"] = 6 * cfg.b * cfg.s * cfg.h * cfg.hff
    table["n_attBMM"] = 6 * cfg.b * cfg.s * cfg.s * cfg.h
    table["n_ffBMM"] = 6 * cfg.b * cfg.s * cfg.s * cfg.hff
    table["n_softmax"] = 13 * cfg.a * cfg.b * cfg.s * cfg.s
    table["n_headCast"] = 3 * cfg.a * cfg.b * cfg.s * cfg.s
    table["n_gather"] = cfg.b * cfg.s * cfg.h * (cfg.t - 1)
    table["n_ffAct"] = 21 * cfg.b * cfg.hff

    table["n_normOp"] = 30 * cfg.b * cfg.s * cfg.h * cfg.t / cfg.sp
    table["n_dropout"] = (
        3 * cfg.b * cfg.s * max(cfg.a * cfg.s, 3 * cfg.h * cfg.t / cfg.sp)
    )
    if cfg.dc_kv != 0:  # Deepseek
        table["n_attMM"] = (
            3
            / 2
            * (
                2 * cfg.dc_kv * cfg.n_kv * cfg.dh
                + cfg.dc_q * cfg.a * (cfg.dh + cfg.dhr)
                + cfg.h * (cfg.a * cfg.dh + cfg.dhr)
            )
            * cfg.b
            * cfg.s
        )
    for op in table:
        table[op] *= cfg.bytes_p / cfg.t / cfg.cp
    # cfg.s *= cfg.cp
    return table


# Evaluation functions
def estimate_op_bulk_comp(cfg, ccfg, stages, with_recomp=False, debugger=None):
    """FW + BW"""
    _ = debugger
    table = op_table(cfg)

    table_exp = deepcopy(table)  # Verify this with MF MoEV2
    table_exp["n_ffMM"] *= (
        cfg.hff_exp / cfg.hff * max(1, cfg.n_chosen_exp) * cfg.cap_fact
    )
    table_exp["n_ffBMM"] *= (
        cfg.hff_exp / cfg.hff * max(1, cfg.n_chosen_exp) * cfg.cap_fact
    )

    lccfgs = get_layer_custom_configs(cfg)
    layer_count = 0
    idx_lccfg = 0

    flops = []
    for stage in stages:
        flops += [0]
        for chunk in stage:
            for layer in chunk:
                if layer == LayerType.EMBEDDING_LAYER:
                    continue

                if layer == LayerType.OUTPUT_LAYER:
                    flops[-1] += (1 if cfg.dc_kv == 0 else cfg.n_mtp) * (
                        1
                        / 16  # bias_imbalance
                        * 6
                        * cfg.b
                        * cfg.v
                        * cfg.h
                        * cfg.s
                        * cfg.bytes_p
                        / cfg.t
                    )
                    continue

                layer_count += 1
                if (
                    idx_lccfg + 1 < len(lccfgs)
                    and lccfgs[idx_lccfg][1] <= layer_count
                ):
                    layer_count = 0
                    idx_lccfg += 1

                flop = get_table_quantity(
                    lccfgs[idx_lccfg][0],
                    table_exp if (lccfgs[idx_lccfg][0].n_exp > 1) else table,
                    layer,
                    with_recomp,
                )

                if ccfg.ttype == PerformanceType.TIME:
                    flop = estimate_comp_flop_time(lccfgs[idx_lccfg][0], flop)

                flops[-1] += flop

    return flops


def estimate_comp(cfg, ccfg, stages, with_recomp=False, debugger=None):
    """wrapper"""
    return estimate_op_bulk_comp(
        cfg, ccfg, stages, with_recomp, debugger=debugger
    )


# Experimental : Flop time


def efficiency(x):
    """obtained via extrapolation"""
    eff = min(
        1.0, max(0.1, 0.00004694 * x**3 + 0.0014 * x**2 - 0.0336 * x + 0.1)
    )
    return eff

def throughput(precision_bytes, flop):
    """assumes matrix"""
    eff = efficiency(flop / (10.0**12))
    return precision_bytes**2 * (10.0**12) * eff


def estimate_comp_flop_time(cfg, flop, is_softmax=False):
    """flop from throughput"""
    th = throughput(
        cfg.bytes_softmax if is_softmax else cfg.bytes_compute, flop
    )
    return flop / th


##############################################################


def get_dynamic_ratio(cfg):
    """comm/comp"""
    if cfg.n_exp == 1:
        return 3 / 2 * (cfg.hff + cfg.s) * (8192 / (cfg.h + cfg.s))
    return 3 / 2 * (cfg.hff_exp + cfg.s) * (8192 / (cfg.h + cfg.s))

def estimate_stage(*args, **kwargs):
    """stage level estimation"""
    cfg = args[0]
    ccfg = args[1]
    compute_perfs = args[2]
    comm_perfs = args[3]
    recompute_perfs = args[4]
    recomm_perfs = args[5]
    debugger = kwargs.get("debugger", args[6] if len(args) > 6 else None)
    comp_w = 1
    comm_w = 1
    if ccfg.rtype == RatioType.COMM_ONLY:
        comp_w = 0
    elif ccfg.rtype == RatioType.COMPUTE_ONLY:
        comm_w = 0
    elif ccfg.rtype == RatioType.STATIC:
        comm_w = 10**4
        ccfg.static_ratio = comm_w
    elif ccfg.rtype == RatioType.DYNAMIC:
        comm_w = get_dynamic_ratio(cfg)
        ccfg.dynamic_ratio = comm_w
    perf = [
        comp_w * compute_perfs[i] + comm_w * comm_perfs[i]
        for i in range(len(compute_perfs))
    ]
    logger.info("ratio = %s", comm_w)
    # ignores comm recomp, to improve
    re_perf = [
        (
            max(0, comp_w * (recompute_perfs[i] - compute_perfs[i]))
            + max(0, comm_w * (recomm_perfs[i] - comm_perfs[i]))
        )
        / (1 + BACKWARD_RATIO)
        for i in range(len(compute_perfs))
    ]

    if debugger and debugger.is_enabled():
        for p in [PerfParts.DP_COMM, PerfParts.MP_COMM, PerfParts.EP_COMM, PerfParts.CP_COMM]:
            debugger.info[p] = [
                comm_w * c for c in debugger.info[p]
            ]
        debugger.info[PerfParts.FW_COMPUTE] = [
            comp_w * comp / (1 + BACKWARD_RATIO) for comp in compute_perfs
        ]
        debugger.info[PerfParts.BW_COMPUTE] = [
            fw * BACKWARD_RATIO for fw in debugger.info[PerfParts.FW_COMPUTE]
        ]
        debugger.info[PerfParts.RECOMPUTE] = re_perf

    return [perf[i] + re_perf[i] for i in range(len(perf))]
    #penalty_fn(stage)
    #return stage

def estimate_pipeline(cfg, stage_perfs, stage_focused=None, debugger=None):
    """pipeline level estimation"""
    logger.info("stage_perfs = %s", stage_perfs)
    straggler_time = max(stage_perfs)
    sum_time = sum(stage_perfs)
    last_straggler_idx = cfg.p - 1 - np.argmax(stage_perfs[::-1])
    logger.info(
        "straggler estim is %s and its stage is %s",
        straggler_time,
        last_straggler_idx,
    )

    non_steady_perf = 0
    steady_perf = 0
    if cfg.p == 1:
        assert len(stage_perfs) == 1
        steady_perf = sum_time * cfg.m
    elif cfg.vp == 1:
        non_steady_perf = sum_time
        if GENERALIZE_PIPELINE_CALCULATION:
            last_idx = last_straggler_idx + 1
            steady_perf = (
                cfg.m - cfg.p + last_straggler_idx
            ) * straggler_time + sum(stage_perfs[last_idx:])
        else:
            steady_perf = (cfg.m - 1) * straggler_time
    else:
        less_extra = cfg.p * (cfg.vp - 1)
        # big_extra = less_extra + cfg.p

        # we assume that times of all micro-blocks in one vp chunk are the same
        straggler_time /= cfg.vp
        sum_time /= cfg.vp

        # more_memory has more micro blocks in warm-up but it does
        # not matter since they will overlap with steady phase
        non_steady_perf = sum_time + less_extra * straggler_time

        # more_memory is a boost to performance and a nerf to memory
        # if cfg.vp_less_memory or True:
        #     steady_perf = (cfg.m * cfg.vp - less_extra - 1) * straggler_time
        # else:
        #     steady_perf = (cfg.m * cfg.vp - big_extra - 1) * straggler_time
        steady_perf = (cfg.m * cfg.vp - less_extra - 1) * straggler_time
        straggler_time *= cfg.vp
        sum_time *= cfg.vp

    pipeline_perf = non_steady_perf + steady_perf
    logger.info(
        "pipeline_perf = non_steady_perf(%.2E) + steady_perf(%.2E)",
        non_steady_perf,
        steady_perf,
    )

    if stage_focused is not None:
        last_straggler_idx = stage_focused
    if debugger and debugger.is_enabled():
        time_sum = 0
        for k in [
            PerfParts.DP_COMM,
            PerfParts.MP_COMM,
            PerfParts.EP_COMM,
            PerfParts.CP_COMM,
            PerfParts.FW_COMPUTE,
            PerfParts.BW_COMPUTE,
            PerfParts.RECOMPUTE,
        ]:
            debugger.info[k] = (
                debugger.info[k][last_straggler_idx] * cfg.m
            )  # / cfg.vp
            logger.info(
                "time_sum += debugger[%s] = %.2E",
                k,
                debugger.info[k],
            )
            time_sum += debugger.info[k]

        if abs(time_sum - straggler_time * cfg.m) < 1e-9:
            logger.warning("Inconsistency found in straggler time calculation")
            time_sum = straggler_time * cfg.m
        logger.info(
            "straggler time = %.2E. %s x stragglers = %.2E",
            straggler_time,
            cfg.m,
            straggler_time * cfg.m,
        )

        bubble = pipeline_perf - time_sum
        logger.info(
            "bubble(%.2E) = pipeline_perf(%.2E) - time_sum(%.2E)",
            bubble,
            pipeline_perf,
            time_sum,
        )
        debugger.info[PerfParts.BUBBLE] = bubble
    return pipeline_perf


def estimate_p2p_comm(cfg, straggler, ratio=MANUAL_P2P_RATIO, debugger=None):
    """pipeline comm"""
    nb_send_recv = 0
    if cfg.vp == 1:
        nb_send_recv = (
            0
            if cfg.p == 1
            else (
                4 * cfg.m
                if cfg.p == 2
                else 4 * cfg.p * cfg.m + 4 * cfg.p * cfg.p - 14 * cfg.p
            )
        )
    else:
        nb_send_recv = (
            0
            if cfg.p == 1
            else (
                8 * cfg.m * cfg.vp - 4 * cfg.m
                if cfg.p == 2
                else (
                    16 * cfg.m * cfg.vp + 12
                    if cfg.p == 4
                    else 4 * cfg.p * cfg.m * cfg.vp
                    + 4 * cfg.p * cfg.p
                    - 13 * cfg.p
                )
            )
        )
    pp_comm = ratio * nb_send_recv / cfg.p * straggler / cfg.sp
    if debugger and debugger.is_enabled():
        debugger.info[PerfParts.PP_COMM] = pp_comm

    return pp_comm


def estimate_perf(cfg, _, stage_perfs, stage_focused=None, debugger=None):
    """wrapper"""
    return estimate_pipeline(cfg, stage_perfs, stage_focused=stage_focused, debugger=debugger)


def estimate_p2p(cfg, ccfg, stage_perfs, debugger=None):
    """wrapper"""
    if ccfg.ptype != P2PCommType.MANUAL:
        p2p = 0
    else:
        p2p = estimate_p2p_comm(cfg, max(stage_perfs), debugger=debugger)
    if debugger and debugger.is_enabled():
        debugger.info[PerfParts.PP_COMM] = p2p
    return p2p


def estimate_layer_perf(*args, **kwargs):
    """for PPB"""
    cfg = args[0]
    stages = kwargs.get("stages", args[2] if len(args) > 2 else None)
    extra_custom_func = kwargs.get(
        "extra_custom_func", args[3] if len(args) > 3 else None
    )
    ccfg = kwargs.get("ccfg", args[4] if len(args) > 4 else CustomConfig())
    debugger = kwargs.get("debugger", args[5] if len(args) > 5 else None)
    # cfg = CostModelConfig(mf_config)
    # Process custom model config
    if extra_custom_func:
        extra_custom_func(cfg)
    else:
        logger.info("auto applying custom model config")
        check_and_apply_custom_hook(cfg)

    new_layer_config = []
    stages = [[LayerType.EMBEDDING_LAYER]]
    for _, layer in cfg.layer_custom_config:
        new_layer_config.append((1, layer))
        stages.append([LayerType.NOT_REC_LAYER])
    stages.append([LayerType.OUTPUT_LAYER])
    cfg.layer_custom_config = new_layer_config

    logger.output("cfg.layer_custom_config = %s", cfg.layer_custom_config)
    logger.output("stages = %s", stages)

    cfg.n = cfg.d * cfg.t * cfg.p

    cfg.n_headCast = 1
    cfg.n_ffAct = 1

    logger.info(str(cfg))
    logger.info(stages)
    logger.info(ccfg)

    perfs = {}
    perfs["compute_perfs"] = estimate_comp(
        cfg, ccfg, stages, with_recomp=False, debugger=debugger
    )
    logger.info("PerfEst: compute_perfs %s", perfs["compute_perfs"])
    perfs["recompute_perfs"] = (
        [0] * cfg.p
        if ccfg.retype not in {RecType.COMPUTE_ONLY, RecType.WITH}
        else estimate_comp(
            cfg, ccfg, stages, with_recomp=True, debugger=debugger
        )
    )

    perfs["comm_perfs"] = estimate_comm(
        cfg, ccfg, stages, args[1], with_recomp=False, debugger=debugger
    )
    logger.info("PerfEst: comm_perfs %s", perfs["comm_perfs"])
    perfs["recomm_perfs"] = (
        [0] * cfg.p
        if ccfg.retype not in {RecType.COMM_ONLY, RecType.WITH}
        else estimate_comm(
            cfg, ccfg, stages, args[1], with_recomp=True, debugger=debugger
        )
    )
    stage_perfs = estimate_stage(
        cfg,
        ccfg,
        perfs["compute_perfs"],
        perfs["comm_perfs"],
        perfs["recompute_perfs"],
        perfs["recomm_perfs"],
        debugger=debugger,
    )
    logger.output("PerfEst: stage_perfs %s", stage_perfs)

    for s, perf in enumerate(stage_perfs):
        stage_perfs[s] = int(perf / 10**12)

    return stage_perfs



def apply_regression_coefficients(coeffs, debugger, old_perf):
    """
    applies the coefficients present in regression's cache_file
    """
    compute_ratio = coeffs.get("COMPUTE")
    for part, raw in list(debugger.info.items()):
        if part in (PerfParts.TOTAL, PerfParts.MEMORY): continue
        if part in (PerfParts.FW_COMPUTE,
                   PerfParts.BW_COMPUTE,
                   PerfParts.RECOMPUTE):
            ratio = compute_ratio
        else:
            ratio = coeffs.get(part.name)
        new_val = 0.0 if raw == 0.0 else raw * ratio
        debugger.info[part] = new_val

    max_idx = max(p.value for p in PerfParts) -1
    estimations = [0.0] * max_idx
    for part in PerfParts:
        if part in (PerfParts.TOTAL, PerfParts.MEMORY): continue
        estimations[part.value - 1] = debugger.info.get(part) or 0.0

    real_buckets = {rp: [] for rp in RealParts}
    real_buckets = estimation_in_real_parts(real_buckets, estimations, old_perf)

    perf = (
            real_buckets[RealParts.COMP][-1]
            + real_buckets[RealParts.DP_WAIT][-1]
            + real_buckets[RealParts.MP_WAIT][-1]
            + real_buckets[RealParts.EP_WAIT][-1]
            + real_buckets[RealParts.CP_WAIT][-1]
            + real_buckets[RealParts.PP_WAIT][-1]
    )
    debugger.info[PerfParts.TOTAL] = perf
    return perf


# performance estimation
def estimate_performance(*args, **kwargs):
    """main estimation"""
    stages = kwargs.get("stages", args[1] if len(args) > 1 else None)
    extra_custom_func = kwargs.get(
        "extra_custom_func", args[2] if len(args) > 2 else None
    )
    ccfg = kwargs.get("ccfg", args[3] if len(args) > 3 else CustomConfig())
    debugger = kwargs.get("debugger", args[4] if len(args) > 4 else None)
    device_type = kwargs.get(
        "device_type", args[5] if len(args) > 5 else Hard.device_map["A2"]
    )
    memory = kwargs.get("memory", args[6] if len(args) > 6 else None)

    # cfg = CostModelConfig(args[0])
    if isinstance(args[0], CostModelConfig):
        cfg = args[0]
    else:
        cfg = CostModelConfig(args[0])

    # Process custom model config
    if extra_custom_func:
        extra_custom_func(cfg)
    else:
        logger.info("auto applying custom model config")
        check_and_apply_custom_hook(cfg)

    # Process partition generation
    if not stages:
        logger.info("stage partitions are generated")
        stages = cfg.generate_partitions_vpp()

    # print(f"DP = {cfg.d}; MP = {cfg.t}; EP = {cfg.ep}; PP = {cfg.p}")

    # print(list(map(list,
    # zip(*list(map(lambda x: list(map(len, x)),stages))))))
    # print(stages)

    cfg.n = cfg.d * cfg.t * cfg.p
    cfg.n_headCast = 1
    cfg.n_ffAct = 1

    logger.debug(
        "perf_model: DP = %d, TP = %d, EP = %d, PP = %d, MB = %d",
        cfg.d,
        cfg.t,
        cfg.ep,
        cfg.p,
        cfg.m,
    )

    logger.info(str(cfg))
    logger.info(stages)
    logger.info(ccfg)

    compute_perfs = estimate_comp(
        cfg, ccfg, stages, with_recomp=False, debugger=debugger
    )
    recompute_perfs = (
        [0] * cfg.p
        if ccfg.retype not in {RecType.COMPUTE_ONLY, RecType.WITH}
        else estimate_comp(
            cfg, ccfg, stages, with_recomp=True, debugger=debugger
        )
    )
    comm_perfs = estimate_comm(
        cfg, ccfg, stages, device_type, with_recomp=False, debugger=debugger
    )
    logger.info("PerfEst: comm_perfs %s", comm_perfs)
    recomm_perfs = (
        [0] * cfg.p
        if ccfg.retype not in {RecType.COMM_ONLY, RecType.WITH}
        else estimate_comm(
            cfg, ccfg, stages, device_type, with_recomp=True, debugger=debugger
        )
    )

    stage_perfs = estimate_stage(
        cfg,
        ccfg,
        compute_perfs,
        comm_perfs,
        recompute_perfs,
        recomm_perfs,
        debugger=debugger,
    )
    logger.info("PerfEst: stage_perfs %s", stage_perfs)

    stage_focused = kwargs.get("stage_focused", None)
    perf = estimate_perf(
        cfg, ccfg, stage_perfs, stage_focused=stage_focused, debugger=debugger
    )
    perf += estimate_p2p(cfg, ccfg, stage_perfs, debugger=debugger)
    logger.info("PerfEst: perf %s", perf)

    cache_file = kwargs.get("cache_file")
    coeffs = None
    cache = False
    if cache_file is not None:
        with open(cache_file, 'r', encoding='utf-8') as f:
            coeffs = json.load(f)
        cache = True

    if debugger and debugger.is_enabled():
        if cache:
            perf = apply_regression_coefficients(coeffs, debugger, perf)
        debugger.info[PerfParts.TOTAL] = perf
        if memory is not None:
            debugger.info[PerfParts.MEMORY] = memory
    return perf  # / cfg.gbs

# TO-DO
# Fix More Memory
# Add Context Parallelism
# Fix PerformanceType.TIME

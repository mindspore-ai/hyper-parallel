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
"""performance estimation"""
import functools
import json
import math
import os
from copy import deepcopy
from dataclasses import replace
from types import MappingProxyType
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple
import numpy as np

from hyper_parallel.auto_parallel.sapp_nd.nd.logger import logger as nd_logger
from hyper_parallel.auto_parallel.sapp_nd.nd.logger import perf_logger as logger
from hyper_parallel.auto_parallel.sapp_nd.nd.common.config import Config
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType
from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_config import CostModelConfig
from hyper_parallel.auto_parallel._layer_stack import LayerStack, StackGroup
from hyper_parallel.auto_parallel._op_profiles import LayerKind
from hyper_parallel.auto_parallel.sapp_nd.nd.common.arch_hooks import CWrap, check_and_apply_custom_hook
import hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware as Hard
from hyper_parallel.auto_parallel.sapp_nd.nd.debug import Debug, PerfParts, RealParts, estimation_in_real_parts

from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.utils_classes import (
    RatioType,
    PerformanceType,
    P2PCommType,
    RecType,
    CustomConfig,
)
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.comm_time import estimate_comm
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.getters import (
    get_layer_configs_by_position,
    get_layer_switches_by_position,
    get_table_quantity,
    mla_weights,
    selective_shares,
)


GENERALIZE_PIPELINE_CALCULATION = False
MANUAL_P2P_RATIO = 0.002
BACKWARD_RATIO = 2
# The chunk the gated delta rule's kernel and Transformers' own
# implementation run a sequence in.
GDN_CHUNK = 64
# What a MoE layer's dispatch costs a rank, per token, per hidden unit, per
# expert a token routes to and per rank holding experts.  The work is the
# sorting, permuting, zeroing and accumulating around the all-to-all, which
# no matmul count sees, and it is the one cost that measurably grows with the
# expert-parallel degree.  Over 15 profiled points of Qwen3.5-35B-A3B at 8192
# tokens on 64 dies, device compute averages 5.85, 6.19, 6.59, 7.58 and 9.43 s
# at EP 1, 2, 4, 8 and 16, a span of 61%, where this model without the term
# spans 8%.  The increments over EP 1 are 222, 217, 230 and 230 ms a unit of
# EP, so they follow EP itself rather than EP - 1: a layer sorts and permutes
# its tokens by expert even where it holds every one of them, and the work
# grows with the ranks it spreads them over.  With the term the estimate runs
# 3.8 to 9.5% above the measured compute at every one of those degrees, where
# it was 31% below at EP 16, and the rank correlation over the 15 points rises
# from +0.51 to +0.64.  It does not change which strategy the search picks:
# what holds that at EP 1 is the gather priced at 370 ms where 204 was
# measured, an overlap threshold rather than a compute term.
#
# EMPIRICAL, and fitted at one shape.  It follows the tokens a rank routes, so
# on the 128-token sweeps it adds 3.2 ms a unit of EP, where the measured
# compute is flat from EP 4 to EP 64, 196 to 210 ms, and this would add 205 ms
# by EP 64: at a shape whose step is launch bound, 53 to 67% of it idle,
# neither this term nor the compute model under it holds.  The kernel tables
# per EP at the training shape name what grows: the routing's sort and index
# kernels, launched as often at every degree above 1 and each running longer,
# a cost linear in the tokens the all-to-all hands a rank.  This coefficient
# stands in for that until it is priced from the tokens, so read it as a
# calibration carrying the measured shape, not as a model of the dispatch.  A
# cluster that has measured its own says so with ``context.moe_dispatch``, as
# it states the ratios of the parts in a file, and a model whose compute does
# not grow with EP states 0, which prices no dispatch: DeepSeek V4.1's crop
# measures 3.3% less compute at EP 16 than at EP 1, where ND with this term
# prices 60% more.
MOE_DISPATCH = 1040
# How far a stage's recorded parts may stray from its time: they are its time
# taken apart, so only by rounding.
PARTS_REL_TOL = 1e-9
# The ratio of a ratios file that scales the compute parts and turns ND's
# units into milliseconds.
COMPUTE_RATIO = "COMPUTE"
# Where a ratio a file leaves out is taken from, as ``nd.ratios.fit_ratios``
# fills a part it cannot fit: the all-reduce from the DP traffic it was split
# out of, the bubble from P2P time, which measured data cannot tell it from,
# and any other part from compute's, which leaves it as ND weighs it.
RATIO_FALLBACKS = {
    PerfParts.DP_REDUCE.name: PerfParts.DP_COMM.name,
    PerfParts.BUBBLE.name: PerfParts.PP_COMM.name,
}
# The ratios a file was found without, each named once a process.
_MISSING_RATIOS = set()
# The parts a communication walk records on the debugger, one value a stage.
RECORDED_COMM_PARTS = (
    PerfParts.DP_COMM,
    PerfParts.DP_REDUCE,
    PerfParts.MP_COMM,
    PerfParts.EP_COMM,
    PerfParts.CP_COMM,
)


def op_table(cfg, attn=None):
    """op compute load formulas

    Args:
        cfg: the cost-model config.
        attn: a layer group whose attention flavour differs from *cfg*'s.
            Only the attention entries read it; everything else stays on
            *cfg*, so the expert scaling applied by the caller is unaffected.
    """
    table = {}
    # cfg.s /= cfg.cp
    # Q,O cost h x (a*dh) each, K,V cost h x (n_kv*dh); the h*h form is the
    # head_dim = h/a special case. A fused output gate doubles Q.
    att = attn if attn is not None else cfg
    d_h = att.dh or (cfg.h / att.a if att.a else 0)
    d_q = att.a * d_h
    table["n_attMM"] = (
        1.5
        * cfg.b
        * cfg.s
        * cfg.h
        * (
            ((2 if getattr(att, "attn_output_gate", False) else 1) + 1) * d_q
            + 2 * att.n_kv * d_h
        )
    )
    # The chunked gated delta rule, per value head: within a chunk of
    # GDN_CHUNK tokens, its keys against its keys and its queries, its
    # solved weights against its values and its decayed keys, and its
    # scores against its new values; the state read twice and written once
    # a chunk.  Linear in the sequence, where an attention score is
    # quadratic.  The entry exists only for a group that declares the
    # flavour, so no other config needs to carry the count.
    n_v, d_k, d_v = (getattr(att, name, 0) for name in ("lin_n_v", "lin_d_k", "lin_d_v"))
    if n_v * d_k * d_v:
        table["n_linrec"] = (3 * cfg.b * cfg.s * n_v
                             * (2 * GDN_CHUNK * (3 * d_k + 2 * d_v) + 6 * d_k * d_v))
    table["n_ffMM"] = 6 * cfg.b * cfg.s * cfg.h * cfg.hff
    # Every head's queries against every key, as wide as a head's queries
    # and keys, an MLA head's with its rotary part; then the weights
    # against the values, as wide as a head's values.
    d_qk = (getattr(att, "qk_nope_head_dim", None) or d_h) + (getattr(att, "dhr", 0) or 0)
    table["n_attBMM"] = 3 * cfg.b * cfg.s * cfg.s * att.a * (d_qk + d_h)
    table["n_softmax"] = 13 * cfg.a * cfg.b * cfg.s * cfg.s
    table["n_headCast"] = 3 * cfg.a * cfg.b * cfg.s * cfg.s
    table["n_gather"] = cfg.b * cfg.s * cfg.h * (cfg.t - 1)
    table["n_ffAct"] = 21 * cfg.b * cfg.s * cfg.hff

    table["n_normOp"] = 30 * cfg.b * cfg.s * cfg.h * cfg.t / cfg.sp
    # A QK-norm runs over every head's queries and keys, which TP splits.
    # The entry exists only for a model that has one.
    if getattr(cfg, "n_qknorm", 0):
        table["n_qknorm"] = 30 * cfg.b * cfg.s * (att.a + att.n_kv) * d_h
    table["n_dropout"] = (
        3 * cfg.b * cfg.s * max(cfg.a * cfg.s, 3 * cfg.h * cfg.t / cfg.sp)
    )
    if cfg.dc_kv != 0:  # MLA
        weights, _ = mla_weights(cfg)
        table["n_attMM"] = 6 * cfg.b * cfg.s * weights / (getattr(cfg, "n_attMM", 0) or 4)
    for op in table:
        table[op] *= cfg.bytes_p / cfg.t / cfg.cp
    # cfg.s *= cfg.cp
    return table


def _flavour_tables(cfg, attn=None):
    """One (dense, expert) table pair for a given attention flavour.

    A MoE layer's feed-forward entry prices the experts each token runs,
    the shared experts, as wide as a routed one, and, spread over the
    layer's feed-forward matmuls, the router and the shared experts' gate;
    its activation function's entry the same experts, which run it at
    their width, where a dense layer's runs it at the dense width.  A MoE
    layer also dispatches its tokens, which the table prices apart
    (:data:`MOE_DISPATCH`, or the cost the run states, 0 pricing none); a
    dense layer does not, so only the expert table carries that entry.
    """
    base = op_table(cfg, attn)
    exp = deepcopy(base)  # Verify this with MF MoEV2
    n_ff = getattr(attn if attn is not None else cfg, "n_ffMM", 0) or 3
    gate = 1 if cfg.n_shared_exp and getattr(cfg, "shared_expert_gate", None) else 0
    experts = cfg.hff_exp * (max(1, cfg.n_chosen_exp) * cfg.cap_fact + cfg.n_shared_exp)
    width = experts + (cfg.n_exp + gate) / n_ff
    exp["n_ffMM"] *= width / cfg.hff
    exp["n_ffAct"] *= experts / cfg.hff
    dispatch = getattr(cfg, "moe_dispatch", None)
    if dispatch is None:
        dispatch = MOE_DISPATCH
    exp["n_dispatch"] = (
        dispatch * cfg.b * cfg.s * cfg.h * max(1, cfg.n_chosen_exp)
        * max(1, getattr(cfg, "ep", 1) or 1) * cfg.bytes_p / cfg.t / cfg.cp
    )
    return base, exp


def _regular_layer_flop(cfg, ccfg, lcfg, tables, layer, with_recomp, switches=None):
    """Price one regular layer with its group's config, and its own recompute *switches* where it has them.

    *tables* holds one table pair per attention flavour and gains one the
    first time a flavour is priced.
    """
    kind = getattr(lcfg, "attn_kind", "full")
    if kind not in tables:
        tables[kind] = _flavour_tables(cfg, lcfg)
    flop = get_table_quantity(
        lcfg,
        tables[kind][1] if (lcfg.n_exp > 1) else tables[kind][0],
        layer,
        with_recomp,
        shares=selective_shares(lcfg, layer, switches),
        switches=switches,
    )
    if ccfg.ttype == PerformanceType.TIME:
        flop = estimate_comp_flop_time(lcfg, flop)
    return flop


# Evaluation functions
def estimate_op_bulk_comp(cfg, ccfg, stages, with_recomp=False, debugger=None, both=False):
    """FW + BW; with ``both``, ``(plain, recomputed)`` from one walk over the layers"""
    _ = debugger
    # Full attention prices from the model's own attention dimensions.
    tables = {"full": _flavour_tables(cfg)}
    lccfg_at = get_layer_configs_by_position(cfg, stages)
    # Each layer's own recompute switches, where the config states several.
    switches_at = get_layer_switches_by_position(cfg, stages)

    # One sum a mode, each taking its terms in the order a walk of that mode alone did.
    modes = (False, True) if both else (with_recomp,)
    sums = tuple([] for _ in modes)
    for stage_id, stage in enumerate(stages):
        for flops in sums:
            flops += [0]
        for chunk_id, chunk in enumerate(stage):
            for lay_id, layer in enumerate(chunk):
                if layer == LayerType.EMBEDDING_LAYER:
                    continue

                if layer == LayerType.OUTPUT_LAYER:
                    output = (1 if cfg.dc_kv == 0 else cfg.n_mtp) * (
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
                    for flops in sums:
                        flops[-1] += output
                    continue

                position = (stage_id, chunk_id, lay_id)
                for flops, mode in zip(sums, modes):
                    flops[-1] += _regular_layer_flop(
                        cfg, ccfg, lccfg_at[position], tables, layer, mode, switches_at.get(position)
                    )

    return sums if both else sums[0]


def estimate_comp(cfg, ccfg, stages, with_recomp=False, debugger=None, both=False):
    """wrapper"""
    return estimate_op_bulk_comp(
        cfg, ccfg, stages, with_recomp, debugger=debugger, both=both
    )


# Experimental : Flop time


def efficiency(x):
    """obtained via extrapolation"""
    eff = min(
        1.0, max(0.1, 0.00004694 * x**3 + 0.0014 * x**2 - 0.0336 * x + 0.1)
    )
    return eff


def throughput(precision_bytes, flop):
    """assumes matrix

    Only PerformanceType.TIME reaches this, and nothing in production asks
    for it: the live estimate is a relative score whose scale the fitted
    ratios own.  Its precision_bytes**2 * 1e12 is no device's rate,
    4 TFLOP/s at bf16, and no other rate prices a candidate's parts: the
    one ND states, a host link's sustained_tflops, only turns an offload
    copy's seconds into the score's units where no fitted ratio does.
    """
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


def _weigh_comm_parts(debugger, comm_w):
    """Weigh each stage's recorded communication parts as the stage time weighs them.

    DP_REDUCE, the all-reduce's share of DP_COMM, is weighed where the
    communication walk recorded it (``estimate_from_mem_comm``).
    """
    for p in [PerfParts.DP_COMM, PerfParts.MP_COMM, PerfParts.EP_COMM, PerfParts.CP_COMM]:
        debugger.info[p] = [
            comm_w * c for c in debugger.info[p]
        ]
    if isinstance(debugger.info[PerfParts.DP_REDUCE], list):
        debugger.info[PerfParts.DP_REDUCE] = [comm_w * c for c in debugger.info[PerfParts.DP_REDUCE]]


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
        _weigh_comm_parts(debugger, comm_w)
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


def _check_stage_parts(time_sum, stage_time, stage):
    """Say so when the parts recorded for a stage do not add up to its time.

    The parts are the stage's time taken apart, so they agree with it to
    rounding.  A gap means a walk recorded less, or more, than the time
    counts, and the bubble, what is left of the pipeline's time once the
    parts are taken out, silently takes the difference.
    """
    if math.isclose(time_sum, stage_time, rel_tol=PARTS_REL_TOL, abs_tol=PARTS_REL_TOL):
        return
    nd_logger.error(
        "the parts recorded for stage %d sum to %.6E where its time is %.6E (%+.3f%%), "
        "and the bubble takes the difference",
        stage,
        time_sum,
        stage_time,
        100 * (time_sum - stage_time) / stage_time if stage_time else 0.0,
    )


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
        if len(stage_perfs) != 1:
            raise ValueError("Expected exactly one stage performance")
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

        # The bubble stays the pipeline's time less the parts, so the parts
        # and the bubble still sum to the total the ratios are fitted on.
        _check_stage_parts(time_sum, stage_perfs[last_straggler_idx] * cfg.m, last_straggler_idx)
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
        # DP_COMM timed the stage's whole DP traffic, the time and the bubble
        # above with it; its all-reduce is DP_REDUCE's, the rest FSDP's.
        reduced = debugger.info[PerfParts.DP_REDUCE]
        if isinstance(reduced, list):
            debugger.info[PerfParts.DP_REDUCE] = reduced[last_straggler_idx] * cfg.m
            debugger.info[PerfParts.DP_COMM] -= debugger.info[PerfParts.DP_REDUCE]
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


def _one_layer_of(stack: Optional[LayerStack], kind: Optional[LayerKind]) -> Optional[LayerStack]:
    """The model's stack, reduced to one layer of *kind*.

    Its other groups stay, with no layer, so the estimates apply the layer's
    kind exactly when they would for the whole model: to a copy of the config
    for the compute, whose op tables come from the model's own config, and in
    place for the communication. A layer with no kind has no stack.
    """
    if kind is None or stack is None:
        return None
    return replace(stack, groups=(StackGroup(kind, 1),) + tuple(replace(group, count=0) for group in stack.groups))


class PlainTimes(NamedTuple):
    """What a layer costs before any recompute, the same for every option of its kind.

    Attributes:
        compute: The compute estimate without recompute.
        comm: The communication estimate without recompute.
        config: The config the communication estimate leaves, which the stage
            estimate reads.
    """

    compute: float
    comm: float
    config: CostModelConfig


def _layer_alone(
    cfg: CostModelConfig, kind: Optional[LayerKind], layer_type: LayerType, switches: Optional[Dict[str, int]]
) -> Tuple[CostModelConfig, list]:
    """One *layer_type* layer of *kind*, alone in the first stage of a fresh copy of *cfg*.

    A fresh copy each time: the estimates leave state on the config they are
    given.
    """
    single = deepcopy(cfg)
    if switches is not None:
        single.rec_op = Config(dict(switches))
    # The one layer runs *switches*, else the config's rec_op, whatever
    # the layers of the model run.
    single.layer_switches = None
    single.layer_stack = _one_layer_of(single.layer_stack, kind)
    single.n = single.d * single.t * single.p
    stages = [[[] for _ in range(single.vp)] for _ in range(single.p)]
    stages[0][0] = [layer_type]
    return single, stages


def plain_layer_times(
    cfg: CostModelConfig,
    kind: Optional[LayerKind],
    layer_type: LayerType,
    device_type: Any,
    ccfg: Optional[CustomConfig] = None,
) -> PlainTimes:
    """The compute and communication of one layer of *kind* alone, without recompute.

    They do not depend on which recompute option the layer runs, so every
    option of a kind can share them; the embedding and the output layer have
    their own.
    """
    ccfg = ccfg if ccfg is not None else CustomConfig()
    single, stages = _layer_alone(cfg, kind, layer_type, None)
    compute = estimate_comp(single, ccfg, stages)[0]
    grouped, stages = _layer_alone(cfg, kind, layer_type, None)
    comm = estimate_comm(grouped, ccfg, stages, device_type)[0]
    return PlainTimes(compute, comm, grouped)


def estimate_layer_times(
    cfg: CostModelConfig,
    kind: Optional[LayerKind],
    layer_type: LayerType,
    device_type: Any,
    ccfg: Optional[CustomConfig] = None,
    switches: Optional[Dict[str, int]] = None,
    plain: Optional[PlainTimes] = None,
) -> Tuple[float, float]:
    """Forward and backward time of one layer, priced as a stage prices it.

    Runs the compute, communication and stage estimates of
    :func:`estimate_performance` on one layer of *kind*, alone in the first
    stage of a copy of *cfg*. For a stage whose layers are all of one kind,
    the sum of their times is what the search charges the stage before
    pipeline bubbles and point-to-point communication. The search prices each
    layer on a config the previous layers' kinds have been applied to, so a
    stage that mixes kinds, or holds the embedding or output layer, can
    differ by a few percent; here every layer is priced on its own kind,
    wherever it sits.

    Args:
        cfg: The model's cost-model config, before any layer kind is applied.
        kind: The layer's kind; ``None`` for the embedding and output, and
            for a layer priced on the config as it stands.
        layer_type: How the layer runs: plain, selective or full recompute,
            embedding or output.
        device_type: The device the communication is priced on.
        ccfg: Estimator options; the search's defaults when omitted.
        switches: The recompute switches a selective layer runs with, 1 to
            keep an op and 0 to recompute it; the config's own when omitted.
        plain: The layer's :func:`plain_layer_times`, when already priced.

    Returns:
        ``(forward, backward)``, *backward* including the layer's recompute.
    """
    ccfg = ccfg if ccfg is not None else CustomConfig()
    if plain is None:
        plain = plain_layer_times(cfg, kind, layer_type, device_type, ccfg)
    comp, comm, grouped = plain
    # A layer that recomputes nothing runs nothing again.
    recomputes = layer_type in (LayerType.FULL_REC_LAYER, LayerType.SEL_REC_LAYER)
    recomp = comp
    if recomputes and ccfg.retype in {RecType.COMPUTE_ONLY, RecType.WITH}:
        single, stages = _layer_alone(cfg, kind, layer_type, switches)
        recomp = estimate_comp(single, ccfg, stages, with_recomp=True)[0]
    recomm = comm
    if recomputes and ccfg.retype in {RecType.COMM_ONLY, RecType.WITH}:
        fresh, fresh_stages = _layer_alone(cfg, kind, layer_type, switches)
        recomm = estimate_comm(fresh, ccfg, fresh_stages, device_type, with_recomp=True)[0]

    # The stage estimate weighs the config the communication estimate leaves.
    once = estimate_stage(grouped, ccfg, [comp], [comm], [comp], [comm])[0]
    total = estimate_stage(grouped, ccfg, [comp], [comm], [recomp], [recomm])[0]
    forward = once / (1 + BACKWARD_RATIO)
    return forward, total - forward


class LayerTimes:
    """Prices the layers of a pipeline balancer's layer description.

    The memory backbone calls it for every layer it describes, with the config
    it walks and the layer's kind. A config is copied the first time it is
    seen, at its embedding layer, before the walk applies any layer kind to
    it, and each kind, layer type and set of switches is priced once. What a
    kind costs without recompute is priced once for all its options.
    """

    def __init__(self, device_type: Any, ccfg: Optional[CustomConfig] = None) -> None:
        """Price on *device_type*, with the estimator options *ccfg*."""
        self._device_type = device_type
        self._ccfg = ccfg
        self._base = {}
        self._times = {}
        self._plain = {}

    def __call__(
        self,
        cfg: CostModelConfig,
        kind: Optional[LayerKind],
        layer_type: LayerType,
        switches: Optional[Dict[str, int]] = None,
    ) -> Tuple[float, float]:
        """``(forward, backward)`` of a *layer_type* layer of *kind*, run with *switches*."""
        if id(cfg) not in self._base:
            self._base[id(cfg)] = deepcopy(cfg)
        key = (id(cfg), kind, layer_type, None if switches is None else tuple(sorted(switches.items())))
        if key not in self._times:
            # Every body option of a kind shares its plain times; the
            # embedding and the output layer have their own.
            role = layer_type if layer_type in (LayerType.EMBEDDING_LAYER, LayerType.OUTPUT_LAYER) else None
            plain_key = (id(cfg), kind, role)
            if plain_key not in self._plain:
                self._plain[plain_key] = plain_layer_times(
                    self._base[id(cfg)], kind, layer_type, self._device_type, self._ccfg
                )
            self._times[key] = estimate_layer_times(
                self._base[id(cfg)], kind, layer_type, self._device_type, self._ccfg, switches,
                plain=self._plain[plain_key],
            )
        return self._times[key]


def _part_ratio(coeffs, key):
    """The ratio a ratios file states for *key*, or the one fit_ratios would give it.

    A file ``fit_ratios`` writes states every key, but one edited by hand,
    or fitted by a tree older than a part, may lack some.  A missing ratio
    falls back as ``fit_ratios`` fills a part it cannot fit (``RATIO_FALLBACKS``),
    and each missing key is named once.  Compute's ratio is the unit every
    other one falls back on, so a file without it is refused.

    Raises:
        ValueError: When the file states no COMPUTE ratio.
    """
    ratio = coeffs.get(key)
    if ratio is not None:
        return ratio
    if key == COMPUTE_RATIO:
        raise ValueError(
            "the ratios file states no COMPUTE ratio, the one that turns ND's "
            "units into milliseconds and that every missing ratio falls back on"
        )
    fallback = RATIO_FALLBACKS.get(key, COMPUTE_RATIO)
    if key not in _MISSING_RATIOS:
        _MISSING_RATIOS.add(key)
        nd_logger.error("the ratios file states no %s ratio: that part takes %s's", key, fallback)
    return _part_ratio(coeffs, fallback)


def apply_regression_coefficients(coeffs, debugger, old_perf):
    """
    applies the coefficients present in regression's cache_file
    """
    compute_ratio = _part_ratio(coeffs, COMPUTE_RATIO)
    for part, raw in list(debugger.info.items()):
        if part in (PerfParts.TOTAL, PerfParts.MEMORY):
            continue
        if part in (PerfParts.FW_COMPUTE,
                   PerfParts.BW_COMPUTE,
                   PerfParts.RECOMPUTE):
            ratio = compute_ratio
        else:
            ratio = _part_ratio(coeffs, part.name)
        new_val = 0.0 if raw == 0.0 else raw * ratio
        debugger.info[part] = new_val

    max_idx = max(p.value for p in PerfParts) -1
    estimations = [0.0] * max_idx
    for part in PerfParts:
        if part in (PerfParts.TOTAL, PerfParts.MEMORY):
            continue
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


class _EstimateArgs(NamedTuple):
    """Resolved inputs of estimate_performance, unpackable as a tuple."""

    cfg: CostModelConfig
    stages: Optional[list]
    extra_custom_func: Optional[Callable]
    ccfg: Any
    debugger: Any
    device_type: Any
    memory: Any


def _resolve_estimate_args(args, kwargs):
    """Resolve positional/keyword inputs for estimate_performance.

    Returns:
        _EstimateArgs of (cfg, stages, extra_custom_func, ccfg, debugger,
        device_type, memory).
    """
    cfg_input = args[0]
    # A copy: the estimate applies layer kinds to its config in place, and the
    # caller's config is the one the next estimate starts from.
    cfg = (
        deepcopy(cfg_input)
        if isinstance(cfg_input, CostModelConfig)
        else CostModelConfig(cfg_input)
    )
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
    return _EstimateArgs(cfg, stages, extra_custom_func, ccfg, debugger, device_type, memory)


def _save_recompute(stage_perfs, savings, debugger):
    """Stage times less the recompute time each stage saves.

    Args:
        stage_perfs: Each stage's time, its layers recomputed as the config
            says.
        savings: The time each stage's layers save with the recompute
            options chosen for them instead; ``None`` when none are.
        debugger: Records the recompute part of each stage, which is where
            the time is saved.

    Returns:
        Each stage's time.
    """
    if savings is None:
        return stage_perfs
    if len(savings) != len(stage_perfs):
        raise ValueError(f"{len(savings)} stage savings for {len(stage_perfs)} stages")
    if debugger and debugger.is_enabled():
        debugger.info[PerfParts.RECOMPUTE] = [
            recompute - saved for recompute, saved in zip(debugger.info[PerfParts.RECOMPUTE], savings)
        ]
    return [perf - saved for perf, saved in zip(stage_perfs, savings)]


def _user_hook(cfg: Any) -> Optional[Callable]:
    """The hook a unimodal config's hook class gives it, as the memory estimate applies it."""
    hooks = getattr(cfg, "hooks_dict", None)
    if not hooks or getattr(cfg, "multimodal", False):
        return None
    return next(iter(hooks.values()))


def _ratios(cache_file):
    """The ratios *cache_file* states, parsed again only when the file changes.

    A search applies them to every candidate it orders: read and parsed for
    each one, they cost 0.7 s on the Qwen3.5 example's 7216 candidates.
    """
    stat = os.stat(cache_file)
    return _parsed_ratios(cache_file, stat.st_mtime_ns, stat.st_size)


@functools.lru_cache(maxsize=8)
def _parsed_ratios(cache_file, mtime_ns, size):  # pylint: disable=unused-argument
    """The ratios file as one read-only mapping; *mtime_ns* and *size* only key the cache."""
    with open(cache_file, 'r', encoding='utf-8') as f:
        return MappingProxyType(json.load(f))


def _finalize_perf(perf, cache_file, debugger, memory):
    """Apply cached regression coefficients and record debug info.

    Returns:
        The final performance value.
    """
    coeffs = None
    cache = False
    if cache_file is not None:
        coeffs = _ratios(cache_file)
        cache = True

    if debugger and debugger.is_enabled():
        if cache:
            perf = apply_regression_coefficients(coeffs, debugger, perf)
        debugger.info[PerfParts.TOTAL] = perf
        if memory is not None:
            debugger.info[PerfParts.MEMORY] = memory
    return perf


def _stage_parts(cfg, ccfg, stages, device_type, debugger):
    """One config's per-stage compute, recompute, communication and recomputed communication."""
    # Only the plain walks record parts. A stage's time takes what the
    # recompute walks add as RECOMPUTE alone; recorded as communication too,
    # it made the parts outgrow the time, and the bubble, their difference,
    # went negative.
    # Where the recompute is priced, one walk prices both modes, each sum
    # taking its terms in the order a walk of that mode alone did.
    if ccfg.retype in {RecType.COMPUTE_ONLY, RecType.WITH}:
        compute_perfs, recompute_perfs = estimate_comp(cfg, ccfg, stages, debugger=debugger, both=True)
    else:
        compute_perfs = estimate_comp(cfg, ccfg, stages, with_recomp=False, debugger=debugger)
        recompute_perfs = [0] * cfg.p
    if ccfg.retype in {RecType.COMM_ONLY, RecType.WITH}:
        comm_perfs, recomm_perfs = estimate_comm(cfg, ccfg, stages, device_type, debugger=debugger, both=True)
    else:
        comm_perfs = estimate_comm(cfg, ccfg, stages, device_type, with_recomp=False, debugger=debugger)
        recomm_perfs = [0] * cfg.p
    return compute_perfs, recompute_perfs, comm_perfs, recomm_perfs


def _submodule_parts(cfg, ccfg, device_type, debugger):
    """A multimodal model's per-stage parts, every submodule's summed.

    The submodules of a vision-language model run on one pipeline, so a
    stage's time is the work of the layers on it, whichever submodule they
    belong to, as the memory model sums what they keep there
    (``combine_partition_multimodal``).  Each is priced on its own
    partitions and under its own family, on a copy, so no submodule leaves
    state for the next.

    Each submodule's walk also records its communication on the debugger,
    stage by stage, and the record is summed the same way.  Kept as the
    last walk left it, it held the language model's share alone: a vision
    tower's collectives went into the bubble, the time less the parts, and
    the parts a ratio is fitted on missed them.
    """
    partitions = cfg.generate_partitions_vpp()
    totals = None
    recorded = {}
    for name in cfg.mm_order:
        sub = deepcopy(cfg.mm_ccfgs[name])
        check_and_apply_custom_hook(sub)
        parts = _stage_parts(sub, ccfg, partitions[name], device_type, debugger)
        totals = parts if totals is None else tuple(
            [left + right for left, right in zip(*pair)] for pair in zip(totals, parts)
        )
        if debugger and debugger.is_enabled():
            _add_recorded_comm(recorded, debugger.info)
    if recorded:
        debugger.info.update(recorded)
    return totals


def _add_recorded_comm(recorded, info):
    """Add the communication a walk recorded, one value a stage, to *recorded*."""
    for part in RECORDED_COMM_PARTS:
        stages = info.get(part)
        if not isinstance(stages, list):
            continue
        recorded[part] = (
            [left + right for left, right in zip(recorded[part], stages)]
            if part in recorded
            else list(stages)
        )


# performance estimation
def estimate_performance(*args, **kwargs):
    """main estimation

    With ``stage_savings``, each stage's time is lowered by the recompute
    time its layers save under the options chosen for them, before the
    pipeline is timed.
    """
    (
        cfg,
        stages,
        extra_custom_func,
        ccfg,
        debugger,
        device_type,
        memory,
    ) = _resolve_estimate_args(args, kwargs)
    cache_file = kwargs.get("cache_file")
    if cache_file is not None and not (debugger and debugger.is_enabled()):
        # The ratios rescale the parts a debugger records, so the parts are
        # recorded whether or not anyone asked to see them: a search that
        # writes no debug output must not rank on the uncalibrated score.
        debugger = Debug(None, info_type=PerfParts)

    # Process custom model config: the config's own hook when its hook class
    # gives it one, as the memory estimate applies it, else its family.
    user_hook = _user_hook(cfg)
    if extra_custom_func:
        extra_custom_func(cfg)
    elif user_hook is not None:
        user_hook(CWrap(cfg))
    else:
        logger.info("auto applying custom model config")
        check_and_apply_custom_hook(cfg)

    # Process partition generation
    if not stages:
        logger.info("stage partitions are generated")
        stages = cfg.generate_partitions_vpp()

    cfg.n = cfg.d * cfg.t * cfg.p

    logger.debug(
        "perf_model: DP = %d, TP = %d, EP = %d, PP = %d, MB = %d",
        cfg.d,
        cfg.t,
        cfg.ep,
        cfg.p,
        cfg.m,
    )

    logger.info("%s", cfg)
    logger.info(stages)
    logger.info(ccfg)

    if getattr(cfg, "multimodal", False):
        compute_perfs, recompute_perfs, comm_perfs, recomm_perfs = _submodule_parts(
            cfg, ccfg, device_type, debugger
        )
        timed = cfg.mm_ccfgs[getattr(cfg, "mm_main", None) or cfg.mm_order[-1]]
        timed.n = cfg.n
    else:
        compute_perfs, recompute_perfs, comm_perfs, recomm_perfs = _stage_parts(
            cfg, ccfg, stages, device_type, debugger
        )
        timed = cfg
    logger.info("PerfEst: comm_perfs %s", comm_perfs)

    stage_perfs = estimate_stage(
        timed,
        ccfg,
        compute_perfs,
        comm_perfs,
        recompute_perfs,
        recomm_perfs,
        debugger=debugger,
    )
    stage_perfs = _save_recompute(stage_perfs, kwargs.get("stage_savings"), debugger)
    logger.info("PerfEst: stage_perfs %s", stage_perfs)

    stage_focused = kwargs.get("stage_focused", None)
    perf = estimate_perf(
        timed, ccfg, stage_perfs, stage_focused=stage_focused, debugger=debugger
    )
    perf += estimate_p2p(timed, ccfg, stage_perfs, debugger=debugger)
    logger.info("PerfEst: perf %s", perf)

    return _finalize_perf(perf, cache_file, debugger, memory)  # / cfg.gbs

# TO-DO
# Fix More Memory
# Add Context Parallelism

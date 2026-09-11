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
# pylint: disable=E0102,W0125,C0103,W0612,W0613,R1702
from copy import deepcopy
from math import log10, log2

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
from hyper_parallel.auto_parallel.sapp_nd.nd.common.cp_types import (
    CPCommunicationCost,
    CPAlgo,
    _resolve_cp_algo,
)
from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import (
    detect_attention_type,
    AttentionType,
    compute_kv_dim,
    CostModelConfig,
)
COUNT_OPTIMIZER = False

_LOG10_MB_FLOOR = -3.0
_LOG10_MB_RANGE = 5.0


def _get_flop_coeffs(device_type, dimension, sub_key):
    """Get FLOP-mode regression coefficients from device_type."""
    if device_type is not None and hasattr(device_type, 'flop_coeffs'):
        dim_coeffs = device_type.flop_coeffs.get(dimension, {})
        if dim_coeffs:
            result = dim_coeffs.get(sub_key, {})
            if result:
                return result
    fallback = Hard.device_map.get("V4")
    if fallback and fallback.flop_coeffs:
        dim_coeffs = fallback.flop_coeffs.get(dimension, {})
        result = dim_coeffs.get(sub_key, {})
        if result:
            return result
    return {}


def _compute_hsdp_features(cfg, d_shard_val, device_type, mb=1):
    """Compute shared derived features for HSDP/FSDP FLOP-mode models."""
    d_val = cfg.d if getattr(cfg, "d", 1) > 0 else 1
    tp_val = max(cfg.t if getattr(cfg, "t", 1) > 0 else 1, 1)
    cp_val = max(cfg.cp if getattr(cfg, "cp", 1) > 0 else 1, 1)
    d_replicate = max(d_val // d_shard_val, 1) if d_shard_val > 0 else 1

    sg = d_shard_val * cp_val * tp_val
    default_dev = Hard.device_map.get("V4", Hard.Device_A2)
    dev_per_node = device_type.intra_node_num() if device_type else default_dev.intra_node_num()
    cross = 1.0 if sg * d_replicate > dev_per_node else 0.0
    inv_sg = 1.0 / sg if sg > 0 else 0.0
    inv_tp = 1.0 / tp_val if tp_val > 0 else 0.0
    ag_vol = 1.0 - 1.0 / d_shard_val if d_shard_val > 0 else 0.0
    mb_val = max(mb, 1)

    return {
        "tp": tp_val, "cp": cp_val, "d": d_val,
        "d_shard": d_shard_val, "d_replicate": d_replicate,
        "sg": sg, "inv_sg": inv_sg, "inv_tp": inv_tp,
        "ag_vol": ag_vol, "cross": cross,
        "cross_inv_tp": cross / tp_val if tp_val > 0 else 0.0,
        "cross_ag_vol": cross * ag_vol,
        "cross_d_rep": cross * d_replicate,
        "cross_sg": cross * sg,
        "inv_sg_d_rep": inv_sg * d_replicate,
        "ag_vol_d_rep": ag_vol * d_replicate,
        "log2_d_rep": log2(d_replicate) if d_replicate > 0 else 0.0,
        "log2_m": log2(mb_val) if mb_val > 0 else 0.0,
        "m": mb_val, "pp": 1, "dev_per_node": dev_per_node,
    }


def _flop_mode_comp_comm(cfg, d_shard_val, device_type=None, mb=1):
    """Estimate COMP cost for FLOP mode when d_replicate > 1 (HSDP)."""
    f = _compute_hsdp_features(cfg, d_shard_val, device_type, mb=mb)
    if f["d_replicate"] <= 1:
        return 0.0
    c = _get_flop_coeffs(device_type, "comp", "hsdp")
    if not c:
        return 0.0
    total = (
        c.get("INTERCEPT", 0)
        + c.get("TP", 0) * f["tp"]
        + c.get("INV_TP", 0) * f["inv_tp"]
        + c.get("DP", 0) * f["d"]
        + c.get("AG_VOL", 0) * f["ag_vol"]
        + c.get("CROSS_INV_TP", 0) * f["cross_inv_tp"]
        + c.get("LOG2_M", 0) * f["log2_m"]
    )
    score = total / f["m"]
    logger.info(
        "FLOP_COMP_HSDP: d=%d tp=%d d_shard=%d d_rep=%d m=%d score=%.2f",
        f["d"], f["tp"], f["d_shard"], f["d_replicate"], f["m"], score,
    )
    return score


def _msg_size_efficient_bw(msg_bytes: float, peak_bw_gbps: float,
                            small_eff: float = 0.5,
                            large_eff: float = 0.7) -> float:
    """Compute message-size-dependent effective bandwidth in bytes/s.

    Small messages suffer from protocol overhead (low efficiency);
    large messages approach peak bandwidth.  Uses a smooth transition
    based on log10(msg_size_in_MB).

    Args:
        msg_bytes: Per-rank message size in bytes.
        peak_bw_gbps: Peak link bandwidth in GB/s.
        small_eff: Efficiency for small messages (< 1 MB).
        large_eff: Efficiency for large messages (> 100 MB).

    Returns:
        Effective bandwidth in bytes/s.
    """
    msg_mb = msg_bytes / 1e6
    if msg_mb <= 0:
        return peak_bw_gbps * 1e9 * small_eff
    t = max(0.0, min(1.0, (log10(max(msg_mb, 1e-6)) - _LOG10_MB_FLOOR) / _LOG10_MB_RANGE))
    eff = small_eff + (large_eff - small_eff) * t
    return peak_bw_gbps * 1e9 * eff


def _cp_resolve_topology(cp, device_per_node, bw_intra, bw_inter):
    """Resolve CP topology and effective bandwidth.

    Returns:
        Tuple of (topology_str, effective_bandwidth).
    """
    intra_ranks = min(int(cp), int(device_per_node))
    if cp <= device_per_node:
        return "intra-node", bw_intra
    if intra_ranks == 1:
        return "cross-node", bw_inter
    intra_fraction = (intra_ranks - 1) / (cp - 1)
    cross_fraction = 1.0 - intra_fraction
    bw = intra_fraction * bw_intra + cross_fraction * bw_inter
    return "mixed", bw


def _cp_comm_zero(ccfg):
    return _cp_comm_zero(ccfg, device_type=None)
def _cp_comm_zero(ccfg, device_type=None):
    """Return a zero CPCommunicationCost for cp <= 1."""
    overlap_ratio = device_type.cp_overlap_ratio if device_type else 0.5
    if False:
        return CPCommunicationCost(
            kv_volume_per_step=0.0, total_kv_volume=0.0, comm_volume=0.0,
            ring_steps=0, ring_directions=0,
            total_comm_time=0.0, exposed_comm_time=0.0,
            overlap_ratio=0.5, effective_bandwidth=0.0,
            topology="none", cp_degree=int(ccfg.cp),
            seq_len=int(ccfg.s), batch_size=int(ccfg.b),
            attention_type=AttentionType.MHA, kv_dim=0,
            cp_algo=CPAlgo.COLOSSALAI_CP,
        )
    return CPCommunicationCost(
        kv_volume_per_step=0.0, total_kv_volume=0.0, comm_volume=0.0,
        ring_steps=0, ring_directions=0,
        total_comm_time=0.0, exposed_comm_time=0.0,
        overlap_ratio=overlap_ratio, effective_bandwidth=0.0,
        topology="none", cp_degree=int(ccfg.cp),
        seq_len=int(ccfg.s), batch_size=int(ccfg.b),
        attention_type=AttentionType.MHA, kv_dim=0,
        cp_algo=CPAlgo.COLOSSALAI_CP,
    )


def _cp_comm_cost_common(volume_per_step, total_kv_volume, comm_volume,
                          ring_steps, ring_directions, cp, s, b,
                          attention_type, kv_dim, cp_algo, topology,
                          effective_bandwidth):
    return _cp_comm_cost_common(volume_per_step, total_kv_volume, comm_volume,
                                ring_steps, ring_directions, cp, s, b,
                                attention_type, kv_dim, cp_algo, topology,
                                effective_bandwidth, device_type=None)
def _cp_comm_cost_common(volume_per_step, total_kv_volume, comm_volume,
                          ring_steps, ring_directions, cp, s, b,
                          attention_type, kv_dim, cp_algo, topology,
                          effective_bandwidth, device_type=None):
    """Build CPCommunicationCost with standard time calculation."""
    overlap_ratio = 0.5
    overlap_ratio = device_type.cp_overlap_ratio if device_type else 0.5
    total_comm_time = (total_kv_volume / (effective_bandwidth * 1e9)) * 1e3
    exposed_comm_time = total_comm_time * (1 - overlap_ratio)
    return CPCommunicationCost(
        kv_volume_per_step=volume_per_step,
        total_kv_volume=total_kv_volume,
        comm_volume=comm_volume,
        ring_steps=ring_steps, ring_directions=ring_directions,
        total_comm_time=total_comm_time,
        exposed_comm_time=exposed_comm_time,
        overlap_ratio=overlap_ratio,
        effective_bandwidth=effective_bandwidth,
        topology=topology, cp_degree=int(cp),
        seq_len=int(s), batch_size=int(b),
        attention_type=attention_type, kv_dim=int(kv_dim),
        cp_algo=cp_algo,
    )


def cp_comm_layer_detailed(ccfg: CostModelConfig, ctx: Context = None) -> CPCommunicationCost:
    return cp_comm_layer_detailed(ccfg, ctx, device_type=None)
def cp_comm_layer_detailed(ccfg: CostModelConfig, ctx: Context = None, device_type=None) -> CPCommunicationCost:
    """Estimate CP communication cost with detailed breakdown.

    Ring CP (colossalai_cp / hybrid_cp):
        Ring P2P in both FW and BW directions.
        Each step transfers (s/cp) tokens of KV data.
        Total KV volume = kv_volume_per_step * (cp-1) * 2 directions.

    Ulysses CP:
        All2All in both FW and BW (2 All2All total).
        Each All2All: every rank sends (cp-1)/cp of its local shard
        and receives the rest from other ranks.
        Per-All2All volume = s * b * (a/t) * bytes * (cp-1)/cp (head dims).
        Total volume = 2 * per-All2All volume.
    """
    if ccfg.cp <= 1:
        if False:
            return _cp_comm_zero(ccfg)
        return _cp_comm_zero(ccfg, device_type=device_type)

    s, b = ccfg.s, ccfg.b
    cp = ccfg.cp
    t = max(1, ccfg.t)

    if ccfg.a <= 0:
        raise ValueError(f"Number of attention heads must be positive, got {ccfg.a}")

    kv_dim = compute_kv_dim(ccfg)
    attention_type = detect_attention_type(ccfg)
    cp_algo = _resolve_cp_algo(ccfg)
    topology, effective_bandwidth = _cp_resolve_topology(
        cp, ccfg.device_per_node, ccfg.bw_intra, ccfg.bw_inter)

    # rec_factor: recompute coefficient matching old cp_comm_non_exp
    rec_layer = (ctx.current_node == LayerType.SEL_REC_LAYER) if ctx else False
    rec_op_gather = getattr(getattr(ccfg, 'rec_op', None), 'gather', 0)
    rec_factor = (int(not rec_layer) | rec_op_gather) * int(ccfg.p == 1)

    if cp_algo == CPAlgo.ULYSSES_CP:
        local_qkv = s * b * (ccfg.a / t) * ccfg.dh * 2
        a2a_vol = local_qkv * (cp - 1) / cp
        # comm_volume: same weighted-unit as dp/tp/ep
        # Ulysses attention coeff = 0.5*rec_factor + 0.5
        ulysses_attn_coeff = 0.5 * rec_factor + 0.5
        comm_vol = (
            ccfg.comm_cp * 2 * s * b
            * (ulysses_attn_coeff * ccfg.n_attMM * ccfg.h
               + ccfg.n_ffMM * ccfg.hff)
            / t
        )
        if False:
            return _cp_comm_cost_common(
                a2a_vol, a2a_vol * 2, comm_vol, 0, 2, cp, s, b,
                attention_type, kv_dim, cp_algo, topology, effective_bandwidth)
        return _cp_comm_cost_common(
            a2a_vol, a2a_vol * 2, comm_vol, 0, 2, cp, s, b,
            attention_type, kv_dim, cp_algo, topology, effective_bandwidth,
            device_type=device_type)

    kv_bytes = 4
    kv_vol_step = (s / cp) * b * kv_dim * kv_bytes
    total_kv = kv_vol_step * (cp - 1) * 2
    # comm_volume: same weighted-unit as dp/tp/ep
    # Ring attention coeff = 2*0.5*rec_factor + 0.5 (extra /cp from (s/cp)^2)
    ring_attn_coeff = 2 * 0.5 * rec_factor + 0.5
    comm_vol = (
        ccfg.comm_cp * 2 * s * b
        * (ring_attn_coeff * ccfg.n_attMM * ccfg.h
           + ccfg.n_ffMM * ccfg.hff)
        / t
    )
    if False:
        return _cp_comm_cost_common(
            kv_vol_step, total_kv, comm_vol, int(cp - 1), 2, cp, s, b,
            attention_type, kv_dim, cp_algo, topology, effective_bandwidth)
    return _cp_comm_cost_common(
        kv_vol_step, total_kv, comm_vol, int(cp - 1), 2, cp, s, b,
        attention_type, kv_dim, cp_algo, topology, effective_bandwidth,
        device_type=device_type)


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


def comm_embed_ouput(cfg):
    """ "formula"""
    comm_embed = cfg.bytes_compute * cfg.h * cfg.v / cfg.shard_embed
    comm_output = cfg.h * cfg.v / cfg.t
    return comm_embed, comm_output


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


def _accumulate_layer_comm(comm, param):
    """Accumulate per-layer communication volumes for DP, FSDP, TP, EP, CP."""
    cfg = param["cfg"]
    ctx = param["ctx"]
    layer = ctx.current_node

    fsdp_intra_vol = 0.0
    hsdp_inter_vol = 0.0
    tp_layer_count = 0
    fsdp_layer_count = 0

    is_fsdp_layer = (
        layer not in [LayerType.EMBEDDING_LAYER, LayerType.OUTPUT_LAYER]
    )
    if is_fsdp_layer and param["flatten"]:
        custom_fun = param["flatten"].pop(0)
        if custom_fun:
            custom_fun(cfg)
        logger.info("is layer moe ? %s", cfg.n_exp > 1)
        ctx.current_node = LayerType.NOT_REC_LAYER
        logger.info("param ctx %s", ctx)
        comm[Dim.DP] += EvalLayerComm.dp_comm_layer(cfg, ctx)
        comm[Dim.FSDP] += EvalLayerComm.fsdp_comm_layer(cfg, ctx)
        non_exp, routed, shared = ctx.eval.num_p(cfg, ctx)
        exp = routed + shared
        bc = cfg.bytes_compute
        fsdp_intra_vol += non_exp * bc * 2
        if cfg.n_exp > 1:
            fsdp_intra_vol += exp * bc * 2
        d_shard_local = cfg.d_shard_or_d
        if getattr(cfg, "comm_hsdp", 0) > 0 and d_shard_local < cfg.d:
            sharded_non_exp = non_exp / (d_shard_local * cfg.cp * cfg.t)
            sharded_exp = exp / (d_shard_local * cfg.cp * cfg.t_exp) if cfg.n_exp > 1 else 0
            hsdp_inter_vol += (sharded_non_exp + sharded_exp) * bc
    fsdp_layer_count = 1 if is_fsdp_layer else 0

    comm[Dim.TP] += EvalLayerComm.tp_comm_layer(cfg, ctx, 1)
    tp_layer_count += 1
    comm[Dim.EP] += EvalLayerComm.ep_comm_layer(cfg, ctx, 1)
    comm[Dim.CP] += cp_comm_layer_detailed(cfg, ctx, device_type=param.get("device_type")).comm_volume

    return fsdp_intra_vol, hsdp_inter_vol, tp_layer_count, fsdp_layer_count


def _accumulate_layer_comm_inline_legacy(comm, param):
    """Legacy inline layer comm accumulation (preserved for reference)."""
    for _stage in [None]:
        for _chunk in [None]:
            for layer in [None]:
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
                comm[Dim.CP] += cp_comm_layer_detailed(
                    param["cfg"], param["ctx"]
                ).comm_volume
                # min(device_type.level_bound_number[0], param["cfg"].ep)
                # comm_cp += EvalLayerComm.cp_comm_layer
                # (param["cfg"], param["ctx"])


def _flop_mode_fsdp_comm(cfg, fsdp_layer_count, d_shard_val, device_type, pp=1, mb=1):
    """Estimate FSDP/HSDP communication cost for FLOP mode."""
    if fsdp_layer_count <= 0 or d_shard_val <= 0:
        return 0.0

    f = _compute_hsdp_features(cfg, d_shard_val, device_type, mb=mb)
    f["pp"] = max(pp, 1)
    is_hsdp = f["d_replicate"] > 1

    sub_key = "hsdp" if is_hsdp else "fsdp"
    c = _get_flop_coeffs(device_type, "shard", sub_key)
    if not c:
        return 0.0

    if is_hsdp:
        total = (
            c.get("INTERCEPT", 0)
            + c.get("TP", 0) * f["tp"]
            + c.get("SG", 0) * f["sg"]
            + c.get("CROSS_AG_VOL", 0) * f["cross_ag_vol"]
            + c.get("CROSS_INV_TP", 0) * f["cross_inv_tp"]
            + c.get("INV_SG_D_REP", 0) * f["inv_sg_d_rep"]
            + c.get("AG_VOL_D_REP", 0) * f["ag_vol_d_rep"]
        )
        return total / f["pp"] / f["m"]
    per_layer = (
        c.get("INTERCEPT", 0)
        + c.get("INV_SG", 0) * f["inv_sg"]
        + c.get("CROSS_INV_TP", 0) * f["cross_inv_tp"]
        + c.get("D_SHARD", 0) * f["d_shard"]
        + c.get("TP", 0) * f["tp"]
    )
    return per_layer * 2 * fsdp_layer_count / f["pp"] / f["m"]


def _flop_mode_dp_comm(comm_dp_raw, cfg, d_shard_val, device_type, mb=1):
    """Estimate DP communication cost for FLOP mode."""
    if comm_dp_raw <= 0:
        return 0.0

    f = _compute_hsdp_features(cfg, d_shard_val, device_type, mb=mb)

    if (cfg.fsdp or d_shard_val > 1) and d_shard_val > 0:
        if f["sg"] <= 1:
            return 0.0

        if f["d_replicate"] > 1:
            c = _get_flop_coeffs(device_type, "dp", "hsdp")
            if not c:
                return 0.0
            dp_val = f["d_replicate"] * f["d_shard"]
            total = (
                c.get("INTERCEPT", 0)
                + c.get("TP", 0) * f["tp"]
                + c.get("DP", 0) * dp_val
                + c.get("SG", 0) * f["sg"]
                + c.get("AG_VOL", 0) * f["ag_vol"]
                + c.get("D_REP", 0) * f["d_replicate"]
                + c.get("CROSS_AG_VOL", 0) * f["cross_ag_vol"]
            )
            score = total / f["m"]
            logger.info(
                "FLOP_DP_HSDP: d=%d tp=%d d_shard=%d sg=%d d_rep=%d "
                "cross=%.1f score=%.2f mb=%d raw=%.4f",
                f["d"], f["tp"], f["d_shard"], f["sg"], f["d_replicate"],
                f["cross"], score, f["m"], comm_dp_raw,
            )
            return score

        return 0.0

    logger.info(
        "FLOP_DP_FALLBACK: d=%d tp=%d fsdp=%s d_shard=%d raw=%.4f result=0.0",
        f["d"], f["tp"], cfg.fsdp, f["d_shard"],
        comm_dp_raw,
    )
    return 0.0


def _flop_mode_tp_comm(cfg, d_shard_val, device_type, mb=1):
    """Estimate TP communication cost for FLOP mode when d_replicate > 1."""
    f = _compute_hsdp_features(cfg, d_shard_val, device_type, mb=mb)
    if f["d_replicate"] <= 1:
        return 0.0
    c = _get_flop_coeffs(device_type, "tp", "hsdp")
    if not c:
        return 0.0
    total = (
        c.get("INTERCEPT", 0)
        + c.get("INV_TP", 0) * f["inv_tp"]
        + c.get("INV_SG", 0) * f["inv_sg"]
        + c.get("LOG2_D_REP", 0) * f["log2_d_rep"]
        + c.get("CROSS_D_REP", 0) * f["cross_d_rep"]
        + c.get("CROSS_SG", 0) * f["cross_sg"]
        + c.get("M", 0) * f["m"]
    )
    score = total / f["m"]
    logger.info(
        "FLOP_TP_HSDP: d=%d tp=%d d_shard=%d sg=%d d_rep=%d "
        "cross=%.1f m=%d score=%.2f",
        f["d"], f["tp"], f["d_shard"], f["sg"], f["d_replicate"],
        f["cross"], f["m"], score,
    )
    return score


def _flop_mode_pp_total_comm(cfg, d_shard_val, device_type, mb=1):
    """Estimate PP_TOTAL cost for FLOP mode when d_replicate > 1 (HSDP)."""
    f = _compute_hsdp_features(cfg, d_shard_val, device_type, mb=mb)
    if f["d_replicate"] <= 1:
        return 0.0
    c = _get_flop_coeffs(device_type, "pp", "hsdp")
    if not c:
        return 0.0
    total = (
        c.get("INTERCEPT", 0)
        + c.get("AG_VOL_D_REP", 0) * f["ag_vol_d_rep"]
        + c.get("CROSS_SG", 0) * f["cross_sg"]
        + c.get("M", 0) * f["m"]
    )
    score = total / f["m"]
    logger.info(
        "FLOP_PP_TOTAL_HSDP: d=%d tp=%d d_shard=%d sg=%d d_rep=%d "
        "cross=%.1f m=%d score=%.2f",
        f["d"], f["tp"], f["d_shard"], f["sg"], f["d_replicate"],
        f["cross"], f["m"], score,
    )
    return score


def compute_hsdp_flop_total(cfg, d_shard_val, device_type, fsdp_layer_count, mb=1, pp=1):
    """Estimate performance using HSDP FLOP model."""
    comp_score = _flop_mode_comp_comm(cfg, d_shard_val, device_type=device_type, mb=mb)
    tp_score = _flop_mode_tp_comm(cfg, d_shard_val, device_type=device_type, mb=mb)
    dp_score = _flop_mode_dp_comm(1.0, cfg, d_shard_val, device_type, mb=mb)
    shard_score = _flop_mode_fsdp_comm(
        cfg, fsdp_layer_count, d_shard_val, device_type, pp=pp, mb=mb,
    )
    pp_score = _flop_mode_pp_total_comm(cfg, d_shard_val, device_type, mb=mb)
    total_score = comp_score + tp_score + shard_score + dp_score + pp_score
    logger.info(
        "HSDP_FLOP_TOTAL: comp=%.2f tp=%.2f shard=%.2f dp=%.2f pp=%.2f total=%.2f",
        comp_score, tp_score, shard_score, dp_score, pp_score, total_score,
    )
    return total_score


def _apply_flop_mode(comm, param, fsdp_layer_count):
    """Apply FLOP-mode estimation to comm_time calculation."""
    cfg = param["cfg"]
    d_shard_val = cfg.d_shard_or_d
    mb = cfg.m if hasattr(cfg, "m") and cfg.m > 0 else 1
    pp = cfg.p if hasattr(cfg, "p") and cfg.p > 0 else 1

    f = _compute_hsdp_features(cfg, d_shard_val, param["device_type"], mb=mb)
    dev_per_node = f["dev_per_node"]

    if f["d_replicate"] > 1:
        comm[Dim.TP] = _flop_mode_tp_comm(
            cfg, d_shard_val, param["device_type"], mb=mb,
        )
    else:
        c = _get_flop_coeffs(param["device_type"], "tp", "fsdp")
        if c:
            tp_scaling = (c.get("A", 0) + c.get("B", 0) * f["tp"] + c.get("C", 0) / f["tp"]) / max(f["d"], 1)
            comm[Dim.TP] *= max(1, f["tp"] // dev_per_node) * tp_scaling

    comm[Dim.EP] *= max(1, param["cfg"].ep // dev_per_node)
    comm[Dim.CP] *= max(1, param["cfg"].cp // dev_per_node)

    dp_result = _flop_mode_dp_comm(comm[Dim.DP], cfg, d_shard_val, param["device_type"], mb=mb)
    if dp_result > 0:
        comm[Dim.DP] = dp_result

    if (cfg.fsdp or d_shard_val > 1) and fsdp_layer_count > 0:
        comm[Dim.FSDP] = _flop_mode_fsdp_comm(
            cfg, fsdp_layer_count, d_shard_val, param["device_type"],
            pp=pp, mb=mb,
        )


def _apply_dev_per_node_scaling_legacy(comm, param):
    """Legacy dev_per_node scaling (preserved for reference)."""
    for _ in [None]:
        dev_per_node = param["device_type"].level_bound_number[0]
        comm[Dim.TP] *= max(1, param["cfg"].t // dev_per_node)
        comm[Dim.EP] *= max(1, param["cfg"].ep // dev_per_node)
        comm[Dim.CP] *= max(1, param["cfg"].cp // dev_per_node)


def _apply_overlap_correction_legacy(comm, param):
    """Legacy overlap correction (preserved for reference)."""
    for _ in [None]:
        # Transitional overlap correction.
        # The search runs the FLOP path, which has no other overlap
        # modeling; these factors are the only overlap correction on that
        # path.  The TIME path's estimate_comm_score(overlap=...) call
        # above is zeroed, so this is the single source of overlap for
        # both paths.
        # Defaults (dp=0.9, tp=0.5) are MindFormers-validated overlap, not
        # test hacks: they made the model match real MindFormers step times.
        # Re-validating for the hyper-parallel target is a follow-up.
        # Follow-up: source from hardware, fix estimate_comm_score's dim
        # list and add latency, then fold this into estimate_comm_score.
        comm[Dim.DP] *= (1 - param["cfg"].comm_dp_overlap)
        comm[Dim.TP] *= (1 - param["cfg"].comm_tp_overlap)


def _apply_a3_ratio_legacy(comm, param):
    """Legacy A3 ratio correction (preserved for reference)."""
    for _ in [None]:
        if param["device_type"].name == "A3":
            logger.info("A3 ratio")
            comm[Dim.DP] /= 2
            comm[Dim.TP] /= 2
            comm[Dim.EP] /= 2
            comm[Dim.CP] /= 2


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
    comms = {Dim.DP: [], Dim.TP: [], Dim.EP: []}
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
        # Transitional overlap correction.
        # The search runs the FLOP path, which has no other overlap
        # modeling; these factors are the only overlap correction on that
        # path.  The TIME path's estimate_comm_score(overlap=...) call
        # above is zeroed, so this is the single source of overlap for
        # both paths.
        # Defaults (dp=0.9, tp=0.5) are MindFormers-validated overlap, not
        # test hacks: they made the model match real MindFormers step times.
        # Re-validating for the hyper-parallel target is a follow-up.
        # Follow-up: source from hardware, fix estimate_comm_score's dim
        # list and add latency, then fold this into estimate_comm_score.
        if False:
            comm[Dim.DP] *= (1 - param["cfg"].comm_dp_overlap)
            comm[Dim.TP] *= (1 - param["cfg"].comm_tp_overlap)

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
        if False:
            res += [c + comms[Dim.DP][i] + comms[Dim.EP][i] + comms[Dim.CP][i]]
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
    comms = {Dim.DP: [], Dim.TP: [], Dim.EP: [], Dim.CP: [], Dim.FSDP: []}
    for stage in param["stages"]:
        comm = {Dim.DP: 0.0, Dim.TP: 0.0, Dim.EP: 0.0, Dim.CP: 0.0, Dim.FSDP: 0.0}
        stage_fsdp_count = 0
        for chunk in stage:
            for layer in chunk:
                param["ctx"].current_node = layer
                _, _, _, fc = _accumulate_layer_comm(comm, param)
                stage_fsdp_count += fc

        _apply_flop_mode(comm, param, stage_fsdp_count)

        if param["ccfg"].ttype == PerformanceType.TIME:
            for dim, ov in zip([Dim.DP, Dim.TP, Dim.CP], [0.0, 0.0, 0.0]):
                comm[dim] = estimate_comm_score(
                    param["cfg"],
                    comm[dim],
                    dim,
                    overlap=ov,
                    device=param["device_type"],
                )

        comm[Dim.TP] *= param["cfg"].comm_t
        comm[Dim.EP] *= param["cfg"].comm_ep

        d_shard_val = param["cfg"].d_shard_or_d
        if not (param["cfg"].fsdp or d_shard_val > 1):
            comm[Dim.DP] *= (1 - param["cfg"].comm_dp_overlap)
            comm[Dim.TP] *= (1 - param["cfg"].comm_tp_overlap)

        scale = param["device_type"].comm_scale_factor
        if scale != 1.0:
            logger.info("comm scale factor: %.2f", scale)
            for dim in (Dim.DP, Dim.TP, Dim.EP, Dim.CP, Dim.FSDP):
                comm[dim] *= scale

        comms[Dim.DP].append(comm[Dim.DP])
        comms[Dim.TP].append(comm[Dim.TP])
        comms[Dim.EP].append(comm[Dim.EP])
        comms[Dim.CP].append(comm[Dim.CP])
        comms[Dim.FSDP].append(comm[Dim.FSDP])

    if param["debugger"] and param["debugger"].is_enabled():
        logger.info("DP_COMM = %s", comms[Dim.DP])
        logger.info("TP(MP)_COMM = %s", comms[Dim.TP])
        logger.info("EP_COMM = %s", comms[Dim.EP])
        logger.info("CP_COMM = %s", comms[Dim.CP])
        logger.info("FSDP_COMM = %s", comms[Dim.FSDP])
        param["debugger"].info[PerfParts.DP_COMM] = comms[Dim.DP]
        param["debugger"].info[PerfParts.MP_COMM] = comms[Dim.TP]
        param["debugger"].info[PerfParts.EP_COMM] = comms[Dim.EP]
        param["debugger"].info[PerfParts.CP_COMM] = comms[Dim.CP]
        param["debugger"].info[PerfParts.FSDP_COMM] = comms[Dim.FSDP]
        if param["cfg"].cp > 1:
            cp_comm_details = cp_comm_layer_detailed(param["cfg"], param["ctx"],
                                                      device_type=param.get("device_type"))
            param["debugger"].info["CP_KV_VOLUME"] = cp_comm_details.total_kv_volume
            param["debugger"].info["CP_EXPOSED_TIME"] = cp_comm_details.exposed_comm_time
            param["debugger"].info["CP_TOPOLOGY"] = cp_comm_details.topology
            param["debugger"].info["CP_BANDWIDTH"] = cp_comm_details.effective_bandwidth

    res = []
    for i, c in enumerate(comms[Dim.TP]):
        res += [c + comms[Dim.DP][i] + comms[Dim.EP][i] + comms[Dim.CP][i] + comms[Dim.FSDP][i]]

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
def level_efficiency(level, device=None):
    """to improve for Ascend A2"""
    if device is None:
        if level == NetworkLevel.NODE:
            return 0.7
        if level == NetworkLevel.CLUSTER:
            return 0.9
        raise ValueError
    idx = level.value - 1 if isinstance(level, NetworkLevel) else level - 1
    p2p_eff = getattr(device, 'p2p_efficiency', None)
    if p2p_eff is not None and 0 <= idx < len(p2p_eff):
        return p2p_eff[idx]
    if 0 <= idx < len(device.level_efficiency):
        return device.level_efficiency[idx]
    raise ValueError(
        f"No efficiency for level {level}; device required"
    )


def level_bandwidth(level):
    """to improve for Ascend A2"""
    if level == NetworkLevel.NODE:
        return 300
    if level == NetworkLevel.CLUSTER:
        return 25
    raise ValueError
def level_bandwidth(level, device=None):
    """to improve for Ascend A2"""
    if device is None:
        if level == NetworkLevel.NODE:
            return 300
        if level == NetworkLevel.CLUSTER:
            return 25
        raise ValueError
    idx = level.value - 1 if isinstance(level, NetworkLevel) else level - 1
    p2p = getattr(device, 'p2p_bandwidth', None)
    if p2p is not None and idx < len(p2p):
        return p2p[idx]
    if 0 <= idx < len(device.level_bandwidth):
        return device.level_bandwidth[idx]
    raise ValueError(
        f"No P2P bandwidth for level {level}; device required"
    )


def level_latency(level):
    """to improve for Ascend A2"""
    if level == NetworkLevel.NODE:
        return 0.00001
    if level == NetworkLevel.CLUSTER:
        return 0.00002
    raise ValueError
def level_latency(level, device=None):
    """to improve for Ascend A2"""
    if device is None:
        if level == NetworkLevel.NODE:
            return 0.00001
        if level == NetworkLevel.CLUSTER:
            return 0.00002
        raise ValueError
    idx = level.value - 1 if isinstance(level, NetworkLevel) else level - 1
    if 0 <= idx < len(device.level_latency):
        return device.level_latency[idx]
    raise ValueError(
        f"No latency for level {level}; device required"
    )


def comm_throughput(level):
    """formula"""
    eff = level_efficiency(level)
    bw = level_bandwidth(level)
def comm_throughput(level, device=None):
    """formula"""
    if device is None:
        eff = level_efficiency(level)
        bw = level_bandwidth(level)
    else:
        eff = level_efficiency(level, device=device)
        bw = level_bandwidth(level, device=device)
    return bw * eff


def estimate_comm_size_time(_, comm_size, level):
    """formula"""
    th = comm_throughput(level)
    lat = level_latency(level)
def estimate_comm_size_time(_, comm_size, level, device=None):
    """formula"""
    if device is None:
        th = comm_throughput(level)
        lat = level_latency(level)
    else:
        th = comm_throughput(level, device=device)
        lat = level_latency(level, device=device)
    return lat + comm_size / th


def _shard_group_levels(device, shard_size):
    """Compute message-size-dependent effective bandwidth."""
    remaining = shard_size
    levels = []
    for level in range(device.levels):
        bound = device.level_bound_number[level]
        if bound:
            n = min(remaining, bound)
            remaining = remaining // n if n > 0 else remaining
        else:
            n = remaining
        levels.append(n)
    return levels


def estimate_comm_score(
    cfg,
    comm_volume,
    dim,
    overlap = 0.0,
    device = Hard.device_map['A2'],
    a2a_efficiency = None,
    shard_group_size = 0,
    per_rank_msg = False,
):
    """Estimate communication time in seconds from byte volume.

    Uses a standard collective-communication model:
      time = (1 - overlap) * sum over levels [
          2 * (n_level - 1) / n_level * msg_per_level / bw_level
          + latency_level * 2 * (n_level - 1)
      ]
    where n_level is the number of ranks at that level participating
    in the collective, and msg_per_level is the per-rank message size.

    level_bandwidth is stored in GB/s; converted to B/s here.

    a2a_efficiency: if set, overrides level_efficiency for all-to-all
    operations.  Can be:
      - float: applied to all levels
      - dict mapping level index (0-based) to float: per-level override
    FSDP needs different efficiencies for intra-node (level 0) vs
    inter-node (level 1) due to kernel launch overhead and protocol
    differences.

    shard_group_size: if > 0, use this instead of level_assign for
    hierarchy decomposition.  Needed for FSDP where the all-gather
    group is d_shard*cp*t ranks, which may not align with any single
    dimension in level_assign (e.g., DP=1,TP=8,FSDP=True).

    per_rank_msg: if True, comm_volume is already the per-rank message
    size (not total).  Skip the /n_level division.  Use for FSDP/HSDP
    where the volume is pre-computed as a per-rank shard.
    """
    if comm_volume <= 0:
        return 0

    d_shard = cfg.d_shard_or_d

    if shard_group_size > 0:
        n_levels = _shard_group_levels(device, shard_group_size)
    else:
        assignment = device.level_assign(
            dp=cfg.d, tp=cfg.t, cp=cfg.cp, pp=cfg.p, d_shard=d_shard
        )
        lookup_dim = dim if dim in assignment else Dim.DP
        n_levels = assignment[lookup_dim]

    time_s = 0
    for level in range(device.levels):
        n_level = n_levels[level]
        if n_level <= 1:
            continue
        if per_rank_msg:
            msg_per_rank = comm_volume
        elif shard_group_size > 0:
            prod_from_l = 1
            for k in range(level, len(n_levels)):
                prod_from_l *= n_levels[k]
            msg_per_rank = comm_volume / prod_from_l
        else:
            msg_per_rank = comm_volume / n_level
        if isinstance(a2a_efficiency, dict):
            eff = a2a_efficiency.get(level, level_efficiency(NetworkLevel(level + 1), device=device))
        elif a2a_efficiency is not None:
            eff = a2a_efficiency
        else:
            eff = level_efficiency(NetworkLevel(level + 1), device=device)
        bw_bps = device.level_bandwidth[level] * 1e9 * eff
        lat = level_latency(NetworkLevel(level + 1), device=device)
        if per_rank_msg:
            collective_time = (
                (n_level - 1) * msg_per_rank / bw_bps
                + lat * 2 * (n_level - 1)
            )
        else:
            collective_time = (
                2 * (n_level - 1) / n_level * msg_per_rank / bw_bps
                + lat * 2 * (n_level - 1)
            )
        time_s += collective_time

    return time_s * (1 - overlap)


def estimate_comm_score_legacy(
    cfg, comm_volume, dim, overlap=0.0, device=Hard.device_map["A2"]
):
    """score assignment"""
    assignment = device.level_assign(dp=cfg.d, tp=cfg.t, cp=cfg.cp, pp=cfg.p)
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

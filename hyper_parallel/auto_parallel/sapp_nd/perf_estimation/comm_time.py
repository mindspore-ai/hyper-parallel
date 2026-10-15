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
from copy import copy, deepcopy
from typing import Any, NamedTuple
from hyper_parallel.auto_parallel.sapp_nd.nd.logger import perf_logger as logger
import hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware as Hard
import hyper_parallel.auto_parallel.sapp_nd.nd.dimensions as Dim
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType
from hyper_parallel.auto_parallel.sapp_nd.nd.debug import PerfParts
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.comm import EvalLayerComm
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.utils import EvalUtils
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
    get_layer_group_configs,
    get_layer_switches_by_position,
    get_model_order,
    get_table_quantity,
)
from hyper_parallel.auto_parallel.sapp_nd.nd.common.cp_types import (
    CPCommunicationCost,
    CPAlgo,
    _resolve_cp_algo,
)
from hyper_parallel.auto_parallel.sapp_nd.nd.common.arch_hooks import apply_layer_kind, layer_kinds
from hyper_parallel.auto_parallel.sapp_nd.nd.common.derive import param_cp
from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import (
    detect_attention_type,
    AttentionType,
    compute_kv_dim,
    CostModelConfig,
)

COUNT_OPTIMIZER = False


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
    """Return a zero CPCommunicationCost for cp <= 1."""
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


class _CPVolumes(NamedTuple):
    """Algorithm-specific CP communication volumes of one layer.

    ``comm_volume`` is what :func:`cp_traffic` moves, the only field a score
    reads; the others are logged.
    """

    kv_volume_per_step: float
    total_kv_volume: float
    comm_volume: float
    ring_steps: int
    ring_directions: int


def _cp_comm_cost_common(ccfg, volumes, attention_type, kv_dim, cp_algo,
                          topology, effective_bandwidth):
    """Build CPCommunicationCost with standard time calculation."""
    overlap_ratio = 0.5
    total_comm_time = (volumes.total_kv_volume / (effective_bandwidth * 1e9)) * 1e3
    exposed_comm_time = total_comm_time * (1 - overlap_ratio)
    return CPCommunicationCost(
        kv_volume_per_step=volumes.kv_volume_per_step,
        total_kv_volume=volumes.total_kv_volume,
        comm_volume=volumes.comm_volume,
        ring_steps=volumes.ring_steps, ring_directions=volumes.ring_directions,
        total_comm_time=total_comm_time,
        exposed_comm_time=exposed_comm_time,
        overlap_ratio=overlap_ratio,
        effective_bandwidth=effective_bandwidth,
        topology=topology, cp_degree=int(ccfg.cp),
        seq_len=int(ccfg.s), batch_size=int(ccfg.b),
        attention_type=attention_type, kv_dim=int(kv_dim),
        cp_algo=cp_algo,
    )


def _ulysses_cp_volumes(ccfg):
    """Ulysses CP volumes: two All2All over (cp-1)/cp of the local shard."""
    s, b = ccfg.s, ccfg.b
    cp = ccfg.cp
    t = max(1, ccfg.t)
    local_qkv = s * b * (ccfg.a / t) * ccfg.dh * 2
    a2a_vol = local_qkv * (cp - 1) / cp
    return _CPVolumes(a2a_vol, a2a_vol * 2, cp_traffic(ccfg, CPAlgo.ULYSSES_CP), 0, 2)


def _ring_cp_volumes(ccfg, kv_dim, cp_algo):
    """Ring CP volumes: cp-1 P2P steps of s/cp tokens of KV, both directions."""
    s, b = ccfg.s, ccfg.b
    cp = ccfg.cp
    kv_bytes = 4
    kv_vol_step = (s / cp) * b * kv_dim * kv_bytes
    total_kv = kv_vol_step * (cp - 1) * 2
    return _CPVolumes(kv_vol_step, total_kv, cp_traffic(ccfg, cp_algo), int(cp - 1), 2)


def cp_traffic(ccfg: CostModelConfig, cp_algo: CPAlgo) -> float:
    """The bytes a rank moves a micro-batch for one layer's context parallelism.

    Attention exchanges its keys and values, the MLP nothing.  colossalai
    and hybrid CP all-gather K and V over the sequence in the forward and
    reduce-scatter their gradients in the backward, as HyperParallel runs
    them (a ring passes the same chunks), each moving (cp - 1) / cp of the
    sequence a rank.  Ulysses all-to-alls the local query, key, value and
    output between sequence and heads, forward and backward.  A
    linear-attention layer passes its recurrent state to the next rank,
    and its gradient back, unless it all-to-alls as Ulysses does.
    """
    cp, t = ccfg.cp, max(1, ccfg.t)
    ring = (cp - 1) / cp
    head = ccfg.dh or ccfg.h / max(1, ccfg.a)
    kv_width = compute_kv_dim(ccfg)
    if cp_algo == CPAlgo.ULYSSES_CP:
        rope = ccfg.dhr if detect_attention_type(ccfg) == AttentionType.MLA else 0
        widths = ccfg.a * (head + rope) / t + 2 * kv_width + ccfg.a * head / t
        return 2 * ring * ccfg.s / cp * ccfg.b * widths * ccfg.bytes_compute
    if ccfg.n_linrec:
        state_bytes = 4
        return 2 * ccfg.a / t * head * head * state_bytes
    return 2 * ring * ccfg.s * ccfg.b * 2 * kv_width * ccfg.bytes_compute


def cp_comm_layer_detailed(ccfg: CostModelConfig, ctx: Context = None) -> CPCommunicationCost:
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

    *ctx* is not read: the exchange a recompute sends again is priced by
    the walk that recomputes it (``_recomputed_comm``).
    """
    del ctx
    if ccfg.cp <= 1:
        return _cp_comm_zero(ccfg)

    if ccfg.a <= 0:
        raise ValueError(f"Number of attention heads must be positive, got {ccfg.a}")

    kv_dim = compute_kv_dim(ccfg)
    attention_type = detect_attention_type(ccfg)
    cp_algo = _resolve_cp_algo(ccfg)
    topology, effective_bandwidth = _cp_resolve_topology(
        ccfg.cp, ccfg.device_per_node, ccfg.bw_intra, ccfg.bw_inter)

    if cp_algo == CPAlgo.ULYSSES_CP:
        volumes = _ulysses_cp_volumes(ccfg)
    else:
        volumes = _ring_cp_volumes(ccfg, kv_dim, cp_algo)
    return _cp_comm_cost_common(
        ccfg, volumes, attention_type, kv_dim, cp_algo, topology, effective_bandwidth)


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

    lccfgs = get_layer_group_configs(param["cfg"])
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
    # A MoE layer's experts, counted as the memory path's eval config counts them.
    ctx.ffn_routed_num_p = EvalFFn.num_params_routed_expert
    ctx.ffn_shared_num_p = EvalFFn.num_params_shared_expert
    ctx.ffn_router_num_p = EvalFFn.num_params_router
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


def _recomputed_comm(cfg, ctx, layer):
    """TP, EP and CP volume a recomputed layer transfers again.

    It is the communication whose buffers the layer's memory no longer keeps:
    all of it for a fully recomputed layer, and for a selective one what its
    switches drop, which the memory model's own terms give as the plain
    volume less the selective one. Parameter traffic is not recomputed. Of
    CP's traffic, a recompute runs the forward's exchange again, the half
    the gather switch keeps in a selective layer.
    """
    def _volumes(node):
        ctx.current_node = node
        kept = EvalUtils.rec_coeff(node == LayerType.SEL_REC_LAYER, EvalUtils.switch(cfg, ctx, "gather"))
        return (
            EvalLayerComm.tp_comm_layer(cfg, ctx, 1),
            EvalLayerComm.ep_comm_layer(cfg, ctx, 1),
            cp_comm_layer_detailed(cfg, ctx).comm_volume / 2 * kept,
        )

    kept = ctx.current_node
    try:
        plain = _volumes(LayerType.NOT_REC_LAYER)
        if layer == LayerType.FULL_REC_LAYER:
            return plain
        selective = _volumes(LayerType.SEL_REC_LAYER)
        return tuple(whole - left for whole, left in zip(plain, selective))
    finally:
        ctx.current_node = kept


def fsdp_root_parts(cfg: Any, layer: Any) -> tuple:
    """The root's table, the embedding's or the output layer's, as FSDP holds it.

    The parts are as :meth:`EvalLayerComm.fsdp_layer_parts` gives a layer's;
    a table the output layer shares is its.
    """
    if layer == LayerType.EMBEDDING_LAYER:
        if EvalHead.shares_output_table(cfg):
            return ()
        tp = cfg.shard_embed / max(1, cfg.gather_embed or 1)
        return ((EvalHead.num_params_embed(cfg, None), tp, cfg.shard_embed, cfg.d * tp),)
    return ((EvalTail.num_params_output(cfg, None), cfg.t, cfg.shard_p_os_non_exp_partial, cfg.d * cfg.cp * cfg.t),)


def _fsdp_rounds(cfg) -> tuple:
    """How many times a micro-batch FSDP gathers a layer's and the root's parameters, and all-reduces.

    Resharded after its forward and its backward, a layer is gathered for
    both, and the root, kept through its backward, once; kept gathered, each
    is gathered once a step.  The copies of a shard all-reduce once a step,
    after the reduce-scatters, where an FSDP group shards the parameters,
    and every micro-batch where none does.
    """
    micro = max(1, cfg.m)
    reshards = bool(getattr(cfg, "reshards", False))
    # Sharded beyond TP and whatever CP adds to the width (param_cp): with
    # TP times CP, a HyperParallel shard no wider than CP read as none.
    shards = cfg.shard_p_os_non_exp_partial > cfg.t * param_cp(cfg)
    return (2 if reshards else 1 / micro), (1 if reshards else 1 / micro), (1 / micro if shards else 1)


def _accumulate_stage_comm(param, stage, stage_id):
    """Sum the per-layer DP, TP, EP and CP communication volumes of one stage.

    With ``param["with_recomp"]`` or ``param["both"]``, the walk also keeps
    the same sums with the volume each recomputed layer's recompute transfers
    again added, the way the compute estimate counts a recomputed op twice.
    Both sums take every term in the same order as a walk keeping one of
    them, so each comes out as that walk's would.

    A body layer is priced on the walk's config, in its own kind; the
    embedding and the output layer on the model's config, as its family left
    it, whichever layer the walk reached last.  Under FSDP (grad_shard_as_params),
    a layer's DP volume, and the root's, is the traffic its collectives
    move a micro-batch (:meth:`EvalLayerComm.fsdp_traffic`).

    Returns:
        ``(comm, again, reduced)``: the volumes by dimension; the same with
        each recomputed layer's traffic added again, or ``None`` where the
        walk keeps no such sums; and the part of the DP volume the copies of
        a shard all-reduce (:meth:`EvalLayerComm.fsdp_reduce_traffic`).
    """
    comm = {Dim.DP: 0.0, Dim.TP: 0.0, Dim.EP: 0.0, Dim.CP: 0.0}
    again = dict(comm) if param["with_recomp"] or param["both"] else None
    sums = (comm,) if again is None else (comm, again)
    reduced = 0.0
    fsdp = bool(getattr(param["cfg"], "grad_shard_as_params", False))
    layer_gathers, root_gathers, reduces = _fsdp_rounds(param["cfg"]) if fsdp else (0, 0, 0)
    for chunk_id, chunk in enumerate(stage):
        for lay_id, layer in enumerate(chunk):
            param["ctx"].current_node = layer
            position = (stage_id, chunk_id, lay_id)
            is_body = layer not in [LayerType.EMBEDDING_LAYER, LayerType.OUTPUT_LAYER]
            cfg = param["walk"] if is_body else param["cfg"]
            # A layer's own recompute switches, where the config states several.
            param["ctx"].switches = param["switches"].get(position) if is_body else None
            if is_body and position in param["kinds"]:
                kind = param["kinds"][position]
                if kind is not None:
                    apply_layer_kind(cfg, kind)
                logger.info("is layer moe ? %s", cfg.n_exp > 1)
                param["ctx"].current_node = LayerType.NOT_REC_LAYER
                logger.info("param ctx %s", param["ctx"])
                if fsdp:
                    parts = EvalLayerComm.fsdp_layer_parts(cfg, param["ctx"])
                    _add(sums, Dim.DP, EvalLayerComm.fsdp_traffic(cfg, parts, layer_gathers, reduces))
                    reduced += EvalLayerComm.fsdp_reduce_traffic(cfg, parts, reduces)
                else:
                    _add(sums, Dim.DP, EvalLayerComm.dp_comm_layer(cfg, param["ctx"]))
            elif fsdp and not is_body:
                parts = fsdp_root_parts(cfg, layer)
                _add(sums, Dim.DP, EvalLayerComm.fsdp_traffic(cfg, parts, root_gathers, reduces))
                reduced += EvalLayerComm.fsdp_reduce_traffic(cfg, parts, reduces)

            _add(sums, Dim.TP, EvalLayerComm.tp_comm_layer(
                cfg, param["ctx"], 1
            ))  # / 4 #* (param["cfg"].t - 1)
            _add(sums, Dim.EP, EvalLayerComm.ep_comm_layer(
                cfg, param["ctx"], 1
            ))  # * param["cfg"].ep
            if is_body:
                _add(sums, Dim.CP, cp_comm_layer_detailed(
                    cfg, param["ctx"]
                ).comm_volume)
            # min(device_type.level_bound_number[0], param["cfg"].ep)
            if again is not None and layer in (LayerType.FULL_REC_LAYER, LayerType.SEL_REC_LAYER):
                tp_again, ep_again, cp_again = _recomputed_comm(cfg, param["ctx"], layer)
                again[Dim.TP] += tp_again
                again[Dim.EP] += ep_again
                again[Dim.CP] += cp_again
    return comm, again, reduced


def _add(sums, dim, volume):
    """Add *volume* to dimension *dim* of each of *sums*."""
    for comm in sums:
        comm[dim] += volume


def _stage_comm_time(param, comm):
    """Turn one stage's volumes into the walk's units, in place: the time model, the nodes, the overlap, A3."""
    if param["ccfg"].ttype == PerformanceType.TIME:
        for dim, ov in zip([Dim.DP, Dim.TP, Dim.CP], [0.0, 0.0, 0.0]):
            comm[dim] = estimate_comm_score(
                param["cfg"],
                comm[dim],
                dim,
                overlap=ov,
                device=param["device_type"],
            )

    dev_per_node = param["device_type"].level_bound_number[0]
    comm[Dim.TP] *= max(1, param["cfg"].t // dev_per_node)
    comm[Dim.EP] *= max(1, param["cfg"].ep // dev_per_node)
    comm[Dim.CP] *= max(1, param["cfg"].cp // dev_per_node)

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

    if param["device_type"].name == "A3":
        logger.info("A3 ratio")
        comm[Dim.DP] /= 2
        comm[Dim.TP] /= 2
        comm[Dim.EP] /= 2
        comm[Dim.CP] /= 2


def _stage_totals(comms):
    """Each stage's volume over the four dimensions, from per-dimension lists."""
    return [c + comms[Dim.DP][i] + comms[Dim.EP][i] + comms[Dim.CP][i] for i, c in enumerate(comms[Dim.TP])]


def estimate_from_mem_comm(*args, **kwargs):
    """Each stage's communication, from the memory model's volumes.

    With ``both``, one walk returns ``(plain, recomputed)``: each stage's
    communication, then the same with each recomputed layer's traffic added
    again, which ``with_recomp`` alone returns.  The debugger records the
    plain parts.
    """

    param = {
        "cfg": args[0],
        "ccfg": args[1],
        "stages": args[2],
        "device_type": args[3],
    }
    param["debugger"] = kwargs.get(
        "debugger", args[5] if len(args) > 5 else None
    )
    param["with_recomp"] = kwargs.get(
        "with_recomp", args[4] if len(args) > 4 else False
    )
    param["both"] = kwargs.get("both", False)
    param["ctx"] = prepare_context()
    # The layers' kinds go on a copy: the model's config leaves the walk as
    # it entered it, for the next walk and every read after.
    param["walk"] = copy(param["cfg"])

    # Each layer's kind, in model order; layers past the stack get no entry,
    # so no kind and no DP term.
    param["kinds"] = dict(zip(get_model_order(param["cfg"], param["stages"]), layer_kinds(param["cfg"])))
    param["switches"] = get_layer_switches_by_position(param["cfg"], param["stages"])
    comms = {Dim.DP: [], Dim.TP: [], Dim.EP: [], Dim.CP: []}
    recomms = {Dim.DP: [], Dim.TP: [], Dim.EP: [], Dim.CP: []}
    # Each stage's all-reduce, the share of its DP time the copies of a shard
    # spend reducing it: estimate_pipeline takes it out of DP_COMM.
    reduces = []
    for stage_id, stage in enumerate(param["stages"]):
        plain, again, reduced = _accumulate_stage_comm(param, stage, stage_id)
        comm = again if param["with_recomp"] and not param["both"] else plain
        reduce_share = reduced / comm[Dim.DP] if comm[Dim.DP] else 0.0

        _stage_comm_time(param, comm)
        for dim, values in comms.items():
            values.append(comm[dim])
        reduces.append(comm[Dim.DP] * reduce_share)
        if param["both"]:
            _stage_comm_time(param, again)
            for dim, values in recomms.items():
                values.append(again[dim])

    if param["debugger"] and param["debugger"].is_enabled():
        logger.info("DP_COMM = %s", comms[Dim.DP])
        logger.info("MP_COMM = %s", comms[Dim.TP])
        logger.info("EP_COMM = %s", comms[Dim.EP])
        logger.info("CP_COMM = %s", comms[Dim.CP])
        param["debugger"].info[PerfParts.DP_COMM] = comms[Dim.DP]
        param["debugger"].info[PerfParts.DP_REDUCE] = reduces
        param["debugger"].info[PerfParts.MP_COMM] = comms[Dim.TP]
        param["debugger"].info[PerfParts.EP_COMM] = comms[Dim.EP]
        param["debugger"].info[PerfParts.CP_COMM] = comms[Dim.CP]
        if param["cfg"].cp > 1:
            # Logged, not stored: debugger.info must hold only numeric PerfParts,
            # which the debug CSV and the score table are built from.
            cp_comm_details = cp_comm_layer_detailed(param["cfg"], param["ctx"])
            logger.info("CP_KV_VOLUME = %s", cp_comm_details.total_kv_volume)
            logger.info("CP_EXPOSED_TIME = %s", cp_comm_details.exposed_comm_time)
            logger.info("CP_TOPOLOGY = %s", cp_comm_details.topology)
            logger.info("CP_BANDWIDTH = %s", cp_comm_details.effective_bandwidth)

    if param["both"]:
        return _stage_totals(comms), _stage_totals(recomms)
    return _stage_totals(comms)


def estimate_comm(*args, **kwargs):
    """wrapper; with ``both``, the plain and the recomputed communication from one walk"""
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
        both=kwargs.get("both", False),
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

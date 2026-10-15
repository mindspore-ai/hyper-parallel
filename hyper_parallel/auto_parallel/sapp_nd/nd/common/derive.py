# Copyright 2026 Huawei Technologies Co., Ltd
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
"""The derived fields of a cost-model config, computed in one place.

A parser states primary facts: the model's dimensions, the parallel strategy
and the fixed facts of the run.  :func:`derive` computes what follows from
them, once when a config is parsed and again whenever its strategy changes,
where each parser used to compute it.  What a producer does not state, its
model's family says, from its op profile: :func:`derive_family`.
"""
import logging
import math
from typing import Any, Callable, Dict, Mapping, Union

from hyper_parallel.auto_parallel._exec_spec import RECOMPUTE_OPS
from hyper_parallel.auto_parallel._op_profiles import VISION_ARCH, family_profile
from hyper_parallel.auto_parallel.sapp_nd.nd.common.config import Config
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_order import stated_recompute

logger = logging.getLogger(__name__)

# The recompute switches of HyperParallel's selective activation checkpointing
# (hyper_parallel/distributed/activation_checkpoint.py). It keeps the outputs of
# matmul and attention kernels and of reduce-scatter, all-to-all and all-reduce,
# and recomputes everything else, all-gathers included. A switch at 1 keeps the
# op's activation and 0 recomputes it. The policy also recomputes every other
# projection matmul, which has no switch, so that part is not priced; an MLA
# layer's up-projections, matmuls too, are kept.
HYPER_SELECTIVE_REC_OP = {
    "attBMM": 1,
    "headCast": 0,
    "dropout": 0,
    "softmax": 0,
    "normOp": 0,
    "gather": 0,
    "ffAct": 0,
    "attUp": 1,
}


def runs_hyper_selective(ccfg: Any, switches: Any = None) -> bool:
    """Whether *ccfg*'s selective layers run HyperParallel's policy: their switches, or *switches*, are its switches.

    A switch they leave out keeps its op.
    """
    stated = getattr(ccfg, "rec_op", None) if switches is None else switches
    values = stated if isinstance(stated, Mapping) else vars(stated) if stated is not None else {}
    return all(values.get(name, 1) == state for name, state in HYPER_SELECTIVE_REC_OP.items())


def derive_sequence_parallel(ccfg: Any) -> None:
    """Set the sequence-parallel factor ``sp``: the TP degree, or 1 without."""
    ccfg.sp = ccfg.t if ccfg.sequence_parallel else 1


def derive_expert_degrees(ccfg: Any, strict: bool = True) -> None:
    """Set the degrees an expert layer runs with, ``t_exp`` and ``d_exp``.

    Args:
        ccfg: The config, with its degrees and expert layout set.
        strict: Refuse degrees that leave an expert group, the expert width
            or the expert count below 1.  When false, clamp them to 1
            instead, with a warning, so that a search can still reach the
            strategy and reject it by memory.

    Raises:
        TypeError: When *strict* and the degrees cannot hold the experts.
    """
    if ccfg.etp > 1:
        ccfg.t_exp = ccfg.etp
        # d * t = inner dp * outer dp * etp
        # inner dp = EP, outer dp = the rest
        ccfg.d_exp = ccfg.d * ccfg.t * ccfg.cp // ccfg.t_exp // ccfg.ep
    else:
        ccfg.t_exp = ccfg.t
        if ccfg.d >= ccfg.ep:
            ccfg.d_exp = ccfg.d // ccfg.ep
        else:
            ccfg.d_exp = ccfg.d * ccfg.t // ccfg.ep
        if ccfg.t_exp * ccfg.ep > ccfg.d * ccfg.t:
            ccfg.t_exp = 1

    # The group the runtime forms, which is what every consumer wants; a
    # strategy is impossible only when the stage holds fewer ranks than it
    # spreads experts over.  d_exp stays the older form, and is clamped
    # because a group of one still leaves it at zero under CP.
    ccfg.edp_group = expert_dp_group(ccfg)
    ccfg.d_exp = max(1, ccfg.d_exp)
    # A MoE layer dispatches its tokens once and takes them back once.
    ccfg.n_dispatch = 1 if ccfg.n_exp > 1 else 0

    exp_group1_invalid = ccfg.edp_group < 1 or ccfg.t_exp < 1
    exp_group2_invalid = ccfg.hff_exp < 1 or ccfg.n_exp < 1
    if not (exp_group1_invalid or exp_group2_invalid):
        return
    if strict:
        raise TypeError(
            f"MoE parsing error: edp_group({ccfg.edp_group})/"
            f"t_exp({ccfg.t_exp})/"
            f"hff_exp({ccfg.hff_exp})/n_exp({ccfg.n_exp})/"
            f"DP = {ccfg.d}, TP = {ccfg.t}, EP = {ccfg.ep}, CP = {ccfg.cp}/"
        )
    logger.warning(
        "Expert degrees invalid for d=%d t=%d ep=%d etp=%d n_exp=%d; clamping to minimum values.",
        ccfg.d, ccfg.t, ccfg.ep, ccfg.etp, ccfg.n_exp,
    )
    ccfg.d_exp = max(1, ccfg.d_exp)
    ccfg.edp_group = max(1, ccfg.edp_group)
    ccfg.t_exp = max(1, ccfg.t_exp)
    ccfg.hff_exp = max(1, ccfg.hff_exp)
    ccfg.n_exp = max(1, ccfg.n_exp)


def optimizer_ranks(ccfg: Any) -> int:
    """How many data-parallel ranks the optimizer shards a parameter over.

    ``os_max_shard`` counts them, as MindSpore's ``optimizer_weight_shard_size``
    and HyperParallel's ``dp_shard`` do, on top of TP's sharding.  A count
    that does not divide DP shards over all of it, as MindSpore does.
    HyperParallel's domain is DP times CP, which its shard spans
    (:func:`param_cp`).
    """
    domain = ccfg.d * (ccfg.cp if getattr(ccfg, "shard_spans_cp", False) else 1)
    ranks = int(ccfg.os_max_shard or 0)
    return ranks if ranks >= 1 and domain % ranks == 0 else domain


def param_cp(ccfg: Any) -> float:
    """What CP's ranks divide a parameter's state by, beyond the shard stated for it.

    The legacy schemas shard a parameter, its gradient and its optimizer
    states over CP's ranks on top of the optimizer's, and a table they gather
    is a CP rank's share of it.  HyperParallel's ``dp_shard_size`` already
    spans the DP x CP domain, its dense FSDP mesh being
    ``(dp_replicate, dp_shard, tp)`` over ``dp * cp`` ranks
    (``MeshContext.build_meshs``), and the table it gathers is whole: so 1
    there (``shard_spans_cp``), CP elsewhere (C1).
    """
    return 1 if getattr(ccfg, "shard_spans_cp", False) else ccfg.cp


def expert_dp_group(ccfg: Any) -> int:
    """How many ranks hold the same experts.

    The runtime spreads the experts over a whole pipeline stage: its expert
    mesh is ``(edp_replicate, edp_shard, ep)`` over the ``dp * cp * tp``
    ranks of the device mesh, so ``dp * cp * tp / ep`` ranks hold the same
    experts, and an expert tensor shard divides that group again
    (``hyper_parallel/distributed/mesh.py``,
    ``MeshContext._build_expert_parallel_mesh``).

    ``d_exp`` cannot stand in for this group.  It is floored by EP against
    DP alone, so it loses the remainder when EP does not divide DP and
    falls to zero once EP passes DP, which is what context parallelism
    does: DP counts the stage's ranks apart from CP and TP, so a strategy
    keeps its group while its DP shrinks with CP.  Returning zero made a
    runnable strategy look invalid, and the consumers that rebuild the
    group as ``d_exp * cp * t_exp`` cannot express a group narrower than
    CP.
    """
    expert_tp = int(ccfg.etp) if ccfg.etp > 1 else 1
    return int(ccfg.d * ccfg.t * ccfg.cp) // int(expert_tp * ccfg.ep)


def expert_dp_ranks(ccfg: Any) -> int:
    """How many ranks hold the same slice of a routed expert, so reduce its gradient together.

    The stage's ranks over EP and over the expert's tensor shard
    ``t_exp``, whose ranks hold different slices: context parallelism's
    ranks hold the same weights and reduce their gradients with the
    data-parallel ones, where ``d_exp`` counts the latter alone outside an
    expert tensor shard, and :func:`expert_dp_group` counts TP's ranks
    whatever slice they hold.
    """
    return max(1, int(ccfg.d * ccfg.t * ccfg.cp) // (max(1, int(ccfg.t_exp)) * max(1, int(ccfg.ep))))


def routed_expert_shard(ccfg: Any, ranks: int) -> int:
    """How many ranks a routed expert's parameters and optimizer states are sharded over.

    A run that states its expert shard, as HyperParallel's ``edp_shard_size``,
    shards an expert under expert parallelism over as many ranks of its expert
    data-parallel group: the stage's ranks over EP, whatever their DP, CP or
    TP, the group of ranks that hold the same experts.  Without expert
    parallelism its FSDP shards the experts with the other parameters, over
    the optimizer's *ranks*.  Stated by no one, the optimizer shards them over
    the whole group.  A run that shards each strategy's experts over its
    whole group (``expert_shard_group``) does so under expert parallelism
    whatever shard it states, since the group changes with the strategy.
    """
    stated = getattr(ccfg, "expert_shard", None)
    # The group as the runtime forms it, once a strategy has been laid over
    # the config; a dense model and the parser's own doubles state none, and
    # keep the older reconstruction from d_exp, which agrees with it
    # wherever EP divides DP.
    whole = max(1, int(ccfg.d * ccfg.cp * ccfg.t // max(1, int(ccfg.ep))))
    group = max(1, int(ccfg.edp_group)) if ccfg.edp_group else whole
    if getattr(ccfg, "expert_shard_group", False) and ccfg.ep > 1:
        return group
    if stated is None:
        if not ccfg.has_op:
            return param_cp(ccfg) * ccfg.t_exp
        return group if ccfg.edp_group else ccfg.d_exp * ccfg.cp * ccfg.t_exp
    if ccfg.ep > 1:
        return math.gcd(int(stated), group)
    return ranks * param_cp(ccfg) * ccfg.t_exp


def derive_optimizer_sharding(ccfg: Any) -> None:
    """Set how parameters, optimizer states and gradients are sharded.

    With optimizer sharding, a parameter is sharded over TP and then over
    :func:`optimizer_ranks` data-parallel ranks, and a routed expert as
    :func:`routed_expert_shard` says.  Gradients are sharded as
    the parameters are when the run says so (``grad_shard_as_params``), as
    FSDP shards them; else over the whole optimizer shard when the run
    shards them (``has_grad_shard``), and over TP alone otherwise.
    """
    ranks = optimizer_ranks(ccfg) if ccfg.has_op else 1
    # Non expert params, sharded over CP's ranks too where the stated shard
    # does not span them already (param_cp)
    ccfg.shard_p_os_non_exp_partial = ranks * ccfg.t * param_cp(ccfg)
    ccfg.shard_p_os_non_exp = (
        (ccfg.d if ccfg.has_op else 1) * ccfg.cp * ccfg.t
    )

    # Expert params
    ccfg.shard_p_os_exp_partial = math.gcd(ccfg.n_exp, ranks * ccfg.t_exp)
    ccfg.shard_p_os_exp = routed_expert_shard(ccfg, ranks)

    # Gradients
    if getattr(ccfg, "grad_shard_as_params", False):
        grads = (ccfg.shard_p_os_non_exp_partial, ccfg.shard_p_os_exp, ccfg.shard_p_os_exp_partial)
    elif ccfg.has_grad_shard:
        grads = (ccfg.shard_p_os_non_exp, ccfg.shard_p_os_exp, ccfg.shard_p_os_exp_partial)
    else:
        grads = (ccfg.t, ccfg.t_exp, ccfg.t_exp)
    ccfg.shard_grad_non_exp, ccfg.shard_grad_exp, ccfg.shard_grad_exp_partial = grads


def derive_param_casts(ccfg: Any) -> None:
    """Set whether a layer keeps the weight casts the formulas price beside its matmuls, ``keeps_param_casts``.

    The run states it as ``param_casts``; unstated, a layer keeps them where
    the optimizer does not shard, the formulas' convention.
    """
    stated = getattr(ccfg, "param_casts", None)
    ccfg.keeps_param_casts = (not ccfg.has_op) if stated is None else bool(stated)


def derive_comm_flags(ccfg: Any) -> None:
    """Set the communication factors and the transitional overlaps."""
    ccfg.comm_d_non_exp = (
        0
        if ((ccfg.d == 1) or not ccfg.has_op)
        else (2 if not ccfg.has_grad_shard else 3)
    )  # data parallel comm factor
    # Whether another rank holds the same slice of a routed expert, as a
    # shard or a copy, so that its gradient is reduced: d_exp left CP's
    # ranks out and read 1 where they reduce it (X3).
    ccfg.comm_d_exp = (
        0
        if ((expert_dp_ranks(ccfg) == 1) or not ccfg.has_op)
        else (2 if not ccfg.has_grad_shard else 3)
    )  # data parallel comm factor
    ccfg.comm_t = float(ccfg.t > 1)  # tensor parallel comm factor
    ccfg.comm_ep = float(
        ccfg.ep > 1 or ccfg.n_exp > 1
    )  # expert parallel comm factor
    ccfg.comm_cp = float(ccfg.cp > 1)  # context parallel comm factor
    ccfg.comm_dp_overlap = 0.9  # transitional overlap, see _cost_model_variables.py
    ccfg.comm_tp_overlap = 0.5  # transitional overlap, see _cost_model_variables.py


def derive_embedding_sharding(ccfg: Any) -> None:
    """Set how the embedding table is sharded, ``shard_embed``, and over how many of those
    ranks it is gathered to compute with it, ``gather_embed``.

    The table is split over tensor parallelism unless the vocabulary
    embedding runs data parallel without pipelining, and over data
    parallelism unless the config says it is not.  FSDP, which shards the
    gradients as the parameters (``grad_shard_as_params``), shards the
    table as every parameter, over :func:`optimizer_ranks`, its FSDP group
    and not the whole of data parallelism under HSDP, and gathers it over
    them to compute with it.
    """
    tp = 1 if (ccfg.vocab_emb_dp and ccfg.p == 1) else ccfg.t
    fsdp = getattr(ccfg, "grad_shard_as_params", False)
    ranks = (optimizer_ranks(ccfg) if ccfg.has_op else 1) if fsdp else ccfg.d
    ccfg.shard_embed = (ranks if ccfg.emb_dp_sharded else 1) * tp
    ccfg.gather_embed = ranks if fsdp and ccfg.emb_dp_sharded else 1


def hyper_rec_op(selective: Union[bool, list]) -> dict[str, int]:
    """Recompute switches for a HyperParallel activation checkpoint mode.

    Args:
        selective: The parsed ``sel_rec``: truthy when the run uses
            selective activation checkpointing.

    Returns:
        The switches for ``ccfg.rec_op``: :data:`HYPER_SELECTIVE_REC_OP`
        for a selective run, and every op kept otherwise.
    """
    if selective:
        return dict(HYPER_SELECTIVE_REC_OP)
    return dict.fromkeys(HYPER_SELECTIVE_REC_OP, 1)


def mindformers_rec_op(ccfg: Any, selective: Any = None) -> dict[str, int]:
    """Recompute switches of MindFormers' selective recompute.

    What ``select_recompute`` recomputes depends on flash attention and
    sequence parallelism; ``select_comm_recompute`` recomputes the
    sequence-parallel all-gather.

    Args:
        ccfg: The config, with ``sel_rec``, ``sel_comm_rec``, ``has_fa`` and
            ``sp`` set.
        selective: Whether the run recomputes selectively, ``sel_rec`` by
            default.

    Returns:
        The switches for ``ccfg.rec_op``: 1 keeps the op's activation and 0
        recomputes it.
    """
    selective = ccfg.sel_rec if selective is None else selective
    return {
        "attBMM": int(not (selective and not ccfg.has_fa and ccfg.sp > 1)),
        "headCast": int(not (selective and ccfg.has_fa)),
        "dropout": 1,
        "softmax": int(not (selective and not ccfg.has_fa)),
        "normOp": int(not (selective and ccfg.sp > 1)),
        "gather": int(not (ccfg.sel_comm_rec and ccfg.sp > 1)),
        "ffAct": int(not (selective and ccfg.sp > 1)),
    }


def derive_recompute_ranges(ccfg: Any) -> None:
    """Check the recompute ranges a config states against its layers.

    Raises:
        ValueError: When a range reaches past the model's last layer.
    """
    ranges = stated_recompute(ccfg)
    if not ranges:
        return
    layers = int(ccfg.n_lay + ccfg.n_mtp)
    for item in ranges:
        last = item.first + (item.count or 1) - 1
        if last >= layers:
            raise ValueError(
                f"{ccfg.model_name}: recompute range {item.to_dict()} reaches layer {last}, "
                f"past the model's {layers} layers"
            )


def derive_recompute_switches(ccfg: Any) -> None:
    """Set which activations a recomputed layer keeps: ``rec_op``, and each layer's own, ``layer_switches``.

    ``sel_rec_rule`` names the framework whose selective recompute the run
    uses: HyperParallel's recomputes a fixed set of ops, MindFormers' a set
    that depends on flash attention and sequence parallelism.  A selective
    recompute range that states its ops sets them instead.  Where the
    ranges state several selective settings, ``layer_switches`` gives each
    layer in model order its range's, None for a layer that is not
    selective, and the estimates read each layer's through its context,
    the per-layer channel; ``rec_op`` holds the first range's.  Otherwise
    ``layer_switches`` is None and ``rec_op`` holds the one setting.

    Raises:
        ValueError: When the config names no known rule, or it is a
            multimodal config whose ranges state several selective
            settings.
    """
    ranges = stated_recompute(ccfg)
    selective = [item for item in ranges or () if item.option == "selective"]
    run_selective = ccfg.sel_rec if ranges is None else bool(selective)
    if ccfg.sel_rec_rule == "hyperparallel":
        switches = hyper_rec_op(run_selective)
    elif ccfg.sel_rec_rule == "mindformers":
        switches = mindformers_rec_op(ccfg, run_selective)
    else:
        raise ValueError(f"Unknown selective recompute rule {ccfg.sel_rec_rule!r}")
    # Every switch, as a range's own state them: one the rule leaves out keeps its op.
    switches = {op: switches.get(op, 1) for op in RECOMPUTE_OPS}
    settings = {tuple(sorted((item.switches() or switches).items())) for item in selective}
    ccfg.layer_switches = None
    if len(settings) > 1:
        if getattr(ccfg, "multimodal", False):
            raise ValueError(
                f"{ccfg.model_name}: its recompute ranges state {len(settings)} selective settings; "
                "a multimodal config prices one"
            )
        ccfg.layer_switches = _per_layer_switches(ccfg, selective, switches)
        switches = dict(selective[0].switches() or switches)
    elif settings:
        switches = dict(next(iter(settings)))
    ccfg.rec_op = Config(switches)


def _per_layer_switches(ccfg: Any, selective: list, rule: Dict[str, int]) -> tuple:
    """Each layer's switches in model order, its selective range's or else *rule*'s; None where none covers it."""
    layers = int(ccfg.n_lay + ccfg.n_mtp)
    per_layer = [None] * layers
    for item in selective:
        stop = layers if item.count is None else min(layers, item.first + item.count)
        per_layer[item.first:stop] = [dict(item.switches() or rule)] * max(0, stop - item.first)
    return tuple(per_layer)


def derive_flash_attention_factor(ccfg: Any) -> None:
    """Set the flash attention factor, ``s_fa``.

    The attention scores are priced over ``s / a`` rather than ``s`` when
    flash attention is on.
    """
    ccfg.s_fa = ccfg.s / ccfg.a if ccfg.has_fa and ccfg.a > 0 else ccfg.s


def family_run(ccfg: Any) -> Dict[str, Any]:
    """Return the run defaults of the config's family, from its op profile.

    A vision tower takes the vision profile's, but for its activation
    sharding, which is its language model's family's.
    """
    arch = getattr(ccfg, "arch", None)
    run = dict(family_profile(arch).run)
    inherited = getattr(ccfg, "inherited_arch", None)
    if arch == VISION_ARCH and inherited is not None:
        run["shard_activations"] = family_profile(inherited).run["shard_activations"]
    return run


def _stated(ccfg: Any, name: str, run: Mapping[str, Any]) -> Any:
    """Return the run fact *name* the config states, else its family's."""
    value = getattr(ccfg, name, None)
    return run[name] if value is None else value


def derive_byte_widths(ccfg: Any, run: Mapping[str, Any]) -> None:
    """Set the byte widths the estimators read from those the run states, else its family's.

    Gradients take memory only under pipeline parallelism, unless the run
    accumulates them without it too (``grad_accumulation``, which it gives as
    ``accumulates_grads``: the search then gives PP 1 several micro-batches,
    its accumulation steps).  The optimizer
    keeps, per parameter, its states and any copy of the parameters: a
    layer's parameter as many states as its optimizer has, the embedding
    and output tables' AdamW's two.
    """
    ccfg.accumulates_grads = bool(_stated(ccfg, "grad_accumulation", run))
    accumulates = ccfg.p > 1 or ccfg.accumulates_grads
    ccfg.bytes_grad = _stated(ccfg, "grad_bytes", run) if accumulates else 0
    ccfg.bytes_os = _stated(ccfg, "optimizer_state_bytes", run)
    main_copy = _stated(ccfg, "main_param_bytes", run)
    ccfg.bytes_optim = _stated(ccfg, "optimizer_states", run) * ccfg.bytes_os + main_copy
    ccfg.bytes_optim_table = 2 * ccfg.bytes_os + main_copy
    ccfg.bytes_norm = _stated(ccfg, "norm_bytes", run)
    ccfg.bytes_dropout = _stated(ccfg, "dropout_bytes", run)


def derive_resharding(ccfg: Any, run: Mapping[str, Any]) -> None:
    """Set whether FSDP frees a layer's gathered parameters once it has run, ``reshards``,
    whether it holds each layer's reduce-scatter output until the backward ends,
    ``defers_grads``, and whether it reduces a layer's gradients while the next
    layer's backward runs, ``overlaps_grad_reduce``."""
    ccfg.reshards = bool(_stated(ccfg, "reshard_params", run))
    ccfg.defers_grads = bool(_stated(ccfg, "deferred_grad_accumulation", run))
    ccfg.overlaps_grad_reduce = bool(_stated(ccfg, "overlapped_grad_reduce", run))


def derive_activation_sharding(ccfg: Any, run: Mapping[str, Any]) -> None:
    """Set how tensor parallelism shards what a layer keeps.

    ``shard_recompute_input`` divides the input a fully recomputed layer
    keeps, and ``shard_output_activ`` the output layer's activations.  Both
    are the TP degree when the run shards the activations between layers
    (``shard_activations``), and 1 otherwise; a recomputed layer's input is
    sliced over TP too when the run's recompute slices it, and the output
    layer's activations only where the loss runs on logits sharded over the
    vocabulary (``loss_parallel``, which derive gives as ``shards_logits``):
    otherwise every rank gathers them whole.
    """
    sharded = _stated(ccfg, "shard_activations", run)
    sliced = sharded or getattr(ccfg, "recompute_slice_activation", False)
    ccfg.shards_logits = bool(_stated(ccfg, "loss_parallel", run))
    ccfg.shard_recompute_input = ccfg.t if sliced else 1
    ccfg.shard_output_activ = ccfg.t if sharded and ccfg.shards_logits else 1


def derive_qk_norm(ccfg: Any) -> None:
    """Count the QK-norm each layer runs, ``n_qknorm``: one where the model has one.

    A layer kind whose attention has none, a linear one, states 0 instead.
    """
    ccfg.n_qknorm = 1 if getattr(ccfg, "qk_norm", False) else 0


def derive_head_dim(ccfg: Any) -> None:
    """Price an MLA family's heads at the value heads' width, ``dh``.

    The model's ``v_head_dim``, else its family's.  A family whose profile
    states no value-head width keeps the head width its parser read.
    """
    width = family_profile(getattr(ccfg, "arch", None)).model.get("v_head_dim")
    if width is not None:
        ccfg.dh = getattr(ccfg, "v_head_dim", None) or width


def cm_layer_fields(ccfg: Any) -> Dict[str, Any]:
    """Return what every layer of a cm model takes on top of its kind's fields.

    Each layer shards its expert states as the model shards its partial
    ones, its other states over the common divisor of the expert count and
    their sharding, and its embedding over TP.
    """
    return {
        "shard_p_os_exp": ccfg.shard_p_os_exp_partial,
        "shard_p_os_non_exp_partial": math.gcd(ccfg.n_exp, ccfg.shard_p_os_non_exp),
        "shard_embed": ccfg.t,
    }


# The fields a family gives every layer, by family: a rule in code, where the
# family's op profile cannot say it.
LAYER_FIELD_RULES: Dict[str, Callable[[Any], Dict[str, Any]]] = {"cm": cm_layer_fields}


def derive_layer_fields(ccfg: Any) -> None:
    """Set the fields the family gives every layer, ``layer_fields``, or none."""
    rule = LAYER_FIELD_RULES.get(getattr(ccfg, "arch", None))
    ccfg.layer_fields = rule(ccfg) if rule is not None else None


def derive_family(ccfg: Any) -> None:
    """Set what the config's family decides where its producer states nothing.

    The byte widths, the resharding and the activation sharding, from the
    run facts the config states and else its family's (:func:`family_run`); an MLA
    family's head width; and the fields cm gives every layer.
    """
    run = family_run(ccfg)
    derive_byte_widths(ccfg, run)
    derive_resharding(ccfg, run)
    derive_activation_sharding(ccfg, run)
    derive_head_dim(ccfg)
    derive_layer_fields(ccfg)


def derive(ccfg: Any, strict: bool = True) -> None:
    """Compute the config's derived fields from its primary ones.

    Args:
        ccfg: The config, with its model, strategy and run facts set.
        strict: As in :func:`derive_expert_degrees`.

    Raises:
        TypeError: When *strict* and the degrees cannot hold the experts.
        ValueError: When the config names no known selective recompute rule.
    """
    derive_sequence_parallel(ccfg)
    derive_expert_degrees(ccfg, strict)
    derive_optimizer_sharding(ccfg)
    derive_param_casts(ccfg)
    derive_comm_flags(ccfg)
    derive_embedding_sharding(ccfg)
    derive_recompute_ranges(ccfg)
    derive_recompute_switches(ccfg)
    derive_flash_attention_factor(ccfg)
    derive_qk_norm(ccfg)
    derive_family(ccfg)


RECOMPUTE_MODES = ("off", "selective", "full")
# Older configs state no recompute as "none".
_MODE_ALIASES = {"none": "off"}


def read_activation_checkpoint_mode(stated: Any, where: str) -> str:
    """Read one activation checkpoint mode the way the trainer reads ``activation_checkpoint.mode``.

    Nothing stated is the trainer's default, ``"off"``, as is the older
    ``"none"``. YAML reads an unquoted ``off`` as False, which the trainer's
    config resolver turns back into ``"off"``, as here. Anything else that
    is not one of :data:`RECOMPUTE_MODES` is refused, True included: ND used
    to price a mode it did not know as no recompute, so a typo of ``full``
    was costed at the off memory, and a False fell through to the legacy
    default (I13).

    Args:
        stated: The mode as the yaml states it.
        where: Where it was stated, for the error message.

    Returns:
        One of :data:`RECOMPUTE_MODES`.

    Raises:
        ValueError: A list, True, or a name that is no mode.
    """
    if stated is None:
        return "off"
    mode = "off" if stated is False else str(stated).strip().lower()
    mode = _MODE_ALIASES.get(mode, mode)
    if isinstance(stated, (list, tuple)) or mode not in RECOMPUTE_MODES:
        raise ValueError(f"{where}: {stated!r} is not one activation checkpoint mode; expected one of "
                         f"{', '.join(RECOMPUTE_MODES)}")
    return mode

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
"""Body module"""
from __future__ import annotations
from typing import TYPE_CHECKING, NamedTuple
from hyper_parallel.auto_parallel._model_spec import KindActivations
from hyper_parallel.auto_parallel._op_records import load_op_records
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.logger import logger
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.utils import EvalUtils
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.comm import EvalLayerComm
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.layer_block import EvalAttn
from hyper_parallel.auto_parallel.sapp_nd.nd.common.cp_types import (
    CPMemoryBreakdown,
    CPAlgo,
    _resolve_cp_algo,
)
from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import (
    detect_attention_type,
    compute_kv_dim,
)
from hyper_parallel.auto_parallel.sapp_nd.nd.common.derive import runs_hyper_selective
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType

if TYPE_CHECKING:
    from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import CostModelConfig
    from hyper_parallel.auto_parallel.sapp_nd.memory_estimation._context import Context
    from typing import Optional, Tuple

# Bytes per element of the attention tensors that CP shards.
_CP_ATTENTION_SCORES_BYTES = 4
_CP_SOFTMAX_OUTPUTS_BYTES = 4
_CP_DROPOUT_MASK_BYTES = 1
_CP_KV_BYTES = 2 * 2  # fp16 key and value


class _CPAttentionTerms(NamedTuple):
    """Attention memory of one layer under CP, and what CP saves."""

    kv_cache_memory: float
    attention_scores_memory: float
    softmax_outputs_memory: float
    dropout_mask_memory: float
    s2_reduction: float
    kv_reduction: float


def _ulysses_cp_attention_terms(
    s: float, b: float, cp: float, kv_dim: float, a_per_rank: float
) -> _CPAttentionTerms:
    """Ulysses CP: each rank holds all s tokens but a/(t*cp) heads."""
    a_per_cp_rank = a_per_rank / cp
    kv_dim_per_cp_rank = kv_dim / cp
    s2_items_no_cp = (
        (_CP_ATTENTION_SCORES_BYTES + _CP_SOFTMAX_OUTPUTS_BYTES + _CP_DROPOUT_MASK_BYTES)
        * s * s * b * a_per_rank
    )
    s2_items_with_cp = (
        (_CP_ATTENTION_SCORES_BYTES + _CP_SOFTMAX_OUTPUTS_BYTES + _CP_DROPOUT_MASK_BYTES)
        * s * s * b * a_per_cp_rank
    )
    return _CPAttentionTerms(
        kv_cache_memory=_CP_KV_BYTES * s * b * kv_dim_per_cp_rank,
        attention_scores_memory=_CP_ATTENTION_SCORES_BYTES * s * s * b * a_per_cp_rank,
        softmax_outputs_memory=_CP_SOFTMAX_OUTPUTS_BYTES * s * s * b * a_per_cp_rank,
        dropout_mask_memory=_CP_DROPOUT_MASK_BYTES * s * s * b * a_per_cp_rank,
        s2_reduction=s2_items_no_cp - s2_items_with_cp,
        kv_reduction=_CP_KV_BYTES * s * b * kv_dim * ((cp - 1) / cp),
    )


def _ring_cp_attention_terms(
    s: float, b: float, cp: float, kv_dim: float, a_per_rank: float
) -> _CPAttentionTerms:
    """Ring CP: each rank holds s/cp tokens and a/t heads, KV is all-gathered."""
    return _CPAttentionTerms(
        kv_cache_memory=_CP_KV_BYTES * (s / cp) * b * kv_dim,
        attention_scores_memory=_CP_ATTENTION_SCORES_BYTES * (s / cp) * s * b * a_per_rank,
        softmax_outputs_memory=_CP_SOFTMAX_OUTPUTS_BYTES * (s / cp) * s * b * a_per_rank,
        dropout_mask_memory=_CP_DROPOUT_MASK_BYTES * (s / cp) * s * b * a_per_rank,
        s2_reduction=(
            (_CP_ATTENTION_SCORES_BYTES + _CP_SOFTMAX_OUTPUTS_BYTES + _CP_DROPOUT_MASK_BYTES)
            * s * s * b * a_per_rank * ((cp - 1) / cp)
        ),
        kv_reduction=_CP_KV_BYTES * s * b * kv_dim * ((cp - 1) / cp),
    )


class EvalBody:
    """Body layer formulas class"""

    @staticmethod
    def num_params_layer(
        ccfg: CostModelConfig, ctx: Context
    ) -> Tuple[float, float, float]:
        """Parameters count.

        Returns a 3-tuple (non_exp, routed, shared):
          - non_exp: attention + norm params (and dense FFN if n_exp==1,
            the router otherwise)
          - routed:  routed expert params (0 if n_exp==1)
          - shared:  shared expert params  (0 if n_shared_exp==0 or no pointer)
        """
        non_exp = ctx.attn_num_p(ccfg, ctx) + ctx.norm_num_p(ccfg, ctx)
        routed = 0.0
        shared = 0.0
        if ccfg.n_exp == 1:
            non_exp += ctx.ffn_num_p(ccfg, ctx)
        else:
            if ctx.ffn_router_num_p is not None:
                non_exp += ctx.ffn_router_num_p(ccfg, ctx)
            if ctx.ffn_routed_num_p is not None:
                routed = ctx.ffn_routed_num_p(ccfg, ctx)
            if ctx.ffn_shared_num_p is not None:
                shared = ctx.ffn_shared_num_p(ccfg, ctx)
        return (non_exp, routed, shared)

    @staticmethod
    def stat_p_layer(ccfg: CostModelConfig, ctx: Context) -> float:
        """model param"""
        non_exp_p, routed_p, shared_p = ctx.eval.num_p(ccfg, ctx)
        # Routed experts: EP sharding
        routed_mem = routed_p / ccfg.ep * ccfg.bytes_p / ccfg.shard_p_os_exp
        # Shared experts: partial DP sharding
        shared_mem = shared_p * ccfg.bytes_p / ccfg.shard_p_os_exp_partial
        # Non expert
        non_exp_mem = non_exp_p * ccfg.bytes_p / ccfg.shard_p_os_non_exp_partial
        return non_exp_mem + routed_mem + shared_mem

    @staticmethod
    def stat_os_layer(ccfg: CostModelConfig, ctx: Context) -> float:
        """optim state"""
        if ctx.swap_os:
            return 0
        non_exp_p, routed_p, shared_p = ctx.eval.num_p(ccfg, ctx)
        # Routed experts
        routed_mem = routed_p / ccfg.ep * ccfg.bytes_optim / ccfg.shard_p_os_exp
        # Shared experts
        shared_mem = shared_p * ccfg.bytes_optim / ccfg.shard_p_os_exp_partial
        # Non expert
        non_exp_mem = non_exp_p * ccfg.bytes_optim / ccfg.shard_p_os_non_exp_partial
        return non_exp_mem + routed_mem + shared_mem

    @staticmethod
    def stat_grad_layer(ccfg: CostModelConfig, ctx: Context) -> float:
        """gradients"""
        non_exp_p, routed_p, shared_p = ctx.eval.num_p(ccfg, ctx)
        # Routed experts
        routed_mem = routed_p / ccfg.ep * ccfg.bytes_grad / ccfg.shard_grad_exp
        # Shared experts: use shard_grad_exp_partial (independent of os sharding)
        shared_mem = shared_p * ccfg.bytes_grad / ccfg.shard_grad_exp_partial
        # Non expert
        non_exp_mem = non_exp_p * ccfg.bytes_grad / ccfg.shard_grad_non_exp
        return non_exp_mem + routed_mem + shared_mem

    @staticmethod
    def reduced_grad_layer(ccfg: CostModelConfig, ctx: Context) -> Tuple[float, float]:
        """The layer's gradients FSDP reduce-scatters, whole as its backward computes them and sharded."""
        non_exp_p, routed_p, shared_p = ctx.eval.num_p(ccfg, ctx)
        return EvalUtils.reduced_grads(ccfg, (
            (non_exp_p, ccfg.t, ccfg.shard_grad_non_exp),
            (routed_p / ccfg.ep, ccfg.t_exp, ccfg.shard_grad_exp),
            (shared_p, ccfg.t_exp, ccfg.shard_grad_exp_partial),
        ))

    # No recompute and select recompute

    @staticmethod
    def layer_activ(ccfg: CostModelConfig, ctx: Context) -> float:
        """activations"""
        census = getattr(ccfg, "kind_activations", None)
        if isinstance(census, KindActivations) and (
                ctx.current_node != LayerType.SEL_REC_LAYER or census.ops is not None
                or census.selective is not None and runs_hyper_selective(ccfg, EvalUtils.switches(ccfg, ctx))):
            return EvalBody.census_activ(ccfg, ctx, census)
        attn_size = sum(
            [
                ctx.attn_qkv_activ(ccfg, ctx),
                ctx.attn_score_activ(ccfg, ctx),
                ctx.attn_proj_activ(ccfg, ctx),
            ]
        )
        if ccfg.n_exp == 1:
            ffn_size = ctx.ffn_activ(ccfg, ctx)
        else:
            ffn_size = ctx.ffn_moe_activ(ccfg, ctx)
        norm_size = ctx.norm_activ(ccfg, ctx)
        return attn_size + ffn_size + norm_size

    @staticmethod
    def census_activ(ccfg: CostModelConfig, ctx: Context, census: KindActivations) -> float:
        """A layer's activations, as the census of its kind measured them.

        What the layer keeps between its passes, or its backward's working
        set (:meth:`EvalUtils.census_bytes`), per token of a CP rank's share
        of the sequence: TP splits one part, sequence parallelism the
        other, and CP that gathers keys and values leaves them whole
        (:meth:`EvalAttn.kv_shards`).  A selective layer keeps what
        :meth:`census_kept` says, and its backward holds what the plain
        layer's does.
        """
        tokens = ctx.micro_factor * ccfg.s * ccfg.b / max(1, ccfg.cp)
        held = census.working / max(1, ccfg.sp) + census.working_tp / max(1, ccfg.t)
        # A census counts a rank's share of the keys and values; where CP
        # gathers them the attention keeps the rest of the sequence's too.
        gathered = EvalAttn.gathered_kv_bytes(ccfg)
        if ctx.current_node == LayerType.SEL_REC_LAYER:
            return EvalUtils.census_bytes(ctx, tokens, EvalBody.census_kept(ccfg, ctx, census), held + gathered)
        kept = census.saved / max(1, ccfg.sp) + census.saved_tp / max(1, ccfg.t)
        return EvalUtils.census_bytes(ctx, tokens, kept + gathered, held + gathered)

    @staticmethod
    def census_kept(ccfg: CostModelConfig, ctx: Context, census: KindActivations) -> float:
        """What a selective layer of a kind *census* prices keeps, bytes per token of a CP rank's share.

        Under HyperParallel's selective checkpointing, whose switches it has
        (:func:`runs_hyper_selective`), what the census measured under it;
        the keys and values CP gathers it gathers again to recompute.  Under
        other switches, what the census's records per op state for each op
        they keep (:func:`load_op_records`): an op no switch names, and the
        census's ``other``, whatever they say; the keys and values CP
        gathers where the attention's batched matmuls are kept.  Split as
        what the plain layer keeps (:meth:`census_activ`).
        """
        if census.selective is not None and runs_hyper_selective(ccfg, EvalUtils.switches(ccfg, ctx)):
            return census.selective / max(1, ccfg.sp) + census.selective_tp / max(1, ccfg.t)
        records = load_op_records().ops

        def keeps(op: str) -> bool:
            """Whether the layer keeps what the census states for *op*."""
            switch = records[op].switch if op in records else None
            return switch is None or bool(EvalUtils.switch(ccfg, ctx, switch))

        kept = sum(size for op, size in census.ops.items() if keeps(op)) / max(1, ccfg.sp)
        kept += sum(size for op, size in census.ops_tp.items() if keeps(op)) / max(1, ccfg.t)
        return kept + (EvalAttn.gathered_kv_bytes(ccfg) if keeps("attBMM") else 0)

    # Full recompute

    @staticmethod
    def fullrec_layer_activ(ccfg: CostModelConfig, ctx: Context) -> float:
        """activations"""
        micro_factor = ctx.micro_factor
        forward_activation = (
            micro_factor * ccfg.bytes_compute * ccfg.s * ccfg.b * ccfg.h
        )
        forward_activation /= ccfg.shard_recompute_input
        return forward_activation

    @staticmethod
    def fullrec_layer_activ_gradclip(
        ccfg: CostModelConfig, ctx: Context
    ) -> float:
        """special case with gradient clipping"""
        non_exp_p, routed_p, shared_p = ctx.eval.num_p(ccfg, ctx)
        grad_clip_mem = (
            non_exp_p
            + routed_p / ccfg.ep * ccfg.bytes_os / ccfg.shard_p_os_exp
            + shared_p * ccfg.bytes_os / ccfg.shard_p_os_exp_partial
        )
        grad_clip_mem *= ccfg.bytes_os / ccfg.shard_p_os_non_exp_partial
        grad_clip_mem *= int(ccfg.has_clip)
        forward_activation = EvalBody.fullrec_layer_activ(ccfg, ctx)
        dp_comm_size = ctx.eval.dyn.comm.dp(ccfg, ctx)
        if forward_activation + dp_comm_size > grad_clip_mem:
            return forward_activation
        logger.debug(
            "gradient clipping %s > %s",
            EvalUtils.mb(grad_clip_mem),
            EvalUtils.mb(forward_activation + dp_comm_size),
        )
        return grad_clip_mem

    @staticmethod
    def fullrec_layer_comm_gradclip(
        ccfg: CostModelConfig, ctx: Context
    ) -> float:
        """special case with gradient clipping"""
        if EvalBody.fullrec_layer_activ_gradclip(ccfg, ctx) > 0:
            return ctx.eval.dyn.comm.dp(ccfg, ctx)
        return 0

    @staticmethod
    def act_cp_layer(
        ccfg: CostModelConfig,
        ctx: Context
    ) -> CPMemoryBreakdown:
        """Estimate CP activation memory impact for one transformer layer.

        Ring CP (colossalai_cp / hybrid_cp):
            Each rank holds s/cp tokens (Q sharded along seq) and a/t heads;
            KV is all-gathered, so only one S dim of the S² score tensor
            is divided.
            KV cache:   (s/cp) × b × kv_dim_per_rank
            Attn scores: (s/cp) × s × b × (a/t)

        Ulysses CP:
            Each rank holds all s tokens but a/(t*cp) heads.
            KV cache:   s × b × kv_dim_per_rank / cp
            Attn scores: s × s × b × (a/(t*cp))
        """
        s, b = ccfg.s, ccfg.b
        a = ccfg.a
        t = max(1, ccfg.t)

        if a <= 0:
            raise ValueError(f"Number of attention heads must be positive, got {a}")

        cp = ccfg.cp

        if cp <= 0:
            raise ValueError(f"CP degree must be positive, got {cp}")

        attention_type = detect_attention_type(ccfg)
        cp_algo = _resolve_cp_algo(ccfg)

        kv_dim = compute_kv_dim(ccfg)
        a_per_rank = a / t

        if cp_algo == CPAlgo.ULYSSES_CP:
            terms = _ulysses_cp_attention_terms(s, b, cp, kv_dim, a_per_rank)
        else:
            terms = _ring_cp_attention_terms(s, b, cp, kv_dim, a_per_rank)

        comm_buffer = EvalLayerComm.cp_comm_buffer(ccfg, ctx)

        total_memory = (
            terms.kv_cache_memory + terms.attention_scores_memory +
            terms.softmax_outputs_memory + terms.dropout_mask_memory + comm_buffer
        )
        total_reduction = terms.s2_reduction + terms.kv_reduction - comm_buffer

        return CPMemoryBreakdown(
            kv_cache_memory=terms.kv_cache_memory,
            attention_scores_memory=terms.attention_scores_memory,
            softmax_outputs_memory=terms.softmax_outputs_memory,
            dropout_mask_memory=terms.dropout_mask_memory,
            comm_buffer_memory=comm_buffer,
            kv_reduction=terms.kv_reduction,
            s2_reduction=terms.s2_reduction,
            total_reduction=total_reduction,
            total_memory=total_memory,
            cp_degree=int(cp),
            seq_len=int(s),
            attention_type=attention_type,
            cp_algo=cp_algo,
        )

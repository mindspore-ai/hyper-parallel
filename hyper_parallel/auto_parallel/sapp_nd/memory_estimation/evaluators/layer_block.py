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
"""Layer's blocks submodule"""
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Mapping, Optional
from hyper_parallel.auto_parallel._op_records import load_op_records
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType
from hyper_parallel.auto_parallel.sapp_nd.nd.common.cp_types import CPAlgo, _resolve_cp_algo
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.utils import EvalUtils

if TYPE_CHECKING:
    from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import CostModelConfig
    from hyper_parallel.auto_parallel.sapp_nd.memory_estimation._context import Context

mb = EvalUtils.mb


def _bias(ccfg: CostModelConfig, stated: str, width: float) -> float:
    """A projection's bias: *width* where the model states one, none where it states none.

    Where the model states neither, as a parser that reads no bias does,
    each projection is counted a bias of the hidden width, the formulas'
    convention.
    """
    has_bias = getattr(ccfg, stated, None)
    if has_bias is None:
        return ccfg.h
    return width if has_bias else 0


def _mlp_biases(ccfg: CostModelConfig, width: float) -> float:
    """The biases of one feed-forward of *width*, or one expert's, as :func:`_bias` counts a projection's.

    Stated, each projection into the width has one of it and the last,
    back to the hidden width, one of that; unstated, each projection one of
    the width.
    """
    has_bias = getattr(ccfg, "mlp_bias", None)
    if has_bias is None:
        return ccfg.n_ffMM * width
    return (ccfg.n_ffMM - 1) * width + ccfg.h if has_bias else 0


class EvalAttn:
    """Attention formulas class"""

    @staticmethod
    def num_params_mla(ccfg: CostModelConfig, _) -> float:
        """Parameters count for Multi-Head Latent Attention.

        The queries: a down-projection to their latent, its norm and an
        up-projection to every head's non-rotary and rotary part, or one
        projection where the model has no query latent.  The keys and
        values: one down-projection to their shared latent beside the
        rotary key, its norm, and an up-projection to every head's
        non-rotary key and value.  The output projection from the values.
        Each head's non-rotary key is ``qk_nope_head_dim`` wide, ``dh``,
        its value head's width, unless the model states it.
        """
        d_qk = getattr(ccfg, "qk_nope_head_dim", None) or ccfg.dh
        heads_q = ccfg.a * (d_qk + ccfg.dhr)
        if ccfg.dc_q:
            query = ccfg.h * ccfg.dc_q + ccfg.dc_q + ccfg.dc_q * heads_q
        else:
            query = ccfg.h * heads_q
        key_value = (ccfg.h * (ccfg.dc_kv + ccfg.dhr) + ccfg.dc_kv
                     + ccfg.dc_kv * ccfg.n_kv * (d_qk + ccfg.dh))
        output = ccfg.a * ccfg.dh * ccfg.h
        return 0.25 * ccfg.n_attMM * (query + key_value + output)

    @staticmethod
    def num_params_attn(ccfg: CostModelConfig, ctx: Context) -> float:
        """Parameters count for Multi-Head/Grouped-Q./Multi-Q. Attention"""
        if ccfg.dc_kv == 0:
            # Q,O and K,V have distinct shapes. Q and O are h x (a*dh),
            # which equals h x h only when head_dim = h/a; a fused output
            # gate (Qwen3.5) doubles the Q projection.
            d_h = ccfg.dh or (ccfg.h / ccfg.a if ccfg.a else 0)
            d_q = ccfg.a * d_h
            q_fact = 2 if ccfg.attn_output_gate else 1
            return 0.25 * ccfg.n_attMM * (
                q_fact * ccfg.h * d_q + _bias(ccfg, "qkv_bias", q_fact * d_q)
            ) + 0.25 * ccfg.n_attMM * (
                ccfg.h * d_q + _bias(ccfg, "o_bias", ccfg.h)
            ) + 0.5 * ccfg.n_attMM * (
                ccfg.h * ccfg.n_kv * d_h + _bias(ccfg, "qkv_bias", ccfg.n_kv * d_h)
            ) + ccfg.attn_extra_p
        return EvalAttn.num_params_mla(ccfg, ctx)

    @staticmethod
    def kv_shards(ccfg: CostModelConfig) -> float:
        """Over how many CP ranks a layer's keys and values are split.

        Colossal-AI and hybrid CP all-gather the keys and values, and the
        attention kernel keeps them whole for the backward: none.  Ulysses
        CP gives each rank its heads' share, and a linear-attention layer
        passes its state rather than gathering them: all of CP.
        """
        if _resolve_cp_algo(ccfg) == CPAlgo.ULYSSES_CP or getattr(ccfg, "n_linrec", 0):
            return ccfg.cp
        return 1

    @staticmethod
    def gathered_kv_bytes(ccfg: CostModelConfig) -> float:
        """The bytes per token of a rank's share of the sequence it keeps of the other ranks' keys and values.

        Where CP gathers the keys and values (:meth:`kv_shards`), the
        attention keeps the whole sequence's, the rank's own share and
        ``cp - 1`` others, each split over TP.
        """
        if ccfg.cp <= 1 or EvalAttn.kv_shards(ccfg) > 1:
            return 0
        return (ccfg.cp - 1) * 2 * ccfg.n_kv * ccfg.dh * ccfg.bytes_compute / max(1, ccfg.t)

    @staticmethod
    def attn_qkv_activations(ccfg: CostModelConfig, ctx: Context) -> float:
        """QKV linear Activations: the op records' ``qkv`` slot"""
        return EvalRecords.slot_bytes(ccfg, ctx, "qkv")

    @staticmethod
    def attn_score_activations(ccfg: CostModelConfig, ctx: Context) -> float:
        """Score/Softmax Activations: the op records' ``score`` slot.

        Ring CP shards the queries along the sequence and gathers the keys and
        values, so the score tensor has one sequence dimension divided, B H S^2
        / cp; only a blockwise ring, which shards both, would divide it again.
        """
        return EvalRecords.slot_bytes(ccfg, ctx, "score")

    @staticmethod
    def attn_proj_activations(ccfg: CostModelConfig, ctx: Context) -> float:
        """Output projection Activations: the op records' ``proj`` slot"""
        return EvalRecords.slot_bytes(ccfg, ctx, "proj")


class EvalFFn:
    """Feed-forward formulas class"""

    @staticmethod
    def num_params_ffn(ccfg: CostModelConfig, _) -> float:
        """Parameters count"""
        return (ccfg.n_exp + ccfg.n_shared_exp) * (
            ccfg.n_ffMM * ccfg.hff * ccfg.h + _mlp_biases(ccfg, ccfg.hff))

    @staticmethod
    def num_params_routed_expert(ccfg: CostModelConfig, _) -> float:
        """Routed expert parameters count (with ETP correction)"""
        hff_sliced = ccfg.hff_exp / max(ccfg.etp, 1)
        return ccfg.n_exp * (ccfg.n_ffMM * hff_sliced * ccfg.h + _mlp_biases(ccfg, hff_sliced))

    @staticmethod
    def num_params_router(ccfg: CostModelConfig, _: Any) -> float:
        """The router's parameters: a weight per routed expert, the hidden width wide, where the layer routes."""
        return ccfg.h * ccfg.n_exp if ccfg.n_exp > 1 else 0

    @staticmethod
    def num_params_shared_expert(ccfg: CostModelConfig, _) -> float:
        """Shared expert parameters count, and the weight that gates its output where the model has one.

        Every producer states the shared experts as a count of experts of
        the routed ones' width, ``hff_exp``: one wide shared expert as that
        many.
        """
        gate = ccfg.h if ccfg.n_shared_exp and getattr(ccfg, "shared_expert_gate", None) else 0
        width = ccfg.hff_exp
        return ccfg.n_shared_exp * (ccfg.n_ffMM * width * ccfg.h + _mlp_biases(ccfg, width)) + gate

    @staticmethod
    def ffn_activations(ccfg: CostModelConfig, ctx: Context, width: Optional[float] = None) -> float:
        """Activations of a feed-forward *width* wide, the model's dense width unless given: the ``ffn`` slot"""
        return EvalRecords.slot_bytes(ccfg, ctx, "ffn", overrides=None if width is None else {"hff": width})

    @staticmethod
    def ffn_router_and_concat_activations(
        ccfg: CostModelConfig, ctx: Context
    ) -> float:
        """MoE router and output activations: the ``router`` slot"""
        return EvalRecords.slot_bytes(ccfg, ctx, "router")

    @staticmethod
    def shared_exp_activations(ccfg: CostModelConfig, ctx: Context) -> float:
        """Shared expert activations, each shared expert of the routed ones' width: the ``shared`` slot"""
        return EvalRecords.slot_bytes(ccfg, ctx, "shared")

    @staticmethod
    def routed_exp_activations(ccfg: CostModelConfig, ctx: Context) -> float:
        """MoE topK activations, each expert at its width: the ``routed`` slot"""
        return EvalRecords.slot_bytes(ccfg, ctx, "routed")

    @staticmethod
    def ffn_moe_activations(ccfg: CostModelConfig, ctx: Context) -> float:
        """Sum of Activations"""
        return (
            EvalFFn.routed_exp_activations(ccfg, ctx)
            + EvalFFn.shared_exp_activations(ccfg, ctx)
            + EvalFFn.ffn_router_and_concat_activations(ccfg, ctx)
        )


class EvalNorm:
    """Normalization formulas class"""

    @staticmethod
    def head_dim(ccfg: CostModelConfig) -> float:
        """The width a QK-norm normalizes over: one attention head's"""
        return ccfg.dh or (ccfg.h / ccfg.a if ccfg.a else 0)

    @staticmethod
    def num_params_norm(ccfg: CostModelConfig, _) -> float:
        """Parameters count: the layer's norms, and a QK-norm's query and key weights.

        A model that states its norms holds a weight in each, and a bias
        beside it in a LayerNorm; one that does not is counted two vectors
        per norm op.
        """
        layer_norms = getattr(ccfg, "layer_norms", None)
        if layer_norms is None:
            vectors = ccfg.n_normOp * 2
        else:
            vectors = layer_norms * (2 if getattr(ccfg, "norm_bias", None) else 1)
        return vectors * ccfg.h + getattr(ccfg, "n_qknorm", 0) * 2 * EvalNorm.head_dim(ccfg)

    @staticmethod
    def norm_activations(ccfg: CostModelConfig, ctx: Context) -> float:
        """Activations: the norms' inputs, a QK-norm's every head's queries and keys: the ``norm`` slot"""
        return EvalRecords.slot_bytes(ccfg, ctx, "norm")


class EvalRecords:
    """The op records (shared decision S1), priced on a layer config.

    The activation formulas of :class:`EvalAttn`, :class:`EvalFFn` and
    :class:`EvalNorm` are the records' slots, so a caller prices a layer
    under any setting of the recompute switches, or one op alone, without
    setting anything on its config.
    """

    @staticmethod
    def slot_bytes(ccfg: CostModelConfig, ctx: Context, slot: str,
                   switches: Optional[Mapping[str, Any]] = None,
                   overrides: Optional[Mapping[str, Any]] = None, only: Optional[str] = None) -> float:
        """The bytes a layer of *ccfg* keeps in *slot*, one rank's share of a micro-batch.

        Args:
            ccfg: The layer's config.
            ctx: The evaluation: its micro factor, and its node, a selective
                layer's (``SEL_REC_LAYER``) dropping what its switches drop.
            slot: The slot.
            switches: A selective layer's switches, 1 to keep an op's
                activations and 0 to drop them; where omitted, those the
                context carries, else the config's own.
            overrides: Values that stand for the config's fields.
            only: An op to price alone, its parts with it, or a part, whatever
                the switches say.
        """
        records = load_op_records()
        rec_layer = ctx.current_node == LayerType.SEL_REC_LAYER
        stated = EvalUtils.switches(ccfg, ctx) if switches is None else switches

        def keep(op: str) -> Any:
            """Whether this layer keeps *op*'s activation: 1 keeps it, 0 recomputes it."""
            switch = records.record(op).switch
            if switch is None:
                return 1
            return EvalUtils.rec_coeff(rec_layer, EvalUtils.state(stated, switch))

        return records.evaluate(slot, EvalRecords.values(ccfg, ctx, overrides), keep, only)

    @staticmethod
    def values(ccfg: CostModelConfig, ctx: Context, overrides: Optional[Mapping[str, Any]] = None) -> Any:
        """The value of each name a record reads: an override, the evaluation's, or the config's field."""
        records = load_op_records()
        overrides = overrides or {}

        def value(name: str) -> Any:
            """The value of *name*: an override, the evaluation's, a record's default or the config's field."""
            if name in overrides:
                return overrides[name]
            if name in ("micro_factor", "dropless_tok_factor"):
                return getattr(ctx, name)
            if name == "kv_shards":
                return EvalAttn.kv_shards(ccfg)
            if name in records.defaults:
                return getattr(ccfg, name, records.defaults[name])
            return getattr(ccfg, name)

        return value

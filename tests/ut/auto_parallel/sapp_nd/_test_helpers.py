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
"""Shared test helpers for sapp_nd unit tests."""
# pylint: disable=missing-class-docstring
from types import SimpleNamespace
from typing import Any

from hyper_parallel.auto_parallel.sapp_nd.memory_estimation._context import (
    Context,
    NodeCommEval,
    NodeDynEval,
    NodeEval,
    NodeStatEval,
)
from hyper_parallel.auto_parallel.sapp_nd.nd.common.framework_parsers._cost_model_parser import (
    _CostModelParser,
)
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType


def _make_ccfg(d: int = 8,
    t: int = 2,
    cp: int = 1,
    ep: int = 1,
    n_exp: int = 1,
    d_shard: int = 0,
    has_op: bool = True,
    has_grad_shard: bool = True,
    fsdp: bool = True,
    os_max_shard: int = 0,
    **kwargs: Any,
) -> SimpleNamespace:
    """Create a SimpleNamespace mimicking a cost-model config for tests."""
    fsdp_active = fsdp and d_shard != d
    if d_shard <= 0:
        d_shard = d if fsdp else 1
    d_replicate = d // d_shard
    d_exp = d // ep if d >= ep else d * t // ep
    t_exp = t
    shard_ne = d_shard * cp * t
    shard_exp = (d_exp * ep) * cp * t_exp
    shard_grad_ne = shard_ne
    shard_grad_exp = shard_exp
    shard_embed_val = t * d
    defaults = {
        "config": None, "d": d, "t": t, "cp": cp, "ep": ep, "n_exp": n_exp,
        "d_exp": d_exp, "t_exp": t_exp, "d_shard": d_shard,
        "d_shard_or_d": d_shard if d_shard > 0 else d, "d_replicate": d_replicate,
        "comm_hsdp": 1.0 if 1 < d_shard < d else 0.0,
        "has_op": has_op, "has_grad_shard": has_grad_shard, "fsdp": fsdp,
        "os_max_shard": os_max_shard if os_max_shard else d * t, "sp": 1,
        "h": 16, "hff": 32, "hff_exp": 64, "s": 8, "b": 1, "v": 64,
        "bytes_p": 2, "bytes_compute": 2, "bytes_os": 12, "bytes_grad": 2,
        "bytes_norm": 4, "bytes_softmax": 4,
        "comm_d_non_exp": 3, "comm_d_exp": 3, "comm_t": 1.0,
        "comm_ep": 0.0, "comm_cp": 0.0,
        "comm_fsdp": 1.0 if fsdp_active else 0.0,
        "fsdp_all_gather_buffer": 1.0 if fsdp else 0.0,
        "framework_overhead": 0,
        "shard_p_os_non_exp_partial": shard_ne, "shard_p_os_non_exp": shard_ne,
        "shard_grad_non_exp": shard_grad_ne,
        "shard_p_os_exp_partial": shard_exp, "shard_p_os_exp": shard_exp,
        "shard_grad_exp": shard_exp, "shard_grad_exp_partial": shard_grad_exp,
        "shard_p_fsdp_non_exp": shard_ne if fsdp else 0,
        "shard_os_fsdp_non_exp": shard_ne if fsdp else 0,
        "shard_grad_fsdp_non_exp": shard_grad_ne if fsdp else 0,
        "shard_p_fsdp_exp": shard_exp if fsdp else 0,
        "shard_os_fsdp_exp": shard_exp if fsdp else 0,
        "shard_grad_fsdp_exp": shard_grad_exp if fsdp else 0,
        "shard_embed": shard_embed_val,
        "shard_output_activ": 1, "shard_recompute_input": 1,
        "n_chosen_exp": 2, "n_shared_exp": 1, "cap_fact": 1,
        "overlap_dp": True, "n_lay": 2, "n_mtp": 1,
        "p": 1, "vp": 1, "m": 1, "gbs": 8, "pp_sched": "1f1b",
        "layer_custom_config": [(2, None)],
        "dc_kv": 0, "dc_q": 0, "dh": 8, "dhr": 0, "is_shard_mtp_param": True,
        "n_attMM": 1, "n_ffMM": 1, "n_attBMM": 1, "n_ffBMM": 1,
        "n_softmax": 1, "n_headCast": 1, "n_gather": 1, "n_ffAct": 1,
        "n_normOp": 1, "n_dropout": 1,
        "rec_op": SimpleNamespace(
            attMM=1, ffMM=1, attBMM=1, ffBMM=1, softmax=1,
            headCast=1, gather=1, ffAct=1, normOp=1, dropout=1,
        ),
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _make_ctx_with_num_p(
    non_exp: float = 100.0,
    exp: float = 50.0,
    num_p_returns_tuple: bool = True,
    routed: float = 0.0,
    shared: float = 0.0,
) -> Context:
    """Create a Context with a stubbed num_p evaluator."""
    if routed == 0.0 and shared == 0.0 and exp > 0.0:
        routed = exp

    if num_p_returns_tuple:
        def _fake_num_p(ccfg: Any, ctx: Any) -> tuple:
            return non_exp, routed, shared
    else:
        def _fake_num_p(ccfg: Any, ctx: Any) -> float:
            return non_exp

    def _eval_fn(*_a, **_k):
        return 1
    stat_eval = NodeStatEval(_eval_fn, _eval_fn, _eval_fn)
    comm_eval = NodeCommEval(_eval_fn, _eval_fn, _eval_fn, _eval_fn, _eval_fn, _eval_fn)
    dyn_eval = NodeDynEval(_eval_fn, comm_eval)
    node_eval = NodeEval(_fake_num_p, stat_eval, dyn_eval)
    ctx = Context()
    ctx.node_eval[LayerType.NOT_REC_LAYER] = node_eval
    ctx.current_node = LayerType.NOT_REC_LAYER
    return ctx


# Backward-compatible aliases
_make_ccfg_hsdp = _make_ccfg
_make_ccfg_fsdp = _make_ccfg


class _ConcreteParser(_CostModelParser):
    def parse(self) -> None:
        pass

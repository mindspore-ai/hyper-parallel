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
"""Getters"""

from copy import copy
from typing import Any, Dict, List, Mapping, Optional, Tuple
from hyper_parallel.auto_parallel.sapp_nd.nd.common.arch_hooks import CWrap, apply_layer_kind, layer_groups
from hyper_parallel.auto_parallel.sapp_nd.nd.common.derive import runs_hyper_selective
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_order import get_model_order, layer_switches
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType
from hyper_parallel.auto_parallel.sapp_nd.nd.logger import perf_logger as logger


def get_layer_group_configs(cfg: Any) -> List[Tuple[Any, int]]:
    """Return each group of layers' config and layer count, in model order.

    A config whose layers run on it as it stands is one group, priced with
    ``cfg`` itself.  Otherwise each kind is applied once, to a copy of
    ``cfg`` that every group of that kind shares.  A kind sets top-level
    fields only (``apply_layer_kind``), so a shallow copy leaves ``cfg``
    intact, as ``comm_time``'s walk copies it for the same reason; a deep
    copy cost 0.36 ms, twice a candidate and once a kind.
    """
    groups = layer_groups(cfg)
    if all(kind is None for kind, _ in groups):
        return [(cfg, cfg.n_lay)]
    configs = {}
    for kind, _ in groups:
        if kind.name not in configs:
            configs[kind.name] = copy(cfg)
            apply_layer_kind(CWrap(configs[kind.name]), kind)
    return [(configs[kind.name], count) for kind, count in groups]


def get_layer_configs_by_position(cfg: Any, stages: List) -> Dict[Tuple[int, int, int], Any]:
    """Map each regular layer's position to the config it is priced with.

    Each group of the layer stack covers exactly its own count of layers in
    model order; layers past the stack's count keep the last group, as a
    config whose layers need no kind keeps ``cfg``.
    """
    lccfgs = get_layer_group_configs(cfg)
    per_layer = [lccfg for lccfg, count in lccfgs for _ in range(count)]
    last = lccfgs[-1][0]
    return {
        position: per_layer[idx] if idx < len(per_layer) else last
        for idx, position in enumerate(get_model_order(cfg, stages))
    }


def get_layer_switches_by_position(cfg: Any, stages: List) -> Dict[Tuple[int, int, int], Any]:
    """Map each regular layer's position to its own recompute switches, where its config's ranges state several.

    Empty where the config's ``rec_op`` holds the one setting every
    selective layer runs (:func:`layer_switches`).
    """
    switches = layer_switches(cfg)
    if switches is None:
        return {}
    return {position: switches[idx] for idx, position in enumerate(get_model_order(cfg, stages))
            if idx < len(switches)}


# The switch an op answers to where it has none of its own: a QK-norm is a norm.
_SWITCH_OF = {"qknorm": "normOp"}

# The ops whose share of FLOPs a selective layer of HyperParallel's policy
# runs again a census states, and the fields that state them.
_CENSUS_SHARES = {"attMM": "selective_attention_mm", "ffMM": "selective_ffn_mm"}


def mla_weights(cfg: Any) -> Tuple[Any, Any]:
    """An MLA layer's projection weights, all of them and its up-projections', as the time model prices them.

    The queries' latent and its up-projection, or one projection to every
    head; the keys' and values' shared down-projection beside the rotary
    key; their up-projections, a head's key at its own width and its value
    at the value heads'; the output projection.  The up-projections build
    the queries' heads from their latent, where they have one, and the
    keys' and values' from theirs.

    Returns:
        ``(weights, up)``.
    """
    d_nope = getattr(cfg, "qk_nope_head_dim", None) or cfg.dh
    heads = cfg.a * (d_nope + cfg.dhr)
    query = cfg.dc_q * (cfg.h + heads) if cfg.dc_q else cfg.h * heads
    weights = (query + cfg.h * (cfg.dc_kv + cfg.dhr) + cfg.dc_kv * cfg.n_kv * (d_nope + cfg.dh)
               + cfg.a * cfg.dh * cfg.h)
    up = (cfg.dc_q * heads if cfg.dc_q else 0) + cfg.dc_kv * cfg.n_kv * (d_nope + cfg.dh)
    return weights, up


def selective_shares(lccfg: Any, layer: Any, switches: Optional[Mapping[str, int]] = None) -> Dict[str, float]:
    """The share of each matmul op's FLOPs a selective layer runs again, where no switch prices it.

    A layer of HyperParallel's selective policy, whose switches it has
    (:func:`runs_hyper_selective`), *switches* where the layer has its own,
    runs again the shares its kind's census measured: the policy recomputes
    every other matmul, which no switch covers.  An MLA layer whose
    ``attUp`` switch is 0 runs its up-projections again, their share of its
    projections' FLOPs.  Empty otherwise, and the switches price every op.
    """
    if layer != LayerType.SEL_REC_LAYER:
        return {}
    census = getattr(lccfg, "kind_activations", None)
    if census is not None and runs_hyper_selective(lccfg, switches):
        return {op: getattr(census, name) for op, name in _CENSUS_SHARES.items()
                if getattr(census, name, None) is not None}
    if not getattr(lccfg, "dc_kv", 0):
        return {}
    stated = switches if switches is not None else vars(lccfg.rec_op) if lccfg.rec_op is not None else {}
    if stated.get("attUp", 1):
        return {}
    weights, up = mla_weights(lccfg)
    return {"attMM": up / weights}


def get_recomp_factor(lccfg, layer, op_name, switches=None):
    """Whether a layer of this type runs the op again in its backward pass.

    A selective layer recomputes exactly the ops whose switch in
    ``lccfg.rec_op`` is 0. A switch at 1 keeps the op's activation, which is
    how the memory model's ``EvalUtils.rec_coeff`` reads it, and an op with no
    switch is kept. The switches are read from ``vars`` because a ``Config``
    answers 0 for any attribute it lacks.  *switches*, a layer's own where
    it has them, stand for the config's.
    """
    if layer == LayerType.FULL_REC_LAYER:
        return 1
    if layer == LayerType.NOT_REC_LAYER:
        return 0
    if layer == LayerType.SEL_REC_LAYER:
        if switches is None:
            switches = vars(lccfg.rec_op) if lccfg.rec_op is not None else {}
        return int(not switches.get(_SWITCH_OF.get(op_name, op_name), 1))
    logger.warning("Unrecognized recompute type %s", layer)
    return 0


def get_table_quantity(lccfg, table, layer, with_recomp, shares=None, switches=None):
    """op compute load from given table; *shares* sets the recompute factor of the ops it names, *switches* the rest"""
    shares = shares or {}
    qt_layer = 0
    for op, quantity in table.items():
        op_name = op[2:]
        factor = shares[op_name] if op_name in shares else get_recomp_factor(lccfg, layer, op_name, switches)

        qt_layer += (
            (1 + with_recomp * factor)
            * getattr(lccfg, op)
            * quantity
        )
    return qt_layer

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
"""Verify mode: the parameters and FLOPs ND prices of a model, beside those of the layers Transformers builds of it.

``run_nd -f hyper_v2 -y <train.yaml> -V`` builds the first layer of each
kind of the model the train.yaml trains, on fake tensors, and counts its
parameters part by part
(:func:`~hyper_parallel.auto_parallel._layer_census.census_parameters`).
Beside each part it sets what ND's formulas price of it, and the same for
the embedding and the output layer.  It reports each part's two counts and
their difference, rather than a pass or a fail: a fact the parser misread
shows as the part it feeds.

It then counts each kind's forward FLOPs on one sequence of the run's
length (:func:`~hyper_parallel.auto_parallel._layer_census.census_flops`)
beside what the time model's op table prices of the same forward: a third
of each matmul entry, which prices the backward as twice the forward.  The
table's feed-forward entry covers the whole feed-forward, its experts,
shared expert and router included.

It sets what each kind keeps for its backward for each op, per token of
the whole layer, as the op records price it (shared decision S1,
:class:`~hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.layer_block.EvalRecords`)
beside what the census measures each op saving
(:func:`~hyper_parallel.auto_parallel._layer_census.census_saved_ops`).

Last, it sets the model spec the resolver reads from the checkpoint's
config beside the one the census measures on the layers Transformers
builds of it (:func:`~hyper_parallel.auto_parallel._spec_census.census_model_spec`),
field by field as ND prices each (:func:`verify_spec`), and prices the run
with each (:func:`verify_estimate`).

It also sets the bytes each kind's forward moves
(:func:`~hyper_parallel.auto_parallel._layer_census.census_traffic`) beside
the FLOPs the op table prices of the same forward.  The time model counts
the FLOPs and nothing else, and the cluster's kernel tables put those at
6.3 to 7.1% of a profiled step of Qwen3.5-35B-A3B at 8192 tokens, so this
section reports the other quantity a step could be priced by rather than a
difference: the two are in different units, and what a round compares is
each against the time it took.
"""
from __future__ import annotations

import copy
import os
import tempfile
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, NamedTuple, Tuple

import yaml

from hyper_parallel.auto_parallel._hf_model_spec import (
    checkpoint_configs,
    is_auto_models_schema,
    resolve_hf_model_spec,
)
from hyper_parallel.auto_parallel._layer_census import (
    census_final_norm,
    census_flops,
    census_parameters,
    census_saved_ops,
    census_traffic,
)
from hyper_parallel.auto_parallel._model_spec import ModelSpec
from hyper_parallel.auto_parallel._op_profiles import family_profile, resolve_ops
from hyper_parallel.auto_parallel._op_records import load_op_records
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.estimate_v2 import EvaluatorV2
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.layer_block import EvalRecords
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.head import EvalHead
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.tail import EvalTail
from hyper_parallel.auto_parallel.sapp_nd.nd.common.arch_hooks import check_and_apply_custom_hook, layer_groups
from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import CostModelConfig
from hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware import device_map
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.comm_time import prepare_context
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.estimate import (
    BACKWARD_RATIO,
    _flavour_tables,
    estimate_performance,
)
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.getters import get_layer_group_configs

# The time model's matmul entries, and the part of a layer each prices.
_FLOP_PARTS = (("n_attMM", "attention"), ("n_attBMM", "scores"), ("n_linrec", "linrec"), ("n_ffMM", "ffn"))

# The census's parts of a feed-forward, which the time model prices as one.
_FFN_PARTS = ("ffn", "routed", "shared", "router")

# The model facts a spec states as true or false.
_FLAGS = ("attn_output_gate", "tie_word_embeddings", "qk_norm", "qkv_bias", "o_bias", "mlp_bias", "norm_bias",
          "shared_expert_gate")

# What a spec states that verify mode does not compare: its name, what the
# AutoModels parser drops (MTP layers, which Transformers does not build),
# the census records, and the vision tower, which the census does not walk.
_NOT_COMPARED = ("name", "mtp_depth", "layer_types", "activations", "output_activations", "vision", "ops", "layers")


class VerifyRow(NamedTuple):
    """One part's parameters: ND's count, the census's, and how many layers hold it."""

    where: str
    part: str
    nd: float
    census: float
    count: int = 1


class SpecRow(NamedTuple):
    """One field of the model spec, as ND prices it: the resolver's value and the census's."""

    field: str
    resolved: Any
    census: Any


class TrafficRow(NamedTuple):
    """One part of a forward: the bytes it moves, the parameters among them, its ops, and the FLOPs ND prices of it."""

    where: str
    part: str
    moved: float
    parameter: float
    launches: int
    flops: float
    count: int = 1


def nd_parameters(lccfg: Any, ctx: Any) -> Dict[str, float]:
    """The parameters ND prices of a layer of *lccfg*, by part, with the formulas *ctx* names."""
    parts = {"attention": ctx.attn_num_p(lccfg, ctx), "norm": ctx.norm_num_p(lccfg, ctx)}
    if lccfg.n_exp == 1:
        parts["ffn"] = ctx.ffn_num_p(lccfg, ctx)
    else:
        parts["routed"] = ctx.ffn_routed_num_p(lccfg, ctx)
        parts["shared"] = ctx.ffn_shared_num_p(lccfg, ctx)
        parts["router"] = ctx.ffn_router_num_p(lccfg, ctx)
    return parts


def nd_flops(ccfg: Any, lccfg: Any) -> Dict[str, float]:
    """The forward FLOPs the time model prices one sequence of a layer of *lccfg* at, by part.

    A third of each matmul entry of the layer's op table, for one sequence
    of the whole layer: the table prices the backward as twice the
    forward, a micro-batch of ``b`` sequences, and one TP and CP rank's
    share.
    """
    base, experts = _flavour_tables(ccfg, lccfg)
    table = experts if lccfg.n_exp > 1 else base
    scale = ccfg.b * (1 + BACKWARD_RATIO) * ccfg.bytes_p / ccfg.t / ccfg.cp
    return {part: getattr(lccfg, op) * table[op] / scale for op, part in _FLOP_PARTS if op in table}


def nd_saved_ops(lccfg: Any) -> Dict[str, float]:
    """What the op records price a layer of *lccfg* keeping for each op, per token of the whole layer.

    The records' slots evaluated on a copy of the layer's config at TP, CP
    and sequence parallelism 1 and micro-batch 1, divided by its tokens.
    """
    whole = copy.copy(lccfg)
    whole.t = whole.cp = whole.sp = whole.b = 1
    ctx = SimpleNamespace(current_node=LayerType.NOT_REC_LAYER, micro_factor=1, dropless_tok_factor=1)
    return {op: sum(EvalRecords.op_bytes(whole, ctx, op).values()) / whole.s for op in load_op_records().ops}


def _text_model(ccfg: Any) -> Any:
    """The config of *ccfg*'s language model: the config itself, or a multimodal config's main submodule's."""
    if not getattr(ccfg, "multimodal", False):
        return ccfg
    return ccfg.mm_ccfgs[getattr(ccfg, "mm_main", None) or ccfg.mm_order[-1]]


def _priced(yaml_path: str) -> Tuple[Any, Any, Any]:
    """The Transformers configs a train.yaml trains, its model's and its language model's, and ND's config of it.

    ND's config is its language model's, with the family's op counts and
    fields applied as the estimates apply them, on a copy.

    Raises:
        ValueError: The train.yaml names no Transformers checkpoint.
    """
    config, text = checkpoint_configs(_train_yaml(yaml_path)["model"])
    ccfg = copy.deepcopy(_text_model(CostModelConfig(yaml_path, framework="hyper_v2")))
    check_and_apply_custom_hook(ccfg)
    return config, text, ccfg


def _kinds(ccfg: Any) -> Dict[str, List[Any]]:
    """Each layer kind of *ccfg*'s stack: its first layer, how many layers it has, and their config."""
    kinds: Dict[str, List[Any]] = {}
    first = 0
    for (kind, count), (lccfg, _) in zip(layer_groups(ccfg), get_layer_group_configs(ccfg)):
        name = kind.name if kind is not None else "decoder"
        kinds.setdefault(name, [first, 0, lccfg])[1] += count
        first += count
    return kinds


def verify_parameters(yaml_path: str) -> List[VerifyRow]:
    """The parameters ND prices of the model a train.yaml trains, part by part, beside the census's.

    Args:
        yaml_path: An AutoModels train.yaml, whose model names a Transformers
            checkpoint.

    Returns:
        A row per part of the first layer of each kind of its language
        model, then the embedding's, the output layer's, and the model's
        whole.  A table the embedding and the output layer share on one
        stage is counted at the output, as ND prices it.

    Raises:
        ValueError: The train.yaml names no Transformers checkpoint.
    """
    config, text, ccfg = _priced(yaml_path)
    ctx = prepare_context()
    rows = []
    for name, (index, count, lccfg) in _kinds(ccfg).items():
        nd, census = nd_parameters(lccfg, ctx), census_parameters(text, index)
        rows += [VerifyRow(f"{name} x{count}", part, nd.get(part, 0.0), census.get(part, 0), count)
                 for part in sorted(set(nd) | set(census))]
    table = text.vocab_size * text.hidden_size
    shared = bool(getattr(config, "tie_word_embeddings", False)) and ccfg.p <= 1
    rows.append(VerifyRow("embedding", "table", 0.0 if EvalHead.shares_output_table(ccfg)
                          else EvalHead.num_params_embed(ccfg, ctx), 0 if shared else table))
    rows.append(VerifyRow("output", "table, norm", EvalTail.num_params_output(ccfg, ctx),
                          table + census_final_norm(text)))
    rows.append(VerifyRow("model", "total", sum(row.count * row.nd for row in rows),
                          sum(row.count * row.census for row in rows)))
    return rows


def verify_activations(yaml_path: str) -> List[VerifyRow]:
    """What each kind of the model a train.yaml trains keeps for each op, the records' beside the census's.

    Args:
        yaml_path: An AutoModels train.yaml, whose model names a Transformers
            checkpoint.

    Returns:
        A row per op either prices or measures of the first layer of each
        kind of its language model, in bytes per token of the whole layer
        at the run's length (:func:`nd_saved_ops`,
        :func:`~hyper_parallel.auto_parallel._layer_census.census_saved_ops`),
        the census's ``other`` its own code's, then the kind's whole.

    Raises:
        ValueError: The train.yaml names no Transformers checkpoint.
    """
    _, text, ccfg = _priced(yaml_path)
    rows = []
    for name, (index, count, lccfg) in _kinds(ccfg).items():
        nd, census = nd_saved_ops(lccfg), census_saved_ops(text, index, int(ccfg.s))
        rows += [VerifyRow(f"{name} x{count}", op, nd.get(op, 0.0), census.get(op, 0.0), count)
                 for op in [*nd, "other"] if nd.get(op) or census.get(op)]
        rows.append(VerifyRow(f"{name} x{count}", "total", sum(nd.values()), sum(census.values()), count))
    return rows


def verify_flops(yaml_path: str) -> List[VerifyRow]:
    """The forward FLOPs ND prices of the model a train.yaml trains, part by part, beside the census's.

    Args:
        yaml_path: An AutoModels train.yaml, whose model names a Transformers
            checkpoint.

    Returns:
        A row per part of the first layer of each kind of its language
        model on one sequence of the run's length, the census's feed-forward
        parts summed as the time model prices them, then the layers' whole.

    Raises:
        ValueError: The train.yaml names no Transformers checkpoint.
    """
    _, text, ccfg = _priced(yaml_path)
    rows = []
    for name, (index, count, lccfg) in _kinds(ccfg).items():
        nd, census = nd_flops(ccfg, lccfg), census_flops(text, index, int(ccfg.s))
        census["ffn"] = sum(census.pop(part, 0) for part in _FFN_PARTS)
        rows += [VerifyRow(f"{name} x{count}", part, nd.get(part, 0.0), census.get(part, 0), count)
                 for _, part in _FLOP_PARTS if nd.get(part) or census.get(part)]
    rows.append(VerifyRow("layers", "total", sum(row.count * row.nd for row in rows),
                          sum(row.count * row.census for row in rows)))
    return rows


def spec_as_priced(data: Mapping[str, Any]) -> Dict[str, Any]:
    """The fields of the model spec *data*, each as ND prices it.

    A field left unstated is compared as the value ND prices for it: a head
    as wide as derive prices it, an MLA family's at its value heads' width;
    a flag as false; the routed and shared experts' widths as the spec
    resolves them; each kind's op counts as its profile states them, one
    field per op.  The layer stack is its body, the groups in model order:
    the AutoModels parser prices no MTP layer.

    Args:
        data: A model spec, as a resolver states it.

    Returns:
        ``{field: value}``, ``ops.<kind>.<op>`` for each op count.
    """
    spec = ModelSpec.from_dict(data)
    out = {key: value for key, value in spec.to_dict().items() if key not in _NOT_COMPARED}
    value_width = family_profile(spec.arch).model.get("v_head_dim")
    out["head_dim"] = (spec.v_head_dim or value_width) if value_width is not None else spec.effective_head_dim
    out.update({flag: bool(getattr(spec, flag)) for flag in _FLAGS})
    if spec.is_moe:
        out["moe_intermediate_size"] = spec.routed_expert_width
    if spec.num_shared_experts:
        out["shared_expert_intermediate_size"] = spec.shared_expert_width
    for kind, counts in resolve_ops(spec.arch, spec.ops).items():
        out.update({f"ops.{kind}.{op}": count for op, count in counts.to_dict().items()})
    out["layers"] = tuple((group.kind, group.count) for group in spec.layers or () if not group.mtp)
    return out


def _train_yaml(yaml_path: str) -> Dict[str, Any]:
    """The AutoModels train.yaml at *yaml_path*, whose model names a Transformers checkpoint.

    Raises:
        ValueError: The yaml is not one.
    """
    with open(yaml_path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict) or not is_auto_models_schema(raw) or not isinstance(raw.get("model"), dict):
        raise ValueError(f"{yaml_path}: verify mode needs an AutoModels train.yaml naming a Transformers checkpoint")
    return raw


def verify_spec(yaml_path: str) -> List[SpecRow]:
    """The model spec the resolver reads of the model a train.yaml trains, beside the census's, field by field.

    Args:
        yaml_path: An AutoModels train.yaml, whose model names a Transformers
            checkpoint.

    Returns:
        A row per field either spec states, each as ND prices it
        (:func:`spec_as_priced`), in the order of the fields' names.

    Raises:
        ValueError: The train.yaml names no Transformers checkpoint.
    """
    model = _train_yaml(yaml_path)["model"]
    resolved = spec_as_priced(resolve_hf_model_spec(model))
    census = spec_as_priced(resolve_hf_model_spec(model, census_spec=True))
    return [SpecRow(field, resolved.get(field), census.get(field)) for field in sorted(set(resolved) | set(census))]


def census_spec_yaml(yaml_path: str, folder: str) -> str:
    """Write the census's spec of the model a train.yaml trains to *folder*, as the run would price it.

    The spec carries the census records of the layers' activations where
    the train.yaml states ``context.census``, measured at its dataset's
    length, else at 4096 tokens; a train.yaml's ``model.config_overrides``
    takes it as it is.

    Returns:
        The path written, ``model_spec.yaml`` in *folder*.
    """
    raw = _train_yaml(yaml_path)
    seq_len = 0
    if (raw.get("context") or {}).get("census"):
        seq_len = int(((raw.get("dataset") or {}).get("data_config") or {}).get("max_seq_len") or 4096)
    spec = resolve_hf_model_spec(raw["model"], census_seq_len=seq_len, census_spec=True)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "model_spec.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(spec, handle, sort_keys=False)
    return path


def _estimate(yaml_path: str, device_type: str) -> Tuple[float, float]:
    """The peak memory, in MB, and the step's time score ND prices the run of a train.yaml at."""
    ccfg = CostModelConfig(yaml_path, framework="hyper_v2")
    peak = EvaluatorV2(yaml_path, framework="hyper_v2", log_level=0).estimate_peak()
    return float(peak), float(estimate_performance(_text_model(ccfg), device_type=device_map[device_type]))


def verify_estimate(yaml_path: str, device_type: str = "A2") -> List[VerifyRow]:
    """The run a train.yaml states, priced with the resolver's model spec and with the census's.

    Args:
        yaml_path: An AutoModels train.yaml, whose model names a Transformers
            checkpoint, and whose run is priced as it states it.
        device_type: The device to price the step's time on.

    Returns:
        The peak memory and the step's time score, with the resolver's spec
        in the ND column and the census's in the census column.

    Raises:
        ValueError: The train.yaml names no Transformers checkpoint.
    """
    raw = _train_yaml(yaml_path)
    resolved = _estimate(yaml_path, device_type)
    raw.setdefault("context", {})["census_spec"] = True
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, os.path.basename(yaml_path))
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(raw, handle)
        census = _estimate(path, device_type)
    return [VerifyRow("estimate", "peak MB", resolved[0], census[0]),
            VerifyRow("estimate", "time score", resolved[1], census[1])]


def _spec_value(value: Any) -> str:
    """One field's value as a spec report shows it: a layer stack by its kinds' layers and groups."""
    if isinstance(value, tuple):
        layers: Dict[str, int] = {}
        for kind, count in value:
            layers[kind] = layers.get(kind, 0) + count
        return f"{len(value)} groups: " + ", ".join(f"{kind} x{count}" for kind, count in layers.items())
    return "unstated" if value is None else str(value)


def report_spec(rows: List[SpecRow]) -> List[str]:
    """The lines of a spec report: each field's two values, the fields that differ marked."""
    shown = [(row.field, _spec_value(row.resolved), _spec_value(row.census), row.resolved != row.census)
             for row in rows]
    widths = [max([len(title)] + [len(line[column]) for line in shown])
              for column, title in enumerate(("field", "resolver", "census"))]
    lines = [f"{'field':{widths[0]}s}  {'resolver':>{widths[1]}s}  {'census':>{widths[2]}s}"]
    for field, resolved, census, differs in shown:
        mark = "  <- differs" if differs else ""
        lines.append(f"{field:{widths[0]}s}  {resolved:>{widths[1]}s}  {census:>{widths[2]}s}{mark}")
    differ = sum(line[3] for line in shown)
    lines.append(f"{len(rows) - differ} fields agree, {differ} differ")
    return lines


def report(rows: List[VerifyRow]) -> List[str]:
    """The lines of a verify report: each part's two counts, their difference and its share of the census's."""
    lines = [f"{'where':26s} {'part':12s} {'ND':>17s} {'census':>17s} {'ND - census':>15s} {'%':>9s}"]
    for row in rows:
        difference = row.nd - row.census
        share = f"{100 * difference / row.census:+8.3f}%" if row.census else ""
        lines.append(f"{row.where:26s} {row.part:12s} {row.nd:17,.0f} {row.census:17,.0f} "
                     f"{difference:+15,.0f} {share}")
    return lines


def verify_traffic(yaml_path: str) -> List[TrafficRow]:
    """The bytes each kind of the model a train.yaml trains moves in a forward, beside the FLOPs ND prices.

    Args:
        yaml_path: An AutoModels train.yaml, whose model names a Transformers
            checkpoint.

    Returns:
        A row per part of the first layer of each kind of its language model
        on one sequence of the run's length, the census's feed-forward parts
        summed as the time model prices them, each kind's whole, then the
        layers' whole.  A part's parameters are the share of its bytes that
        a strategy shards: on a layer that routes, the feed-forward row's
        are its experts'.

    Raises:
        ValueError: The train.yaml names no Transformers checkpoint.
    """
    _, text, ccfg = _priced(yaml_path)
    rows = []
    for name, (index, count, lccfg) in _kinds(ccfg).items():
        flops, traffic = nd_flops(ccfg, lccfg), census_traffic(text, index, int(ccfg.s))
        moved, parameter, launches = (dict(field) for field in traffic)
        for part in _FFN_PARTS[1:]:
            for field in (moved, parameter, launches):
                field["ffn"] = field.get("ffn", 0) + field.pop(part, 0)
        where = f"{name} x{count}"
        rows += [TrafficRow(where, part, moved.get(part, 0.0), parameter.get(part, 0.0),
                            launches.get(part, 0), flops.get(part, 0.0), count)
                 for part in sorted(set(moved) | set(flops)) if moved.get(part) or flops.get(part)]
        rows.append(TrafficRow(where, "total", sum(moved.values()), sum(parameter.values()),
                               sum(launches.values()), sum(flops.values()), count))
    rows.append(TrafficRow("layers", "total", *(sum(row.count * getattr(row, field) for row in rows
                                                    if row.part == "total")
                                                for field in ("moved", "parameter", "launches", "flops"))))
    return rows


def traffic_report(rows: List[TrafficRow]) -> List[str]:
    """The lines of a traffic report: each part's bytes, the parameters among them, its ops and ND's FLOPs.

    The last column is the bytes a part moves for each FLOP ND prices of it,
    and the spread down it is the reading: a part that moves much and
    multiplies little is priced near nothing.  The FLOPs are the op table's
    matmul entries, as the FLOPs section reports them, so a part with no
    matmul of its own shows none.
    """
    lines = [f"{'where':26s} {'part':12s} {'bytes moved':>17s} {'parameters':>15s} "
             f"{'ops':>8s} {'ND FLOPs':>17s} {'bytes/FLOP':>11s}"]
    for row in rows:
        ratio = f"{row.moved / row.flops:11.4f}" if row.flops else f"{'':>11s}"
        lines.append(f"{row.where:26s} {row.part:12s} {row.moved:17,.0f} {row.parameter:15,.0f} "
                     f"{row.launches:8,d} {row.flops:17,.0f} {ratio}")
    return lines

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
"""Describe the DeepSeek-V4.1 validation crop to the ND cost model, and check the description.

Transformers does not know the ``deepseek_v41`` model type, so ND cannot resolve
the released checkpoint: neither its model-spec reader nor verify mode nor the
layer census can load it.  Every fact ND prices therefore has to be stated by
hand, and nothing in ND checks it.  This script supplies both halves.

``spec``
    Derive ND's ``model.config_overrides`` from the released ``config.json`` and
    the crop's own arguments, and print it as a YAML fragment.  This is the
    model description: the released dimensions divided by the crop's parameter
    divisor, with the per-head widths and the layer count it preserves.

``inventory``
    Build the crop on the CPU, with no device and no checkpoint, and count its
    parameters under the part names ND uses, by layer kind.  This is the ground
    truth a description is checked against.

``verify``
    Both, side by side: ND's parameters per part of each layer kind against the
    crop's own, and the blocks ND has no field for listed as unpriced.  It is
    what ``run_nd -V`` does for a model Transformers can load.

The crop builds on the CPU rather than on the meta device because Engram's hash
mapping reads its token map with ``.item()``, which a meta tensor refuses.
"""
import argparse
import collections
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml

# Released fields the crop divides by its text parameter divisor, and those it
# keeps: per-head widths, the vocabulary and the decoder depth.
_SCALED = ("hidden_size", "moe_intermediate_size", "num_attention_heads",
           "q_lora_rank", "o_lora_rank", "index_n_heads", "engram_head_dim")
_KEPT = ("head_dim", "qk_rope_head_dim", "index_head_dim", "o_groups",
         "num_hidden_layers", "vocab_size", "num_key_value_heads",
         "num_experts_per_tok", "n_shared_experts")

# Where each of the crop's parameters belongs, by a substring of its qualified
# name, in the part names ``nd.verify.nd_parameters`` returns.  ``self_attn``
# covers the compressor and the Indexer too, which sit inside it, so that
# ND's one attention figure is compared with the whole attention block.
_PARTS: Tuple[Tuple[str, str], ...] = (
    ("engram", "engram"),
    ("_hc", "mhc"),
    ("mlp.experts", "routed"),
    ("shared_expert", "shared"),
    ("mlp.gate", "router"),
    ("self_attn", "attention"),
    ("layernorm", "norm"),
)

# The parts ND prices for a layer, so an unpriced block is named as such
# rather than silently compared with zero.
_ND_PARTS = ("attention", "norm", "routed", "shared", "router")


def _released(model_dir: str) -> Dict[str, Any]:
    """Return the released text configuration of the checkpoint at *model_dir*.

    Args:
        model_dir: A directory holding the released ``config.json``.

    Returns:
        Its ``text_config`` section.

    Raises:
        ValueError: The config states no ``text_config``.
    """
    with open(Path(model_dir) / "config.json", encoding="utf-8") as handle:
        config = json.load(handle)
    text = config.get("text_config")
    if not isinstance(text, dict):
        raise ValueError(f"{model_dir}/config.json states no text_config")
    return text


def cropped_dimensions(model_dir: str, divisor: int, experts: int) -> Dict[str, int]:
    """Return the crop's own dimensions, as ``build_cropped_deepseek_v41`` derives them.

    Args:
        model_dir: The released checkpoint directory.
        divisor: The crop's ``text_parameter_divisor``.
        experts: The crop's ``num_routed_experts``.

    Returns:
        Every released field the crop scales or keeps, under its released name,
        with ``num_experts`` holding the crop's routed expert count.

    Raises:
        ValueError: A field the crop scales is not divisible by the divisor.
    """
    text = _released(model_dir)
    dims: Dict[str, int] = {}
    for name in _SCALED:
        if name not in text:
            continue
        value = int(text[name])
        if value % divisor:
            raise ValueError(f"{name}={value} is not divisible by the parameter divisor {divisor}")
        dims[name] = value // divisor
    for name in _KEPT:
        if name in text:
            dims[name] = int(text[name])
    dims["num_experts"] = int(experts)
    return dims


def nd_spec(model_dir: str, divisor: int, experts: int) -> Dict[str, Any]:
    """Return the ``model.config_overrides`` ND needs for the crop.

    Only fields ND reads are emitted, and only where the crop actually builds
    what they describe: the released ``num_nextn_predict_layers`` and every
    DSpark field are left out because the validation crop builds neither, and
    stating them would price layers that never run.  No dtype is emitted: the
    width a run loads, keeps and computes in is the run's, stated as the
    launcher states it (``model.torch_dtype``, ``model_init_dtype`` and FSDP's
    ``mix_precision``), and ND refuses ``compute_dtype`` among the model's
    facts.

    Args:
        model_dir: The released checkpoint directory.
        divisor: The crop's ``text_parameter_divisor``.
        experts: The crop's ``num_routed_experts``.

    Returns:
        The overrides, in the order they are useful to read.
    """
    dims = cropped_dimensions(model_dir, divisor, experts)
    shared_width = dims["moe_intermediate_size"]
    spec: Dict[str, Any] = {
        "hidden_size": dims["hidden_size"],
        "num_hidden_layers": dims["num_hidden_layers"],
        "num_attention_heads": dims["num_attention_heads"],
        "num_key_value_heads": dims.get("num_key_value_heads", 1),
        "head_dim": dims["head_dim"],
        "qk_rope_head_dim": dims.get("qk_rope_head_dim", 0),
        "q_lora_rank": dims["q_lora_rank"],
        "vocab_size": dims["vocab_size"],
        # An all-MoE config declares no dense feed-forward width, and ND
        # divides by it on the FLOPs path: the shared expert's own width is
        # what ND prices the shared expert at.
        "intermediate_size": shared_width,
        "moe_intermediate_size": shared_width,
        "num_experts": dims["num_experts"],
        "num_experts_per_tok": dims.get("num_experts_per_tok", 1),
        "num_shared_experts": dims.get("n_shared_experts", 0),
    }
    return spec


def unpriced(model_dir: str) -> List[Tuple[str, str]]:
    """Return the released blocks ND has no field for, each with what it is.

    Args:
        model_dir: The released checkpoint directory.

    Returns:
        ``(block, what it is)`` for every block of the crop that no ND field
        carries, in the order they cost the most.
    """
    text = _released(model_dir)
    blocks = [
        ("o_lora_rank, o_groups",
         f"the output projection through rank {text.get('o_lora_rank')} in {text.get('o_groups')} groups"),
        ("index_*",
         f"the Lightning Indexer, {text.get('index_n_heads')} heads of {text.get('index_head_dim')}, "
         f"top {text.get('index_topk')}, on layers {text.get('index_source_layer_ids')}"),
        ("compress_ratios, sliding_window, kv_source_layer_ids",
         f"compressed and sparse attention, window {text.get('sliding_window')}, "
         f"key-value sources on layers {text.get('kv_source_layer_ids')}"),
        ("hc_mult, hc_sinkhorn_iters",
         f"mHC hyper-connections, multiplier {text.get('hc_mult')}, "
         f"{text.get('hc_sinkhorn_iters')} Sinkhorn iterations, on every layer"),
        ("engram_*", f"Engram on layers {text.get('engram_layer_ids')}"),
        ("candidate_*", f"the candidate pool from layer {text.get('candidate_source_layer_id')}"),
        ("scoring_func, swiglu_limit",
         f"{text.get('scoring_func')} routing and a SwiGLU clamped at {text.get('swiglu_limit')}"),
    ]
    return [(name, what) for name, what in blocks if "None" not in what]


def _stated_mtp(spec_yaml: str) -> List[str]:
    """Return the multi-token-prediction fields *spec_yaml* states, if any.

    No spelling of the field reaches the cost model through
    ``config_overrides``, so one stated there is silently not priced.

    Args:
        spec_yaml: A train.yaml whose model states ``config_overrides``.

    Returns:
        The names it states, in the order they are read.
    """
    with open(spec_yaml, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    overrides = ((raw.get("model") or {}).get("config_overrides") or {})
    return [name for name in ("mtp_depth", "num_nextn_predict_layers", "mtp_num_hidden_layers")
            if overrides.get(name)]


def _part_of(name: str) -> str:
    """Return the part a parameter named *name* belongs to."""
    for key, part in _PARTS:
        if key in name:
            return part
    return "other"


def build_crop(model_dir: str, assets: str, divisor: int, experts: int) -> Any:
    """Build the validation crop on the CPU, without a device or a checkpoint.

    Args:
        model_dir: The released checkpoint directory.
        assets: The prepared Engram assets file for this divisor.
        divisor: The crop's ``text_parameter_divisor``.
        experts: The crop's ``num_routed_experts``.

    Returns:
        The model the trainer would build, on the CPU.
    """
    # Imported here, not at the top: describing a crop reads config.json alone, and must
    # work in a checkout or an environment where neither torch nor the model is installed.
    # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=forbidden-backend-import
    from hyper_parallel.models.deepseek_v41.adapter.validation.cropped_model import (
        build_cropped_deepseek_v41,
    )
    return build_cropped_deepseek_v41(
        config_path=model_dir, engram_assets_path=assets,
        text_parameter_divisor=divisor, num_routed_experts=experts,
        torch_dtype=torch.bfloat16, validate_placement=False,
    )


def inventory(model: Any) -> Dict[str, Any]:
    """Return the crop's parameters by part, by layer index, and in total.

    Args:
        model: The crop, as :func:`build_crop` builds it.

    Returns:
        ``parts`` summed over the body, ``layers`` mapping a layer index to its
        parts, ``outside`` for what sits outside the body, and ``total``.
    """
    parts: Dict[str, int] = collections.Counter()
    layers: Dict[int, Dict[str, int]] = collections.defaultdict(collections.Counter)
    outside: Dict[str, int] = collections.Counter()
    total = 0
    for name, param in model.named_parameters():
        count = param.numel()
        total += count
        found = re.search(r"layers\.(\d+)\.", name)
        if found:
            part = _part_of(name)
            parts[part] += count
            layers[int(found.group(1))][part] += count
        elif "embed" in name:
            outside["embedding"] += count
        else:
            outside["output"] += count
    return {"parts": dict(parts), "layers": {index: dict(held) for index, held in layers.items()},
            "outside": dict(outside), "total": total}


def layer_kinds(layers: Mapping[int, Mapping[str, int]]) -> List[Tuple[Tuple[int, ...], Dict[str, int]]]:
    """Group body layers that hold the same parameters, in model order.

    Args:
        layers: Each layer index with its parameters by part.

    Returns:
        ``(layer indices, parts)`` per kind, ordered by the first layer of each.
    """
    kinds: Dict[Tuple[Tuple[str, int], ...], List[int]] = collections.defaultdict(list)
    for index in sorted(layers):
        kinds[tuple(sorted(layers[index].items()))].append(index)
    return sorted(((tuple(indices), dict(signature)) for signature, indices in kinds.items()),
                  key=lambda pair: pair[0][0])


def nd_layer_parameters(spec_yaml: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Return what ND makes of the spec at *spec_yaml*: one body layer's parts, and the stack around it.

    Args:
        spec_yaml: A train.yaml whose model states ``config_overrides``.

    Returns:
        ND's parameters per part of one body layer, and the stack: ``layers``,
        ``mtp_layers``, ``embedding`` and ``output``.  The stack is what catches
        a description that states layers the run does not build.
    """
    # verify first: importing the cost model's own modules in another order
    # leaves _cost_model_variables partly initialized (a circular import).  Imported here
    # for the same reason as above, and in this order.
    # pylint: disable=import-outside-toplevel
    from hyper_parallel.auto_parallel.sapp_nd.nd.verify import nd_parameters
    from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.head import EvalHead
    from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.tail import EvalTail
    from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.comm_time import prepare_context
    from hyper_parallel.auto_parallel.sapp_nd.nd.common.arch_hooks import check_and_apply_custom_hook
    from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import CostModelConfig

    ccfg = copy.deepcopy(CostModelConfig(spec_yaml, framework="hyper_v2"))
    check_and_apply_custom_hook(ccfg)
    ctx = prepare_context()
    stack = {
        "layers": float(getattr(ccfg, "n_lay", 0) or 0),
        "mtp_layers": float(getattr(ccfg, "n_mtp", 0) or 0),
        "embedding": 0.0 if EvalHead.shares_output_table(ccfg) else float(EvalHead.num_params_embed(ccfg, ctx)),
        "output": float(EvalTail.num_params_output(ccfg, ctx)),
    }
    return nd_parameters(ccfg, ctx), stack


def _print_spec(args: argparse.Namespace) -> None:
    """Print the ND model description of the crop."""
    spec = nd_spec(args.model_dir, args.divisor, args.experts)
    print(yaml.safe_dump({"model": {"name": "deepseek_v41_crop", "config_overrides": spec}},
                         sort_keys=False, default_flow_style=False).rstrip())
    print("\n# Blocks no ND field carries, so not priced at all:")
    for name, what in unpriced(args.model_dir):
        print(f"#   {name}: {what}")
    print("# Also left out on purpose: num_nextn_predict_layers and every dspark_* field,")
    print("#   because the validation crop builds neither an MTP nor a DSpark layer.")


def layer_tensors(model: Any, layer: int) -> List[Tuple[str, Tuple[int, ...], int, str]]:
    """Return one layer's parameter tensors: name, shape, count and the part each belongs to.

    Args:
        model: The crop, as :func:`build_crop` builds it.
        layer: The body layer to list, by its index in the released stack.

    Returns:
        ``(name within the layer, shape, parameters, part)`` in model order.
    """
    prefix = f"model.layers.{layer}."
    return [(name[len(prefix):], tuple(param.shape), param.numel(), _part_of(name))
            for name, param in model.named_parameters() if name.startswith(prefix)]


def _print_inventory(args: argparse.Namespace) -> None:
    """Print the crop's measured parameter inventory."""
    model = build_crop(args.model_dir, args.assets, args.divisor, args.experts)
    if args.tensors is not None:
        rows = layer_tensors(model, args.tensors)
        if not rows:
            raise SystemExit(f"the crop has no body layer {args.tensors}")
        print(f"layer {args.tensors}, tensor by tensor:\n")
        print(f"  {'tensor':44s} {'shape':24s} {'parameters':>12s}  part")
        for name, shape, count, part in rows:
            flag = part if part in _ND_PARTS else f"{part} (no ND field)"
            print(f"  {name:44s} {str(shape):24s} {count:>12,}  {flag}")
        print(f"\n  {'whole layer':44s} {'':24s} {sum(row[2] for row in rows):>12,}")
        return
    held = inventory(model)
    print(f"total parameters: {held['total']:,}\n")
    print("body, by part:")
    for part, count in sorted(held["parts"].items(), key=lambda pair: -pair[1]):
        flag = "" if part in _ND_PARTS else "   <- no ND field"
        print(f"  {part:12s} {count:>14,}  {100 * count / held['total']:5.1f}%{flag}")
    print("\noutside the body:")
    for part, count in sorted(held["outside"].items()):
        print(f"  {part:12s} {count:>14,}")
    kinds = layer_kinds(held["layers"])
    print(f"\n{len(held['layers'])} body layers, {len(kinds)} kinds:")
    for indices, parts in kinds:
        whole = sum(parts.values())
        extra = sorted(part for part in parts if part not in ("attention", "norm", "routed", "shared", "router"))
        shown = f"{indices[0]}..{indices[-1]}" if len(indices) > 3 else ", ".join(str(i) for i in indices)
        print(f"  {len(indices):2d} layer(s) [{shown}] {whole:>12,} params"
              f"{'  + ' + ', '.join(extra) if extra else ''}")


def _print_verify(args: argparse.Namespace) -> None:
    """Print ND's parameters per part beside the crop's own."""
    if not args.spec_yaml:
        raise SystemExit("verify needs --spec-yaml, a train.yaml whose model states config_overrides")
    held = inventory(build_crop(args.model_dir, args.assets, args.divisor, args.experts))
    nd, stack = nd_layer_parameters(args.spec_yaml)
    kinds = layer_kinds(held["layers"])
    plain = max(kinds, key=lambda pair: len(pair[0]))[1]
    print(f"ND's spec: {args.spec_yaml}")
    print(f"one plain body layer, {len(max(kinds, key=lambda pair: len(pair[0]))[0])} of "
          f"{len(held['layers'])} layers\n")
    print(f"  {'part':12s} {'ND':>14s} {'the crop':>14s} {'difference':>12s}")
    for part in sorted(set(nd) | set(plain)):
        mine, real = float(nd.get(part, 0.0)), float(plain.get(part, 0))
        gap = f"{100 * (mine / real - 1):+11.1f}%" if real else "   unpriced"
        print(f"  {part:12s} {mine:>14,.0f} {real:>14,.0f} {gap:>12s}")
    priced = sum(float(plain.get(part, 0)) for part in _ND_PARTS)
    missing = sum(count for part, count in plain.items() if part not in _ND_PARTS)
    print(f"\n  priced parts of a plain layer: {priced:,.0f}")
    print(f"  ND's total for the same layer:  {sum(nd.values()):,.0f} "
          f"({100 * (sum(nd.values()) / priced - 1):+.1f}%)")
    print(f"  parts with no ND field:        {missing:,.0f} "
          f"({100 * missing / (priced + missing):.1f}% of the layer)")

    # The stack, where a description that states layers the run never builds
    # shows itself: a per-layer table cannot see an extra layer count.
    body = len(held["layers"])
    print(f"\n  {'the stack':26s} {'ND':>14s} {'the crop':>14s}")
    print(f"  {'body layers':26s} {stack['layers']:>14,.0f} {body:>14,}"
          f"{'   <- MISMATCH' if int(stack['layers']) != body else ''}")
    print(f"  {'extra MTP layers':26s} {stack['mtp_layers']:>14,.0f} {0:>14,}"
          f"{'   <- MISMATCH: the crop builds none' if stack['mtp_layers'] else ''}")
    # A config_overrides spec cannot state MTP at all: no spelling of the field
    # reaches n_mtp on this path. Say so, rather than let a stated field look
    # priced because the row above happens to read 0.
    named = _stated_mtp(args.spec_yaml)
    if named and not stack["mtp_layers"]:
        print(f"      your spec states {', '.join(named)}, and ND read none of it:"
              f" MTP cannot be stated through config_overrides")
    for part in ("embedding", "output"):
        real = float(held["outside"].get(part, 0))
        gap = f"{100 * (stack[part] / real - 1):+13.1f}%" if real else ""
        print(f"  {part:26s} {stack[part]:>14,.0f} {real:>14,.0f}{gap}")

    whole = sum(nd.values()) * (stack["layers"] + stack["mtp_layers"]) + stack["embedding"] + stack["output"]
    print(f"\n  {'whole model':26s} {whole:>14,.0f} {held['total']:>14,}"
          f"  {100 * (whole / held['total'] - 1):+.1f}%")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Return the parsed command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("spec", "inventory", "verify"))
    parser.add_argument("model_dir", help="the released checkpoint directory, holding config.json")
    parser.add_argument("--assets", help="the prepared Engram assets file (inventory and verify)")
    parser.add_argument("--divisor", type=int, default=8, help="the crop's text_parameter_divisor")
    parser.add_argument("--experts", type=int, default=16, help="the crop's num_routed_experts")
    parser.add_argument("--spec-yaml", help="a train.yaml stating config_overrides, for verify")
    parser.add_argument("--tensors", type=int, metavar="LAYER",
                        help="with inventory, list this body layer's tensors with their shapes "
                             "instead of the whole model's parts")
    args = parser.parse_args(argv)
    if args.command in ("inventory", "verify") and not args.assets:
        parser.error(f"{args.command} needs --assets, the prepared Engram assets file")
    return args


def main(argv: Optional[List[str]] = None) -> None:
    """Run the requested command."""
    args = parse_args(argv)
    {"spec": _print_spec, "inventory": _print_inventory, "verify": _print_verify}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())

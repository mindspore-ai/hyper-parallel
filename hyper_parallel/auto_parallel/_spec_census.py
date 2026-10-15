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
"""The census's ModelSpec: a model's stack, dimensions and op counts, measured on the layers it builds.

The resolver (:mod:`hyper_parallel.auto_parallel._hf_model_spec`) states a
model from its Transformers config's field names, completed by rules a
family's name selects.  The census states it from the model Transformers
builds of the config instead, the block walk of IR phase 5:

- every decoder layer is built on the meta device, and its attention and
  feed-forward are told apart by what they hold: a gated delta rule makes
  a linear attention, routed experts a MoE feed-forward;
- one layer of each shape runs a forward on fake tensors, under the
  kernels the runtime runs (:mod:`~hyper_parallel.auto_parallel._layer_census`),
  and the kernels' inputs state its heads and their widths, the router's
  top-k its experts per token;
- the parameters state the widths, the biases, the norms and the gates.

A layer's op counts are counted in ND's units: an attention's projections
by role, the queries', keys' and values' the kernel takes and the output's
after it; its batched matmuls, softmaxes and score casts per softmax kernel;
a feed-forward's projections as its weights over one ``h x width``
projection's; its norms and activation functions as the modules it runs.
The kinds are the model's family's (``arch``), each layer taking the kind
whose flavours its own match, and :func:`census_model_spec` states the whole
as a ModelSpec mapping, the resolver's form, which a train.yaml's
``model.config_overrides`` takes as it is.
"""
from __future__ import annotations

import dataclasses
import functools
import importlib
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch  # pylint: disable=forbidden-backend-import
from torch.utils._python_dispatch import TorchDispatchMode  # pylint: disable=forbidden-backend-import

from hyper_parallel.auto_parallel._layer_census import _fake_layer, _positions, _run, delta_rule_module
from hyper_parallel.auto_parallel._model_spec import LayerGroup, ModelSpec, ModelSpecError, OpCounts
from hyper_parallel.auto_parallel._op_profiles import OpProfile, infer_arch, load_op_profile

# The modules an activation function's module is defined in.
_ACTIVATION_MODULES = ("transformers.activations", "torch.nn.modules.activation")

# The names of an attention's output projection, the projection after its kernel.
_OUTPUT_PROJECTIONS = ("o_proj", "out_proj", "dense")


@dataclasses.dataclass(frozen=True)
class LayerShape:
    """How one decoder layer is built.

    Attributes:
        attention: ``linear`` for an attention running a gated delta rule,
            ``full`` otherwise.
        ffn: ``moe`` for a feed-forward holding routed experts, ``dense``
            otherwise.
        params: Every parameter's path and shape, which tell two layers of
            one flavour apart.
    """

    attention: str
    ffn: str
    params: Tuple[Tuple[str, Tuple[int, ...]], ...]


@dataclasses.dataclass
class KindCensus:
    """What one forward of a layer states of its kind: its op counts and the dimensions its kernels take.

    Attributes:
        ops: The op counts, in ND's units.
        dims: The model fields the layer states, by spec field name.
    """

    ops: OpCounts
    dims: Dict[str, Any]


def _parts(layer: Any) -> Tuple[Any, Tuple[str, Any], List[Any]]:
    """A layer's attention, its feed-forward's name and module, and its own norms, as its children are named."""
    attention = ffn = None
    norms = []
    for name, child in layer.named_children():
        if "norm" in type(child).__name__.lower():
            norms.append(child)
        elif "attn" in name or "attention" in name:
            attention = child
        elif any(True for _ in child.parameters()):
            ffn = (name, child)
    if attention is None or ffn is None:
        raise ModelSpecError(f"{type(layer).__name__} holds no attention or no feed-forward the census knows")
    return attention, ffn, norms


def _experts(ffn: Any) -> Optional[Any]:
    """A feed-forward's routed experts, or ``None`` for a dense one."""
    return getattr(ffn, "experts", None)


def layer_shape(layer: Any) -> LayerShape:
    """The flavours and the parameters of *layer*."""
    attention, (_, ffn), _ = _parts(layer)
    linear = any(delta_rule_module(module) for module in attention.modules())
    return LayerShape("linear" if linear else "full", "moe" if _experts(ffn) is not None else "dense",
                      tuple((name, tuple(param.shape)) for name, param in layer.named_parameters()))


def _linears(module: Any) -> List[Tuple[str, Any]]:
    """The linear projections of *module*, by path."""
    return [(name, child) for name, child in module.named_modules() if isinstance(child, torch.nn.Linear)]


def _down_width(module: Any, hidden: int) -> int:
    """The width of a feed-forward *module*: the input of its projection back to the hidden width.

    Where every projection maps the hidden width to itself, the
    feed-forward is as wide as the model.
    """
    back = [child.in_features for _, child in _linears(module) if child.out_features == hidden]
    if not back:
        raise ModelSpecError(f"{type(module).__name__} holds no projection back to width {hidden}")
    return next((width for width in back if width != hidden), hidden)


def _is_activation(module: Any) -> bool:
    """Whether *module* is an activation function's module."""
    return type(module).__module__ in _ACTIVATION_MODULES


class _KindOps(TorchDispatchMode):
    """What a layer's forward runs: its kernels and their inputs, its routers' top-k, its dropouts and projections."""

    def __init__(self, layer: Any) -> None:
        """Follow which of *layer*'s modules runs."""
        super().__init__()
        self.running: List[str] = []
        self.kernels: List[Tuple[Tuple[int, ...], ...]] = []
        self.delta_rules = 0
        self.top_k: Optional[int] = None
        self.dropouts = 0
        self.after_kernel = 0
        self.activations: List[str] = []
        self.modules = dict(layer.named_modules())
        for name, module in self.modules.items():
            if name:
                module.register_forward_pre_hook(functools.partial(self._enter, name))
                module.register_forward_hook(self._leave)
            if hasattr(module, "chunk_gated_delta_rule"):
                module.chunk_gated_delta_rule = self._counted(module.chunk_gated_delta_rule)

    def _counted(self, rule: Any) -> Any:
        """*rule*, counting its calls."""
        def run(*args: Any, **kwargs: Any) -> Any:
            """The delta rule, its call counted."""
            self.delta_rules += 1
            return rule(*args, **kwargs)
        return run

    def _enter(self, name: str, *_: Any) -> None:
        """Note that the module at *name* runs: an output projection after the kernel, an activation function."""
        module = self.modules[name]
        if (self.kernels or self.delta_rules) and name.rsplit(".", 1)[-1] in _OUTPUT_PROJECTIONS:
            self.after_kernel += 1
        if _is_activation(module):
            self.activations.append(name)
        self.running.append(name)

    def _leave(self, *_: Any) -> None:
        """Note that the innermost module running has returned."""
        self.running.pop()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # pylint: disable=unused-argument
        out = func(*args, **(kwargs or {}))
        if func is torch.ops.nd_census.flash_attention.default:
            self.kernels.append(tuple(tuple(tensor.shape) for tensor in args[:3]))
        elif func is torch.ops.aten.topk.default:
            self.top_k = int(args[1])
        elif func is torch.ops.aten.native_dropout.default and args[1] > 0:
            self.dropouts += 1
        return out


def _biased(module: Any) -> bool:
    """Whether a linear projection of *module* holds a bias."""
    return any(child.bias is not None for _, child in _linears(module))


def _shared_dims(ffn: Any, hidden: int, width: int) -> Dict[str, Any]:
    """A MoE feed-forward's shared experts' fields: as many experts of the routed ones' width as they are wide.

    Its gate is a projection of the hidden state to one scale.
    """
    shared = [child for name, child in ffn.named_children() if "shared" in name
              and any(inner.out_features == hidden for _, inner in _linears(child))]
    dims: Dict[str, Any] = {"shared_expert_gate": any(
        "shared" in name and "gate" in name and isinstance(child, torch.nn.Linear)
        for name, child in ffn.named_children())}
    if shared:
        shared_width = sum(_down_width(child, hidden) for child in shared)
        dims.update(shared_expert_intermediate_size=shared_width,
                    num_shared_experts=shared_width // width if shared_width % width == 0 else len(shared),
                    mlp_bias=any(_biased(child) for child in shared))
    return dims


def _ffn_dims(ffn: Any, hidden: int, top_k: Optional[int]) -> Tuple[Dict[str, Any], int]:
    """A feed-forward's fields, and its projections in ND's units: its weights over an ``h x width`` projection's."""
    experts = _experts(ffn)
    if experts is None:
        width = _down_width(ffn, hidden)
        weights = sum(child.weight.numel() for _, child in _linears(ffn))
        return {"dense_width": width, "mlp_bias": _biased(ffn)}, round(weights / (hidden * width))
    num, width = int(experts.num_experts), int(experts.intermediate_dim)
    weights = sum(param.numel() for name, param in experts.named_parameters() if "bias" not in name)
    shared = _shared_dims(ffn, hidden, width)
    dims = {"num_experts": num, "moe_intermediate_size": width, "num_experts_per_tok": top_k, **shared,
            "mlp_bias": bool(getattr(experts, "has_bias", False)) or shared.get("mlp_bias", False)}
    return dims, round(weights / (num * hidden * width))


def _attention_dims(attention: Any, counter: _KindOps) -> Dict[str, Any]:
    """An attention's fields: its heads and widths as its kernel takes them, its projections' shapes."""
    projections = dict(_linears(attention))
    inputs = [child for name, child in projections.items() if name.rsplit(".", 1)[-1] not in _OUTPUT_PROJECTIONS]
    outputs = [child for name, child in projections.items() if name.rsplit(".", 1)[-1] in _OUTPUT_PROJECTIONS]
    dims: Dict[str, Any] = {
        "qkv_bias": any(child.bias is not None for child in inputs),
        "o_bias": any(child.bias is not None for child in outputs),
        "qk_norm": all(hasattr(attention, name) for name in ("q_norm", "k_norm")),
    }
    if counter.delta_rules:
        for field, attribute in (("linear_num_key_heads", "num_k_heads"), ("linear_key_head_dim", "head_k_dim"),
                                 ("linear_num_value_heads", "num_v_heads"), ("linear_value_head_dim", "head_v_dim"),
                                 ("linear_conv_kernel_dim", "conv_kernel_size")):
            dims[field] = int(getattr(attention, attribute))
        return dims
    query, key, value = counter.kernels[0]
    heads, width, value_width = query[1], query[-1], value[-1]
    dims.update(num_attention_heads=heads, num_key_value_heads=key[1], head_dim=value_width)
    latent = projections.get("kv_b_proj")
    if latent is not None:
        # MLA: its keys and values are up-projected from a latent, beside a
        # rotary key the heads share.
        rope = projections["kv_a_proj_with_mqa"].out_features - latent.in_features
        dims.update(kv_lora_rank=latent.in_features, qk_rope_head_dim=rope, qk_nope_head_dim=width - rope,
                    v_head_dim=value_width)
        if "q_a_proj" in projections:
            dims["q_lora_rank"] = projections["q_a_proj"].out_features
    elif "q_proj" in projections:
        dims["attn_output_gate"] = projections["q_proj"].out_features == 2 * heads * width
    return dims


def census_kind(config: Any, layer_index: int, seq_length: int = 64) -> KindCensus:
    """The op counts and the fields one forward of layer *layer_index* of *config* states.

    Args:
        config: The language model's Transformers config.
        layer_index: The layer to build.
        seq_length: Tokens of the sequence it runs; the counts and the
            fields do not depend on it.

    Returns:
        The layer's :class:`KindCensus`.
    """
    with _fake_layer(config, layer_index) as (layer, rotary):
        hidden = torch.randn(1, seq_length, config.hidden_size, dtype=torch.bfloat16)
        counter = _KindOps(layer)
        with counter:
            _run(layer, hidden, _positions(rotary, hidden, seq_length))
        attention, (ffn_name, ffn), norms = _parts(layer)
        dims = _attention_dims(attention, counter)
        ffn_dims, projections = _ffn_dims(ffn, config.hidden_size, counter.top_k)
        dims.update(ffn_dims)
        dims.update(layer_norms=len(norms), norm_bias=any(norm.bias is not None for norm in norms
                                                          if hasattr(norm, "bias")))
        # The activation functions of one expert, or of the dense feed-forward.
        runs = f"{ffn_name}.experts." if _experts(ffn) is not None else f"{ffn_name}."
    softmaxes = len(counter.kernels)
    routed = [name for name in counter.activations if name.startswith(runs)]
    ops = OpCounts(
        attMM=(3 + counter.after_kernel) if softmaxes or counter.delta_rules else len(_linears(attention)),
        attBMM=2 * softmaxes, ffMM=projections, softmax=softmaxes, dropout=counter.dropouts,
        normOp=len(norms), gather=4, headCast=softmaxes, ffAct=len(routed), linrec=counter.delta_rules)
    return KindCensus(ops, dims)


def _kind_name(profile: OpProfile, shape: LayerShape) -> str:
    """The kind of *profile* a layer of *shape* is: the one whose flavours match its own."""
    for ffn in (shape.ffn, None):
        names = [kind.name for kind in profile.layer_kinds.values()
                 if kind.attention == shape.attention and kind.ffn == ffn]
        if len(names) == 1:
            return names[0]
    raise ModelSpecError(
        f"op profile {profile.arch!r} has no single kind for a layer of {shape.attention} attention "
        f"and a {shape.ffn} feed-forward")


def _language_model(config: Any) -> Any:
    """The Transformers causal LM of *config*, built on the meta device."""
    transformers = importlib.import_module("transformers")
    with torch.device("meta"):
        return transformers.AutoModelForCausalLM.from_config(config)


def _stack(profile: OpProfile, shapes: List[LayerShape]) -> Tuple[Tuple[LayerGroup, ...], Dict[str, int]]:
    """The layer stack *shapes* make, each layer of its kind, and each kind's first layer.

    Raises:
        ModelSpecError: If the profile gives two differently built layers
            one kind.
    """
    kinds = [_kind_name(profile, shape) for shape in shapes]
    firsts: Dict[str, int] = {}
    for index, (kind, shape) in enumerate(zip(kinds, shapes)):
        first = firsts.setdefault(kind, index)
        if shapes[first] != shape:
            raise ModelSpecError(
                f"op profile {profile.arch!r} prices layers {first} and {index} as one kind {kind!r}, "
                f"but they are built differently: {shapes[first].ffn} and {shape.ffn} feed-forwards")
    groups: List[List[Any]] = []
    for kind in kinds:
        if groups and groups[-1][0] == kind:
            groups[-1][1] += 1
        else:
            groups.append([kind, 1])
    return tuple(LayerGroup(kind, count) for kind, count in groups), firsts


def _model_fields(model: Any) -> Dict[str, Any]:
    """The fields of the model around its layers: its width, vocabulary, tied table and positions."""
    table = model.get_input_embeddings().weight
    body = model.get_decoder()
    fields = {"hidden_size": int(table.shape[1]), "vocab_size": int(table.shape[0]),
              "num_hidden_layers": len(body.layers),
              "tie_word_embeddings": model.get_output_embeddings().weight is table}
    positions = getattr(getattr(body, "rotary_emb", None), "original_max_seq_len", None)
    if positions:
        fields["max_position_embeddings"] = int(positions)
    return fields


def _widths(spec: Dict[str, Any], stack_dims: Mapping[str, Mapping[str, Any]]) -> None:
    """Fill *spec*'s widths from its kinds' fields: attention's, linear attention's, feed-forward's."""
    ordered = sorted(stack_dims.items(), key=lambda item: "linear_num_key_heads" in item[1])
    for _, dims in ordered:
        for field, value in dims.items():
            if field != "dense_width" and value is not None:
                spec.setdefault(field, value)
    for field in ("qkv_bias", "o_bias", "qk_norm", "mlp_bias", "norm_bias", "shared_expert_gate",
                  "attn_output_gate"):
        # A flag no layer states is one no layer has: a dense model's
        # shared expert gate, a linear attention's output gate.
        spec[field] = any(dims.get(field, False) for dims in stack_dims.values())
    dense = [dims["dense_width"] for dims in stack_dims.values() if "dense_width" in dims]
    # A model whose every layer routes its tokens states the dense width
    # as the resolver does: its shared expert's, else its experts'.
    width = dense[0] if dense else spec.get("shared_expert_intermediate_size") or spec.get("moe_intermediate_size")
    if width:
        spec["intermediate_size"] = width


def _first_dense(shapes: List[LayerShape]) -> Optional[int]:
    """How many dense layers lead a stack whose other layers route their tokens, or ``None``."""
    moe = [index for index, shape in enumerate(shapes) if shape.ffn == "moe"]
    if not moe or not moe[0] or any(shape.ffn != "dense" for shape in shapes[:moe[0]]):
        return None
    return moe[0]


def census_model_spec(config: Any, arch: Optional[str] = None, name: Optional[str] = None) -> Dict[str, Any]:
    """The ModelSpec the model Transformers builds of *config* states, as a mapping.

    Args:
        config: The model's Transformers config; a multimodal config's
            language model is the one described.
        arch: The family whose kinds and op profile the spec names; the one
            the model's name belongs to where omitted.
        name: The spec's name, the config's model type where omitted.

    Returns:
        The spec, in the resolver's form: every field the layers state,
        the layer stack in ``layers``, each kind's op counts in ``ops``.

    Raises:
        ModelSpecError: If the family has no kind for a layer, or prices
            two differently built layers as one kind.
    """
    name = str(name or config.model_type)
    text = getattr(config, "text_config", None) or config
    profile = load_op_profile(arch or infer_arch(name))
    model = _language_model(text)
    spec: Dict[str, Any] = {"name": name, "arch": profile.arch, **_model_fields(model)}
    shapes = [layer_shape(layer) for layer in model.get_decoder().layers]
    layers, firsts = _stack(profile, shapes)
    censuses = {kind: census_kind(text, index) for kind, index in firsts.items()}
    _widths(spec, {kind: record.dims for kind, record in censuses.items()})
    spec["first_k_dense_replace"] = _first_dense(shapes)
    # A kind of the family no layer is keeps its profile's counts.
    spec["ops"] = {kind: counts.to_dict() for kind, counts in profile.kinds.items()}
    spec["ops"].update({kind: record.ops.to_dict() for kind, record in censuses.items()})
    spec["layers"] = [group.to_dict() for group in layers]
    return ModelSpec.from_dict(spec).to_dict()

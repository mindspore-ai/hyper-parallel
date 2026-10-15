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
"""A model's family, applied to its config, and its layer kinds.

A family is data: its op profile in ``auto_parallel/op_profiles``, or the op
counts its model spec declares, chosen by ``ccfg.arch``, which the parser
settles, never by matching the model name.  :func:`check_and_apply_custom_hook`
applies it where the estimators price a config: what the family decides
where the run states nothing (:func:`derive_family`), then the op counts of
its default kind.

The layer stack is data too (``ccfg.layer_stack``).  The estimators read each
layer's kind from :func:`layer_groups`, and :func:`apply_layer_kind` gives a
layer its kind from the fields :func:`bind_layer_stack` recorded when the
family was applied.  A family whose layers take some fields per layer rather
than on the model, cm's sharding, leaves them in ``ccfg.layer_fields`` for
every kind to assign.
"""
from typing import Any, Dict, List, Optional, Tuple
from hyper_parallel.auto_parallel._layer_stack import LayerStack, LinearAttentionDims
from hyper_parallel.auto_parallel._model_spec import ModelSpecError, OpCounts
from hyper_parallel.auto_parallel._op_profiles import (
    VISION_ARCH, LayerKind, family_profile, known_archs, load_op_profile,
)
from hyper_parallel.auto_parallel.sapp_nd.nd.common.apply_exec import apply_layer_strategy
from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import CostModelConfig
from hyper_parallel.auto_parallel.sapp_nd.nd.common.derive import derive_family
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.logger import logger


class CWrap:
    """Temporary evaluator-like instance"""

    def __init__(self, e) -> None:
        self.ccfg = e

    def set_ccfg(self, hook):
        """Apply a hook to the wrapped config object."""
        return hook(self.ccfg)

    def get_model_name(self):
        """Return model name from the wrapped config."""
        return self.ccfg.model_name

    def reset(self, e):
        """Replace the wrapped config object."""
        self.ccfg = e

    def __getattr__(self, attr):
        if attr not in self.__dict__:
            return lambda *args, **kwargs: None
        return self.__dict__[attr]

    def set_strategy(self, **kwargs):
        """Forward strategy updates to the wrapped config."""
        self.ccfg.set_strategy(**kwargs)

    def get_strategy(self):
        """Return strategy from the wrapped config."""
        return self.ccfg.get_strategy()


def layer_op_counts(ccfg: Any, arch: str, kind: str) -> OpCounts:
    """Return the op counts of one layer kind.

    The counts the parser recorded from the model spec when there are any,
    else the family's own profile, which is what a config assembled without
    a parser gets.

    Raises:
        ModelSpecError: If the recorded counts have no such kind.
    """
    table = getattr(ccfg, "op_counts", None)
    if not table:
        return load_op_profile(arch).counts(kind)
    if kind not in table:
        raise ModelSpecError(
            f"{ccfg.model_name}: no op counts for layer kind {kind!r}, "
            f"the spec declares {sorted(table)}"
        )
    return table[kind]


def apply_op_counts(ccfg: Any, counts: OpCounts) -> None:
    """Set one layer kind's op counts on *ccfg*, each as ``n_<op>``."""
    for name, count in counts.to_dict().items():
        setattr(ccfg, "n_" + name, count)
    # A layer keeps a cast beside each matmul where the run says it does,
    # derive's keeps_param_casts; a config derive has not seen, where the
    # optimizer does not shard.
    casts = getattr(ccfg, "keeps_param_casts", None)
    if casts is None:
        casts = not ccfg.has_op
    ccfg.n_attParamCast = ccfg.n_attMM if casts else 0
    ccfg.n_ffParamCast = ccfg.n_ffMM if casts else 0


# The fields an attention flavour assigns.  Every kind of a stack whose kinds
# differ in attention writes all of them, so applying kinds in place, one
# layer after another and in any order, leaves each layer the same config.
_ATTENTION_FIELDS = (
    "attn_kind", "a", "dh", "n_kv", "attn_output_gate", "attn_extra_p", "n_qknorm",
    "lin_n_k", "lin_d_k", "lin_n_v", "lin_d_v", "lin_conv",
)

# The fields a feed-forward flavour assigns, likewise.
_FFN_FIELDS = ("hff", "n_chosen_exp", "n_exp", "n_shared_exp", "ep")

# Strategy a kind still sets.  The guard refuses a strategy write through
# set_ccfg, so the applier gives them through apply_layer_strategy.
_STRATEGY_FIELDS = ("ep",)


def _linear_attention(snapshot: Any, linear: LinearAttentionDims) -> Dict[str, Any]:
    """The attention fields of a gated-DeltaNet layer.

    The flavour maps onto the q/k/v/o formula: the value heads carry the
    q-side width, the key heads the kv-side, and the output gate is a second
    q-wide tensor.  What the formula does not describe is stated apart: the
    short convolution over the projected stream, the two per-head gates'
    projections, each head's decay and time-step bias and the gated output
    norm's weight as extra parameters, the recurrent state update as the
    ``linrec`` op.
    """
    n_k, d_k = linear.num_key_heads, linear.key_head_dim
    n_v, d_v = linear.num_value_heads, linear.value_head_dim
    qkv_width = 2 * n_k * d_k + n_v * d_v
    return {
        "attn_kind": "linear",
        "a": n_v,
        "dh": d_v,
        "n_kv": n_k * d_k / d_v,
        "attn_output_gate": True,
        "attn_extra_p": linear.conv_kernel_dim * qkv_width + 2 * snapshot.h * n_v + 2 * n_v + d_v,
        # The kernel normalizes its queries and keys itself, with no weights.
        "n_qknorm": 0,
        "lin_n_k": n_k,
        "lin_d_k": d_k,
        "lin_n_v": n_v,
        "lin_d_v": d_v,
        "lin_conv": linear.conv_kernel_dim,
    }


def _feed_forward(snapshot: Any, flavour: Optional[str]) -> Dict[str, Any]:
    """The feed-forward fields of a layer of *flavour*, from the bound config.

    A MoE layer runs the routed experts at their width.  A dense layer runs
    one expert at the model's feed-forward width, the parser's ``hff``, with
    no shared expert and no expert parallelism.  A kind without a flavour
    keeps the model's own.
    """
    if flavour == "dense":
        return {"hff": snapshot.hff, "n_chosen_exp": 1, "n_exp": 1, "n_shared_exp": 0, "ep": 1}
    fields = {name: getattr(snapshot, name) for name in _FFN_FIELDS}
    if flavour == "moe":
        fields["hff"] = snapshot.hff_exp
    return fields


def _kind_fields(snapshot: Any, stack: LayerStack, kind: LayerKind) -> Dict[str, Any]:
    """The fields *kind* assigns beyond its op counts, from the bound config.

    For every flavour some kind of the stack states, the fields it assigns,
    with the kind's values or the model's; then the fields the family gives
    every layer, and the kind's census record, None where the spec states
    none.
    """
    kinds = stack.distinct_kinds()
    fields: Dict[str, Any] = {}
    if any(other.attention != "full" for other in kinds):
        if kind.attention == "linear":
            fields.update(_linear_attention(snapshot, stack.linear))
        else:
            fields.update({name: getattr(snapshot, name) for name in _ATTENTION_FIELDS})
    if any(other.ffn is not None for other in kinds):
        fields.update(_feed_forward(snapshot, kind.ffn))
    fields.update(getattr(snapshot, "layer_fields", None) or {})
    fields["kind_activations"] = (getattr(snapshot, "census", None) or {}).get(kind.name)
    return fields


def bind_layer_stack(ccfg: Any) -> None:
    """Record, per kind of the config's stack, the fields it assigns.

    Called where the family is applied, per search candidate and per
    estimate, so the values a kind restores are the model's own, as
    applying its family left them.  A config without a stack gets no
    binding.
    """
    stack = getattr(ccfg, "layer_stack", None)
    if stack is None:
        ccfg.layer_binding = None
        return
    ccfg.layer_binding = {
        kind.name: _kind_fields(ccfg, stack, kind) for kind in stack.distinct_kinds()
    }


def apply_layer_kind(e: Any, kind: LayerKind) -> None:
    """Make the layer about to be priced one of *kind*.

    The memory backbone calls it with an evaluator, the performance path
    with a bare config.  A config without a stack gives the kind its op
    counts only.
    """
    if isinstance(e, CostModelConfig):
        e = CWrap(e)
    if getattr(e.ccfg, "layer_binding", None) is None:
        bind_layer_stack(e.ccfg)
    binding = e.ccfg.layer_binding
    fields = binding[kind.name] if binding is not None else {}

    def assign(c: Any) -> None:
        """Give config *c* the kind's counts, then its model fields."""
        apply_op_counts(c, kind.ops)
        for name, value in fields.items():
            if name not in _STRATEGY_FIELDS:
                setattr(c, name, value)

    e.set_ccfg(assign)
    apply_layer_strategy(e.ccfg, {name: fields[name] for name in _STRATEGY_FIELDS if name in fields})


def _needs_kinds(stack: LayerStack) -> bool:
    """Whether a stack's layers differ from the config applying its family leaves."""
    kinds = stack.distinct_kinds()
    return len(kinds) > 1 or any(kind.attention != "full" or kind.ffn is not None for kind in kinds)


def layer_groups(ccfg: Any) -> List[Tuple[Optional[LayerKind], int]]:
    """Return the groups of layers the estimators price alike, in model order.

    Returns:
        ``(kind, count)`` per group of the config's stack.  A config without
        a stack, or whose stack's one kind is the config applying its family
        leaves, is one group of all its layers, MTP included, with no kind:
        they are priced on the config as it stands.
    """
    stack = getattr(ccfg, "layer_stack", None)
    if stack is None or not _needs_kinds(stack):
        return [(None, int(ccfg.n_lay + ccfg.n_mtp))]
    return [(group.kind, group.count) for group in stack.groups]


def layer_kinds(ccfg: Any) -> List[Optional[LayerKind]]:
    """Return the kind of every layer in model order, as :func:`layer_groups` groups them."""
    return [kind for kind, count in layer_groups(ccfg) for _ in range(count)]


def apply_family(ccfg: Any) -> None:
    """Give *ccfg* what its family decides, and the op counts of its default kind.

    A vision tower takes the vision profile's encoder counts, whatever
    counts it carries from its language model.  Each layer of a stack takes
    its own kind's counts; the default kind's are the ones the model's
    config prices its embedding and output layer with, so t5 names its
    encoder (I15).  A family that names none gives the model no counts.
    """
    derive_family(ccfg)
    arch = getattr(ccfg, "arch", None)
    if arch == VISION_ARCH:
        apply_op_counts(ccfg, load_op_profile(VISION_ARCH).counts("encoder"))
        return
    profile = family_profile(arch)
    if profile.default is not None:
        apply_op_counts(ccfg, layer_op_counts(ccfg, profile.arch, profile.default))


def check_and_apply_custom_hook(e: Any) -> None:
    """Apply the family the config declares in ``ccfg.arch``, :func:`apply_family`.

    A config whose arch names no profile is priced as the default family.
    Then bind the config's layer stack, so each kind restores the values
    applying the family left.
    """
    if isinstance(e, CostModelConfig):
        e = CWrap(e)
    arch = getattr(e.ccfg, "arch", None)
    if arch not in known_archs():
        logger.warning(
            "No op profile for: %s (arch %r). The default one is chosen",
            e.get_model_name(), arch,
        )
    e.set_ccfg(apply_family)
    bind_layer_stack(e.ccfg)

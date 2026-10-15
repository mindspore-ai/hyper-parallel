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
"""The layer stack: which kind of layer sits where, as data.

A model's stack is a tuple of :class:`LayerGroup`, consecutive layers of one
kind of its op profile, body first and MTP layers last.  A spec may state it
in ``layers``.  Otherwise :func:`derive_layers` builds it from the fields every
config already carries, by the rules the cost model's hooks used to encode:

- ``layer_types`` names each layer's kind, when the profile has those kinds
  (Qwen3.5 alternates ``linear_attention`` and ``full_attention``);
- a profile with ``dense`` and ``moe`` kinds runs ``first_k_dense_replace``
  dense layers and then MoE layers (DeepSeek);
- a profile with ``encoder`` and ``decoder`` kinds runs one half of each (t5);
- any other model runs the profile's default kind throughout;
- the ``mtp_depth`` MTP layers repeat the kind of the last body layer.

A vision tower's stack is the same, of kinds of the vision profile: one
encoder group unless the tower states otherwise.

:func:`resolve_layers` checks a stack against its profile and the model's
dimensions and returns it as a :class:`LayerStack`, each kind resolved.
"""
import logging
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from hyper_parallel.auto_parallel._model_spec import (
    LayerGroup,
    ModelSpec,
    ModelSpecError,
    OpCounts,
    VisionSpec,
)
from hyper_parallel.auto_parallel._op_profiles import (
    VISION_ARCH,
    LayerKind,
    OpProfile,
    load_op_profile,
    resolve_ops,
)

logger = logging.getLogger(__name__)

# The spec fields of a gated-DeltaNet layer, and the dimension each one sets.
_LINEAR_FIELDS = (
    ("linear_num_key_heads", "num_key_heads"),
    ("linear_key_head_dim", "key_head_dim"),
    ("linear_num_value_heads", "num_value_heads"),
    ("linear_value_head_dim", "value_head_dim"),
    ("linear_conv_kernel_dim", "conv_kernel_dim"),
)


@dataclass(frozen=True)
class LinearAttentionDims:
    """The dimensions a linear-attention (gated DeltaNet) layer is priced on."""

    num_key_heads: int
    key_head_dim: int
    num_value_heads: int
    value_head_dim: int
    conv_kernel_dim: int

    @classmethod
    def from_fields(cls, values: Mapping[str, Any]) -> Optional["LinearAttentionDims"]:
        """Build the dimensions from ``linear_*`` fields, or ``None`` if none is set.

        Raises:
            ModelSpecError: If some are set but not all, or one is not a
                positive whole number.
        """
        found = {name: values.get(name) for name, _ in _LINEAR_FIELDS}
        if all(value is None for value in found.values()):
            return None
        missing = [name for name, value in found.items() if value is None]
        if missing:
            raise ModelSpecError(f"linear attention is missing {missing}; declare all of {list(found)}")
        dims: Dict[str, int] = {}
        for name, attr in _LINEAR_FIELDS:
            value = found[name]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ModelSpecError(f"{name} must be a positive whole number, got {value!r}")
            dims[attr] = value
        return cls(**dims)

    def __deepcopy__(self, memo: Dict[int, Any]) -> "LinearAttentionDims":
        """Frozen and made of immutable fields: a deep copy is the record itself."""
        del memo
        return self


@dataclass(frozen=True)
class StackGroup:
    """A group of a resolved stack: its kind, as the profile defines it."""

    kind: LayerKind
    count: int
    mtp: bool = False

    def __deepcopy__(self, memo: Dict[int, Any]) -> "StackGroup":
        """Frozen and made of immutable fields: a deep copy is the record itself."""
        del memo
        return self


@dataclass(frozen=True)
class LayerStack:
    """A model's layer stack, checked against its profile.

    Attributes:
        arch: The profile the kinds come from.
        groups: The groups in model order, body first and MTP layers last.
        linear: The linear-attention dimensions, when a kind needs them.
    """

    arch: str
    groups: Tuple[StackGroup, ...]
    linear: Optional[LinearAttentionDims] = None

    def kinds(self) -> Tuple[LayerKind, ...]:
        """Return the kind of every layer, in model order."""
        return tuple(group.kind for group in self.groups for _ in range(group.count))

    def distinct_kinds(self) -> Tuple[LayerKind, ...]:
        """Return each kind the stack uses once, in order of first use."""
        seen: Dict[str, LayerKind] = {}
        for group in self.groups:
            seen.setdefault(group.kind.name, group.kind)
        return tuple(seen.values())

    def to_layers(self) -> Tuple[LayerGroup, ...]:
        """Return the stack in its serialised form, the groups of ``ModelSpec.layers``."""
        return tuple(LayerGroup(group.kind.name, group.count, group.mtp) for group in self.groups)

    def __deepcopy__(self, memo: Dict[int, Any]) -> "LayerStack":
        """Frozen and made of immutable fields: a deep copy is the record itself."""
        del memo
        return self


def _groups(kinds: Sequence[str]) -> Tuple[LayerGroup, ...]:
    """Run-length group a sequence of kind names."""
    groups = []
    for kind in kinds:
        if groups and groups[-1][0] == kind:
            groups[-1][1] += 1
        else:
            groups.append([kind, 1])
    return tuple(LayerGroup(kind, count) for kind, count in groups)


def _body_from_layer_types(profile: OpProfile, num_layers: int,
                           layer_types: Sequence[Any]) -> Optional[Tuple[LayerGroup, ...]]:
    """The body named by ``layer_types``, or ``None`` if the profile lacks its kinds."""
    if len(layer_types) < num_layers:
        raise ModelSpecError(
            f"layer_types lists {len(layer_types)} layers, but num_hidden_layers is {num_layers}"
        )
    kinds = [str(kind) for kind in layer_types[:num_layers]]
    unknown = sorted(set(kinds) - set(profile.layer_kinds))
    if unknown:
        logger.warning(
            "layer_types names %s, which op profile %r does not declare; every layer is priced as %r",
            unknown, profile.arch, profile.default,
        )
        return None
    return _groups(kinds)


def _structural_body(profile: OpProfile, num_layers: int, first_k_dense: int) -> Tuple[LayerGroup, ...]:
    """The body the profile's kinds imply: dense then MoE, two halves, or one kind."""
    names = set(profile.layer_kinds)
    if {"dense", "moe"} <= names:
        dense = min(max(int(first_k_dense), 0), num_layers)
        groups = (LayerGroup("dense", dense), LayerGroup("moe", num_layers - dense))
        return tuple(group for group in groups if group.count)
    if {"encoder", "decoder"} <= names:
        if num_layers % 2:
            raise ModelSpecError(
                f"an encoder-decoder stack needs an even layer count, got {num_layers}"
            )
        half = num_layers // 2
        return (LayerGroup("encoder", half), LayerGroup("decoder", half)) if half else ()
    if profile.default is None:
        raise ModelSpecError(
            f"op profile {profile.arch!r} has several kinds and names no default; state the stack in layers"
        )
    return (LayerGroup(profile.default, num_layers),) if num_layers else ()


def derive_layers(profile: OpProfile, num_layers: int, mtp_depth: int = 0,
                  layer_types: Optional[Sequence[Any]] = None,
                  first_k_dense: int = 0) -> Tuple[LayerGroup, ...]:
    """Build the stack a config implies, when it does not state one.

    Args:
        profile: The model's op profile, which names the kinds.
        num_layers: The number of body layers.
        mtp_depth: The number of MTP layers after the body.
        layer_types: The per-layer kind names a config such as Qwen3.5 lists.
        first_k_dense: How many leading layers are dense in a dense-then-MoE
            family.

    Returns:
        The groups in model order, body first and MTP layers last.

    Raises:
        ModelSpecError: If ``layer_types`` lists fewer layers than the body,
            or no rule gives the body a kind.
    """
    body = None
    if layer_types:
        body = _body_from_layer_types(profile, num_layers, layer_types)
    if body is None:
        body = _structural_body(profile, num_layers, first_k_dense)
    if not mtp_depth:
        return body
    last = body[-1].kind if body else profile.default
    if last is None:
        raise ModelSpecError(f"op profile {profile.arch!r} gives the MTP layers no kind")
    return body + (LayerGroup(last, int(mtp_depth), mtp=True),)


def resolve_kinds(profile: OpProfile, ops: Optional[Mapping[str, OpCounts]] = None) -> Dict[str, LayerKind]:
    """Return the profile's kinds, with declared *ops* replacing their counts.

    Raises:
        ModelSpecError: If *ops* names other kinds than the profile's.
    """
    counts = resolve_ops(profile.arch, ops)
    return {
        name: LayerKind(name, counts[name], kind.attention, kind.ffn)
        for name, kind in profile.layer_kinds.items()
    }


def resolve_layers(arch: str, layers: Sequence[LayerGroup],
                   ops: Optional[Mapping[str, OpCounts]] = None,
                   linear: Optional[LinearAttentionDims] = None) -> LayerStack:
    """Check a stack against its profile and resolve each group's kind.

    Args:
        arch: The model's op profile.
        layers: The groups, in model order.
        ops: Declared op counts that replace the profile's.
        linear: The model's linear-attention dimensions, if it declares any.

    Returns:
        The stack with every kind resolved.

    Raises:
        ModelSpecError: If a group names a kind the profile lacks, or a
            linear kind has no linear-attention dimensions to be priced on.
    """
    kinds = resolve_kinds(load_op_profile(arch), ops)
    groups = []
    for index, group in enumerate(layers):
        if group.kind not in kinds:
            raise ModelSpecError(
                f"layers[{index}] names kind {group.kind!r}, but op profile {arch!r} "
                f"has {sorted(kinds)}"
            )
        groups.append(StackGroup(kinds[group.kind], group.count, group.mtp))
    needs_linear = any(group.kind.attention == "linear" for group in groups)
    if needs_linear and linear is None:
        raise ModelSpecError(
            f"op profile {arch!r} prices linear-attention layers, but the model declares no "
            f"{', '.join(name for name, _ in _LINEAR_FIELDS)}"
        )
    return LayerStack(arch, tuple(groups), linear if needs_linear else None)


def spec_layer_stack(spec: ModelSpec) -> LayerStack:
    """Return a spec's stack, as it states it or as its fields imply.

    Raises:
        ModelSpecError: If the spec has no arch, or its stack does not fit
            its profile.
    """
    if spec.arch is None:
        raise ModelSpecError(f"{spec.name}: a layer stack needs the spec's arch")
    linear = LinearAttentionDims.from_fields(
        {name: getattr(spec, name) for name, _ in _LINEAR_FIELDS}
    )
    layers = spec.layers
    if layers is None:
        layers = derive_layers(
            load_op_profile(spec.arch),
            spec.num_hidden_layers,
            spec.mtp_depth or 0,
            layer_types=spec.layer_types,
            first_k_dense=spec.first_k_dense_replace or 0,
        )
    return resolve_layers(spec.arch, layers, spec.ops, linear)


def tower_layer_stack(vision: VisionSpec) -> LayerStack:
    """Return a vision tower's stack, as it states it or as one encoder group.

    Raises:
        ModelSpecError: If the tower names a kind the vision profile lacks.
    """
    layers = vision.layers
    if layers is None:
        layers = derive_layers(load_op_profile(VISION_ARCH), vision.num_hidden_layers)
    return resolve_layers(VISION_ARCH, layers)

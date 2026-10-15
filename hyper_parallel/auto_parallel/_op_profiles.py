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
"""Op profiles: the layer kinds of each known architecture family, as data.

Every file under ``op_profiles/`` names a family and declares its layer kinds.
A kind states how many times one layer runs each op the cost model prices,
the numbers the ND arch hooks used to assign in code, and two flavours from a
closed set that say how the layer is shaped:

- ``attention``: ``full`` (the default) or ``linear``, a gated-DeltaNet layer
  priced on the model's linear-attention dimensions;
- ``ffn``: ``dense`` or ``moe``, or absent to keep the model's own
  feed-forward, for families whose layers do not differ in it.

A profile with more than one kind may name a ``default``, the kind a layer
takes when nothing says otherwise; a profile with one kind defaults to it.

A profile may also state what a model of its family has when its producer
does not say: under ``run``, the run's byte widths, whether its gradients
take memory without pipeline parallelism and whether tensor parallelism
shards the activations between layers (:data:`DEFAULT_RUN` otherwise); under
``model``, an MLA family's value-head width.  The cost model's ``derive``
reads them.

A model spec names its profile with ``arch``, and may declare its own counts
in ``ops`` instead.  The cost model reads only what the spec declares.  A
producer that has nothing better than a free-text model name, such as a
MindFormers ``trainer.model_name``, falls back on :func:`infer_arch`.
"""
import functools
import logging
import os
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple

import yaml

from hyper_parallel.auto_parallel._exec_spec import ExecSpec, ExecSpecError
from hyper_parallel.auto_parallel._model_spec import ModelSpecError, OpCounts

logger = logging.getLogger(__name__)

PROFILE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "op_profiles")
DEFAULT_ARCH = "default"
# The profile every vision tower is priced with.
VISION_ARCH = "vision"

ATTENTION_FLAVOURS = ("full", "linear")
FFN_FLAVOURS = ("dense", "moe")

# The run of a family's model when its producer states none of it, and the
# only execution-spec fields a profile's ``run`` may change: the byte widths
# of gradients, optimizer states, norm activations and dropout masks, whether
# gradients take memory without pipeline parallelism, whether tensor
# parallelism shards the activations between layers, and whether the loss
# runs on logits sharded over the vocabulary.
DEFAULT_RUN = MappingProxyType({
    "grad_bytes": 4,
    "optimizer_state_bytes": 4,
    "optimizer_states": 2,
    "main_param_bytes": 0,
    "norm_bytes": 4,
    "dropout_bytes": 0,
    "grad_accumulation": False,
    "shard_activations": False,
    "loss_parallel": True,
    "reshard_params": False,
    "deferred_grad_accumulation": False,
    "overlapped_grad_reduce": False,
})

# The model facts a profile's ``model`` may give a model that states none.
MODEL_DEFAULT_KEYS = ("v_head_dim",)

# What a model name opens with, with the family it means.  The longest that
# a name opens with wins, so Qwen3.5, under both the Transformers spelling
# and the release name, is not taken for the Qwen family it opens with too,
# and the order they are listed in decides nothing.
_NAME_PREFIXES = {
    "llama2": "llama2", "mixtral": "mixtral", "t5": "t5", "pangualpha": "pangualpha",
    "deepseek": "deepseek", "qwen3_5": "qwen3_5", "qwen3.5": "qwen3_5", "qwen": "qwen", "cm": "cm",
}
# The names the default profile was taken for, each said once a process.
_DEFAULTED_NAMES = set()


@dataclass(frozen=True)
class LayerKind:
    """One kind of layer: the ops it runs and how it is shaped.

    Attributes:
        name: The kind's name in its profile, which a layer stack refers to.
        ops: How many times one layer of this kind runs each op.
        attention: ``full``, or ``linear`` for a gated-DeltaNet layer.
        ffn: ``dense``, ``moe``, or ``None`` to keep the model's own.
    """

    name: str
    ops: OpCounts
    attention: str = "full"
    ffn: Optional[str] = None

    def __deepcopy__(self, memo: Dict[int, Any]) -> "LayerKind":
        """Frozen and made of immutable fields: a deep copy is the record itself."""
        del memo
        return self


@dataclass(frozen=True)
class OpProfile:
    """The layer kinds of one architecture family.

    Attributes:
        arch: The family, the profile's file name.
        layer_kinds: Every kind of the family, by name.
        default: The kind a layer takes when nothing says otherwise, or
            ``None`` for a family whose stack always states its kinds.
        run: The run of the family's models where their producer states
            none, every key of :data:`DEFAULT_RUN`.
        model: The model facts the family gives a model that states none.
    """

    arch: str
    layer_kinds: Dict[str, LayerKind]
    default: Optional[str] = None
    run: Mapping[str, Any] = field(default_factory=lambda: DEFAULT_RUN)
    model: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))

    @property
    def kinds(self) -> Dict[str, OpCounts]:
        """Return each kind's op counts, by kind name."""
        return {name: kind.ops for name, kind in self.layer_kinds.items()}

    def counts(self, kind: str) -> OpCounts:
        """Return one layer kind's counts, or raise naming the kinds there are."""
        return self.kind(kind).ops

    def kind(self, name: str) -> LayerKind:
        """Return one layer kind, or raise naming the kinds there are."""
        if name not in self.layer_kinds:
            raise ModelSpecError(
                f"op profile {self.arch!r} has no layer kind {name!r}; "
                f"it has {sorted(self.layer_kinds)}"
            )
        return self.layer_kinds[name]


@functools.lru_cache(maxsize=None)
def known_archs() -> Tuple[str, ...]:
    """Return the name of every family that has a profile file."""
    return tuple(sorted(
        name[:-len(".yaml")] for name in os.listdir(PROFILE_DIR) if name.endswith(".yaml")
    ))


def _flavour(value: Any, allowed: Tuple[str, ...], where: str, optional: bool) -> Optional[str]:
    """Return a flavour value, refusing one outside *allowed*."""
    if value is None and optional:
        return None
    if value not in allowed:
        raise ModelSpecError(f"{where} must be one of {list(allowed)}, got {value!r}")
    return str(value)


def _layer_kind(arch: str, name: str, data: Any) -> LayerKind:
    """Build one kind of a profile from its ``{ops, attention, ffn}`` mapping."""
    where = f"{arch}.kinds.{name}"
    if not isinstance(data, Mapping):
        raise ModelSpecError(f"{where} must map ops and flavours, got {data!r}")
    unknown = sorted(set(data) - {"ops", "attention", "ffn"})
    if unknown:
        raise ModelSpecError(f"{where} has unknown keys {unknown}; a kind has ops, attention and ffn")
    return LayerKind(
        name=str(name),
        ops=OpCounts.from_dict(data.get("ops"), f"{where}.ops"),
        attention=_flavour(data.get("attention", "full"), ATTENTION_FLAVOURS, f"{where}.attention", False),
        ffn=_flavour(data.get("ffn"), FFN_FLAVOURS, f"{where}.ffn", True),
    )


def _run_defaults(arch: str, data: Any) -> Mapping[str, Any]:
    """Return a family's run: :data:`DEFAULT_RUN`, with what its profile states."""
    stated = {} if data is None else data
    if not isinstance(stated, Mapping):
        raise ModelSpecError(f"{arch}.run must map run facts to values, got {data!r}")
    unknown = sorted(set(stated) - set(DEFAULT_RUN))
    if unknown:
        raise ModelSpecError(f"{arch}.run has unknown keys {unknown}; a family states {sorted(DEFAULT_RUN)}")
    try:
        spec = ExecSpec.from_dict(stated)
    except ExecSpecError as exc:
        raise ModelSpecError(f"{arch}.run: {exc}") from exc
    return MappingProxyType({**DEFAULT_RUN, **spec.to_dict()})


def _model_defaults(arch: str, data: Any) -> Mapping[str, int]:
    """Return the model facts a family gives a model that states none."""
    stated = {} if data is None else data
    if not isinstance(stated, Mapping):
        raise ModelSpecError(f"{arch}.model must map model facts to values, got {data!r}")
    unknown = sorted(set(stated) - set(MODEL_DEFAULT_KEYS))
    if unknown:
        raise ModelSpecError(
            f"{arch}.model has unknown keys {unknown}; a family states {list(MODEL_DEFAULT_KEYS)}"
        )
    for key, value in stated.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ModelSpecError(f"{arch}.model.{key} must be a positive whole number, got {value!r}")
    return MappingProxyType(dict(stated))


@functools.lru_cache(maxsize=None)
def load_op_profile(arch: str) -> OpProfile:
    """Read and validate the profile of one family.

    Raises:
        ModelSpecError: If no profile has that name, or the file declares no
            layer kind, an unknown key, an incomplete op vector, an unknown
            flavour, a default that is not one of its kinds, or a run or
            model default it cannot state.
    """
    if arch not in known_archs():
        raise ModelSpecError(
            f"unknown arch {arch!r}; the op profiles are {list(known_archs())}"
        )
    with open(os.path.join(PROFILE_DIR, f"{arch}.yaml"), encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    unknown = sorted(set(data) - {"kinds", "default", "run", "model"})
    if unknown:
        raise ModelSpecError(f"op profile {arch!r} has unknown keys {unknown}")
    kinds = data.get("kinds")
    if not isinstance(kinds, Mapping) or not kinds:
        raise ModelSpecError(f"op profile {arch!r} declares no layer kind")
    layer_kinds = {str(name): _layer_kind(arch, name, kind) for name, kind in kinds.items()}
    default = data.get("default")
    if default is None and len(layer_kinds) == 1:
        default = next(iter(layer_kinds))
    if default is not None and default not in layer_kinds:
        raise ModelSpecError(
            f"op profile {arch!r} names default kind {default!r}, but its kinds are {sorted(layer_kinds)}"
        )
    return OpProfile(arch, layer_kinds, default, _run_defaults(arch, data.get("run")),
                     _model_defaults(arch, data.get("model")))


def family_profile(arch: Optional[str]) -> OpProfile:
    """Return the profile of *arch*, or the default family's when no profile has that name."""
    return load_op_profile(arch if arch in known_archs() else DEFAULT_ARCH)


def infer_arch(name: Any) -> str:
    """Return the family a free-text model name belongs to.

    The family the lower-cased name opens with, the longest first, else the
    default profile, said once a name.  Matched anywhere in the name, in a
    fixed order, as the cost model matched names before the family became
    data, ``cm`` took any name holding those two letters and a family listed
    earlier won over the one the name opens with (I8).  Only a producer with
    no better source should use it; a spec that declares ``arch`` never gets
    here.
    """
    lowered = str(name).lower()
    for prefix in sorted(_NAME_PREFIXES, key=len, reverse=True):
        if lowered.startswith(prefix):
            return _NAME_PREFIXES[prefix]
    if lowered not in _DEFAULTED_NAMES:
        # Once: a search parses a config more than once.
        _DEFAULTED_NAMES.add(lowered)
        logger.warning(
            "no op profile matches model %r, pricing it as %r; declare arch to choose one",
            name, DEFAULT_ARCH,
        )
    return DEFAULT_ARCH


# What a lower-cased model name contains when its attention normalizes each
# head's queries and keys: the Qwen3 generation, Qwen3.5 and the Qwen3
# vision-language models included.
_QK_NORM_NAMES = ("qwen3",)


def infer_qk_norm(name: Any) -> bool:
    """Return whether a model named *name* normalizes each head's queries and keys.

    For a producer whose config does not state it, as a Transformers config
    does not: the Qwen3 generation does, and no other family the cost model
    prices does.
    """
    lowered = str(name).lower()
    return any(pattern in lowered for pattern in _QK_NORM_NAMES)


# What a lower-cased model name contains when its attention projects queries,
# keys and values with a bias and its output without, which its Transformers
# config does not state: the Qwen2 generation.
_QKV_BIAS_NAMES = ("qwen2",)

# What a lower-cased model name contains when its MoE layer gates its shared
# expert's output with a weight of its own: Qwen2-MoE, Qwen3.5 and Qwen3-Next.
_SHARED_EXPERT_GATE_NAMES = ("qwen2_moe", "qwen3_5", "qwen3_next")


def infer_qkv_bias(name: Any) -> bool:
    """Return whether a model named *name* biases its query, key and value projections.

    For a Transformers config that states no ``attention_bias``, as the
    Qwen2 generation's, whose projections have one.
    """
    lowered = str(name).lower()
    return any(pattern in lowered for pattern in _QKV_BIAS_NAMES)


def infer_shared_expert_gate(name: Any) -> bool:
    """Return whether a model named *name* gates its shared expert's output with a weight of its own."""
    lowered = str(name).lower()
    return any(pattern in lowered for pattern in _SHARED_EXPERT_GATE_NAMES)


def resolve_ops(arch: str, ops: Optional[Mapping[str, OpCounts]] = None) -> Dict[str, OpCounts]:
    """Return the op counts a model is priced with, per layer kind.

    Declared *ops* replace the profile's, but must name the same layer kinds:
    the family's hooks decide which layers are of which kind, so counts for a
    kind they never assign would be dropped without a word.

    Raises:
        ModelSpecError: If *arch* has no profile, or *ops* names other kinds.
    """
    profile = load_op_profile(arch)
    if ops is None:
        return dict(profile.kinds)
    if set(ops) != set(profile.kinds):
        raise ModelSpecError(
            f"ops declares layer kinds {sorted(ops)}, but arch {arch!r} "
            f"prices {sorted(profile.kinds)}"
        )
    return dict(ops)

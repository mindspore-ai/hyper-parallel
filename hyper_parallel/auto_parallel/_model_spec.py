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
"""The model IR: a typed, serialisable description of what is being trained.

``memory_estimation/README.md`` already calls ``CostModelConfig`` "the
interface between framework input files and the memory module".  It cannot
be one, because it carries derived values, fuses model shape with parallel
strategy, and has no serialised form -- so the interface gets faked, most
visibly by writing a temporary YAML to disk and parsing it back.

:class:`ModelSpec` is that interface made real for the model half: the facts
about an architecture that hold whatever strategy it is trained under.  It is
a schema, not a grammar.  Three producers fill it -- a framework parser, a
census of an instantiated model, or an engineer writing YAML by hand -- and
every consumer downstream reads the same fields.

Two rules make it safe to hand-fill, both learned from Qwen3.5-35B-A3B, which
reached the cost model with its MTP layer dropped, its shared expert priced at
width zero, and half its attention parameters missing, all silently:

- **Absent is not zero.**  Every optional field defaults to ``None``.  A width
  nothing supplied can then be told apart from a width declared as ``0``,
  which is what let an all-MoE config price its shared expert at no width at
  all and then divide by it.
- **Refuse rather than default.**  :meth:`ModelSpec.validate` rejects a spec
  whose fields contradict each other, at parse time and by name, instead of
  resolving to a number that is wrong further downstream.

The strategy half (degrees, sharding, recompute, precision) is the execution
spec's, :class:`~hyper_parallel.auto_parallel._exec_spec.ExecSpec`.  A key that
states neither the model nor the run is refused.
"""
from dataclasses import dataclass, fields
from typing import Any, Dict, Mapping, Optional, Tuple


# The model facts a spec states as true or false.
_FLAGS = ("attn_output_gate", "tie_word_embeddings", "qk_norm", "qkv_bias", "o_bias", "mlp_bias", "norm_bias",
          "shared_expert_gate")

# The legacy keys that state a run's precision.  Refused as model facts like
# any unknown key, but named with the keys a run reads instead: a description
# stating one among its facts was once priced at the default width (D17).
_RUN_PRECISION_KEYS = ("compute_dtype", "param_init_type")


class ModelSpecError(ValueError):
    """A model spec is missing a required fact or contradicts itself.

    Raised at parse time, by field name, so a producer learns what it failed
    to supply instead of a consumer inheriting a zero.
    """


def _as_int(value: Any, name: str) -> Optional[int]:
    """Coerce *value* to ``int``, or raise naming the field it came from."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ModelSpecError(f"{name} must be an integer, got {value!r}") from exc


def _as_count(value: Any, name: str) -> int:
    """Coerce *value* to a non-negative whole number, or raise naming it."""
    if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
        raise ModelSpecError(f"{name} must be a whole number, got {value!r}")
    count = _as_int(value, name)
    if count is None or count < 0:
        raise ModelSpecError(f"{name} must be a non-negative count, got {value!r}")
    return count


@dataclass(frozen=True)
class OpCounts:
    """How many times one layer kind runs each op the cost model prices.

    The estimators already price a layer as a count times a per-op cost
    (``perf_estimation/getters.py``); this is the count half, declared as data
    instead of assigned by a per-family callback.  It belongs to a layer
    *kind*, not to a model: an encoder-decoder, or a stack mixing linear and
    full attention, runs different ops in different layers.

    Every op is required.  A count left out would be priced as zero, which is
    exactly the silent failure the model IR exists to refuse, so a layer that
    does not run an op declares ``0``.  The parameter-cast counts are absent on
    purpose: whether parameters are cast depends on optimizer sharding, which
    is strategy, so the consumer derives them from ``attMM`` and ``ffMM``.
    """

    # The fields are the op names of ND's op vector, spelled as the op
    # profiles' yaml, the recompute switches and every census record spell
    # them (attMM, normOp, ...): renaming one renames those keys.
    # pylint: disable=invalid-name
    attMM: int     # attention projections: q, k, v, o
    attBMM: int    # attention batched matmuls: QK^T and AV
    ffMM: int      # feed-forward projections, e.g. 3 for a gated MLP
    softmax: int
    dropout: int
    normOp: int
    gather: int    # tensor-parallel gathers
    headCast: int  # casts of the per-head score tensor
    ffAct: int     # feed-forward activation functions
    linrec: int    # linear-attention state updates: the delta rule's recurrence
    # pylint: enable=invalid-name

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], where: str = "ops") -> "OpCounts":
        """Build the vector from a mapping, refusing a missing or unknown op.

        Args:
            data: ``{op name: count}``, one entry per op.
            where: Where the mapping came from, for the error message.

        Raises:
            ModelSpecError: If an op is missing, unknown, or not a count.
        """
        if not isinstance(data, Mapping):
            raise ModelSpecError(f"{where} must map op names to counts, got {data!r}")
        names = [f.name for f in fields(cls)]
        unknown = sorted(set(data) - set(names))
        if unknown:
            raise ModelSpecError(f"{where} declares unknown ops {unknown}; the ops are {names}")
        missing = [name for name in names if name not in data]
        if missing:
            raise ModelSpecError(
                f"{where} is missing counts for {missing}; declare 0 for an op "
                "the layer does not run"
            )
        return cls(**{name: _as_count(data[name], f"{where}.{name}") for name in names})

    def to_dict(self) -> Dict[str, int]:
        """Return the vector as ``{op name: count}``, in declaration order."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def __deepcopy__(self, memo: Dict[int, Any]) -> "OpCounts":
        """Frozen and made of immutable fields: a deep copy is the record itself."""
        del memo
        return self


def ops_from_dict(data: Any) -> Dict[str, OpCounts]:
    """Parse ``{layer kind: {op: count}}``, the serialised form of ``ModelSpec.ops``."""
    if not isinstance(data, Mapping) or not data:
        raise ModelSpecError(
            f"ops must map at least one layer kind to its op counts, got {data!r}"
        )
    return {str(kind): OpCounts.from_dict(counts, f"ops.{kind}") for kind, counts in data.items()}


@dataclass(frozen=True)
class LayerGroup:
    """Consecutive layers of one kind, in model order.

    A model's stack is a tuple of groups: DeepSeek-V3 is three dense layers,
    58 MoE layers and one MTP layer; Qwen3.5 alternates groups of linear- and
    full-attention layers.  ``kind`` names a kind of the model's op profile.
    MTP layers are body-shaped layers after the body, marked so that nothing
    has to count them from the end.
    """

    kind: str
    count: int
    mtp: bool = False

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], where: str = "layers") -> "LayerGroup":
        """Build a group from ``{kind, count, mtp}``, refusing anything else.

        Raises:
            ModelSpecError: If the kind is missing, the count is not a
                positive whole number, or an unknown key is present.
        """
        if not isinstance(data, Mapping):
            raise ModelSpecError(f"{where} must be a mapping of kind and count, got {data!r}")
        unknown = sorted(set(data) - {"kind", "count", "mtp"})
        if unknown:
            raise ModelSpecError(f"{where} declares unknown keys {unknown}; the keys are kind, count, mtp")
        kind = data.get("kind")
        if not isinstance(kind, str) or not kind:
            raise ModelSpecError(f"{where}.kind must name a layer kind, got {kind!r}")
        count = _as_count(data.get("count"), f"{where}.count")
        if count == 0:
            raise ModelSpecError(f"{where}.count must be positive; leave out a group with no layers")
        mtp = data.get("mtp", False)
        if not isinstance(mtp, bool):
            raise ModelSpecError(f"{where}.mtp must be true or false, got {mtp!r}")
        return cls(kind=kind, count=count, mtp=mtp)

    def to_dict(self) -> Dict[str, Any]:
        """Return the group as a mapping, omitting ``mtp`` when it is false."""
        out: Dict[str, Any] = {"kind": self.kind, "count": self.count}
        if self.mtp:
            out["mtp"] = True
        return out


def layers_from_list(data: Any, where: str = "layers") -> Tuple[LayerGroup, ...]:
    """Parse a list of groups, the serialised form of ``ModelSpec.layers``."""
    if not isinstance(data, (list, tuple)) or not data:
        raise ModelSpecError(f"{where} must list at least one group, got {data!r}")
    return tuple(LayerGroup.from_dict(group, f"{where}[{index}]") for index, group in enumerate(data))


def check_layer_counts(layers: Tuple[LayerGroup, ...], num_layers: int, mtp_depth: int,
                       where: str = "layers") -> None:
    """Raise unless *layers* covers exactly the body and the MTP layers, in order.

    Args:
        layers: The groups, in model order.
        num_layers: The number of body layers the model declares.
        mtp_depth: The number of MTP layers the model declares.
        where: The field the groups were read from, for the message.

    Raises:
        ModelSpecError: If a body group follows an MTP group, or either
            part does not sum to its declared count.
    """
    seen_mtp = False
    for index, group in enumerate(layers):
        if seen_mtp and not group.mtp:
            raise ModelSpecError(f"{where}[{index}] is a body group after an MTP group; MTP layers come last")
        seen_mtp = seen_mtp or group.mtp
    body = sum(group.count for group in layers if not group.mtp)
    mtp = sum(group.count for group in layers if group.mtp)
    if body != num_layers:
        raise ModelSpecError(f"{where} list {body} body layers, but num_hidden_layers is {num_layers}")
    if mtp != mtp_depth:
        raise ModelSpecError(f"{where} list {mtp} MTP layers, but mtp_depth is {mtp_depth}")


# The fields a census record states in pairs: what a layer keeps under
# HyperParallel's selective activation checkpointing, the shares of its
# matmul FLOPs that recomputes, and what it keeps for each op.
_PAIRED = (("selective", "selective_tp"), ("selective_attention_mm", "selective_ffn_mm"), ("ops", "ops_tp"))

# The census record's fields that map ops to bytes per token.
_BY_OP = ("ops", "ops_tp")


@dataclass(frozen=True)
class KindActivations:
    """What one layer of a kind keeps for its backward, and the most its backward holds, per token.

    Bytes per token at micro-batch 1, as a census measures them on one layer
    (:mod:`hyper_parallel.auto_parallel._layer_census`) at ``seq_length``
    tokens under the runtime's kernels: the part no tensor-parallel rank
    splits, and the part it splits, of which a layer at TP 2 holds half.  The
    backward's working set leaves out the parameters and their gradients,
    which the memory model counts on its own.  ``selective`` and
    ``selective_tp`` are what the layer keeps under HyperParallel's selective
    activation checkpointing, whose backward recomputes the rest and holds
    the same working set, and ``selective_attention_mm`` and
    ``selective_ffn_mm`` the shares of the FLOPs of its attention's
    projections and of the rest of its matmuls that recomputes.  ``ops`` and
    ``ops_tp`` are what it keeps for each op of the op vector, the two parts
    of ``saved`` and ``saved_tp``, and ``other`` for its own code: the op
    records of shared decision S1, as the census fills them.  A record
    states each pair whole or not at all.
    """

    saved: float
    saved_tp: float
    working: float
    working_tp: float
    seq_length: int
    selective: Optional[float] = None
    selective_tp: Optional[float] = None
    selective_attention_mm: Optional[float] = None
    selective_ffn_mm: Optional[float] = None
    ops: Optional[Mapping[str, float]] = None
    ops_tp: Optional[Mapping[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return the record as a plain mapping, the selective part and the ops only when stated."""
        out = {spec_field.name: getattr(self, spec_field.name) for spec_field in fields(self)
               if getattr(self, spec_field.name) is not None}
        for name in _BY_OP:
            if name in out:
                out[name] = dict(out[name])
        return out

    @staticmethod
    def _check_keys(data: Mapping[str, Any], names: Tuple[str, ...], where: str) -> None:
        """Raise unless *data* states the fields of *names*, each of :data:`_PAIRED` whole or not at all."""
        unknown = sorted(set(data) - set(names))
        missing = [name for name in names if name not in data and all(name not in pair for pair in _PAIRED)]
        if unknown or missing:
            raise ModelSpecError(f"{where} has unknown keys {unknown} and lacks {missing}")
        for pair in _PAIRED:
            if len({data.get(name) is None for name in pair}) > 1:
                raise ModelSpecError(f"{where} states one of {list(pair)} without the other")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], where: str = "activations") -> "KindActivations":
        """Build a record, refusing a key it does not know, a missing one, a negative size or a share above 1."""
        if not isinstance(data, Mapping):
            raise ModelSpecError(f"{where} must map the record's fields to their values, got {data!r}")
        names = tuple(spec_field.name for spec_field in fields(cls))
        cls._check_keys(data, names, where)
        sizes = {name: float(data[name]) for name in names
                 if name != "seq_length" and name not in _BY_OP and data.get(name) is not None}
        if any(size < 0 for size in sizes.values()):
            raise ModelSpecError(f"{where}: bytes per token cannot be negative, got {sizes}")
        if any(sizes.get(name, 0) > 1 for name in _PAIRED[1]):
            raise ModelSpecError(f"{where}: a share of FLOPs cannot exceed 1, got {sizes}")
        by_op = {name: _op_bytes(data[name], f"{where}.{name}") for name in _BY_OP if data.get(name) is not None}
        return cls(seq_length=_as_count(data["seq_length"], f"{where}.seq_length"), **sizes, **by_op)


def _op_bytes(data: Any, where: str) -> Dict[str, float]:
    """Parse ``{op: bytes per token}``, the ops of the op vector and ``other``, refusing a negative size."""
    known = [spec_field.name for spec_field in fields(OpCounts)] + ["other"]
    if not isinstance(data, Mapping) or any(op not in known for op in data):
        raise ModelSpecError(f"{where} must map ops of {known} to bytes per token, got {data!r}")
    sizes = {str(op): float(size) for op, size in data.items()}
    if any(size < 0 for size in sizes.values()):
        raise ModelSpecError(f"{where}: bytes per token cannot be negative, got {sizes}")
    return sizes


def activations_from_dict(data: Any) -> Dict[str, KindActivations]:
    """Parse a spec's ``activations``, a mapping of layer kind to its :class:`KindActivations`."""
    if not isinstance(data, Mapping):
        raise ModelSpecError(f"activations must map layer kinds to their records, got {data!r}")
    return {str(kind): KindActivations.from_dict(record, f"activations.{kind}") for kind, record in data.items()}


@dataclass(frozen=True)
class VisionSpec:
    """The vision tower of a multimodal model.

    Towers are nested rather than flattened with a prefix, so a spec can carry
    more than one and a consumer can ask whether there is one at all.
    ``max_position_embeddings`` here is the encoder sequence length in merged
    visual tokens, which depends on the images the dataset serves and is
    therefore an input, not a property of the checkpoint.  ``layers`` is the
    tower's stack, groups of kinds of the vision profile; a producer states
    one encoder group when the config states none.
    """

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    name: str = "vision"
    intermediate_size: Optional[int] = None
    out_hidden_size: Optional[int] = None
    patch_size: Optional[int] = None
    spatial_merge_size: Optional[int] = None
    num_position_embeddings: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    layers: Optional[Tuple[LayerGroup, ...]] = None

    def validate(self) -> None:
        """Raise :class:`ModelSpecError` if the tower cannot be costed."""
        for name in ("hidden_size", "num_hidden_layers", "num_attention_heads"):
            value = getattr(self, name)
            if not value or value <= 0:
                raise ModelSpecError(
                    f"vision.{name} is required and must be positive, got {value!r}"
                )
        if self.layers is not None:
            if any(group.mtp for group in self.layers):
                raise ModelSpecError("vision.layers marks MTP layers; a tower has none")
            check_layer_counts(self.layers, self.num_hidden_layers, 0, "vision.layers")

    def to_dict(self) -> Dict[str, Any]:
        """Return the tower as a plain mapping, unset fields omitted."""
        out: Dict[str, Any] = {}
        for spec_field in fields(self):
            value = getattr(self, spec_field.name)
            if value is None:
                continue
            out[spec_field.name] = [group.to_dict() for group in value] if spec_field.name == "layers" else value
        return out


@dataclass(frozen=True)
class ModelSpec:
    """What is being trained, independent of how it is parallelised.

    The four required fields are the ones every cost path dereferences
    unconditionally; the rest are optional because a dense model genuinely has
    no expert count, and ``None`` says so without claiming the count is zero.

    Attributes:
        arch: The op profile the model is priced with, one of the files in
            ``auto_parallel/op_profiles``.  A producer that knows the family
            declares it; otherwise it is inferred from ``name`` when the spec
            is resolved.
        ops: Per-layer-kind op counts that replace the profile's own, for a
            model no profile describes.  Its kinds must be the profile's.
        layers: The layer stack: groups of consecutive layers of one kind of
            the profile, body first and MTP layers last.  A producer derives
            it from ``layer_types``, ``first_k_dense_replace`` and
            ``mtp_depth`` when the config does not state it.
        layer_types: The attention flavour of each layer, as a hybrid config
            such as Qwen3.5 lists them; a producer states ``layers`` from
            them and drops them.
        tie_word_embeddings: Whether the output head shares the embedding's
            table, as a Transformers config states it.
        qk_norm: Whether attention normalizes each head's queries and keys
            over the head width before the scores, as Qwen3's ``q_norm`` and
            ``k_norm`` do.  A Transformers config does not state it; its
            producer infers it from the model's name.
        qkv_bias, o_bias, mlp_bias: Whether the query, key and value
            projections, the output projection and the feed-forward's hold a
            bias.  Unstated, the parameter formulas count their own.
        norm_bias, layer_norms: Whether each norm holds a bias beside its
            weight, as a LayerNorm does, and how many norms a layer holds,
            its attention's per-head query and key norms apart.
        shared_expert_gate: Whether a MoE layer gates its shared expert's
            output with a weight of its own, as Qwen2-MoE's does.
        torch_dtype: The dtype the checkpoint's weights are saved in, by name,
            which the runtime loads them in under its default ``auto``; how a
            run holds and computes them is its execution spec's.
        activations: What a layer of each kind keeps for its backward and
            holds in it, per token, as a census measures them; the memory
            model prices a layer of a kind stated here with them rather
            than its formulas.
        output_activations: The same record for the output layer, the final
            norm, the output projection and the loss; its TP part is the
            part a vocabulary-parallel loss splits.
    """

    name: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    vocab_size: int

    num_key_value_heads: Optional[int] = None
    head_dim: Optional[int] = None
    intermediate_size: Optional[int] = None
    max_position_embeddings: Optional[int] = None

    num_experts: Optional[int] = None
    num_experts_per_tok: Optional[int] = None
    num_shared_experts: Optional[int] = None
    moe_intermediate_size: Optional[int] = None
    shared_expert_intermediate_size: Optional[int] = None
    first_k_dense_replace: Optional[int] = None

    mtp_depth: Optional[int] = None

    multiple_of: Optional[int] = None
    ffn_dim_multiplier: Optional[float] = None
    kv_lora_rank: Optional[int] = None
    q_lora_rank: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None
    # Qwen3.5 fuses an output gate into the query projection, doubling its width.
    attn_output_gate: Optional[bool] = None
    tie_word_embeddings: Optional[bool] = None
    qk_norm: Optional[bool] = None
    qkv_bias: Optional[bool] = None
    o_bias: Optional[bool] = None
    mlp_bias: Optional[bool] = None
    norm_bias: Optional[bool] = None
    layer_norms: Optional[int] = None
    shared_expert_gate: Optional[bool] = None
    torch_dtype: Optional[str] = None

    # Gated DeltaNet linear attention (Qwen3.5), for the layers a linear kind prices.
    linear_num_key_heads: Optional[int] = None
    linear_key_head_dim: Optional[int] = None
    linear_num_value_heads: Optional[int] = None
    linear_value_head_dim: Optional[int] = None
    linear_conv_kernel_dim: Optional[int] = None

    arch: Optional[str] = None
    ops: Optional[Dict[str, OpCounts]] = None
    layers: Optional[Tuple[LayerGroup, ...]] = None
    layer_types: Optional[Tuple[str, ...]] = None
    activations: Optional[Dict[str, KindActivations]] = None
    output_activations: Optional[KindActivations] = None

    vision: Optional[VisionSpec] = None

    _REQUIRED = ("hidden_size", "num_hidden_layers", "num_attention_heads", "vocab_size")

    # ---- derived reads -------------------------------------------------

    @property
    def effective_head_dim(self) -> int:
        """Per-head width, explicit when declared and derived when not.

        Costing attention as ``hidden_size x hidden_size`` is only correct
        when ``head_dim == hidden_size / num_attention_heads``.  Qwen3.5
        declares ``head_dim: 256`` against ``h=2048, a=16``, so the two
        differ by 2x and half the projection parameters go missing.
        """
        if self.head_dim:
            return int(self.head_dim)
        return self.hidden_size // self.num_attention_heads

    @property
    def is_moe(self) -> bool:
        """Whether the model routes tokens to more than one expert."""
        return bool(self.num_experts and self.num_experts > 1)

    @property
    def routed_expert_width(self) -> int:
        """Feed-forward width of one routed expert, ``0`` when none resolves.

        Mixtral-style configs run their experts at the dense feed-forward
        width and declare no ``moe_intermediate_size``, so the dense width is
        a real fallback rather than a guess.
        """
        width = self.moe_intermediate_size or self.intermediate_size
        return int(width or 0)

    @property
    def shared_expert_width(self) -> int:
        """Feed-forward width of one shared expert, ``0`` when there is none.

        A config may declare the shared expert's own width or inherit the
        routed one.  An all-MoE config declares no dense ``intermediate_size``
        at all, which is the case that used to price this at zero.
        """
        if not self.num_shared_experts:
            return 0
        width = self.shared_expert_intermediate_size or self.routed_expert_width
        return int(width or 0)

    # ---- validation ----------------------------------------------------

    def validate(self) -> "ModelSpec":
        """Raise :class:`ModelSpecError` unless the spec is internally coherent.

        Returns *self*, so a producer can ``return spec.validate()``.
        """
        for name in self._REQUIRED:
            value = getattr(self, name)
            if not value or value <= 0:
                raise ModelSpecError(
                    f"{name} is required and must be positive, got {value!r}. "
                    "Supply it in the model config or in model.config_overrides."
                )

        kv_heads = self.num_key_value_heads
        if kv_heads and self.num_attention_heads % kv_heads:
            raise ModelSpecError(
                f"num_attention_heads ({self.num_attention_heads}) must be a multiple "
                f"of num_key_value_heads ({kv_heads})"
            )

        if self.is_moe and not self.routed_expert_width:
            raise ModelSpecError(
                f"no feed-forward width resolves for {self.num_experts} routed "
                "experts; declare moe_intermediate_size, or intermediate_size if "
                "the experts run at the dense width"
            )
        if self.num_shared_experts and not self.shared_expert_width:
            raise ModelSpecError(
                "num_shared_experts is set but no width resolves for it; declare "
                "shared_expert_intermediate_size or moe_intermediate_size"
            )
        if self.layers is not None:
            check_layer_counts(self.layers, self.num_hidden_layers, self.mtp_depth or 0)
            self._check_activations()
        if self.vision is not None:
            self.vision.validate()
        return self

    def _check_activations(self) -> None:
        """Raise unless every kind the activations state is a kind of the stack."""
        stated = set(self.activations or {}) - {group.kind for group in self.layers or ()}
        if stated:
            raise ModelSpecError(f"activations of kinds {sorted(stated)} no layer of the stack is")

    # ---- serialised form -----------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return the spec as a plain mapping, ready for YAML.

        Unset optional fields are omitted rather than written as ``null``, so
        a hand-written file and a dumped one look alike and a reader cannot
        mistake "not declared" for "declared empty".
        """
        out: Dict[str, Any] = {}
        for spec_field in fields(self):
            if spec_field.name in ("vision", "ops", "layers", "layer_types", "activations", "output_activations"):
                continue
            value = getattr(self, spec_field.name)
            if value is not None:
                out[spec_field.name] = value
        if self.ops is not None:
            out["ops"] = {kind: counts.to_dict() for kind, counts in self.ops.items()}
        if self.layers is not None:
            out["layers"] = [group.to_dict() for group in self.layers]
        if self.layer_types is not None:
            out["layer_types"] = list(self.layer_types)
        if self.activations is not None:
            out["activations"] = {kind: record.to_dict() for kind, record in self.activations.items()}
        if self.output_activations is not None:
            out["output_activations"] = self.output_activations.to_dict()
        if self.vision is not None:
            out["vision"] = self.vision.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], strict: bool = True) -> "ModelSpec":
        """Build a spec from a mapping, refusing a key it does not know.

        Args:
            data: A parsed YAML mapping, or whatever a producer assembled.
                :func:`model_fields` takes the model's keys from a mapping
                that also states the run.
            strict: Validate before returning.  Only a test constructing a
                deliberately partial spec should pass ``False``.

        Raises:
            ModelSpecError: If a key is unknown, a required field is missing,
                a field is not an integer, or the declared fields contradict
                each other.
        """
        unknown = sorted(set(data) - MODEL_KEYS)
        if unknown:
            precision = [key for key in unknown if key in _RUN_PRECISION_KEYS]
            raise ModelSpecError(
                f"unknown model spec keys {unknown}: a model spec states the model only, "
                "and a run's strategy, such as its offset or recompute, belongs in its "
                "execution spec" + (
                    f"; {precision} state the run's precision, which a HyperParallel run takes from "
                    "fsdp_config.mix_precision.param_dtype, else model_init_dtype, else model.torch_dtype"
                    if precision else ""
                )
            )
        kwargs: Dict[str, Any] = {}
        for key, value in data.items():
            if key != "vision":
                kwargs[key] = cls._field_value(key, value)

        for name in cls._REQUIRED:
            if name not in kwargs:
                raise ModelSpecError(
                    f"{name} is required and was not supplied. "
                    "Supply it in the model config or in model.config_overrides."
                )
        kwargs.setdefault("name", "custom")

        raw_vision = data.get("vision")
        if isinstance(raw_vision, Mapping):
            kwargs["vision"] = cls._vision_from_dict(raw_vision)

        spec = cls(**kwargs)
        return spec.validate() if strict else spec

    @staticmethod
    def _field_value(key: str, value: Any) -> Any:
        """Coerce one declared field to the type the schema gives it."""
        if key == "name":
            return str(value)
        if value is None:
            return None
        if key == "arch":
            return str(value)
        if key == "torch_dtype":
            return str(value).replace("torch.", "")
        if key == "ops":
            return ops_from_dict(value)
        if key == "layers":
            return layers_from_list(value)
        if key == "activations":
            return activations_from_dict(value)
        if key == "output_activations":
            return KindActivations.from_dict(value, key)
        if key == "ffn_dim_multiplier":
            return float(value)
        if key in _FLAGS:
            if not isinstance(value, bool):
                raise ModelSpecError(f"{key} must be true or false, got {value!r}")
            return value
        if key == "layer_types":
            if not isinstance(value, (list, tuple)) or not all(isinstance(kind, str) for kind in value):
                raise ModelSpecError(f"layer_types must list layer type names, got {value!r}")
            return tuple(value)
        return _as_int(value, key)

    @staticmethod
    def _vision_from_dict(data: Mapping[str, Any]) -> VisionSpec:
        """Build a :class:`VisionSpec`, dropping keys it does not declare."""
        known = {f.name for f in fields(VisionSpec)}
        kwargs: Dict[str, Any] = {}
        for key, value in data.items():
            if key not in known:
                continue
            if key == "layers":
                kwargs[key] = None if value is None else layers_from_list(value, "vision.layers")
            else:
                kwargs[key] = str(value) if key == "name" else _as_int(value, f"vision.{key}")
        missing = [
            name for name in ("hidden_size", "num_hidden_layers", "num_attention_heads")
            if name not in kwargs
        ]
        if missing:
            raise ModelSpecError(
                f"vision tower is missing required fields: {', '.join(missing)}"
            )
        return VisionSpec(**kwargs)


# The keys a model spec reads.
MODEL_KEYS = frozenset(spec_field.name for spec_field in fields(ModelSpec))


def model_fields(data: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the entries of *data* a model spec reads.

    For a mapping that states the model and the run together, such as a
    search config's model section.
    """
    return {key: value for key, value in data.items() if key in MODEL_KEYS}

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
"""The execution IR: how a model is trained, beside what the model is.

:class:`~hyper_parallel.auto_parallel._model_spec.ModelSpec` states the facts
about an architecture that hold whatever strategy it runs under.
:class:`ExecSpec` states the rest: what a search varies (the parallel
degrees, the micro-batching, the sharding and the recompute policy) and the
fixed facts of the run (the sequence length, the precisions, the kernels and
the device's memory).  The cost model derives everything else from the two.

It follows the model spec's two rules.  **Absent is not zero**: every field
defaults to ``None``, which means "not stated", so an ExecSpec can state a
whole run or only what a search candidate changes, and a consumer never
mistakes an unstated degree for a degree of zero.  **Refuse rather than
default**: :meth:`ExecSpec.validate` rejects a stated value that cannot be
right, and :meth:`ExecSpec.from_dict` refuses a key it does not know, by name.
"""
from dataclasses import dataclass, fields
from typing import Any, Dict, Mapping, Optional, Tuple, Union


class ExecSpecError(ValueError):
    """An execution spec states something it cannot, or something unknown."""


# The rules by which selective recompute picks the ops it recomputes.
SELECTIVE_RULES = ("hyperparallel", "mindformers")

# How a range of layers recomputes.
RECOMPUTE_OPTIONS = ("none", "full", "selective")
# The ops a selective range decides for, the cost model's recompute switches,
# and the state it gives each: the op's output is kept for backward, or
# recomputed.  Offload will add a third state.  attUp is an MLA layer's
# up-projections, which build its query, key and value heads from their
# latents.
RECOMPUTE_OPS = ("attBMM", "headCast", "dropout", "softmax", "normOp", "gather", "ffAct", "attUp")
OP_STATES = ("keep", "recompute")


def _whole(value: Any, least: int) -> bool:
    """Whether *value* is a whole number of at least *least*."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= least


@dataclass(frozen=True)
class RecomputeRange:
    """How consecutive layers recompute, every layer of the range alike.

    Layers are counted in model order from 0, the body layers first and then
    the MTP layers, whatever stage each is placed on.

    Attributes:
        first: The range's first layer.
        count: How many layers it covers, or ``None`` for every layer from
            *first* on, so that ``RecomputeRange(option="full")`` recomputes
            the whole model.
        option: One of :data:`RECOMPUTE_OPTIONS`.
        ops: For a selective range, the state of each op it names, one of
            :data:`OP_STATES`; an op it does not name is kept.  ``None``
            takes the selective recompute of the run's rule
            (``selective_rule``).
    """

    first: int = 0
    count: Optional[int] = None
    option: str = "none"
    ops: Optional[Mapping[str, str]] = None

    def validate(self) -> "RecomputeRange":
        """Raise :class:`ExecSpecError` unless the range can be right; return it."""
        if not _whole(self.first, 0):
            raise ExecSpecError(f"a recompute range's first layer must be a whole number from 0, got {self.first!r}")
        if self.count is not None and not _whole(self.count, 1):
            raise ExecSpecError(f"a recompute range's count must be at least 1, got {self.count!r}")
        if self.option not in RECOMPUTE_OPTIONS:
            raise ExecSpecError(f"a recompute option must be one of {list(RECOMPUTE_OPTIONS)}, got {self.option!r}")
        if self.ops is not None:
            if self.option != "selective":
                raise ExecSpecError(f"only a selective range states ops, not a {self.option!r} one")
            unknown = sorted(set(self.ops) - set(RECOMPUTE_OPS))
            if unknown:
                raise ExecSpecError(f"unknown recompute ops {unknown}; the ops are {list(RECOMPUTE_OPS)}")
            states = sorted({str(state) for state in self.ops.values()} - set(OP_STATES))
            if states:
                raise ExecSpecError(f"an op's state must be one of {list(OP_STATES)}, got {states}")
        return self

    def switches(self) -> Optional[Dict[str, int]]:
        """Return every op's switch, 1 to keep and 0 to recompute, or ``None`` for the rule's."""
        if self.ops is None:
            return None
        return {op: int(self.ops.get(op, "keep") == "keep") for op in RECOMPUTE_OPS}

    def to_dict(self) -> Dict[str, Any]:
        """Return the range as a plain mapping, ready for YAML."""
        out: Dict[str, Any] = {"first": self.first}
        if self.count is not None:
            out["count"] = self.count
        out["option"] = self.option
        if self.ops is not None:
            out["ops"] = dict(self.ops)
        return out

    @classmethod
    def from_dict(cls, data: Any) -> "RecomputeRange":
        """Build a validated range from a mapping, refusing an unknown key."""
        if not isinstance(data, Mapping):
            raise ExecSpecError(f"a recompute range must be a mapping, got {data!r}")
        unknown = sorted(set(data) - {"first", "count", "option", "ops"})
        if unknown:
            raise ExecSpecError(f"unknown recompute range keys {unknown}; a range has first, count, option and ops")
        ops = data.get("ops")
        if ops is not None and not isinstance(ops, Mapping):
            raise ExecSpecError(f"a recompute range's ops must map ops to states, got {ops!r}")
        return cls(
            first=data.get("first", 0),
            count=data.get("count"),
            option=str(data.get("option", "none")),
            ops=None if ops is None else {str(op): str(state) for op, state in ops.items()},
        ).validate()


def _check_ranges(ranges: Any) -> None:
    """Raise :class:`ExecSpecError` unless *ranges* run in model order without overlap."""
    if not isinstance(ranges, tuple):
        raise ExecSpecError(f"recompute must be a tuple of recompute ranges, got {ranges!r}")
    end = 0
    for index, item in enumerate(ranges):
        if not isinstance(item, RecomputeRange):
            raise ExecSpecError(f"recompute must hold recompute ranges, got {item!r}")
        item.validate()
        if item.first < end:
            raise ExecSpecError(
                f"recompute ranges must run in model order without overlap: layer {item.first} "
                f"is covered already"
            )
        if item.count is None and index != len(ranges) - 1:
            raise ExecSpecError("only the last recompute range may run to the last layer")
        end = item.first + (item.count or 0)


@dataclass(frozen=True)
class ExecSpec:
    """How a model is trained: the strategy a search varies, and the run.

    Every field is optional; ``None`` means the spec does not state it.  A
    byte width, ``grad_accumulation`` or ``shard_activations`` that no spec
    states takes the default of the model's family, from its op profile.

    Attributes:
        dp, tp, pp, vpp, cp, ep: The data, tensor, pipeline, virtual
            pipeline, context and expert parallel degrees.
        etp: The expert tensor-parallel degree; at most 1 means an expert
            layer runs with the dense layers' tensor parallelism.
        sequence_parallel: Whether activations are split along the sequence
            over the tensor-parallel group.
        shard_activations: Whether tensor parallelism shards the activations
            between layers, which a fully recomputed layer keeps as its
            input, and the output layer's.
        loss_parallel: Whether the loss runs on logits sharded over the
            vocabulary across the tensor-parallel group, each rank's share,
            rather than on logits every rank gathers whole, as
            HyperParallel's trainer does unless its ``loss_parallel``.
        micro_batch_size, micro_batch_num, global_batch_size: The batch.
        optimizer_parallel: Whether optimizer states are sharded.
        optimizer_shard: How many data-parallel ranks optimizer sharding
            splits each parameter over, on top of TP's split, as MindSpore's
            ``optimizer_weight_shard_size`` and HyperParallel's ``dp_shard``
            count them.
        expert_shard: How many ranks of its expert data-parallel group FSDP
            shards a routed expert over under expert parallelism, as
            HyperParallel's ``edp_shard_size`` counts them; without expert
            parallelism the experts shard with the other parameters.  Stated
            by no one, the optimizer shards them over the whole group.
        expert_shard_group: Whether a launcher shards each strategy's routed
            experts over its whole expert data-parallel group, whatever
            ``expert_shard`` states, as ``context.expert_shard: group`` says.
        grad_shard: Whether gradients are sharded too.
        grad_shard_as_params: Whether each gradient is sharded as its
            parameter is, as FSDP holds it; ``grad_shard`` does not apply.
        shard_spans_cp: Whether the optimizer's shard spans context
            parallelism's ranks too, as HyperParallel's ``dp_shard_size``
            does over its ``dp * cp`` FSDP domain, so that CP shards no
            parameter state beyond it and the tables it gathers are whole;
            unstated, CP shards a parameter's state again (C1).
        grad_accumulation: Whether gradients take memory without pipeline
            parallelism too; under it they always do.
        deferred_grad_accumulation: Whether FSDP holds each layer's
            reduce-scatter output until the micro-batch's backward ends, and
            only then adds it to the accumulated gradient, as HyperParallel's
            does; PyTorch's FSDP2 adds it as soon as it is reduced.
        overlapped_grad_reduce: Whether FSDP reduce-scatters a layer's
            gradients while the next layer's backward runs, holding them
            whole until that one ends, and the root's, the embedding and
            output tables', until the backward ends, as HyperParallel's does.
        reshard_params: Whether FSDP frees a layer's gathered parameters
            once the layer has run, forward or backward, and gathers them
            again when it runs next, as HyperParallel's does by default;
            the root's, the embedding and output tables, stay gathered.
        param_casts: Whether a layer keeps, beside each matmul's input, the
            cast the formulas price where the optimizer does not shard
            (``n_attParamCast``, ``n_ffParamCast``); unstated, it does
            exactly there.  HyperParallel's FSDP gathers its weights in the
            compute dtype, and its layers keep none at any ``dp_shard``.
        pp_schedule: The pipeline schedule, such as ``1f1b``.
        offset: How many layers each pipeline stage holds beyond an even
            split, per stage or per chunk and stage.
        seq_split: How many parts a sequence is split into for a sequence
            pipeline.
        pp_partition: The layers each stage holds, when stated outright.
        mtp_in_offset, emb_out_in_offset: Whether the MTP layers, and the
            embedding and output layers, count in the offset's layers.
        full_recompute, selective_recompute: Which layers are recomputed in
            full or selectively: true for all, or counts per stage.
        recompute: How each layer recomputes, as :class:`RecomputeRange`
            ranges over the layers in model order.  Stated, it replaces the
            per-stage forms above; a layer no range covers is not
            recomputed.
        selective_comm_recompute: Which layers recompute their
            sequence-parallel all-gather, under the MindFormers rule.
        selective_rule: Whose selective recompute the run uses, one of
            :data:`SELECTIVE_RULES`.
        recompute_slice_activation: Whether a recomputed layer keeps its
            input sliced over tensor parallelism.
        param_bytes, compute_bytes, softmax_bytes, grad_bytes,
            optimizer_state_bytes, norm_bytes, dropout_bytes: Bytes per
            element of the parameters, activations, softmax outputs,
            gradients, optimizer states, norm activations and dropout masks.
        optimizer_states: How many states the optimizer keeps per parameter
            of a layer: AdamW's two moments, or Muon's one momentum.  The
            embedding and output tables keep AdamW's two.
        main_param_bytes: Bytes per element of the copy of the parameters
            the optimizer keeps, 0 when it keeps none, as HyperParallel's
            fp32 main parameters keep one.
        flash_attention, grad_clip, grouped_gemm, tie_embeddings: Kernels
            and features the run uses.
        vocab_emb_dp: Whether the vocabulary embedding runs data parallel.
        emb_dp_sharded: Whether the embedding table is sharded over data
            parallelism.
        cp_algo: The context-parallel algorithm.
        capacity_factor: The experts' token capacity factor.
        shard_mtp_param: Whether the MTP layers' parameters are sharded.
        frozen: Whether the model's parameters are frozen.
        optimizer: The optimizer's name.
        seq_length: The training sequence length.
        device_memory: The device's memory, such as ``"64GB"``.
    """

    dp: Optional[int] = None
    tp: Optional[int] = None
    pp: Optional[int] = None
    vpp: Optional[int] = None
    cp: Optional[int] = None
    ep: Optional[int] = None
    etp: Optional[int] = None
    sequence_parallel: Optional[bool] = None
    shard_activations: Optional[bool] = None
    loss_parallel: Optional[bool] = None

    micro_batch_size: Optional[int] = None
    micro_batch_num: Optional[int] = None
    global_batch_size: Optional[int] = None

    optimizer_parallel: Optional[bool] = None
    optimizer_shard: Optional[int] = None
    expert_shard: Optional[int] = None
    expert_shard_group: Optional[bool] = None
    grad_shard: Optional[bool] = None
    grad_shard_as_params: Optional[bool] = None
    shard_spans_cp: Optional[bool] = None
    grad_accumulation: Optional[bool] = None
    deferred_grad_accumulation: Optional[bool] = None
    overlapped_grad_reduce: Optional[bool] = None
    reshard_params: Optional[bool] = None
    param_casts: Optional[bool] = None

    pp_schedule: Optional[str] = None
    offset: Optional[Union[int, list]] = None
    seq_split: Optional[int] = None
    pp_partition: Optional[list] = None
    mtp_in_offset: Optional[bool] = None
    emb_out_in_offset: Optional[bool] = None

    full_recompute: Optional[Union[bool, int, list]] = None
    selective_recompute: Optional[Union[bool, list]] = None
    selective_comm_recompute: Optional[Union[bool, list]] = None
    selective_rule: Optional[str] = None
    recompute_slice_activation: Optional[bool] = None
    recompute: Optional[Tuple[RecomputeRange, ...]] = None

    param_bytes: Optional[int] = None
    compute_bytes: Optional[int] = None
    softmax_bytes: Optional[int] = None
    grad_bytes: Optional[int] = None
    optimizer_state_bytes: Optional[int] = None
    optimizer_states: Optional[int] = None
    main_param_bytes: Optional[int] = None
    norm_bytes: Optional[int] = None
    dropout_bytes: Optional[int] = None

    flash_attention: Optional[bool] = None
    grad_clip: Optional[bool] = None
    grouped_gemm: Optional[bool] = None
    vocab_emb_dp: Optional[bool] = None
    emb_dp_sharded: Optional[bool] = None
    tie_embeddings: Optional[bool] = None
    cp_algo: Optional[str] = None
    capacity_factor: Optional[float] = None
    shard_mtp_param: Optional[bool] = None
    frozen: Optional[bool] = None
    optimizer: Optional[str] = None

    seq_length: Optional[int] = None
    device_memory: Optional[Any] = None

    # ---- validation ----------------------------------------------------

    def validate(self) -> "ExecSpec":
        """Raise :class:`ExecSpecError` unless every stated value can be right.

        Returns *self*, so a producer can ``return spec.validate()``.
        """
        for name in _KINDS["count"]:
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ExecSpecError(f"{name} must be at least 1, got {value!r}")
        for name in _KINDS["size"]:
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ExecSpecError(f"{name} must not be negative, got {value!r}")
        if self.selective_rule is not None and self.selective_rule not in SELECTIVE_RULES:
            raise ExecSpecError(
                f"selective_rule must be one of {list(SELECTIVE_RULES)}, got {self.selective_rule!r}"
            )
        if self.capacity_factor is not None and self.capacity_factor <= 0:
            raise ExecSpecError(f"capacity_factor must be positive, got {self.capacity_factor!r}")
        if self.recompute is not None:
            _check_ranges(self.recompute)
        return self

    # ---- serialised form -----------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return the stated fields as a plain mapping, ready for YAML."""
        out: Dict[str, Any] = {}
        for spec_field in fields(self):
            value = getattr(self, spec_field.name)
            if value is None:
                continue
            if spec_field.name == "recompute":
                value = [item.to_dict() for item in value]
            out[spec_field.name] = value if spec_field.name != "device_memory" else str(value)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecSpec":
        """Build a validated spec from a mapping, refusing an unknown key.

        Raises:
            ExecSpecError: If a key is unknown, a value has the wrong type,
                or a stated value cannot be right.
        """
        if not isinstance(data, Mapping):
            raise ExecSpecError(f"an execution spec must be a mapping, got {data!r}")
        known = [spec_field.name for spec_field in fields(cls)]
        unknown = sorted(set(data) - set(known))
        if unknown:
            raise ExecSpecError(f"unknown execution spec keys {unknown}; the keys are {known}")
        kwargs = {key: _coerce(key, value) for key, value in data.items()}
        return cls(**kwargs).validate()


# The fields by kind, for coercion from YAML and for validation; a field in
# none of them is kept as it is.
_KINDS: Dict[str, tuple] = {
    "count": ("dp", "tp", "pp", "vpp", "cp", "ep", "micro_batch_size", "micro_batch_num",
              "global_batch_size", "optimizer_shard", "expert_shard", "seq_split", "seq_length",
              "optimizer_states"),
    "size": ("etp", "param_bytes", "compute_bytes", "softmax_bytes", "grad_bytes",
             "optimizer_state_bytes", "main_param_bytes", "norm_bytes", "dropout_bytes"),
    "flag": ("sequence_parallel", "shard_activations", "loss_parallel", "optimizer_parallel", "grad_shard",
             "grad_shard_as_params", "shard_spans_cp", "grad_accumulation", "deferred_grad_accumulation",
             "overlapped_grad_reduce", "reshard_params", "param_casts", "expert_shard_group",
             "mtp_in_offset",
             "emb_out_in_offset",
             "recompute_slice_activation", "flash_attention", "grad_clip", "grouped_gemm",
             "vocab_emb_dp", "emb_dp_sharded", "tie_embeddings", "shard_mtp_param", "frozen"),
    "name": ("pp_schedule", "selective_rule", "cp_algo", "optimizer", "device_memory"),
}


def _as_number(value: Any) -> float:
    """Coerce the capacity factor to a number."""
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ExecSpecError(f"capacity_factor must be a number, got {value!r}") from exc


def _as_ranges(value: Any) -> Tuple[RecomputeRange, ...]:
    """Coerce a list of recompute ranges, each a range or its mapping."""
    if not isinstance(value, (list, tuple)):
        raise ExecSpecError(f"recompute must list recompute ranges, got {value!r}")
    return tuple(item if isinstance(item, RecomputeRange) else RecomputeRange.from_dict(item) for item in value)


# The fields with a coercion of their own.
_COERCIONS = {"capacity_factor": _as_number, "recompute": _as_ranges}


def _coerce(key: str, value: Any) -> Any:
    """Coerce one field read from YAML to the type the schema gives it."""
    if value is None:
        return None
    if key in _KINDS["count"] or key in _KINDS["size"]:
        if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
            raise ExecSpecError(f"{key} must be a whole number, got {value!r}")
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ExecSpecError(f"{key} must be a whole number, got {value!r}") from exc
    if key in _KINDS["flag"]:
        if not isinstance(value, bool):
            raise ExecSpecError(f"{key} must be true or false, got {value!r}")
        return value
    if key in _KINDS["name"]:
        return str(value)
    if key in _COERCIONS:
        return _COERCIONS[key](value)
    return value

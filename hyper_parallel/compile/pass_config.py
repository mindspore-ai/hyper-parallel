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
"""
Parallel Configuration - Graph-mode parallel configuration.

Pure configuration dataclass: no environment probes. Whether FSDP actually
runs is decided by ``fsdp_enabled`` here (the user's intent) AND the runtime
distributed guard inside ``FSDPPass`` (which still early-returns on
``world_size == 1``). The previous ``fsdp_enabled`` property returned
``dist.is_initialized() and world_size > 1`` — that turned on FSDP whenever
distributed was initialized, leaving no way to run pure-TP / pure-PP graph
mode. The explicit field fixes that. Pipeline-parallel (``pp_enabled``)
follows the same contract: intent here, runtime guard in ``PpPass``.

Note:
    TP / SP / LP are deliberately NOT configurable here: their collectives
    are baked into module boundary forwards by the automodel flow (shared
    with eager mode), and no graph pass consumes them. ``PassConfig`` only
    drives what the graph passes actually execute.
"""

__all__ = [
    "PassConfig",
    "GradientReduction",
    "PP_SCHEDULES",
    "INDUCTOR_MODES",
    "DP_MODES",
    "AC_MODES",
    "AC_REGION_SOURCES",
    "normalize_dp_mode",
    "build_pass_config_from_trainer_config",
]

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

_LOG = logging.getLogger(__name__)


# Pipeline schedule names accepted by ``pp_schedule``. Kept torch-free here so
# the config dataclass stays importable anywhere; ``pp_schedule``'s registry
# (which does import torch) must expose exactly these keys.
PP_SCHEDULES = ("gpipe", "1f1b", "interleaved_1f1b", "vpp")

# Backend compilation modes for the terminal ``InductorPass``:
#   "off"      - no Inductor compilation (graph runs as plain FX nodes)
#   "regional" - compile tagged compute regions; collectives and
#                side-effectful nodes stay eager (see InductorPass)
#   "full"     - compile the whole joint graph as one Inductor region
INDUCTOR_MODES = ("off", "regional", "full")

# Schedule names that can drive multiple virtual stages per physical rank.
_INTERLEAVED_SCHEDULES = ("interleaved_1f1b", "vpp")

# Data-parallel modes, mirroring simplefsdp's ``data_parallel`` modes:
#   "fsdp" -> fully_shard        (Shard(0) on the fsdp axis)
#   "ddp"  -> replicate          (Replicate; grads all-reduced)
#   "hsdp" -> hybrid_shard       (Replicate x Shard(0); rs + all-reduce)
DP_MODES = ("fsdp", "ddp", "hsdp")

# simplefsdp / fully_shard spellings accepted as aliases.
_DP_MODE_ALIASES = {
    "fully_shard": "fsdp",
    "replicate": "ddp",
    "hybrid_shard": "hsdp",
}

# Static-graph activation checkpointing modes for ``ActivationCheckpointPass``,
# mirroring the dynamic ``ActivationCheckpointConfig.mode`` vocabulary:
#   "off"       - save every forward activation (no recompute pass)
#   "full"      - recompute every checkpoint-scoped module op in backward
#   "selective" - save compute/communication intensive ops, recompute the rest
AC_MODES = ("off", "full", "selective")

# Where the recompute-region shape comes from, mirroring
# ``ActivationCheckpointSelection.source``. ``default`` selects whole blocks
# (and lets ``full`` auto-upgrade to a model-owned RecomputePolicy);
# ``model_adapter_safe_regions`` requires an adapter-declared region.
AC_REGION_SOURCES = ("default", "model_adapter_safe_regions")


def normalize_dp_mode(mode: str) -> str:
    """Return the canonical DP mode for ``mode`` (accepts simplefsdp aliases).

    Args:
        mode: Requested DP mode spelling.

    Returns:
        The canonical ``DP_MODES`` entry.

    Raises:
        ValueError: When ``mode`` is not a string or is unknown.
    """
    if not isinstance(mode, str):
        raise ValueError(f"dp_mode must be a string, got {type(mode).__name__}")
    key = mode.strip().lower()
    return _DP_MODE_ALIASES.get(key, key)


@dataclass(frozen=True)
class GradientReduction:
    """Reduction operation and a divisor independent of the communication mesh.

    The divisor is applied before communication, matching eager FSDP's
    reduction-precision scaling and its singleton-group behavior.
    """

    reduce_op: Literal["avg", "sum"] = "avg"
    divisor: int = 1

    def __post_init__(self) -> None:
        if self.reduce_op not in ("avg", "sum"):
            raise ValueError("Gradient reduction must be 'avg' or 'sum'")
        if isinstance(self.divisor, bool) or not isinstance(self.divisor, int) or self.divisor < 1:
            raise ValueError("Gradient divisor must be a positive integer")

    @property
    def scale(self) -> float | None:
        """Return the pre-reduction multiplier, or None for an identity scale."""
        return 1.0 / self.divisor if self.divisor > 1 else None



@dataclass
class PassConfig:
    """Parallel configuration for graph-mode FSDP (+ optional TP / SP / PP) training.

    Attributes:
        enable_overlap: Drive ``AutoOverlapPass`` to move ``wait_tensor`` for
            communication/compute overlap.
        fsdp_bucketing: Fuse compatible FSDP all-gathers and reduce-scatters before
            overlap. Python-only, disabled by default; configure before compilation.
        fsdp_bucket_modules: Optional module FQN groups, produced by
            ``build_module_bucket_plans`` or supplied as a list of paths/groups.
            None groups by immediate parameter parents; an empty list selects nothing.
            All ranks in each FSDP group must use the same configuration.
        fsdp_allgather_prefetch: Prefetch forward/backward FSDP all-gathers
            and delay their wait/view nodes independently of enable_overlap.
        fsdp_overlap_grad_reduce: Advance backward reduce-scatter launches and
            delay their waits independently of parameter prefetch and enable_overlap.
        max_in_flight_gb: GiB of early FSDP all-gather replicas before first use,
            shared across anchors in each phase. Uses PyTorch's overlap scheduler
            naming and binary units. Zero disables early launches, not wait sinking.
        fsdp_enabled: Drive ``FSDPPass`` (data-parallel partitioning: parameter
            all_gather + gradient reduction + live-model sharding, selected by
            ``dp_mode``). ``False`` skips data parallelism entirely — set this
            for pure-TP / pure-PP graph-mode runs. ``FSDPPass`` itself still
            early-returns when distributed is not initialized or
            ``world_size == 1``, so single-card runs are a no-op regardless.
        fsdp_reshard_after_forward: Drive the FSDP reshard optimization.
            When ``True`` (default, matching FSDP's ``reshard_after_forward``)
            ``FSDPPass`` sinks each param's all_gather to its first forward
            use, frees the replicated full parameter once its last forward
            reader is done, and — when the backward still needs it — inserts a
            fresh all_gather plus a rematerialization of the saved forward
            views just before the first backward consumer. Peak memory then
            tracks the forward working set instead of the sum of all
            replicated parameters. Set ``False`` to keep every gathered
            parameter resident for the whole joint graph (previous behavior).
        fsdp_degree: Size of the FSDP group. ``None`` (default) means
            "resolve at runtime": the trainer back-fills it from the
            automodel ``MeshContext`` (TP+FSDP hybrid, where the FSDP group
            is a proper sub-group of the world), and ``FSDPPass`` falls back
            to ``world_size`` for the FSDP-only path. Mutating this after
            construction is supported but discouraged — prefer passing the
            resolved degree at construction time (see ``GraphTrainer``).
            For ``dp_mode="hsdp"`` this is the *shard* axis degree.
        dp_mode: Data-parallel mode, one of ``DP_MODES``: ``"fsdp"``
            (fully_shard, the default), ``"ddp"`` (replicate; no parameter
            sharding, gradients all-reduced), or ``"hsdp"`` (hybrid_shard;
            parameters replicated on the ``dp_replicate`` axis and sharded on
            the ``fsdp`` axis, gradients reduce-scattered then all-reduced).
            simplefsdp spellings ``"fully_shard"`` / ``"replicate"`` /
            ``"hybrid_shard"`` are accepted as aliases and normalized.
        dp_replicate_degree: Size of the replicate axis for ``"ddp"`` /
            ``"hsdp"``. ``None`` means "resolve at runtime" (from the mesh,
            or ``world_size // fsdp_degree`` for HSDP / ``world_size`` for
            DDP). Kept separate from ``fsdp_degree`` (the shard axis) so an
            HSDP mesh is unambiguous.
        pp_enabled: Drive ``PpPass`` (pipeline-parallel graph split). When
            ``True`` the joint fwd+bwd graph is sliced to this rank's stage
            along module-FQN boundaries declared in ``GraphParallelPlan``, cross-stage
            activations are exchanged via P2P ``isend``/``irecv``, and a
            self-contained pipeline schedule is installed as a ``call_module``
            stub inside the rewritten graph, so the trainer needs no PP
            wiring. ``PpPass`` early-returns when distributed is not
            initialized or ``world_size == 1``.
        pp_degree: Number of pipeline-parallel ranks (physical stages).
            ``None`` means "resolve at runtime": pure-PP resolves it from
            ``world_size``; the PP+DP hybrid resolves it from the 2-D
            ``(pp, DP)`` mesh built by ``GraphCompiler._init_pp_dp_mesh``
            (which also back-fills ``fsdp_degree`` /
            ``dp_replicate_degree``), subject to
            ``pp_degree * dp_degree == world_size``.
        pp_chunks: Number of virtual pipeline stages per physical rank.
            ``1`` (default) is the classic single-stage-per-rank layout used
            by ``"gpipe"`` / ``"1f1b"``. Values ``> 1`` enable VPP
            (interleaved 1F1B): the model is split into
            ``pp_degree * pp_chunks`` virtual stages, rank ``r`` owns virtual
            stages ``r, r + pp_degree, ...``, and the ``"interleaved_1f1b"``
            schedule pipelines the virtual stages. Requires
            ``pp_schedule`` to be an interleaved schedule.
        pp_microbatch_size: Number of samples per microbatch for the
            pipeline schedule. The per-step batch size (the leading dim of
            the trainer's input tensor) must be divisible by this. ``1`` means
            one sample per microbatch (maximum pipeline concurrency, most
            P2P traffic); larger values trade bubble size for fewer P2P
            round-trips.
        pp_schedule: Pipeline schedule name, one of ``PP_SCHEDULES``.
            ``"gpipe"`` (default) runs all forwards before all backwards
            (simplest ordering, largest bubble). ``"1f1b"`` interleaves one
            backward per forward in steady state, shrinking the pipeline
            bubble and freeing each microbatch's activations as soon as its
            backward runs (GPipe retains all of them until the sweep ends);
            it requires ``num_microbatches >= pp_degree``.
            ``"interleaved_1f1b"`` (alias ``"vpp"``) is the VPP schedule: it
            supports ``pp_chunks > 1`` virtual stages per rank and uses the
            classic interleaved-1F1B warmup / steady / cooldown order.
            ``PpPass`` resolves the name to a schedule class via
            ``pp_schedule.get_schedule_class``.
        inductor_mode: Terminal backend compilation for the joint graph, one
            of ``INDUCTOR_MODES``. ``"off"`` (default) runs the graph as
            plain FX nodes; ``"regional"`` compiles tagged compute regions
            with TorchInductor while keeping communication collectives and
            side-effectful nodes (e.g. the FSDP reshard storage free) eager;
            ``"full"`` compiles the whole joint graph as one region. Both
            compiling modes force ``eager_numerics.division_rounding`` on
            torch builds that provide it (2.11+) so compiled code stays
            bitwise-equal to eager, and restore the positional calling
            convention afterwards. Requires a torch with Inductor. With
            pipeline parallel enabled, ``PpPass`` runs first and this pass
            compiles each stage's forward/backward subgraph
            independently (both modes); cross-stage fusion is inherently
            impossible because the schedule's P2P sits between the stages.
        inductor_configs: Optional Inductor config overrides passed to the
            compiled regions (keys validated by ``torch._inductor.config``),
            e.g. ``{"reorder_for_peak_memory": True}``. ``"full"`` mode
            always sets ``reorder_for_peak_memory=True`` unless overridden.
        ac_mode: Static-graph activation checkpointing mode for
            ``ActivationCheckpointPass``, one of ``AC_MODES``. Mirrors the
            dynamic ``ActivationCheckpointConfig.mode`` vocabulary and is
            projected from a trainer config's ``activation_checkpoint``
            section by ``build_pass_config_from_trainer_config``. ``"off"``
            (default) saves every forward activation; ``"full"`` recomputes
            every checkpoint-scoped module op in backward; ``"selective"``
            saves the shared compute/communication operator set and
            recomputes the rest.
        ac_layer_count: Optional layer selector: checkpoint only the first
            ``ac_layer_count`` discovered transformer blocks (the same
            HuggingFace-marker discovery the eager path uses). Mutually
            exclusive with ``ac_layer_indices``.
        ac_layer_indices: Optional explicit zero-based indices into the
            discovered transformer blocks; only nodes inside the selected
            blocks recompute. Mutually exclusive with ``ac_layer_count``.
        ac_region_source: Where the recompute-region shape comes from, one of
            ``AC_REGION_SOURCES``. ``"default"`` selects whole blocks (and
            lets ``full`` auto-upgrade to a model-owned ``RecomputePolicy``);
            ``"model_adapter_safe_regions"`` requires an adapter-declared
            region and its ``exclude_patterns``. Projected from the trainer
            selection's ``source``.
        ac_save_forward_all_gather: Save forward (TP/SP/CP) all-gathers instead
            of replaying them during backward recompute. Graph-mode only;
            defaults to ``True``, so static-graph recompute does not duplicate
            a forward collective. FSDP parameter all-gathers are inserted after
            this pass and are unaffected. Set ``False`` to replay the
            all-gather, trading communication for less retained memory.

    Note:
        ``fsdp_enabled`` / ``pp_enabled`` no longer probe
        ``torch.distributed``. The distributed-initialized check moved into
        the respective passes' ``run`` (their original location) so this
        dataclass stays torch-free and importable anywhere.
    """

    enable_overlap: bool = True
    fsdp_enabled: bool = True
    fsdp_reshard_after_forward: bool = True
    fsdp_degree: Optional[int] = None
    dp_mode: str = "fsdp"
    dp_replicate_degree: Optional[int] = None
    pp_enabled: bool = False
    pp_degree: Optional[int] = None
    pp_chunks: int = 1
    pp_microbatch_size: int = 1
    pp_schedule: str = "gpipe"
    fsdp_bucketing: bool = False
    fsdp_bucket_modules: Optional[
        list[str | list[str]] | tuple[tuple[str, ...], ...]
    ] = None
    fsdp_allgather_prefetch: bool = True
    fsdp_overlap_grad_reduce: bool = True
    max_in_flight_gb: float = 0.25
    inductor_mode: str = "off"
    inductor_configs: Optional[Dict[str, Any]] = None
    ac_mode: str = "off"
    ac_layer_count: Optional[int] = None
    ac_layer_indices: Optional[List[int]] = None
    ac_region_source: str = "default"
    ac_save_forward_all_gather: bool = True

    def __post_init__(self) -> None:
        self.dp_mode = normalize_dp_mode(self.dp_mode)
        self.validate()

    def validate(self) -> None:
        """Sanity-check invariants; also re-run after manual field mutation.

        Raises:
            ValueError: On a non-positive explicit ``fsdp_degree`` /
                ``pp_degree``, a non-positive ``pp_microbatch_size``, an
                unknown ``pp_schedule`` / ``dp_mode``, or a non-positive
                ``dp_replicate_degree``, or ``pp_chunks > 1`` with a
                non-interleaved schedule, or an invalid activation
                checkpointing mode / layer selector.
        """
        self._validate_flags()
        self._validate_dp()
        self._validate_pp()
        self._validate_ac()

    def _validate_ac(self) -> None:
        """Validate the activation checkpointing mode and layer selectors."""
        if not isinstance(self.ac_mode, str) or self.ac_mode not in AC_MODES:
            raise ValueError(f"ac_mode must be one of {AC_MODES}, got {self.ac_mode!r}")
        if (
            not isinstance(self.ac_region_source, str)
            or self.ac_region_source not in AC_REGION_SOURCES
        ):
            raise ValueError(
                f"ac_region_source must be one of {AC_REGION_SOURCES}, "
                f"got {self.ac_region_source!r}"
            )
        if not isinstance(self.ac_save_forward_all_gather, bool):
            raise ValueError("ac_save_forward_all_gather must be a bool")
        self._validate_ac_layer_count()
        self._validate_ac_layer_indices()
        if self.ac_layer_count is not None and self.ac_layer_indices is not None:
            raise ValueError(
                "ac_layer_count and ac_layer_indices are mutually exclusive; "
                "set exactly one"
            )
        if (
            self.ac_mode != "off"
            and self.ac_region_source == "model_adapter_safe_regions"
            and (self.ac_layer_count is None and self.ac_layer_indices is None)
        ):
            raise ValueError(
                "ac_region_source=model_adapter_safe_regions requires exactly "
                "one of ac_layer_count or ac_layer_indices"
            )

    def _validate_ac_layer_count(self) -> None:
        """Validate the optional prefix layer selector."""
        if self.ac_layer_count is not None and (
            isinstance(self.ac_layer_count, bool)
            or not isinstance(self.ac_layer_count, int)
            or self.ac_layer_count < 0
        ):
            raise ValueError(
                "ac_layer_count must be None or a non-negative integer, "
                f"got {self.ac_layer_count!r}"
            )

    def _validate_ac_layer_indices(self) -> None:
        """Validate the optional explicit layer index selector."""
        indices = self.ac_layer_indices
        if indices is None:
            return
        invalid = (
            not isinstance(indices, list)
            or not indices
            or any(
                isinstance(index, bool) or not isinstance(index, int) or index < 0
                for index in indices
            )
        )
        if invalid:
            raise ValueError(
                "ac_layer_indices must be None or a non-empty list of "
                f"non-negative integers, got {indices!r}"
            )
        if len(set(indices)) != len(indices):
            raise ValueError(
                f"ac_layer_indices must not contain duplicates, got {indices!r}"
            )

    def _validate_flags(self) -> None:
        """Validate boolean flags and the overlap memory budget."""
        for name in (
            "fsdp_bucketing",
            "fsdp_allgather_prefetch",
            "fsdp_overlap_grad_reduce",
        ):
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a bool")
        if self.fsdp_bucket_modules is not None and not isinstance(
            self.fsdp_bucket_modules, (list, tuple)
        ):
            raise ValueError(
                "fsdp_bucket_modules must be None or a list/tuple of module groups"
            )
        value = self.max_in_flight_gb
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError("max_in_flight_gb must be a finite non-negative number")
        self._validate_inductor()

    def _validate_inductor(self) -> None:
        """Validate the Inductor compilation mode and config overrides."""
        if not isinstance(self.inductor_mode, str) or (
            self.inductor_mode.lower() not in INDUCTOR_MODES
        ):
            raise ValueError(
                f"inductor_mode must be one of {INDUCTOR_MODES}, "
                f"got {self.inductor_mode!r}"
            )
        # Canonicalize to one spelling so the pass's mode dispatch cannot be
        # defeated by an accepted-but-unnormalized value like "FULL".
        self.inductor_mode = self.inductor_mode.lower()
        if self.inductor_configs is not None and (
            not isinstance(self.inductor_configs, dict)
            or not all(isinstance(key, str) for key in self.inductor_configs)
        ):
            raise ValueError(
                "inductor_configs must be None or a dict of "
                "torch._inductor.config option names (str) to values"
            )

    def _validate_dp(self) -> None:
        """Validate the tensor-parallel and data-parallel degrees."""
        if self.fsdp_degree is not None and self.fsdp_degree < 1:
            raise ValueError(
                f"fsdp_degree must be None or a positive int, got {self.fsdp_degree}"
            )
        if self.dp_mode not in DP_MODES:
            raise ValueError(
                f"dp_mode must be one of {DP_MODES} "
                f"(aliases: {tuple(_DP_MODE_ALIASES)}), got {self.dp_mode!r}"
            )
        if self.dp_replicate_degree is not None and self.dp_replicate_degree < 1:
            raise ValueError(
                f"dp_replicate_degree must be None or a positive int, "
                f"got {self.dp_replicate_degree}"
            )

    def _validate_pp(self) -> None:
        """Validate the pipeline-parallel degree, microbatch size, and schedule."""
        if self.pp_degree is not None and self.pp_degree < 1:
            raise ValueError(
                f"pp_degree must be None or a positive int, got {self.pp_degree}"
            )
        if self.pp_chunks < 1:
            raise ValueError(f"pp_chunks must be >= 1, got {self.pp_chunks}")
        if self.pp_microbatch_size < 1:
            raise ValueError(
                f"pp_microbatch_size must be >= 1, got {self.pp_microbatch_size}"
            )
        if not isinstance(self.pp_schedule, str) or (
            self.pp_schedule.lower() not in PP_SCHEDULES
        ):
            raise ValueError(
                f"pp_schedule must be one of {PP_SCHEDULES}, got {self.pp_schedule!r}"
            )
        if (
            self.pp_chunks > 1
            and self.pp_schedule.lower() not in _INTERLEAVED_SCHEDULES
        ):
            raise ValueError(
                f"pp_chunks={self.pp_chunks} requires an interleaved schedule "
                f"(one of {_INTERLEAVED_SCHEDULES}), got {self.pp_schedule!r}"
            )


def build_pass_config_from_trainer_config(
    config: Any,
    *,
    fsdp_enabled: Optional[bool] = None,
    enable_overlap: bool = False,
) -> PassConfig:
    """Project trainer topology intent onto graph-mode pass config.

    This helper stays in the pass-config module because it computes the
    graph-pass surface from the trainer topology without altering runtime
    objects. Accepting ``Any`` keeps this module importable without the
    trainer package. Only fields the graph passes actually consume are
    projected: TP / SP / LP topology (``accelerator.tp_size`` /
    ``sequence_parallel`` / ``loss_parallel``) is owned by the automodel
    flow and deliberately not copied.

    The dynamic ``activation_checkpoint`` section is projected onto the
    static-graph AC fields: ``mode`` becomes ``ac_mode`` and a
    ``model_adapter_safe_regions`` selection contributes its layer
    selectors. ``swap_inputs`` and ``allow_cache_entry_mutation`` have no
    static-graph counterpart (no eager checkpoint cache) and are ignored
    with a warning when set, mirroring the dynamic compile path.

    Args:
        config: Trainer topology (e.g. a ``TrainerConfig``); only the
            ``fsdp_config``, ``compile`` and ``activation_checkpoint``
            surfaces are read.
        fsdp_enabled: Overrides the sharded-DP intent derived from
            ``config.fsdp_config`` when provided.
        enable_overlap: Value for the legacy ``AutoOverlapPass`` switch.

    Returns:
        The graph-mode ``PassConfig`` for this trainer topology.
    """
    compile_config = getattr(config, "compile", None)
    if fsdp_enabled is None:
        fsdp_enabled = (
            config.fsdp_config.dp_shard_size > 1
            or config.fsdp_config.edp_shard_size > 1
        )
    fsdp_degree = config.fsdp_config.dp_shard_size if fsdp_enabled else None

    ac_mode = "off"
    ac_layer_count = None
    ac_layer_indices = None
    ac_region_source = "default"
    ac_config = getattr(config, "activation_checkpoint", None)
    if ac_config is not None and getattr(ac_config, "mode", None) not in (None, "off"):
        ac_mode = ac_config.mode
        selection = getattr(ac_config, "selection", None)
        ac_region_source = getattr(selection, "source", "default")
        if ac_region_source == "model_adapter_safe_regions":
            ac_layer_count = getattr(selection, "layer_count", None)
            ac_layer_indices = getattr(selection, "layer_indices", None)
        if getattr(ac_config, "swap_inputs", False):
            _LOG.warning(
                "activation_checkpoint.swap_inputs is not supported in graph "
                "mode; input swapping will be disabled."
            )
        if getattr(ac_config, "allow_cache_entry_mutation", False):
            _LOG.info(
                "activation_checkpoint.allow_cache_entry_mutation has no "
                "effect in graph mode; ignoring it."
            )

    return PassConfig(
        enable_overlap=enable_overlap,
        fsdp_enabled=fsdp_enabled,
        fsdp_degree=fsdp_degree,
        fsdp_reshard_after_forward=getattr(
            config.fsdp_config, "reshard_after_forward", True
        ),
        fsdp_allgather_prefetch=getattr(
            compile_config, "fsdp_allgather_prefetch", True
        ),
        fsdp_overlap_grad_reduce=getattr(
            compile_config, "fsdp_overlap_grad_reduce", True
        ),
        max_in_flight_gb=getattr(compile_config, "max_in_flight_gb", 0.25),
        ac_mode=ac_mode,
        ac_layer_count=ac_layer_count,
        ac_layer_indices=ac_layer_indices,
        ac_region_source=ac_region_source,
    )

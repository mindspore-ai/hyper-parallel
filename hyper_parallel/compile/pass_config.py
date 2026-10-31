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
    "PP_SCHEDULES",
    "INDUCTOR_MODES",
    "DP_MODES",
    "normalize_dp_mode",
    "build_pass_config_from_trainer_config",
]

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional


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


def normalize_dp_mode(mode: str) -> str:
    """Return the canonical DP mode for ``mode`` (accepts simplefsdp aliases).

    Raises:
        ValueError: When ``mode`` is not a string or is unknown.
    """
    if not isinstance(mode, str):
        raise ValueError(f"dp_mode must be a string, got {type(mode).__name__}")
    key = mode.strip().lower()
    return _DP_MODE_ALIASES.get(key, key)


@dataclass
class PassConfig:
    """Parallel configuration for graph-mode FSDP (+ optional TP / SP / PP) training.

    Attributes:
        enable_overlap: Drive ``AutoOverlapPass`` to move ``wait_tensor`` for
            communication/compute overlap.
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
            compiling modes force ``eager_numerics.division_rounding`` so
            compiled code stays bitwise-equal to eager, and restore the
            positional calling convention afterwards. Requires a torch with
            Inductor. With pipeline parallel enabled, ``PpPass`` runs first
            and this pass compiles each stage's forward/backward subgraph
            independently (both modes); cross-stage fusion is inherently
            impossible because the schedule's P2P sits between the stages.
        inductor_configs: Optional Inductor config overrides passed to the
            compiled regions (keys validated by ``torch._inductor.config``),
            e.g. ``{"reorder_for_peak_memory": True}``. ``"full"`` mode
            always sets ``reorder_for_peak_memory=True`` unless overridden.

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
    fsdp_allgather_prefetch: bool = True
    fsdp_overlap_grad_reduce: bool = True
    max_in_flight_gb: float = 0.25
    inductor_mode: str = "off"
    inductor_configs: Optional[Dict[str, Any]] = None

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
                non-interleaved schedule.
        """
        self._validate_flags()
        self._validate_dp()
        self._validate_pp()

    def _validate_flags(self) -> None:
        """Validate boolean flags and the overlap memory budget."""
        for name in ("fsdp_allgather_prefetch", "fsdp_overlap_grad_reduce"):
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a bool")
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
    """
    compile_config = getattr(config, "compile", None)
    if fsdp_enabled is None:
        fsdp_enabled = (
            config.fsdp_config.dp_shard_size > 1
            or config.fsdp_config.edp_shard_size > 1
        )
    fsdp_degree = config.fsdp_config.dp_shard_size if fsdp_enabled else None
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
    )

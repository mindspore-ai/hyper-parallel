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
"""

__all__ = ["PassConfig", "PP_SCHEDULES", "DP_MODES", "normalize_dp_mode"]

from dataclasses import dataclass
from typing import Any, Optional


# Pipeline schedule names accepted by ``pp_schedule``. Kept torch-free here so
# the config dataclass stays importable anywhere; ``pp_schedule``'s registry
# (which does import torch) must expose exactly these keys.
PP_SCHEDULES = ("gpipe", "1f1b")

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
        tp_size: Tensor-parallel degree. Informational today (TP collectives
            live inside boundary forwards baked by automodel, not in the
            graph-mode passes); kept so a future TP-aware pass can read it
            without API churn.
        sequence_parallel: Enable sequence parallel (SP) on the TP axis.
        loss_parallel: Enable loss parallel (LP) on the TP axis.
        pp_enabled: Drive ``PpPass`` (pipeline-parallel graph split). When
            ``True`` the joint fwd+bwd graph is sliced to this rank's stage
            along module-FQN boundaries declared in ``GraphParallelPlan``, cross-stage
            activations are exchanged via P2P ``isend``/``irecv``, and a
            self-contained GPipe schedule is installed as a ``call_module``
            stub inside the rewritten graph, so the trainer needs no PP
            wiring. ``PpPass`` early-returns when distributed is not
            initialized or ``world_size == 1``.
        pp_degree: Number of pipeline-parallel ranks (== number of stages
            for the single-virtual-stage schedule shipped here). ``None``
            means "resolve at runtime" from ``world_size`` (the pure-PP
            path); for a PP+FSDP hybrid the FSDP group is a proper
            sub-group of the world and the trainer back-fills this from
            the mesh, exactly like ``fsdp_degree``.
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
            ``PpPass`` resolves the name to a schedule class via
            ``pp_schedule.get_schedule_class``.

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
    tp_size: int = 1
    sequence_parallel: bool = False
    loss_parallel: bool = False
    pp_enabled: bool = False
    pp_degree: Optional[int] = None
    pp_microbatch_size: int = 1
    pp_schedule: str = "gpipe"

    def __post_init__(self) -> None:
        self.dp_mode = normalize_dp_mode(self.dp_mode)
        self.validate()

    def validate(self) -> None:
        """Sanity-check invariants; also re-run after manual field mutation.

        Raises:
            ValueError: On a negative ``tp_size``, a non-positive explicit
                ``fsdp_degree`` / ``pp_degree``, a non-positive
                ``pp_microbatch_size``, an unknown ``pp_schedule`` /
                ``dp_mode``, or a non-positive ``dp_replicate_degree``.
        """
        if self.tp_size < 1:
            raise ValueError(f"tp_size must be >= 1, got {self.tp_size}")
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
        if self.pp_degree is not None and self.pp_degree < 1:
            raise ValueError(
                f"pp_degree must be None or a positive int, got {self.pp_degree}"
            )
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
    trainer package.
    """
    accelerator = config.accelerator
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
        tp_size=accelerator.tp_size,
        sequence_parallel=accelerator.sequence_parallel,
        loss_parallel=accelerator.loss_parallel,
    )


__all__ = ["PassConfig", "build_pass_config_from_trainer_config"]

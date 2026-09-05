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
mode. The explicit field fixes that.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PassConfig:
    """Parallel configuration for graph-mode FSDP, EP and optional TP/SP training.

    Attributes:
        enable_overlap: Drive ``AutoOverlapPass`` to move ``wait_tensor`` for
            communication/compute overlap.
        fsdp_enabled: Drive ``FSDPPass`` (parameter all_gather + gradient
            reduce_scatter + live-model sharding). ``False`` skips FSDP
            entirely — set this for pure-TP / pure-PP graph-mode runs.
            ``FSDPPass`` itself still early-returns when distributed is not
            initialized or ``world_size == 1``, so single-card runs are a
            no-op regardless.
        fsdp_degree: Size of the FSDP group. ``None`` (default) means
            "resolve at runtime": the trainer back-fills it from the
            automodel ``MeshContext`` (TP+FSDP hybrid, where the FSDP group
            is a proper sub-group of the world), and ``FSDPPass`` falls back
            to ``world_size`` for the FSDP-only path. Mutating this after
            construction is supported but discouraged — prefer passing the
            resolved degree at construction time (see ``GraphTrainer``).
        tp_size: Tensor-parallel degree. Informational today (TP collectives
            live inside boundary forwards baked by automodel, not in the
            graph-mode passes); kept so a future TP-aware pass can read it
            without API churn.
        sequence_parallel: Enable sequence parallel (SP) on the TP axis.
        loss_parallel: Enable loss parallel (LP) on the TP axis.
        ep_degree: Expert-parallel degree. Values greater than one capture the
            already-applied AutoModels dynamic EP region into the graph.
        require_ep_collectives: Whether EP graph validation requires captured
            all-to-all evidence.

    Note:
        This dataclass does not probe ``torch.distributed``. Runtime process
        group validation remains in the corresponding graph passes.
    """

    enable_overlap: bool = True
    fsdp_enabled: bool = True
    fsdp_degree: Optional[int] = None
    tp_size: int = 1
    sequence_parallel: bool = False
    loss_parallel: bool = False
    ep_degree: int = 1
    require_ep_collectives: bool = True

    def __post_init__(self) -> None:
        self.validate()

    @property
    def ep_enabled(self) -> bool:
        """Whether dynamic EP should be captured into the graph."""
        return self.ep_degree > 1

    def validate(self) -> None:
        """Sanity-check invariants; also re-run after manual field mutation.

        Raises:
            ValueError: On a non-positive or non-integer parallel degree.
            NotImplementedError: When EP is combined with TP.
        """
        degrees = {"tp_size": self.tp_size, "ep_degree": self.ep_degree}
        if self.fsdp_degree is not None:
            degrees["fsdp_degree"] = self.fsdp_degree
        for name, degree in degrees.items():
            if isinstance(degree, bool) or not isinstance(degree, int) or degree < 1:
                raise ValueError(f"{name} must be a positive integer")

        if self.ep_enabled and self.tp_size != 1:
            raise NotImplementedError("Static EP currently requires tp_size to be 1")


__all__ = ["PassConfig"]

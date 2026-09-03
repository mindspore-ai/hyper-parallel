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
    """Parallel configuration for graph-mode FSDP (+ optional TP) training.

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
            resolved degree at construction time (see ``GraphTextTrainer``).
        tp_size: Tensor-parallel degree. Informational today (TP collectives
            live inside boundary forwards baked by automodel, not in the
            graph-mode passes); kept so a future TP-aware pass can read it
            without API churn.

    Note:
        ``fsdp_enabled`` no longer probes ``torch.distributed``. The
        distributed-initialized check moved into ``FSDPPass.run`` (its
        original location) so this dataclass stays torch-free and importable
        anywhere.
    """

    enable_overlap: bool = True
    fsdp_enabled: bool = True
    fsdp_degree: Optional[int] = None
    tp_size: int = 1

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Sanity-check invariants; also re-run after manual field mutation.

        Raises:
            ValueError: On a negative ``tp_size`` or a non-positive
                explicit ``fsdp_degree``.
        """
        if self.tp_size < 1:
            raise ValueError(f"tp_size must be >= 1, got {self.tp_size}")
        if self.fsdp_degree is not None and self.fsdp_degree < 1:
            raise ValueError(
                f"fsdp_degree must be None or a positive int, got {self.fsdp_degree}"
            )


__all__ = ["PassConfig"]

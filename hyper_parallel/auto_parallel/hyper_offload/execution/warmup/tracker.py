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
"""Activation identity and lifecycle management.

:class:`ActivationTracker` is the single source of truth for:
* Storage identity (``storage_id`` resolution).
* Activation policy (which tensors should be shadowed/tracked).
* Activation storage registration (activations produced inside the trace).

It holds **no mutable runtime state** beyond the set of trace-created
storage IDs.  Callers that need physical state transitions should use
:class:`~offload.runtime.residency.ResidencyManager` instead.
"""

from __future__ import annotations

import logging
import weakref

import torch

from hyper_parallel.auto_parallel.hyper_offload.execution.tensor import ShadowTensor

logger = logging.getLogger(__name__)


class ActivationTracker:
    """Identity resolution + activation lifecycle management.

    Owns
    ----
    * ``_storage_tracker`` (weak dictionary) for stable ``storage_id``.
    * ``_storage_sizes`` — mapping from storage ID to size in bytes,
      accumulated across all recorded ops.
    * ``_activation_sids`` — storage IDs produced inside the current trace.

    High-level API (:meth:`get_activation_sid`, :meth:`register_op_activations`)
    should be preferred over the low-level private helpers.
    """

    def __init__(self) -> None:
        self._storage_tracker: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._next_storage_id = 1
        self._storage_sizes: dict[int, int] = {}
        self._activation_sids: set[int] = set()

    # ------------------------------------------------------------------
    # Private low-level identity API
    # ------------------------------------------------------------------

    def _ensure_id(self, tensor: torch.Tensor) -> int | None:
        """Get-or-create the unique storage ID for *tensor* (private).

        If the storage has been seen before the existing ID is returned;
        otherwise a fresh ID is assigned and recorded.

        Returns ``None`` for tensors without a stable storage identity
        (e.g. meta / quantized tensors that do not support
        :meth:`torch.Tensor.untyped_storage`).
        """
        try:
            s = tensor.untyped_storage()
        except (AttributeError, RuntimeError):
            return None

        try:
            sid = self._storage_tracker[s]
        except KeyError:
            sid = self._next_storage_id
            self._storage_tracker[s] = sid
            self._next_storage_id += 1

        self._storage_sizes.setdefault(sid, s.size())
        logger.debug("_ensure_id: sid=%d shape=%s", sid, tensor.shape)
        return sid

    # ------------------------------------------------------------------
    # Unified SID resolution
    # ------------------------------------------------------------------

    def get_activation_sid(self, tensor: torch.Tensor) -> int | None:
        """Return the tracked storage ID for *tensor*, or ``None``.

        Handles both :class:`ShadowTensor` (which carries its SID inline)
        and raw tensors via a read-only storage lookup.

        This is the unified replacement for ``WarmupExecutor._sid_of``.

        Notes on the raw-tensor eligibility heuristics:
        * CPU tensors are never activations — the trace runs on CUDA.
        * Tensors without ``untyped_storage`` (e.g. meta device) cannot
          have a stable storage identity.
        """
        if isinstance(tensor, ShadowTensor):
            return tensor.storage_id
        if tensor.device.type == "cpu":
            return None
        if not hasattr(tensor, "untyped_storage"):
            return None
        try:
            s = tensor.untyped_storage()
        except (AttributeError, RuntimeError):
            return None
        sid = self._storage_tracker.get(s)
        return sid if sid is not None and sid in self._activation_sids else None

    # ------------------------------------------------------------------
    # Step lifecycle
    # ------------------------------------------------------------------

    def register_op_activations(
        self,
        input_tensors: list[torch.Tensor],
        output_tensors: list[torch.Tensor],
    ) -> None:
        """Register new activations produced by an op.

        Compares the storage IDs of *output_tensors* against those of
        *input_tensors* and marks any previously unseen output storage
        as a trace-created activation.

        """
        # Collect storage IDs of all op inputs.
        input_sids = {sid for t in input_tensors if (sid := self._ensure_id(t)) is not None}

        # Register newly-created storages that appear for the first time
        # as op outputs, collecting their sizes.
        for t in output_tensors:
            sid = self._ensure_id(t)
            if sid is not None and sid not in input_sids:
                self._activation_sids.add(sid)

    # ------------------------------------------------------------------
    # Lifecycle reset
    # ------------------------------------------------------------------

    @property
    def storage_sizes(self) -> dict[int, int]:
        """Return a copy of the accumulated storage size map."""
        return dict(self._storage_sizes)

    def clear_activations(self) -> None:
        """Clear the activation set (called at the start of warmup)."""
        self._activation_sids.clear()
        self._storage_sizes.clear()

    def __repr__(self) -> str:
        """Return the string representation."""
        return f"{type(self).__name__}(activations={len(self._activation_sids)})"

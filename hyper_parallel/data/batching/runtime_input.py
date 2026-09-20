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
"""Resolve sequence metadata and model-owned runtime inputs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch


class OnlineBoundaryResolver:
    """Read global cumulative sequence boundaries emitted by Online packing."""

    @staticmethod
    def resolve(canonical_batch: Mapping[str, Any]) -> Any:
        """Read leading-zero ``cu_seq_lens`` from an Online batch."""
        raw_cu_seq_lens = canonical_batch["cu_seq_lens"]
        cu_seq_lens = raw_cu_seq_lens.to(torch.int32)
        return cu_seq_lens


class IndexedBoundaryResolver:
    """Recover global cumulative sequence boundaries from Indexed input IDs."""

    def __init__(self, eod_token_id: int | None) -> None:
        """Store the token separating packed Indexed sequences."""
        self.eod_token_id = eod_token_id

    def resolve(self, canonical_batch: Mapping[str, Any]) -> Any:
        """Recover leading-zero ``cu_seq_lens`` from Indexed tokens."""
        input_ids = canonical_batch["input_ids"]
        batch_size, seq_len = input_ids.shape
        token_indices = torch.arange(seq_len, dtype=input_ids.dtype, device=input_ids.device)

        seq_ends = []
        for batch_idx in range(batch_size):
            if self.eod_token_id is None:
                eod_indices = token_indices[:0]
            else:
                eod_indices = token_indices[input_ids[batch_idx] == self.eod_token_id]

            prev_eod_idx = -1
            for eod_idx in eod_indices:
                eod_idx = int(eod_idx.item())
                if eod_idx == prev_eod_idx:
                    break

                seq_end = batch_idx * seq_len + eod_idx + 1
                seq_ends.append(seq_end)
                prev_eod_idx = eod_idx + 1

            row_end = (batch_idx + 1) * seq_len
            if not seq_ends or seq_ends[-1] != row_end:
                seq_ends.append(row_end)

        cu_seq_lens = torch.tensor([0, *seq_ends], dtype=torch.int32)
        return cu_seq_lens


@dataclass(frozen=True)
class RuntimeInputContext:
    """Framework execution facts available to a model runtime adapter.

    ``options`` deliberately carries feature-specific policy. The framework
    owns batch movement and parallel coordinates, while a model adapter may
    translate those facts into attention metadata, modality routing inputs,
    cache descriptors, or another forward-only contract.
    """

    local_input_shape: Sequence[int]
    parallel_ranks: Mapping[str, int]
    parallel_sizes: Mapping[str, int]
    options: Mapping[str, Any]

    @classmethod
    def from_batch(
            cls,
            *,
            batch: Mapping[str, Any],
            parallel_context: Any,
            options: Mapping[str, Any] | None = None,
    ) -> RuntimeInputContext:
        """Build runtime context from one parallel-local batch."""
        runtime_context = cls(
            local_input_shape=batch["input_ids"].shape,
            parallel_ranks={
                "tp": parallel_context.tp_rank,
                "cp": parallel_context.cp_rank,
            },
            parallel_sizes={
                "tp": parallel_context.tp_world_size,
                "cp": parallel_context.cp_world_size,
            },
            options={} if options is None else options,
        )
        return runtime_context


class RuntimeInputAdapter(ABC):
    """Extend a generic batch with model-owned forward inputs."""

    def build(
            self,
            *,
            batch: Mapping[str, Any],
            parallel_context: Any,
            options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build runtime context and produce one runtime's model inputs."""
        runtime_context = RuntimeInputContext.from_batch(
            batch=batch,
            parallel_context=parallel_context,
            options=options,
        )
        runtime_inputs = self.build_runtime_inputs(
            batch=batch,
            context=runtime_context,
        )
        if not isinstance(runtime_inputs, Mapping):
            raise TypeError(
                "RuntimeInputAdapter.build_runtime_inputs must return a mapping, "
                f"got {type(runtime_inputs).__name__}"
            )
        return dict(runtime_inputs)

    @abstractmethod
    def build_runtime_inputs(
            self,
            *,
            batch: Mapping[str, Any],
            context: RuntimeInputContext,
    ) -> Mapping[str, Any]:
        """Build additional model inputs without mutating ``batch``.

        Args:
            batch: Device-resident, parallel-local batch. Global metadata such
                as ``cu_seq_lens`` remains available when required.
            context: Local geometry, parallel coordinates, and feature-specific
                recipe options.

        Returns:
            Mapping merged into the model forward inputs. Keys must not replace
            fields already owned by the generic batch path.
        """
        raise NotImplementedError


class AttentionRuntime(RuntimeInputAdapter):
    """Build dense attention fields from common runtime context."""

    def __init__(
            self,
            *,
            mode: str,
            create_mask: bool,
            reset_mask: bool,
            sliding_window: int | None,
    ) -> None:
        """Store the attention representation and mask policy."""
        self.mode = mode
        self.create_mask = create_mask
        self.reset_mask = reset_mask
        self.sliding_window = sliding_window

    def build_runtime_inputs(
            self,
            *,
            batch: Mapping[str, Any],
            context: RuntimeInputContext,
    ) -> Mapping[str, Any]:
        """Build attention fields without owning model-specific packed metadata."""
        attention_inputs = {"attention_mask": None}
        if not self.create_mask or self.mode == "compressed":
            return attention_inputs

        micro_batch_size, local_sequence_length = batch["input_ids"].shape
        cp_size = context.parallel_sizes["cp"]
        global_sequence_length = local_sequence_length * cp_size
        attention_mask, sliding_window_mask = self._build_dense_attention_masks(
            cu_seq_lens=batch["cu_seq_lens"],
            micro_batch_size=micro_batch_size,
            seq_length=global_sequence_length,
            device=batch["input_ids"].device,
        )
        attention_inputs["attention_mask"] = attention_mask
        if sliding_window_mask is not None:
            attention_inputs["swa_mask"] = sliding_window_mask

        return attention_inputs

    def _build_dense_attention_masks(
            self,
            *,
            cu_seq_lens: Any,
            micro_batch_size: int,
            seq_length: int,
            device: Any,
    ) -> tuple[Any, Any | None]:
        """Build global dense attention and optional sliding-window masks."""
        boundaries = [int(boundary) for boundary in cu_seq_lens.tolist()]
        if len(boundaries) < 2 or boundaries[0] != 0:
            raise ValueError("cu_seq_lens must contain a leading zero and at least one sequence")

        if any(end <= start for start, end in zip(boundaries[:-1], boundaries[1:])):
            raise ValueError("cu_seq_lens must be strictly increasing")

        physical_sequence_length = micro_batch_size * seq_length
        if boundaries[-1] != physical_sequence_length:
            raise ValueError(
                "cu_seq_lens must cover the physical batch length "
                f"({physical_sequence_length}), but got {boundaries[-1]}"
            )

        attention_batch_size = micro_batch_size if self.reset_mask else 1
        attention_mask = torch.ones(
            (attention_batch_size, seq_length, seq_length),
            dtype=torch.bool,
            device=device,
        ).tril()
        attention_mask = attention_mask.view(attention_batch_size, 1, seq_length, seq_length)

        if self.reset_mask:
            for sequence_start in boundaries[:-1]:
                batch_index, row_sequence_start = divmod(sequence_start, seq_length)
                attention_mask[batch_index, 0, row_sequence_start:, :row_sequence_start] = False

        sliding_window_mask = None
        if self.sliding_window is not None:
            positions = torch.arange(seq_length, dtype=torch.int64, device=device)
            outside_window = positions.unsqueeze(1) - positions.unsqueeze(0) > self.sliding_window
            sliding_window_mask = attention_mask & ~outside_window.unsqueeze(0).unsqueeze(0)

        return attention_mask, sliding_window_mask


__all__ = [
    "AttentionRuntime",
    "IndexedBoundaryResolver",
    "OnlineBoundaryResolver",
    "RuntimeInputAdapter",
    "RuntimeInputContext",
]

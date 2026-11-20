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
from numbers import Integral
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
    """Resolve global cumulative sequence boundaries for Indexed batches."""

    def __init__(self, eod_token_id: int | None) -> None:
        """Store the token separating packed Indexed sequences."""
        self.eod_token_id = eod_token_id

    def resolve(self, canonical_batch: Mapping[str, Any]) -> torch.Tensor:
        """Read explicit boundaries or recover them from EOD tokens."""
        input_ids = canonical_batch["input_ids"]
        batch_size, seq_len = input_ids.shape

        if "cu_seq_lens" in canonical_batch:
            cu_seq_lens = self._merge_cu_seqlens_across_micro_batch(
                canonical_batch["cu_seq_lens"], batch_size, seq_len
            )
            return cu_seq_lens

        token_indices = torch.arange(seq_len, dtype=input_ids.dtype, device=input_ids.device)
        seq_ends = []

        for batch_idx in range(batch_size):
            if self.eod_token_id is None:
                eod_indices = token_indices[:0]
            else:
                eod_indices = token_indices[input_ids[batch_idx] == self.eod_token_id]

            for eod_idx in eod_indices:
                seq_end = batch_idx * seq_len + int(eod_idx.item()) + 1
                seq_ends.append(seq_end)

            row_end = (batch_idx + 1) * seq_len
            if not seq_ends or seq_ends[-1] != row_end:
                seq_ends.append(row_end)

        cu_seq_lens = torch.tensor([0, *seq_ends], dtype=torch.int32)

        return cu_seq_lens

    @staticmethod
    def _merge_cu_seqlens_across_micro_batch(
            cu_seqlens: Sequence[Any] | torch.Tensor, micro_batch_size: int, seq_length: int
    ) -> torch.Tensor:
        """Merge per-sample boundaries into offsets over the entire batch.

        Adapted from Megatron-LM's megatron/core/utils.py. See MEGATRON_LICENSE.txt.
        """
        if isinstance(cu_seqlens, torch.Tensor) and cu_seqlens.ndim == 1:
            merged_cu_seqlens = cu_seqlens.to(torch.int32)

            return merged_cu_seqlens

        if len(cu_seqlens) != micro_batch_size:
            raise ValueError("cu_seq_lens must have one boundary array per batch row")

        if micro_batch_size == 0:
            merged_cu_seqlens = torch.zeros(1, dtype=torch.int32, device="cpu")

            return merged_cu_seqlens

        def _strip_padding(row: torch.Tensor) -> torch.Tensor:
            """Return the valid prefix ending at the first occurrence of seq_length."""
            hits = (row == seq_length).nonzero(as_tuple=True)[0]

            if hits.numel() > 0:
                valid = row[:hits[0].item() + 1]

                return valid

            return row

        def _prepare_row(row: Any) -> torch.Tensor:
            """Validate the sample boundaries and remove terminal padding."""
            row = torch.as_tensor(row, device="cpu")

            if row.ndim != 1 or row.dtype not in (torch.int32, torch.int64) or row.numel() < 2:
                raise ValueError("Each cu_seq_lens row must be a 1D integer array with at least two entries")

            if row[0] != 0 or row[-1] != seq_length:
                raise ValueError("Each cu_seq_lens row must start at zero and end at the sequence length")

            valid = _strip_padding(row)

            if torch.any(valid[1:] <= valid[:-1]) or torch.any(row[valid.numel():] != seq_length):
                raise ValueError("cu_seq_lens must strictly increase before terminal padding")

            return valid

        if micro_batch_size == 1:
            merged_cu_seqlens = _prepare_row(cu_seqlens[0]).to(dtype=torch.int32, copy=True)
            return merged_cu_seqlens

        parts = [_prepare_row(cu_seqlens[0])]
        for i in range(1, micro_batch_size):
            offset = i * seq_length
            valid = _prepare_row(cu_seqlens[i])
            parts.append(valid[1:] + offset)

        merged_cu_seqlens = torch.cat(parts).to(torch.int32)

        return merged_cu_seqlens


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

    def prepare_graph_inputs(self, model_inputs: dict[str, Any]) -> dict[str, Any]:
        """Return inputs unchanged unless the model needs graph-only metadata.

        Args:
            model_inputs: Real model inputs immediately before graph execution.

        Returns:
            Inputs used for both graph capture and replay. Eager never calls this hook.
        """
        return model_inputs

    def runtime_input_fields(self) -> tuple[str, ...]:
        """Return the forward fields exclusively owned by this adapter.

        Dynamic values may vary by batch, but field ownership must be stable so
        diagnostics can reject collisions before interpreting precision runs.
        """
        return ()

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


class PackedAttentionRuntime(RuntimeInputAdapter):
    """Share global document boundaries across attention layers without a dense mask.

    The selected attention implementation must consume cumulative lengths and
    bypass model-side dense mask construction. Position IDs and loss ownership
    remain with TextParallelBatch; this adapter does not shard document bounds.
    """

    def runtime_input_fields(self) -> tuple[str, ...]:
        """Declare the two leading-zero FA length arguments."""
        return ("cu_seq_lens_q", "cu_seq_lens_k")

    def build_runtime_inputs(self, *, batch: Mapping[str, Any], context: RuntimeInputContext) -> Mapping[str, Any]:
        """Read boundaries once per batch and preserve their global batch-major coordinates."""
        if context.options.get("attention_mode") != "compressed":
            raise ValueError("PackedAttentionRuntime requires attention_mode='compressed'")
        shape = context.local_input_shape
        if len(shape) != 2:
            raise ValueError("Packed attention requires input_ids with shape [batch, local_sequence]")
        boundaries = batch.get("cu_seq_lens")
        if isinstance(boundaries, torch.Tensor):
            if boundaries.ndim != 1:
                raise ValueError("Packed attention requires one-dimensional global cu_seq_lens")
            boundaries = boundaries.tolist()
        if boundaries is None or any(isinstance(value, bool) or not isinstance(value, Integral)
                                     for value in boundaries):
            raise ValueError("Packed attention requires integer global cu_seq_lens")
        boundaries = tuple(int(value) for value in boundaries)
        if len(boundaries) < 2 or boundaries[0] != 0:
            raise ValueError("Packed attention cu_seq_lens must start with zero and contain a document")
        if any(end <= start for start, end in zip(boundaries, boundaries[1:])):
            raise ValueError("Packed attention cu_seq_lens must be strictly increasing")
        tokens = shape[0] * shape[1] * context.parallel_sizes["cp"]
        if boundaries[-1] != tokens:
            raise ValueError("Packed attention cu_seq_lens must cover the global batch token count")
        return {"cu_seq_lens_q": boundaries, "cu_seq_lens_k": boundaries}


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
    "PackedAttentionRuntime",
    "RuntimeInputAdapter",
    "RuntimeInputContext",
]

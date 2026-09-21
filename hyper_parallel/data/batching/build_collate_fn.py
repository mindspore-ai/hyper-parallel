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
"""Build final batch collators for Indexed and Online sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from math import lcm
from typing import Any

import torch
from torch.utils.data import default_collate

from hyper_parallel.data.constants import IGNORE_INDEX


def get_sequence_parallel_size(mesh_context: Any | None) -> int:
    """Return the unified sequence-sharding degree used for batch alignment.

    Args:
        mesh_context: Runtime topology exposing TP, CP, and sequence-parallel
            settings, or ``None`` for a single-rank context.

    Returns:
        Number of ranks across which one sequence is physically partitioned.
    """
    if mesh_context is None:
        return 1

    cp_size = int(getattr(mesh_context, "cp_size", 1))
    tp_size = int(getattr(mesh_context, "tp_size", 1))
    sequence_parallel = bool(getattr(mesh_context, "sequence_parallel", False))
    if cp_size <= 0 or tp_size <= 0:
        raise ValueError("mesh_context tp_size and cp_size must be positive")

    return cp_size * (tp_size if sequence_parallel else 1)


@dataclass(frozen=True)
class DataBatchContext:
    """Framework facts available throughout data-batch construction.

    The context is deliberately model-neutral. An adapter may use these facts
    to calculate physical item cost, prepare selected items, declare final
    batch constraints, or validate checkpoint-compatible state.

    Args:
        source_type: Dataset contract such as ``online`` or ``indexed``.
        sequence_parallel_size: Physical sequence-sharding degree.
        token_budget: Optional physical token budget for dynamic batching.
        pad_token_id: Optional tokenizer/model padding token.
    """

    source_type: str
    sequence_parallel_size: int = 1
    token_budget: int | None = None
    pad_token_id: int | None = None

    def __post_init__(self) -> None:
        """Validate generic batch geometry."""
        if not self.source_type:
            raise ValueError("source_type must be a non-empty string")
        if self.sequence_parallel_size <= 0:
            raise ValueError("sequence_parallel_size must be positive")
        if self.token_budget is not None and self.token_budget <= 0:
            raise ValueError("token_budget must be positive when provided")


@dataclass(frozen=True)
class BatchConstraints:
    """Physical constraints declared by a model-owned data adapter.

    Args:
        sequence_multiple: Required multiple for the final physical sequence.
    """

    sequence_multiple: int = 1

    def __post_init__(self) -> None:
        """Reject invalid physical constraints before collation."""
        if self.sequence_multiple <= 0:
            raise ValueError("sequence_multiple must be positive")


class DataBatchAdapter:
    """Extend generic selection and collation without owning a DataLoader.

    The default implementation retains items, applies standard field
    collation, and retains the resulting batch. Model integrations can override
    only the lifecycle stages they need. ``item_cost`` runs before dynamic
    selection, ``prepare_items`` runs after selection but before the generic
    collator, ``collate_items`` controls model-specific field merging, and
    ``finalize_batch`` runs after collation.
    """

    def item_cost(
            self,
            item: Mapping[str, Any],
            context: DataBatchContext,
    ) -> int:
        """Return the physical budget contribution of one source item.

        Args:
            item: One transformed Dataset output.
            context: Generic batch construction facts.

        Returns:
            Positive physical cost used by dynamic batching.
        """
        del context
        return int(item["input_ids"].shape[-1])

    def prepare_items(
            self,
            items: Sequence[Mapping[str, Any]],
            context: DataBatchContext,
    ) -> Sequence[Mapping[str, Any]]:
        """Prepare selected items without mutating the source objects.

        Args:
            items: Items selected for one forward-backward batch.
            context: Generic batch construction facts.

        Returns:
            Items consumed by the generic collator.
        """
        del context
        return items

    def constraints(self, context: DataBatchContext) -> BatchConstraints:
        """Return model-owned physical batch constraints.

        Args:
            context: Generic batch construction facts.

        Returns:
            Constraints combined with framework parallel requirements.
        """
        del context
        return BatchConstraints()

    def collate_items(
            self,
            items: Sequence[Mapping[str, Any]],
            context: DataBatchContext,
    ) -> Mapping[str, Any]:
        """Merge prepared items into one batch mapping.

        The default policy delegates to PyTorch collation and therefore
        supports mappings whose corresponding fields have compatible shapes.
        Model adapters may override this hook for variable-size modality
        fields, cumulative offsets, or derived ownership metadata.

        Args:
            items: Prepared items selected for one micro-batch.
            context: Generic batch construction facts.

        Returns:
            Collated batch mapping.
        """
        del context
        return default_collate(items)

    def finalize_batch(
            self,
            batch: Mapping[str, Any],
            context: DataBatchContext,
    ) -> Mapping[str, Any]:
        """Finalize a generic collated batch.

        Args:
            batch: Batch produced by the generic collator.
            context: Generic batch construction facts.

        Returns:
            Final DataLoader batch mapping.
        """
        del context
        return batch

    def state_signature(self, context: DataBatchContext) -> Mapping[str, Any]:
        """Return deterministic settings that affect resumable batch state.

        Args:
            context: Generic batch construction facts.

        Returns:
            A serializable mapping stored with dynamic DataLoader state.
        """
        del context
        return {}


class DataCollator(ABC):
    """Convert Dataset samples into one forward-backward micro-batch."""

    @abstractmethod
    def __call__(self, model_samples: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        """Collate one sequence of model samples.

        Args:
            model_samples: Samples selected for one forward-backward step by the
                fixed or dynamic batching policy.

        Returns:
            A collated batch mapping.
        """
        raise NotImplementedError


@dataclass
class TextPackingCollator(DataCollator):
    """Pack unpadded Online text samples and emit ``cu_seq_lens``.

    Only ``input_ids`` and pre-shifted ``labels`` are packed here. Unified
    ``get_batch`` constructs loss, position, mask, and CP runtime fields.

    Args:
        sequence_parallel_size: Compatibility argument for direct callers.
            Trainer paths provide the same value through ``context``.
        context: Shared framework batch facts.
        batch_adapter: Model-owned batch lifecycle extension.
    """

    sequence_parallel_size: int | None = None
    context: DataBatchContext = field(
        default_factory=lambda: DataBatchContext(source_type="online")
    )
    batch_adapter: DataBatchAdapter = field(default_factory=DataBatchAdapter)

    def __post_init__(self) -> None:
        """Normalize the legacy sequence-parallel constructor argument."""
        if self.sequence_parallel_size is None:
            return
        if self.sequence_parallel_size <= 0:
            raise ValueError("sequence_parallel_size must be positive")
        if self.context.sequence_parallel_size not in (1, self.sequence_parallel_size):
            raise ValueError(
                "sequence_parallel_size conflicts with context.sequence_parallel_size"
            )
        self.context = replace(
            self.context,
            sequence_parallel_size=self.sequence_parallel_size,
        )

    def __call__(self, model_samples: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        """Pack samples into one ``[1, packed_length]`` forward-backward batch.

        Args:
            model_samples: Online samples selected for one forward-backward step.

        Returns:
            Packed ``input_ids`` and ``labels`` plus int32 ``cu_seq_lens``.

        Raises:
            ValueError: If no samples are provided.
        """
        if not model_samples:
            raise ValueError("model_samples must contain at least one Online sample")

        prepared_samples = self.batch_adapter.prepare_items(model_samples, self.context)
        if not isinstance(prepared_samples, Sequence) or isinstance(prepared_samples, (str, bytes)):
            raise TypeError("DataBatchAdapter.prepare_items must return a sequence of mappings")
        if not prepared_samples:
            raise ValueError("DataBatchAdapter.prepare_items must retain at least one item")
        if any(not isinstance(item, Mapping) for item in prepared_samples):
            raise TypeError("DataBatchAdapter.prepare_items must return only mappings")

        sequence_lengths = []
        values_by_field = {"input_ids": [], "labels": []}
        for model_sample in prepared_samples:
            sample_length = int(model_sample["input_ids"].shape[-1])
            if sample_length <= 0:
                raise ValueError("Online packing items must contain at least one token")
            if int(model_sample["labels"].shape[-1]) != sample_length:
                raise ValueError("Online packing input_ids and labels must have equal lengths")
            sequence_lengths.append(sample_length)
            values_by_field["input_ids"].append(model_sample["input_ids"])
            values_by_field["labels"].append(model_sample["labels"])

        packed_batch = {
            field: torch.cat(values, dim=-1).unsqueeze(0)
            for field, values in values_by_field.items()
        }

        packed_seq_len = packed_batch["input_ids"].shape[-1]
        constraints = self.batch_adapter.constraints(self.context)
        if not isinstance(constraints, BatchConstraints):
            raise TypeError("DataBatchAdapter.constraints must return BatchConstraints")
        physical_alignment = lcm(
            self.context.sequence_parallel_size,
            constraints.sequence_multiple,
        )
        pad_len = (-packed_seq_len) % physical_alignment
        if pad_len:
            pad_token_id = 0 if self.context.pad_token_id is None else self.context.pad_token_id
            input_padding = packed_batch["input_ids"].new_full((1, pad_len), pad_token_id)
            label_padding = packed_batch["labels"].new_full((1, pad_len), IGNORE_INDEX)
            packed_batch["input_ids"] = torch.cat((packed_batch["input_ids"], input_padding), dim=-1)
            packed_batch["labels"] = torch.cat((packed_batch["labels"], label_padding), dim=-1)

        seq_lens = prepared_samples[0]["input_ids"].new_tensor(sequence_lengths)
        zero = seq_lens.new_zeros(1)
        seq_ends = seq_lens.cumsum(dim=0)
        if pad_len:
            # Represent the alignment tail as one synthetic packed sequence so
            # attention metadata covers every physical Q/KV token. Its labels
            # remain IGNORE_INDEX and therefore do not contribute to the loss.
            padded_end = seq_ends[-1:] + pad_len
            seq_ends = torch.cat((seq_ends, padded_end))
        cu_seq_lens = torch.cat((zero, seq_ends))
        cu_seq_lens = cu_seq_lens.to(torch.int32)
        packed_batch["cu_seq_lens"] = cu_seq_lens

        finalized_batch = self.batch_adapter.finalize_batch(packed_batch, self.context)
        if not isinstance(finalized_batch, Mapping):
            raise TypeError("DataBatchAdapter.finalize_batch must return a mapping")
        required_fields = {"input_ids", "labels", "cu_seq_lens"}
        missing_fields = required_fields.difference(finalized_batch)
        if missing_fields:
            raise ValueError(
                "DataBatchAdapter.finalize_batch removed required fields: "
                f"{sorted(missing_fields)}"
            )

        return finalized_batch


@dataclass
class MainCollator(DataCollator):
    """Apply modality packing after fixed or dynamic sample selection.

    Args:
        packing_collator: Text or multimodal packing implementation.
    """

    packing_collator: DataCollator

    def __call__(self, model_samples: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        """Pack selected samples and retain compact sequence boundaries."""
        packed_batch = self.packing_collator(model_samples)

        return packed_batch


def build_indexed_collate_fn() -> Callable[[list[Any]], Any]:
    """Build default collation for fixed-length Indexed samples.

    Returns:
        PyTorch default collation.
    """
    collate_fn = default_collate

    return collate_fn


def build_online_text_collate_fn(
        mesh_context: Any | None = None,
        tokenizer: Any | None = None,
        batch_adapter: DataBatchAdapter | None = None,
        batch_context: DataBatchContext | None = None,
) -> DataCollator:
    """Build Online text collation shared by fixed N and dynamic K batching.

    Packing concatenates ``input_ids`` and ``labels`` and emits ``cu_seq_lens``.
    Tail padding is derived from the runtime topology rather than user config.
    The unified alignment size is ``cp_size * tp_size`` when TP sequence
    parallelism is enabled, otherwise it is ``cp_size``. Real sample boundaries
    remain unchanged and one synthetic final boundary covers the physical
    padding segment.

    Args:
        mesh_context: Runtime TP/CP topology injected by the Trainer.
        tokenizer: Optional tokenizer providing the generic padding token.
        batch_adapter: Model-owned batch lifecycle extension.
        batch_context: Shared Trainer context. When omitted, this builder
            derives the collation subset from ``mesh_context`` and tokenizer.

    Returns:
        A collator producing one forward-backward batch.
    """
    if batch_context is None:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        batch_context = DataBatchContext(
            source_type="online",
            sequence_parallel_size=get_sequence_parallel_size(mesh_context),
            pad_token_id=pad_token_id,
        )
    packing_collator = TextPackingCollator(
        context=batch_context,
        batch_adapter=batch_adapter or DataBatchAdapter(),
    )
    collate_fn = MainCollator(packing_collator=packing_collator)

    return collate_fn

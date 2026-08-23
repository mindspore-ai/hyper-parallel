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
from dataclasses import dataclass
from typing import Any

from torch.utils.data import default_collate

from hyper_parallel.platform import get_platform
from hyper_models.components.utils.constants import IGNORE_INDEX


platform = get_platform()


def _get_sequence_parallel_size(mesh_context: Any | None) -> int:
    """Return the unified sequence-sharding degree used for batch alignment."""
    if mesh_context is None:
        return 1

    cp_size = int(getattr(mesh_context, "cp_size", 1))
    tp_size = int(getattr(mesh_context, "tp_size", 1))
    sequence_parallel = bool(getattr(mesh_context, "sequence_parallel", False))
    if cp_size <= 0 or tp_size <= 0:
        raise ValueError("mesh_context tp_size and cp_size must be positive")

    return cp_size * (tp_size if sequence_parallel else 1)


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

    Only ``input_ids`` and ``labels`` are packed here. Unified ``get_batch``
    constructs loss, position, mask, and CP runtime fields.
    """

    sequence_parallel_size: int = 1

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

        packed_batch = {}
        for field in ("input_ids", "labels"):
            values = [model_sample[field] for model_sample in model_samples]
            packed_batch[field] = platform.cat(values, dim=-1).unsqueeze(0)

        packed_seq_len = packed_batch["input_ids"].shape[-1]
        pad_len = (-packed_seq_len) % self.sequence_parallel_size
        if pad_len:
            input_padding = packed_batch["input_ids"].new_zeros((1, pad_len))
            label_padding = packed_batch["labels"].new_full((1, pad_len), IGNORE_INDEX)
            packed_batch["input_ids"] = platform.cat((packed_batch["input_ids"], input_padding), dim=-1)
            packed_batch["labels"] = platform.cat((packed_batch["labels"], label_padding), dim=-1)

        seq_lens = model_samples[0]["input_ids"].new_tensor(
            [model_sample["input_ids"].shape[-1] for model_sample in model_samples]
        )
        zero = seq_lens.new_zeros(1)
        seq_ends = seq_lens.cumsum(dim=0)
        if pad_len:
            # Represent the alignment tail as one synthetic packed sequence so
            # attention metadata covers every physical Q/KV token. Its labels
            # remain IGNORE_INDEX and therefore do not contribute to the loss.
            padded_end = seq_ends[-1:] + pad_len
            seq_ends = platform.cat((seq_ends, padded_end))
        cu_seq_lens = platform.cat((zero, seq_ends))
        cu_seq_lens = platform.tensor_type_cast(cu_seq_lens, "int32")
        packed_batch["cu_seq_lens"] = cu_seq_lens

        return packed_batch


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


def build_online_text_collate_fn(mesh_context: Any | None = None) -> DataCollator:
    """Build Online text collation shared by fixed N and dynamic K batching.

    Packing concatenates ``input_ids`` and ``labels`` and emits ``cu_seq_lens``.
    Tail padding is derived from the runtime topology rather than user config.
    The unified alignment size is ``cp_size * tp_size`` when TP sequence
    parallelism is enabled, otherwise it is ``cp_size``. Real sample boundaries
    remain unchanged and one synthetic final boundary covers the physical
    padding segment.

    Args:
        mesh_context: Runtime TP/CP topology injected by the Trainer.

    Returns:
        A collator producing one forward-backward batch.
    """
    sequence_parallel_size = _get_sequence_parallel_size(mesh_context)
    packing_collator = TextPackingCollator(sequence_parallel_size=sequence_parallel_size)
    collate_fn = MainCollator(packing_collator=packing_collator)

    return collate_fn

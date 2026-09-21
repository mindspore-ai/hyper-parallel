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
"""Build the model-neutral VLM micro-batch collator."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

from hyper_parallel.data.batching.build_collate_fn import (
    DataBatchAdapter,
    DataBatchContext,
    DataCollator,
    get_sequence_parallel_size,
)
from hyper_parallel.data.constants import IGNORE_INDEX


@dataclass
class VLMCollator(DataCollator):
    """Run the generic adapter lifecycle for one VLM micro-batch.

    The framework owns validation and lifecycle ordering. Field names, merge
    rules, cumulative offsets, and derived modality metadata belong to the
    configured model adapter.

    Args:
        context: Shared framework batch facts.
        batch_adapter: Model-owned batch lifecycle extension.
    """

    context: DataBatchContext = field(
        default_factory=lambda: DataBatchContext(source_type="online")
    )
    batch_adapter: DataBatchAdapter = field(default_factory=DataBatchAdapter)

    def __call__(self, samples: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        """Collate one micro-batch through the configured model adapter.

        Args:
            samples: Dataset items selected for one micro-batch.

        Returns:
            A collated VLM batch mapping.
        """
        if not samples:
            raise ValueError("VLM samples must contain at least one item")
        if any(not isinstance(sample, Mapping) for sample in samples):
            raise TypeError("VLM samples must contain only mappings")

        prepared_samples = self.batch_adapter.prepare_items(samples, self.context)
        if not isinstance(prepared_samples, Sequence) or isinstance(prepared_samples, (str, bytes)):
            raise TypeError("DataBatchAdapter.prepare_items must return a sequence of mappings")
        if not prepared_samples:
            raise ValueError("DataBatchAdapter.prepare_items must retain at least one item")
        if any(not isinstance(sample, Mapping) for sample in prepared_samples):
            raise TypeError("DataBatchAdapter.prepare_items must return only mappings")

        batch = self.batch_adapter.collate_items(prepared_samples, self.context)
        if not isinstance(batch, Mapping):
            raise TypeError("DataBatchAdapter.collate_items must return a mapping")
        finalized_batch = self.batch_adapter.finalize_batch(batch, self.context)
        if not isinstance(finalized_batch, Mapping):
            raise TypeError("DataBatchAdapter.finalize_batch must return a mapping")
        required_fields = {"input_ids", "labels"}
        missing_fields = required_fields.difference(finalized_batch)
        if missing_fields:
            raise ValueError(
                "DataBatchAdapter VLM collation omitted required fields: "
                f"{sorted(missing_fields)}"
            )
        return finalized_batch


def build_vlm_collator(
        *,
        packing: bool = False,
        pad_token_id: int = 0,
        ignore_index: int = IGNORE_INDEX,
        pad_to_length: Optional[int] = None,
        mesh_context: Any | None = None,
        tokenizer: Any | None = None,
        batch_adapter: DataBatchAdapter | None = None,
        batch_context: DataBatchContext | None = None,
) -> VLMCollator:
    """Build the VLM micro-batch collator.

    Args:
        packing: Reserved switch for VeOmni-style text packing.
        pad_token_id: Reserved padding value for text input IDs.
        ignore_index: Reserved label value excluded from loss computation.
        pad_to_length: Reserved packed text sequence length.
        mesh_context: Runtime topology used when ``batch_context`` is omitted.
        tokenizer: Optional tokenizer providing the generic padding token.
        batch_adapter: Model-owned batch lifecycle extension.
        batch_context: Shared Trainer context.

    Returns:
        A collator producing one VLM micro-batch dictionary.
    """
    if packing:
        raise NotImplementedError("The VLM collator does not support packing")
    if pad_token_id != 0 or ignore_index != IGNORE_INDEX or pad_to_length is not None:
        raise NotImplementedError("The VLM collator does not support custom text padding")
    if batch_context is None:
        tokenizer_pad_token_id = getattr(tokenizer, "pad_token_id", None)
        batch_context = DataBatchContext(
            source_type="online",
            sequence_parallel_size=get_sequence_parallel_size(mesh_context),
            pad_token_id=tokenizer_pad_token_id,
        )
    return VLMCollator(
        context=batch_context,
        batch_adapter=batch_adapter or DataBatchAdapter(),
    )


__all__ = ["VLMCollator", "build_vlm_collator"]

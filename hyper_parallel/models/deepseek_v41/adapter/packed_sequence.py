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
"""Compact Online-packing metadata for DeepSeek-V4.1 CSA2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import lcm
from typing import Any

from hyper_parallel.components.modules.shared_compressed_dsa_attention import (
    SharedCompressedPackedSequence,
)
from hyper_parallel.data.batching.build_collate_fn import (
    BatchConstraints,
    DataBatchAdapter,
    DataBatchContext,
)
from hyper_parallel.data.batching.runtime_input import (
    RuntimeInputAdapter,
    RuntimeInputContext,
)
from hyper_parallel.data.constants import IGNORE_INDEX


class DeepseekV41BatchAdapter(DataBatchAdapter, RuntimeInputAdapter):
    """Apply V4.1 data constraints and build CSA2 runtime metadata."""

    def __init__(self, model_config: Any) -> None:
        """Derive packing rules from the instantiated model configuration.

        Args:
            model_config: Final V4.1 config used to instantiate the model.
        """
        raw_ratios = getattr(model_config, "v41_compress_ratios", None)
        if raw_ratios is None:
            raise ValueError("DeepSeek-V4.1 model_config.v41_compress_ratios must be defined")
        ratios = [
            int(ratio)
            for ratio in raw_ratios
            if int(ratio) > 1
        ]
        self.compression_alignment = lcm(*ratios) if ratios else 1
        pad_token_id = getattr(model_config, "pad_token_id", None)
        if pad_token_id is None:
            raise ValueError("DeepSeek-V4.1 model_config.pad_token_id must be defined")
        self.pad_token_id = int(pad_token_id)

    def item_cost(
            self,
            item: Mapping[str, Any],
            context: DataBatchContext,
    ) -> int:
        """Charge dynamic batching for compression-aligned physical tokens.

        Args:
            item: One source sample before collation.
            context: Batch context shared by preparation, collation and finalization.
        """
        del context
        sequence_length = int(item["input_ids"].shape[-1])
        return sequence_length + (-sequence_length) % self.compression_alignment

    def prepare_items(
            self,
            items: Sequence[Mapping[str, Any]],
            context: DataBatchContext,
    ) -> list[Mapping[str, Any]]:
        """Pad each packed text item so CSA2 groups never cross boundaries.

        Args:
            items: Source samples to prepare and collate together.
            context: Batch context shared by preparation, collation and finalization.
        """
        del context
        prepared_items = []
        for item in items:
            input_ids = item["input_ids"]
            labels = item["labels"]
            sequence_length = int(input_ids.shape[-1])
            if int(labels.shape[-1]) != sequence_length:
                raise ValueError("DeepSeek-V4.1 input_ids and labels must have equal lengths")
            padding = (-sequence_length) % self.compression_alignment
            prepared_item = dict(item)
            if padding:
                prepared_item["input_ids"] = input_ids.new_full(
                    (sequence_length + padding,),
                    self.pad_token_id,
                )
                prepared_item["input_ids"][:sequence_length].copy_(input_ids)
                prepared_item["labels"] = labels.new_full(
                    (sequence_length + padding,),
                    IGNORE_INDEX,
                )
                prepared_item["labels"][:sequence_length].copy_(labels)
            prepared_items.append(prepared_item)
        return prepared_items

    def constraints(self, context: DataBatchContext) -> BatchConstraints:
        """Require the final physical sequence to align with every CSA2 rate.

        Args:
            context: Batch context shared by preparation, collation and finalization.
        """
        del context
        return BatchConstraints(sequence_multiple=self.compression_alignment)

    def state_signature(self, context: DataBatchContext) -> Mapping[str, Any]:
        """Describe settings that determine dynamic selection and padding.

        Args:
            context: Batch context shared by preparation, collation and finalization.
        """
        del context
        return {
            "compression_alignment": self.compression_alignment,
            "pad_token_id": self.pad_token_id,
        }

    def build_runtime_inputs(
            self,
            *,
            batch: Mapping[str, Any],
            context: RuntimeInputContext,
    ) -> Mapping[str, Any]:
        """Build compact CSA2 boundaries without an O(sequence squared) mask."""
        options = context.options
        attention_mode = options.get("attention_mode")
        if attention_mode != "compressed":
            raise ValueError(
                "DeepSeek-V4.1 runtime inputs require compressed attention, "
                f"got attention_mode={attention_mode!r}"
            )
        causal = bool(options.get("causal", True))
        cp_algorithm = str(options.get("cp_algorithm", "ulysses"))
        if not causal:
            raise ValueError("DeepSeek-V4.1 CSA2 packed attention must be causal")
        if cp_algorithm != "colossal":
            raise ValueError(
                "DeepSeek-V4.1 CSA2 requires contiguous Colossal CP shards, "
                f"got cp_algorithm={cp_algorithm!r}"
            )
        local_input_shape = context.local_input_shape
        if len(local_input_shape) != 2 or int(local_input_shape[0]) != 1:
            raise ValueError(
                "DeepSeek-V4.1 compact packing requires local input shape [1, sequence], "
                f"got {tuple(local_input_shape)}"
            )
        local_sequence_length = int(local_input_shape[1])
        cp_rank = context.parallel_ranks.get("cp", 0)
        cp_size = context.parallel_sizes.get("cp", 1)
        return {
            "packed_seq_params": SharedCompressedPackedSequence(
                cu_seq_lens=batch["cu_seq_lens"],
                local_query_start=cp_rank * local_sequence_length,
                local_query_length=local_sequence_length,
                global_sequence_length=cp_size * local_sequence_length,
            )
        }


__all__ = ["DeepseekV41BatchAdapter"]

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
"""Explicit global document metadata and mask ownership for head-chunk MLA."""

from collections.abc import Mapping, Sequence
from typing import Any

from transformers import AttentionInterface

from hyper_parallel.components.functional.npu_fusion_attention import (
    npu_fusion_attention_forward,
    resolve_packed_sequence_lengths,
)
from hyper_parallel.data.batching import build_indexed_collate_fn
from hyper_parallel.data.batching.runtime_input import RuntimeInputAdapter, RuntimeInputContext


class DeepseekV3PackedMLARuntime(RuntimeInputAdapter):
    """Translate global boundaries; never infer documents from a CP-local shard.

    TextParallelBatch owns CP slicing and document-local position IDs. The
    adapter only translates the preserved global leading-zero boundary list.
    """

    def runtime_input_fields(self) -> tuple[str, ...]:
        """Declare the one additional model input."""
        return ("actual_seq_len",)

    def build_runtime_inputs(self, *, batch: Mapping[str, Any], context: RuntimeInputContext) -> Mapping[str, Any]:
        """Validate global coverage and return cumulative ends without zero.

        Args:
            batch: CP-local token batch retaining global ``cu_seq_lens``.
            context: Local shape and parallel degrees from TextParallelBatch.

        Returns:
            Global cumulative ends under ``actual_seq_len``.
        """
        shape = context.local_input_shape
        if len(shape) != 2 or shape[0] != 1:
            raise ValueError("packed head-chunk MLA requires input_ids with shape [1, local_sequence]")
        if context.parallel_sizes.get("tp", 1) != 1:
            raise ValueError("packed head-chunk MLA does not support TP")
        if batch.get("cu_seq_lens") is None:
            raise ValueError("packed head-chunk MLA requires global cu_seq_lens")
        total = shape[1] * context.parallel_sizes.get("cp", 1)
        lengths, _ = resolve_packed_sequence_lengths({"cu_seq_lens": batch["cu_seq_lens"]}, total, total)
        return {"actual_seq_len": tuple(lengths)}


class DeepseekV3PackedIndexedCollator:
    """Mask cross-document next-token targets in already-shifted Indexed data.

    Args:
        eod_token_id: The dataset's actual document-ending token ID.
        mesh_context: Framework-supplied context; collation precedes CP slicing.
    """

    def __init__(self, *, eod_token_id: int, mesh_context: Any = None) -> None:
        """Reuse Indexed collation without changing sampling or label alignment."""
        del mesh_context
        if isinstance(eod_token_id, bool) or not isinstance(eod_token_id, int) or eod_token_id < 0:
            raise ValueError("eod_token_id must be a nonnegative integer from the dataset")
        self.eod_token_id = eod_token_id
        self._collate = build_indexed_collate_fn()

    def __call__(self, samples: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        """Write ignored targets into labels so CE and Trainer counts agree."""
        batch = self._collate(samples)
        batch["labels"] = batch["labels"].masked_fill(batch["input_ids"] == self.eod_token_id, -100)
        return batch


def register_head_chunk_mla_attention() -> str:
    """Register explicit native-kernel mask ownership before HF construction.

    Returns:
        The name to pass as the model's ``attn_implementation``.

    Note:
        No HF dense-mask registry entry is installed: native FA owns causal/TND
        masking. Use with the matching replacement and packed runtime; do not
        apply this selection to an unrelated HF attention implementation.
    """
    name = "head_chunk_mla_native"
    AttentionInterface.register(name, npu_fusion_attention_forward)
    return name

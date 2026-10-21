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
"""Build compact sample metadata for linear and full attention in kimi_k3."""

from collections.abc import Mapping
from typing import Any

from hyper_parallel.components.functional.linear_attention_packed import resolve_packed_sequence
from hyper_parallel.data.batching.runtime_input import RuntimeInputAdapter, RuntimeInputContext


class KimiK3PackedRuntime(RuntimeInputAdapter):
    """Describe one contiguous CP interval without allocating a dense attention mask.

    Position IDs, shifted labels and loss masks remain owned by TextParallelBatch.
    cu_seq_lens is the global leading-zero sample list emitted by the collator.
    """

    def runtime_input_fields(self) -> tuple[str, ...]:
        """Declare the metadata shared by the model's linear and full attention."""
        return ("packed_sequence_info", "cu_seq_lens_q", "cu_seq_lens_k")

    def build_runtime_inputs(self, *, batch: Mapping[str, Any], context: RuntimeInputContext) -> Mapping[str, Any]:
        """Snapshot global boundaries once, then reuse their immutable local plan."""
        shape = context.local_input_shape
        if len(shape) != 2 or shape[0] != 1:
            raise ValueError("Packed linear attention requires flattened input_ids [1,T].")
        if context.parallel_sizes.get("tp", 1) != 1:
            raise NotImplementedError("Packed linear attention TP with CP is not qualified.")
        if batch.get("cu_seq_lens") is None:
            raise ValueError("Packed linear attention requires global cu_seq_lens.")
        info = resolve_packed_sequence(
            batch, shape[1], cp_rank=context.parallel_ranks.get("cp", 0), cp_size=context.parallel_sizes.get("cp", 1)
        )
        # Tuples avoid device-to-host conversion in downstream full-attention adapters.
        return {
            "packed_sequence_info": info,
            "cu_seq_lens_q": info.global_offsets,
            "cu_seq_lens_k": info.global_offsets,
        }

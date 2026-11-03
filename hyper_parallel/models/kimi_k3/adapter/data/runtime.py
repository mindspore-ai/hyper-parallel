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


class KimiK3RuntimeInputAdapter(RuntimeInputAdapter):
    """Retain global media metadata while describing decoder-local documents."""

    def runtime_input_fields(self) -> tuple[str, ...]:
        """Declare packed attention metadata owned by this adapter."""
        return KimiK3PackedRuntime().runtime_input_fields()

    def build_runtime_inputs(
            self, *, batch: Mapping[str, Any], context: RuntimeInputContext,
    ) -> Mapping[str, Any]:
        """Validate local token fields and globally owned media coordinates.

        Args:
            batch: CP-local text tensors and complete replicated media metadata.
            context: Decoder coordinates supplied by OmniParallelBatch.

        Returns:
            Immutable packed metadata when the collator packed multiple documents.
        """
        if any(int(context.parallel_sizes.get(axis, 1)) != 1 for axis in ("tp", "pp")):
            raise ValueError("K3 media pretraining runtime requires TP=PP=1")
        if context.options.get("sequence_parallel", False):
            raise ValueError("K3 media pretraining does not support sequence parallelism")
        shape = tuple(context.local_input_shape)
        if len(shape) != 2 or shape[0] < 1:
            raise ValueError("K3 media pretraining requires nonempty [batch, sequence] inputs")
        for field in ("labels", "attention_mask", "position_ids"):
            if field not in batch or tuple(batch[field].shape) != shape:
                raise ValueError(f"K3 runtime {field} must match local input_ids shape {shape}")
        cp_size = int(context.parallel_sizes.get("cp", 1))
        self._validate_global_media_fields(batch, (shape[0], shape[1] * cp_size))
        if cp_size > 1 and not bool(batch.get("labels_are_shifted", False)):
            raise ValueError("K3 media CP requires labels shifted globally by the collator")
        if "cu_seq_lens" not in batch:
            return {}
        return self._packed_media_inputs(batch, context)

    @staticmethod
    def _validate_global_media_fields(batch: Mapping[str, Any], global_shape: tuple[int, int]) -> None:
        """Check replicated media provenance and ownership before any encoder work."""
        plan = batch.get("encoder_parallel_plan")
        fields = ("media_token_mask",) if plan is not None else (
            "pixel_values", "grid_thws", "media_spans", "media_patch_offsets", "media_token_mask",
        )
        for field in fields:
            if field not in batch:
                raise ValueError(f"K3 runtime requires {field}; use the K3 transform and collator")
        metadata = batch.get("media_metadata")
        media_count = (sum(len(bucket) for bucket in plan.buckets) if plan is not None
                       else batch["grid_thws"].shape[0])
        if metadata is not None and len(metadata) != media_count:
            raise ValueError("media_metadata must retain all global media groups across CP ranks")
        if tuple(batch["media_token_mask"].shape) != global_shape:
            raise ValueError("media_token_mask must retain the complete global token interval")

    @staticmethod
    def _packed_media_inputs(batch: Mapping[str, Any], context: RuntimeInputContext) -> Mapping[str, Any]:
        """Require each media interval to remain inside its complete source document."""
        info = KimiK3PackedRuntime().build_runtime_inputs(batch=batch, context=context)
        boundaries = info["packed_sequence_info"].global_offsets
        plan = batch.get("encoder_parallel_plan")
        spans = ([(item.batch_index, item.decoder_token_start,
                   item.decoder_token_start + item.decoder_token_length)
                  for bucket in plan.buckets for item in bucket] if plan is not None
                 else batch["media_spans"].tolist())
        for batch_index, start, stop in spans:
            if batch_index != 0 or not any(left <= start < stop <= right
                                           for left, right in zip(boundaries, boundaries[1:])):
                raise ValueError("A packed media span must remain inside its source document")
        return info


def build_kimi_k3_runtime_input_adapter() -> KimiK3RuntimeInputAdapter:
    """Build global-media ownership checks and document-isolated attention metadata."""
    return KimiK3RuntimeInputAdapter()

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
"""Image-only Qwen VLM training with contiguous decoder context parallelism."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

# This adapter implements the Torch-only Qwen Transformers training contract.
import torch  # pylint: disable=forbidden-backend-import
from torch.nn import functional  # pylint: disable=forbidden-backend-import

from hyper_parallel.data.batching.get_batch import OmniParallelBatch
from hyper_parallel.data.batching.runtime_input import RuntimeInputAdapter, RuntimeInputContext
from hyper_parallel.distributed.encoder_parallel.contracts import EncoderFeature, EncoderWorkItem
from hyper_parallel.models.qwen3_5_moe.adapter.encoder_parallel import QwenVisionBridge


@dataclass(frozen=True)
class QwenVLMCPInputs:
    """Precomputed full-conversation geometry and local next-token targets."""

    start: int
    stop: int
    position_ids: torch.Tensor
    targets: torch.Tensor
    images: tuple[EncoderWorkItem, ...]


def select_local_image_features(records: list[EncoderFeature], geometry: QwenVLMCPInputs,
                                hidden: torch.Tensor, input_ids: torch.Tensor,
                                image_token_id: int) -> torch.Tensor:
    """Insert only image rows intersecting this rank's global token interval.

    Args:
        records: Image work items paired with their encoded features.
        geometry: Global token interval assigned to this CP rank.
        hidden: Local text embeddings receiving image features.
        input_ids: Local token IDs identifying image placeholders.
        image_token_id: Token ID used for image placeholders.

    Returns:
        Local embeddings with image placeholders replaced by image features.
    """
    rows = []
    for item, features in sorted(records, key=lambda record: record[0].decoder_token_start):
        begin = max(geometry.start, item.decoder_token_start)
        end = min(geometry.stop, item.decoder_token_start + item.output_rows)
        if begin < end:
            rows.append(features[begin - item.decoder_token_start:end - item.decoder_token_start])
    mask = input_ids.eq(image_token_id).unsqueeze(-1).expand_as(hidden)
    features = torch.cat(rows).to(hidden.dtype) if rows else hidden.new_empty((0, hidden.shape[-1]))
    if int(mask[..., 0].sum()) != features.shape[0]:
        raise ValueError("CP-local image placeholders do not match routed image features")
    return hidden.masked_scatter(mask, features)


class QwenVLMParallelBatch(OmniParallelBatch, RuntimeInputAdapter):
    """Prepare full-image geometry before slicing an unpacked conversation over CP.

    TP/SP, video, packing and interior/left padding are intentionally unsupported.
    Decoder attention/GDN may use Ulysses or KV all-gather/P2P respectively.
    CP=1 retains native HF forward and can serve as the pure-FSDP baseline.
    """

    def __init__(self, *, mesh_context: Any, device: Any, sequence_alignment: int = 128,
                 runtime_input_adapter: Any = None, **kwargs: Any) -> None:
        """Use the same global tail alignment for the baseline and CP comparisons."""
        if mesh_context.tp_size != 1 or mesh_context.sequence_parallel:
            raise NotImplementedError("Qwen VLM P2P CP currently requires TP=1 and SP=false")
        if isinstance(sequence_alignment, bool) or not isinstance(sequence_alignment, int) or sequence_alignment <= 0:
            raise ValueError("sequence_alignment must be a positive integer")
        if sequence_alignment % (64 * mesh_context.cp_size):
            raise ValueError("sequence_alignment must be divisible by 64 * cp_size")
        self.model = None
        self.bridge = None
        self.sequence_alignment = sequence_alignment
        super().__init__(mesh_context=mesh_context, device=device,
                         runtime_input_adapter=runtime_input_adapter or self, **kwargs)

    def runtime_input_fields(self) -> tuple[str, ...]:
        """The batch prepares its geometry after the shared runtime hook."""
        return ()

    def build_runtime_inputs(self, *, batch: Mapping[str, Any], context: RuntimeInputContext) -> dict[str, Any]:
        """Keep the model-specific geometry owned by the batch adapter."""
        return {}

    def bind_model(self, model: Any) -> None:
        """Bind native mRoPE and visual metadata without changing model execution.

        Args:
            model: Qwen multimodal conditional-generation model.
        """
        if model.config.model_type not in ("qwen3_5", "qwen3_5_moe"):
            raise ValueError("QwenVLMParallelBatch requires the conditional-generation Qwen Dense or MoE model")
        self.model, self.bridge = model, QwenVisionBridge(model)

    def _pad(self, source: Mapping[str, Any]) -> dict[str, Any]:
        """Right-pad token fields together without truncating any image span."""
        batch = dict(source)
        ids = batch["input_ids"]
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1:
            raise ValueError("Qwen VLM CP requires one nonempty conversation")
        if batch.get("cu_seq_lens") is not None and batch["cu_seq_lens"].numel() > 2:
            raise NotImplementedError("Qwen VLM CP does not support packed conversations")
        mask = batch.get("attention_mask", torch.ones_like(ids))
        if not bool(mask.bool().all()):
            raise ValueError("Provide an unpadded conversation; QwenVLMParallelBatch owns tail padding")
        batch["attention_mask"] = mask
        pad = (-ids.shape[1]) % self.sequence_alignment
        for field, value in tuple(batch.items()):
            if torch.is_tensor(value) and value.shape == ids.shape and field not in self.bridge.input_fields:
                batch[field] = functional.pad(value, (0, pad), value=-100 if field == "labels" else 0)
        batch["loss_mask"] = batch["labels"].ne(-100)
        return batch

    def __call__(self, data_iterator: Any, *, external_batch: Mapping[str, Any] | None = None
                 ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Render global positions/targets, then expose only local tokens to the decoder."""
        if self.model is None:
            raise RuntimeError("bind_model must run before reading a Qwen VLM CP batch")
        source = external_batch if external_batch is not None else next(data_iterator)
        batch = self.tp_broadcaster.broadcast(
            self._pad(self._normalize_source_batch(source)), broadcast_all_fields=True,
        )
        images = self.bridge.parse_images(batch)
        cp_size = self.parallel_context.cp_world_size
        if cp_size == 1:
            runtime_inputs = self._build_runtime_inputs(batch)
            model_inputs, loss_inputs = self._split_model_and_loss_inputs(batch)
            model_inputs.update(runtime_inputs)
            if getattr(self.model, "_hp_qwen_vlm_chunk_loss_bound", False):
                targets = functional.pad(batch["labels"], (0, 1), value=-100)[:, 1:].contiguous()
                self._add_chunk_targets(model_inputs, loss_inputs, targets)
            return model_inputs, loss_inputs
        global_ids = batch["input_ids"]
        token_types = batch.get("mm_token_type_ids", global_ids.eq(self.model.config.image_token_id).long())
        positions, _ = self.model.model.get_rope_index(
            input_ids=global_ids, mm_token_type_ids=token_types,
            image_grid_thw=batch.get("image_grid_thw"), attention_mask=batch["attention_mask"],
        )
        length = global_ids.shape[1] // cp_size
        start = self.parallel_context.cp_rank * length
        stop = start + length
        targets = functional.pad(batch["labels"], (0, 1), value=-100)[:, 1:][:, start:stop].contiguous()
        geometry = QwenVLMCPInputs(start, stop, positions[..., start:stop].contiguous(), targets, images)
        for field, value in tuple(batch.items()):
            if torch.is_tensor(value) and value.shape == global_ids.shape and field not in self.bridge.input_fields:
                batch[field] = value[:, start:stop].contiguous()
        # The native trainer counts labels[..., 1:]. This sentinel preserves that
        # contract while the forward bridge uses the already globally shifted targets.
        batch["labels"] = functional.pad(targets, (1, 0), value=-100)
        batch["loss_mask"] = batch["labels"].ne(-100)
        batch["vlm_global_input_ids"] = global_ids
        runtime_inputs = self._build_runtime_inputs(batch)
        batch.pop("vlm_global_input_ids")
        model_inputs, loss_inputs = self._split_model_and_loss_inputs(batch)
        model_inputs.update(runtime_inputs)
        model_inputs["vlm_cp_inputs"] = geometry
        self._add_chunk_targets(model_inputs, loss_inputs, targets)
        return model_inputs, loss_inputs

    def _add_chunk_targets(self, model_inputs: dict[str, Any], loss_inputs: dict[str, Any],
                           targets: torch.Tensor) -> None:
        """Expose aligned labels only to the opt-in chunk objective and token counter."""
        if getattr(self.model, "_hp_qwen_vlm_chunk_loss_bound", False):
            for inputs in (model_inputs, loss_inputs):
                inputs["shift_labels"] = targets
                inputs["loss_mask"] = targets.ne(-100)

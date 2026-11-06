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


@dataclass(frozen=True)
class QwenVLMCPInputs:
    """Global sequence fields and the local next-token interval, independent of a model."""

    start: int
    stop: int
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor | None
    targets: torch.Tensor
    cp_size: int


class QwenVLMParallelBatch(OmniParallelBatch, RuntimeInputAdapter):
    """Retain global sequence metadata while slicing an unpacked conversation over CP.

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
        self.sequence_alignment = sequence_alignment
        super().__init__(mesh_context=mesh_context, device=device,
                         runtime_input_adapter=runtime_input_adapter or self, **kwargs)

    def runtime_input_fields(self) -> tuple[str, ...]:
        """The batch prepares its geometry after the shared runtime hook."""
        return ()

    def build_runtime_inputs(self, *, batch: Mapping[str, Any], context: RuntimeInputContext) -> dict[str, Any]:
        """Keep the model-specific geometry owned by the batch adapter."""
        return {}

    def _pad(self, source: Mapping[str, Any]) -> dict[str, Any]:
        """Right-pad token fields together without truncating any image span."""
        batch = dict(source)
        ids = batch["input_ids"]
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1:
            raise ValueError("Qwen VLM CP requires one nonempty conversation")
        if batch.get("cu_seq_lens") is not None and batch["cu_seq_lens"].numel() > 2:
            raise NotImplementedError("Qwen VLM CP does not support packed conversations")
        mask = batch.get("attention_mask", torch.ones_like(ids))
        if mask.shape != ids.shape or not bool(mask[:, 0].all()):
            raise ValueError("Qwen VLM CP requires a nonempty conversation without left padding")
        valid = mask.bool()
        if bool((valid[:, 1:] & ~valid[:, :-1]).any()):
            raise ValueError("Qwen VLM CP does not support interior padding")
        if bool((batch["labels"][~valid] != -100).any()):
            raise ValueError("Qwen VLM CP padding must not contain supervised labels")
        batch["attention_mask"] = mask
        pad = (-ids.shape[1]) % self.sequence_alignment
        for field, value in tuple(batch.items()):
            if torch.is_tensor(value) and value.shape == ids.shape and field not in ("pixel_values", "image_grid_thw"):
                batch[field] = functional.pad(value, (0, pad), value=-100 if field == "labels" else 0)
        batch["loss_mask"] = batch["labels"].ne(-100)
        return batch

    def __call__(self, data_iterator: Any, *, external_batch: Mapping[str, Any] | None = None
                 ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Shift targets globally and retain model-independent metadata for the adapter."""
        source = external_batch if external_batch is not None else next(data_iterator)
        batch = self.tp_broadcaster.broadcast(
            self._pad(self._normalize_source_batch(source)), broadcast_all_fields=True,
        )
        cp_size = self.parallel_context.cp_world_size
        global_ids = batch["input_ids"]
        length = global_ids.shape[1] // cp_size
        start = self.parallel_context.cp_rank * length
        stop = start + length
        targets = functional.pad(batch["labels"], (0, 1), value=-100)[:, 1:][:, start:stop].contiguous()
        geometry = QwenVLMCPInputs(
            start, stop, global_ids, batch["attention_mask"], batch.get("mm_token_type_ids"), targets, cp_size,
        )
        if cp_size == 1:
            runtime_inputs = self._build_runtime_inputs(batch)
            model_inputs, loss_inputs = self._split_model_and_loss_inputs(batch)
            model_inputs.update(runtime_inputs)
            model_inputs["vlm_cp_inputs"] = geometry
            self._add_chunk_targets(loss_inputs, targets)
            return model_inputs, loss_inputs
        for field, value in tuple(batch.items()):
            if (torch.is_tensor(value) and value.shape == global_ids.shape
                    and field not in ("pixel_values", "image_grid_thw")):
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
        self._add_chunk_targets(loss_inputs, targets)
        return model_inputs, loss_inputs

    @staticmethod
    def _add_chunk_targets(loss_inputs: dict[str, Any], targets: torch.Tensor) -> None:
        """Provide shifted targets without depending on the selected model loss adapter."""
        loss_inputs["shift_labels"] = targets
        loss_inputs["loss_mask"] = targets.ne(-100)

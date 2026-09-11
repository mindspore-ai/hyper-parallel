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
"""Temporary self-contained VLM batch preparation."""

from collections.abc import Mapping
from typing import Any

import torch

from hyper_parallel.data.batching.attention_runtime import AttentionRuntimeAdapter


_MODEL_INPUT_FIELDS = {
    "input_ids",
    "labels",
    "attention_mask",
    "position_ids",
    "text_position_ids",
    "router_attention_mask",
    "mm_token_type_ids",
    "token_types",
    "pixel_values",
    "pixel_values_videos",
    "input_features",
    "audio_features",
    "audio_attention_mask",
    "audio_feature_lengths",
    "image_mask",
    "video_mask",
    "audio_mask",
    "image_grid_hw",
    "image_grid_thw",
    "image_patch_offsets",
    "image_vit_grid_hw",
    "image_llm_grid_hw",
    "image_batch_indices",
    "image_token_starts",
    "packed_seq_params",
    "video_grid_thw",
    "video_timestamp",
}
_LOSS_INPUT_FIELDS = {"labels", "loss_mask", "stream_loss_mask"}


class VLMBatchProcessor:
    """Normalize and classify one VLM batch without shared LLM adapters."""

    @staticmethod
    def normalize_source_batch(source_batch: Mapping[str, Any]) -> dict[str, Any]:
        """Normalize one collated batch into the temporary VLM contract."""
        batch = dict(source_batch)
        if "input_ids" not in batch:
            raise ValueError("VLM batch must contain 'input_ids'")
        if "labels" not in batch:
            raise ValueError("VLM batch must contain 'labels'")
        if "loss_mask" not in batch:
            batch["loss_mask"] = batch["labels"] >= 0
        return batch

    @staticmethod
    def prepare_batch(batch: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Split device-resident VLM fields into model and loss inputs."""
        model_inputs = {field: value for field, value in batch.items() if field in _MODEL_INPUT_FIELDS}
        loss_inputs = {field: value for field, value in batch.items() if field in _LOSS_INPUT_FIELDS}
        return model_inputs, loss_inputs


class VLMGetBatch:
    """Prepare VLM batches for the temporary TP=CP=PP=1 training path."""

    def __init__(
            self,
            *,
            mesh_context: Any,
            device: Any,
            pp_shared_data: bool = False,
            attention_mode: str = "dense",
            attention_runtime_adapter: AttentionRuntimeAdapter | None = None,
            cp_algorithm: str = "ulysses",
            causal: bool = True,
            sliding_window: int | None = None,
    ) -> None:
        """Validate the temporary VLM parallel boundary and store the device.

        Args:
            mesh_context: Trainer mesh exposing TP, CP, and PP sizes.
            device: Destination model device.
            pp_shared_data: Whether pipeline stages share the source batch.
            attention_mode: ``dense`` preserves the source mask; ``compressed``
                converts right-padding into compact sequence boundaries.
            attention_runtime_adapter: Model-owned adapter used for compressed
                attention metadata.
            cp_algorithm: Context-parallel algorithm passed to the adapter.
            causal: Whether the model uses causal attention.
            sliding_window: Optional local-attention window passed to the adapter.

        Raises:
            NotImplementedError: If model parallelism or pipeline batch sharing is enabled.
        """
        parallel_sizes = {
            "tp_size": int(getattr(mesh_context, "tp_size", 1)),
            "cp_size": int(getattr(mesh_context, "cp_size", 1)),
            "pp_size": int(getattr(mesh_context, "pp_size", 1)),
        }
        unsupported_sizes = {name: size for name, size in parallel_sizes.items() if size != 1}
        if unsupported_sizes:
            raise NotImplementedError(
                "The temporary VLM batch path requires TP=CP=PP=1, but got "
                + ", ".join(f"{name}={size}" for name, size in unsupported_sizes.items())
            )
        if pp_shared_data:
            raise NotImplementedError("The temporary VLM batch path does not support pp_shared_data")
        if attention_mode not in {"dense", "compressed"}:
            raise ValueError(f"unsupported VLM attention_mode: {attention_mode!r}")
        if attention_mode == "compressed" and attention_runtime_adapter is None:
            raise ValueError("compressed VLM attention requires attention_runtime_adapter")
        self.device = device
        self.processor = VLMBatchProcessor()
        self.attention_mode = attention_mode
        self.attention_runtime_adapter = attention_runtime_adapter
        self.cp_algorithm = cp_algorithm
        self.causal = causal
        self.sliding_window = sliding_window

    def __call__(
            self,
            data_iterator: Any,
            *,
            external_batch: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Read, normalize, transfer, and classify one VLM batch.

        Args:
            data_iterator: Iterator yielding collated VLM batches.
            external_batch: Optional batch supplied without advancing the iterator.

        Returns:
            Model inputs and loss inputs on the configured device.
        """
        source_batch = external_batch if external_batch is not None else next(data_iterator)
        if not isinstance(source_batch, Mapping):
            raise ValueError("VLM DataLoader must yield a mapping batch")
        normalized_batch = self.processor.normalize_source_batch(source_batch)
        device_batch = {
            field: value.to(self.device, non_blocking=True)
            if torch.is_tensor(value) else value
            for field, value in normalized_batch.items()
        }
        if self.attention_mode == "compressed":
            device_batch["packed_seq_params"] = self._build_packed_seq_params(device_batch)
            device_batch.pop("attention_mask", None)
        return self.processor.prepare_batch(device_batch)

    def _build_packed_seq_params(self, batch: Mapping[str, Any]) -> object:
        """Convert right-padding into compact causal-attention boundaries."""
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise ValueError("compressed VLM attention requires input_ids with shape [batch, sequence]")
        if not isinstance(attention_mask, torch.Tensor) or attention_mask.shape != input_ids.shape:
            raise ValueError("compressed VLM attention requires attention_mask with input_ids shape")

        batch_size, sequence_length = input_ids.shape
        boundaries = [0]
        for batch_index, row in enumerate(attention_mask.tolist()):
            valid_length = sum(bool(value) for value in row)
            expected = [True] * valid_length + [False] * (sequence_length - valid_length)
            if [bool(value) for value in row] != expected:
                raise ValueError(
                    "compressed VLM attention supports only right-padded attention_mask rows; "
                    f"sample {batch_index} is not right-padded"
                )
            sample_start = batch_index * sequence_length
            if valid_length:
                boundaries.append(sample_start + valid_length)
            if valid_length < sequence_length:
                boundaries.append(sample_start + sequence_length)
        if len(boundaries) < 2 or boundaries[-1] != batch_size * sequence_length:
            raise ValueError("compressed VLM attention failed to cover the physical token batch")

        cu_seq_lens = torch.tensor(boundaries, dtype=torch.int64, device=input_ids.device)
        return self.attention_runtime_adapter.build_packed_seq_params(
            cu_seq_lens=cu_seq_lens,
            local_input_shape=input_ids.shape,
            cp_rank=0,
            cp_size=1,
            cp_algorithm=self.cp_algorithm,
            causal=self.causal,
            sliding_window=self.sliding_window,
        )


def build_vlm_get_batch(
        *,
        mesh_context: Any,
        device: Any,
        pp_shared_data: bool = False,
        attention_mode: str = "dense",
        attention_runtime_adapter: AttentionRuntimeAdapter | None = None,
        cp_algorithm: str = "ulysses",
        causal: bool = True,
        sliding_window: int | None = None,
) -> VLMGetBatch:
    """Build the temporary self-contained VLM batch adapter.

    Args:
        mesh_context: Trainer mesh used to validate VLM parallel sizes.
        device: Destination model device.
        pp_shared_data: Reserved pipeline batch-sharing option.
        attention_mode: Dense or compact compressed-attention input format.
        attention_runtime_adapter: Optional model-owned compact metadata adapter.
        cp_algorithm: Context-parallel algorithm selected by the model recipe.
        causal: Whether compact attention is causal.
        sliding_window: Optional compact sliding window.

    Returns:
        Callable VLM batch adapter.
    """
    return VLMGetBatch(
        mesh_context=mesh_context,
        device=device,
        pp_shared_data=pp_shared_data,
        attention_mode=attention_mode,
        attention_runtime_adapter=attention_runtime_adapter,
        cp_algorithm=cp_algorithm,
        causal=causal,
        sliding_window=sliding_window,
    )


__all__ = ["VLMBatchProcessor", "VLMGetBatch", "build_vlm_get_batch"]

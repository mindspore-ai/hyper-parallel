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
"""Opt-in Omni transport and model-input adaptation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from hyper_parallel.data.omni.constants import (
    OMNI_CP_PAD_VALUES, OMNI_CP_TOKEN_FIELDS, OMNI_INTERNAL_FIELDS, OMNI_LOSS_INPUT_FIELDS,
)
from hyper_parallel.data.omni.parallel import OmniCPBatchSharder, OmniTPBatchBroadcaster
from hyper_parallel.data.parallel import create_dataloader_parallel_context


class OmniParallelBatch:
    """Shard Omni token fields over CP and replicate modality fields over TP."""

    def __init__(
            self,
            *,
            mesh_context: Any,
            device: Any,
            pp_shared_data: bool = False,
            encoder_dp: bool = False,
            runtime_input_adapter: Any = None,
            data_config: Mapping[str, Any] | None = None,
    ) -> None:
        """Create the DataLoader ownership and TP/CP transport context.

        Args:
            mesh_context: Trainer mesh exposing TP, CP, and PP sizes.
            device: Destination model device.
            pp_shared_data: Whether pipeline stages share the source batch.
            encoder_dp: Reserved for image-bucket distribution; not implemented.
            runtime_input_adapter: Model-owned hook for additional forward inputs.
            data_config: Dataset cache settings used by DataLoader ownership.
        """
        if encoder_dp:
            raise NotImplementedError("OmniParallelBatch encoder_dp image-bucket distribution is not implemented")
        if runtime_input_adapter is not None and not callable(getattr(runtime_input_adapter, "build", None)):
            raise TypeError("runtime_input_adapter must expose build(batch=..., parallel_context=...)")
        if int(getattr(mesh_context, "pp_size", 1)) != 1 or pp_shared_data:
            raise NotImplementedError("OmniParallelBatch does not support pipeline parallelism or pp_shared_data")
        source_config = data_config or {}
        self.parallel_context = create_dataloader_parallel_context(
            mesh_context,
            data_index_cache=bool(source_config.get("data_index_cache", False)),
            shared_storage=not bool(source_config.get("no_shared_storage", False)),
        )
        if not self.parallel_context.distributed_enabled and (
                int(getattr(mesh_context, "tp_size", 1)) > 1 or int(getattr(mesh_context, "cp_size", 1)) > 1
        ):
            raise ValueError("OmniParallelBatch requires an initialized process group for TP or CP")
        self.device = device
        self.runtime_input_adapter = runtime_input_adapter
        if self.parallel_context.cp_world_size > 1 and runtime_input_adapter is None:
            raise ValueError("Omni CP requires a model runtime adapter for positions, attention and label shifting")
        token_pad_values = {field: OMNI_CP_PAD_VALUES.get(field, 0) for field in OMNI_CP_TOKEN_FIELDS}
        self.cp_sharder = OmniCPBatchSharder(self.parallel_context, token_pad_values=token_pad_values)
        self.tp_broadcaster = OmniTPBatchBroadcaster(self.parallel_context, device)

    def __call__(
            self,
            data_iterator: Any,
            *,
            external_batch: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Read on TP0, slice token fields for CP, then broadcast within TP."""
        local_batch = None
        if self.parallel_context.build_on_rank():
            source_batch = external_batch if external_batch is not None else next(data_iterator)
            if not isinstance(source_batch, Mapping):
                raise TypeError("Omni DataLoader must yield a mapping batch")
            normalized_batch = self._normalize_source_batch(source_batch)
            local_batch = self.cp_sharder.shard(normalized_batch)

        device_batch = self.tp_broadcaster.broadcast(local_batch)
        runtime_inputs = self._build_runtime_inputs(device_batch)
        model_inputs, loss_inputs = self._split_model_and_loss_inputs(device_batch)
        collisions = set(model_inputs).intersection(runtime_inputs)
        if collisions:
            raise ValueError(f"Omni runtime adapter cannot overwrite model fields: {sorted(collisions)}")
        model_inputs.update(runtime_inputs)

        return model_inputs, loss_inputs

    @staticmethod
    def _normalize_source_batch(source_batch: Mapping[str, Any]) -> dict[str, Any]:
        """Validate required fields and derive the default loss mask."""
        batch = dict(source_batch)
        if "input_ids" not in batch:
            raise ValueError("Omni batch must contain input_ids")
        if "labels" not in batch:
            raise ValueError("Omni batch must contain labels")
        if "loss_mask" not in batch:
            batch["loss_mask"] = batch["labels"] >= 0
        if "cu_seq_lens" in batch:
            batch["cu_seq_lens"] = batch["cu_seq_lens"].reshape(-1)
        return batch

    def _build_runtime_inputs(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Call the configured model hook after TP/CP batch transport."""
        if self.runtime_input_adapter is None:
            boundaries = batch.get("cu_seq_lens")
            if boundaries is not None and boundaries.numel() > 2:
                raise ValueError("Multi-sample Omni packing requires a model runtime adapter for attention isolation")
            return {}
        runtime_inputs = self.runtime_input_adapter.build(
            batch=batch, parallel_context=self.parallel_context
        )
        if not isinstance(runtime_inputs, Mapping):
            raise TypeError("Omni runtime adapter must return a mapping of model inputs")
        return dict(runtime_inputs)

    @staticmethod
    def _split_model_and_loss_inputs(
            batch: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Keep processor fields as model inputs and isolate internal metadata."""
        model_inputs = {}
        loss_inputs = {}
        for field, value in batch.items():
            if field not in OMNI_INTERNAL_FIELDS:
                model_inputs[field] = value
            if field in OMNI_LOSS_INPUT_FIELDS:
                loss_inputs[field] = value
        return model_inputs, loss_inputs

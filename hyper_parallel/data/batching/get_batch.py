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
"""Build one model and loss batch from a DataLoader iterator."""
# pylint: disable=forbidden-backend-import

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from hyper_parallel.data.batching.runtime_input import (
    AttentionRuntime,
    IndexedBoundaryResolver,
    OnlineBoundaryResolver,
    RuntimeInputAdapter,
)
from hyper_parallel.data.constants import (
    OMNI_CP_PAD_VALUES,
    OMNI_CP_TOKEN_FIELDS,
    OMNI_INTERNAL_FIELDS,
    OMNI_LOSS_INPUT_FIELDS,
)
from hyper_parallel.data.dataset_logging import get_dataset_logger
from hyper_parallel.data.parallel import (
    CPBatchSharder,
    DataLoaderParallelContext,
    TPBatchBroadcaster,
    create_dataloader_parallel_context,
)

logger = get_dataset_logger(__name__)


class OmniParallelBatch:
    """Shard Omni token fields over CP and replicate modality fields over TP."""

    def __init__(
            self,
            *,
            mesh_context: Any,
            device: Any,
            pp_shared_data: bool = False,
            encoder_dp: bool = False,
            runtime_input_adapter: RuntimeInputAdapter | None = None,
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
        if runtime_input_adapter is not None and not isinstance(runtime_input_adapter, RuntimeInputAdapter):
            raise TypeError("OmniParallelBatch runtime_input_adapter must be a RuntimeInputAdapter")
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
        token_pad_values = {field: OMNI_CP_PAD_VALUES.get(field, 0) for field in OMNI_CP_TOKEN_FIELDS}
        self.cp_sharder = CPBatchSharder(self.parallel_context, token_pad_values=token_pad_values)
        self.tp_broadcaster = TPBatchBroadcaster(self.parallel_context, device)

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

        device_batch = self.tp_broadcaster.broadcast(local_batch, broadcast_all_fields=True)
        runtime_inputs = self._build_runtime_inputs(device_batch)
        model_inputs, loss_inputs = self._split_model_and_loss_inputs(device_batch)
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
            return {}
        runtime_inputs = self.runtime_input_adapter.build(
            batch=batch, parallel_context=self.parallel_context
        )
        collisions = set(batch).intersection(runtime_inputs)
        if collisions:
            raise ValueError(
                "[HP-DATA-001] runtime inputs cannot replace Omni model inputs: "
                f"{sorted(collisions)}"
            )
        return runtime_inputs

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


class TextParallelBatch:
    """Define the DataLoader-to-forward-batch processing boundary.

    The runtime reads data only on TP rank zero at every CP coordinate, keeps
    global sequence boundaries, shards token fields for CP, broadcasts the
    local fields across TP, and then builds common attention inputs.
    """

    def __init__(
            self,
            mesh_context: Any,
            device: Any,
            tokenizer: Any,
            data_config: Mapping[str, Any],
            pp_shared_data: bool,
            *,
            source_type: str,
            attention_mode: str = "dense",
            preserve_loss_mask: bool = False,
            causal: bool = True,
            sliding_window: int | None = None,
            reset_position_ids: bool = False,
            reset_attention_mask: bool = False,
            eod_mask_loss: bool = False,
            runtime_input_adapter: RuntimeInputAdapter | None = None,
    ) -> None:
        """Initialize the batch runtime and its parallel execution context.

        Args:
            mesh_context: Trainer mesh used to create DataLoader parallel state.
            device: Device receiving the DataLoader batch.
            tokenizer: Tokenizer providing EOD and padding token semantics.
            data_config: Dataset options used to build runtime LTR fields.
            pp_shared_data: Whether pipeline stages share the prepared batch.
            source_type: ``online`` or ``indexed`` DataLoader batch contract.
            attention_mode: ``dense`` or ``compressed`` attention representation.
            preserve_loss_mask: Preserve explicit Dataset weights and their dtype.
            causal: Whether attention uses left-to-right causal semantics.
            sliding_window: Sliding-window size, or ``None`` for full attention.
            reset_position_ids: Whether positions restart at sequence boundaries.
            reset_attention_mask: Whether EOD starts an independent attention sequence.
            eod_mask_loss: Whether EOD tokens are excluded from the loss.
            runtime_input_adapter: Optional model-owned forward-input extension.
        """
        # pylint: disable=too-many-locals

        self.parallel_context: DataLoaderParallelContext = create_dataloader_parallel_context(
            mesh_context,
            data_index_cache=bool(data_config.get("data_index_cache", False)),
            shared_storage=not bool(data_config.get("no_shared_storage", False)),
        )
        self.device = device
        self.tokenizer = tokenizer
        self.data_config = dict(data_config)
        self.source_type = source_type
        self.causal = causal
        self.sliding_window = sliding_window
        if runtime_input_adapter is not None and not isinstance(
                runtime_input_adapter, RuntimeInputAdapter
        ):
            raise TypeError("TextParallelBatch runtime_input_adapter must be a RuntimeInputAdapter")
        self.runtime_input_adapter = runtime_input_adapter
        self._batch_flow_logged = False

        if source_type == "online":
            self.boundary_resolver = OnlineBoundaryResolver()
            self.source_input_field = "input_ids"
        elif source_type == "indexed":
            eod_token_id = getattr(tokenizer, "eod", None)
            self.boundary_resolver = IndexedBoundaryResolver(eod_token_id)
            self.source_input_field = "tokens"
        else:
            raise ValueError(f"Unsupported batch source type: {source_type!r}")

        self.pp_shared_data = pp_shared_data
        self.labels_are_shifted = bool(self.data_config.get("labels_are_shifted", True))
        self.preserve_loss_mask = preserve_loss_mask or bool(self.data_config.get("preserve_loss_mask", False))
        create_attention_mask = bool(
            self.data_config.get("create_attention_mask_in_dataloader", attention_mode == "dense")
        )
        self.cp_sharder = CPBatchSharder(self.parallel_context)
        self.tp_broadcaster = TPBatchBroadcaster(self.parallel_context, device)

        self.reset_position_ids = reset_position_ids or bool(self.data_config.get("reset_position_ids", False))
        resolved_reset_attention_mask = (
            reset_attention_mask
            or bool(self.data_config.get("reset_attention_mask", False))
            or source_type == "online"
        )
        self.eod_mask_loss = eod_mask_loss or bool(self.data_config.get("eod_mask_loss", False))
        self.attention_runtime = AttentionRuntime(
            mode=attention_mode,
            create_mask=create_attention_mask,
            reset_mask=resolved_reset_attention_mask,
            sliding_window=sliding_window,
        )

    def __call__(
            self,
            data_iterator: Any,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        """Build model and loss inputs for one forward-backward step.

        Args:
            data_iterator: Iterator yielding one collated Indexed or Online batch.

        Returns:
            Model inputs and loss inputs for one forward-backward step.
        """
        source_batch = self._read_source_batch(data_iterator)
        canonical_batch = self._normalize_source_batch(source_batch)
        cu_seq_lens = self._resolve_sequence_boundaries(canonical_batch)
        cp_local_batch = self.cp_sharder.shard(canonical_batch)
        if self.preserve_loss_mask:
            if cp_local_batch is not None:
                cp_local_batch["cu_seq_lens"] = cu_seq_lens
            parallel_batch = self.tp_broadcaster.broadcast(cp_local_batch, broadcast_all_fields=True)
        else:
            parallel_batch = self.tp_broadcaster.broadcast(cp_local_batch, cu_seq_lens)
        self._log_batch_flow(canonical_batch, parallel_batch)

        position_ids = self._build_local_position_ids(
            parallel_batch["input_ids"],
            parallel_batch["cu_seq_lens"],
        )
        parallel_batch["position_ids"] = position_ids

        loss_mask = self._build_loss_mask(parallel_batch)
        parallel_batch["loss_mask"] = loss_mask
        runtime_inputs = self._build_runtime_inputs(parallel_batch)

        model_inputs, loss_inputs = self._split_model_and_loss_inputs(
            parallel_batch,
            runtime_inputs,
        )

        return model_inputs, loss_inputs

    def _build_runtime_inputs(
            self,
            parallel_batch: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Build common attention inputs plus a model-owned extension."""
        runtime_inputs = self.attention_runtime.build(
            batch=parallel_batch,
            parallel_context=self.parallel_context,
        )
        if self.runtime_input_adapter is None:
            return runtime_inputs

        model_runtime_inputs = self.runtime_input_adapter.build(
            batch=parallel_batch,
            parallel_context=self.parallel_context,
            options={
                "source_type": self.source_type,
                "attention_mode": self.attention_runtime.mode,
                "causal": self.causal,
                "sliding_window": self.sliding_window,
            },
        )
        collisions = set(runtime_inputs).intersection(model_runtime_inputs)
        if collisions:
            raise ValueError(
                "[HP-DATA-001] model runtime inputs cannot replace common attention inputs: "
                f"{sorted(collisions)}"
            )
        runtime_inputs.update(model_runtime_inputs)
        return runtime_inputs

    def _log_batch_flow(
            self,
            canonical_batch: Mapping[str, Any] | None,
            parallel_batch: Mapping[str, Any],
    ) -> None:
        """Log the resolved source and parallel batch shapes once."""
        if not self._batch_flow_logged:
            source_shape = None if canonical_batch is None else tuple(canonical_batch["input_ids"].shape)
            local_shape = tuple(parallel_batch["input_ids"].shape)
            num_boundaries = 0 if parallel_batch["cu_seq_lens"] is None else parallel_batch["cu_seq_lens"].numel()
            logger.debug(
                "Parallel batch flow: source=%s, tp_rank=%d/%d, cp_rank=%d/%d, "
                "source_owner=%s, source_shape=%s, local_shape=%s, global_boundaries=%d",
                self.source_type,
                self.parallel_context.tp_rank,
                self.parallel_context.tp_world_size,
                self.parallel_context.cp_rank,
                self.parallel_context.cp_world_size,
                canonical_batch is not None,
                source_shape,
                local_shape,
                num_boundaries,
                enabled=True,
            )
            self._batch_flow_logged = True

    def _read_source_batch(self, data_iterator: Any) -> Mapping[str, Any] | None:
        """Read one complete batch on TP rank zero of each CP coordinate."""
        if self.parallel_context.build_on_rank():
            source_batch = next(data_iterator)
        else:
            source_batch = None

        return source_batch

    def _normalize_source_batch(
            self,
            source_batch: Mapping[str, Any] | None,
    ) -> Mapping[str, Any] | None:
        """Normalize Indexed and Online token fields before parallel processing."""
        if source_batch is None:
            return None

        canonical_batch = dict(source_batch)
        if self.source_input_field != "input_ids":
            canonical_batch["input_ids"] = canonical_batch.pop(self.source_input_field)

        return canonical_batch

    def _resolve_sequence_boundaries(
            self,
            canonical_batch: Mapping[str, Any] | None,
    ) -> Any:
        """Resolve global leading-zero cumulative sequence boundaries."""
        if canonical_batch is None:
            cu_seq_lens = None
        else:
            cu_seq_lens = self.boundary_resolver.resolve(canonical_batch)

        return cu_seq_lens

    def _build_local_position_ids(
            self,
            input_ids: Any,
            cu_seq_lens: Any,
    ) -> Any:
        """Build position IDs for the current CP slice on every TP rank.

        Packed boundaries define the position semantics. CP only determines
        which global sequence interval is materialized by this rank.
        """
        # pylint: disable=too-many-locals
        batch_size, local_seq_len = input_ids.shape
        cp_size = self.parallel_context.cp_world_size
        global_seq_len = local_seq_len * cp_size
        cp_rank = self.parallel_context.cp_rank
        cp_seq_start = cp_rank * local_seq_len
        cp_seq_end = cp_seq_start + local_seq_len

        # This produces the same values as building global [B, S] positions
        # and then taking the CP slice, without allocating the global tensor.
        position_ids = torch.arange(
            cp_seq_start, cp_seq_end, dtype=torch.int64, device=input_ids.device
        )
        local_position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        if self.reset_position_ids:
            local_position_ids = local_position_ids.clone()
            boundaries = cu_seq_lens.tolist()
            for seq_start, seq_end in zip(boundaries[:-1], boundaries[1:]):
                seq_len = seq_end - seq_start
                batch_idx, seq_start_in_batch = divmod(seq_start, global_seq_len)
                seq_end_in_batch = seq_start_in_batch + seq_len
                # Intersect the sequence with the current CP slice.
                slice_start = max(seq_start_in_batch, cp_seq_start)
                slice_end = min(seq_end_in_batch, cp_seq_end)
                if slice_start >= slice_end:
                    continue

                local_start = slice_start - cp_seq_start
                local_end = slice_end - cp_seq_start
                local_position_ids[batch_idx, local_start:local_end] -= seq_start_in_batch

        local_position_ids = local_position_ids.contiguous()

        return local_position_ids

    def _build_loss_mask(self, parallel_batch: Mapping[str, Any]) -> Any:
        """Build the local loss mask from labels and input IDs."""
        if self.preserve_loss_mask:
            if "loss_mask" not in parallel_batch:
                raise ValueError("preserve_loss_mask requires explicit Dataset loss_mask")
            loss_mask = parallel_batch["loss_mask"].masked_fill(parallel_batch["labels"] < 0, 0)
        else:
            loss_mask = (parallel_batch["labels"] >= 0).to(dtype=torch.int64)

        if self.eod_mask_loss:
            eod_token_id = getattr(self.tokenizer, "eod", None)
            loss_mask = loss_mask.masked_fill(parallel_batch["input_ids"] == eod_token_id, 0)

        return loss_mask

    def _split_model_and_loss_inputs(
            self,
            parallel_batch: Mapping[str, Any],
            runtime_inputs: Mapping[str, Any] | None = None,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        """Split forward fields from loss and token-accounting fields."""
        model_inputs = {
            "input_ids": parallel_batch["input_ids"],
            "labels": parallel_batch["labels"],
            "position_ids": parallel_batch["position_ids"],
        }
        if self.labels_are_shifted:
            model_inputs["shift_labels"] = parallel_batch["labels"]

        runtime_inputs = runtime_inputs or {}
        collisions = set(model_inputs).intersection(runtime_inputs)
        if collisions:
            raise ValueError(
                "[HP-DATA-001] runtime inputs cannot replace framework-owned model inputs: "
                f"{sorted(collisions)}"
            )
        model_inputs.update(runtime_inputs)

        loss_inputs = {
            "labels": parallel_batch["labels"],
            "loss_mask": parallel_batch["loss_mask"],
        }
        if self.labels_are_shifted:
            loss_inputs["shift_labels"] = parallel_batch["labels"]

        return model_inputs, loss_inputs

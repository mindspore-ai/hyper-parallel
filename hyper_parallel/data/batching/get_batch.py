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

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from hyper_parallel.platform import get_platform
from hyper_parallel.data.batching.attention_runtime import (
    AttentionRuntimeAdapter,
    build_dense_attention_masks,
)
from hyper_parallel.data.batching.sequence_boundaries import (
    IndexedBoundaryResolver,
    OnlineBoundaryResolver,
)
from hyper_parallel.data.dataset_logging import get_dataset_logger
from hyper_parallel.data.parallel import (
    CPBatchSharder,
    DataLoaderParallelContext,
    TPBatchBroadcaster,
    create_dataloader_parallel_context,
)

platform = get_platform()
logger = get_dataset_logger(__name__)

class ParallelBatch:
    """Define the DataLoader-to-forward-batch processing boundary.

    The runtime reads data only on TP rank zero at every CP coordinate, keeps
    global sequence boundaries, shards token fields for CP, broadcasts the
    local fields across TP, and then builds dense or compressed attention data.
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
            cp_algorithm: str = "ulysses",
            causal: bool = True,
            sliding_window: int | None = None,
            reset_position_ids: bool = False,
            reset_attention_mask: bool = False,
            eod_mask_loss: bool = False,
            attention_runtime_adapter: AttentionRuntimeAdapter | None = None,
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
            cp_algorithm: Context-parallel sequence sharding algorithm.
            causal: Whether attention uses left-to-right causal semantics.
            sliding_window: Sliding-window size, or ``None`` for full attention.
            reset_position_ids: Whether positions restart at sequence boundaries.
            reset_attention_mask: Whether EOD starts an independent attention sequence.
            eod_mask_loss: Whether EOD tokens are excluded from the loss.
            attention_runtime_adapter: Compressed Attention/CP metadata adapter.
        """

        self.parallel_context: DataLoaderParallelContext = create_dataloader_parallel_context(
            mesh_context,
            data_index_cache=bool(data_config.get("data_index_cache", False)),
            shared_storage=not bool(data_config.get("no_shared_storage", False)),
        )
        self.device = device
        self.tokenizer = tokenizer
        self.data_config = dict(data_config)
        self.source_type = source_type
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
        self.attention_mode = attention_mode
        self.cp_algorithm = cp_algorithm
        self.labels_are_shifted = bool(self.data_config.get("labels_are_shifted", True))
        self.create_attention_mask = bool(
            self.data_config.get("create_attention_mask_in_dataloader", attention_mode == "dense")
        )
        self.cp_sharder = CPBatchSharder(self.parallel_context)
        self.tp_broadcaster = TPBatchBroadcaster(self.parallel_context, device)

        self.causal = causal
        self.sliding_window = sliding_window
        self.reset_position_ids = reset_position_ids or bool(self.data_config.get("reset_position_ids", False))
        self.reset_attention_mask = (
            reset_attention_mask
            or bool(self.data_config.get("reset_attention_mask", False))
            or source_type == "online"
        )
        self.eod_mask_loss = eod_mask_loss or bool(self.data_config.get("eod_mask_loss", False))
        self.attention_runtime_adapter = attention_runtime_adapter

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
        parallel_batch = self.tp_broadcaster.broadcast(
            cp_local_batch,
            cu_seq_lens,
        )
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

        position_ids = self._build_local_position_ids(
            parallel_batch["input_ids"],
            parallel_batch["cu_seq_lens"],
        )
        parallel_batch["position_ids"] = position_ids

        loss_mask = self._build_loss_mask(parallel_batch)
        attention_mask, swa_mask, packed_seq_params = self._build_attention_data(
            parallel_batch,
        )
        parallel_batch["loss_mask"] = loss_mask
        parallel_batch["attention_mask"] = attention_mask
        parallel_batch["swa_mask"] = swa_mask
        parallel_batch["packed_seq_params"] = packed_seq_params

        model_inputs, loss_inputs = self._split_model_and_loss_inputs(parallel_batch)

        return model_inputs, loss_inputs

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
        batch_size, local_seq_len = input_ids.shape
        cp_size = self.parallel_context.cp_world_size
        global_seq_len = local_seq_len * cp_size
        cp_rank = self.parallel_context.cp_rank
        cp_seq_start = cp_rank * local_seq_len
        cp_seq_end = cp_seq_start + local_seq_len

        # This produces the same values as building global [B, S] positions
        # and then taking the CP slice, without allocating the global tensor.
        position_ids = platform.arange(
            cp_seq_start, cp_seq_end, dtype=platform.tensor_dtype.int64, device=input_ids.device
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
        loss_mask = (parallel_batch["labels"] >= 0).to(dtype=platform.tensor_dtype.int64)

        if self.eod_mask_loss:
            eod_token_id = getattr(self.tokenizer, "eod", None)
            loss_mask = loss_mask.masked_fill(parallel_batch["input_ids"] == eod_token_id, 0)

        return loss_mask

    def _build_attention_data(
            self,
            parallel_batch: Mapping[str, Any],
    ) -> tuple[Any | None, Any | None, object | None]:
        """Build dense masks or compressed Attention/CP metadata."""
        attention_mask = None
        swa_mask = None
        packed_seq_params = None
        if not self.create_attention_mask:
            return attention_mask, swa_mask, packed_seq_params

        mask_compress = self.attention_mode == "compressed"

        if mask_compress:
            if self.attention_runtime_adapter is not None:
                packed_seq_params = self.attention_runtime_adapter.build_packed_seq_params(
                    cu_seq_lens=parallel_batch["cu_seq_lens"],
                    local_input_shape=parallel_batch["input_ids"].shape,
                    cp_rank=self.parallel_context.cp_rank,
                    cp_size=self.parallel_context.cp_world_size,
                    cp_algorithm=self.cp_algorithm,
                    causal=self.causal,
                    sliding_window=self.sliding_window,
                )
        else:
            # Dense masks remain global although input IDs are already CP-local.
            micro_batch_size, local_seq_length = parallel_batch["input_ids"].shape
            seq_length = local_seq_length * self.parallel_context.cp_world_size
            attention_mask, swa_mask = build_dense_attention_masks(
                cu_seq_lens=parallel_batch["cu_seq_lens"],
                micro_batch_size=micro_batch_size,
                seq_length=seq_length,
                device=parallel_batch["input_ids"].device,
                reset_attention_mask=self.reset_attention_mask,
                sliding_window=self.sliding_window,
            )

        return attention_mask, swa_mask, packed_seq_params

    def _split_model_and_loss_inputs(
            self,
            parallel_batch: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        """Split forward fields from loss and token-accounting fields."""
        model_inputs = {
            "input_ids": parallel_batch["input_ids"],
            "labels": parallel_batch["labels"],
            "position_ids": parallel_batch["position_ids"],
            "attention_mask": parallel_batch["attention_mask"],
        }
        if self.labels_are_shifted:
            model_inputs["shift_labels"] = parallel_batch["labels"]
        if parallel_batch["swa_mask"] is not None:
            model_inputs["swa_mask"] = parallel_batch["swa_mask"]
        if parallel_batch["packed_seq_params"] is not None:
            model_inputs["packed_seq_params"] = parallel_batch["packed_seq_params"]

        loss_inputs = {
            "labels": parallel_batch["labels"],
            "loss_mask": parallel_batch["loss_mask"],
        }
        if self.labels_are_shifted:
            loss_inputs["shift_labels"] = parallel_batch["labels"]

        return model_inputs, loss_inputs

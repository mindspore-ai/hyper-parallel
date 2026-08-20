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
"""TP source-rank reading, device transfer, and batch broadcasting."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

from hyper_parallel.platform import get_platform
from hyper_models.components.datasets.parallel.batch_context import BatchParallelContext


platform = get_platform()


class DistributedBatchTransport:
    """Move and distribute one source batch within its TP group.

    Pipeline field routing is intentionally reserved. In particular, Omni
    inputs must eventually route raw modality fields to encoder ranks, encoded
    features to the first LLM stage, and loss fields to the last LLM stage.
    Broadcasting the complete batch across PP would obscure that contract and
    waste bandwidth.
    """

    def __init__(
        self,
        *,
        parallel_context: BatchParallelContext,
        device: Any,
        field_names: Collection[str] | None = None,
    ) -> None:
        """Store runtime topology and destination device."""
        if parallel_context.tp_size <= 0:
            raise ValueError("tp_size must be positive")
        if not 0 <= parallel_context.tp_rank < parallel_context.tp_size:
            raise ValueError(
                f"tp_rank must be in [0, {parallel_context.tp_size}), "
                f"got {parallel_context.tp_rank}"
            )
        if parallel_context.tp_size > 1 and parallel_context.tp_group is None:
            raise ValueError("TP batch transport requires a TP process group")
        self.parallel_context = parallel_context
        self.device = device
        self.field_names = frozenset(field_names) if field_names is not None else None

    def read_source_batch(
        self,
        data_iterator: Any,
        external_batch: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any] | None, bool]:
        """Read only on the configured TP/PP source rank.

        Args:
            data_iterator: Iterator over collated CPU micro-batches.
            external_batch: Optional batch supplied without a DataLoader.

        Returns:
            Source batch and whether the source iterator is exhausted.
        """
        if not self.parallel_context.reads_data():
            return None, False
        if external_batch is not None:
            if not isinstance(external_batch, Mapping):
                raise ValueError("external_batch must be a mapping")
            source_batch = dict(external_batch)
            return source_batch, False
        if data_iterator is None:
            raise ValueError("data_iterator is required on the batch source rank")
        try:
            raw_batch = next(data_iterator)
        except StopIteration:
            return None, True
        if not isinstance(raw_batch, Mapping):
            raise ValueError("DataLoader must yield a mapping batch")
        source_batch = dict(raw_batch)
        return source_batch, False

    def broadcast_batch(
        self,
        source_batch: Mapping[str, Any] | None,
        *,
        source_exhausted: bool,
    ) -> dict[str, Any]:
        """Move source tensors to device and broadcast them across TP.

        Args:
            source_batch: Normalized batch on TP rank zero, otherwise ``None``.
            source_exhausted: Whether the source iterator raised ``StopIteration``.

        Returns:
            Device batch available on every TP rank.

        Raises:
            StopIteration: If the source DataLoader has no next batch.
        """
        if self.parallel_context.tp_size <= 1:
            if source_exhausted:
                raise StopIteration
            if source_batch is None:
                raise ValueError("source batch is required when TP is disabled")
            device_batch = self._move_batch_to_device(source_batch)
            return device_batch

        distributed_batch, distributed_exhausted = self._broadcast_over_group(
            source_batch,
            source_exhausted=source_exhausted,
            group=self.parallel_context.tp_group,
            group_size=self.parallel_context.tp_size,
            group_rank=self.parallel_context.tp_rank,
        )
        if distributed_exhausted:
            raise StopIteration
        if distributed_batch is None:
            raise ValueError("batch source rank did not provide a batch")
        device_batch = distributed_batch
        return device_batch

    def _broadcast_over_group(
        self,
        source_batch: Mapping[str, Any] | None,
        *,
        source_exhausted: bool,
        group: Any,
        group_size: int,
        group_rank: int,
    ) -> tuple[dict[str, Any] | None, bool]:
        """Broadcast a dynamically shaped batch from group rank zero."""
        owns_source = group_rank == 0
        payload = self._build_source_payload(
            source_batch,
            source_exhausted,
            owns_source=owns_source,
        )
        gathered_payloads = [None] * group_size
        platform.all_gather_object(
            gathered_payloads,
            payload,
            group=group,
        )
        source_payload = gathered_payloads[0]
        if not isinstance(source_payload, dict):
            raise ValueError("batch source rank did not provide batch metadata")
        if source_payload.get("exhausted", False):
            return None, True

        device_batch = self._materialize_and_broadcast(
            source_batch,
            source_payload,
            owns_source=owns_source,
            group=group,
        )
        return device_batch, False

    def _build_source_payload(
        self,
        source_batch: Mapping[str, Any] | None,
        source_exhausted: bool,
        *,
        owns_source: bool,
    ) -> dict[str, Any] | None:
        """Describe source tensors and carry non-tensor values."""
        if not owns_source:
            return None
        if source_exhausted:
            return {"exhausted": True}
        if source_batch is None:
            raise ValueError("source batch cannot be None before TP broadcast")

        tensor_schema = {}
        object_values = {}
        for field_name, field_value in source_batch.items():
            if self.field_names is not None and field_name not in self.field_names:
                continue
            if platform.is_tensor(field_value):
                tensor_schema[field_name] = {
                    "shape": tuple(field_value.shape),
                    "dtype": field_value.dtype,
                }
            else:
                object_values[field_name] = field_value
        payload = {
            "exhausted": False,
            "tensor_schema": tensor_schema,
            "object_values": object_values,
        }
        return payload

    def _materialize_and_broadcast(
        self,
        source_batch: Mapping[str, Any] | None,
        source_payload: Mapping[str, Any],
        *,
        owns_source: bool,
        group: Any,
    ) -> dict[str, Any]:
        """Allocate peer tensors and execute process-group broadcasts."""
        object_values = source_payload.get("object_values", {})
        tensor_schema = source_payload.get("tensor_schema", {})
        device_batch = dict(object_values)
        for field_name, field_schema in tensor_schema.items():
            if owns_source:
                if source_batch is None:
                    raise ValueError("source batch disappeared during broadcast")
                field_tensor = platform.move_to_device(
                    source_batch[field_name],
                    self.device,
                    non_blocking=True,
                )
            else:
                field_tensor = platform.empty(
                    field_schema["shape"],
                    dtype=field_schema["dtype"],
                    device=self.device,
                )
            platform.broadcast(
                field_tensor,
                group=group,
                group_src=0,
            )
            device_batch[field_name] = field_tensor
        return device_batch

    def _move_batch_to_device(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Move top-level tensor fields while retaining Python metadata."""
        device_batch = {}
        for field_name, field_value in batch.items():
            if platform.is_tensor(field_value):
                device_batch[field_name] = platform.move_to_device(
                    field_value,
                    self.device,
                    non_blocking=True,
                )
            else:
                device_batch[field_name] = field_value
        return device_batch


__all__ = ["DistributedBatchTransport"]

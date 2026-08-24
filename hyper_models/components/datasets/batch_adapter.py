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
"""Shared runtime orchestration around model-specific batch processors."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from hyper_models.components.datasets.parallel.batch_context import BatchParallelContext
from hyper_models.components.datasets.parallel.batch_transport import DistributedBatchTransport
from hyper_models.components.datasets.parallel.pipeline_router import PipelineBatchRouter


class BatchProcessor(Protocol):
    """Model-family-specific batch normalization and classification."""

    def normalize_source_batch(
        self,
        source_batch: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Normalize the Dataset field contract before communication."""

    def prepare_batch(self, batch: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Apply model-specific sharding and classify runtime fields."""


class RuntimeBatchAdapter:
    """Compose source reading, optional PP routing, TP transport, and processing."""

    def __init__(
        self,
        *,
        parallel_context: BatchParallelContext,
        transport: DistributedBatchTransport,
        processor: BatchProcessor,
        pipeline_router: PipelineBatchRouter | None = None,
    ) -> None:
        """Store the independently replaceable runtime batch components."""
        if (
            parallel_context.pp_shared_data
            and parallel_context.pp_size > 1
            and pipeline_router is None
        ):
            raise NotImplementedError(
                "pp_shared_data requires a stage-aware PipelineBatchRouter"
            )
        self.parallel_context = parallel_context
        self.transport = transport
        self.processor = processor
        self.pipeline_router = pipeline_router

    def __call__(
        self,
        data_iterator: Any,
        *,
        external_batch: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build one distributed, model-ready micro-batch.

        Args:
            data_iterator: Iterator over collated CPU micro-batches.
            external_batch: Optional batch supplied without advancing an iterator.

        Returns:
            Model inputs and loss inputs.
        """
        source_batch, source_exhausted = self.transport.read_source_batch(
            data_iterator,
            external_batch,
        )
        normalized_batch = self.processor.normalize_source_batch(source_batch)
        if self.pipeline_router is not None:
            normalized_batch, source_exhausted = self.pipeline_router.route_source_batch(
                normalized_batch,
                source_exhausted=source_exhausted,
            )
        distributed_batch = self.transport.broadcast_batch(
            normalized_batch,
            source_exhausted=source_exhausted,
        )
        model_inputs, loss_inputs = self.processor.prepare_batch(distributed_batch)
        return model_inputs, loss_inputs

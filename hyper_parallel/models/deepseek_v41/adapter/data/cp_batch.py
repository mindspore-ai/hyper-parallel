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
"""Optional host-only CP planning for TP1 eager text training.

This adapter wraps an already constructed CPU source. Distributed/device-prefetch
loaders and graph capture require their own producer integration; they must not
be moved onto this background thread.
"""
from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.data.batching import TextParallelBatch
from hyper_parallel.distributed.context_parallel.batch_plan import (
    CPBatchPlan, CPBatchPlanConfig, prepare_cp_batch,
)
from hyper_parallel.trainer.runtime.data_iterator import BackgroundPrefetcher


@dataclass(frozen=True)
class PreparedCPBatch:
    """Keep a CPU payload and its immutable plan together in the prefetch queue."""

    payload: dict[str, Any]
    plan: CPBatchPlan


class _PlanningSource:
    """Map CPU batches without changing source advancement or checkpoint ownership."""

    def __init__(self, source: Any, config: CPBatchPlanConfig) -> None:
        self.source = source
        self.config = config

    def __iter__(self) -> Iterator[PreparedCPBatch]:
        """Prepare host metadata while retaining the source payload order."""
        for batch in self.source:
            tokens = batch["input_ids"]
            boundaries = batch["cu_seq_lens"]
            if not isinstance(tokens, torch.Tensor) or not isinstance(boundaries, torch.Tensor):
                raise TypeError("CP batch prefetch requires tensor token IDs and cu_seq_lens")
            if tokens.device.type != "cpu" or boundaries.device.type != "cpu":
                raise ValueError("CP batch prefetch requires a CPU-only source before device transfer")
            if tokens.ndim != 2 or tokens.shape[0] != 1 or boundaries.ndim != 1:
                raise ValueError("CP batch prefetch requires [1,S] tokens and flat global cu_seq_lens")
            ends = tuple(int(value) for value in boundaries.tolist())
            if not ends or ends[0] != 0 or ends[-1] != tokens.shape[1]:
                raise ValueError("CP batch boundaries must cover the CPU token payload")
            lengths = tuple(end - start for start, end in zip(ends, ends[1:]))
            yield PreparedCPBatch(dict(batch), prepare_cp_batch(lengths, self.config))

    def state_dict(self) -> dict:
        """Snapshot the source cursor before its next speculative advance."""
        state = getattr(self.source, "state_dict", None)
        return deepcopy(state()) if callable(state) else {}


class CPPlannedDataLoader:
    """Wrap a stateful CPU loader with optional one-batch host preparation.

    The checkpoint cursor follows consumed batches, including within GAS. Queued
    plans are disposable metadata and are rebuilt after restoring the source.
    This eager/TP1 integration does not enable itself by default.
    """

    def __init__(self, source: Any, config: CPBatchPlanConfig, *, prefetch: bool = False) -> None:
        """Configure preparation without starting a worker or advancing the source."""
        if config.size < 2:
            raise ValueError("CP batch preparation requires CP > 1")
        if hasattr(source, "wait_for_prefetch") or getattr(source, "use_background_prefetcher", False):
            raise ValueError("Use the collective loader's own producer hook, not a background thread")
        self.source = source
        self.config = config
        self.prefetch = bool(prefetch)
        self._prepared_source = _PlanningSource(source, config)
        self._iterator = None
        self._consumed_state = self._prepared_source.state_dict()

    def __len__(self) -> int:
        """Return the original number of micro-batches."""
        return len(self.source)

    def __iter__(self) -> CPPlannedDataLoader:
        """Start a bounded producer for this source iteration."""
        self.close()
        self._consumed_state = self._prepared_source.state_dict()
        self._iterator = (BackgroundPrefetcher(self._prepared_source, maxsize=1) if self.prefetch
                          else iter(self._prepared_source))
        return self

    def __next__(self) -> PreparedCPBatch:
        """Consume data and its matching plan together before publishing the cursor."""
        if self._iterator is None:
            raise RuntimeError("Iterate the CP loader before consuming batches")
        result = next(self._iterator)
        self._consumed_state = (deepcopy(self._iterator.state_dict()) if self.prefetch
                                else self._prepared_source.state_dict())
        return result

    def state_dict(self) -> dict:
        """Return only the consumed source state; never checkpoint queued plans."""
        return deepcopy(self._consumed_state)

    def load_state_dict(self, state: dict) -> None:
        """Stop the producer before restoring the original source cursor.

        Args:
            state: Previously consumed source state returned by state_dict.
        """
        self.close()
        self.source.load_state_dict(deepcopy(state))
        self._consumed_state = deepcopy(state)

    def set_epoch(self, epoch: int) -> None:
        """Keep source epoch changes outside the producer lifetime.

        Args:
            epoch: Epoch index to forward to the underlying source.
        """
        self.close()
        setter = getattr(self.source, "set_epoch", None)
        if callable(setter):
            setter(epoch)
        self._consumed_state = self._prepared_source.state_dict()

    def close(self) -> None:
        """Join host preparation and release queued payloads on completion or failure."""
        if isinstance(self._iterator, BackgroundPrefetcher):
            self._iterator.stop(timeout=None)
        self._iterator = None


class CPPlannedGetBatch:
    """Compose prepared host metadata with the normal TP1 text batch runtime."""

    def __init__(self, get_batch: Any, config: CPBatchPlanConfig, *, prefetch: bool = False) -> None:
        """Require a source-owning TP1 rank; device movement stays in get_batch."""
        context = get_batch.parallel_context
        if context.tp_world_size != 1 or not context.build_on_rank():
            raise ValueError("Prepared CP text batches currently require TP1 source-owning ranks")
        if (context.cp_world_size, context.cp_rank) != (config.size, config.rank):
            raise ValueError("Prepared CP config differs from the text batch runtime")
        self.get_batch = get_batch
        self.config = config
        self.runtime_input_adapter = get_batch.runtime_input_adapter
        self.prefetch = prefetch
        self._loader = None

    def prepare_dataloader(self, source: Any, *, graph_mode: bool = False) -> CPPlannedDataLoader:
        """Attach optional host preparation through TextTrainer's batch-runtime hook.

        Args:
            source: Original stateful CPU data loader.
            graph_mode: Whether the Trainer selected graph compilation.
        """
        if graph_mode:
            raise ValueError("CP CPU plan prefetch currently supports eager TextTrainer only")
        if self._loader is not None:
            raise RuntimeError("A CP batch runtime cannot prepare two active data loaders")
        self._loader = CPPlannedDataLoader(source, self.config, prefetch=self.prefetch)
        return self._loader

    def close_data_pipeline(self) -> None:
        """Stop speculative CPU planning before Trainer teardown, including on errors."""
        if self._loader is not None:
            self._loader.close()

    def __call__(self, data_iterator: Any) -> tuple[dict, dict]:
        """Use a plan belonging to this payload without any mutable current-batch state."""
        prepared = next(data_iterator)
        if not isinstance(prepared, PreparedCPBatch):
            raise TypeError("CPPlannedGetBatch must consume CPPlannedDataLoader batches")
        model_inputs, loss_inputs = self.get_batch(iter((prepared.payload,)))
        packed = model_inputs["packed_seq_params"]
        local = sum(prepared.plan.lengths) // self.config.size
        if (packed.local_query_start, packed.local_query_length, packed.global_sequence_length) != (
                self.config.rank * local, local, self.config.size * local):
            raise ValueError("Prepared CPU plan differs from the runtime CP shard")
        return {**model_inputs, "cp_batch_plan": prepared.plan}, loss_inputs


def build_cp_planned_text_batch(*, plan_options: dict[str, Any], prefetch: bool = False,
                                **runtime_kwargs: Any) -> CPPlannedGetBatch:
    """Build an opt-in V4.1 eager TextTrainer batch runtime from a YAML target.

    Args:
        plan_options: Ratios, windows and CP routing options matching the model.
            CP size/rank come from the Trainer and must not be supplied here.
        prefetch: Prepare the following CPU batch on a bounded background thread.
        **runtime_kwargs: The normal TextParallelBatch constructor arguments.

    Returns:
        A batch runtime that attaches a matching immutable plan to each payload.
    """
    if not isinstance(prefetch, bool):
        raise ValueError("CP planning prefetch must be a bool")
    get_batch = TextParallelBatch(**runtime_kwargs)
    context = get_batch.parallel_context
    config = CPBatchPlanConfig(context.cp_world_size, context.cp_rank, **plan_options)
    return CPPlannedGetBatch(get_batch, config, prefetch=prefetch)

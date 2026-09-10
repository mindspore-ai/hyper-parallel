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
"""One-shot normal-training data probe owned exclusively by Dry-run."""
# Dry-run deliberately reuses TextTrainer's existing private build stages.
# pylint: disable=protected-access

from dataclasses import dataclass
from typing import Any, Mapping

from hyper_parallel.trainer.base import BaseTrainer
from hyper_parallel.trainer.runtime.loss_aggregation import count_loss_token
from hyper_parallel.trainer.text_trainer import TextTrainer


@dataclass(frozen=True)
class PreparedDryRunBatch:
    """One normal-training batch and its CPU-derived token counts."""

    model_inputs: Mapping[str, Any]
    loss_inputs: Mapping[str, Any]
    token_counts: Mapping[str, int]


class DryRunDataProbe:
    """Reuse TextTrainer data methods without constructing a real trainer."""

    def __init__(self, base: BaseTrainer) -> None:
        """Attach a temporary TextTrainer facade to Dry-run's BaseTrainer state."""
        self.base = base
        self._iterator: Any = None
        self._trainer = TextTrainer.__new__(TextTrainer)
        self._trainer.base = base

    def build(self) -> None:
        """Invoke the unchanged TextTrainer data construction sequence."""
        self._trainer._build_model_assets()
        self._trainer._build_data_transform()
        self.base._build_dataset()
        self._trainer._build_collate_fn()
        self.base._build_dataloader()
        self._trainer._build_get_batch()

    def read_first_batch(self) -> PreparedDryRunBatch:
        """Read this rank's first fully prepared training micro-batch."""
        dataloader = getattr(self.base, "train_dataloader", None)
        if dataloader is None:
            raise ValueError("A non-empty train dataloader is required")
        self._iterator = iter(dataloader)
        try:
            model_inputs, loss_inputs = self.base.get_batch(self._iterator)
        except StopIteration as exc:
            raise ValueError("Train dataloader produced no batches") from exc
        if not isinstance(model_inputs, Mapping) or not isinstance(loss_inputs, Mapping):
            raise ValueError("dataloader.get_batch must return model_inputs and loss_inputs mappings")
        counts = {
            name: int(value.item())
            for name, value in count_loss_token(dict(loss_inputs)).items()
        }
        return PreparedDryRunBatch(dict(model_inputs), dict(loss_inputs), counts)

    def close(self) -> None:
        """Stop workers owned by the one-shot iterator, if any."""
        iterator = self._iterator
        self._iterator = None
        shutdown_workers = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown_workers):
            shutdown_workers()  # pylint: disable=not-callable


__all__ = ["DryRunDataProbe", "PreparedDryRunBatch"]

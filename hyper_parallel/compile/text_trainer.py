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
"""Graph-mode text trainer reusing the eager TextTrainer runtime."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from hyper_parallel.trainer.runtime.loss_aggregation import count_loss_token
from hyper_parallel.trainer.text_trainer import TextTrainer
from hyper_parallel.trainer.config import TrainerConfig

from .trainer import GraphTrainer


class GraphTextTrainer(TextTrainer):
    """Reuse TextTrainer and delegate graph execution to ``GraphTrainer``."""

    def __init__(
        self,
        config: TrainerConfig,
    ) -> None:
        """Build graph-mode text training on top of the eager TextTrainer stages."""
        super().__init__(config)
        self._graph_train_fn = self._default_train_fn
        self.base_graph_trainer = GraphTrainer(
            model=self.base.model,
            train_fn=self._graph_train_fn,
            trainer_config=config,
            device=self.base.device,
            mesh_context=self.base.mesh,
            manage_optimizer=False,
        )

    def _default_train_fn(
        self,
        model: torch.nn.Module,
        model_inputs: Mapping[str, Any],
        loss_inputs: Mapping[str, Any],
    ) -> torch.Tensor:
        """Default graph trace function for Transformer-style text training."""
        outputs = model(**dict(model_inputs), use_cache=False)
        labels = loss_inputs.get("labels")
        loss = self.base.loss_fn(model_output=outputs, labels=labels)
        if isinstance(loss, dict):
            return torch.stack(list(loss.values())).sum()
        return loss

    def set_pytree_pre_hook(self, hook: Any) -> "GraphTextTrainer":
        """Register a tracer pre-hook on the underlying graph executor."""
        self.base_graph_trainer.set_pytree_pre_hook(hook)
        return self

    def forward_backward_step(
        self,
        data_iterator: Any,
        num_micro_steps: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Fetch one text batch and execute graph-mode forward/backward."""
        model_inputs, loss_inputs = self.base.get_batch(data_iterator)
        self.base.current_token_counts = count_loss_token(loss_inputs)
        self.base.step_token_counts = {
            name: token_count * num_micro_steps
            for name, token_count in self.base.current_token_counts.items()
        }
        loss = self.base_graph_trainer.train_step(model_inputs, loss_inputs)
        return loss, _loss_to_metrics(loss)


def _loss_to_metrics(loss: Any) -> dict[str, Any]:
    """Normalize graph loss output to a callback-friendly metrics mapping."""
    if isinstance(loss, dict):
        return {
            str(name): value.detach() if hasattr(value, "detach") else value
            for name, value in loss.items()
        }
    return {"graph_loss": loss.detach() if hasattr(loss, "detach") else loss}

__all__ = ["GraphTextTrainer"]

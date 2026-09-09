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
"""Graph-mode text trainer that reuses the eager Trainer/AutoModel runtime."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Optional

import torch

from hyper_parallel.trainer.config import TrainerConfig
from hyper_parallel.trainer.runtime.loss_aggregation import count_loss_token
from hyper_parallel.trainer.text_trainer import TextTrainer

from .dependency_bridge import (
    build_pass_config_from_trainer_config,
    clone_config_for_graph_mode,
    loss_to_metrics,
)
from .parallel_config import PassConfig
from .sharding_config import PassPlan
from .trainer import GraphTrainer as GraphExecutionEngine


class GraphTextTrainer(TextTrainer):
    """Run the text Trainer flow with graph-mode forward/backward execution.

    Phase-1 scope:
    - Reuse eager ``TextTrainer`` for setup, data, optimizer, scheduler, and
      callbacks.
    - Reuse AutoModel for model preparation.
    - Replace only the forward/backward micro-step with the graph tracer and
      graph pass pipeline.

    Advanced graph-specific features (graph FSDP, overlap scheduling, plugin
    registration) stay optional and are intentionally off by default in this
    first integration milestone.
    """

    def __init__(
        self,
        config: TrainerConfig,
        *,
        train_fn: Optional[Callable[[Any, Any, Any], Any]] = None,
        pass_config: Optional[PassConfig] = None,
        pass_plan: Optional[PassPlan] = None,
    ) -> None:
        graph_config = clone_config_for_graph_mode(config)
        super().__init__(graph_config)
        self.graph_pass_config = pass_config or build_pass_config_from_trainer_config(
            graph_config
        )
        self.graph_pass_plan = pass_plan
        self._graph_train_fn = train_fn or self._default_train_fn
        self.graph_executor = GraphExecutionEngine(
            model=self.base.model,
            train_fn=self._graph_train_fn,
            pass_config=self.graph_pass_config,
            pass_plan=self.graph_pass_plan,
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
        """Default graph trace function for Transformers-style text models."""
        outputs = model(**dict(model_inputs), use_cache=False)
        labels = loss_inputs.get("labels")
        loss = self.base.loss_fn(model_output=outputs, labels=labels)
        if isinstance(loss, dict):
            return torch.stack(list(loss.values())).sum()
        return loss

    def set_pytree_pre_hook(
        self,
        hook: Callable[[], None],
    ) -> "GraphTextTrainer":
        """Register a tracer pre-hook on the underlying graph executor."""
        self.graph_executor.set_pytree_pre_hook(hook)
        return self

    def forward_backward_step(
        self,
        data_iterator: Any,
        num_micro_steps: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Fetch one batch and run the graph-mode forward/backward step."""
        model_inputs, loss_inputs = self.base.get_batch(data_iterator)
        self.base.current_token_counts = count_loss_token(loss_inputs)
        self.base.step_token_counts = {
            name: token_count * num_micro_steps
            for name, token_count in self.base.current_token_counts.items()
        }
        loss = self.graph_executor.train_step(model_inputs, loss_inputs)
        return loss, loss_to_metrics(loss)


__all__ = ["GraphTextTrainer"]

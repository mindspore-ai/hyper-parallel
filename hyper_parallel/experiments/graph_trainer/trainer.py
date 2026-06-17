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
"""Minimal TorchTitan-style train-step tracer.

The trainer traces ``forward + loss + torch.autograd.grad`` instead of only
``model.forward``. This is the narrow experiment needed to check whether
SimpleFSDP's all-gather and reduce-scatter communication can live inside a
full train-step FX graph.
"""
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

import torch
import torch.distributed as dist  # pylint: disable=C0415
from torch import nn
from torch.fx import GraphModule
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.utils import stateless

from hyper_parallel.experiments.graph_trainer.config import GraphTrainerConfig
from hyper_parallel.experiments.graph_trainer.graph_debug import (
    CollectiveGraphSummary,
    dump_graph_debug,
    summarize_collectives,
)


LossFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class GraphStepResult:
    """Result from one traced train step."""

    loss: torch.Tensor
    grads: tuple[torch.Tensor, ...]
    graph_summary: CollectiveGraphSummary


class GraphTrainerTraceError(RuntimeError):
    """Raised when a full train-step graph cannot be traced."""


def _named_trainable_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    """Return trainable parameters in stable module order."""
    return [
        (name, param)
        for name, param in model.named_parameters(remove_duplicate=False)
        if param.requires_grad
    ]


def _named_buffers(model: nn.Module) -> list[tuple[str, torch.Tensor]]:
    """Return buffers in stable module order."""
    return list(model.named_buffers(remove_duplicate=False))


def _extract_module_state(model: nn.Module) -> OrderedDict[str, torch.Tensor]:
    """Return a merged ordered dict of the module parameters and buffers."""
    return OrderedDict(
        [
            *model.named_parameters(remove_duplicate=False),
            *model.named_buffers(remove_duplicate=False),
        ]
    )


def _normalize_step_outputs(outputs) -> tuple[torch.Tensor, ...]:
    """Normalize FX graph outputs to a tensor tuple."""
    if isinstance(outputs, torch.Tensor):
        return (outputs,)
    if isinstance(outputs, list):
        return tuple(outputs)
    if isinstance(outputs, tuple):
        return outputs
    raise TypeError(f"Expected traced step to return Tensor/list/tuple, got {type(outputs)!r}.")


def make_fwd_bwd_step(
    model: nn.Module,
    loss_fn: LossFn,
) -> Callable[..., tuple[torch.Tensor, ...]]:
    """Build a function that traces forward, loss, and backward gradients.

    Args:
        model: Model captured in the train-step closure.
        loss_fn: Function mapping model output to a scalar loss.

    Returns:
        Callable that runs ``model -> loss -> torch.autograd.grad``.
    """

    def fwd_bwd_step(
        grad_params: tuple[torch.Tensor, ...],
        *model_args: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        output = model(*model_args)
        loss = loss_fn(output)
        grads = torch.autograd.grad(loss, grad_params)
        return (loss, *grads)

    return fwd_bwd_step


class SimpleGraphTrainer:
    """Trace and run a SimpleFSDP train step as an FX graph."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: LossFn,
        optimizer: torch.optim.Optimizer | None = None,
        config: GraphTrainerConfig | None = None,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.config = config or GraphTrainerConfig()
        self.param_items = _named_trainable_parameters(model)
        self.buffer_items = _named_buffers(model)
        self.state_names = tuple(_extract_module_state(model).keys())
        self.graph_module: GraphModule | None = None
        self.callable_graph = None
        self.graph_summary = CollectiveGraphSummary(total_nodes=0, collective_nodes=[])

        if not self.param_items:
            raise ValueError("SimpleGraphTrainer requires at least one trainable parameter.")

    def _state_values(self) -> tuple[torch.Tensor, ...]:
        """Return live parameter and buffer tensors used as graph inputs."""
        return tuple(_extract_module_state(self.model).values())

    def _state_dict_from_flat(self, flat_state: tuple[torch.Tensor, ...]) -> OrderedDict[str, torch.Tensor]:
        """Rebuild a module state dict from flattened graph inputs."""
        return OrderedDict(zip(self.state_names, flat_state))

    def trace(self, *model_args: torch.Tensor) -> GraphModule:
        """Trace ``forward + loss + backward`` with the current model state."""
        step_fn = make_fwd_bwd_step(self.model, self.loss_fn)

        def stateless_step(*flat_inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
            flat_state = flat_inputs[:len(self.state_names)]
            user_args = flat_inputs[len(self.state_names):]
            state = self._state_dict_from_flat(flat_state)
            grad_params = flat_state[:len(self.param_items)]
            with stateless._reparametrize_module(self.model, state):  # pylint: disable=W0212
                return step_fn(grad_params, *user_args)

        example_inputs = (*self._state_values(), *model_args)
        try:
            graph_module = make_fx(
                stateless_step,
                tracing_mode=self.config.tracing_mode,
            )(*example_inputs)
        except Exception as error:
            raise GraphTrainerTraceError(
                "Failed to trace forward + loss + torch.autograd.grad. "
                "This usually means a parameter access path, custom autograd "
                "collective, or torch.distributed collective is not FX-traceable yet."
            ) from error
        self.graph_module = graph_module
        self.graph_summary = summarize_collectives(graph_module)

        if self.config.dump_graph and self._should_dump_graph():
            dump_graph_debug(
                graph_module,
                self.config.dump_dir,
                "train_step",
                self.graph_summary,
            )

        if self.config.compile_backend is None:
            self.callable_graph = graph_module
        else:
            self.callable_graph = torch.compile(
                graph_module,
                backend=self.config.compile_backend,
                fullgraph=self.config.fullgraph,
            )
        return graph_module

    def step(self, *model_args: torch.Tensor) -> GraphStepResult:
        """Run one traced train step and optionally step the optimizer."""
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
        if self.callable_graph is None:
            self.trace(*model_args)

        outputs = _normalize_step_outputs(
            self.callable_graph(*self._state_values(), *model_args)
        )
        loss = outputs[0]
        grads = tuple(outputs[1:])
        self._writeback_grads(grads)

        if self.optimizer is not None:
            self.optimizer.step()

        return GraphStepResult(
            loss=loss,
            grads=grads,
            graph_summary=self.graph_summary,
        )

    def _writeback_grads(self, grads: tuple[torch.Tensor, ...]) -> None:
        """Write traced graph gradient outputs back to live parameters."""
        if len(grads) != len(self.param_items):
            raise ValueError(
                f"Expected {len(self.param_items)} grads from traced graph, got {len(grads)}."
            )
        for (_, param), grad in zip(self.param_items, grads):
            param.grad = grad.detach().clone()

    @staticmethod
    def _should_dump_graph() -> bool:
        """Dump graph only on rank 0 when distributed is initialized."""
        return not dist.is_initialized() or dist.get_rank() == 0

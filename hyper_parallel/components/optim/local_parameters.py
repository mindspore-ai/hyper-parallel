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
"""Zero-copy optimizer views for locally executed sharded parameters."""

from functools import partial
from typing import Any, Sequence

import torch
import torch.distributed as dist

from hyper_parallel.core.dtensor.dtensor import DTensor


def _sum_gradient(group: Any, gradient: torch.Tensor) -> torch.Tensor:
    """Reduce each micro-batch contribution before accumulating it locally."""
    result = gradient.clone()
    dist.all_reduce(result, group=group)
    return result


def _bind_gradient(view: torch.nn.Parameter, parameter: torch.nn.Parameter) -> None:
    """Share the accumulated local gradient with its distributed optimizer view."""
    gradient = DTensor.from_local(parameter.grad, view.device_mesh, view.placements)
    view.grad = gradient
    parameter.main_grad = gradient


def _clear_local_gradients(model: torch.nn.Module, optimizer: Any, args: tuple, kwargs: dict) -> None:
    """Release local gradient aliases after the last optimizer leaf completes."""
    del optimizer, args, kwargs
    for parameter in model.parameters():
        parameter.grad = None
        parameter.main_grad = None


def bind_local_parameters(model: torch.nn.Module, optimizer: Any, *, replicated_names: Sequence[str]) -> None:
    """Attach zero-copy distributed views using the planner's source layouts.

    Args:
        model: Model with local parameters and source_shard_info from its planner.
        optimizer: Public chained optimizer to bind before its first update.
        replicated_names: Parameters whose local gradients need source-mesh summation.
    """
    layouts = model.source_shard_info
    views = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        placements, mesh = layouts[name]
        view = torch.nn.Parameter(DTensor.from_local(parameter.detach(), mesh, placements))
        views[parameter] = view
        if name in replicated_names:
            parameter.register_hook(partial(_sum_gradient, mesh.get_group()))
        parameter.register_post_accumulate_grad_hook(partial(_bind_gradient, view))
    for leaf in optimizer.chained_optimizers:
        for group in leaf.param_groups:
            group["params"] = [views[p] for p in group["params"]]
    optimizer.reset_optimizer_parameters({view: parameter for parameter, view in views.items()})
    optimizer.chained_optimizers[-1].register_step_post_hook(partial(_clear_local_gradients, model))

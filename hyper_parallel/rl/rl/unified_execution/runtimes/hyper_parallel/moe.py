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
"""Training-role dynamic EP binding and execution-path validation."""

from typing import Any, Callable

from hyper_parallel.distributed.recipe_spec import local_compute
from rl.unified_execution.models.bindings import bind_expert_ownership, validate_expert_binding
from rl.unified_execution.ops.moe.training_ep import training_ep


def bind_training_ep(module: Any, ep_mesh: Any) -> Callable:
    """Bind differentiable routing at construction time; eval never changes the role."""
    router_fn, options = bind_expert_ownership(module, ep_mesh)
    transport = training_ep

    def compute_fn(module: Any, hidden_states: Any) -> Any:
        """Route flattened token rows and restore the HP boundary shape."""
        states = hidden_states.reshape(-1, hidden_states.shape[-1])
        indices, weights = router_fn(module, states)
        output = transport(module, states, indices, weights, **options)
        return output.view(hidden_states.shape)

    return compute_fn


@local_compute
def unified_ep_compute_fn(*, module: Any, mesh: Any, tp_mesh: Any, cp_mesh: Any,
                          ep_mesh: Any, mc2_capacity: int = 256, inference_backend: str = "auto") -> Callable:
    """Train with HP dynamic EP; shared inference options never limit training rows."""
    del mesh, tp_mesh, cp_mesh
    if not isinstance(mc2_capacity, int) or isinstance(mc2_capacity, bool) or mc2_capacity < 1:
        raise ValueError("MC2 capacity must be a positive integer")
    if inference_backend not in ("auto", "mc2", "allgather"):
        raise ValueError("Unified EP inference_backend must be auto, mc2 or allgather")
    return bind_training_ep(module, ep_mesh)


def validate_training_ep(module: Any, function: Any, ep_size: int, path: str,
                         router_fn: Callable, recipe: dict) -> None:
    """Require the differentiable transport even during Actor probability evaluation."""
    del recipe
    validate_expert_binding(module, function, ep_size, path, router_fn, bind_training_ep, training_ep)

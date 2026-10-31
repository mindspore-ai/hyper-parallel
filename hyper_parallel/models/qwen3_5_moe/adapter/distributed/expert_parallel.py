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
"""Expert-parallel compute factory for Qwen3.5-MoE and Qwen3.6."""

from typing import Any, Callable

from hyper_parallel.distributed.expert_parallel.experts import (
    bind_local_expert_forward,
    ep_routed_forward,
    require_attrs,
)
from hyper_parallel.distributed.expert_parallel.routing import (
    MOE_ROUTER_ADAPTERS,
)
from hyper_parallel.distributed.recipe_spec import local_compute


@local_compute
def qwen3_5_moe_ep_compute_fn(
    *,
    module: Any,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
    use_grouped_gemm: bool = False,
) -> Callable:
    """Build Qwen3.5-MoE compute with routed and shared experts.

    The execution order follows the Transformers implementation exactly:
    shared expert, router and routed experts, shared-expert gate, then merge.
    """
    del mesh, tp_mesh, cp_mesh
    if ep_mesh is None:
        raise ValueError(
            "qwen3_5_moe_ep_compute_fn requires an active ep_mesh; "
            "set ep_size greater than 1 before selecting this factory"
        )

    require_attrs(
        module,
        "gate",
        "experts",
        "shared_expert",
        "shared_expert_gate",
        owner="Qwen3.5-MoE EP compute",
    )
    ep_group = ep_mesh.get_group("ep")
    bind_local_expert_forward(
        module,
        ep_mesh["ep"].size(),
        use_grouped_gemm=use_grouped_gemm,
    )

    def compute_fn(module: Any, hidden_states):
        """Run the Qwen3.5-MoE block while preserving branch order."""
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flattened_states = hidden_states.reshape(-1, hidden_dim)
        shared_output = module.shared_expert(flattened_states)
        routed_output = ep_routed_forward(
            module,
            hidden_states,
            router_fn=MOE_ROUTER_ADAPTERS["qwen3moe"],
            ep_group=ep_group,
        ).reshape(-1, hidden_dim)
        shared_gate = module.shared_expert_gate(flattened_states).sigmoid()
        output = routed_output + shared_gate * shared_output
        return output.reshape(batch_size, sequence_length, hidden_dim)

    return compute_fn


__all__ = ["qwen3_5_moe_ep_compute_fn"]

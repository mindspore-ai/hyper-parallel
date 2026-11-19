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
"""Resolve inference EP buckets outside Dynamo using native host schedule metadata."""

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.distributed as dist

from hyper_parallel.distributed.recipe_spec import local_compute
from rl.unified_execution.models.bindings import bind_expert_ownership, validate_expert_binding
from rl.unified_execution.ops.moe.allgather_ep import allgather_ep
from rl.unified_execution.ops.moe.mc2_ep import mc2_ep


@dataclass(frozen=True)
class InferenceExpertExecutor:
    """Worker-owned experts and immutable limits shared by all captured buckets."""

    experts: Any
    options: dict
    backend: str
    comm_name: str
    mc2_capacity: int
    max_token_rows: int
    row_alignment: int
    dp_size: int

    def select_backend_and_bucket(self, rows: int, context: Any) -> tuple[str, int]:
        """Use a common physical row bound on all EP members, including empty ranks.

        Args:
            rows: Local model rows after HP SP alignment, before EP padding.
            context: Native forward context with synchronized host DP counts or
                the actual graph batch descriptor; no device routing histogram.

        Returns:
            (backend, bucket_rows). MC2 uses its configured common capacity;
            AllGather uses the current aligned schedule, bounded by startup limits.
        """
        metadata = context.dp_metadata
        if metadata is not None:
            # Reuse host scheduler counts already synchronized by vLLM. Reading
            # routing counts from the device here would break graph capture.
            counts = metadata.num_tokens_across_dp_cpu
            if counts.device.type != "cpu":
                raise ValueError("Inference EP requires native CPU scheduling metadata")
            scheduled = int(counts.max())
        elif self.dp_size > 1:
            raise ValueError("Inference EP requires synchronized DP scheduling metadata")
        elif context.batch_descriptor is not None:
            scheduled = context.batch_descriptor.num_tokens
        else:
            scheduled = rows
        bucket = max(1, ((scheduled + self.row_alignment - 1) // self.row_alignment) * self.row_alignment)
        if rows > bucket or bucket > self.max_token_rows:
            raise ValueError(
                f"Inference EP schedule/layout mismatch: rows={rows}, bucket={bucket}, "
                f"scheduler/graph bound={self.max_token_rows}")
        if self.backend == "mc2" and bucket <= self.mc2_capacity:
            return "mc2", self.mc2_capacity
        # A legal large batch exceeds only the MC2 fast path, not inference
        # limits. AllGather uses this batch's bucket rather than the global maximum.
        return "allgather", bucket

    def forward(self, states: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor,
                context: Any) -> torch.Tensor:
        """Select before communication; communication failures are never retried."""
        backend, bucket_rows = self.select_backend_and_bucket(states.shape[0], context)
        if backend == "mc2":
            return mc2_ep(states, indices, weights, experts=self.experts, comm_name=self.comm_name,
                          rank=self.options["rank"], world_size=self.options["world_size"],
                          local_experts=self.options["local_experts"], mc2_capacity=bucket_rows)
        return allgather_ep(states, indices, weights, experts=self.experts, bucket_rows=bucket_rows, **self.options)


def register_expert_executor(executor: InferenceExpertExecutor) -> str:
    """Use vLLM's model-owned registry instead of a process-global expert cache."""
    # vLLM is optional in training and configuration processes.
    from vllm.config import get_current_vllm_config  # pylint: disable=C0415

    layers = get_current_vllm_config().compilation_config.static_forward_context
    name = f"hyper_rl.ep.{len(layers)}"
    layers[name] = executor
    return name


@torch.library.custom_op("hyper_rl::inference_ep", mutates_args=())
def execute_inference_ep(states: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor,
                          layer_name: str) -> torch.Tensor:
    """Keep per-batch transport selection out of symbolic tracing, but inside ACL capture."""
    # Resolve metadata at execution/capture time, never freeze it during Dynamo tracing.
    from vllm.forward_context import get_forward_context  # pylint: disable=C0415

    context = get_forward_context()
    executor = context.no_compile_layers[layer_name]
    return executor.forward(states, indices, weights, context)


@execute_inference_ep.register_fake
def _execute_inference_ep_fake(states: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor,
                                layer_name: str) -> torch.Tensor:
    del indices, weights, layer_name
    return torch.empty_like(states)


def bind_inference_ep(module: Any, ep_mesh: Any, *, mc2_capacity: int, backend: str,
                      comm_name: str, max_token_rows: int, row_alignment: int, dp_size: int) -> Callable:
    """Register worker-owned execution before tracing or graph capture."""
    if not isinstance(mc2_capacity, int) or isinstance(mc2_capacity, bool) or mc2_capacity < 1:
        raise ValueError("MC2 capacity must be a positive integer")
    if max_token_rows < 1:
        raise ValueError("Inference EP requires a resolved scheduler/graph row bound")
    router_fn, options = bind_expert_ownership(module, ep_mesh)
    executor = InferenceExpertExecutor(module.experts, options, backend, comm_name, mc2_capacity,
                                       max_token_rows, row_alignment, dp_size)
    executor_key = register_expert_executor(executor)
    transport = execute_inference_ep

    def compute_fn(module: Any, hidden_states: Any) -> Any:
        """Keep the router outside the opaque communication/GMM graph operation."""
        states = hidden_states.reshape(-1, hidden_states.shape[-1])
        indices, weights = router_fn(module, states)
        output = transport(states, indices, weights, executor_key)
        return output.view(hidden_states.shape)

    # Audits retain both ownership and the exact object reached by the opaque op.
    compute_fn.expert_executor = executor
    compute_fn.expert_options = options
    return compute_fn


@local_compute
def inference_ep_compute_fn(*, module: Any, mesh: Any, tp_mesh: Any, cp_mesh: Any,
                            ep_mesh: Any, mc2_capacity: int, backend: str, comm_name: str,
                            max_token_rows: int, row_alignment: int, dp_size: int) -> Callable:
    """Apply the worker-resolved backend without switching transport on eval/train."""
    del mesh, tp_mesh, cp_mesh
    if backend not in ("mc2", "allgather") or (backend == "mc2") != bool(comm_name):
        raise ValueError("Inference EP backend and dedicated communicator disagree")
    return bind_inference_ep(module, ep_mesh, mc2_capacity=mc2_capacity, backend=backend,
                             comm_name=comm_name, max_token_rows=max_token_rows,
                             row_alignment=row_alignment, dp_size=dp_size)


def validate_inference_ep(module: Any, function: Any, ep_size: int, path: str,
                          router_fn: Callable, recipe: dict, *, runtime: Any) -> None:
    """Match the live graph registry, worker transport and HP expert ownership."""
    # These dependencies are needed only when auditing a realized inference model.
    from vllm.config import get_current_vllm_config  # pylint: disable=C0415

    from hyper_parallel.components.modules.grouped_experts import GroupedExperts  # pylint: disable=C0415

    if runtime is None:
        raise ValueError("Inference EP audit requires its worker-owned runtime")
    closure = validate_expert_binding(module, function, ep_size, path, router_fn,
                                      bind_inference_ep, execute_inference_ep)
    options = function.func.expert_options
    executor = function.func.expert_executor
    registered = get_current_vllm_config().compilation_config.static_forward_context[closure["executor_key"]]
    if (executor is not registered or executor.experts is not module.experts or executor.options is not options
            or executor.mc2_capacity != recipe["mc2_capacity"] or executor.backend != runtime.backend
            or executor.comm_name != runtime.comm_name or executor.max_token_rows != runtime.max_token_rows
            or executor.row_alignment != runtime.row_alignment or executor.dp_size != runtime.dp_size):
        raise ValueError("Inference EP executor differs from the audited scheduler/transport contract")
    if recipe["inference_backend"] not in ("auto", runtime.backend):
        raise ValueError("Inference EP backend differs from the explicit recipe")
    if getattr(module.experts.forward_expert_major, "__func__", None) is not GroupedExperts.forward_expert_major:
        raise ValueError("Inference EP requires HP GroupedExperts EIO computation")
    if runtime.backend == "mc2":
        if (runtime.group is options["group"] or runtime.group is None or not runtime.comm_name
                or dist.get_process_group_ranks(runtime.group) != dist.get_process_group_ranks(options["group"])):
            raise ValueError("MC2 must own a dedicated group with the HP expert rank order")

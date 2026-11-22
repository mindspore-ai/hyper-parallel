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
"""EP-only MegaMoE execution and its parameterless runtime holder."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import Tensor, nn

from hyper_parallel.components.modules.moe.grouped_experts.grouped_experts import (
    GroupedExpertsConfig, GroupedExpertsModule, match_grouped_experts_inputs,
)
from hyper_parallel.components.modules.moe.grouped_experts.ep.ep_grouped_experts import validate_ep_layout
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.distributed.module_binding import ParallelBinding, ParallelModuleSpec


class _MegaExpertsRuntime:
    """Keep native scratch resources without registering a second parameter owner."""

    def __init__(self, config: Any, *, dispatch_mode: str = "push",
                 initial_capacity_factor: float | None = None,
                 capacity_growth_factor: float | None = None) -> None:
        """Record algorithm facts and the existing Mega transport options."""
        if config.top_k is None:
            raise ValueError("MegaKernel requires config.top_k from the model or explicit standalone config")
        if config.activation is not None:
            raise ValueError(
                "MegaKernel supports SwiGLU with optional swiglu_limit, not arbitrary activation callables"
            )
        if dispatch_mode not in ("push", "pull"):
            raise ValueError("Mega dispatch_mode must be push or pull")
        self.config = config
        self.tuning = {"dispatch_mode": dispatch_mode, "initial_capacity_factor": initial_capacity_factor,
                       "capacity_growth_factor": capacity_growth_factor}
        self.executor = None

    def configure(self, group: Any) -> None:
        """Borrow the existing ordered EP group and the upstream numerical implementation.

        Args:
            group: Job-owned process group with contiguous expert ownership.
        """
        if self.executor is not None:
            return
        # The optional accelerator package is loaded only for an explicitly selected Mega member.
        from hyper_multicore import MegaMoeExperts, MegaMoeNumericsOptions, SwiGLUParameters  # pylint: disable=C0415
        from hyper_multicore.experimental import MegaMoeTuningOptions  # pylint: disable=C0415

        config = self.config
        executor = MegaMoeExperts(
            hidden_size=config.hidden_size, intermediate_size=config.intermediate_size,
            num_experts=config.num_experts, top_k=config.top_k, ep_group=group,
            activation=SwiGLUParameters(clamp_limit=config.swiglu_limit, clamp_after_silu=False),
            numerics_options=MegaMoeNumericsOptions(
                apply_probs_before_fc2=config.routing_weight_mode == "before_fc2",
            ),
            tuning_options=MegaMoeTuningOptions(**self.tuning),
        )
        if any(True for _ in executor.parameters()):
            executor.close()
            raise RuntimeError("MegaKernel requires the parameterless HyperMulticore executor API")
        self.executor = executor

    def __call__(self, hidden: Tensor, ids: Tensor, weights: Tensor,
                 gate_up: Tensor, down: Tensor) -> Tensor:
        """Pass current canonical weights directly; the native executor infers EOI layout."""
        if self.executor is None:
            raise RuntimeError("prepare_module_runtimes must run before MegaKernel forward")
        return self.executor(
            hidden, ids, weights,
            expert_weights=(gate_up, down),
        )

    @staticmethod
    def share(runtimes: list["_MegaExpertsRuntime"]) -> None:
        """Reuse the established serial-layer workspace sharing implementation.

        Args:
            runtimes: Configured peers with compatible serial execution resources.
        """
        # The native package owns allocation, sharing and in-flight lifetime validation.
        from hyper_multicore import MegaMoeExperts  # pylint: disable=C0415

        MegaMoeExperts.share_execution_resources([runtime.executor for runtime in runtimes])

    def close(self) -> None:
        """Release native storage after its final reader has completed."""
        if self.executor is not None:
            self.executor.close()
            self.executor = None


def _validate_mega_layout(*, module, sharding, context) -> None:
    if context.axis_sizes["ep"] <= 1:
        raise ValueError("MegaMoE requires EP>1; use ordinary GroupedExperts for EP=1")
    validate_ep_layout(module=module, sharding=sharding, context=context)


class EPMegaGroupedExperts(GroupedExpertsModule):
    """Run one fused token-level call; the kernel owns its transport and restoration."""

    shared_expert_overlap_supported = False

    def __init__(self, *, config: GroupedExpertsConfig | None = None, module: nn.Module | None = None,
                 module_fqn: str = "", context: Mapping[str, Any] | None = None,
                 dispatch_mode: str = "push", initial_capacity_factor: float | None = None,
                 capacity_growth_factor: float | None = None) -> None:
        """Adopt canonical state and record a parameterless native runtime holder.

        Args:
            config: Global expert dimensions and routing/activation semantics.
            module: Unsharded packed source, exclusive with config.
            module_fqn: Expert checkpoint scope.
            context: Model configuration and parallel topology facts.
            dispatch_mode: Existing Mega push or pull transport policy.
            initial_capacity_factor: Optional push receive-capacity factor.
            capacity_growth_factor: Optional push growth factor.
        """
        super().__init__(config=config, module=module, module_fqn=module_fqn, context=context)
        self._runtime = _MegaExpertsRuntime(
            self.config, dispatch_mode=dispatch_mode, initial_capacity_factor=initial_capacity_factor,
            capacity_growth_factor=capacity_growth_factor,
        )
        self._ep_group: Any = None
        self._parallel_bound = False

    def bind_parallel(self, binding: ParallelBinding) -> None:
        """Borrow the actual expert group; preserve the existing fused backend limits.

        Args:
            binding: Existing job topology; no new process group or ordinary EPContext.
        """
        mesh = binding.mesh_binding.mesh_context
        if mesh.ep_size <= 1:
            raise ValueError("MegaMoE requires EP>1; use ordinary GroupedExperts for EP=1")
        if mesh.pp_size > 1:
            raise ValueError("the existing MegaMoe lifecycle requires PP=1")
        if mesh.tp_size > 1 and not mesh.sequence_parallel:
            raise ValueError("MegaMoe with dense TP requires EP>1 and sequence_parallel=true")
        axis = binding.mesh_binding.axis_mesh("ep")
        group = axis.get_group()
        if self._parallel_bound and self._ep_group is not group:
            raise ValueError("MegaKernel already has a different communication owner")
        self._ep_group = group
        self._parallel_bound = True

    def parallel_module_spec(self) -> ParallelModuleSpec:
        """Declare canonical ownership and a fused token-level protocol."""
        return ParallelModuleSpec(
            protocol="grouped_experts.forward.mega", binding_kind="mesh",
            stages={"forward": ("hidden_states", "topk_ids", "topk_weights")}, validate=_validate_mega_layout,
        )

    def runtime_group_key(self) -> tuple:
        """Group only serial native executors with compatible scratch requirements."""
        config = self.config
        if not self._parallel_bound:
            raise RuntimeError("bind_parallel must run before preparing MegaKernel runtimes")
        return (type(self), self._ep_group, config, self.gate_up_proj.device, self.gate_up_proj.dtype,
                tuple(self._runtime.tuning.items()))

    def _bound_runtime(self) -> tuple[_MegaExpertsRuntime, Any]:
        """Return the native holder and EP group after parallel binding."""
        if not self._parallel_bound:
            raise RuntimeError("bind_parallel must run before preparing MegaKernel runtimes")
        return self._runtime, self._ep_group

    @classmethod
    def prepare_runtime_group(cls, experts: list["EPMegaGroupedExperts"]) -> None:
        """Configure after materialization and share scratch across compatible serial layers.

        Args:
            experts: Peers with the same runtime_group_key, before their first forward.
        """
        peers = [cls._bound_runtime(expert) for expert in experts]
        if all(runtime.executor is not None for runtime, _ in peers):
            return
        created = []
        try:
            for runtime, group in peers:
                if runtime.executor is None:
                    runtime.configure(group)
                    created.append(runtime)
            _MegaExpertsRuntime.share([runtime for runtime, _ in peers])
        except Exception:
            for runtime in created:
                runtime.close()
            raise

    def close_runtime(self) -> None:
        """Close this native executor after its final backward."""
        self._runtime.close()

    def forward(self, hidden_states: Tensor, topk_ids: Tensor | None = None,
                topk_weights: Tensor | None = None, **kwargs: Tensor) -> Tensor:
        """Keep original token routes and current FSDP-restored parameters in one fused call.

        Args:
            hidden_states: Source token activations [..., H].
            topk_ids: Global expert IDs [T, K], where T is the flattened token count.
            topk_weights: Final model-selected routing probabilities [T, K].
            **kwargs: Recognized aliases for the top-k ID and probability inputs.

        Returns:
            Expert contributions restored to the original hidden_states shape.
        """
        topk_ids, topk_weights = match_grouped_experts_inputs(topk_ids, topk_weights, kwargs)
        self.validate_and_flatten_inputs(hidden_states, topk_ids, topk_weights)
        if topk_ids.shape[1] != self.config.top_k:
            raise ValueError("MegaKernel top-k width must match its configured native executor")
        parameters = tuple(value.to_local() if isinstance(value, DTensor) else value
                           for value in (self.gate_up_proj, self.down_proj))
        return self._runtime(hidden_states, topk_ids, topk_weights, *parameters)

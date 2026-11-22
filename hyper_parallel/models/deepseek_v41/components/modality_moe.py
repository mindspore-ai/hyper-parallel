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
"""Complete modality-aware MoE composed from parameter-owning public expert families."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from types import MethodType, SimpleNamespace
from typing import Any, Callable

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.checkpoint import checkpoint

from hyper_parallel.components.modules import GroupedExperts, SharedExpert
from hyper_parallel.components.modules.common.swiglu_mlp import swiglu_activation
from hyper_parallel.components.modules.engram import local_token_mask
from hyper_parallel.components.modules.engram.parallel_engram import engram_ep_compute_fn
from hyper_parallel.components.modules.module_family import ModuleFamily, build_family, register_module_family
from hyper_parallel.components.modules.moe.grouped_experts import GroupedExpertsConfig
from hyper_parallel.components.modules.moe.grouped_experts.ep.ep_mega_grouped_experts import EPMegaGroupedExperts
from hyper_parallel.distributed.expert_parallel.experts import (
    bind_local_expert_forward,
    ep_routed_forward,
    require_attrs,
)
from hyper_parallel.distributed.recipe_spec import local_compute
from hyper_parallel.models.deepseek_v41.components.modality_router import (
    ModalityTopKRouter,
)
from hyper_parallel.models.deepseek_v41.components.modality_router import (
    configure_router_statistics as _configure_router_statistics,
)
from hyper_parallel.models.deepseek_v41.components.packed_mega_experts import (
    DeepseekV41MegaMoeExperts,
    _swiglu_limit,
)
from hyper_parallel.models.replacement import module_replacement

# Keep the finite historical recipe symbol without maintaining a second Engram implementation.
deepseek_v41_engram_compute_fn = engram_ep_compute_fn


class ModalityMoEModule(nn.Module):
    """Own modality routing and routed/shared composition without forward-method installation."""

    def __init__(
        self,
        *,
        config: Any = None,
        module: nn.Module | None = None,
        module_fqn: str = "",
        context: Any = None,
        use_grouped_gemm: bool = False,
        recompute_shared: bool = False,
        activation_dtype: str = "float32",
    ) -> None:
        """Construct native matrices or adopt canonical V4.1 children with their source keys."""
        super().__init__()
        del module_fqn
        if (config is None) == (module is None):
            raise ValueError("provide exactly one of config or module")
        if module is not None:
            config = getattr(module, "config", None)
            if config is None:
                configs = (context or {}).get("module_configs", ())
                config = next((item for item in configs if hasattr(item, "swiglu_limit")), None)
            if config is None:
                raise ValueError("V4.1 MoE source adoption requires its architecture config")
        if activation_dtype not in ("input", "float32"):
            raise ValueError("V4.1 expert activation_dtype must be input or float32")
        self.activation_dtype = activation_dtype
        self.config = config
        self.is_hash = False
        if not isinstance(recompute_shared, bool):
            raise ValueError("recompute_shared must be bool")
        self.recompute_shared = recompute_shared
        self.use_grouped_gemm = use_grouped_gemm
        self.initializer_range = float(config.initializer_range)
        limit = float(config.swiglu_limit)
        if module is None:
            self.gate = ModalityTopKRouter(config=config, context=context)
            self.experts = GroupedExperts(
                config=GroupedExpertsConfig(
                    num_experts=config.num_local_experts,
                    top_k=config.num_experts_per_tok,
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    num_layers=config.num_hidden_layers,
                    routing_weight_mode="output",
                    activation_dtype=activation_dtype,
                    swiglu_limit=limit,
                    init_method_std=self.initializer_range,
                ),
                context=context,
                use_grouped_gemm=use_grouped_gemm,
            )
            shared_config = SimpleNamespace(
                hidden_size=config.hidden_size,
                num_hidden_layers=config.num_hidden_layers,
                initializer_range=config.initializer_range,
                init_method_std=config.initializer_range,
                intermediate_size=config.intermediate_size,
                hidden_act="silu",
                bias=False,
                return_bias_tuple=False,
                swiglu_limit=limit,
            )
            self.shared_experts = SharedExpert(
                config=shared_config, context=context, preserve_source_layout=True,
                activation_dtype=activation_dtype, swiglu_limit=limit
            )
            self.reset_parameters()
        else:
            if set(dict(module.named_children())) != {"gate", "experts", "shared_experts"}:
                raise ValueError("V4.1 MoE source must own exactly router, routed experts and shared experts")
            self.gate = ModalityTopKRouter(module=module.gate, context=context)
            self.experts = GroupedExperts(
                module=module.experts,
                context=context,
                use_grouped_gemm=use_grouped_gemm,
                activation_dtype=activation_dtype,
                swiglu_limit=limit,
            )
            self.shared_experts = SharedExpert(
                module=module.shared_experts,
                context=context,
                preserve_source_layout=True,
                activation_dtype=activation_dtype,
                swiglu_limit=limit,
            )
            self.training = module.training

    def reset_parameters(self) -> None:
        """Retain the V4.1 normal initialization for both packed expert projections."""
        with torch.no_grad():
            nn.init.normal_(self.experts.gate_up_proj, std=self.initializer_range)
            nn.init.normal_(self.experts.down_proj, std=self.initializer_range)
            for parameter in self.shared_experts.parameters():
                nn.init.normal_(parameter, std=self.initializer_range)
            self.gate.reset_parameters()

    def make_transforms(self) -> list:
        """Let child families own their canonical packed and separate projection keys."""
        return []

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
        router_token_mask: torch.Tensor | None = None,
        router_sequence_ids: torch.Tensor | None = None,
        router_num_sequences: int | None = None,
    ) -> torch.Tensor:
        """Route text/image tokens and add the shared branch using complete child family calls."""
        del input_ids
        _, weights, indices = self.gate(
            hidden_states,
            image_mask=image_mask,
            token_mask=router_token_mask,
            sequence_ids=router_sequence_ids,
            num_sequences=router_num_sequences,
        )
        routed = self.experts(hidden_states.reshape(-1, hidden_states.shape[-1]), indices, weights)
        shared = (
            checkpoint(self.shared_experts, hidden_states, use_reentrant=False)
            if self.recompute_shared and self.training and torch.is_grad_enabled()
            else self.shared_experts(hidden_states)
        )
        return routed.reshape_as(hidden_states) + shared


def _validate(*, config: Any, module: nn.Module | None, context: Any, member: Any) -> None:
    """Compose parallel child families only for declared token ownership and supported pipeline scope."""
    del config, module
    if member.axis_sizes["pp"] > 1:
        raise NotImplementedError("V4.1 modality MoE currently requires PP=1")
    if member.axis_sizes["tp"] > 1 and (member.axis_sizes["ep"] == 1 or not context.get("sequence_parallel", False)):
        raise ValueError("V4.1 TP requires sequence parallelism and TP-extend-EP with EP>1")


@module_replacement
def ModalityMoE(  # pylint: disable=invalid-name
    *,
    config: Any = None,
    module: nn.Module | None = None,
    module_fqn: str = "",
    context: Any = None,
    mesh: Any = None,
    use_grouped_gemm: bool = False,
    recompute_shared: bool = False,
    activation_dtype: str = "float32",
) -> nn.Module:
    """Construct a complete model-owned MoE; child families own parameter layouts and optimizer matrices."""
    return build_family(
        ModalityMoE,
        config=config,
        module=module,
        module_fqn=module_fqn,
        context=context,
        mesh=mesh,
        use_grouped_gemm=use_grouped_gemm,
        recompute_shared=recompute_shared,
        activation_dtype=activation_dtype,
    )


register_module_family(
    ModuleFamily(
        name="deepseek_v41_modality_moe",
        baseline=ModalityMoE,
        variants={"local": "hyper_parallel.models.deepseek_v41.components.modality_moe.ModalityMoEModule"},
        default_variant="local",
        validate=_validate,
    )
)


# Original recipe method targets remain explicit compatibility entries. New
# composition calls class-owned router and Experts families directly.
def _deepseek_v41_unclamped_gate(gate_up: torch.Tensor) -> torch.Tensor:
    """Apply native FP32 SwiGLU when a nonpositive limit disables clamping."""
    gate, up = gate_up.float().chunk(2, dim=-1)
    return swiglu_activation(gate, up, limit=0.0, activation_dtype="float32").to(gate_up.dtype)


def _deepseek_v41_grouped_expert_forward(
        experts: torch.nn.Module,
        hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    """Run FP32 clamped SwiGLU between two grouped GEMMs.

    The generic EP path has already sorted ``hidden_states`` into expert-major
    order. Routed and shared experts reuse the native FP32 clamp and SwiGLU
    implementation. Nonpositive limits disable clamping; the down projection
    still receives the original hidden-state dtype.
    """
    # The shared primitive imports optional torch_npu; keep CPU/GPU imports lazy.
    from hyper_parallel.components.functional import grouped_matmul  # pylint: disable=C0415

    group_list = torch.cumsum(
        tokens_per_expert.to(device=hidden_states.device, dtype=torch.int64),
        dim=0,
    )
    gate_up = grouped_matmul(  # pylint: disable=not-callable
        hidden_states,
        experts.gate_up_proj.transpose(1, 2),
        bias=None,
        group_list=group_list,
        group_type=0,
        group_list_type=0,
    )
    # Cast the packed projections once; the shared helper's casts are then no-ops.
    gate, up = gate_up.float().chunk(2, dim=-1)
    intermediate = swiglu_activation(gate, up, limit=experts.limit, activation_dtype="float32")
    return grouped_matmul(  # pylint: disable=not-callable
        intermediate.to(hidden_states.dtype),
        experts.down_proj.transpose(1, 2),
        bias=None,
        group_list=group_list,
        group_type=0,
        group_list_type=0,
    )


def _local_router_metadata(tensor: torch.Tensor | None, states: torch.Tensor, tp_mesh: Any) -> torch.Tensor | None:
    """Slice CP-local metadata when an EP router keeps the TP sequence shard."""
    if tensor is None or tensor.shape == states.shape[:-1]:
        return tensor
    local_length = states.shape[-2]
    if tp_mesh is None or tensor.shape[-1] != local_length * tp_mesh.size():
        raise ValueError("router metadata must match the local or TP-gathered token dimensions")
    return tensor.narrow(-1, tp_mesh.get_local_rank() * local_length, local_length)


@module_replacement
def replace_deepseek_v41_grouped_experts(
        *, module: torch.nn.Module, module_fqn: str, context: Mapping[str, Any],
        moonep: bool = False, moonep_replica_capacity: int | None = None,
) -> torch.nn.Module:
    """Adopt the packed V4.1 projections with native FP32 clamped SwiGLU.

    Args:
        module: Original routed experts with the packed Transformers layout.
        module_fqn: Path of the routed experts in the model.
        context: Builder context for selecting the local or EP member.
        moonep: Use collective MoonEP while preserving the model-owned activation policy.
        moonep_replica_capacity: Maximum distinct remote experts per execution rank.

    Returns:
        GroupedExperts family member preserving the source parameters.

    Note:
        The source uses the Transformers 5.11+ packed expert contract. Only
        the activation differs: clamp and SwiGLU use the model's FP32 helper.
        Parameter packing, router decisions and shared experts are unchanged.
    """
    return GroupedExperts(
        module=module, module_fqn=module_fqn, context=context,
        activation_dtype="float32", swiglu_limit=module.limit,
        moonep=moonep, moonep_replica_capacity=moonep_replica_capacity,
    )


def _router(
        module: Any,
        hidden_states: torch.Tensor,
        image_mask: torch.Tensor | None = None,
        router_token_mask: torch.Tensor | None = None,
        router_sequence_ids: torch.Tensor | None = None,
        router_num_sequences: int | None = None,
        sequence_partition_groups: tuple[Any, ...] = (),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the V4.1 learned router's selected experts and weights."""
    output = module.gate(
        hidden_states, image_mask=image_mask, token_mask=router_token_mask,
        sequence_ids=router_sequence_ids, num_sequences=router_num_sequences,
        sequence_partition_groups=sequence_partition_groups,
    )
    if not isinstance(output, (tuple, list)) or len(output) != 3:
        raise TypeError("DeepSeek-V4.1 gate must return logits, weights, and indices")
    _, weights, indices = output
    return indices, weights


def _routed_and_shared_forward(
        module: Any,
        hidden_states: torch.Tensor,
        image_mask: torch.Tensor | None,
        ep_group: Any,
        router_token_mask: torch.Tensor | None = None,
        router_sequence_ids: torch.Tensor | None = None,
        router_num_sequences: int | None = None,
        sequence_partition_groups: tuple[Any, ...] = (),
        *,
        ep_size: int,
        fix_router: bool = False,
) -> torch.Tensor:
    """Dispatch routed experts and add the V4.1 shared expert branch."""

    def router_fn(target: Any, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Route with statistics over the current token partitions.

        Args:
            target: MoE block exposing the learned router.
            states: Token activations on the current partition.

        Returns:
            Selected expert indices and differentiable routing weights.
        """
        return _router(
            target, states, image_mask, router_token_mask, router_sequence_ids,
            router_num_sequences, sequence_partition_groups=sequence_partition_groups,
        )

    routed = ep_routed_forward(
        module, hidden_states,
        router_fn=_resolve_router(router_fn, ep_size=ep_size, fix_router=fix_router), ep_group=ep_group,
    )
    return routed + module.shared_experts(hidden_states)


def _resolve_router(router_fn: Callable, ep_size: int, fix_router: bool = False) -> Callable:
    """Keep learned gate work and optionally balance destinations for benchmarks.

    Only dispatch indices change; weights keep their original autograd graph.
    Fixed assignments ignore gate scores and must only be used for performance tests.
    Both native EP and MegaMoe supply the size of their expert-parallel mesh.
    """
    if not fix_router:
        return router_fn

    def balanced_router(module: Any, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Spread slots across destination ranks before cycling their experts.

        Args:
            module: MoE block exposing the learned gate.
            hidden_states: Local token activations passed to the learned router.

        Returns:
            Balanced expert indices and unchanged learned routing weights.
        """
        indices, weights = router_fn(module, hidden_states)
        expert_count = module.gate.num_experts
        if expert_count <= 0 or ep_size <= 0 or expert_count % ep_size != 0:
            raise ValueError("The positive global expert count must be divisible by the positive EP size")
        local_count = expert_count // ep_size
        slots = torch.arange(indices.numel(), device=indices.device, dtype=torch.long)
        balanced = (slots % ep_size) * local_count + (slots // ep_size) % local_count
        return balanced.view_as(indices).to(indices.dtype), weights

    return balanced_router


@local_compute
def deepseek_v41_ep_compute_fn(
        *,
        module: Any,
        mesh: Any,
        tp_mesh: Any,
        cp_mesh: Any,
        ep_mesh: Any,
        use_grouped_gemm: bool = False,
        fix_router: bool = False,
) -> Callable:
    """Build V4.1 routed/shared computation with training-aware routing.

    Sequence statistics reduce CP/TP token partitions; bias updates also reduce
    DP axes. Fixed routing changes dispatch indices after the gate, so enabled
    auxiliary statistics still describe learned decisions. Pure throughput runs
    can disable both router coefficients.

    Args:
        module: V4.1 MoE block exposing gate, experts, and shared_experts.
        mesh: Region mesh whose root supplies data-parallel bias statistics groups.
        tp_mesh: Tensor-parallel mesh used to align router metadata and statistics.
        cp_mesh: Context-parallel mesh supplying token-partition statistics groups.
        ep_mesh: Expert-parallel mesh owning the routed experts.
        use_grouped_gemm: Use the existing V4.1 clamped grouped-GEMM expert path.
        fix_router: Enable benchmark-only balanced destinations while retaining
            learned Top-K weights and their gradients. Normal training and
            correctness runs keep this disabled.

    Returns:
        The source-compatible local compute callable.
    """
    require_attrs(module, "gate", "experts", "shared_experts", owner="DeepSeek-V4.1 EP")
    if ep_mesh is None:
        raise ValueError("DeepSeek-V4.1 EP forward requires an active ep_mesh")
    if module.is_hash:
        raise ValueError("DeepSeek-V4.1 validation layers must use learned routing")
    if not isinstance(fix_router, bool):
        raise ValueError("fix_router must be a boolean")
    sequence_groups = _configure_router_statistics(module.gate, mesh, tp_mesh, cp_mesh)
    ep_group = ep_mesh.get_group("ep")
    ep_size = ep_mesh["ep"].size()
    tp_size = 1 if tp_mesh is None else tp_mesh.size()
    tp_rank = 0 if tp_mesh is None else tp_mesh.get_local_rank()
    apply_gate = (
        module.experts._apply_gate  # pylint: disable=protected-access
        if module.experts.limit > 0 else _deepseek_v41_unclamped_gate
    )
    if use_grouped_gemm:
        module.experts.forward_expert_major = MethodType(
            _deepseek_v41_grouped_expert_forward, module.experts,
        )
    bind_local_expert_forward(
        module, ep_size, use_grouped_gemm=use_grouped_gemm,
        apply_gate=None if use_grouped_gemm else apply_gate,
    )

    def auxiliary_compute_fn(
            module: Any,
            hidden_states: torch.Tensor,
            input_ids: torch.Tensor | None = None,
            image_mask: torch.Tensor | None = None,
            router_token_mask: torch.Tensor | None = None,
            router_sequence_ids: torch.Tensor | None = None,
            router_num_sequences: int | None = None,
    ) -> torch.Tensor:
        """Run text and image routing through the same dispatch/combine path.

        Args:
            module: MLP owning routed and shared experts.
            hidden_states: Token activations before expert dispatch.
            input_ids: Unused by learned routing.
            image_mask: Optional image-token mask.
            router_token_mask: Optional valid-token mask.
            router_sequence_ids: Optional packed logical sample IDs.
            router_num_sequences: Global sample count within this microbatch.
        """
        del input_ids
        return _routed_and_shared_forward(
            module, hidden_states, _local_router_metadata(image_mask, hidden_states, tp_mesh), ep_group,
            router_token_mask=_local_router_metadata(router_token_mask, hidden_states, tp_mesh),
            router_sequence_ids=_local_router_metadata(router_sequence_ids, hidden_states, tp_mesh),
            router_num_sequences=router_num_sequences, sequence_partition_groups=sequence_groups,
            ep_size=ep_size, fix_router=fix_router,
        )

    if "router_token_mask" in inspect.signature(module.forward).parameters:
        return auxiliary_compute_fn

    if "image_mask" not in inspect.signature(module.forward).parameters:
        def text_compute_fn(
                module: Any,
                hidden_states: torch.Tensor,
                input_ids: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """Run the text-only source contract without visual routing state.

            Args:
                module: Routed and shared expert module.
                hidden_states: Current decoder hidden rows.
                input_ids: Unused token IDs from the local compute contract.
            """
            del input_ids
            return _routed_and_shared_forward(
                module, hidden_states, None, ep_group, ep_size=ep_size, fix_router=fix_router,
            )

        return text_compute_fn

    def multimodal_compute_fn(
            module: Any,
            hidden_states: torch.Tensor,
            input_ids: torch.Tensor | None = None,
            image_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the multimodal source contract with optional visual routing state.

        Args:
            module: Routed and shared expert module.
            hidden_states: TP-local decoder hidden rows.
            input_ids: Unused token IDs from the local compute contract.
            image_mask: CP-local mask identifying image tokens.
        """
        del input_ids
        local_image_mask = local_token_mask(
            image_mask, hidden_states, tp_rank=tp_rank, tp_size=tp_size,
        )
        return _routed_and_shared_forward(
            module, hidden_states, local_image_mask, ep_group, ep_size=ep_size, fix_router=fix_router,
        )

    return multimodal_compute_fn


def _validate_megamoe_composition(module: Any, tp_size: int, ep_size: int) -> bool:
    """Validate shared/routed semantics before selecting canonical or historical runtime ownership."""
    canonical_mega = isinstance(module.experts, EPMegaGroupedExperts)
    if not isinstance(module.experts, (DeepseekV41MegaMoeExperts, EPMegaGroupedExperts)):
        raise TypeError("MegaMoe compute requires the DeepseekV41MegaMoeExperts module replacement")
    expert_limit = module.experts.config.swiglu_limit if canonical_mega else module.experts.swiglu_limit
    if _swiglu_limit(module.shared_experts.limit) != expert_limit:
        raise ValueError("Routed and shared experts must have the same swiglu_limit")
    if tp_size > 1:
        if ep_size <= 1:
            raise ValueError("DSV4.1 MegaMoe with TP>1 requires EP>1")
        if not canonical_mega and not module.experts.sequence_parallel:
            raise ValueError("DSV4.1 MegaMoe with TP>1 requires sequence_parallel=true")
    return canonical_mega


@local_compute
def deepseek_v41_megamoe_compute_fn(
        *, module: Any, mesh: Any, tp_mesh: Any, cp_mesh: Any, ep_mesh: Any,
        fix_router: bool = False,
) -> Callable:
    """Bind a declaratively replaced expert module to the full EP mesh.

    Router statistics reduce CP/TP token partitions; bias updates also reduce
    DP axes. Parameter sharding and FSDP remain framework-owned.

    Args:
        module: V4.1 MoE block with the MegaMoe expert replacement.
        mesh: Region mesh whose root supplies data-parallel bias statistics groups.
        tp_mesh: Tensor-parallel mesh used to align router metadata and statistics.
        cp_mesh: Context-parallel mesh supplying token-partition statistics groups.
        ep_mesh: Expert-parallel mesh owning the routed experts, or None for EP=1.
        fix_router: Enable benchmark-only balanced destinations while preserving
            learned weights and gradients. Keep false for normal training.
            Enabled auxiliary statistics still describe learned gate decisions;
            pure throughput runs can disable both router coefficients.

    Returns:
        The local compute callable matching the source forward signature.
    """
    require_attrs(module, "gate", "experts", "shared_experts", owner="DeepSeek-V4.1 MegaMoe")
    if module.is_hash:
        raise ValueError("DeepSeek-V4.1 MegaMoe requires learned routing")
    if not isinstance(fix_router, bool):
        raise ValueError("fix_router must be a boolean")
    ep_group = None if ep_mesh is None else ep_mesh.get_group("ep")
    ep_size = 1 if ep_mesh is None else ep_mesh["ep"].size()
    tp_size = 1 if tp_mesh is None else tp_mesh.size()
    canonical_mega = _validate_megamoe_composition(module, tp_size, ep_size)
    tp_rank = 0 if tp_mesh is None else tp_mesh.get_local_rank()
    if dist.is_initialized() and dist.get_world_size(ep_group) != ep_size:
        raise ValueError("MegaMoe EP mesh and process group sizes must match")
    sequence_groups = _configure_router_statistics(module.gate, mesh, tp_mesh, cp_mesh)
    if not canonical_mega:
        module.experts.configure(ep_group, module.gate.top_k)

    def compute_fn(
            module: Any, hidden_states: torch.Tensor,
            input_ids: torch.Tensor | None = None, image_mask: torch.Tensor | None = None,
            router_token_mask: torch.Tensor | None = None,
            router_sequence_ids: torch.Tensor | None = None,
            router_num_sequences: int | None = None,
    ) -> torch.Tensor:
        """Run training-aware routing and shared experts around the MegaMoe call.

        Args:
            module: MLP owning the router and experts.
            hidden_states: Token activations before expert dispatch.
            input_ids: Unused by learned routing.
            image_mask: Optional CP-local image-token mask.
            router_token_mask: Optional CP-local valid-token mask.
            router_sequence_ids: Optional CP-local packed sample IDs.
            router_num_sequences: Total logical sample count in the microbatch.
        """
        del input_ids
        local_image_mask = local_token_mask(
            image_mask, hidden_states, tp_rank=tp_rank, tp_size=tp_size,
        )
        if local_image_mask is not None:
            # Slicing multiple batch rows leaves gaps, which native MegaGate rejects.
            local_image_mask = local_image_mask.contiguous()

        def router_fn(target: Any, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Keep training metadata attached to the learned gate before balancing.

            Args:
                target: MoE block exposing the learned router.
                states: Token activations on the current partition.

            Returns:
                Selected expert indices and differentiable routing weights.
            """
            return _router(
                target, states, local_image_mask,
                router_token_mask=_local_router_metadata(router_token_mask, states, tp_mesh),
                router_sequence_ids=_local_router_metadata(router_sequence_ids, states, tp_mesh),
                router_num_sequences=router_num_sequences, sequence_partition_groups=sequence_groups,
            )

        indices, weights = _resolve_router(router_fn, fix_router=fix_router, ep_size=ep_size)(module, hidden_states)
        return module.experts(hidden_states, indices, weights) + module.shared_experts(hidden_states)

    if "image_mask" in inspect.signature(module.forward).parameters:
        return compute_fn

    def text_compute_fn(
            module: Any,
            hidden_states: torch.Tensor,
            input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Match the text-only source forward without visual routing arguments.

        Args:
            module: Routed and shared expert module.
            hidden_states: Current decoder hidden rows.
            input_ids: Unused token IDs from the local compute contract.
        """
        return compute_fn(module, hidden_states, input_ids)

    return text_compute_fn

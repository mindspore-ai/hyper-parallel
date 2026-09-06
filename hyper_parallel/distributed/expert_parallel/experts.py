# Copyright 2025-2026 Huawei Technologies Co., Ltd
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

"""expert_parallel.experts: EP expert execution and binding.

- Routed-experts pipeline (``ep_routed_forward``, 05 §6.4.8): router (local
  chunk) -> a2a dispatch (extended EP group) -> local SwiGLU (complete
  expert weights, no internal communication) -> a2a combine -> weighted
  aggregation. SP-in -> SP-out, all communication cohesive inside. NO
  shared-expert/gate branch — that composition is the caller's job.
- Expert entry point (``bind_local_expert_forward`` /
  ``resolve_swiglu_weights``): installs ``experts.forward`` (via the forward
  rewriter's bound-forward install point, 05 §15.2.3) so nested FSDP hooks
  unshard/reshard around the local SwiGLU.
- Interface helpers (``require_attrs`` / ``describe_moe_module``):
  build-time interface assertions with teaching errors, and a structural
  diagnostic for mapping a concrete MoE module to an archetype.

Nothing in this module probes model structure with getattr fallback chains.
Split out of components/distributed/ep_utils.py in stage 4e.
"""

from typing import Any, Callable, Optional
import torch
import torch.distributed as dist
import torch.nn.functional as F
from hyper_parallel.components.functional.npu_grouped_swiglu import (
    npu_grouped_swiglu,
)
from hyper_parallel.distributed._builder.forward_rewriter import (
    _install_bound_forward,
)
from hyper_parallel.distributed.expert_parallel.collectives import (
    ep_all_to_all,
)


def resolve_swiglu_weights(
    experts: Any,
) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Resolve SwiGLU weights from the stacked holder (three layouts).

    Returns (w_gate, w_up, w_down):
    - Separate naming: gate_proj/up_proj/down_proj or w1/w3/w2 (all
      [E, I, H]/[E, H, I]);
    - **fused layout** (after the HF 2025 refactor, D-11): gate_up_proj
      [E, 2I, H] + down_proj [E, H, I] -> returns (gate_up_proj, None,
      down_proj); w_up=None marks the fused case (the compute side chunks
      out gate/up).
    """
    w_fused = getattr(experts, "gate_up_proj", None)
    if w_fused is None:
        w_fused = getattr(experts, "gate_and_up_projs", None)
    w_down = getattr(experts, "down_proj", None)
    w_down = w_down if w_down is not None else getattr(experts, "down_projs", None)
    w_down = w_down if w_down is not None else getattr(experts, "w2", None)
    if w_fused is not None and w_down is not None:
        return w_fused, None, w_down

    w_gate = getattr(experts, "gate_proj", None)
    w_gate = w_gate if w_gate is not None else getattr(experts, "w1", None)
    w_up = getattr(experts, "up_proj", None)
    w_up = w_up if w_up is not None else getattr(experts, "w3", None)
    if w_gate is None or w_up is None or w_down is None:
        raise NotImplementedError(
            f"{type(experts).__name__}: only SwiGLU experts are supported "
            "(fused gate_up_proj, or separate gate_proj/up_proj/down_proj, "
            "w1/w2/w3 three-matrix layouts); use an EP-aware MoE module instead"
        )
    return w_gate, w_up, w_down


def _local_swiglu_expert_forward(experts, dispatched_states, local_expert_indices):
    """Compute dispatched tokens with the local stacked SwiGLU experts.

    This function is installed as ``experts.forward`` for the HF-native EP
    path. Calling the expert module, instead of indexing its parameters from
    the parent MoE forward, allows nested FSDP forward hooks to unshard and
    reshard expert parameters around the local computation.
    """
    token_order = local_expert_indices.argsort()
    sorted_states = dispatched_states[token_order]
    local_expert_counts = torch.bincount(
        local_expert_indices,
        minlength=experts.local_expert_count,
    )
    if getattr(experts, "_ep_use_grouped_gemm", False):
        grouped_forward = getattr(experts, "forward_expert_major", None)
        if callable(grouped_forward):
            sorted_output = grouped_forward(sorted_states, local_expert_counts)
        else:
            gate_weight, up_weight, down_weight = resolve_swiglu_weights(experts)
            if up_weight is not None:
                raise ValueError(
                    "EP grouped GEMM currently requires packed gate_up_proj weights"
                )
            sorted_output = npu_grouped_swiglu(
                sorted_states,
                gate_weight,
                down_weight,
                local_expert_counts,
            )
        output = torch.empty_like(sorted_output)
        output[token_order] = sorted_output
        return output

    gate_weight, up_weight, down_weight = resolve_swiglu_weights(experts)
    sorted_outputs = []
    token_start = 0
    for local_expert_index in range(experts.local_expert_count):
        expert_token_count = int(local_expert_counts[local_expert_index])
        expert_states = sorted_states[token_start:token_start + expert_token_count]
        if up_weight is None:
            gate_states, up_states = F.linear(  # pylint: disable=not-callable
                expert_states,
                gate_weight[local_expert_index],
            ).chunk(2, dim=-1)
        else:
            gate_states = F.linear(  # pylint: disable=not-callable
                expert_states, gate_weight[local_expert_index]
            )
            up_states = F.linear(  # pylint: disable=not-callable
                expert_states, up_weight[local_expert_index]
            )
        activation = getattr(experts, "_ep_act_fn", F.silu)
        sorted_outputs.append(
            F.linear(  # pylint: disable=not-callable
                activation(gate_states) * up_states,
                down_weight[local_expert_index],
            )
        )
        token_start += expert_token_count

    sorted_output = torch.cat(sorted_outputs)
    output = torch.empty_like(sorted_output)
    output[token_order] = sorted_output
    return output


def _get_global_expert_count(module):
    """Return the model-level routed expert count for an MoE module."""
    if hasattr(module.experts, "num_experts"):
        return module.experts.num_experts
    if hasattr(module, "num_experts"):
        return module.num_experts
    if hasattr(module, "config") and hasattr(module.config, "num_experts"):
        return module.config.num_experts
    if hasattr(module, "config") and hasattr(module.config, "n_routed_experts"):
        return module.config.n_routed_experts
    raise ValueError(
        f"{type(module).__name__}: cannot determine the global routed expert count"
    )


def bind_local_expert_forward(
    module: Any,
    ep_size: int,
    use_grouped_gemm: bool = False,
) -> None:
    """Install the local expert compute entry used by TP-extend-EP.

    Called by the EP compute factory (archetype or user-written) at apply
    time: sets ``module.experts.local_expert_count`` and installs
    ``experts.forward`` (via the forward rewriter's bound-forward install
    point) so nested FSDP hooks unshard/reshard around the local SwiGLU
    computation.
    """
    global_expert_count = _get_global_expert_count(module)
    if global_expert_count % ep_size != 0:
        raise ValueError(
            f"num_experts ({global_expert_count}) must be divisible by ep_size ({ep_size})"
        )
    module.experts.local_expert_count = global_expert_count // ep_size
    activation = getattr(module.experts, "act_fn", None)
    if activation is None:
        hidden_act = getattr(getattr(module, "config", None), "hidden_act", "silu")
        activation = {
            "gelu": F.gelu,
            "relu": F.relu,
            "silu": F.silu,
            "swish": F.silu,
        }.get(hidden_act)
        if activation is None:
            raise ValueError(
                f"{type(module).__name__}: unsupported expert activation {hidden_act!r}; "
                "provide experts.act_fn or extend the EP activation registry"
            )
    module.experts._ep_act_fn = activation
    module.experts._ep_use_grouped_gemm = use_grouped_gemm
    # The forward write itself lives in the forward rewriter (05 §15.2.3:
    # the single MethodType/assignment site); this binder only sets the
    # companion attributes above.
    _install_bound_forward(module.experts, _local_swiglu_expert_forward)


def _prepare_ep_dispatch(
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    local_expert_count: int,
    global_expert_count: int,
    ep_size: int,
    ep_group: Any,
):
    """Sort routed tokens and exchange per-rank dispatch counts."""
    flattened_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    token_count = flattened_states.shape[0]
    experts_per_token = topk_indices.shape[1]
    expert_indices = topk_indices.reshape(-1)
    expert_weights = topk_weights.reshape(-1).to(flattened_states.dtype)
    source_indices = torch.arange(token_count, device=flattened_states.device).repeat_interleave(experts_per_token)
    destination_ranks = torch.div(expert_indices, local_expert_count, rounding_mode="floor")
    dispatch_order = (destination_ranks * global_expert_count + expert_indices).argsort()
    dispatched_states = flattened_states[source_indices[dispatch_order]].contiguous()
    dispatched_indices = expert_indices[dispatch_order].unsqueeze(-1).contiguous()
    send_counts_tensor = torch.bincount(destination_ranks, minlength=ep_size)
    receive_counts_tensor = torch.empty_like(send_counts_tensor)
    dist.all_to_all_single(receive_counts_tensor, send_counts_tensor, group=ep_group)
    return (
        source_indices,
        expert_weights,
        dispatch_order,
        dispatched_states,
        dispatched_indices,
        send_counts_tensor.tolist(),
        receive_counts_tensor.tolist(),
    )


def _run_ep_local_experts(
    module: Any,
    dispatched_states: torch.Tensor,
    dispatched_indices: torch.Tensor,
    send_counts: list[int],
    receive_counts: list[int],
    ep_group: Any,
    expert_offset: int,
) -> torch.Tensor:
    """Dispatch tokens, run local experts, and return combined expert outputs."""
    received_states = ep_all_to_all(dispatched_states, send_counts, receive_counts, ep_group)
    received_indices = ep_all_to_all(
        dispatched_indices, send_counts, receive_counts, ep_group
    ).squeeze(-1)
    local_outputs = module.experts(received_states, received_indices - expert_offset)
    return ep_all_to_all(local_outputs.contiguous(), receive_counts, send_counts, ep_group)


def _aggregate_ep_outputs(
    combined_outputs: torch.Tensor,
    expert_weights: torch.Tensor,
    source_indices: torch.Tensor,
    dispatch_order: torch.Tensor,
    output_shape: tuple[int, int, int],
) -> torch.Tensor:
    """Undo expert-major routing and aggregate weighted top-k outputs."""
    weighted_outputs = combined_outputs * expert_weights[dispatch_order].unsqueeze(-1)
    output = torch.zeros(
        output_shape[0] * output_shape[1],
        output_shape[-1],
        dtype=weighted_outputs.dtype,
        device=weighted_outputs.device,
    )
    output.index_add_(0, source_indices[dispatch_order], weighted_outputs)
    return output.view(*output_shape)


def ep_routed_forward(
    module: Any,
    hidden_states: torch.Tensor,
    *,
    router_fn: Callable,
    ep_group: Any,
) -> torch.Tensor:
    """Routed-experts pipeline: SP-in (local chunk) -> all communication
    inside -> SP-out. **Routed branch only.**

    This primitive deliberately does NOT handle shared experts / scalar
    gates / branch merging — that composition is model semantics and belongs
    to the caller (an ep_compute.py archetype or a user-written factory,
    accuracy_fix_plan.md §3). There is no ``tp_group`` parameter: if the
    caller invokes a nested-boundary submodule (e.g. ``module.shared_expert``),
    that submodule's own boundary performs its TP communication — the
    **nested-boundary call contract**:

    1. the input is the parent local region's current logical local layout;
    2. the nested boundary exclusively owns its parameter layout and its TP
       communication (entry/exit via its own PrecompiledBoundary);
    3. the return value is already the nested boundary's out_dst logical
       layout (e.g. under SP: the complete per-token values of the local
       sequence chunk);
    4. the caller MUST NOT repeat any compensating collective
       (all-reduce / reduce-scatter / all-gather) on the returned value
       over the nested boundary's mesh.

    Communication flow (isomorphic to Megatron token_dispatcher.py
    MoEAlltoAllTokenDispatcher):
    router (local chunk, no communication) -> a2a dispatch (extended EP
    group, including TP ranks) -> local SwiGLU (complete expert weights, no
    internal communication, no Partial) -> a2a combine (returns over the
    same group) -> weighted aggregation.

    Input hidden [B, S/tp, H] (local sequence chunk, boundary identity);
    output [B, S/tp, H] (complete, boundary identity).

    ``router_fn`` is supplied BY THE CALLER (explicit choice, e.g. an entry
    of MOE_ROUTER_ADAPTERS picked by name in the factory code).

    Extended EP group = the ep axis of the derived expert mesh (flatten
    ep_size consecutive ranks: first span the TP group, then extend to
    adjacent dp/cp ranks; MindSpeed TP-extend-EP / Megatron etp=1 + ep
    homogeneous across TP). Expert weights are only Shard(0) along the
    expert dim -- each rank holds num_experts/ep_size complete experts, so
    there is no all_gather/reduce_scatter pair.
    """
    ep_size = ep_group.size()
    ep_rank = dist.get_rank(group=ep_group)
    local_expert_count = module.experts.local_expert_count
    global_expert_count = local_expert_count * ep_size
    expert_offset = ep_rank * local_expert_count

    batch_size, sequence_length, hidden_size = hidden_states.shape
    topk_indices, topk_weights = router_fn(module, hidden_states)  # [T, K]
    dispatch = _prepare_ep_dispatch(
        hidden_states,
        topk_indices,
        topk_weights,
        local_expert_count=local_expert_count,
        global_expert_count=global_expert_count,
        ep_size=ep_size,
        ep_group=ep_group,
    )
    (
        source_token_indices,
        flattened_expert_weights,
        dispatch_order,
        dispatched_states,
        dispatched_expert_indices,
        send_counts,
        receive_counts,
    ) = dispatch
    combined_expert_outputs = _run_ep_local_experts(
        module,
        dispatched_states,
        dispatched_expert_indices,
        send_counts,
        receive_counts,
        ep_group,
        expert_offset,
    )
    return _aggregate_ep_outputs(
        combined_expert_outputs,
        flattened_expert_weights,
        source_token_indices,
        dispatch_order,
        (batch_size, sequence_length, hidden_size),
    )


def require_attrs(module: Any, *names: str, owner: str = "") -> None:
    """Assert that ``module`` has every attribute in ``names``; raise a
    teaching ValueError listing the module's ACTUAL children otherwise.

    Used by EP compute factories (archetype or user-written) at apply time:
    the factory body runs ONCE at apply time (before the wrapped forward
    ever executes), so an interface mismatch fails the model build in
    seconds instead of surfacing as a runtime AttributeError at step N.
    The check is structural (the names the implementation will call exist);
    it does not prove semantic correctness — that is vouched by numeric
    verification.
    """
    missing = [n for n in names if not hasattr(module, n)]
    if not missing:
        return
    children = [n for n, _ in module.named_children()]
    who = f"{owner} " if owner else ""
    raise ValueError(
        f"{who}expects MoE module attribute(s) {missing} on "
        f"{type(module).__name__}, but they do not exist; the module's "
        f"actual children are {children}. Pick the matching EP archetype "
        f"(see the archetype table in ep_compute.py), or write your own "
        f"factory (reference: examples/distributed/ep_factories.py) — the "
        f"names your compute_fn calls on module.<child> must match the "
        f"model's actual attribute names"
    )


def describe_moe_module(module: Any) -> str:
    """Structural diagnostic for a MoE module: child submodules, direct
    parameter shapes, and expert-related attributes — the facts needed to
    pick an EP archetype or write a custom factory. Returns the report;
    also logged at INFO."""
    lines = [f"MoE module: {type(module).__name__}"]
    children = list(module.named_children())
    lines.append(f"children ({len(children)}):")
    for name, child in children:
        lines.append(f"  - {name}: {type(child).__name__}")
    direct_params = list(module.named_parameters(recurse=False))
    if direct_params:
        lines.append("direct parameters:")
        for name, p in direct_params:
            lines.append(f"  - {name}: shape={tuple(p.shape)}")
    experts = getattr(module, "experts", None)
    if experts is not None:
        lines.append(
            f"experts: {type(experts).__name__}, "
            f"local_expert_count={getattr(experts, 'local_expert_count', '<unset>')}"
        )
    report = "\n".join(lines)
    return report

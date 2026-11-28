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
"""Routed-slot transport, independent of routing and expert computation."""

from __future__ import annotations

from contextvars import ContextVar
from weakref import ref
from typing import Any

import torch
from torch import Tensor, nn
from torch.distributed import _functional_collectives

from hyper_parallel.components.ops import moe_token_permute, moe_token_unpermute
from hyper_parallel.distributed.expert_parallel.collectives import (
    ep_all_to_all, ep_all_to_all_async, ep_exchange_counts, ep_split_sizes,
)
from hyper_parallel.core.activation_memory.recompute_state import get_recompute_state


def _validate_slot_inputs(hidden: Tensor, ids: Tensor, weights: Tensor | None) -> None:
    """Validate resource-free source slots before entering any collective."""
    if hidden.ndim != 2 or hidden.shape[1] < 1 or not hidden.is_floating_point():
        raise ValueError("slot hidden_states must be floating-point [slots,H] with H>0")
    if ids.shape != (hidden.shape[0],) or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("expert_ids must have integer dtype and shape [slots]")
    if ids.device != hidden.device:
        raise ValueError("slot activations and expert IDs must share the same device")
    if weights is not None:
        if weights.shape != ids.shape or not weights.is_floating_point() or weights.device != hidden.device:
            raise ValueError("slot routing_weights must be floating-point [slots] on the activation device")


class EPContext:
    """Contiguous expert ownership, slot transport and optional shared-expert overlap."""

    def __init__(self, mesh: Any, *, num_experts: int, overlap_shared_expert: bool = True) -> None:
        """Borrow the EP mesh; never create a group or reduce parameter gradients.

        Args:
            mesh: Existing one-dimensional expert ownership mesh.
            num_experts: Global expert count divisible by the mesh size.
            overlap_shared_expert: Schedule bound shared stages during token exchange.
        """
        if mesh is None or mesh.ndim != 1:
            raise ValueError("EPContext requires an existing one-dimensional HP mesh")
        if num_experts < 1 or num_experts % mesh.size():
            raise ValueError("num_experts must be positive and divisible by EP size")
        if not isinstance(overlap_shared_expert, bool):
            raise ValueError("overlap_shared_expert must be bool")
        self.overlap_shared_expert = overlap_shared_expert
        self._shared_expert = None
        self._shared_state: ContextVar | None = None
        self.mesh = mesh
        self.num_experts = num_experts
        self.local_experts = num_experts // mesh.size()
        self.rank = mesh.get_local_rank()

    def _exchange(self, tensor: Tensor, send: list[int], receive: list[int], *, async_op: bool = False) -> Tensor:
        """Share one synchronous/lazy exchange entry between dispatch and combine."""
        if self.mesh.size() == 1:
            return tensor
        if async_op:
            return ep_all_to_all_async(tensor, send, receive, self.mesh.get_group())
        return ep_all_to_all(tensor, send, receive, self.mesh.get_group())

    def bind_shared_expert(self, owner: nn.Module, shared: nn.Module, experts: nn.Module) -> None:
        """Bind shared stages without changing the containing MoE's forward.

        Args:
            owner: MoE whose first activation input also feeds its shared expert.
            shared: SharedExpert member with the existing FC1/FC2 module-call protocol.
            experts: Routed expert member whose checkpoint replay needs the same shared stages.
        """
        if self._shared_state is not None:
            raise ValueError("EP context already has a shared-expert owner")
        shared.bind_ep_context(self)
        self._shared_expert = ref(shared)
        self._shared_state = ContextVar("ep_shared_expert", default=None)
        owner.register_forward_pre_hook(self._capture_shared_input, prepend=True, with_kwargs=True)
        owner.register_forward_hook(self._release_shared_state, with_kwargs=True, always_call=True)
        experts.register_forward_pre_hook(self._capture_expert_input, prepend=True, with_kwargs=True)

    def _capture_shared_input(self, module: nn.Module, args: tuple, kwargs: dict) -> None:
        """Keep original token shape and invocation-local state across both communication windows."""
        del module
        if not self.overlap_shared_expert or self.mesh.size() == 1:
            return
        hidden = args[0] if args else kwargs.get("hidden_states", kwargs.get("x"))
        if not isinstance(hidden, Tensor):
            raise ValueError("shared-expert overlap requires the MoE activation as its first input")
        state = {"hidden": hidden, "shared_called": False, "started": False,
                 "grad_enabled": torch.is_grad_enabled()}
        state["token"] = self._shared_state.set(state)

    def _release_shared_state(self, module: nn.Module, args: tuple, kwargs: dict, output: Any) -> None:
        """Release graph references on normal exit, checkpoint replay and exceptions."""
        del module, args, kwargs, output
        state = self._shared_state.get()
        if state:
            token = state.pop("token")
            self._shared_state.reset(token)
            state.clear()

    def take_shared_output(self, hidden_states: Tensor) -> Any:
        """Consume the prepared result at the model's unchanged shared-expert call.

        Args:
            hidden_states: Original activation supplied to the shared module.

        Returns:
            Prepared shared output, or None to execute the ordinary shared forward.
        """
        state = self._shared_state.get() if self._shared_state is not None else None
        if not state or not isinstance(hidden_states, Tensor):
            return None
        if hidden_states is not state["hidden"] and not hidden_states.is_set_to(state["hidden"]):
            return None
        state["shared_called"] = True
        return state.pop("output", None)

    def _capture_expert_input(self, module: nn.Module, args: tuple, kwargs: dict) -> None:
        """Keep only shared shape metadata; replay uses the restored expert input tensor."""
        del module
        state = self._shared_state.get()
        hidden = args[0] if args else kwargs.get("hidden_states")
        if state and not state["started"] and not state["shared_called"]:
            # Use the containing MoE's post-FSDP activation and gradient hooks,
            # rather than the raw tensor seen before its forward pre-hooks.
            state["hidden"] = hidden.reshape(state["hidden"].shape)
        recompute = get_recompute_state()
        if not self.overlap_shared_expert or recompute is None:
            return
        saved = recompute.get_resource(self, dict)
        if state:
            if not state["shared_called"] and (not state["grad_enabled"] or torch.is_grad_enabled()):
                saved.setdefault("shape", tuple(state["hidden"].shape))
        elif recompute.is_recomputing and saved:
            saved.update(hidden=hidden.reshape(saved["shape"]), shared_called=False, started=False, grad_enabled=True)
            self._shared_state.set(saved)

    def _overlap_ready(self, stage: str) -> bool:
        """Only an active, unconsumed invocation can schedule shared work."""
        state = self._shared_state.get() if self._shared_state is not None else None
        if not self.overlap_shared_expert or not state or state["shared_called"]:
            return False
        # Reentrant experts-only checkpointing must not move the shared branch
        # into its no-grad forward; its ordinary model call still needs autograd.
        if state["grad_enabled"] and not torch.is_grad_enabled() and state["hidden"].requires_grad:
            return False
        return not state["started"] if stage == "fc1" else "intermediate" in state

    def _run_overlap(self, stage: str, *payloads: Tensor | None) -> None:
        """Run a bound shared stage and drain outstanding exchanges if it fails."""
        state = self._shared_state.get()
        shared = self._shared_expert()
        try:
            if stage == "fc1":
                state["started"] = True
                state["intermediate"] = shared(state["hidden"], stage="fc1")
            else:
                state["output"] = shared(state.pop("intermediate"), stage="fc2")
        except Exception:
            for payload in payloads:
                if payload is not None:
                    _functional_collectives.wait_tensor(payload)
            raise

    def combine(
        self, values: Tensor, send_splits: list[int], recv_splits: list[int], recv_inverse: Tensor,
    ) -> Tensor:
        """Return outputs to source prepared-slot order, without weighting or reduction.

        Args:
            values: Local expert-major outputs [received_slots,H].
            send_splits: Original per-peer send sizes from this call's dispatch.
            recv_splits: Original per-peer receive sizes, reversed for combine.
            recv_inverse: Mapping from peer-major rows to local expert-major rows [R].

        Returns:
            Source-side slot outputs; the Experts algorithm restores original tokens.
        """
        if len(send_splits) != self.mesh.size() or len(recv_splits) != self.mesh.size():
            raise ValueError("slot split lists must follow the current EP group's rank order")
        if values.ndim != 2 or values.shape[0] != sum(recv_splits):
            raise ValueError("slot output must have one row per received slot")
        if recv_inverse.shape != (values.shape[0],) or recv_inverse.dtype != torch.int32:
            raise ValueError("recv_inverse must have dtype int32 and one index per received slot")
        if recv_inverse.device != values.device:
            raise ValueError("slot outputs and recv_inverse must be on the same device")
        if values.shape[0]:
            probabilities = values.new_ones((values.shape[0], 1))
            values = moe_token_unpermute(values.contiguous(), recv_inverse.long(), probabilities)
        asynchronous = self._overlap_ready("fc2")
        returned = self._exchange(values, recv_splits, send_splits, async_op=asynchronous)
        if asynchronous:
            self._run_overlap("fc2", returned)
        return returned

    def _slot_split_sizes(self, counts: Tensor, weighted: bool) -> tuple[list[int], list[int]]:
        """Exchange counts and verify optional probability transport agrees across peers."""
        metadata = torch.stack((counts, torch.full_like(counts, int(weighted))), dim=1)
        received = (ep_exchange_counts(metadata, self.mesh.get_group())
                    if self.mesh.size() > 1 else metadata)
        fields = ep_split_sizes(received.reshape(-1))
        if any(flag != int(weighted) for flag in fields[1::2]):
            raise ValueError("all EP ranks must agree on whether slot routing_weights are transported")
        return ep_split_sizes(counts), fields[::2]

    def dispatch(
        self, hidden_states: Tensor, expert_ids: Tensor, routing_weights: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None, list[int], list[int], Tensor]:
        """Transport prepared slots and group received rows by local expert.

        Args:
            hidden_states: Source expert-major slot activations [S,H].
            expert_ids: Matching global expert IDs [S], sorted in ascending order.
            routing_weights: Optional differentiable per-slot probabilities [S].

        Returns:
            Owner-local hidden [R,H], counts [E_local], optional weights [R],
            send/receive split lists and the inverse mapping from received rows to
            owner-local expert-major rows [R]. The caller retains them for combine.
        """
        _validate_slot_inputs(hidden_states, expert_ids, routing_weights)
        ids = expert_ids.to(torch.int64)
        # Fixed-size counting rejects invalid/drop IDs on device and retains empty experts.
        global_counts = ids.new_zeros(self.num_experts).scatter_add_(0, ids, torch.ones_like(ids))
        counts = global_counts.reshape(self.mesh.size(), self.local_experts).sum(dim=1)
        send, receive = self._slot_split_sizes(counts, routing_weights is not None)
        # Ascending global IDs already group rows by rank under contiguous expert ownership.
        asynchronous = self._overlap_ready("fc1")
        hidden = self._exchange(hidden_states.contiguous(), send, receive, async_op=asynchronous)
        global_ids = self._exchange(ids[:, None].contiguous(), send, receive, async_op=asynchronous).reshape(-1)
        weights = None
        if routing_weights is not None:
            weights = self._exchange(routing_weights[:, None].contiguous(), send, receive,
                                     async_op=asynchronous).reshape(-1)
        if asynchronous:
            self._run_overlap("fc1", hidden, global_ids, weights)
        # AsyncCollectiveTensor inserts the wait at these first non-view consumers.
        local_ids = global_ids - self.rank * self.local_experts
        local_counts = local_ids.new_zeros(self.local_experts).scatter_add_(
            0, local_ids, torch.ones_like(local_ids),
        )
        if not hidden.shape[0]:
            return hidden, local_counts, weights, send, receive, local_ids.to(torch.int32)

        hidden, _, recv_inverse = moe_token_permute(hidden.contiguous(), local_ids.view(-1, 1))
        recv_inverse = recv_inverse.to(torch.int32)
        if weights is not None:
            weights = torch.empty_like(weights).scatter_(0, recv_inverse.to(torch.int64), weights)
        return hidden, local_counts, weights, send, receive, recv_inverse

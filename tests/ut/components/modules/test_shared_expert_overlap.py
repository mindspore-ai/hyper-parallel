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
"""Context-owned shared scheduling with unchanged model calls and no callback arguments."""

from __future__ import annotations

import copy
import inspect
from itertools import product
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.checkpoint import DefaultDeviceType, checkpoint as torch_checkpoint

from examples.torch.module_family.config import SharedExpertConfig
from hyper_parallel.components.modules.moe.shared_expert import SharedExpertModule
from hyper_parallel.components.modules.moe.shared_expert_overlap import bind_shared_expert_overlap
from hyper_parallel.components.modules.moe.grouped_experts.ep.ep_grouped_experts import EPGroupedExperts
from hyper_parallel.components.modules.moe.grouped_experts.grouped_experts import GroupedExpertsConfig
from hyper_parallel.core.activation_memory.api import checkpoint
from hyper_parallel.distributed.expert_parallel.ep_context import EPContext
from hyper_parallel.distributed.module_binding import (
    MeshBinding, ParallelBinding, ParallelModuleSpec, bind_parallel_modules,
)
from tests.common.mark_utils import arg_mark


def _mesh(size: int = 2) -> SimpleNamespace:
    """Provide a process-group-free resource borrowed by transport doubles."""
    return SimpleNamespace(ndim=1, size=lambda: size, get_local_rank=lambda: 0, get_group=lambda: None)


class _Experts(nn.Module):
    """Exercise real dispatch/combine while replacing expert GEMM with a CPU projection."""

    def __init__(self, context: EPContext) -> None:
        super().__init__()
        self.proj = nn.Linear(8, 8, bias=False, dtype=torch.float64)
        self.ep_context = context

    def bind_parallel(self, binding: ParallelBinding) -> None:
        """Borrow the automatically constructed test EP context."""
        self.ep_context = binding.ep_context

    def forward(self, hidden: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Retain the old three-tensor call throughout both transport variants."""
        context = self.ep_context
        hidden, _, _, send, receive, inverse = context.dispatch(hidden, ids.flatten())
        values = self.proj(hidden) * weights
        return context.combine(values, send, receive, inverse)


class _MoE(nn.Module):
    """An unmodified model algorithm: routed experts followed by the shared expert."""

    def __init__(self, context: EPContext, *, experts_checkpoint: bool | None = None,
                 shared_first: bool = False) -> None:
        super().__init__()
        self.experts = _Experts(context)
        self.shared_experts = SharedExpertModule(config=SharedExpertConfig(8, 12, bias=True)).double()
        self.experts_checkpoint = experts_checkpoint
        self.shared_first = shared_first

    def forward(self, hidden_states: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Keep scheduling details out of the model's algorithm and interface."""
        shared = self.shared_experts(hidden_states) if self.shared_first else None
        flat = hidden_states.reshape(-1, 8)
        if self.experts_checkpoint is None:
            routed = self.experts(flat, ids, weights)
        else:
            routed = checkpoint(self.experts, flat, ids, weights, use_reentrant=self.experts_checkpoint)
        if shared is None:
            shared = self.shared_experts(hidden_states)
        return routed.reshape_as(hidden_states) + shared


def _record_transport(context: EPContext, events: list) -> list:
    """Emulate token exchange, retaining exact transport arithmetic and autograd."""
    def exchange(payload: torch.Tensor, send: list[int], receive: list[int], *,
                 async_op: bool = False) -> torch.Tensor:
        """Record collective scheduling while retaining the original autograd graph."""
        del send, receive
        events.append(("exchange", async_op))
        return payload

    context._exchange = exchange
    context._slot_split_sizes = lambda counts, _weighted: ([int(counts.sum()), 0], [int(counts.sum()), 0])
    return events


class TestSharedExpertOverlap(unittest.TestCase):
    """Verify transparent binding, both communication windows and invocation cleanup."""

    def setUp(self) -> None:
        """Keep RNG and checkpoint device changes local to each test."""
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        self.addCleanup(DefaultDeviceType.set_device_type, DefaultDeviceType.get_device_type())
        DefaultDeviceType.set_device_type("cpu")
        torch.manual_seed(81)

    @staticmethod
    def _bind(model: _MoE, events: list) -> EPContext:
        """Bind shared stages and record the module calls in scheduling order."""
        context = model.experts.ep_context
        _record_transport(context, events)
        original = model.forward.__func__
        bind_shared_expert_overlap(model, "experts", context)
        if model.forward.__func__ is not original:
            raise AssertionError("framework binding replaced the model forward")

        def record_stage(module: nn.Module, args: tuple, kwargs: dict[str, Any]) -> None:
            """Observe normal module calls without changing inputs or outputs."""
            del module, args
            events.append(("shared", kwargs.get("stage", "full")))

        model.shared_experts.register_forward_pre_hook(record_stage, with_kwargs=True)
        return context

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_outputs_and_gradients_match_serial_with_parent_and_expert_replay(self) -> None:
        """Feature: Transparent shared expert overlap.
        Description: Compare both EP contexts under full-MoE and experts-only checkpointing.
        Expectation: Outputs, all gradients and accumulated microbatches match serial exactly.
        """
        for context_type, replay in product((EPContext,),
                                           ("none", "parent", "parent_reentrant", "experts", "experts_reentrant")):
            with self.subTest(context=context_type.__name__, replay=replay):
                expert_replay = {"experts": False, "experts_reentrant": True}.get(replay)
                model = _MoE(context_type(_mesh(), num_experts=4), experts_checkpoint=expert_replay)
                reference = copy.deepcopy(model)
                reference.experts.ep_context.overlap_shared_expert = False
                events = []
                context = self._bind(model, events)
                self._bind(reference, [])
                for _ in range(2):
                    hidden = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
                    other = hidden.detach().clone().requires_grad_()
                    weights = torch.randn(6, 1, dtype=torch.float64, requires_grad=True)
                    other_weights = weights.detach().clone().requires_grad_()
                    ids = torch.zeros(6, 1, dtype=torch.int64)
                    if replay.startswith("parent"):
                        actual = torch_checkpoint(model, hidden, ids, weights,
                                                  use_reentrant=replay == "parent_reentrant")
                    else:
                        actual = model(hidden, ids, weights)
                    expected = reference(other, ids, other_weights)
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    actual.square().sum().backward()
                    expected.square().sum().backward()
                    torch.testing.assert_close(hidden.grad, other.grad, atol=0, rtol=0)
                    torch.testing.assert_close(weights.grad, other_weights.grad, atol=0, rtol=0)
                    self.assertFalse(context._shared_state.get())
                for actual, expected in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(actual.grad, expected.grad, atol=0, rtol=0)
                if replay != "experts_reentrant":
                    self.assertIn(("shared", "fc1"), events)
                    self.assertIn(("shared", "fc2"), events)
                    self.assertNotIn(("shared", "full"), events)
                    self.assertLess(events.index(("exchange", True)), events.index(("shared", "fc1")))
                else:
                    self.assertIn(("shared", "full"), events)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_common_binder_enables_context_flag_and_preserves_shared_parameters(self) -> None:
        """Feature: Framework-only overlap setup.
        Description: Bind the planned expert without a model-specific forward replacement.
        Expectation: The context borrows the sibling shared module and retains checkpoint names and identities.
        """
        for moon, enabled in product((False,), (False, True)):
            with self.subTest(moon=moon, enabled=enabled):
                model = _MoE(EPContext(_mesh(), num_experts=4))
                keys = set(model.state_dict())
                parameters = dict(model.named_parameters())
                parallel = ParallelModuleSpec("test.ep", "ep", {"forward": ("hidden", "ids", "weights")},
                    validate=lambda **_kwargs: None,
                    options={"num_experts": 4, "moonep": moon, "overlap_shared_expert": enabled})
                plan = SimpleNamespace(parallel_modules={"experts": parallel})
                mesh = SimpleNamespace(fsdp_moe_mesh={"ep": _mesh()}, device_mesh=None)
                bind_parallel_modules(model, plan, mesh)
                context = model.experts.ep_context
                self.assertEqual(context.overlap_shared_expert, enabled)
                self.assertIsInstance(context, EPContext)
                self.assertIs(context._shared_expert(), model.shared_experts)
                self.assertEqual(keys, set(model.state_dict()))
                for name, parameter in parameters.items():
                    self.assertIs(model.get_parameter(name), parameter)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_fsdp_input_alias_is_consumed_without_a_third_shared_call(self) -> None:
        """Feature: Shared input after framework pre-hooks.
        Description: Emulate a containing FSDP pre-hook returning an activation alias.
        Expectation: Staged work uses the post-hook input and the normal shared call consumes its result.
        """
        model = _MoE(EPContext(_mesh(), num_experts=4))
        reference = copy.deepcopy(model)
        reference.experts.ep_context.overlap_shared_expert = False
        events = []
        self._bind(model, events)
        self._bind(reference, [])

        def alias_input(module: nn.Module, args: tuple) -> tuple:
            """Reproduce a parent FSDP pre-hook returning an activation alias."""
            del module
            return (args[0].view_as(args[0]), *args[1:])

        model.register_forward_pre_hook(alias_input)
        hidden = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
        other = hidden.detach().clone().requires_grad_()
        ids, weights = torch.zeros(6, 1, dtype=torch.long), torch.ones(6, 1)
        actual, expected = model(hidden, ids, weights), reference(other, ids, weights)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(hidden.grad, other.grad, atol=0, rtol=0)
        self.assertNotIn(("shared", "full"), events)
        self.assertEqual(events.count(("shared", "fc1")), 1)
        self.assertEqual(events.count(("shared", "fc2")), 1)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_disabled_single_rank_and_shared_first_stay_serial(self) -> None:
        """Feature: Serial fallback.
        Description: Disable overlap, use one owner, or evaluate the shared branch before routed experts.
        Expectation: The ordinary shared call executes once and no asynchronous payload is launched.
        """
        for owners, enabled, shared_first in ((2, False, False), (1, True, False), (2, True, True)):
            model = _MoE(EPContext(_mesh(owners), num_experts=4, overlap_shared_expert=enabled),
                         shared_first=shared_first)
            events = []
            context = self._bind(model, events)
            hidden = torch.randn(2, 3, 8, dtype=torch.float64)
            if owners == 1:
                context._slot_split_sizes = lambda counts, _weighted: ([int(counts.sum())], [int(counts.sum())])
            model(hidden, torch.zeros(6, 1, dtype=torch.long), torch.ones(6, 1))
            self.assertEqual(events.count(("shared", "full")), 1)
            self.assertNotIn(("shared", "fc1"), events)
            self.assertNotIn(("exchange", True), events)
            self.assertFalse(context._shared_state.get())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_stage_failure_drains_payloads_and_next_forward_has_fresh_state(self) -> None:
        """Feature: Shared-stage exception safety.
        Description: Fail FC1 or FC2 after starting communication, then retry the same model.
        Expectation: In-flight tensors are waited, the frame is released, and retry runs both stages once.
        """
        for stage in ("fc1", "fc2"):
            model = _MoE(EPContext(_mesh(), num_experts=4))
            events = []
            context = self._bind(model, events)
            hidden = torch.randn(2, 3, 8, dtype=torch.float64)
            arguments = (hidden, torch.zeros(6, 1, dtype=torch.long), torch.ones(6, 1))
            with patch.object(model.shared_experts, "forward_" + stage, side_effect=ValueError(stage)), patch(
                "hyper_parallel.distributed.expert_parallel.ep_context._functional_collectives.wait_tensor",
            ) as wait:
                with self.assertRaisesRegex(ValueError, stage):
                    model(*arguments)
                self.assertEqual(wait.call_count, 2 if stage == "fc1" else 1)
            self.assertIsNone(context._shared_state.get())
            events.clear()
            model(*arguments)
            self.assertEqual(events.count(("shared", "fc1")), 1)
            self.assertEqual(events.count(("shared", "fc2")), 1)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_plain_shared_module_stays_unbound_and_transport_has_no_callback_arguments(self) -> None:
        """Feature: Native compatibility and transport signatures.
        Description: Keep an unconverted shared MLP and inspect the public EP/MoonEP calls.
        Expectation: No native behavior is replaced and no dispatch/combine callback argument exists.
        """
        model = _MoE(EPContext(_mesh(), num_experts=4))
        model.shared_experts = nn.Linear(8, 8).double()
        bind_shared_expert_overlap(model, "experts", model.experts.ep_context)
        self.assertIsNone(model.experts.ep_context._shared_state)
        for cls in (EPContext,):
            for method in (cls.dispatch, cls.combine):
                self.assertNotIn("overlap_compute", inspect.signature(method).parameters)

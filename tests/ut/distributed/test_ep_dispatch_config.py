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

"""Exercise dispatch configuration and expert ordering."""

import itertools
import unittest
from unittest.mock import MagicMock, Mock, patch

import torch
from torch import nn
import yaml

from hyper_parallel.distributed._builder.rule_resolver import _resolve_local_compute_fn
from hyper_parallel.distributed.expert_parallel import experts as ep_experts
from hyper_parallel.distributed.expert_parallel.experts import _argsort_keys
from hyper_parallel.trainer.config.parallelism import PlanOverride, entries_to_plan_overrides
from hyper_parallel.trainer.config.resolver import normalize_value


class _Moe(nn.Module):
    """Minimal routed boundary for the shipped Qwen3 EP factory."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Identity()
        self.experts = nn.Module()
        self.experts.num_experts = 4

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Use the model path before EP compute is installed."""
        return hidden_states


class TestEpDispatchConfig(unittest.TestCase):
    """Configure capacity independently of the model factory signature."""

    def test_applies_capacity_to_existing_factory(self) -> None:
        """Feature: Shared EP capacity setting.
        Description: Load a YAML override with a nested model factory.
        Expectation: EP=2 binds the capacity and EP=1 skips the rule.
        """
        config = yaml.safe_load("""
match: mlp.forward
when: ep
replace:
  kind: method
  implementation:
    _target_: hyper_parallel.distributed.expert_parallel.recipes.ep_compute_with_dispatch_options
    compute:
      _target_: hyper_parallel.models.qwen3_moe.adapter.distributed.expert_parallel.qwen3moe_ep_compute_fn
      use_grouped_gemm: true
    capacity_factor: 1.25
""")
        override = normalize_value(config, PlanOverride, path="plan_overrides[0]")
        self.assertEqual(entries_to_plan_overrides([override], ep_size=1), {})

        spec = entries_to_plan_overrides([override], ep_size=2)["mlp"]
        ep_mesh = MagicMock()
        ep_mesh["ep"].size.return_value = 2
        module = _Moe()
        compute = _resolve_local_compute_fn(module, spec, None, (), ep_mesh)

        self.assertTrue(callable(compute))
        self.assertEqual(module.experts.local_expert_count, 2)
        self.assertTrue(module.experts._ep_use_grouped_gemm)  # pylint: disable=protected-access
        self.assertEqual(module.experts.capacity_factor, 1.25)


class TestEpExpertSort(unittest.TestCase):
    """Exercise the integer precision boundary of the default expert sort."""

    def test_uses_fp32_only_when_bound_is_safe(self) -> None:
        """Feature: FP32 ordering of routed expert IDs.
        Description: Sort IDs at the upper bound, then sort IDs that FP32 would merge.
        Expectation: Safe IDs use FP32 and out-of-range IDs preserve their integer order.
        """
        limit = 2**24
        original_argsort = torch.Tensor.argsort
        sorted_dtypes = []

        def record_dtype(keys: torch.Tensor) -> torch.Tensor:
            sorted_dtypes.append(keys.dtype)
            return original_argsort(keys)

        with patch.object(torch.Tensor, "argsort", record_dtype):
            safe_order = _argsort_keys(torch.tensor([limit - 1, 0, 2]), bound=limit)
            large_order = _argsort_keys(torch.tensor([limit + 1, limit]), bound=limit + 2)

        self.assertEqual(sorted_dtypes, [torch.float32, torch.int64])
        torch.testing.assert_close(safe_order, torch.tensor([1, 2, 0]))
        torch.testing.assert_close(large_order, torch.tensor([1, 0]))


class TestEpDispatchPayloads(unittest.TestCase):
    """Exercise the two dispatch payloads through both EP entry points."""

    def test_dispatch_exchanges_states_and_indices_separately(self) -> None:
        """Feature: EP dispatch and combine.
        Description: Use the direct and split entry points with the same router.
        Expectation: Dispatch sends states and indices, then combines expert outputs and preserves gradients.
        """
        group = Mock()
        group.size.return_value = 2

        def echo(tensor, *_args):
            """Model symmetric peer messages while recording exchange calls."""
            return tensor

        for rank, split_entry in itertools.product(range(2), (False, True)):
            expert_ids = torch.tensor([[0], [rank], [1]])
            counts = [2, 1] if rank == 0 else [1, 2]
            collective = "ep_all_to_all_async" if split_entry else "ep_all_to_all"
            with self.subTest(rank=rank, split_entry=split_entry):
                module = _Moe()
                compute = Mock(side_effect=lambda states, _indices: states * 2)
                module.experts.forward = compute
                module.experts.local_expert_count = 1
                states = torch.arange(6.0).reshape(1, 3, 2).requires_grad_()
                weights = torch.ones(3, 1, requires_grad=True)
                router = Mock(return_value=(expert_ids, weights))
                with patch.object(ep_experts.dist, "get_rank", return_value=rank), \
                        patch.object(ep_experts, "ep_exchange_counts", side_effect=echo), \
                        patch.object(ep_experts, collective, side_effect=echo) as exchange:
                    if split_entry:
                        state = ep_experts.ep_routed_dispatch(module, states, router_fn=router, ep_group=group)
                        output = ep_experts.ep_routed_experts_and_combine(module, state, group)
                    else:
                        output = ep_experts.ep_routed_forward(module, states, router_fn=router, ep_group=group)
                output.sum().backward()

                compute.assert_called_once()
                self.assertEqual(exchange.call_count, 3)
                self.assertEqual([call.args[0].dtype for call in exchange.call_args_list],
                                 [torch.float32, torch.int64, torch.float32])
                for call in exchange.call_args_list:
                    self.assertEqual(call.args[1:3], (counts, counts))
                torch.testing.assert_close(output, states * 2)
                torch.testing.assert_close(states.grad, torch.full_like(states, 2))
                torch.testing.assert_close(weights.grad, states.detach().sum(-1).reshape(3, 1) * 2)

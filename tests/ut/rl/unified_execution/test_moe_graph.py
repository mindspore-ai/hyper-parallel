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
"""CPU reference checks for route ownership, padding and ordered reduction."""

import unittest

import torch

from rl.unified_execution.ops.moe.fixed_capacity_ep import fixed_capacity_ep, ordered_combine, owner_routes
from rl.unified_execution.execution_plan import FIXED_EP_TARGET, fixed_capacity_ep_capacity
from rl.unified_execution.contract import build_unified_payload, parse_unified_payload


class TestGraphMoE(unittest.TestCase):
    """Check mathematical results independently of NPU or distributed setup."""

    def test_owner_selection_does_not_sum_other_ranks(self) -> None:
        received = torch.full((3, 2, 2, 4), float("nan"))
        ids = torch.tensor([[0, 3], [5, -1]])
        expected = torch.tensor([[[1.] * 4, [2.] * 4], [[3.] * 4, [0.] * 4]])
        received[0, 0, 0] = 1
        received[1, 0, 1] = 2
        received[2, 1, 0] = 3
        torch.testing.assert_close(owner_routes(received, ids, 2), expected, rtol=0, atol=0)

    def test_one_rank_padding_and_autograd(self) -> None:
        """Compare fixed-capacity expert padding with the unpadded gradient reference."""
        x = torch.tensor([[2., 3.], [4., 5.]], requires_grad=True)
        ids = torch.tensor([[0, 1], [1, 0]])
        weights = torch.tensor([[.25, .75], [.6, .4]], requires_grad=True)

        def experts(values: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
            """Independent per-expert linear map with a known derivative."""
            return values * (selected[:, None] + 1)

        result = fixed_capacity_ep(x, ids, weights, expert_forward=experts, group=None, rank=0,
                                   world_size=1, local_experts=2, capacity=8)
        expected = x * ((ids + 1) * weights).sum(-1, keepdim=True)
        torch.testing.assert_close(result, expected)
        actual_grads = torch.autograd.grad(result.sum(), (x, weights), retain_graph=True)
        expected_grads = torch.autograd.grad(expected.sum(), (x, weights))
        for actual, reference in zip(actual_grads, expected_grads, strict=True):
            torch.testing.assert_close(actual, reference)

    def test_slot_order_and_final_rounding(self) -> None:
        routes = torch.tensor([[[1e8], [1.], [-1e8], [3.]]])
        self.assertEqual(ordered_combine(routes, torch.ones(1, 4)).item(), 3.)
        with self.assertRaisesRegex(ValueError, "combine requires"):
            ordered_combine(routes, torch.ones(1, 3))

    def test_explicit_plan_and_invalid_capacity(self) -> None:
        entry = {"match": "*.mlp.forward", "when": "ep", "replace": {"kind": "method",
                 "implementation": {"_target_": FIXED_EP_TARGET, "capacity": 32}}}
        self.assertEqual(fixed_capacity_ep_capacity([entry]), 32)
        self.assertIsNone(fixed_capacity_ep_capacity([]))
        with self.assertRaisesRegex(ValueError, "unambiguous"):
            fixed_capacity_ep_capacity([entry, entry])
        entry["replace"]["implementation"]["capacity"] = True
        with self.assertRaisesRegex(ValueError, "positive integer"):
            fixed_capacity_ep_capacity([entry])

    def test_shared_recipe_survives_worker_contract(self) -> None:
        """Retain the shared expert recipe across worker contract serialization."""
        entry = {"match": "*.mlp.forward", "when": "ep", "replace": {"kind": "method",
                 "implementation": {"_target_": FIXED_EP_TARGET, "capacity": 32}}}
        config = {
            "model": {"name": "qwen3_moe"},
            "consistency": {"schema_version": 1, "mode": "batch_invariant",
                            "attention_backend": "flash_attention_npu_v4",
                            "batch_invariant": {"provider": "torch_npu_level3", "fill_uninitialized_memory": True}},
            "train": {"plan_overrides": [entry], "accelerator": {"tp": 2, "ep": 2}},
            "rollout": {"vllm": {"enforce_eager": False}},
        }
        payload = parse_unified_payload(build_unified_payload(config))
        self.assertEqual(fixed_capacity_ep_capacity(payload["plan_overrides"]), 32)
        self.assertEqual(payload["expert_parallelism"], {"training": 2, "inference": 2})
        config["train"]["plan_overrides"] = []
        with self.assertRaisesRegex(ValueError, "shared graph EP recipe"):
            build_unified_payload(config)

    def test_capacity_overflow_never_drops_tokens(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceed"):
            fixed_capacity_ep(torch.zeros(3, 4), torch.zeros(3, 2, dtype=torch.long), torch.ones(3, 2),
                              expert_forward=None, group=None, rank=0, world_size=1, local_experts=2, capacity=2)

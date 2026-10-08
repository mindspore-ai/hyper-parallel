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
"""CPU contracts for the Qwen3.5-MoE adapter used by Qwen3.6."""

import os
import unittest
from unittest.mock import patch

import torch

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

from hyper_parallel.models.registry import get_model_adapter
from hyper_parallel.models.qwen3_5_moe.adapter.distributed import expert_parallel
from tests.common.mark_utils import arg_mark


class _FakeEpAxis:
    """Minimal EP-axis contract used by the factory test."""

    @staticmethod
    def size():
        """Return the configured EP degree."""
        return 2


class _FakeEpMesh:
    """Minimal EP-mesh contract used by the factory test."""

    def __init__(self):
        self.group = object()

    def get_group(self, name):
        """Return the fake EP process group."""
        if name != "ep":
            raise KeyError(name)
        return self.group

    def __getitem__(self, name):
        """Return the fake EP mesh axis."""
        if name != "ep":
            raise KeyError(name)
        return _FakeEpAxis()


class _FakeMoeBlock:
    """Small Qwen3.5-MoE-shaped object that records branch order."""

    def __init__(self):
        self.gate = object()
        self.experts = object()
        self.calls = []

    def shared_expert(self, hidden_states):
        """Record and return the shared-expert branch."""
        self.calls.append("shared_expert")
        return hidden_states * 2

    def shared_expert_gate(self, hidden_states):
        """Record and return a neutral sigmoid gate."""
        self.calls.append("shared_expert_gate")
        return torch.zeros((*hidden_states.shape[:-1], 1))


class TestQwen35MoeAdapterContracts(unittest.TestCase):
    """Pin the outer and text configuration registration aliases."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_registration_resolves_outer_and_text_identities(self):
        """Validate Qwen3.5-MoE adapter discovery.

        Feature: Qwen3.5-MoE adapter registration.
        Description: Resolve the Transformers outer and text identities through the lazy model registry.
        Expectation: Both identities resolve to their respective Qwen3.5-MoE adapter specifications.
        """
        text_spec = get_model_adapter("qwen3_5_moe_text")
        outer_spec = get_model_adapter("Qwen3_5MoeForConditionalGeneration")

        self.assertIsNotNone(text_spec, f"Qwen3.5-MoE text adapter must be registered, got {text_spec}")
        self.assertIsNotNone(outer_spec, f"Qwen3.5-MoE outer adapter must be registered, got {outer_spec}")
        self.assertEqual(
            text_spec.model_type,
            "qwen3_5_moe_text",
            f"Unexpected text model type: expected=qwen3_5_moe_text, got={text_spec.model_type}",
        )
        self.assertEqual(
            outer_spec.model_type,
            "qwen3_5_moe",
            f"Unexpected outer model type: expected=qwen3_5_moe, got={outer_spec.model_type}",
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_ep_factory_preserves_qwen36_branch_order(self):
        """Validate Qwen3.6 shared/routed/gate ordering and output merge.

        Feature: Qwen3.5-MoE expert-parallel compute factory.
        Description: Build the EP compute function and replace distributed routing with a deterministic stub.
        Expectation: Branches run in Transformers order and merge without changing the tensor shape.
        """
        module = _FakeMoeBlock()
        ep_mesh = _FakeEpMesh()

        def routed_forward(*args, **kwargs):
            del args, kwargs
            module.calls.append("routed_experts")
            return torch.ones((1, 2, 3))

        with (
            patch.object(expert_parallel, "bind_local_expert_forward") as bind_forward,
            patch.object(expert_parallel, "ep_routed_forward", side_effect=routed_forward),
        ):
            compute_fn = expert_parallel.qwen3_5_moe_ep_compute_fn(
                module=module,
                mesh=None,
                tp_mesh=None,
                cp_mesh=None,
                ep_mesh=ep_mesh,
                use_grouped_gemm=True,
            )
            output = compute_fn(module, torch.ones((1, 2, 3)))

        bind_forward.assert_called_once_with(module, 2, use_grouped_gemm=True)
        self.assertEqual(
            module.calls,
            ["shared_expert", "routed_experts", "shared_expert_gate"],
        )
        self.assertEqual(output.shape, (1, 2, 3))
        torch.testing.assert_close(output, torch.full((1, 2, 3), 2.0))


if __name__ == "__main__":
    unittest.main()

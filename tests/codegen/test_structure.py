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
"""Structural discovery must not depend on model aliases or run a router."""

from types import SimpleNamespace
import unittest

import torch
from torch import nn
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

from hyper_parallel.distributed.expert_parallel.structure import (
    UnsupportedModuleStructure, detect_moe_structure, detect_router_kind,
)


class TestMoeStructure(unittest.TestCase):
    """Exercise real source contracts using allocation-free meta models."""

    def test_tuple_router_preserves_scoring_for_both_architectures(self):
        """Grouped sigmoid routers already return final weights and indices."""
        configurations = (
            (Qwen3MoeSparseMoeBlock, Qwen3MoeConfig(), "none"),
            (DeepseekV3MoE, DeepseekV3Config(), "additive"),
        )
        for constructor, config, shared in configurations:
            with self.subTest(constructor=constructor.__name__), torch.device("meta"):
                model = constructor(config)
                structure = detect_moe_structure(model)
                self.assertEqual(structure.router, "topk_router_module")
                self.assertEqual(structure.shared_experts, shared)
                self.assertEqual(structure.expert_storage, "batched_parameters")

    def test_unregistered_identity_uses_identical_source_contract(self):
        """Inherited forward source is sufficient; model registration is irrelevant."""
        class UnregisteredBlock(Qwen3MoeSparseMoeBlock):
            """A model name absent from all registries."""

        with torch.device("meta"):
            structure = detect_moe_structure(UnregisteredBlock(Qwen3MoeConfig()))
        self.assertEqual(structure.router, "topk_router_module")

    def test_ambiguous_shared_branch_fails(self):
        """An extra branch must not silently disappear from generated compute."""
        with torch.device("meta"):
            model = Qwen3MoeSparseMoeBlock(Qwen3MoeConfig())
            model.shared_expert = nn.Linear(4, 4)
        with self.assertRaisesRegex(UnsupportedModuleStructure, "shared expert"):
            detect_moe_structure(model)

    def test_linear_requires_explicit_topk(self):
        """Absent routing settings must not select an invented default."""
        module = SimpleNamespace(gate=nn.Linear(4, 8))
        with self.assertRaises(UnsupportedModuleStructure):
            detect_router_kind(module)
        module.top_k = 2
        self.assertEqual(detect_router_kind(module), "softmax_topk")

    def test_linear_sigmoid_group_markers_select_sigmoid(self):
        """A bare-linear gate backed by sigmoid/group markers must not be
        misread as a softmax top-k router (DeepSeek-V3 / GLM-4-MoE-style)."""
        module = SimpleNamespace(
            gate=nn.Linear(4, 8), top_k=2, routed_scaling_factor=2.0, n_group=2, topk_group=1
        )
        self.assertEqual(detect_router_kind(module), "sigmoid_group")

    def test_linear_without_sigmoid_markers_stays_softmax(self):
        """Plain linear top-k routers keep softmax semantics."""
        module = SimpleNamespace(gate=nn.Linear(4, 8), top_k=2, norm_topk_prob=True)
        self.assertEqual(detect_router_kind(module), "softmax_topk")

    def test_unknown_gate_fails_without_running(self):
        """The generation path never probes a forward with fabricated tensors."""
        class UnknownGate(nn.Module):
            def forward(self, value):
                raise AssertionError("must not execute")

        with self.assertRaisesRegex(UnsupportedModuleStructure, "contract"):
            detect_router_kind(SimpleNamespace(gate=UnknownGate()))

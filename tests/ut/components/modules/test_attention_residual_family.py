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
"""Tests for the common Attention Residual family."""

from types import SimpleNamespace
import unittest

import torch

from hyper_parallel.components.modules import AttentionResidual, AttentionResidualModule
from hyper_parallel.components.modules.common.attention_residual import attention_residual_sharding
from hyper_parallel.components.modules.module_family import get_family_build_info
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard


class TestAttentionResidualFamily(unittest.TestCase):
    """Check config construction, source adoption and declared sequence layouts."""

    def setUp(self) -> None:
        self.config = SimpleNamespace(
            hidden_size=8,
            rms_norm_eps=1e-6,
            initializer_range=0.02,
            attention_residual_backend="eager",
        )
        self.module = AttentionResidual(config=self.config)

    def test_factory_uses_zero_query_unit_scale_and_computes_gradients(self) -> None:
        """Feature: common Attention Residual family.
        Description: Construct from config and run the shared eager implementation.
        Expectation: Query starts at zero, RMS scale at one, and the query can learn.
        """
        self.assertIsInstance(self.module, AttentionResidualModule)
        info = get_family_build_info(self.module)
        self.assertIsNotNone(info)
        self.assertEqual(info.resolved.family.name, "attention_residual")
        self.assertEqual(info.resolved.variant, "local")
        torch.testing.assert_close(self.module.projection, torch.zeros_like(self.module.projection))
        torch.testing.assert_close(self.module.norm_weight, torch.ones_like(self.module.norm_weight))

        prefix = torch.randn(2, 3, 8, requires_grad=True)
        history = torch.randn(2, 3, 2, 8, requires_grad=True)
        output = self.module(prefix, history)
        output.square().sum().backward()
        self.assertEqual(output.shape, prefix.shape)
        self.assertIsNotNone(prefix.grad)
        self.assertIsNotNone(history.grad)
        self.assertIsNotNone(self.module.projection.grad)
        self.assertIsNotNone(self.module.norm_weight.grad)
        self.assertGreater(self.module.projection.grad.abs().sum().item(), 0.0)

    def test_source_replacement_preserves_parameter_identity(self) -> None:
        """Feature: common family source conversion.
        Description: Rebuild an Attention Residual member from a compatible source.
        Expectation: Existing trainable parameter objects are reused.
        """
        replacement = AttentionResidual(module=self.module)
        self.assertIs(replacement.projection, self.module.projection)
        self.assertIs(replacement.norm_weight, self.module.norm_weight)

    def test_backend_and_recompute_are_read_from_config(self) -> None:
        """Feature: config-owned Attention Residual execution options.
        Description: Build the family with runtime choices carried by its config.
        Expectation: Backend aliases are normalized and checkpoint recompute is enabled.
        """
        config = SimpleNamespace(
            hidden_size=8,
            rms_norm_eps=1e-6,
            initializer_range=0.02,
            attention_residual_backend="torch",
            attention_residual_recompute=True,
        )
        module = AttentionResidual(config=config)
        self.assertEqual(module.backend, "eager")
        self.assertTrue(module.recompute)

    def test_sharding_tracks_cp_and_tensor_sequence_parallelism(self) -> None:
        """Feature: module-owned token layout.
        Description: Describe active CP and TP sequence parallel axes.
        Expectation: Parameters replicate and token ports shard sequence dimension one.
        """
        context = SimpleNamespace(
            mesh_dim_names=("tp", "cp", "dp"),
            axis_sizes={"tp": 2, "cp": 4, "dp": 1},
            sequence_parallel=True,
        )
        spec = attention_residual_sharding(
            module=self.module,
            module_fqn="layer.attention_residual",
            member=None,
            context=context,
        )
        self.assertEqual(spec.params["projection"], {"tp": Replicate(), "cp": Replicate(), "dp": Replicate()})
        self.assertEqual(spec.in_src["prefix"], {"tp": Shard(1), "cp": Shard(1), "dp": Replicate()})
        self.assertEqual(spec.in_src["history"], spec.in_src["prefix"])
        self.assertEqual(spec.out_dst["output"], spec.in_src["prefix"])

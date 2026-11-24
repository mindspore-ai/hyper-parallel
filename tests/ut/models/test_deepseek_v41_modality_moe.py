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
"""Native modality-MoE construction, adoption and nested initialization contracts."""

import unittest
from types import SimpleNamespace

import torch

from hyper_parallel.distributed import ShardingPlanner
from hyper_parallel.distributed.mesh import LogicalDeviceMesh
from hyper_parallel.models._transformers.model_builder import _initialize_model_weights
from hyper_parallel.models.deepseek_v41.components.modality_moe import ModalityMoE
from hyper_parallel.models.module_build import build_module_context, prepare_model_components
from hyper_parallel.models.pipeline import PipelineConfig
from tests.common.mark_utils import arg_mark


class TestModalityMoE(unittest.TestCase):
    """Exercise real public child families without optional grouped kernels on CPU."""


    def test_meta_training_and_identity_adoption(self) -> None:
        """Feature: Complete native modality-MoE family.
        Description: Materialize composed families, route natural text/image tokens, and adopt their state.
        Expectation: Bare state initializes, forward/backward stays finite, keys and parameter objects survive.
        """
        torch.set_num_threads(2)
        config = SimpleNamespace(
            hidden_size=8,
            intermediate_size=12,
            num_local_experts=4,
            num_hidden_layers=2,
            initializer_range=0.02,
            swiglu_limit=1.5,
            num_experts_per_tok=2,
            scoring_func="sigmoid",
            routed_scaling_factor=1.0,
            router_aux_loss_coef=0.0,
            router_bias_update_rate=0.0,
            v41_vision_enabled=True,
        )
        with torch.device("meta"):
            model = ModalityMoE(config=config, use_grouped_gemm=False)
        model.to_empty(device="cpu")
        _initialize_model_weights(model)
        self.assertTrue(all(torch.isfinite(parameter).all() for parameter in model.parameters()))
        torch.testing.assert_close(model.gate.bias, torch.zeros(4), atol=0, rtol=0)
        torch.testing.assert_close(model.gate.bias_vl, torch.zeros(4), atol=0, rtol=0)
        hidden = torch.randn(2, 7, 8, requires_grad=True)
        image_mask = torch.arange(7).remainder(2).bool().expand(2, -1)
        expected = model(hidden, image_mask=image_mask)
        adopted = ModalityMoE(module=model, context={"num_layers": 2}, use_grouped_gemm=False)
        self.assertEqual(tuple(dict(model.named_parameters())), tuple(dict(adopted.named_parameters())))
        for name, parameter in model.named_parameters():
            self.assertIs(adopted.get_parameter(name), parameter)
        torch.testing.assert_close(adopted(hidden, image_mask=image_mask), expected, atol=0, rtol=0)
        expected.float().square().mean().backward()
        self.assertTrue(torch.isfinite(hidden.grad).all())
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_pipeline_preparation_requires_eager_partition(self) -> None:
        """Feature: Native modality MoE in eager pipeline stages.
        Description: Prepare PP2 with and without an eager configuration, then add TP2.
        Expectation: Only the configured PP2 build with a supported TP layout succeeds.
        """
        mesh = SimpleNamespace(pp_size=2)
        setup = SimpleNamespace(mesh_context=mesh)
        config = SimpleNamespace(hidden_size=8, intermediate_size=16, num_hidden_layers=2,
                                 num_local_experts=4, num_experts_per_tok=2, initializer_range=0.02,
                                 swiglu_limit=1.5, scoring_func="sigmoid", routed_scaling_factor=1.0,
                                 router_aux_loss_coef=0.0, router_bias_update_rate=0.0, v41_vision_enabled=True)
        model = torch.nn.Sequential(ModalityMoE(config=config, use_grouped_gemm=False))
        with self.assertRaisesRegex(NotImplementedError, "eager pipeline partition"):
            prepare_model_components(model, build_module_context(setup))
        setup.pipeline_config = PipelineConfig(schedule="interleaved_1f1b")
        prepare_model_components(model, build_module_context(setup))
        mesh.tp_size = 2
        with self.assertRaisesRegex(ValueError, "TP requires sequence parallelism"):
            other = torch.nn.Sequential(ModalityMoE(config=config, use_grouped_gemm=False))
            prepare_model_components(other, build_module_context(setup))

    def test_parallel_children_own_expert_transport_and_router_statistics(self) -> None:
        """EP preparation retains class-owned sibling protocols without a whole-MoE forward callback."""
        config = SimpleNamespace(
            hidden_size=8, intermediate_size=16, num_hidden_layers=2, num_local_experts=4,
            num_experts_per_tok=2, scoring_func="sqrtsoftplus", routed_scaling_factor=1.5,
            router_aux_loss_coef=0.01, router_bias_update_rate=0.001, swiglu_limit=1.5,
            n_shared_experts=1, initializer_range=0.02, v41_vision_enabled=True,
        )
        model = torch.nn.Sequential(ModalityMoE(config=config, use_grouped_gemm=False))
        context = {**build_module_context(), "ep_size": 2, "ep": True}
        model, _ = prepare_model_components(model, context)
        plan = ShardingPlanner().plan(model, LogicalDeviceMesh(("dp",), (2,)), ep_size=2)
        self.assertEqual(plan.parallel_modules["0.gate"].binding_kind, "mesh")
        self.assertEqual(plan.parallel_modules["0.experts"].binding_kind, "ep")
        self.assertNotIn("forward", vars(model[0]))

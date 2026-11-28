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
"""mHC family construction, state and portable execution regression."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4UnweightedRMSNorm

from examples.torch.module_family.config import MHCPreConfig, MHCPostConfig, PipelinedMhcConfig
from hyper_parallel.components.modules import (
    MHCPre, MHCPost, PipelinedMhc, get_family_build_info,
    MhcPreModule, MhcPostModule, PipelinedMhcModule,
)
from hyper_parallel.models.module_build import prepare_model_components
from hyper_parallel.models.replacement import ModuleReplacementSpec
from hyper_parallel.components.modules.module_family import FamilyShardingContext
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.distributed._builder.family_layout import collect_owned_layouts, validate_owned_layouts
from tests.common.mark_utils import arg_mark


class TestMHCFamily(unittest.TestCase):
    """Verify the three distinct component contracts."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_model_owned_config(self):
        """
        Feature: mHC construction.
        Description: Independent configs validate direct and adopted construction.
        Expectation: All factories retain model-owned settings and reject invalid settings.
        """
        settings = {"num_stream": 4, "hidden_size": 16, "sinkhorn_iters": 3,
                    "hc_eps": 1e-6, "norm_eps": 1e-6, "use_gamma": True}
        for factory in (MHCPre, MHCPost, PipelinedMhc):
            with self.subTest(factory=factory.__name__):
                member = factory(config=SimpleNamespace(**settings))
                self.assertIsNotNone(get_family_build_info(member))
                restored = factory(module=member)
                for name, parameter in member.named_parameters():
                    self.assertIs(parameter, restored.get_parameter(name))
                with self.assertRaises(ValueError):
                    factory(config=SimpleNamespace(**{**settings, "num_stream": 0}))
        with self.assertRaises(ValueError):
            MHCPre(config=SimpleNamespace(**{**settings, "hc_eps": float("nan")}))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_build_source_and_prepare(self):
        """
        Feature: mHC replacement.
        Description: Prepare frozen source modules through each public family.
        Expectation: Parameters retain identity and frozen state across repeated preparation.
        """
        for factory, config in (
            (MHCPre, MHCPreConfig(4, 16, use_gamma=True)),
            (MHCPost, MHCPostConfig(4)),
            (PipelinedMhc, PipelinedMhcConfig(4, 16)),
        ):
            with self.subTest(factory=factory.__name__):
                concrete = {MHCPre: MhcPreModule, MHCPost: MhcPostModule,
                            PipelinedMhc: PipelinedMhcModule}[factory]
                source = concrete(config=config)
                source.eval()
                source.requires_grad_(False)
                weights = dict(source.named_parameters())
                model = torch.nn.Sequential(source)
                context = {"module_replacements": (ModuleReplacementSpec(("0",), factory),)}
                model, _ = prepare_model_components(model, context)
                self.assertFalse(model[0].training)
                for name, parameter in model[0].named_parameters():
                    self.assertIs(parameter, weights[name])
                    self.assertFalse(parameter.requires_grad)
                self.assertEqual(get_family_build_info(model[0]).origin, "replacement")
                prepare_model_components(model, context)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_meta_reset_and_checkpoint(self):
        """
        Feature: mHC materialization.
        Description: Materialize meta members, reset parameters, and reload their state.
        Expectation: All parameters are finite and checkpoint values restore exactly.
        """
        for factory, config in (
            (MHCPre, MHCPreConfig(4, 16)),
            (MHCPre, MHCPreConfig(4, 16, use_gamma=True)),
            (PipelinedMhc, PipelinedMhcConfig(4, 16)),
        ):
            with self.subTest(factory=factory.__name__):
                with torch.device("meta"):
                    member = factory(config=config)
                member.to_empty(device="cpu")
                member.reset_parameters()
                restored = factory(config=config)
                restored.load_state_dict(member.state_dict())
                for name, parameter in member.named_parameters():
                    self.assertTrue(torch.isfinite(parameter).all())
                    torch.testing.assert_close(parameter, restored.state_dict()[name])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_coefficients_and_post_backward(self):
        """
        Feature: mHC precision.
        Description: Compare portable coefficient and post composition with independent stream math.
        Expectation: Outputs and gradients agree and checkpoint state remains compatible.
        """
        member = PipelinedMhc(config=PipelinedMhcConfig(4, 16, sinkhorn_iters=3))
        reference = PipelinedMhc(config=PipelinedMhcConfig(4, 16, sinkhorn_iters=3))
        reference.load_state_dict(member.state_dict())
        x = torch.randn(2, 3, 4, 16, requires_grad=True)
        rx = x.detach().clone().requires_grad_()
        actual, expected = member(x), reference(rx)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b)
        sum(t.square().sum() for t in actual).backward()
        sum(t.square().sum() for t in expected).backward()
        torch.testing.assert_close(x.grad, rx.grad)
        for p, q in zip(member.parameters(), reference.parameters()):
            torch.testing.assert_close(p.grad, q.grad)
        post = MHCPost(config=MHCPostConfig(4))
        output = torch.randn(2, 3, 16, requires_grad=True)
        residual = torch.randn(2, 3, 64, requires_grad=True)
        h_post, h_res = actual[1].detach(), actual[2].detach()
        combined = post(output, residual, h_post, h_res)
        independent = h_post.unsqueeze(-1) * output.unsqueeze(-2)
        independent = independent + torch.einsum("bsij,bsih->bsjh", h_res, residual.unflatten(-1, (4, 16)))
        torch.testing.assert_close(combined, independent.flatten(2))
        combined.sum().backward()
        self.assertIsNotNone(output.grad)
        self.assertIsNotNone(residual.grad)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_adopted_unweighted_norm_materialization(self) -> None:
        """
        Feature: mHC source materialization.
        Description: Adopt the HF unweighted norm and materialize its meta coefficient owner.
        Expectation: Initialization and backward succeed without requiring a norm reset hook.
        """
        with torch.device("meta"):
            source = PipelinedMhc(config=PipelinedMhcConfig(4, 16, sinkhorn_iters=3))
            source.input_norm = DeepseekV4UnweightedRMSNorm(eps=1e-6)
            member = PipelinedMhc(module=source)
        self.assertIs(member.input_norm, source.input_norm)
        member.to_empty(device="cpu")
        member.reset_parameters()
        self.assertEqual(set(member.state_dict()), {"fn", "base", "scale"})
        for parameter in member.parameters():
            self.assertTrue(torch.isfinite(parameter).all())
        hidden = torch.randn(2, 3, 4, 16, requires_grad=True)
        outputs = member(hidden)
        sum(value.square().sum() for value in outputs).backward()
        self.assertTrue(torch.isfinite(hidden.grad).all())
        for parameter in member.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_reset_distributed_scale(self) -> None:
        """
        Feature: mHC distributed initialization.
        Description: Reset an actual DTensor scale on a CPU mesh with mocked rank lookup.
        Expectation: The scale retains its DTensor placement and all local values become 0.01.
        """
        member = PipelinedMhc(config=PipelinedMhcConfig(4, 16))
        with (
            patch("hyper_parallel.core.dtensor.device_mesh.dist.get_rank", return_value=0),
            patch.dict("hyper_parallel.core.dtensor.device_mesh._DEVICE_MESH_MAP", {}, clear=True),
            patch.dict("hyper_parallel.core.dtensor.dtensor._LAYOUT_CACHE", {}, clear=True),
        ):
            mesh = DeviceMesh("cpu", [0], mesh_dim_names=("mhc_reset",), _init_backend=False)
            scale = DTensor.from_local(torch.empty(3), mesh, (Replicate(),))
            member.scale = torch.nn.Parameter(scale)
            member.reset_parameters()
        self.assertIsInstance(member.scale, DTensor)
        self.assertEqual(member.scale.placements, (Replicate(),))
        torch.testing.assert_close(member.scale.to_local(), torch.full((3,), 0.01))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_invalid_config(self):
        """
        Feature: mHC input validation.
        Description: Construct families with invalid dimensions, epsilon, or ambiguous sources.
        Expectation: Each invalid configuration raises ValueError.
        """
        for values in ((0, 16), (True, 16), (4, 0)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                MHCPreConfig(*values)
        with self.assertRaises(ValueError):
            PipelinedMhcConfig(4, 16, hc_eps=float("nan"))
        with self.assertRaises(ValueError):
            MHCPost(config=MHCPostConfig(4), module=MHCPost(config=MHCPostConfig(4)))

    def test_owned_layout_covers_normalization_and_cp_sp_tokens(self) -> None:
        """Default ownership includes child scales and follows CP/SP without splitting coefficient matrices."""
        for sequence_parallel in (False, True):
            with self.subTest(sequence_parallel=sequence_parallel):
                member = PipelinedMhc(config=PipelinedMhcConfig(4, 16))
                model = torch.nn.Sequential(member)
                context = FamilyShardingContext(
                    ("tp", "cp", "ep"), {"tp": 2, "cp": 2, "ep": 1}, {}, sequence_parallel,
                )
                owned = collect_owned_layouts(model, context)
                spec = owned.specs["0"]
                self.assertEqual(set(spec.params), set(dict(member.named_parameters())))
                self.assertIn("input_norm.weight", spec.params)
                for placements in spec.params.values():
                    self.assertTrue(all(placement == Replicate() for placement in placements.values()))
                self.assertEqual(spec.in_dst["hidden_streams"]["cp"], Shard(1))
                self.assertEqual(spec.in_dst["hidden_streams"]["tp"],
                                 Shard(1) if sequence_parallel else Replicate())
                self.assertEqual(spec.out_names, ("pre", "post", "residual"))
                plan = SimpleNamespace(modules=owned.specs)
                validate_owned_layouts(model, plan, owned, context)
                spec.region_dispatch = False
                with self.assertRaisesRegex(ValueError, "legacy compute injection"):
                    validate_owned_layouts(model, plan, owned, context)

    def test_fused_shifted_member_retains_explicit_parallel_limit(self) -> None:
        """Unqualified fused advance boundaries fail before any device kernel is imported."""
        config = SimpleNamespace(num_stream=4, hidden_size=16, sinkhorn_iters=20,
                                 hc_eps=1e-6, norm_eps=1e-6, use_gamma=False)
        for context in ({"tp_size": 2}, {"cp_size": 4}):
            with self.subTest(context=context), self.assertRaisesRegex(NotImplementedError, "TP=CP=1"):
                PipelinedMhc(config=config, context=context, shifted=True, use_fused_ops=True)

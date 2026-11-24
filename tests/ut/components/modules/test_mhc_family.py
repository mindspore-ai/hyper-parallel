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

import torch

from examples.torch.module_family.config import MHCPostConfig, MHCPreConfig, PipelinedMhcConfig
from hyper_parallel.components.modules import (
    MHCPost,
    MhcPostModule,
    MHCPre,
    MhcPreModule,
    PipelinedMhc,
    PipelinedMhcModule,
    get_family_build_info,
)
from hyper_parallel.models.module_build import prepare_model_components
from hyper_parallel.models.replacement import ModuleReplacementSpec


class TestMHCFamily(unittest.TestCase):
    """Verify the three distinct component contracts."""


    def test_model_owned_config(self):
        """Independent model configs retain direct and source construction validation."""
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

    def test_build_source_and_prepare(self):
        """Source installation reuses parameters, buffers and frozen state."""
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

    def test_meta_reset_and_checkpoint(self):
        """Meta construction resets owned parameters and restores direct state."""
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

    def test_coefficients_and_post_backward(self):
        """Portable coefficient/post composition preserves gradients and checkpoint state."""
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

    def test_invalid_config(self):
        """Reject non-finite epsilons and ambiguous construction sources."""
        for values in ((0, 16), (True, 16), (4, 0)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                MHCPreConfig(*values)
        with self.assertRaises(ValueError):
            PipelinedMhcConfig(4, 16, hc_eps=float("nan"))
        with self.assertRaises(ValueError):
            MHCPost(config=MHCPostConfig(4), module=MHCPost(config=MHCPostConfig(4)))

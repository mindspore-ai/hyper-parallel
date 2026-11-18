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

"""MLP and shared-expert production construction parity."""
import unittest
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.checkpoint import DefaultDeviceType, checkpoint
from examples.torch.module_family.config import SwiGLUMLPConfig, SharedExpertConfig
from hyper_parallel.components.checkpoint import ConcatenateWithSections

from hyper_parallel.components.modules import (
    SwiGLUMLP, SharedExpert, get_family_build_info,
)
from hyper_parallel.models.module_build import prepare_model_components
from hyper_parallel.models.replacement import ModuleReplacementSpec


class SourceMLP(nn.Module):
    """Independent separate-projection SwiGLU reference."""

    def __init__(self, bias=True, device="cpu", dtype=torch.float32):
        super().__init__()
        self.config = SimpleNamespace(hidden_act="silu")
        self.gate_proj = nn.Linear(16, 32, bias=bias, device=device, dtype=dtype)
        self.up_proj = nn.Linear(16, 32, bias=bias, device=device, dtype=dtype)
        self.down_proj = nn.Linear(32, 16, bias=bias, device=device, dtype=dtype)

    def compute(self, x):
        """Apply the independent source expression."""
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

    def forward(self, x):
        """Preserve the MLP source keyword."""
        return self.compute(x)


class SourceShared(SourceMLP):
    """Shared expert source keyword contract."""

    def forward(self, hidden_states):
        """Apply source math with shared-expert keyword."""
        return self.compute(hidden_states)


def check_parity(factory, source_type, device="cpu", dtype=torch.float32, bias=True):
    """Compare real replacement, gradients, one update and checkpoint state."""
    source = source_type(device=device, dtype=dtype, bias=bias)
    reference = source_type(device=device, dtype=dtype, bias=bias)
    reference.load_state_dict(source.state_dict())
    model = nn.Sequential(source)
    context = {"module_replacements": (ModuleReplacementSpec(("0",), factory),)}
    model, _ = prepare_model_components(model, context)
    torch.testing.assert_close(model[0].linear_fc1.weight,
                               torch.cat((reference.gate_proj.weight, reference.up_proj.weight)))
    assert model[0].linear_fc2.weight is source.down_proj.weight
    x = torch.randn(4, 16, device=device, dtype=dtype, requires_grad=True)
    rx = x.detach().clone().requires_grad_()
    upstream = torch.randn_like(x)
    actual, expected = model(x), reference(rx)
    tolerance = {"atol": 2e-5, "rtol": 2e-4} if dtype == torch.float32 else {"atol": 0.02, "rtol": 0.02}
    torch.testing.assert_close(actual, expected, **tolerance)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(x.grad, rx.grad, **tolerance)
    for name in ("weight", "bias"):
        if name == "bias" and not bias:
            continue
        torch.testing.assert_close(getattr(model[0].linear_fc1, name).grad,
                                   torch.cat((getattr(reference.gate_proj, name).grad,
                                              getattr(reference.up_proj, name).grad)), **tolerance)
        torch.testing.assert_close(getattr(model[0].linear_fc2, name).grad,
                                   getattr(reference.down_proj, name).grad, **tolerance)
    for module in (model, reference):
        torch.optim.SGD(module.parameters(), lr=0.01, foreach=False).step()
    torch.testing.assert_close(model(x.detach()), reference(rx.detach()), **tolerance)
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    torch.testing.assert_close(model(x.detach()), reference(rx.detach()), **tolerance)
    prepare_model_components(model, context)


class TestMLPFamily(unittest.TestCase):
    """CPU coverage for actual family members."""

    def setUp(self) -> None:
        """Keep CPU recomputation independent of the registered accelerator default."""
        self.addCleanup(DefaultDeviceType.set_device_type, DefaultDeviceType.get_device_type())
        DefaultDeviceType.set_device_type("cpu")

    def test_model_owned_config(self):
        """Projection validation is independent of example configuration classes."""
        for factory in (SwiGLUMLP, SharedExpert):
            with self.subTest(factory=factory.__name__):
                config = SimpleNamespace(hidden_size=8, intermediate_size=16, bias=False,
                                         return_bias_tuple=False)
                member = factory(config=config)
                self.assertEqual(member(torch.randn(2, 8)).shape, (2, 8))
                config.intermediate_size = 0
                with self.assertRaises(ValueError):
                    factory(config=config)
        with self.assertRaises(TypeError):
            SharedExpert(config=SimpleNamespace(hidden_size=8, intermediate_size=16,
                                                      bias=False, return_bias_tuple=1))

    def test_replacement_parity(self):
        """Preserve loaded projections through actual model preparation."""
        for factory, source in ((SwiGLUMLP, SourceMLP), (SharedExpert, SourceShared)):
            for bias in (False, True):
                with self.subTest(factory=factory.__name__, bias=bias):
                    check_parity(factory, source, bias=bias)

    def test_fused_shared_identity_and_bias(self):
        """Already fused shared sources retain projections and output-bias tuples."""
        source = SharedExpert(config=SharedExpertConfig(16, 32, True, True))
        candidate = SharedExpert(module=source)
        self.assertIs(candidate.linear_fc1, source.linear_fc1)
        self.assertIs(candidate.linear_fc2, source.linear_fc2)
        x = torch.randn(2, 16)
        left, bias = candidate(x)
        right, expected_bias = source(x)
        torch.testing.assert_close(left, right)
        self.assertIs(bias, expected_bias)
        self.assertEqual(candidate.make_transforms(), [])
        tensor_source = SharedExpert(config=SharedExpertConfig(16, 32))
        tensor_candidate = SharedExpert(module=tensor_source)
        self.assertIsInstance(tensor_candidate(x), torch.Tensor)
        torch.testing.assert_close(tensor_candidate(x), tensor_source(x))

    def test_freezing_and_recomputation(self):
        """Packed projections keep freezing and recompute the same gradients."""
        for factory, source_type in ((SwiGLUMLP, SourceMLP), (SharedExpert, SourceShared)):
            source = source_type()
            source.eval()
            source.gate_proj.requires_grad_(False)
            source.up_proj.requires_grad_(False)
            candidate = factory(module=source)
            self.assertFalse(candidate.training)
            self.assertFalse(candidate.linear_fc1.weight.requires_grad)
            x = torch.randn(2, 16, requires_grad=True)
            y = x.detach().clone().requires_grad_()
            candidate(x).sum().backward()
            checkpoint(candidate, y, use_reentrant=False).sum().backward()
            torch.testing.assert_close(x.grad, y.grad)

    def test_packed_checkpoint_sections(self):
        """The declared packing operation reverses both weight and bias sections."""
        source = SourceMLP()
        for name in ("weight", "bias"):
            operation = ConcatenateWithSections((32, 32))
            values = {"gate": getattr(source.gate_proj, name), "up": getattr(source.up_proj, name)}
            packed = operation.convert(values, ["gate", "up"], ["packed"])
            restored = operation.reverse_op.convert(packed, ["packed"], ["gate", "up"])
            for key, value in values.items():
                torch.testing.assert_close(restored[key], value)

    def test_direct_meta_and_tuple(self):
        """Direct and meta construction preserve explicit return protocols."""
        for factory, config in (
            (SwiGLUMLP, SwiGLUMLPConfig(16, 32)),
            (SharedExpert, SharedExpertConfig(16, 32)),
            (SharedExpert, SharedExpertConfig(16, 32, True, True)),
        ):
            with self.subTest(config=config):
                with torch.device("meta"):
                    member = factory(config=config)
                member.to_empty(device="cpu")
                member.reset_parameters()
                output = member(torch.randn(2, 16))
                if isinstance(config, SharedExpertConfig) and config.return_bias_tuple:
                    self.assertIsInstance(output, tuple)
                    self.assertEqual(output[1].shape, (16,))
                else:
                    self.assertEqual(output.shape, (2, 16))
                self.assertEqual(get_family_build_info(member).origin, "direct")

    def test_invalid_config(self):
        """Reject ambiguous dimensions and conflicting construction sources."""
        for size in (0, -1, True, 1.5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                SwiGLUMLPConfig(size, 32)
        with self.assertRaises(ValueError):
            SwiGLUMLP(config=SwiGLUMLPConfig(16, 32), module=SourceMLP())

    def test_shared_x_source_keyword(self):
        """The equivalent x input spelling is accepted without weakening matching."""
        model = nn.Sequential(SourceMLP())
        context = {"module_replacements": (ModuleReplacementSpec(("0",), SharedExpert),)}
        model, _ = prepare_model_components(model, context)
        x = torch.randn(2, 16)
        torch.testing.assert_close(model[0](x=x), model[0](hidden_states=x))
        with self.assertRaises(ValueError):
            model[0](x=x, hidden_states=x)

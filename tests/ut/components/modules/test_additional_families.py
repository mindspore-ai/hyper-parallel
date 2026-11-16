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
"""Independent algorithm and lifecycle acceptance for additional local families."""
# Torch builtins are exposed as descriptors in some local type stubs.
# pylint: disable=not-callable
import unittest
from copy import deepcopy
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from examples.torch.module_family.config import (
    MegaMhcConfig, SpatialMergeProjectorConfig, RotaryEmbeddingConfig, RMSNormConfig,
    SwiGLUMLPConfig, SharedExpertConfig, TokenChoiceTopKRouterConfig, EngramConfig,
)
from hyper_parallel.components.modules import (
    MegaMhc, SpatialMergeProjector, RotaryEmbedding, RMSNorm, OffsetRMSNorm,
    SwiGLUMLP, SharedExpert, Engram, get_family_build_info,
)
from hyper_parallel.components.modules.router import TokenChoiceTopKRouter, TokenChoiceTopKRouterModule
from hyper_parallel.components.modules.mhc.mega_mhc import MegaMhcModule, FusedMegaMhcModule
from hyper_parallel.components.modules.engram.host_engram_table import HostEngramTable
from hyper_parallel.models.deepseek_v41.vision import DeepseekV41VisionAligner
from hyper_parallel.models.module_build import prepare_model_components
from hyper_parallel.models.replacement import ModuleReplacementSpec
from hyper_parallel.models.materialization import MaterializationContext, rebuild_materialized_state


class TestAdditionalFamilies(unittest.TestCase):
    """Compare independent formulas, source state and family replacement boundaries."""

    def test_spatial_source_padding_grad_and_restore(self):
        """Odd grids preserve channel/row/column ordering, loaded state and gradients."""
        source = DeepseekV41VisionAligner(SimpleNamespace(
            v41_vision_downsample_ratio=2, v41_vision_hidden_size=3, hidden_size=5))
        source.eval()
        member = SpatialMergeProjector(module=source)
        self.assertIs(member.w1.weight, source.w1.weight)
        self.assertFalse(member.training)
        inputs = torch.randn(15, 3, requires_grad=True)
        reference_inputs = inputs.detach().clone().requires_grad_()
        grid = F.pad(reference_inputs.reshape(3, 5, 3).permute(2, 0, 1), (0, 1, 0, 1))
        patches = torch.stack([grid[:, row:row + 2, col:col + 2].reshape(-1)
                               for row in (0, 2) for col in (0, 2, 4)])
        expected = source.w2(F.gelu(source.w1(patches)))
        actual = member(inputs, 3, 5)
        torch.testing.assert_close(actual, expected)
        gradients = torch.autograd.grad(actual.sum(), (inputs, *member.parameters()))
        targets = torch.autograd.grad(expected.sum(), (reference_inputs, *source.parameters()))
        for gradient, target in zip(gradients, targets):
            torch.testing.assert_close(gradient, target)
        restored = SpatialMergeProjector(config=SpatialMergeProjectorConfig(3, 5, 5))
        restored.load_state_dict(member.state_dict())
        torch.testing.assert_close(restored(inputs, 3, 5), actual)
        with torch.device("meta"):
            meta = SpatialMergeProjector(config=SpatialMergeProjectorConfig(3, 5, 5))
        meta.to_empty(device="cpu")
        meta.reset_parameters()
        self.assertTrue(torch.isfinite(meta(inputs, 3, 5)).all())

    def test_rotary_scaling_axes_and_stateless_dynamic(self):
        """Frequency equations and sectioned positions agree without request-history state."""
        x = torch.empty(1, 4, 8)
        positions = torch.tensor([[0, 1, 3, 7]])
        for rope_type in ("default", "linear", "dynamic"):
            with self.subTest(rope_type=rope_type):
                config = RotaryEmbeddingConfig(8, factor=2, rope_type=rope_type, max_position_embeddings=4)
                member = RotaryEmbedding(config=config)
                base = 10000.0 * (3 ** (8 / 6)) if rope_type == "dynamic" else 10000.0
                frequency = base ** (-torch.arange(0, 8, 2).float() / 8)
                if rope_type == "linear":
                    frequency /= 2
                angles = positions.unsqueeze(-1) * frequency
                expected = torch.cat((angles, angles), -1)
                actual = member(x, positions, sequence_length=8)
                torch.testing.assert_close(actual[0], expected.cos())
                torch.testing.assert_close(actual[1], expected.sin())
                member(x, positions, sequence_length=32)
                torch.testing.assert_close(member(x, positions, sequence_length=8)[0], actual[0])
                self.assertEqual(member.state_dict(), {})
        multi = RotaryEmbedding(config=RotaryEmbeddingConfig(8, mrope_section=(1, 1, 2)))
        axes = torch.stack((positions, positions + 1, positions + 2))
        frequency = 10000.0 ** (-torch.arange(0, 8, 2).float() / 8)
        chosen = torch.cat((axes[0, ..., None] * frequency[:1], axes[1, ..., None] * frequency[1:2],
                            axes[2, ..., None] * frequency[2:]), -1)
        torch.testing.assert_close(multi(x, axes)[0], torch.cat((chosen, chosen), -1).cos())
        source = RotaryEmbedding(config=RotaryEmbeddingConfig(8))
        adopted = RotaryEmbedding(module=source)
        self.assertIs(adopted.inv_freq, source.inv_freq)
        frequency = source.inv_freq.clone()
        source.bfloat16()
        self.assertEqual(source.inv_freq.dtype, torch.float32)
        torch.testing.assert_close(source.inv_freq, frequency, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            RotaryEmbedding(config=RotaryEmbeddingConfig(8, rope_type="yarn"))
        with torch.device("meta"):
            meta = RotaryEmbedding(config=RotaryEmbeddingConfig(8))
        meta.to_empty(device="cpu")
        rebuild_materialized_state(meta, MaterializationContext("random_init", torch.device("cpu")))
        torch.testing.assert_close(meta(x, positions)[0], source(x, positions)[0])

    def test_mhc_reference_grad_state_and_source(self):
        """Shifted pre/post mapping orientation and all parameter gradients match a separate reference."""
        member = MegaMhc(config=MegaMhcConfig(8, num_iters=3))
        reference = MegaMhcModule(config=MegaMhcConfig(8, num_iters=3))
        reference.load_state_dict(member.state_dict())
        state = (torch.randn(2, 8), torch.randn(2, 4, 8), torch.rand(2, 4),
                 torch.rand(2, 4), torch.rand(2, 4, 4))
        state = tuple(tensor.requires_grad_() for tensor in state)
        independent = tuple(tensor.detach().clone().requires_grad_() for tensor in state)
        output, residual, old_pre, old_post, old_mix = independent
        updated = torch.stack([sum(old_mix[:, j, i, None] * residual[:, j] for j in range(4))
                               + old_post[:, i, None] * output for i in range(4)], 1)
        # Account for epsilon in RMS rather than using a norm-only approximation.
        normalized = updated.flatten(1) / (updated.square().mean((1, 2), keepdim=False) + 1e-6).sqrt()[:, None]
        logits = F.linear(normalized, reference.phi)
        pre = torch.sigmoid(logits[:, :4] * reference.alpha[0] + reference.bias[:4]) + 1e-6
        post = 2 * torch.sigmoid(logits[:, 4:8] * reference.alpha[1] + reference.bias[4:8])
        mix = (logits[:, 8:] * reference.alpha[2] + reference.bias[8:]).reshape(2, 4, 4).softmax(-1) + 1e-6
        mix = mix / (mix.sum(1, keepdim=True) + 1e-6)
        for _ in range(2):
            mix = mix / (mix.sum(2, keepdim=True) + 1e-6)
            mix = mix / (mix.sum(1, keepdim=True) + 1e-6)
        block = sum(old_pre[:, i, None] * updated[:, i] for i in range(4))
        block = block / (block.square().mean(-1, keepdim=True) + 1e-6).sqrt() * reference.norm_weight.float()
        expected = (updated, pre, post, mix, block)
        actual = member(*state)
        for result, target in zip(actual, expected):
            torch.testing.assert_close(result, target)
        sum(value.square().sum() for value in actual).backward()
        sum(value.square().sum() for value in expected).backward()
        for parameter, ref_parameter in zip(member.parameters(), reference.parameters()):
            torch.testing.assert_close(parameter.grad, ref_parameter.grad, atol=3e-5, rtol=3e-5)
        for value, ref_value in zip(state, independent):
            torch.testing.assert_close(value.grad, ref_value.grad, atol=3e-5, rtol=3e-5)
        adopted = MegaMhc(module=member)
        self.assertIs(adopted.phi, member.phi)
        with self.assertRaises(ValueError):
            MegaMhc(module=member, backend="fused")
        fused = MegaMhc(config=MegaMhcConfig(128), backend="fused")
        self.assertIsInstance(fused, FusedMegaMhcModule)
        restored = MegaMhc(config=MegaMhcConfig(8, num_iters=3))
        restored.load_state_dict(member.state_dict())
        for result, target in zip(restored(*state), actual):
            torch.testing.assert_close(result, target)
        member.bfloat16()
        self.assertEqual(member.phi.dtype, torch.float32)

    def test_eager_norm_and_clamped_mlp_shared(self):
        """Ordinary norm and clamped activation retain dtype and gradient contracts."""
        for factory in (RMSNorm, OffsetRMSNorm):
            member = factory(config=RMSNormConfig(8))
            x = torch.randn(2, 8, dtype=torch.bfloat16, requires_grad=True)
            expected = (x.float() / (x.float().square().mean(-1, keepdim=True) + 1e-5).sqrt()).to(x.dtype)
            torch.testing.assert_close(member(x), expected)
            member(x).sum().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
        for factory, config in ((SwiGLUMLP, SwiGLUMLPConfig(4, 6, swiglu_limit=0.2)),
                                (SharedExpert, SharedExpertConfig(4, 6, swiglu_limit=0.2))):
            member = factory(config=config)
            x = torch.randn(3, 4, requires_grad=True)
            packed = F.linear(x, member.linear_fc1.weight)
            gate, up = packed.float().chunk(2, -1)
            expected = F.linear(F.silu(gate.clamp(max=0.2)) * up.clamp(-0.2, 0.2), member.linear_fc2.weight)
            torch.testing.assert_close(member(x), expected)
            member(x).sum().backward()
            self.assertTrue(torch.isfinite(member.linear_fc1.weight.grad).all())
            if factory is SwiGLUMLP:
                adopted = SwiGLUMLP(module=member)
                self.assertIs(adopted.linear_fc1.weight, member.linear_fc1.weight)
                self.assertEqual(adopted.make_transforms(), [])

    def test_router_replacement_and_metadata(self):
        """Only public factories declare replacement; model bias is supplied at each call."""
        source = TokenChoiceTopKRouterModule(config=TokenChoiceTopKRouterConfig(4, 6, top_k=2))
        model = nn.Sequential(source)
        model, _ = prepare_model_components(model, {"module_replacements": (
            ModuleReplacementSpec(("0",), TokenChoiceTopKRouter),)})
        self.assertIs(model[0].gate.weight, source.gate.weight)
        x, bias = torch.randn(3, 4), torch.rand(6)
        for actual, expected in zip(model[0](x, bias), source(x, bias)):
            torch.testing.assert_close(actual, expected)
        direct = TokenChoiceTopKRouter(config=TokenChoiceTopKRouterConfig(4, 6, top_k=2))
        self.assertIsNotNone(get_family_build_info(direct))
        for concrete in (MegaMhcModule, FusedMegaMhcModule, type(model[0])):
            self.assertFalse(getattr(concrete, "_hp_module_replacement", False))
        self.assertTrue(getattr(TokenChoiceTopKRouter, "_hp_module_replacement", False))

    def test_host_table_sparse_step_and_restore(self):
        """Host lookup coalesces repeated rows and resumes an optimizer step from saved state."""
        initial = torch.randn(12, 3)
        table = HostEngramTable(source_weight=nn.Parameter(torch.empty(12, 3, device="meta")), logical_rows=11)
        table.bind_planned_shard(ep_rank=0, ep_size=1)
        torch.utils.swap_tensors(table.weight, nn.Parameter(initial.clone()))
        reference = nn.Embedding.from_pretrained(initial.clone(), freeze=False, sparse=True)
        optimizer = torch.optim.SGD(table.parameters(), lr=0.1, momentum=0.9, foreach=False)
        ref_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1, momentum=0.9, foreach=False)
        ids = torch.tensor([1, 3, 1, 7])
        table(ids).square().sum().backward()
        reference(ids).square().sum().backward()
        table.pending = table.coalesce_pending()
        table.install_grad()
        optimizer.step()
        ref_optimizer.step()
        torch.testing.assert_close(table.weight, reference.weight)
        saved_model, saved_optimizer = deepcopy(table.state_dict()), deepcopy(optimizer.state_dict())
        restored = HostEngramTable(source_weight=nn.Parameter(torch.empty(12, 3, device="meta")), logical_rows=11)
        restored.bind_planned_shard(ep_rank=0, ep_size=1)
        torch.utils.swap_tensors(restored.weight, nn.Parameter(torch.zeros_like(initial)))
        restored.load_state_dict(saved_model)
        resumed = torch.optim.SGD(restored.parameters(), lr=0.1, momentum=0.9, foreach=False)
        resumed.load_state_dict(saved_optimizer)
        for current, opt in ((table, optimizer), (restored, resumed)):
            current.clear_step()
            opt.zero_grad(set_to_none=True)
            current(ids).sum().backward()
            current.pending = current.coalesce_pending()
            current.install_grad()
            opt.step()
        torch.testing.assert_close(table.weight, restored.weight)

    def test_host_family_direct_and_source(self):
        """Direct Host construction defers the full table and preserves dense fusion state."""
        assets = {"layer_ids": [0], "max_ngram_size": 2, "num_heads": 2,
                  "token_map": list(range(6)), "pad_token_id": 0,
                  "primes": [[[5, 7]]], "multipliers": [[1, 3]]}
        config = EngramConfig(assets, 0, 4, 1, 3, padded_num_embeddings=16)
        host = Engram(config=config, engram_storage_backend="host")
        self.assertTrue(host.embed.weight.is_meta)
        self.assertEqual(host.embed.weight.dtype, torch.float32)
        self.assertEqual(host.wkv.weight.device.type, "cpu")
        source = Engram(module=host, engram_storage_backend="host")
        self.assertIs(source.embed, host.embed)
        self.assertFalse(getattr(type(host), "_hp_module_replacement", False))
        host.embed.bind_planned_shard(ep_rank=0, ep_size=1)
        torch.utils.swap_tensors(host.embed.weight, nn.Parameter(torch.randn(16, 3)))
        hidden = torch.randn(1, 3, 1, 4)
        ids = torch.tensor([[1, 2, 3]])
        actual = host(hidden, ids)
        parallel = host.parallel_forward(hidden, ids, ep_group=None, ep_rank=0, ep_size=1)
        torch.testing.assert_close(parallel, actual)

    def test_packed_mlp_source_and_clamp(self):
        """Packed model weights retain identity and reversible checkpoint names."""
        source = nn.Module()
        source.w1, source.w2 = nn.Linear(4, 12), nn.Linear(6, 4)
        source.swiglu_limit = 0.3
        source.eval()
        member = SwiGLUMLP(module=source)
        x = torch.randn(2, 4)
        gate, up = source.w1(x).float().chunk(2, -1)
        expected = source.w2(F.silu(gate.clamp(max=0.3)) * up.clamp(-0.3, 0.3))
        torch.testing.assert_close(member(x), expected)
        self.assertIs(member.linear_fc1.weight, source.w1.weight)
        self.assertFalse(member.training)
        self.assertEqual(len(member.make_transforms()), 4)

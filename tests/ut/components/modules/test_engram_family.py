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
"""Device Engram family and deterministic hash-state regression."""
import unittest
from types import SimpleNamespace

import torch
from torch.utils.checkpoint import DefaultDeviceType, checkpoint

from examples.torch.module_family.config import EngramConfig
from hyper_parallel.components.modules.engram.engram import Engram
from hyper_parallel.components.modules.engram import EngramModule, NgramHashMapping
from hyper_parallel.models.materialization import MaterializationContext, rebuild_materialized_state
from hyper_parallel.models.module_build import prepare_model_components
from hyper_parallel.models.replacement import ModuleReplacementSpec
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41Engram


def assets():
    """Return small disjoint n-gram hash tables."""
    return {"layer_ids": [0], "max_ngram_size": 3, "num_heads": 2, "token_map": list(range(16)),
            "pad_token_id": 0, "primes": [[[5, 7], [11, 13]]], "multipliers": [[1, 3, 5]]}


def config():
    """Return a padded device-table configuration."""
    return EngramConfig(assets(), 0, 8, 2, 4, padded_num_embeddings=40)


def independent_fusion(member, hidden, ids, starts=None, mask=None):
    """Use a separately expressed signed-root cosine gate as reference."""
    rows = member.hash_mapping(ids, starts, mask)
    embeddings = member.embed(rows).flatten(-2)
    key, value = member.wkv(embeddings).split((16, 8), dim=-1)
    key = key.float().unflatten(-1, (2, 8))
    h = hidden.float()
    normalized_h = h * (h.square().mean(-1, keepdim=True) + member.eps).rsqrt()
    normalized_k = key * (key.square().mean(-1, keepdim=True) + member.eps).rsqrt()
    score = (normalized_h * normalized_k * member.q_weight.float() * member.k_weight.float()).sum(-1) / 8**0.5
    signed = torch.where(score < 0, -score.abs().clamp_min(member.clamp_value).sqrt(),
                         score.abs().clamp_min(member.clamp_value).sqrt())
    output = (h + signed.sigmoid().unsqueeze(-1) * value.float().unsqueeze(-2)).to(hidden.dtype)
    return output if mask is None else torch.where(mask.bool().unsqueeze(-1).unsqueeze(-1), output, hidden)


def check_device(device="cpu", dtype=torch.float32):
    """Compare actual replacement math, gradients, update and reconstruction."""
    source = EngramModule(config=config()).to(device=device, dtype=dtype)
    reference = EngramModule(config=config()).to(device=device, dtype=dtype)
    reference.load_state_dict(source.state_dict())
    container = torch.nn.Module()
    container.engram = source
    context = {"module_replacements": (ModuleReplacementSpec(("engram",), Engram),)}
    container, _ = prepare_model_components(container, context)
    member = container.engram
    assert member.embed is source.embed
    ids = torch.tensor([[1,2,3,4], [4,3,2,1]], device=device)
    starts = torch.tensor([[1,0,1,0], [1,0,0,0]], device=device)
    mask = torch.tensor([[1,1,1,0], [1,1,0,1]], device=device, dtype=torch.bool)
    hidden = torch.randn(2,4,2,8,device=device,dtype=dtype,requires_grad=True)
    other = hidden.detach().clone().requires_grad_()
    actual = member(hidden, ids, starts, mask)
    expected = independent_fusion(reference, other, ids, starts, mask)
    tolerance = {"atol": 2e-5, "rtol": 2e-4} if dtype == torch.float32 else {"atol": 0.03, "rtol": 0.03}
    torch.testing.assert_close(actual, expected, **tolerance)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(hidden.grad, other.grad, **tolerance)
    for (name, left), (_, right) in zip(member.named_parameters(), reference.named_parameters()):
        torch.testing.assert_close(left.grad, right.grad, **tolerance)
    for model in (member, reference):
        torch.optim.SGD(model.parameters(), lr=0.001, foreach=False).step()
    torch.testing.assert_close(member(hidden.detach(), ids, starts, mask),
                               independent_fusion(reference, other.detach(), ids, starts, mask), **tolerance)
    restored = Engram(config=config()).to(device=device, dtype=dtype)
    restored.load_state_dict(member.state_dict())
    torch.testing.assert_close(restored(hidden.detach(), ids, starts, mask),
                               member(hidden.detach(), ids, starts, mask), **tolerance)


class TestEngramFamily(unittest.TestCase):
    """Construction, assets, masks and actual fusion regression."""

    def setUp(self) -> None:
        """Keep CPU recomputation independent of the registered accelerator default."""
        self.addCleanup(DefaultDeviceType.set_device_type, DefaultDeviceType.get_device_type())
        DefaultDeviceType.set_device_type("cpu")

    def test_model_owned_config(self):
        """Model configs need attributes, without inheriting an example config."""
        settings = {"assets": assets(), "layer_id": 0, "hidden_size": 8, "hc_mult": 2, "embedding_dim": 4,
                    "padded_num_embeddings": None, "eps": 1e-6, "clamp_value": 1e-6}
        member = Engram(config=SimpleNamespace(**settings))
        self.assertEqual(member.padded_num_embeddings, member.logical_num_embeddings)
        replacement = Engram(module=member)
        self.assertIs(replacement.embed, member.embed)
        for change in ({"hidden_size": 0}, {"eps": float("nan")}, {"padded_num_embeddings": 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                Engram(config=SimpleNamespace(**{**settings, **change}))

    def test_math_and_replacement(self):
        """Preserve the supported source contract and actual gradients."""
        check_device()

    def test_hash_oracle_and_segments(self):
        """Hash IDs follow rolling XOR with independent modular offsets."""
        mapping = NgramHashMapping(assets(), 0)
        ids = torch.tensor([[1,2,3,4]])
        actual = mapping(ids)
        rows = []
        for i, token in enumerate((1,2,3,4)):
            previous = (1,2,3,4)[i-1] if i else 0
            older = (1,2,3,4)[i-2] if i > 1 else 0
            first = token ^ (previous*3)
            second = first ^ (older*5)
            rows.append([first%5, first%7+5, second%11+12, second%13+23])
        torch.testing.assert_close(actual, torch.tensor([rows]))
        packed = mapping(ids, torch.tensor([[1,0,1,0]]))
        separate = torch.cat((mapping(ids[:,:2]), mapping(ids[:,2:])), dim=1)
        torch.testing.assert_close(packed, separate)
        self.assertEqual(mapping(ids[:, :0], torch.empty(1, 0)).shape, (1, 0, 4))

    def test_meta_assets_and_restore(self):
        """Non-persistent hash buffers rebuild from assets after to_empty."""
        with torch.device("meta"):
            member = Engram(config=config())
        member.to_empty(device="cpu")
        member.reset_parameters()
        rebuild_materialized_state(member, MaterializationContext("random_init", torch.device("cpu")))
        expected = NgramHashMapping(assets(), 0)
        for name, value in expected.named_buffers():
            torch.testing.assert_close(dict(member.hash_mapping.named_buffers())[name], value)
        self.assertFalse(any("hash_mapping" in name for name in member.state_dict()))
        self.assertEqual(member.checkpoint_logical_shapes(), {"embed.weight": (36, 4)})

    def test_freeze_recompute_and_asset_snapshot(self):
        """Recomputation is stateless and config copies input hash assets."""
        original_assets = assets()
        cfg = EngramConfig(original_assets, 0, 8, 2, 4)
        original_assets["token_map"][1] = 9
        self.assertEqual(cfg.assets["token_map"][1], 1)
        member = Engram(config=cfg)
        member.embed.requires_grad_(False)
        ids = torch.tensor([[1,2,3]])
        x = torch.randn(1,3,2,8,requires_grad=True)
        rx = x.detach().clone().requires_grad_()
        member(x, ids).sum().backward()
        checkpoint(member, rx, ids, use_reentrant=False).sum().backward()
        torch.testing.assert_close(x.grad, rx.grad)
        self.assertIsNone(member.embed.weight.grad)

    def test_invalid_assets(self):
        """Reject tables too small for logical rows and missing layer metadata."""
        with self.assertRaises(ValueError):
            EngramConfig(assets(), 0, 8, 2, 4, padded_num_embeddings=35)
        with self.assertRaises(ValueError):
            EngramConfig(assets(), 9, 8, 2, 4)

    def test_real_v41_source(self):
        """Preserve the actual model-owned Engram state and call protocol."""
        data = {**assets(), "head_dim": 4, "num_embeddings": [36], "padded_num_embeddings": [40]}
        source = DeepseekV41Engram(SimpleNamespace(hidden_size=8, hc_mult=2, rms_norm_eps=1e-6), 0, data)
        source.eval()
        source.embed.requires_grad_(False)
        model = torch.nn.Module()
        model.engram = source
        context = {"module_replacements": (ModuleReplacementSpec(("engram",), Engram),)}
        model, _ = prepare_model_components(model, context)
        for name, parameter in source.named_parameters():
            self.assertIs(dict(model.engram.named_parameters())[name], parameter)
        self.assertFalse(model.engram.training)
        self.assertFalse(model.engram.embed.weight.requires_grad)
        hidden = torch.randn(1, 3, 2, 8)
        ids = torch.tensor([[1, 2, 3]])
        torch.testing.assert_close(model.engram(hidden, ids), source(hidden, ids))

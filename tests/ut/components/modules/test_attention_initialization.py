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
"""Initialization ownership across native attention families and CP selection."""

from itertools import product
from types import SimpleNamespace
from typing import Any, Optional
import unittest
from unittest.mock import patch

import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from hyper_parallel.components.modules import (
    DeepseekSparseAttention, GatedDeltaNet, GatedGroupedQueryAttention, GatedMultiLatentAttention,
    GroupedQueryAttention, KimiDeltaAttention, MultiLatentAttention, OffsetRMSNorm, RMSNorm,
)
from hyper_parallel.models._transformers.model_builder import _initialize_model_weights
from hyper_parallel.models.module_build import prepare_model_components


_FACTORIES = {
    "gqa": GroupedQueryAttention, "split_gqa": GroupedQueryAttention,
    "gated_gqa": GatedGroupedQueryAttention, "mla": MultiLatentAttention,
    "gated_mla": GatedMultiLatentAttention, "gdn": GatedDeltaNet,
    "kda": KimiDeltaAttention, "dsa": DeepseekSparseAttention,
}


def _config() -> SimpleNamespace:
    """Supply small dimensions, including optional norms, biases and indexer weights."""
    return SimpleNamespace(
        hidden_size=32, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        q_lora_rank=8, kv_lora_rank=8, qk_nope_head_dim=4, qk_rope_head_dim=4, v_head_dim=8,
        attention_bias=True, qk_norm=True, rms_norm_eps=1e-6, index_head_dim=8,
        index_n_heads=2, index_topk=1024, num_heads=4, head_k_dim=4, head_v_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=4,
        linear_value_head_dim=8, init_std=.017,
    )


def _attention(kind: str, cp: int = 1, *, module: Optional[nn.Module] = None) -> nn.Module:
    """Select a local/CP member without binding communication resources."""
    arguments: dict[str, Any] = {"context": {"cp_size": cp}}
    arguments.update({"config": _config()} if module is None else {"module": module})
    if kind in ("kda", "gdn"):
        arguments["ulysses_degree"] = None
    if kind == "split_gqa":
        arguments["fuse_qkv"] = False
    if kind == "dsa":
        arguments["kv_cp_size"] = cp
    return _FACTORIES[kind](**arguments)


@torch.no_grad()
def _poison(module: nn.Module) -> None:
    """Detect parameters that were never written after allocating empty storage."""
    for parameter in module.parameters():
        parameter.fill_(float("nan"))


class _NativePolicy(nn.Module):
    """Model whose policy deliberately differs from component defaults."""

    def __init__(self, block: nn.Module, policy: str = "normal") -> None:
        super().__init__()
        self.block = block
        self.policy = policy
        self.config = _config()
        self.native_snapshot = {}

    @torch.no_grad()
    def initialize_weights(self) -> None:
        """Initialize selected leaves or delegate ownership to the complete block."""
        if self.policy == "delegate":
            self.block.reset_parameters()
        elif self.policy == "all_parameters":
            for parameter in self.parameters():
                nn.init.constant_(parameter, .375)
        else:
            for module in self.modules():
                if not isinstance(module, (nn.Linear, nn.Conv1d)):
                    continue
                if self.policy == "normal":
                    nn.init.normal_(module.weight, std=self.config.init_std)
                else:
                    nn.init.trunc_normal_(module.weight, std=.02, a=-.03, b=.03)
                if module.bias is not None:
                    nn.init.constant_(module.bias, -.125)
        self.native_snapshot = {name: parameter.detach().clone() for name, parameter in self.named_parameters()}


class _HFPolicy(PreTrainedModel):
    """Use the real HF dispatch and per-tensor initialization guards."""

    config_class = PretrainedConfig

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__(config)
        self.block = _attention("kda")
        self.post_init()


class TestAttentionInitialization(unittest.TestCase):
    """Preserve model policies without weakening complete component reset or adoption."""

    def test_native_policies_survive_local_and_cp_initialization(self) -> None:
        """Model initialization remains exact for projections, gates, biases and nested norms."""
        policies = ("normal", "truncated", "all_parameters", "delegate")
        for kind, cp, policy in product(_FACTORIES, (1, 2), policies):
            with self.subTest(family=kind, cp=cp, policy=policy):
                with torch.device("meta"):
                    model = _NativePolicy(_attention(kind, cp), policy)
                model.to_empty(device="cpu")
                _poison(model)
                _initialize_model_weights(model)
                for name, parameter in model.named_parameters():
                    self.assertTrue(torch.isfinite(parameter).all(), name)
                    torch.testing.assert_close(parameter, model.native_snapshot[name], rtol=0, atol=0, msg=name)

    def test_standalone_reset_and_fallback_write_all_parameters(self) -> None:
        """Full reset remains usable without an enclosing model initializer."""
        for kind, cp, fallback in product(_FACTORIES, (1, 2), (False, True)):
            with self.subTest(family=kind, cp=cp, fallback=fallback):
                with torch.device("meta"):
                    block = _attention(kind, cp)
                block.to_empty(device="cpu")
                _poison(block)
                if fallback:
                    _initialize_model_weights(nn.Sequential(block))
                else:
                    block.reset_parameters()
                self.assertTrue(all(torch.isfinite(parameter).all() for parameter in block.parameters()))
                if kind in ("kda", "gdn"):
                    expected = torch.zeros_like(block.dt_bias) if kind == "kda" else torch.ones_like(block.dt_bias)
                    torch.testing.assert_close(block.dt_bias, expected, rtol=0, atol=0)
                    self.assertTrue(torch.all(block.A_log <= torch.log(torch.tensor(16.))))

    def test_promotion_preserves_parameters_and_deferred_initialization(self) -> None:
        """Explicit module adoption and model preparation retain the same initialization responsibility."""
        for kind, device, route in product(_FACTORIES, ("cpu", "meta"), ("module", "prepare")):
            with self.subTest(family=kind, device=device, route=route):
                with torch.device(device):
                    source = (DeepseekSparseAttention(config=_config(), kv_cp_size=2)
                              if kind == "dsa" and route == "prepare" else _attention(kind)).eval()
                original = dict(source.named_parameters())
                next(source.parameters()).requires_grad_(False)
                saved = {name: value.detach().clone() for name, value in original.items()} if device == "cpu" else {}
                model = _NativePolicy(source)
                model.alias = source
                with patch.object(source, "reset_parameters", wraps=source.reset_parameters) as reset:
                    if route == "module":
                        model.block = _attention(kind, 2, module=source)
                        model.alias = model.block
                    else:
                        model, _ = prepare_model_components(model, {"cp_size": 2, "tp_size": 1, "ep_size": 1})
                    reset.assert_not_called()
                target = model.block
                self.assertIs(model.alias, target)
                self.assertFalse(target.training)
                self.assertFalse(next(target.parameters()).requires_grad)
                self.assertTrue(getattr(target, "_hp_reset_after_materialization", False))
                for name, value in original.items():
                    self.assertIs(target.get_parameter(name), value)
                    if saved:
                        torch.testing.assert_close(value, saved[name], rtol=0, atol=0)
                if device == "meta":
                    model.to_empty(device="cpu")
                    _poison(model)
                    _initialize_model_weights(model)
                    for name, parameter in model.named_parameters():
                        self.assertTrue(torch.isfinite(parameter).all(), name)
                        torch.testing.assert_close(parameter, model.native_snapshot[name], rtol=0, atol=0, msg=name)

    def test_hf_guards_do_not_hide_the_model_initializer(self) -> None:
        """Component defaults cannot mark HF initialization complete before the model policy."""
        for field in ("initializer_range", "init_std"):
            with self.subTest(field=field):
                with torch.device("meta"):
                    model = _HFPolicy(PretrainedConfig(**{field: .035}))
                model.to_empty(device="cpu")
                _poison(model)
                reset = model.block.reset_parameters

                def initialize_defaults(reset_parameters=reset, block=model.block):
                    """Bind this subtest's defaults and tensors before the next iteration."""
                    reset_parameters()
                    for module in block.modules():
                        module._is_hf_initialized = True
                    for parameter in block.parameters():
                        parameter._is_hf_initialized = True

                with patch.object(model.block, "reset_parameters", side_effect=initialize_defaults):
                    _initialize_model_weights(model)
                projection = model.block.q_proj.weight
                self.assertAlmostEqual(projection.std().item(), .035, delta=.004)
                self.assertTrue(all(torch.isfinite(value).all() for value in model.parameters()))

    def test_external_parameter_guards_survive_defaults_and_native_initialization(self) -> None:
        """Loaded external parameters keep their protection when defaults precede HF initialization."""
        config = PretrainedConfig()
        config.init_std = .035
        with torch.device("meta"):
            model = _HFPolicy(config)
            model.external_projection = nn.Linear(4, 4, bias=False)
        model.to_empty(device="cpu")
        _poison(model)
        parameter = model.external_projection.weight
        nn.init.constant_(parameter, 7.)
        parameter._is_hf_initialized = True
        model._hp_external_state = SimpleNamespace(parameters={"external_projection.weight": parameter})
        _initialize_model_weights(model)
        torch.testing.assert_close(parameter, torch.full((4, 4), 7.), rtol=0, atol=0)
        self.assertTrue(all(torch.isfinite(value).all() for value in model.parameters()))

    def test_norm_defaults_yield_to_explicit_model_values(self) -> None:
        """Direct and offset defaults apply only when the model leaves their parameters alone."""
        for policy, expected in (("normal", (1., 0.)), ("all_parameters", (.375, .375))):
            with self.subTest(policy=policy), torch.device("meta"):
                config = SimpleNamespace(hidden_size=8, eps=1e-6)
                direct = RMSNorm(config=config)
                offset = OffsetRMSNorm(config=config)
                model = _NativePolicy(nn.ModuleList([direct, offset]), policy)
            model.to_empty(device="cpu")
            _poison(model)
            _initialize_model_weights(model)
            for block, value in zip(model.block, expected):
                torch.testing.assert_close(block.weight, torch.full((8,), value), rtol=0, atol=0)

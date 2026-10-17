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
"""CPU contracts for the adapter over the upstream MegaMoe parameter API."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch
from torch import nn

from torch.nn import functional as F

from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Experts, DeepseekV4SparseMoeBlock

from hyper_parallel.components.checkpoint.weight_conversion import revert_weight_conversion
from hyper_parallel.models.replacement import (
    apply_module_replacements, compile_module_replacements,
)
from hyper_parallel.trainer.config.parser import parse_training_args
from hyper_parallel.trainer.config.trainer import TrainerConfig

from hyper_parallel.core.multicore import MegaMoeExperts
from hyper_parallel.distributed._builder.forward_rewriter import validate_local_compute_signature
from hyper_parallel.models.deepseek_v41.adapter.distributed.megamoe import (
    DeepseekV41MegaMoeExperts,
)

from hyper_parallel.models.deepseek_v41.adapter.distributed.moe_engram_expert_parallel import (
    deepseek_v41_megamoe_compute_fn, deepseek_v41_ep_compute_fn,
)
from hyper_parallel.models.deepseek_v41.adapter.distributed import moe_engram_expert_parallel as ep_adapter
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41TopKRouter
from hyper_parallel.trainer.config import entries_to_module_replacements
from tests.common.mark_utils import arg_mark


# Packaged CPU tests do not include repository-only training examples.
_RECIPE_YAML = """
model:
  _target_: torch.nn.Identity
model_init_dtype: float32
optimizer:
  _target_: hyper_parallel.components.optim.Muon
  fp32_main_params: true
fsdp_config:
  mix_precision:
    param_dtype: bfloat16
    reduce_dtype: float32
    cast_forward_inputs: false
plan_overrides:
  - match: model.layers.*.mlp.experts
    module_type: transformers.models.deepseek_v4.modeling_deepseek_v4.DeepseekV4Experts
    replace_module:
      _target_: hyper_parallel.models.deepseek_v41.adapter.distributed.megamoe.DeepseekV41MegaMoeExperts
      local_num_tokens: 4096
      dispatch_mode: push
  - match: [model.layers.*.attn_hc, model.layers.*.ffn_hc]
    module_type: hyper_parallel.models.deepseek_v41.modeling_deepseek_v41.DeepseekV41PipelinedHyperConnection
    replace_module:
      _target_: hyper_parallel.models.deepseek_v41.adapter.distributed.mega_mhc.DeepseekV41HyperMegaMhc
  - match: model.layers.*.mlp.gate
    module_type: hyper_parallel.models.deepseek_v41.modeling_deepseek_v41.DeepseekV41TopKRouter
    replace_module:
      _target_: hyper_parallel.models.deepseek_v41.adapter.distributed.mega_gate.DeepseekV41MegaGate
  - match: model.layers.*.engram
    module_type: hyper_parallel.models.deepseek_v41.modeling_deepseek_v41.DeepseekV41Engram
    replace_module:
      _target_: hyper_parallel.components.modules.engram.EngramModule
  - match: model.layers.*.self_attn
    module_type: hyper_parallel.models.deepseek_v41.modeling_deepseek_v41.DeepseekV41Attention
    replace_module:
      _target_: hyper_parallel.components.modules.shared_compressed_dsa_attention.SharedCompressedDSAAttention
  - match: model.layers.*.mlp
    region_dispatch: false
    local_compute_fn:
      _target_: >-
        hyper_parallel.models.deepseek_v41.adapter.distributed.moe_engram_expert_parallel.deepseek_v41_megamoe_compute_fn
"""
_VISION_OVERRIDE_YAML = """
  - match: model.vision.blocks.*.attn
    module_type: hyper_parallel.models.deepseek_v41.vision.DeepseekV41VisionAttention
    replace_module:
      _target_: hyper_parallel.models.deepseek_v41.adapter.distributed.encoder_parallel.replace_vision_attention
"""


def _parse_recipe(multimodal: bool) -> TrainerConfig:
    """Resolve the installed package's text/VLM replacement and precision contract."""
    with TemporaryDirectory() as directory:
        path = Path(directory) / 'train.yaml'
        path.write_text(_RECIPE_YAML + (_VISION_OVERRIDE_YAML if multimodal else ''), encoding='utf-8')
        return parse_training_args([str(path)])


def _source(limit=10.0):
    source = nn.Module()
    source.num_experts, source.hidden_dim, source.intermediate_dim = 4, 8, 4
    source.limit, source.act_fn = limit, nn.SiLU()
    source.gate_up_proj = nn.Parameter(torch.randn(4, 8, 8))
    source.down_proj = nn.Parameter(torch.randn(4, 8, 4))
    return source

def _native_reference(module, inputs, indices, weights, *, expert_weights=None):
    """Differentiate the routed expert math independently of the native executor."""
    gate_up, down_weight = expert_weights or (module.gate_up_weight, module.down_weight)
    original_shape = inputs.shape
    inputs = inputs.reshape(-1, inputs.shape[-1])
    result = torch.zeros_like(inputs)
    for slot in range(indices.shape[1]):
        up = torch.bmm(inputs.unsqueeze(1), gate_up[indices[:, slot]]).squeeze(1)
        gate, value = up.chunk(2, dim=-1)
        if module.swiglu_limit is not None:
            gate = gate.clamp(max=module.swiglu_limit)
            value = value.clamp(-module.swiglu_limit, module.swiglu_limit)
        act = F.silu(gate) * value
        down = torch.bmm(act.unsqueeze(1), down_weight[indices[:, slot]]).squeeze(1)
        result = result + down * weights[:, slot, None]
    return result.reshape(original_shape)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
class TestDeepseekV41MegaMoe(unittest.TestCase):
    """Exercise target replacement, dynamic token execution and Trainer ownership."""

    # White-box assertions verify executor ownership and parameter lifetimes.
    # pylint: disable=protected-access

    def test_ordered_ep_subgroup_binds_without_world_size_assumption(self) -> None:
        """Expert FSDP may add ranks outside an ordered EP subgroup."""
        experts = DeepseekV41MegaMoeExperts(module=_source(), local_num_tokens=128)
        self.addCleanup(experts.close_runtime)
        group = object()
        mesh = MagicMock()
        mesh.get_group.return_value = group
        mesh.__getitem__.return_value.size.return_value = 2
        module = SimpleNamespace(is_hash=False, experts=experts, gate=SimpleNamespace(top_k=2),
                                 shared_experts=SimpleNamespace(limit=10.0),
                                 forward=lambda hidden_states, input_ids=None: hidden_states)
        with (patch.object(ep_adapter.dist, 'is_initialized', return_value=True),
              patch.object(ep_adapter.dist, 'get_world_size', return_value=2) as world_size,
              patch('hyper_parallel.core.multicore.modules.mega_moe.module.shmem.root_members', return_value=(1, 3))):
            deepseek_v41_megamoe_compute_fn(module=module, mesh=None, tp_mesh=None, cp_mesh=None, ep_mesh=mesh)
        world_size.assert_called_once_with(group)
        self.assertIs(experts._executor._ep_group, group)
        self.assertEqual(experts._executor.max_local_num_tokens, 128)

    def test_declared_replacements_preserve_base_modules_and_precision(self) -> None:
        """Resolve text/VLM YAML fixtures with paired expert and compute targets."""
        for multimodal in (False, True):
            with self.subTest(multimodal=multimodal):
                mega = _parse_recipe(multimodal)
                replacements = entries_to_module_replacements(mega.plan_overrides)
                matches = {match for rule in replacements for match in rule.match}
                expected = {'model.layers.*.mlp.experts', 'model.layers.*.attn_hc', 'model.layers.*.ffn_hc',
                            'model.layers.*.mlp.gate', 'model.layers.*.engram', 'model.layers.*.self_attn'}
                if multimodal:
                    expected.add('model.vision.blocks.*.attn')
                self.assertEqual(matches, expected)
                experts = [entry for entry in mega.plan_overrides if entry.replace_module is not None
                           and entry.replace_module.callable is DeepseekV41MegaMoeExperts]
                self.assertEqual(len(experts), 1)
                self.assertEqual(experts[0].replace_module.local_num_tokens, 4096)
                self.assertEqual(experts[0].replace_module.dispatch_mode, 'push')
                compute = [entry for entry in mega.plan_overrides if entry.local_compute_fn is not None
                           and entry.local_compute_fn.callable is deepseek_v41_megamoe_compute_fn]
                self.assertEqual(len(compute), 1)
                self.assertEqual(compute[0].match, 'model.layers.*.mlp')
                self.assertIsNone(compute[0].when)
                self.assertEqual(mega.model_init_dtype, 'float32')
                self.assertEqual(mega.fsdp_config.mix_precision.param_dtype, 'bfloat16')
                self.assertEqual(mega.fsdp_config.mix_precision.reduce_dtype, 'float32')
                self.assertFalse(mega.fsdp_config.mix_precision.cast_forward_inputs)
                self.assertTrue(mega.optimizer.fp32_main_params)

    def test_external_weights_preserve_gradients_and_state(self) -> None:
        """Use current tensors across calls and never register placeholder weights."""
        torch.manual_seed(9)
        source = _source()
        experts = DeepseekV41MegaMoeExperts(module=source, local_num_tokens=128)
        experts.configure(None, 1, 2)
        self.addCleanup(experts.close_runtime)
        self.assertEqual(set(experts.state_dict()), {"gate_up_proj", "down_proj"})
        self.assertEqual(len(list(experts.parameters())), 2)
        kernel = experts._executor
        self.assertIsNone(kernel.gate_up_weight)
        for step in range(2):
            with self.subTest(step=step):
                inputs = (torch.randn(128, 8) * 5).requires_grad_()
                weights = torch.randn(128, 2, requires_grad=True)
                ids = torch.arange(256).reshape(128, 2) % 4
                reference = SimpleNamespace(gate_up_weight=source.gate_up_proj.transpose(1, 2),
                                            down_weight=source.down_proj.transpose(1, 2), swiglu_limit=10.0)
                expected = _native_reference(reference, inputs, ids, weights)
                with patch.object(MegaMoeExperts, "forward", _native_reference):
                    actual = experts(inputs, ids, weights)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                derivative = torch.randn_like(actual)
                grads = torch.autograd.grad(actual, (inputs, weights, experts.gate_up_proj, experts.down_proj),
                                            derivative, retain_graph=True)
                ref_grads = torch.autograd.grad(expected, (inputs, weights, source.gate_up_proj, source.down_proj),
                                                derivative)
                for left, right in zip(grads[:2], ref_grads[:2]):
                    torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-5)
                for left, right in zip(grads[2:], ref_grads[2:]):
                    torch.testing.assert_close(left, right.transpose(1, 2), rtol=1e-5, atol=1e-5)
                self.assertIsNone(kernel.gate_up_weight)
                with torch.no_grad():
                    source.gate_up_proj.add_(0.1)
                    experts.gate_up_proj.copy_(source.gate_up_proj.transpose(1, 2))
        self.assertEqual(len(experts.make_transforms()), 2)

    def test_exception_does_not_retain_external_parameters(self) -> None:
        """An execution failure must not retain FSDP unsharded tensor views."""
        experts = DeepseekV41MegaMoeExperts(module=_source(), local_num_tokens=128)
        experts.configure(None, 1, 2)
        self.addCleanup(experts.close_runtime)
        with patch.object(MegaMoeExperts, "forward", side_effect=RuntimeError("native failure")):
            with self.assertRaisesRegex(RuntimeError, "native failure"):
                experts(torch.zeros(128, 8), torch.zeros(128, 2, dtype=torch.int32), torch.ones(128, 2))
        self.assertIsNone(experts._executor.gate_up_weight)
        self.assertEqual(len(list(experts.parameters())), 2)

    def test_reject_mismatched_shared_limit_and_parallel_axes(self) -> None:
        """Fail before native initialization when source or topology differs."""
        experts = DeepseekV41MegaMoeExperts(module=_source(), local_num_tokens=128)
        self.addCleanup(experts.close_runtime)
        module = SimpleNamespace(is_hash=False, gate=Mock(), experts=experts, shared_experts=SimpleNamespace(limit=1.0))
        args = {"module": module, "mesh": None, "tp_mesh": None, "cp_mesh": None, "ep_mesh": None}
        with self.assertRaisesRegex(ValueError, "same swiglu_limit"):
            deepseek_v41_megamoe_compute_fn(**args)
        module.shared_experts.limit = 10.0
        args["tp_mesh"] = SimpleNamespace(size=lambda: 2)
        with self.assertRaisesRegex(ValueError, "TP>1 requires EP>1"):
            deepseek_v41_megamoe_compute_fn(**args)

    def test_disabled_ep_retains_native_dispatch_and_router(self) -> None:
        """The original compute target retains the existing EP helpers."""
        experts = SimpleNamespace(_apply_gate=object(), limit=10.0)
        module = SimpleNamespace(is_hash=False, gate=Mock(), experts=experts,
                                 shared_experts=lambda value: value + 1,
                                 forward=lambda hidden_states, input_ids=None: hidden_states)
        mesh = Mock()
        mesh.get_group.return_value = "ep-group"
        mesh.__getitem__ = Mock(return_value=SimpleNamespace(size=lambda: 2))
        values = torch.randn(3, 4)
        with (patch.object(ep_adapter, "bind_local_expert_forward") as bind,
              patch.object(ep_adapter, "ep_routed_forward",
                    return_value=values * 2) as route):
            compute = deepseek_v41_ep_compute_fn(module=module, mesh=None, tp_mesh=None,
                                                 cp_mesh=None, ep_mesh=mesh)
            result = compute(module, values)
            torch.testing.assert_close(result, values * 3 + 1)
            bind.assert_called_once_with(module, 2, use_grouped_gemm=False, apply_gate=experts._apply_gate)
            self.assertEqual(route.call_args.kwargs["ep_group"], "ep-group")

    def test_multimodal_router_and_shared_expert_gradients(self) -> None:
        """Use real image/text router biases and propagate image-token gradients through variable-length experts."""
        torch.manual_seed(81)
        gate = DeepseekV41TopKRouter(SimpleNamespace(
            hidden_size=8, num_local_experts=4, num_experts_per_tok=2, scoring_func="sigmoid",
            routed_scaling_factor=1.0, v41_vision_enabled=True,
        ))
        with torch.no_grad():
            gate.weight.normal_(std=0.1)
            gate.bias.copy_(torch.tensor([2., 2., 0., 0.]))
            gate.bias_vl.copy_(torch.tensor([0., 0., 2., 2.]))
        experts = DeepseekV41MegaMoeExperts(module=_source(), local_num_tokens=128)
        self.addCleanup(experts.close_runtime)
        shared = nn.Linear(8, 8)
        shared.limit = 10.0
        module = SimpleNamespace(experts=experts, shared_experts=shared, gate=gate, is_hash=False,
                                 forward=lambda hidden_states, input_ids=None, image_mask=None: hidden_states)
        compute = deepseek_v41_megamoe_compute_fn(module=module, mesh=None, tp_mesh=None, cp_mesh=None,
                                             ep_mesh=None)
        validate_local_compute_signature(compute, module.forward, owner='VLM MegaMoe')
        hidden = (torch.randn(1, 18, 8) * 5).requires_grad_()
        image_mask = torch.zeros(1, 18, dtype=torch.bool)
        image_mask[:, 3:8] = True
        _, weights, ids = gate(hidden, image_mask)
        self.assertTrue(torch.all(ids[image_mask.flatten()] >= 2))
        self.assertTrue(torch.all(ids[~image_mask.flatten()] < 2))
        expected = _native_reference(experts._executor, hidden.reshape(-1, 8), ids, weights,
                                     expert_weights=(experts.gate_up_proj, experts.down_proj)).reshape_as(hidden)
        expected = expected + shared(hidden)
        with patch.object(MegaMoeExperts, "forward", _native_reference):
            actual = compute(module, hidden, image_mask=image_mask)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        targets = (hidden, gate.weight, experts.gate_up_proj, experts.down_proj, shared.weight, shared.bias)
        gradient = torch.randn_like(actual)
        actual_grads = torch.autograd.grad(actual, targets, gradient)
        expected_grads = torch.autograd.grad(expected, targets, gradient)
        for left, right in zip(actual_grads, expected_grads):
            torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-5)
        self.assertGreater(actual_grads[0][image_mask].abs().sum().item(), 0)

    def test_text_forward_signature_and_gradients(self) -> None:
        """Apply the real text MoE contract and preserve routed/shared gradients."""
        torch.manual_seed(82)
        config = DeepseekV4Config.from_dict({
            'hidden_size': 8, 'moe_intermediate_size': 4, 'num_hidden_layers': 1,
            'num_local_experts': 4, 'num_experts_per_tok': 2, 'mlp_layer_types': ['moe'],
            'swiglu_limit': 10.0, 'scoring_func': 'sigmoid', 'routed_scaling_factor': 1.0,
        })
        config.v41_vision_enabled = False
        module = DeepseekV4SparseMoeBlock(config, 0)
        module.gate = DeepseekV41TopKRouter(config)
        for parameter in module.parameters():
            nn.init.normal_(parameter, std=0.1)
        module.experts = DeepseekV41MegaMoeExperts(module=module.experts, local_num_tokens=128)
        self.addCleanup(module.experts.close_runtime)
        compute = deepseek_v41_megamoe_compute_fn(
            module=module, mesh=None, tp_mesh=None, cp_mesh=None, ep_mesh=None,
        )
        validate_local_compute_signature(compute, module.forward, owner='text MegaMoe')
        hidden = torch.randn(1, 7, 8, requires_grad=True)
        input_ids = torch.arange(7).unsqueeze(0)
        _, weights, ids = module.gate(hidden)
        expected = _native_reference(
            module.experts._executor, hidden, ids, weights,
            expert_weights=(module.experts.gate_up_proj, module.experts.down_proj),
        ) + module.shared_experts(hidden)
        with patch.object(MegaMoeExperts, 'forward', _native_reference):
            actual = compute(module, hidden, input_ids=input_ids)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        targets = (hidden, module.gate.weight, module.experts.gate_up_proj, module.experts.down_proj,
                   *module.shared_experts.parameters())
        derivative = torch.randn_like(actual)
        actual_grads = torch.autograd.grad(actual, targets, derivative)
        expected_grads = torch.autograd.grad(expected, targets, derivative)
        for left, right in zip(actual_grads, expected_grads):
            torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-5)

    def test_real_token_output_and_all_gradients(self) -> None:
        """Real token rows reach the executor unchanged and preserve every gradient."""
        for tokens in (0, 7, 128):
            with self.subTest(tokens=tokens):
                experts = DeepseekV41MegaMoeExperts(module=_source(), local_num_tokens=128)
                experts.configure(None, 1, 2)
                self.addCleanup(experts.close_runtime)
                hidden = torch.randn(tokens, 8, requires_grad=True)
                ids = torch.arange(tokens * 2).reshape(tokens, 2) % 4
                weights = torch.randn(tokens, 2, requires_grad=True)
                expected = _native_reference(experts._executor, hidden, ids, weights,
                                             expert_weights=(experts.gate_up_proj, experts.down_proj))
                with patch.object(MegaMoeExperts, 'forward', autospec=True, side_effect=_native_reference) as run:
                    actual = experts(hidden, ids, weights)
                self.assertIs(run.call_args.args[1], hidden)
                self.assertIs(run.call_args.args[2], ids)
                self.assertIs(run.call_args.args[3], weights)
                torch.testing.assert_close(actual, expected)
                params = (hidden, weights, experts.gate_up_proj, experts.down_proj)
                actual_grads = torch.autograd.grad(actual.sum(), params)
                expected_grads = torch.autograd.grad(expected.sum(), params)
                for left, right in zip(actual_grads, expected_grads):
                    torch.testing.assert_close(left, right)

    def test_declarative_config_and_checkpoint_roundtrip(self) -> None:
        """Resolve both fixtures through the actual YAML Target replacement executor."""
        for multimodal in (False, True):
            with self.subTest(multimodal=multimodal):
                config = _parse_recipe(multimodal)
                self.assertNotIn('megamoe', config.to_dict())
                entry = next(entry for entry in config.plan_overrides
                             if entry.replace_module is not None
                             and entry.replace_module.callable is DeepseekV41MegaMoeExperts)
                rules = entries_to_module_replacements([entry])
                model = nn.Module()
                model.config = DeepseekV4Config.from_dict({
                    'hidden_size': 32, 'moe_intermediate_size': 16,
                    'num_local_experts': 4, 'num_experts_per_tok': 2, 'swiglu_limit': 10.,
                })
                model.model = nn.Module()
                model.model.layers = nn.ModuleList([nn.Module()])
                model.model.layers[0].mlp = nn.Module()
                model.model.layers[0].mlp.experts = DeepseekV4Experts(model.config)
                for parameter in model.parameters():
                    nn.init.normal_(parameter, std=0.02)
                original = {key: value.clone() for key, value in model.state_dict().items()}
                conversions = []
                apply_module_replacements(model, compile_module_replacements(model, rules),
                                          weights_mapping=conversions)
                self.assertIsInstance(model.model.layers[0].mlp.experts, DeepseekV41MegaMoeExperts)
                model._weight_conversions = conversions
                restored = revert_weight_conversion(model, dict(model.state_dict()))
                for name, value in original.items():
                    torch.testing.assert_close(restored[name], value, rtol=0, atol=0)
                with self.assertRaisesRegex(RuntimeError, 'Configure every expert'):
                    DeepseekV41MegaMoeExperts.prepare_runtime_group([model.model.layers[0].mlp.experts])

    def test_dispatch_transports_and_capacity_options(self) -> None:
        """Preserve push growth settings and reject push-only factors for pull."""
        cases = (
            ({}, ("push", 1.25, 1.25)),
            ({"dispatch_mode": "push", "initial_capacity_factor": 2.0,
              "capacity_growth_factor": 1.5}, ("push", 2.0, 1.5)),
            ({"dispatch_mode": "pull"}, ("pull", None, None)),
        )
        for options, expected in cases:
            with self.subTest(options=options):
                experts = DeepseekV41MegaMoeExperts(module=_source(), local_num_tokens=128, **options)
                self.addCleanup(experts.close_runtime)
                experts.configure(None, 1, 2)
                executor = experts._executor
                self.assertEqual((executor.dispatch_mode, executor.initial_capacity_factor,
                                  executor.capacity_growth_factor), expected)
                self.assertEqual(executor.swiglu_limit, 10.0)
                self.assertEqual(set(experts.state_dict()), {"gate_up_proj", "down_proj"})
        for option in ("initial_capacity_factor", "capacity_growth_factor"):
            with self.subTest(option=option), self.assertRaisesRegex(ValueError, "only supported for push"):
                experts = DeepseekV41MegaMoeExperts(module=_source(), local_num_tokens=128,
                                                   dispatch_mode="pull", **{option: 1.5})
                experts.configure(None, 1, 2)

    def test_meta_capacity_topology_and_incorrect_pairing(self) -> None:
        """Reject invalid capacities/topology before any native resources are acquired."""
        with torch.device('meta'):
            experts = DeepseekV41MegaMoeExperts(module=_source(), local_num_tokens=256)
        self.assertTrue(experts.gate_up_proj.is_meta)
        experts.to_empty(device='cpu')
        experts.reset_parameters()
        self.assertTrue(torch.isfinite(experts.gate_up_proj).all())
        for capacity in (0, 127, -128, True):
            with self.subTest(capacity=capacity), self.assertRaises(ValueError):
                DeepseekV41MegaMoeExperts(module=_source(), local_num_tokens=capacity)
        for axis, message in (('tp', 'EP>1'), ('pp', 'PP=1')):
            with self.subTest(axis=axis), self.assertRaisesRegex(ValueError, message):
                DeepseekV41MegaMoeExperts(module=_source(), context={axis: True})
        module = SimpleNamespace(is_hash=False, gate=Mock(), experts=_source(), shared_experts=Mock())
        with self.assertRaisesRegex(TypeError, 'module replacement'):
            deepseek_v41_megamoe_compute_fn(module=module, mesh=None, tp_mesh=None, cp_mesh=None, ep_mesh=None)

    def test_configured_experts_share_runtime_resources(self) -> None:
        """Configured experts share workspace and retain explicit cleanup."""
        model = nn.ModuleList([DeepseekV41MegaMoeExperts(module=_source(), local_num_tokens=128) for _ in range(2)])
        for experts in model:
            self.addCleanup(experts.close_runtime)
            experts.configure(None, 1, 2)
        DeepseekV41MegaMoeExperts.prepare_runtime_group(list(model))
        self.assertIs(model[0]._executor._resource_group, model[1]._executor._resource_group)
        self.assertTrue(all(experts._executor is not None for experts in model))
        for experts in model:
            experts.close_runtime()
        self.assertTrue(all(experts._executor is None for experts in model))

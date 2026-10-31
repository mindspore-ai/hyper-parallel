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
"""CPU training-lifecycle tests of the Xing4.0 model."""

import math
from collections.abc import Callable
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn

from hyper_parallel.components.functional.aux_loss import aux_loss_scale_context, bind_aux_loss_scale
from hyper_parallel.components.modules.moe_utils import MoEMonitorCallback
from hyper_parallel.core.activation_memory.recompute_state import create_recompute_contexts
from hyper_parallel.distributed import ShardingPlanner
from hyper_parallel.models import HyperAutoModelForCausalLM
from hyper_parallel.models.replacement import apply_module_replacements, compile_module_replacements
from hyper_parallel.models.xing40.adapter.builder import build_model
from hyper_parallel.models.xing40.adapter.distributed.expert_parallel import xing40_ep_compute_fn
from hyper_parallel.models.xing40.modeling_xing40 import (
    Xing40Config,
    Xing40ForCausalLM,
    Xing40MoE,
)
from hyper_parallel.trainer.config import entries_to_module_replacements, entries_to_plan_overrides
from hyper_parallel.trainer.config.parser import parse_training_args
from tests.common.mark_utils import arg_mark
from tests.ut.auto_models.distributed.conftest import FakeDeviceMesh

TINY_CONFIG = {
    "vocab_size": 128, "hidden_size": 64, "intermediate_size": 96, "moe_intermediate_size": 32,
    "num_hidden_layers": 3, "first_k_dense_replace": 1, "num_attention_heads": 4, "num_key_value_heads": 4,
    "q_lora_rank": 32, "kv_lora_rank": 16, "qk_rope_head_dim": 8, "qk_nope_head_dim": 8, "v_head_dim": 8,
    "n_routed_experts": 4, "num_experts_per_tok": 2, "n_shared_experts": 1, "n_group": 1, "topk_group": 1,
    "moe_aux_loss_coeff": 1e-4, "moe_router_enable_expert_bias": True, "moe_router_bias_update_rate": 1e-3,
    "use_pad_tokens": True, "num_residual_streams": 4, "mhc_sinkhorn_iterations": 20,
    "mhc_init_gating_factor": 0.01, "mhc_layernorm_epsilon": 1e-6, "num_nextn_predict_layers": 1,
    "mtp_loss_factor": 0.3, "chunk_loss_size": 8,
}
_TO_EMPTY = nn.Module.to_empty


def _config(**overrides) -> Xing40Config:
    """Return the tiny configuration with ``overrides`` applied."""
    return Xing40Config(**{**TINY_CONFIG, **overrides})  # pylint: disable=unexpected-keyword-arg


def _model(**overrides) -> Xing40ForCausalLM:
    """Build the tiny model with fixed random weights."""
    torch.manual_seed(0)
    return Xing40ForCausalLM(_config(**overrides))


def _train_step(model: Xing40ForCausalLM) -> dict[str, torch.Tensor]:
    """Run one forward/backward on a fixed sequence and return the named losses."""
    generator = torch.Generator().manual_seed(1)
    input_ids = torch.randint(0, TINY_CONFIG["vocab_size"], (1, 16), generator=generator)
    losses = model(input_ids, shift_labels=torch.roll(input_ids, -1, dims=1)).loss
    torch.stack(list(losses.values())).sum().backward()
    return losses


def _poisoned_to_empty(module: nn.Module, *, device, recurse: bool = True) -> nn.Module:
    """Materialize like ``to_empty`` but fill every float tensor with NaN, as arbitrary memory may hold."""
    _TO_EMPTY(module, device=device, recurse=recurse)
    with torch.no_grad():
        for tensor in (*module.parameters(), *module.buffers()):
            if tensor.is_floating_point():
                tensor.fill_(float("nan"))
    return module


class TestXing40Training(unittest.TestCase):
    """Initialization, objective coverage and router-statistics lifecycle."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_attention_scale_is_independent_of_rotary_frequencies(self):
        """
        Feature: Explicit attention score scale with ordinary rotary embeddings.
        Description: Save/reload config and run attention through SDPA with the explicit score scale.
        Expectation: The requested scale is applied once, with unchanged frequencies and default behavior.
        """
        scale = (1.0 + 0.1 * math.log(40.0)) ** 2 / math.sqrt(16)
        default = _model()
        config = _config(attention_softmax_scale=scale)
        restored = Xing40Config.from_dict(config.to_dict())
        model = Xing40ForCausalLM(restored)
        torch.testing.assert_close(model.model.rotary_emb.inv_freq, default.model.rotary_emb.inv_freq,
                                   rtol=0, atol=0)
        self.assertEqual(default.model.layers[0].self_attn.scaling, 16 ** -0.5)
        self.assertEqual(model.mtp.layers[0].transformer_layer.self_attn.scaling, scale)
        source = model.model.layers[0].self_attn.double()
        source.config._attn_implementation = "sdpa"
        inputs = torch.linspace(-1, 1, 3 * 64, dtype=torch.float64).reshape(1, 3, 64).requires_grad_()
        cos = torch.ones(1, 3, TINY_CONFIG["qk_rope_head_dim"], dtype=torch.float64)
        with mock.patch("torch.nn.functional.scaled_dot_product_attention",
                        wraps=nn.functional.scaled_dot_product_attention) as attention:
            actual, _ = source(inputs, (cos, torch.zeros_like(cos)), attention_mask=None)
        attention.assert_called_once()
        self.assertEqual(attention.call_args.kwargs["scale"], scale)
        actual.square().sum().backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertGreater(inputs.grad.abs().sum().item(), 0.0)
        for invalid in (0.0, -1.0, float("inf"), float("nan"), True, "0.5"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "attention_softmax_scale"):
                _config(attention_softmax_scale=invalid)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_yarn_attention_scale_default_and_override(self):
        """The YaRN score scale is applied to backbone and MTP unless explicitly overridden."""
        # Reference: (1 + 0.1 * ln(factor)) ** 2 / sqrt(8 + 8).
        cases = ((1.0, None, 0.25), (40.0, None, 0.46846355177315665), (40.0, 0.375, 0.375))
        for factor, override, expected in cases:
            with self.subTest(scaling_factor=factor, override=override):
                model = _model(scaling_factor=factor, attention_softmax_scale=override)
                for attention in (model.model.layers[0].self_attn,
                                  model.mtp.layers[0].transformer_layer.self_attn):
                    self.assertAlmostEqual(attention.scaling, expected, places=14,
                                           msg=f"factor={factor}, override={override}, "
                                               f"expected={expected}, got={attention.scaling}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_recipe_module_replacements_match_tiny_xing40(self):
        """The shipped recipe parses and its three module rules select the intended modules."""
        recipe = Path(__file__).resolve().parents[4] / "hyper_parallel/models/xing40/recipes/train.yaml"
        config = parse_training_args([str(recipe)])
        entries = [entry for entry in config.plan_overrides if entry.replace is not None
                   and entry.replace.kind == "module"]
        targets = {tuple(entry.match) if isinstance(entry.match, list) else (entry.match,):
                   entry.replace.implementation.to_dict()["_target_"] for entry in entries}
        expected_targets = {
            ("*.self_attn",): "hyper_parallel.models.xing40.adapter.conversion.module_replacement.Xing40MLAAttention",
            ("*[!a]_layernorm", "*[.eh]norm"): "hyper_parallel.components.modules.rms_norm.RMSNorm",
            ("*.mlp.experts",): "hyper_parallel.components.modules.GroupedExperts",
        }
        self.assertEqual(targets, expected_targets, f"expected={expected_targets}, got={targets}")
        model = _model(scaling_factor=40.0)
        expected_scale = model.model.layers[0].self_attn.scaling
        plan = compile_module_replacements(model, entries_to_module_replacements(config.plan_overrides))
        selected = {name for target in plan.targets for name in target.module_fqns}
        expected_names = {
            "model.layers.0.self_attn", "model.layers.0.input_layernorm", "model.norm",
            "model.layers.2.mlp.experts", "mtp.layers.0.enorm", "mtp.layers.0.transformer_layer.mlp.experts",
        }
        self.assertTrue(expected_names <= selected, f"expected={expected_names}, got={selected}")
        for name in ("model.layers.0.self_attn.q_a_layernorm", "model.layers.0.self_attn.kv_a_layernorm"):
            self.assertNotIn(name, selected, f"nested latent norm unexpectedly selected: {name}")

        factories = {tuple(entry.match) if isinstance(entry.match, list) else (entry.match,):
                     entry.replace.implementation.callable for entry in entries}
        installed, _ = apply_module_replacements(model, plan, weights_mapping=[])
        self.assertIs(installed, model, f"expected model={model}, got={installed}")
        for name, patterns in (
                ("model.layers.0.self_attn", ("*.self_attn",)),
                ("model.layers.0.input_layernorm", ("*[!a]_layernorm", "*[.eh]norm")),
                ("model.layers.2.mlp.experts", ("*.mlp.experts",))):
            actual = model.get_submodule(name)
            self.assertIsInstance(actual, factories[patterns],
                                  f"expected={factories[patterns]}, got={type(actual)} for {name}")
        for attention in (model.model.layers[0].self_attn,
                          model.mtp.layers[0].transformer_layer.self_attn):
            self.assertEqual(attention.scaling, expected_scale,
                             f"expected scale={expected_scale}, got={attention.scaling}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_recipe_ep_forward_rules_reach_planner(self):
        """EP rules are inactive at degree one and reach backbone and MTP planner boundaries at degree two."""
        recipe = Path(__file__).resolve().parents[4] / "hyper_parallel/models/xing40/recipes/train.yaml"
        config = parse_training_args([str(recipe)])
        entries = [entry for entry in config.plan_overrides if entry.replace is not None
                   and entry.replace.kind == "method"]
        expected = {"model.layers.[2-9].mlp", "model.layers.[1-9][0-9].mlp",
                    "mtp.layers.*.transformer_layer.mlp"}
        inactive = entries_to_plan_overrides(entries, ep_size=1)
        active = entries_to_plan_overrides(entries, ep_size=2)
        self.assertEqual(inactive, {}, f"expected no EP rules, got={inactive}")
        self.assertEqual(set(active), expected, f"expected={expected}, got={set(active)}")
        for name, spec in active.items():
            self.assertIs(spec.local_compute_fn.callable, xing40_ep_compute_fn, f"wrong EP target for {name}")
            self.assertIs(spec.region_dispatch, False, f"wrong dispatch mode for {name}")

        # A metadata-only mesh exercises planner matching without initializing a process group or device.
        model = _model(num_hidden_layers=11, first_k_dense_replace=2)
        replacements = compile_module_replacements(model, entries_to_module_replacements(config.plan_overrides))
        model, _ = apply_module_replacements(model, replacements, weights_mapping=[])
        overrides = entries_to_plan_overrides(config.plan_overrides, ep_size=2)
        plan = ShardingPlanner(plan_overrides=overrides).plan(model, FakeDeviceMesh((2,), ("dp",)), ep_size=2)
        for name in ("model.layers.2.mlp", "model.layers.10.mlp", "mtp.layers.0.transformer_layer.mlp"):
            spec = plan.modules[name]
            self.assertIs(spec.local_compute_fn.callable, xing40_ep_compute_fn, f"wrong EP target for {name}")
            self.assertIs(spec.region_dispatch, False, f"wrong dispatch mode for {name}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_meta_materialization_initializes_every_tensor(self):
        """
        Feature: Meta-device construction through the recipe's builder.
        Description: to_empty() storage is NaN; the framework then runs the random initialization.
        Expectation: No tensor, RoPE tables included, stays NaN and the mHC gates take their reference
            value; mHC shifts are zeroed (checkpoint-dependent, no from-scratch reference).
        """
        with mock.patch.object(nn.Module, "to_empty", _poisoned_to_empty):
            model = build_model(config=TINY_CONFIG)
        uninitialized = [name for name, tensor in (*model.named_parameters(), *model.named_buffers())
                         if tensor.is_floating_point() and torch.isnan(tensor).any()]
        self.assertEqual(uninitialized, [], f"uninitialized tensors: {uninitialized}")
        connection = model.mtp.layers[0].transformer_layer.ffn_hc_pre
        torch.testing.assert_close(connection.branch_alpha.detach(), torch.full((3,), 0.01),
                                   msg="case: gates take mhc_init_gating_factor")
        torch.testing.assert_close(connection.branch_beta.detach(), torch.zeros_like(connection.branch_beta),
                                   msg="case: shifts are zeroed, not NaN")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_pretrained_load_rebuilds_rope_tables(self):
        """
        Feature: Pretrained loading through the framework builder.
        Description: Save a model in the Transformers layout, then load it with NaN-filled to_empty() storage.
        Expectation: Every saved tensor loads exactly, and the RoPE tables, which checkpoints do not hold, are
            rebuilt.
        """
        source = _model()
        with tempfile.TemporaryDirectory() as directory:
            source.save_pretrained(directory)
            with mock.patch.object(nn.Module, "to_empty", _poisoned_to_empty):
                loaded = HyperAutoModelForCausalLM.from_pretrained(directory)
        expected = source.state_dict()
        mismatched = [name for name, tensor in loaded.state_dict().items() if not torch.equal(tensor, expected[name])]
        self.assertEqual(mismatched, [], f"tensors that did not load: {mismatched}")
        torch.testing.assert_close(loaded.model.rotary_emb.inv_freq, source.model.rotary_emb.inv_freq, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_objective_reaches_every_parameter(self):
        """
        Feature: Model-computed training objective.
        Description: One step through the LM and MTP losses and the injected router balance loss.
        Expectation: Both losses are finite and every trainable parameter, MTP included, has a finite gradient.
        """
        model = _model()
        losses = _train_step(model)
        self.assertEqual(sorted(losses), ["foundation_loss", "foundation_loss/mtp"], f"losses: {sorted(losses)}")
        for name, loss in losses.items():
            self.assertTrue(torch.isfinite(loss), f"{name}={loss}")
        missing = [name for name, parameter in model.named_parameters()
                   if parameter.grad is None or not torch.isfinite(parameter.grad).all()]
        self.assertEqual(missing, [], f"parameters without a finite gradient: {missing}")
        mtp_gradient = model.mtp.layers[0].eh_proj.weight.grad.abs().sum()
        self.assertGreater(mtp_gradient.item(), 0.0, f"MTP projection gradient sum={mtp_gradient}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_mtp_depth_follows_mindformers_layout(self):
        """
        Feature: MindFormers multi-token-prediction layout.
        Description: Record the inputs and outputs of the backbone norm and the MTP depth's modules in one step.
        Expectation: The depth reads the backbone's final-normed state, projects [enorm(next-token embeddings),
            hnorm(state)] and collapses its streams with its own head.
        """
        model = _model()
        layer = model.mtp.layers[0]
        seen: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        def _recorder(key: str) -> Callable[[nn.Module, tuple, torch.Tensor], None]:
            """Return a forward hook that keeps the first input and output of one module."""
            def record(_: nn.Module, args: tuple, output: torch.Tensor) -> None:
                """Keep the first call's input and output under ``key``."""
                seen.setdefault(key, (args[0], output))
            return record

        modules = {"backbone": model.model.layers[0], "norm": model.model.norm,
                   "enorm": layer.enorm, "hnorm": layer.hnorm, "eh_proj": layer.eh_proj,
                   "mtp_head": layer.hc_head}
        hooks = [module.register_forward_hook(_recorder(key)) for key, module in modules.items()]
        _train_step(model)
        for hook in hooks:
            hook.remove()
        self.assertEqual(seen["backbone"][0].dtype, torch.bfloat16)
        self.assertEqual(seen["enorm"][0].dtype, torch.bfloat16)
        torch.testing.assert_close(seen["hnorm"][0], seen["norm"][1], rtol=0, atol=0,
                                   msg="case: the MTP depth reads the final-normed backbone state")
        torch.testing.assert_close(seen["eh_proj"][0], torch.cat((seen["enorm"][1], seen["hnorm"][1]), dim=-1),
                                   rtol=0, atol=0, msg="case: the fusion projects [embedding, state]")
        self.assertIsNot(layer.hc_head, model.model.hc_head, "the MTP depth must own its stream-collapse head")
        self.assertIn("mtp_head", seen, "the MTP depth must collapse its streams with its own head")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_balance_loss_reaches_router(self):
        """
        Feature: Sequence-wise router balance loss.
        Description: The same step with the balance loss disabled and enabled.
        Expectation: Only the router gradients change; the reported losses do not.
        """
        disabled, enabled = _model(moe_aux_loss_coeff=0.0), _model(moe_aux_loss_coeff=1.0)
        disabled_losses, enabled_losses = _train_step(disabled), _train_step(enabled)
        torch.testing.assert_close(enabled_losses["foundation_loss"], disabled_losses["foundation_loss"])
        router_delta = (enabled.model.layers[1].mlp.gate.weight.grad
                        - disabled.model.layers[1].mlp.gate.weight.grad).abs().max()
        self.assertGreater(router_delta.item(), 0.0, f"router gradient change={router_delta}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_expert_bias_updates_once_per_step(self):
        """
        Feature: Expert-load bias update by the Trainer's MoE monitor.
        Description: Replay a route under recomputation, then run the monitor's step-end update twice.
        Expectation: Replay does not count tokens; the first update uses rate*sign(mean-count),
            and the second finds no new tokens and changes nothing.
        """
        model = _model()
        _train_step(model)
        layers = [module for module in model.modules() if isinstance(module, Xing40MoE)]
        probes = [torch.zeros(1, layer.padding + 3, model.config.hidden_size) for layer in layers]
        forward_context, recompute_context = create_recompute_contexts()
        with torch.no_grad(), forward_context:
            for layer, probe in zip(layers, probes):
                layer.route(probe)
        counts = [layer.tokens_per_expert.clone() for layer in layers]
        with torch.no_grad(), recompute_context:
            for layer, probe in zip(layers, probes):
                layer.route(probe)
        for layer, count in zip(layers, counts):
            torch.testing.assert_close(layer.tokens_per_expert, count, rtol=0, atol=0)
        monitor = MoEMonitorCallback(model)
        monitor.on_step_end()
        for layer, count in zip(layers, counts):
            expected = (count.mean() - count).sign() * layer.expert_bias_update_rate
            torch.testing.assert_close(layer.gate.e_score_correction_bias, expected)
        monitor.on_step_end()
        for layer, count in zip(layers, counts):
            expected = (count.mean() - count).sign() * layer.expert_bias_update_rate
            torch.testing.assert_close(layer.gate.e_score_correction_bias, expected,
                                       msg="case: no new tokens leave the bias unchanged")


class TestXing40BatchedTraining(unittest.TestCase):
    """A microbatch keeps independent causal sequences and per-sequence router statistics."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_batched_objective_matches_independent_sequences(self):
        """
        Feature: Training with a microbatch larger than one.
        Description: Compare a four-sequence forward/backward to the mean of independent sequences.
        Expectation: LM/MTP losses, parameter gradients and real-token expert counts agree.
        """
        batched = _model(num_hidden_layers=2)
        independent = _model(num_hidden_layers=2)
        generator = torch.Generator().manual_seed(29)
        tokens = torch.randint(0, TINY_CONFIG["vocab_size"], (4, 9), generator=generator)
        targets = torch.roll(tokens, -1, dims=1)
        targets[:, -1] = -100
        with aux_loss_scale_context():
            actual = batched(tokens, shift_labels=targets).loss
            objective = torch.stack(list(actual.values())).sum()
            bind_aux_loss_scale(objective)
            objective.backward()
        expected = {name: torch.zeros_like(value) for name, value in actual.items()}
        for row in range(tokens.shape[0]):
            with aux_loss_scale_context():
                losses = independent(tokens[row:row + 1], shift_labels=targets[row:row + 1]).loss
                objective = torch.stack(list(losses.values())).sum()
                bind_aux_loss_scale(objective)
                (objective / tokens.shape[0]).backward()
            for name, value in losses.items():
                expected[name] += value.detach() / tokens.shape[0]
        for name, value in actual.items():
            torch.testing.assert_close(value, expected[name], rtol=1e-5, atol=1e-6,
                                       msg=f"case: {name} averages independent equal-length sequences")
        for (name, actual_parameter), (_, expected_parameter) in zip(
                batched.named_parameters(), independent.named_parameters()):
            # embed_tokens.weight is the only parameter whose forward value crosses the
            # unconditional BF16 cast (embed_inputs); batched and per-sequence calls round
            # independently, so its gradient needs a looser tolerance than the rest.
            grad_rtol = 3e-2 if name == "model.embed_tokens.weight" else 2e-4
            torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad, rtol=grad_rtol, atol=2e-6,
                                       msg=f"case: batched gradient of {name}")
        for actual_layer, expected_layer in zip(
                (m for m in batched.modules() if isinstance(m, Xing40MoE)),
                (m for m in independent.modules() if isinstance(m, Xing40MoE))):
            torch.testing.assert_close(actual_layer.tokens_per_expert, expected_layer.tokens_per_expert,
                                       rtol=0, atol=0, msg="case: padding contributes no expert counts")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_sequence_auxiliary_loss_does_not_pool_sequences(self):
        """
        Feature: Per-sequence auxiliary balancing in batched MoE routing.
        Description: Two sequences favor opposite experts; compare routes and derivatives to a float64 formula.
        Expectation: The auxiliary objective averages sequence-level products rather than pooled statistics.
        """
        layer = _model(use_pad_tokens=False).model.layers[1].mlp
        with torch.no_grad():
            layer.gate.weight.zero_()
            layer.gate.weight[:, 0] = torch.tensor([1.0, 2.0, -1.0, -2.0])
        hidden = torch.zeros(2, 3, TINY_CONFIG["hidden_size"])
        hidden[0, :, 0], hidden[1, :, 0] = 1.0, -1.0
        hidden.requires_grad_()
        padded = torch.cat((hidden.new_zeros(2, layer.padding, hidden.shape[-1]), hidden), dim=1)
        coefficients = torch.arange(12, dtype=hidden.dtype).reshape(2, 3, 2) / 10
        with aux_loss_scale_context():
            indices, weights = layer.route(padded)
            indices = indices.reshape(2, layer.padding + 3, layer.top_k)[:, layer.padding:]
            real_weights = weights.reshape(2, layer.padding + 3, layer.top_k)[:, layer.padding:]
            objective = (real_weights * coefficients).sum()
            bind_aux_loss_scale(objective)
            actual_gradients = torch.autograd.grad(objective, (hidden, layer.gate.weight))
        reference_hidden = hidden.detach().double().requires_grad_()
        reference_weight = layer.gate.weight.detach().double().requires_grad_()
        scores = torch.sigmoid(reference_hidden @ reference_weight.T)
        selected = scores.gather(-1, indices)
        expected_weights = selected / selected.sum(-1, keepdim=True) * layer.gate.routed_scaling_factor
        unbiased = scores.topk(layer.top_k, dim=-1, sorted=False).indices
        loads = torch.nn.functional.one_hot(  # pylint: disable=not-callable
            unbiased, layer.num_experts).double().sum(dim=(1, 2)) / 6
        affinities = (scores / scores.sum(-1, keepdim=True)).mean(dim=1)
        auxiliary = (loads * affinities).sum(-1).mean() * layer.num_experts
        pooled = (loads.mean(0) * affinities.mean(0)).sum() * layer.num_experts
        self.assertGreater(abs((auxiliary - pooled).item()), 0.1,
                           "case: opposing sequence loads distinguish sequence aux from pooled batch aux")
        reference_objective = (expected_weights * coefficients).sum() + auxiliary * layer.aux_loss_coeff
        expected_gradients = torch.autograd.grad(reference_objective, (reference_hidden, reference_weight))
        torch.testing.assert_close(real_weights.double(), expected_weights, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(layer.last_aux_loss.double(), auxiliary, rtol=1e-6, atol=1e-7)
        for actual, expected in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual.double(), expected, rtol=1e-5, atol=1e-7)


class TestXing40ForcedRouting(unittest.TestCase):
    """Forced balancing preserves affinities and differentiates the selected weights."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_forced_expert_ids_follow_sequence_major_numbering(self):
        """
        Feature: Fixed routing across multiple sequences.
        Description: Route four sequences with eight experts and top-2, independently of their affinities.
        Expectation: Sequence-major token IDs select each sequence's expert pair and gather its own weights.
        """
        layer = _model(n_routed_experts=8, use_pad_tokens=False, moe_aux_loss_coeff=0.0,
                       moe_router_force_expert_balance=True).model.layers[1].mlp
        generator = torch.Generator().manual_seed(37)
        hidden = torch.randn(4, 3, TINY_CONFIG["hidden_size"], generator=generator)
        expected = torch.tensor([[[0, 1]] * 3, [[2, 3]] * 3, [[4, 5]] * 3, [[6, 7]] * 3])
        indices, weights = layer.route(hidden)
        self.assertEqual(indices.reshape(4, 3, 2).tolist(), expected.tolist(),
                         "case: t = sequence_index * batch_size + batch_index")
        scores = torch.sigmoid(hidden.double() @ layer.gate.weight.detach().double().T)
        selected = scores.gather(-1, expected)
        reference = selected / selected.sum(-1, keepdim=True) * layer.gate.routed_scaling_factor
        torch.testing.assert_close(weights.reshape(4, 3, 2).double(), reference, rtol=1e-6, atol=1e-7)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_forced_routes_and_gradients_use_original_affinities(self):
        """
        Feature: Forced balanced expert routing.
        Description: Compare non-divisible cyclic assignments and gradients to a float64 formula.
        Expectation: Routing weights, auxiliary loss and gradients use the selected experts' affinities.
        """
        layer = _model(moe_router_force_expert_balance=True).model.layers[1].mlp
        generator = torch.Generator().manual_seed(19)
        with torch.no_grad():
            layer.gate.weight.copy_(torch.randn(layer.gate.weight.shape, generator=generator) * 0.1)
            layer.gate.e_score_correction_bias[-1] = 1000.0
        hidden = torch.randn(1, 3, TINY_CONFIG["hidden_size"], generator=generator, requires_grad=True)
        padded = torch.cat((hidden.new_zeros(1, layer.padding, hidden.shape[-1]), hidden), dim=1)
        expected_indices = torch.tensor([[0, 1], [2, 3], [0, 1]])
        coefficients = torch.tensor([[0.2, 0.7], [0.5, 0.3], [0.8, -0.4]])
        with aux_loss_scale_context():
            indices, weights = layer.route(padded)
            self.assertEqual(indices[layer.padding:].tolist(), expected_indices.tolist(),
                             f"forced indices={indices[layer.padding:].tolist()}, expected={expected_indices.tolist()}")
            task = (weights[layer.padding:] * coefficients).sum()
            bind_aux_loss_scale(task)
            actual_gradients = torch.autograd.grad(task, (hidden, layer.gate.weight))

        reference_hidden = hidden.detach().double().requires_grad_()
        reference_weight = layer.gate.weight.detach().double().requires_grad_()
        scores = torch.sigmoid(reference_hidden.reshape(3, -1) @ reference_weight.T)
        selected = scores.gather(1, expected_indices)
        reference_weights = selected / selected.sum(-1, keepdim=True) * layer.gate.routed_scaling_factor
        # Three tokens with top-2 cyclic routes have loads [2, 2, 1, 1] / 6.
        loads = scores.new_tensor([2, 2, 1, 1]) / 6
        affinity = (scores / scores.sum(-1, keepdim=True)).mean(0)
        auxiliary = (affinity * loads).sum() * layer.num_experts
        objective = (reference_weights * coefficients).sum() + auxiliary * layer.aux_loss_coeff
        expected_gradients = torch.autograd.grad(objective, (reference_hidden, reference_weight))
        torch.testing.assert_close(weights[layer.padding:].double(), reference_weights, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(layer.last_aux_loss.double(), auxiliary, rtol=1e-6, atol=1e-7)
        for actual, expected in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual.double(), expected, rtol=1e-5, atol=1e-7)


class TestXing40EmbeddingPrecision(unittest.TestCase):
    """Keep lookup/scatter precision independent of the unconditional BF16 output cast."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_fp32_lookup_with_low_precision_output_preserves_scatter(self):
        """
        Feature: Embedding lookup/scatter precision independent of the residual output cast.
        Description: Look up a repeated token through embed_inputs with an FP32 table, then backward
            with per-occurrence coefficients that only round in BF16.
        Expectation: The output is BF16 and equals the FP32 row cast once; the FP32 weight gradient
            sums the un-rounded coefficients at the repeated row and stays zero elsewhere.
        """
        model = _model()
        weight = model.model.embed_tokens.weight
        ids = torch.tensor([[3, 3]])
        output = model.model.embed_inputs(ids)
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(weight.dtype, torch.float32)
        torch.testing.assert_close(output, weight.detach()[ids].to(torch.bfloat16), rtol=0, atol=0)
        coefficients = torch.ones_like(output)
        coefficients[:, 1] = 1 / 256
        (output * coefficients).sum().backward()
        expected = torch.zeros_like(weight)
        expected[3] = 1 + 1 / 256
        torch.testing.assert_close(weight.grad, expected, rtol=0, atol=0)
        self.assertIs(model.model.embed_tokens.weight, weight)


if __name__ == "__main__":
    unittest.main()

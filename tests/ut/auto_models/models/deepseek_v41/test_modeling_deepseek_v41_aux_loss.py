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
"""V4.1 router and native-model auxiliary-loss integration tests."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import Mock, patch

import torch
import yaml
from torch.nn import functional
try:
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4SparseMoeBlock
except ImportError as exc:
    raise unittest.SkipTest(f"DeepSeek-V4 Transformers dependency unavailable: {exc}") from exc

from hyper_parallel.components.functional.aux_loss import aux_loss_scale_context, bind_aux_loss_scale
from hyper_parallel.components.modules.shared_compressed_dsa_attention import SharedCompressedPackedSequence
from hyper_parallel.distributed.apply import apply_sharding_plan
from hyper_parallel.distributed.plan import ShardingPlan
from hyper_parallel.models.deepseek_v41.adapter.distributed.moe_engram_expert_parallel import (
    deepseek_v41_ep_compute_fn,
    deepseek_v41_megamoe_compute_fn,
)
from hyper_parallel.models.deepseek_v41.adapter.distributed.megamoe import DeepseekV41MegaMoeExperts
from hyper_parallel.models.deepseek_v41.adapter.validation.cropped_model import build_deepseek_v41_validation_config
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import (
    DeepseekV41ForCausalLM,
    DeepseekV41TopKRouter,
    _v41_sparse_moe_forward,
)
from hyper_parallel.trainer.config import PlanOverride, entries_to_plan_overrides
from hyper_parallel.trainer.config.resolver import resolve_component
from tests.ut.auto_models.distributed.conftest import FakeDeviceMesh
from tests.common.deepseek_v41_mhc_stubs import stub_mhc_for_cpu
from tests.common.mark_utils import arg_mark


# Match the text/VLM recipe contract without requiring repository-only examples.
_ROUTER_OVERRIDE_YAML = """
plan_overrides:
  - match: "{match}.forward"
    when: ep
    region_dispatch: false
    replace:
      kind: method
      implementation:
        _target_: >-
          hyper_parallel.models.deepseek_v41.adapter.distributed.moe_engram_expert_parallel.deepseek_v41_ep_compute_fn
        use_grouped_gemm: true
"""


def tiny_config(coefficient: float = 0.1) -> DeepseekV4Config:
    """Build a small one-layer smoke configuration, not a structural acceptance crop."""
    config = DeepseekV4Config(  # pylint: disable=unexpected-keyword-arg
        vocab_size=32, hidden_size=16, moe_intermediate_size=24, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=1, head_dim=8, q_lora_rank=8,
        num_experts_per_tok=2, n_routed_experts=4, n_shared_experts=1,
        scoring_func="sqrtsoftplus", norm_topk_prob=True, routed_scaling_factor=1.25,
        layer_types=["sliding_attention"], mlp_layer_types=["moe"], swiglu_limit=1.5,
        sliding_window=4, o_groups=2, o_lora_rank=4, index_n_heads=2,
        index_head_dim=4, index_topk=2, partial_rotary_factor=0.5,
        router_aux_loss_coef=coefficient, use_cache=False,
    )
    config.v41_vision_enabled = False
    config.v41_compress_ratios = [0]
    config.v41_kv_source_layer_ids = []
    config.v41_index_source_layer_ids = []
    return config


class TestV41RouterAuxLoss(unittest.TestCase):
    """Verify the learned text/image router against an independent objective."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_router_gradient_matches_explicit_full_score_loss(self) -> None:
        """
        Feature: Auxiliary-loss training lifecycle.
        Description: Bias chooses experts; normalized unbiased scores carry aux gradients.
        Expectation: Forward outputs match, while router and input gradients equal the explicit objective.
        """
        for scoring in ("sqrtsoftplus", "sigmoid", "softmax"):
            with self.subTest(scoring=scoring):
                torch.manual_seed(31)
                config = tiny_config()
                config.scoring_func = scoring
                config.v41_vision_enabled = True
                actual = DeepseekV41TopKRouter(config)
                torch.nn.init.normal_(actual.weight)
                with torch.no_grad():
                    actual.bias.copy_(torch.tensor([2., 1., -2., -3.]))
                    actual.bias_vl.copy_(torch.tensor([-3., -2., 1., 2.]))
                expected = copy.deepcopy(actual)
                expected.aux_loss_coeff = 0
                hidden = torch.randn(2, 3, config.hidden_size, requires_grad=True)
                reference_input = hidden.detach().clone().requires_grad_()
                image_mask = torch.tensor([[True, False, False], [False, True, True]])
                valid = torch.tensor([[True, True, False], [True, True, True]])
                _, weights, indices = actual(hidden, image_mask, valid)
                logits, reference_weights, reference_indices = expected(reference_input, image_mask)
                torch.testing.assert_close(weights, reference_weights)
                torch.testing.assert_close(indices, reference_indices)
                if scoring == "sqrtsoftplus":
                    scores = functional.softplus(logits).sqrt()  # pylint: disable=not-callable
                elif scoring == "sigmoid":
                    scores = logits.sigmoid()
                else:
                    scores = logits.softmax(-1)
                scores = scores.reshape(2, 3, -1)
                unbiased_indices = scores.topk(config.num_experts_per_tok, dim=-1).indices
                sample_losses = []
                for sample in range(2):
                    selected = valid[sample]
                    route_map = functional.one_hot(  # pylint: disable=not-callable
                        unbiased_indices[sample, selected], config.num_local_experts,
                    ).sum(1).float()
                    frequency = route_map.mean(0) / config.num_experts_per_tok
                    probabilities = functional.normalize(scores[sample, selected], p=1, dim=-1).mean(0)
                    sample_losses.append(config.num_local_experts * torch.dot(frequency, probabilities))
                aux = torch.stack(sample_losses).mean()
                upstream = torch.randn_like(weights)
                (weights * upstream).sum().backward()
                ((reference_weights * upstream).sum() + config.router_aux_loss_coef * aux).backward()
                torch.testing.assert_close(actual.last_aux_loss, aux.detach())
                torch.testing.assert_close(actual.weight.grad, expected.weight.grad)
                torch.testing.assert_close(hidden.grad, reference_input.grad)
                self.assertIsNone(actual.bias.grad)
                self.assertIsNone(actual.bias_vl.grad)
                self.assertFalse(actual.last_aux_loss.requires_grad)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_aux_loss_uses_unbiased_topk_while_routing_uses_bias(self) -> None:
        """
        Feature: Sequence auxiliary loss and modality correction bias.
        Description: Fixed scores prefer expert zero while text/image biases change the routed experts.
        Expectation: Aux values and gradients stay unchanged; actual routing controls each modality's loads.
        """
        config = SimpleNamespace(
            hidden_size=1, num_local_experts=2, num_experts_per_tok=1,
            scoring_func="sqrtsoftplus", routed_scaling_factor=1.0, router_aux_loss_coef=1.0,
            router_bias_update_rate=0.001, v41_vision_enabled=True,
        )
        for text_bias, image_bias, routed in (
                ([0., 0.], [0., 0.], [0, 0]),
                ([-1., 1.], [0., 0.], [1, 0]),
                ([-1., 1.], [-1., 1.], [1, 1]),
        ):
            with self.subTest(routed=routed):
                router = DeepseekV41TopKRouter(config)
                with torch.no_grad():
                    router.weight.copy_(torch.log(torch.expm1(torch.tensor([[0.81], [0.01]]))))
                    router.bias.copy_(torch.tensor(text_bias))
                    router.bias_vl.copy_(torch.tensor(image_bias))
                hidden = torch.ones(1, 2, 1, requires_grad=True)
                logits, weights, indices = router(hidden, image_mask=torch.tensor([[False, True]]))
                torch.testing.assert_close(indices, torch.tensor(routed).unsqueeze(-1))
                scores = functional.softplus(logits).sqrt()  # pylint: disable=not-callable
                expected_loss = 2 * (scores[:, 0] / scores.sum(-1)).mean()
                expected_gradients = torch.autograd.grad(expected_loss, (router.weight, hidden), retain_graph=True)
                torch.testing.assert_close(router.last_aux_loss, torch.tensor(1.8))
                (weights.sum() * 0).backward()
                torch.testing.assert_close(router.weight.grad, expected_gradients[0])
                torch.testing.assert_close(hidden.grad, expected_gradients[1])
                expected_loads = functional.one_hot(torch.tensor(routed), 2)  # pylint: disable=not-callable
                torch.testing.assert_close(router.tokens_per_expert, expected_loads)
                self.assertIsNone(router.bias.grad)
                self.assertIsNone(router.bias_vl.grad)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_disable_eval_and_invalid_coefficient(self) -> None:
        """
        Feature: Auxiliary-loss training lifecycle.
        Description: Disabled/eval paths clear metrics and skip all auxiliary work.
        Expectation: Eval/zero coefficient disables aux; negative and non-finite coefficients are rejected.
        """
        router = DeepseekV41TopKRouter(tiny_config())
        torch.nn.init.normal_(router.weight)
        inputs = torch.randn(1, 3, 16)
        router(inputs)
        self.assertIsNotNone(router.last_aux_loss)
        router.eval()
        with patch("hyper_parallel.models.deepseek_v41.modeling_deepseek_v41.sequence_load_balancing_loss") as loss:
            router(inputs)
            self.assertIsNone(router.last_aux_loss)
            router.train()
            router.aux_loss_coeff = 0
            router(inputs)
            loss.assert_not_called()
        for coefficient in (-0.1, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "router_aux_loss_coef"):
                DeepseekV41TopKRouter(tiny_config(coefficient))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_compute_factory_reduces_only_token_partition_axes(self) -> None:
        """
        Feature: Auxiliary-loss training lifecycle.
        Description: The EP factory reduces sequence statistics over CP and TP token partitions.
        Expectation: CP and TP participate; calling the EP factory without an EP mesh fails.
        """
        cp_mesh = Mock()
        cp_mesh.size.return_value = 2
        tp_mesh = Mock()
        tp_mesh.size.return_value = 2
        inputs = torch.randn(1, 2, 16)
        router = Mock(return_value=(torch.empty(2, 3), torch.ones(2, 1), torch.zeros(2, 1, dtype=torch.long)))
        experts = Mock(return_value=torch.zeros(2, 16))
        experts.limit = 0
        module = SimpleNamespace(
            gate=router, experts=experts, shared_experts=torch.zeros_like, is_hash=False,
            forward=lambda hidden_states, input_ids=None, router_token_mask=None: hidden_states,
        )

        def dispatch(
                target: Any, states: torch.Tensor, *, router_fn: Callable, ep_group: Any,
        ) -> torch.Tensor:
            """Exercise the real gate call without expert communication."""
            del ep_group
            router_fn(target, states)
            return torch.zeros_like(states)

        prefix = "hyper_parallel.models.deepseek_v41.adapter.distributed.moe_engram_expert_parallel"
        expert_mesh = Mock()
        expert_mesh.__getitem__ = Mock(return_value=SimpleNamespace(size=lambda: 2))
        with self.assertRaisesRegex(ValueError, "active ep_mesh"):
            deepseek_v41_ep_compute_fn(module=module, mesh=None, tp_mesh=tp_mesh, cp_mesh=cp_mesh, ep_mesh=None)
        with patch(f"{prefix}.bind_local_expert_forward") as bind, patch(
                f"{prefix}.ep_routed_forward", side_effect=dispatch):
            compute = deepseek_v41_ep_compute_fn(
                module=module, mesh=None, tp_mesh=tp_mesh, cp_mesh=cp_mesh, ep_mesh=expert_mesh,
            )
            compute(module, inputs)
            groups = router.call_args.kwargs["sequence_partition_groups"]
            expected = (cp_mesh.get_group(), tp_mesh.get_group())
            self.assertEqual(groups, expected)
            self.assertEqual(router.expert_bias_update_groups, expected)
            bind.assert_called_once()


class TestV41NativeAuxLoss(unittest.TestCase):
    """Exercise the real causal-LM and EP adapter paths on CPU."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_cropped_config_preserves_aux_coefficient_and_upstream_structure(self) -> None:
        """
        Feature: Auxiliary-loss training lifecycle.
        Description: Aux settings survive the refactored builder without undoing its crop.
        Expectation: Both balancing knobs are set without changing the requested crop structure.
        """
        base = tiny_config()
        text = base.to_dict()
        text.update(
            num_hidden_layers=3, rope_scaling=base.rope_scaling, qk_rope_head_dim=4,
            engram_head_dim=4, compress_ratios=[0, 2, 2],
            kv_source_layer_ids=[1], index_source_layer_ids=[1],
        )
        source = {
            "model_type": "deepseek_v41", "text_config": text, "pad_token_id": 0,
            "bos_token_id": 1, "eos_token_id": 2, "image_token_id": 3,
            "vision_config": {
                "num_hidden_layers": 5, "hidden_size": 16, "num_attention_heads": 4,
                "intermediate_size": 24, "patch_size": 2, "rope_theta": 10000,
                "downsample_ratio": 2, "max_image_tokens": 16, "min_pixels": 4, "max_wh_ratio": 4,
            },
        }
        assets = {
            "source_model_type": "deepseek_v41", "num_hidden_layers": 3,
            "head_dim": 2, "layer_ids": [], "num_embeddings": [], "bucket_base": 16,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(source), encoding="utf-8")
            assets_path = root / "engram.json"
            assets_path.write_text(json.dumps(assets), encoding="utf-8")
            for options in ({}, {"router_aux_loss_coeff": 0.125},
                            {"router_aux_loss_coeff": 0.0, "router_bias_update_rate": 0.0}):
                with self.subTest(options=options):
                    config = build_deepseek_v41_validation_config(
                        str(root), str(assets_path), text_parameter_divisor=2,
                        vision_parameter_divisor=2, enable_vision=True, num_routed_experts=4, **options,
                    )
                    router = DeepseekV41TopKRouter(config)
                    self.assertEqual(router.aux_loss_coeff, options.get("router_aux_loss_coeff", 1.0e-4))
                    self.assertEqual(router.expert_bias_update_rate, options.get("router_bias_update_rate", 0.001))
                    self.assertEqual(router.enable_expert_bias, router.expert_bias_update_rate > 0)
                    self.assertEqual(config.hidden_size, 8)
                    self.assertEqual(config.o_groups, 2)
                    self.assertEqual(config.v41_compress_ratios, [0, 2, 2])
                    self.assertEqual(config.num_hidden_layers, 3)
                    self.assertEqual(config.v41_vision_num_hidden_layers, 5)
                    self.assertEqual(config.v41_vision_hidden_size, 8)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_native_forward_backward_preserves_logits_and_checkpoint_keys(self) -> None:
        """
        Feature: Auxiliary-loss training lifecycle.
        Description: Verify router aux training with shape-preserving mHC stubs on CPU.
        Expectation: Logits and checkpoint keys match while router gradients include the auxiliary objective.
        """
        torch.manual_seed(17)
        with tempfile.TemporaryDirectory() as temporary_dir:
            config = tiny_config()
            assets = Path(temporary_dir) / "engram.json"
            assets.write_text(json.dumps({"layer_ids": []}), encoding="utf-8")
            config.v41_engram_assets_path = str(assets)
            model = DeepseekV41ForCausalLM(config)
            reference = copy.deepcopy(model)
            reference.model.layers[0].mlp.gate.aux_loss_coeff = 0
            tokens = torch.tensor([[1, 2, 3, 4], [4, 2, 5, 6]])
            with stub_mhc_for_cpu():
                with aux_loss_scale_context():
                    output = model(tokens, labels=tokens)
                    bind_aux_loss_scale(output.loss)
                    (output.loss / 2).backward()
                reference_output = reference(tokens, labels=tokens)
                (reference_output.loss / 2).backward()
            torch.testing.assert_close(output.logits, reference_output.logits)
            torch.testing.assert_close(output.loss, reference_output.loss)
            gate = model.model.layers[0].mlp.gate
            self.assertIsNotNone(gate.last_aux_loss)
            self.assertFalse(torch.allclose(gate.weight.grad, reference.model.layers[0].mlp.gate.weight.grad))
            self.assertEqual(set(model.state_dict()), set(reference.state_dict()))
            self.assertTrue(torch.isfinite(gate.weight.grad).all())

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_ep_adapter_retains_router_auxiliary_gradient(self) -> None:
        """
        Feature: Auxiliary-loss training lifecycle.
        Description: Mock only expert communication while executing the actual EP router callable.
        Expectation: The production EP router carrier delivers a finite nonzero auxiliary gradient.
        """
        torch.manual_seed(12)
        router = DeepseekV41TopKRouter(tiny_config())
        torch.nn.init.normal_(router.weight)
        module = SimpleNamespace(
            gate=router, experts=SimpleNamespace(_apply_gate=None, limit=1.5),
            shared_experts=lambda inputs: inputs * 0, is_hash=False,
            forward=lambda hidden_states, input_ids=None, image_mask=None, router_token_mask=None: hidden_states,
        )
        ep_mesh = Mock()
        ep_mesh.__getitem__ = Mock(return_value=SimpleNamespace(size=lambda: 2))

        def _dispatch(target, inputs, *, router_fn, ep_group):
            """Use returned weights exactly as the real dispatch/combine path does."""
            del ep_group
            _, weights = router_fn(target, inputs)
            return weights.sum(-1).reshape(*inputs.shape[:-1], 1).expand_as(inputs)

        prefix = "hyper_parallel.models.deepseek_v41.adapter.distributed.moe_engram_expert_parallel"
        with patch(f"{prefix}.bind_local_expert_forward"), patch(f"{prefix}.ep_routed_forward", side_effect=_dispatch):
            compute = deepseek_v41_ep_compute_fn(module=module, mesh=None, tp_mesh=None, cp_mesh=None, ep_mesh=ep_mesh)
            output = compute(module, torch.randn(1, 4, 16), router_token_mask=torch.tensor([[True, True, False, True]]))
            output.sum().backward()
        self.assertIsNotNone(router.last_aux_loss)
        self.assertGreater(router.weight.grad.abs().sum().item(), 0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_packed_samples_reach_router_as_distinct_sequences(self) -> None:
        """
        Feature: Auxiliary-loss training lifecycle.
        Description: Packing two unequal samples into B=1 must preserve their boundaries.
        Expectation: Packed sample IDs and total count reach the real gate unchanged.
        """
        with tempfile.TemporaryDirectory() as temporary_dir:
            config = tiny_config()
            assets = Path(temporary_dir) / "engram.json"
            assets.write_text(json.dumps({"layer_ids": []}), encoding="utf-8")
            config.v41_engram_assets_path = str(assets)
            model = DeepseekV41ForCausalLM(config)
            tokens = torch.tensor([[1, 2, 3, 4]])
            packed = SharedCompressedPackedSequence(
                cu_seq_lens=torch.tensor([0, 1, 4], dtype=torch.int32),
                local_query_start=0, local_query_length=4, global_sequence_length=4,
            )
            gate = model.model.layers[0].mlp.gate
            with (stub_mhc_for_cpu(),
                  patch.object(gate, "forward", wraps=gate.forward) as forward):
                model(tokens, labels=tokens, packed_seq_params=packed).loss.backward()
            torch.testing.assert_close(forward.call_args.kwargs["sequence_ids"], torch.tensor([[0, 1, 1, 1]]))
            self.assertEqual(forward.call_args.kwargs["num_sequences"], 2)
            self.assertEqual(gate.tokens_per_expert.sum().item(), 4 * config.num_experts_per_tok)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_recipe_compute_installs_in_production_and_validate(self) -> None:
        """
        Feature: Auxiliary-loss training lifecycle.
        Description: Text/VLM YAML activates the compute factory only for EP and preserves gate gradients.
        Expectation: EP-off installs nothing; both EP recipe modes preserve outputs and router gradients.
        """
        for match in ("*.mlp", "model.layers.*.mlp"):
            raw = yaml.safe_load(_ROUTER_OVERRIDE_YAML.format(match=match))
            entries = [
                resolve_component(entry, annotation=PlanOverride, path="plan_overrides")
                for entry in raw["plan_overrides"]
                if isinstance(entry["match"], str) and entry["match"].endswith(".mlp.forward")
            ]
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries_to_plan_overrides(entries, ep_size=1), {})

            def dispatch(
                    target: Any, states: torch.Tensor, *, router_fn: Callable, ep_group: Any,
            ) -> torch.Tensor:
                """Use native experts to validate the EP region without communication."""
                del ep_group
                indices, weights = router_fn(target, states)
                return target.experts(states.reshape(-1, states.shape[-1]), indices, weights).reshape_as(states)

            for validate in (False, True):
                with self.subTest(match=match, validate=validate):
                    module = torch.nn.Module()
                    module.mlp = _RouterCarrier()
                    reference = copy.deepcopy(module.mlp)
                    spec = next(iter(entries_to_plan_overrides(entries, ep_size=2).values()))
                    # The fixture has no active mesh axes; production plans
                    # inherit parameter placements from the MLP template.
                    spec.params = {}
                    plan = ShardingPlan(modules={"mlp": spec}, mesh_dim_names=())
                    expert_mesh = Mock()
                    expert_mesh.__getitem__ = Mock(return_value=SimpleNamespace(size=lambda: 2))
                    prefix = "hyper_parallel.models.deepseek_v41.adapter.distributed.moe_engram_expert_parallel"
                    with (
                        patch("hyper_parallel.distributed.apply._resolve_parameter_source_meshes",
                              return_value=(expert_mesh, None, None)),
                        patch(f"{prefix}.bind_local_expert_forward"),
                        patch(f"{prefix}.ep_routed_forward", side_effect=dispatch),
                    ):
                        apply_sharding_plan(module, plan, FakeDeviceMesh(), validate_mode=validate)
                        inputs = torch.randn(1, 4, 16)
                        actual = module.mlp(inputs)
                        expected = reference(inputs)
                        actual.sum().backward()
                        expected.sum().backward()
                        torch.testing.assert_close(actual, expected)
                        torch.testing.assert_close(module.mlp.gate.weight.grad, reference.gate.weight.grad)
            ep_spec = next(iter(entries_to_plan_overrides(entries, ep_size=2).values()))
            self.assertIsNotNone(ep_spec.local_compute_fn)
            self.assertIsNone(ep_spec.inner_wrapper)


class TestV41MegaMoeAuxLoss(unittest.TestCase):
    """Run real router training state around a CPU replacement for native experts."""

    def test_megamoe_preserves_router_loss_gradients_and_loads(self) -> None:
        """Keep packed IDs, masks and independent aux switches through MegaMoe.

        Feature: MegaMoe auxiliary-loss integration.
        Description: Compare its real router with a direct router reference on CPU.
        Expectation: Outputs, gradients, auxiliary loss and modality counts match.
        """
        torch.manual_seed(47)
        for vision_enabled in (False, True):
            for coefficient, rate in ((0., 0.), (0.1, 0.), (0., 0.001), (0.1, 0.001)):
                for packed in (False, True):
                    with self.subTest(vision=vision_enabled, aux=coefficient, bias=rate, packed=packed):
                        config = tiny_config(coefficient)
                        config.v41_vision_enabled = vision_enabled
                        config.router_bias_update_rate = rate
                        router = DeepseekV41TopKRouter(config)
                        torch.nn.init.normal_(router.weight)
                        reference = copy.deepcopy(router)

                        def execute(
                                states: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor,
                        ) -> torch.Tensor:
                            """Emulate only the native expert gradient carrier on CPU."""
                            del indices
                            return weights.sum(-1).reshape(*states.shape[:-1], 1).expand_as(states)

                        experts = Mock(spec=DeepseekV41MegaMoeExperts, swiglu_limit=1.5, side_effect=execute)
                        module = SimpleNamespace(
                            gate=router, experts=experts, is_hash=False, forward=_v41_sparse_moe_forward,
                            shared_experts=Mock(limit=1.5, side_effect=torch.zeros_like),
                        )
                        compute = deepseek_v41_megamoe_compute_fn(
                            module=module, mesh=None, tp_mesh=None, cp_mesh=None, ep_mesh=None,
                        )
                        states = torch.randn(1, 4, 16, requires_grad=True)
                        reference_states = states.detach().clone().requires_grad_()
                        images = torch.tensor([[False, True, False, True]]) if vision_enabled else None
                        valid = torch.tensor([[True, False, True, True]])
                        ids = torch.tensor([[0, 1, 1, 1]]) if packed else None
                        count = 2 if packed else None
                        actual = compute(
                            module, states, image_mask=images, router_token_mask=valid,
                            router_sequence_ids=ids, router_num_sequences=count,
                        )
                        _, weights, indices = reference(
                            reference_states, image_mask=images, token_mask=valid,
                            sequence_ids=ids, num_sequences=count,
                        )
                        expected = execute(reference_states, indices, weights)
                        upstream = torch.randn_like(actual)
                        (actual * upstream).sum().backward()
                        (expected * upstream).sum().backward()
                        torch.testing.assert_close(actual, expected)
                        torch.testing.assert_close(states.grad, reference_states.grad)
                        torch.testing.assert_close(router.weight.grad, reference.weight.grad)
                        torch.testing.assert_close(router.tokens_per_expert, reference.tokens_per_expert)
                        if coefficient:
                            torch.testing.assert_close(router.last_aux_loss, reference.last_aux_loss)
                        else:
                            self.assertIsNone(router.last_aux_loss)
                        self.assertEqual(router.tokens_per_expert.sum().item(), 6 if rate else 0)
                        experts.configure.assert_called_once_with(None, 1, 2)

    def test_megamoe_aligns_tp_metadata_and_statistics_groups(self) -> None:
        """Align all metadata and preserve separate sequence and bias domains.

        Feature: MegaMoe auxiliary-loss integration.
        Description: Feed CP-local metadata into the second TP shard of a two-row batch.
        Expectation: Router receives sliced masks/IDs, contiguous image mask and CP/TP groups.
        """
        for fixed in (False, True):
            with self.subTest(fixed=fixed):
                self._check_megamoe_metadata(fixed)

    def _check_megamoe_metadata(self, fixed: bool) -> None:
        """Keep CP/TP/DP metadata independent of the synthetic dispatch destinations."""
        tp, cp, dp = Mock(), Mock(), Mock()
        for axis in (tp, cp, dp):
            axis.size.return_value = 2
        tp.get_local_rank.return_value = 1
        mesh = Mock(mesh_dim_names=("dp", "cp", "tp"))
        mesh.__getitem__ = Mock(side_effect={"dp": dp, "cp": cp, "tp": tp}.__getitem__)
        ep = Mock()
        ep.__getitem__ = Mock(return_value=SimpleNamespace(size=lambda: 2))
        router = Mock(num_experts=4, top_k=2,
                      return_value=(torch.empty(4, 4), torch.ones(4, 2), torch.zeros(4, 2).long()))
        experts = Mock(spec=DeepseekV41MegaMoeExperts, swiglu_limit=1.5, sequence_parallel=True)
        experts.side_effect = lambda states, unused_indices, unused_weights: torch.zeros_like(states)
        module = SimpleNamespace(
            gate=router, experts=experts, is_hash=False, forward=_v41_sparse_moe_forward,
            shared_experts=Mock(limit=1.5, side_effect=torch.zeros_like),
        )
        states = torch.randn(2, 2, 16)
        images = torch.tensor([[False, True, False, True], [True, False, True, False]])
        valid = torch.tensor([[True, True, False, True], [True, False, True, True]])
        ids = torch.tensor([[0, 0, 1, 1], [2, 2, 3, 3]])
        compute = deepseek_v41_megamoe_compute_fn(
            module=module, mesh=mesh, tp_mesh=tp, cp_mesh=cp, ep_mesh=ep, fix_router=fixed,
        )
        compute(module, states, image_mask=images, router_token_mask=valid,
                router_sequence_ids=ids, router_num_sequences=4)
        actual = router.call_args.kwargs
        torch.testing.assert_close(actual["image_mask"], images[:, 2:])
        torch.testing.assert_close(actual["token_mask"], valid[:, 2:])
        torch.testing.assert_close(actual["sequence_ids"], ids[:, 2:])
        self.assertTrue(actual["image_mask"].is_contiguous())
        self.assertEqual(actual["num_sequences"], 4)
        self.assertEqual(actual["sequence_partition_groups"], (cp.get_group(), tp.get_group()))
        self.assertEqual(router.expert_bias_update_groups, (dp.get_group(), cp.get_group(), tp.get_group()))
        expected_indices = torch.tensor([[0, 2], [1, 3], [0, 2], [1, 3]]) if fixed else torch.zeros(4, 2).long()
        torch.testing.assert_close(experts.call_args.args[1], expected_indices)
        self.assertIs(experts.call_args.args[2], router.return_value[1])


class _RouterCarrier(DeepseekV4SparseMoeBlock):
    """Minimal MLP boundary exercising the production router callable."""

    def __init__(self) -> None:
        """Construct a router with initialized parameters."""
        config = tiny_config()
        super().__init__(config, 0)
        self.gate = DeepseekV41TopKRouter(config)
        for parameter in self.parameters():
            if parameter.requires_grad:
                torch.nn.init.normal_(parameter)

    forward = _v41_sparse_moe_forward

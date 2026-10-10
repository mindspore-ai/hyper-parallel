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
# pylint: disable=wrong-import-position

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import yaml
from torch.nn import functional
try:
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
except ImportError as exc:
    raise unittest.SkipTest(f"DeepSeek-V4 Transformers dependency unavailable: {exc}") from exc

from hyper_parallel.components.functional.aux_loss import aux_loss_scale_context, bind_aux_loss_scale
from hyper_parallel.components.modules.shared_compressed_dsa_attention import SharedCompressedPackedSequence
from hyper_parallel.distributed.apply import apply_sharding_plan
from hyper_parallel.distributed.plan import ShardingPlan
from hyper_parallel.models.deepseek_v41.adapter.distributed.moe_engram_expert_parallel import (
    deepseek_v41_ep_compute_fn,
    deepseek_v41_router_aux_loss_wrapper,
)
from hyper_parallel.models.deepseek_v41.adapter.validation.cropped_model import build_deepseek_v41_validation_config
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41ForCausalLM, DeepseekV41TopKRouter
from hyper_parallel.trainer.config import PlanOverride, entries_to_plan_overrides
from hyper_parallel.trainer.config.resolver import resolve_component
from tests.ut.auto_models.distributed.conftest import FakeDeviceMesh
from tests.common.mark_utils import arg_mark


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
                sample_losses = []
                for sample in range(2):
                    selected = valid[sample]
                    route_map = functional.one_hot(  # pylint: disable=not-callable
                        indices.reshape(2, 3, -1)[sample, selected], config.num_local_experts,
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
    def test_parallel_wrapper_reduces_only_token_partition_axes(self) -> None:
        """
        Feature: Auxiliary-loss training lifecycle.
        Description: CP is always partitioned; TP is partitioned only inside the EP region.
        Expectation: CP always participates and TP participates only when EP is enabled.
        """
        cp_mesh = Mock()
        cp_mesh.size.return_value = 2
        tp_mesh = Mock()
        tp_mesh.size.return_value = 2
        router = Mock()
        module = SimpleNamespace(gate=router, forward=Mock())
        for ep_mesh in (None, Mock()):
            requests = deepseek_v41_router_aux_loss_wrapper(module, None, tp_mesh, cp_mesh, ep_mesh)
            requests[1].forward(torch.randn(1, 2, 16))
            groups = router.forward.call_args.kwargs["sequence_partition_groups"]
            expected = (cp_mesh.get_group(),) + (() if ep_mesh is None else (tp_mesh.get_group(),))
            self.assertEqual(groups, expected)


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
            for options in ({}, {"router_aux_loss_coeff": 0.125}):
                with self.subTest(options=options):
                    config = build_deepseek_v41_validation_config(
                        str(root), str(assets_path), text_parameter_divisor=2,
                        vision_parameter_divisor=2, enable_vision=True, num_routed_experts=4, **options,
                    )
                    router = DeepseekV41TopKRouter(config)
                    self.assertEqual(router.aux_loss_coeff, options.get("router_aux_loss_coeff", 1.0e-4))
                    self.assertEqual(router.expert_bias_update_rate, 0.001)
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
        Description: The production model runs aux training without module replacement.
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
            gate=router, experts=SimpleNamespace(_apply_gate=None),
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
            with patch.object(gate, "forward", wraps=gate.forward) as forward:
                model(tokens, labels=tokens, packed_seq_params=packed).loss.backward()
            torch.testing.assert_close(forward.call_args.kwargs["sequence_ids"], torch.tensor([[0, 1, 1, 1]]))
            self.assertEqual(forward.call_args.kwargs["num_sequences"], 2)
            self.assertEqual(gate.tokens_per_expert.sum().item(), 4 * config.num_experts_per_tok)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_recipe_wrapper_installs_in_production_and_validate(self) -> None:
        """
        Feature: Auxiliary-loss training lifecycle.
        Description: The actual YAML target preserves the MLP boundary and gate gradient.
        Expectation: Both recipe modes preserve outputs and router gradients through the installed wrapper.
        """
        recipe_dir = Path(__file__).resolve().parents[5] / "examples/training_demo/deepseek_v41"
        for recipe_name in ("train_deepseek_v41_online.yaml", "train_deepseek_v41_vlm_online.yaml"):
            raw = yaml.safe_load((recipe_dir / recipe_name).read_text(encoding="utf-8"))
            entries = [
                resolve_component(entry, annotation=PlanOverride, path="plan_overrides")
                for entry in raw["plan_overrides"]
                if isinstance(entry["match"], str) and entry["match"].endswith(".mlp")
            ]
            for validate in (False, True):
                with self.subTest(recipe=recipe_name, validate=validate):
                    module = torch.nn.Module()
                    module.mlp = _RouterCarrier()
                    reference = copy.deepcopy(module.mlp)
                    spec = next(iter(entries_to_plan_overrides(entries).values()))
                    # The fixture has no active mesh axes; production plans
                    # inherit parameter placements from the MLP template.
                    spec.params = {}
                    plan = ShardingPlan(modules={"mlp": spec}, mesh_dim_names=())
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
            self.assertIsNotNone(ep_spec.inner_wrapper)


class _RouterCarrier(torch.nn.Module):
    """Minimal MLP boundary exercising the production router callable."""

    def __init__(self) -> None:
        """Construct a router with initialized parameters."""
        super().__init__()
        self.gate = DeepseekV41TopKRouter(tiny_config())
        torch.nn.init.normal_(self.gate.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Expose routing weights as the carrier of auxiliary gradients."""
        return self.gate(hidden_states)[1]

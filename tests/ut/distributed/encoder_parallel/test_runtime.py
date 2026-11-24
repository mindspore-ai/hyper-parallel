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
"""One-rank visual runtime compared with the unchanged DeepSeek encoder."""

from types import SimpleNamespace
import unittest

import torch
import yaml
from torch import nn

from hyper_parallel.core.dtensor.device_mesh import _DEVICE_MESH_MAP, init_device_mesh
from hyper_parallel.distributed.encoder_parallel.batch_input import EncoderParallelInputAdapter
from hyper_parallel.distributed.encoder_parallel.config import EncoderParallelConfig
from hyper_parallel.distributed.encoder_parallel.runtime import EncoderParallelRuntime
from hyper_parallel.models.deepseek_v41.adapter.data.runtime import DeepseekV41Runtime
from hyper_parallel.models.deepseek_v41.adapter.distributed.encoder_parallel import (
    get_deepseek_v41_encoder_parallel_spec,
)
from hyper_parallel.models.deepseek_v41.vision import (
    DeepseekV41VisionAligner,
    DeepseekV41VisionTower,
)
from hyper_parallel.models.replacement import apply_module_replacements, compile_module_replacements
from hyper_parallel.trainer.config import entries_to_module_replacements
from hyper_parallel.trainer.config.parallelism import PlanOverride
from hyper_parallel.trainer.config.resolver import resolve_component

_VISUAL_RECIPE_RULE = """
match: "model.vision.blocks.*.attn"
replace:
  kind: module
  source_type: hyper_parallel.models.deepseek_v41.vision.DeepseekV41VisionAttention
  implementation:
    _target_: >-
      hyper_parallel.models.deepseek_v41.adapter.distributed.encoder_parallel.replace_vision_attention
"""


def _make_model(*, apply_recipe: bool = True, vision_depth: int = 2):
    """Build a minimal DeepSeek-V4.1 facade with a real vision tower."""
    config = SimpleNamespace(v41_vision_hidden_size=8, v41_vision_patch_size=2,
                             v41_vision_num_attention_heads=2, v41_vision_rope_theta=10000.0,
                             v41_vision_intermediate_size=16, v41_vision_num_hidden_layers=vision_depth,
                             v41_vision_downsample_ratio=1, hidden_size=8)
    facade = nn.Module()
    facade.config = SimpleNamespace(model_type="deepseek_v41", architectures=["DeepseekV41ForCausalLM"])
    facade.model = nn.Module()
    facade.model.vision = DeepseekV41VisionTower(config)
    facade.model.aligner = DeepseekV41VisionAligner(config)
    facade.model.image_start = nn.Parameter(torch.zeros(8))
    facade.model.image_end = nn.Parameter(torch.zeros(8))
    facade.model.image_newline = nn.Parameter(torch.zeros(8))
    if not apply_recipe:
        return facade
    entry = resolve_component(yaml.safe_load(_VISUAL_RECIPE_RULE), annotation=PlanOverride,
                              path="$.plan_overrides[0]")
    rules = entries_to_module_replacements([entry])
    visual_rules = tuple(rule for rule in rules if rule.match == ("model.vision.blocks.*.attn",))
    if len(visual_rules) != 1:
        raise AssertionError("The VLM recipe must declare exactly one visual attention replacement")
    plan = compile_module_replacements(facade, visual_rules)
    apply_module_replacements(facade, plan, weights_mapping=[])
    return facade


class TestEncoderRuntime(unittest.TestCase):
    """Check a complete local encode/route path and its inactive mode."""

    def setUp(self) -> None:
        """Build an unsharded local runtime with replaced attention blocks."""
        self.model = _make_model()
        mesh = init_device_mesh("cpu", (1, 1, 1), mesh_dim_names=("dp", "cp", "tp"),
                                rank_list=(0,), init_backend=False)
        context = SimpleNamespace(device_mesh=mesh, cp_size=1, tp_size=1, cp_rank=0, tp_rank=0,
                                  pp_size=1, sequence_parallel=False, loss_parallel=False)
        self.context = context
        self.runtime = EncoderParallelRuntime(
            config=EncoderParallelConfig(enabled=True), mesh_context=context,
            model=self.model, spec=get_deepseek_v41_encoder_parallel_spec(),
            execution_options={"micro_batch_size": 1, "compile_enabled": False,
                               "validate_placement": False},
        )

    def tearDown(self) -> None:
        """Evict backend-less cached sub-meshes so later suites rebuild real groups."""
        _DEVICE_MESH_MAP.clear()

    def test_recipe_wrapper_keeps_native_forward_and_backward(self):
        """Feature: recipe-selected ViT wrapper preserves the native path.

        Description: Compare the real recipe replacement with no recipe action.
        Expectation: Inactive-context outputs and patch gradients match exactly.
        """
        native = _make_model(apply_recipe=False)
        native.load_state_dict(self.model.state_dict())
        pixels = torch.randn(4, 3, 2, 2, requires_grad=True)
        native_output = native.model.vision(pixels, 2, 2)
        native_output.sum().backward()
        candidate_pixels = pixels.detach().clone().requires_grad_()
        candidate_output = self.model.model.vision(candidate_pixels, 2, 2)
        candidate_output.sum().backward()
        self.assertTrue(torch.equal(native_output, candidate_output))
        self.assertTrue(torch.equal(pixels.grad, candidate_pixels.grad))

    def test_recipe_matches_all_vision_attention_blocks(self):
        """Feature: one recipe glob covers the released vision block depth.

        Description: Apply the actual VLM recipe to a 32-block vision tower.
        Expectation: Every attention block is replaced exactly once.
        """
        model = _make_model(vision_depth=32)
        matched = set(model._hp_replaced_module_fqns)  # pylint: disable=protected-access
        expected = {f"model.vision.blocks.{index}.attn" for index in range(32)}
        self.assertEqual(matched, expected)

    def test_local_image_features_match_native_vision(self):
        """Feature: one-rank active visual encoding.

        Description: Encode one image through the runtime and native vision tower.
        Expectation: Feature values and patch gradients match.
        """
        pixels = torch.randn(4, 3, 2, 2, requires_grad=True)
        batch = {
            "input_ids": torch.zeros(1, 8, dtype=torch.long),
            "token_types": torch.zeros(1, 8, dtype=torch.long),
            "pixel_values": pixels,
            "image_patch_offsets": torch.tensor([0, 4]),
            "image_vit_grid_hw": torch.tensor([[2, 2]]),
            "image_llm_grid_hw": torch.tensor([[2, 2]]),
            "image_batch_indices": torch.tensor([0]),
            "image_token_starts": torch.tensor([0]),
        }
        plan = self.runtime.finalize_plan(*self.runtime.preflight_batch(batch))
        self.assertTrue(plan.active)
        records, anchor = plan.encode_and_route(batch)
        self.assertEqual(len(records), 1)
        actual = records[0][1]
        actual.sum().backward()
        active_grad = pixels.grad.clone()
        self.model.zero_grad(set_to_none=True)
        pixels.grad = None
        expected = self.model.model.aligner(self.model.model.vision(pixels, 2, 2), 2, 2)
        expected.sum().backward()
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        self.assertTrue(torch.allclose(active_grad, pixels.grad, atol=1e-6))
        self.assertTrue(torch.isfinite(anchor))

    def test_text_only_keeps_visual_path_idle(self):
        """Feature: text-only visual runtime mode.

        Description: Plan a batch with no image metadata.
        Expectation: The visual path returns no records or gradient anchor.
        """
        batch = {"input_ids": torch.zeros(1, 8, dtype=torch.long)}
        plan = self.runtime.finalize_plan(*self.runtime.preflight_batch(batch))
        self.assertFalse(plan.active)
        self.assertEqual(plan.encode_and_route(batch), ([], None))

    def test_composed_input_matches_previous_batch_sequence(self):
        """The real DeepSeek bridge produces the same plan and packed inputs.

        Feature: Batch ownership refactor.
        Description: Compare the former preflight/input/finalize sequence with
            the composed runtime-input adapter on one image microbatch.
        Expectation: Plan fields and all model runtime inputs agree exactly.
        """
        batch = {
            "input_ids": torch.zeros(1, 8, dtype=torch.long),
            "cu_seq_lens": torch.tensor([0, 8], dtype=torch.int32),
            "token_types": torch.zeros(1, 8, dtype=torch.long),
            "pixel_values": torch.randn(4, 3, 2, 2),
            "image_patch_offsets": torch.tensor([0, 4]),
            "image_vit_grid_hw": torch.tensor([[2, 2]]),
            "image_llm_grid_hw": torch.tensor([[2, 2]]),
            "image_batch_indices": torch.tensor([0]),
            "image_token_starts": torch.tensor([0]),
        }
        parallel_context = SimpleNamespace(tp_rank=0, cp_rank=0, tp_world_size=1, cp_world_size=1)
        options = {"sequence_parallel": False}
        inner = DeepseekV41Runtime()
        prepared = self.runtime.preflight_batch(batch)
        expected = inner.build(batch=batch, parallel_context=parallel_context, options=options)
        expected_plan = self.runtime.finalize_plan(*prepared)

        composed = EncoderParallelInputAdapter(runtime=self.runtime, inner=inner)
        actual = composed.build(batch=batch, parallel_context=parallel_context, options=options)
        self.assertEqual(actual["encoder_parallel_plan"], expected_plan)
        self.assertEqual(actual["image_sequence_start"], expected["image_sequence_start"])
        self.assertTrue(torch.equal(actual["position_ids"], expected["position_ids"]))
        for field in ("cu_seq_lens", "local_query_start", "local_query_length", "global_sequence_length"):
            old_value = getattr(expected["packed_seq_params"], field)
            new_value = getattr(actual["packed_seq_params"], field)
            if isinstance(old_value, torch.Tensor):
                self.assertTrue(torch.equal(new_value, old_value))
            else:
                self.assertEqual(new_value, old_value)

    def test_static_guards_fail_before_group_derivation(self):
        """Feature: static Encoder Parallel fail-fast checks.

        Description: Configure invalid microbatch, loss layout, and QKV geometry.
        Expectation: Each case raises its specific diagnostic before mesh setup.
        """
        spec = get_deepseek_v41_encoder_parallel_spec()
        options = {"micro_batch_size": 1, "compile_enabled": False, "validate_placement": False}
        invalid_options = dict(options, micro_batch_size=2)
        with self.assertRaisesRegex(ValueError, "does not support the requested micro batch"):
            EncoderParallelRuntime(config=EncoderParallelConfig(enabled=True),
                                   mesh_context=self.context, model=self.model, spec=spec,
                                   execution_options=invalid_options)
        invalid_context = SimpleNamespace(**{**vars(self.context), "loss_parallel": True})
        with self.assertRaisesRegex(NotImplementedError, "unsupported encoder loss layout"):
            EncoderParallelRuntime(config=EncoderParallelConfig(enabled=True),
                                   mesh_context=invalid_context, model=self.model, spec=spec,
                                   execution_options=options)
        wide_context = SimpleNamespace(**{**vars(self.context), "cp_size": 2, "tp_size": 4})
        with self.assertRaisesRegex(ValueError, "head divisibility check failed"):
            EncoderParallelRuntime(config=EncoderParallelConfig(enabled=True, encoder_cp_size=4),
                                   mesh_context=wide_context, model=self.model, spec=spec,
                                   execution_options=options)

    def test_missing_attention_replacement_is_rejected(self):
        """Feature: structural visual attention replacement audit.

        Description: Clear the replacement record after installing ViT blocks.
        Expectation: Runtime initialization identifies the missing block FQN.
        """
        self.model._hp_replaced_module_fqns = ()  # pylint: disable=protected-access
        with self.assertRaisesRegex(RuntimeError, "incomplete vision attention replacement.*blocks.0.attn"):
            EncoderParallelRuntime(
                config=EncoderParallelConfig(enabled=True), mesh_context=self.context,
                model=self.model, spec=get_deepseek_v41_encoder_parallel_spec(),
                execution_options={"micro_batch_size": 1, "compile_enabled": False,
                                   "validate_placement": False},
            )

    def test_partial_image_bundle_fails_preflight(self):
        """Feature: native image bundle validation.

        Description: Preflight a batch with pixels but missing image metadata.
        Expectation: Validation fails before the active-mode collective.
        """
        batch = {"input_ids": torch.zeros(1, 8, dtype=torch.long),
                 "pixel_values": torch.zeros(4, 3, 2, 2)}
        with self.assertRaisesRegex(ValueError, "incomplete image metadata bundle"):
            self.runtime.preflight_batch(batch)

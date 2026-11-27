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
"""Numerical and ownership checks for model-owned pipeline partitions."""

import copy
import tempfile
from types import SimpleNamespace
import unittest

import torch

from hyper_parallel.models.pipeline import PipelineConfig, partition_model
from hyper_parallel.models._transformers.model_builder import _prune_pipeline_parameter_contracts, apply_model_infrastructure
from hyper_parallel.models.deepseek_v41.adapter.distributed.mega_mhc import DeepseekV41HyperMegaMhc
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41ForCausalLM
from hyper_parallel.distributed.mesh import MeshContext
from tests.common.deepseek_v41_pipeline import make_config
from tests.common.mark_utils import arg_mark


class TestPipelinePartition(unittest.TestCase):
    """Compare a segmented model's outputs and trainable gradients with its source."""

    def _check_four_stages(self, shifted: bool, vision: bool = False, dspark: bool = False) -> None:
        """Compare the complete parameter and gradient sets against the source model."""
        torch.manual_seed(71)
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(directory)
            if vision:
                config.v41_vision_enabled = True
                config.v41_vision_hidden_size = 16
                config.v41_vision_intermediate_size = 24
                config.v41_vision_patch_size = 2
                config.v41_vision_num_attention_heads = 2
                config.v41_vision_rope_theta = 10000.0
                config.v41_vision_num_hidden_layers = 2
                config.v41_vision_downsample_ratio = 1
            if dspark:
                config.v41_dspark_depth = 1
                config.v41_dspark_confidence_coeff = 0.1
                config.v41_dspark_target_layer_ids = [37, 38, 39]
                config.v41_dspark_block_size = 2
                config.v41_dspark_window = 4
                config.v41_dspark_markov_rank = 8
                config.v41_dspark_noise_token_id = 63
                config.v41_dspark_n_routed_experts = 8
                config.v41_dspark_top_k = 3
            reference = DeepseekV41ForCausalLM(config)
            if shifted:
                for layer in reference.model.layers:
                    layer.attn_hc = DeepseekV41HyperMegaMhc(module=layer.attn_hc)
                    layer.ffn_hc = DeepseekV41HyperMegaMhc(module=layer.ffn_hc)
            stages = [copy.deepcopy(reference) for _ in range(4)]
            for rank, model in enumerate(stages):
                mesh = SimpleNamespace(pp_size=4, pp_rank=rank)
                partition_model(model, mesh, PipelineConfig(split_points=[10, 22, 30]))
            tokens = torch.randint(1, 64, (1, 8))
            media = {}
            if vision:
                media = {
                    "pixel_values": torch.randn(1, 3, 2, 2),
                    "token_types": torch.tensor([[0, 1, 2, 3, -1, -1, -1, -1]]),
                    "image_patch_offsets": torch.tensor([0, 1]),
                    "image_vit_grid_hw": torch.tensor([[1, 1]]),
                    "image_llm_grid_hw": torch.tensor([[1, 1]]),
                    "image_batch_indices": torch.tensor([0]),
                    "image_token_starts": torch.tensor([0]),
                }
            if dspark:
                media["labels"] = tokens
            expected_output = reference(input_ids=tokens, use_cache=False, **media)
            expected = expected_output.logits
            expected_loss = expected_output.loss if dspark else expected.float().square().mean()
            expected_loss.backward()
            state = ()
            for stage in stages:
                state = stage(input_ids=tokens, pipeline_inputs=state, use_cache=False, **media)
            actual = state.logits
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            actual_loss = state.loss if dspark else actual.float().square().mean()
            torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-5, atol=1e-6)
            actual_loss.backward()
            expected_parameters = dict(reference.named_parameters())
            seen = set()
            for stage in stages:
                for name, parameter in stage.named_parameters():
                    self.assertNotIn(name, seen)
                    seen.add(name)
                    baseline = expected_parameters[name].grad
                    self.assertEqual(parameter.grad is None, baseline is None, name)
                    if baseline is not None:
                        torch.testing.assert_close(parameter.grad, baseline, rtol=1e-5, atol=1e-6, msg=name)
            self.assertEqual(seen, set(expected_parameters))

    def test_partition_prunes_only_retired_parameter_contracts(self) -> None:
        """Discard retired endpoint layouts while preserving unknown names for validation."""
        model = torch.nn.Linear(2, 2)
        model._hp_pipeline_removed_parameters = frozenset({"backbone.embedding"})
        layout = {"tp": "replicate"}
        spec = SimpleNamespace(params={"embedding": layout, "kept": layout, "typo": layout})
        plan = SimpleNamespace(modules={"backbone": spec})
        _prune_pipeline_parameter_contracts(model, plan)
        self.assertEqual(set(spec.params), {"kept", "typo"})
        untouched = SimpleNamespace(params={"embedding": layout})
        _prune_pipeline_parameter_contracts(torch.nn.Linear(2, 2), SimpleNamespace(modules={"backbone": untouched}))
        self.assertEqual(untouched.params, {"embedding": layout})

    def test_dspark_partition_preserves_joint_loss_and_gradients(self) -> None:
        """Keep all three drafter targets on the tail and preserve the joint objective."""
        self._check_four_stages(False, vision=True, dspark=True)

    def test_vision_partition_preserves_outputs_and_gradients(self) -> None:
        """Keep vision parameters unique and their gradients equal across four decoder stages."""
        self._check_four_stages(False, vision=True)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_deepseek_shared_state_crosses_multiple_cuts(self) -> None:
        """Feature: DeepSeek V4.1 pipeline partitioning.
        Description: Cut the decoder into four stages with shared KV and index state.
        Expectation: Logits and gradients for all owned parameters match the unsplit model.
        """
        self._check_four_stages(False)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_shifted_mhc_across_multiple_cuts(self) -> None:
        """Feature: DeepSeek V4.1 pipeline partitioning.
        Description: Replace the mHC modules and run the four-stage decoder.
        Expectation: Deferred FFN outputs and mixing coefficients preserve gradients across each cut.
        """
        self._check_four_stages(True)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_invalid_split_is_rejected_before_pruning(self) -> None:
        """Feature: DeepSeek V4.1 pipeline partitioning.
        Description: Provide two decreasing layer boundaries to a three-stage partition.
        Expectation: Partitioning raises ValueError and leaves all forty layers in the source.
        """
        with tempfile.TemporaryDirectory() as directory:
            model = DeepseekV41ForCausalLM(make_config(directory))
            with self.assertRaisesRegex(ValueError, "nonempty ordered"):
                partition_model(model, SimpleNamespace(pp_size=3, pp_rank=0), PipelineConfig(split_points=[20, 10]))
            self.assertEqual(len(model.model.layers), 40)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_virtual_stage_configuration(self) -> None:
        """Feature: Interleaved Trainer pipeline layout.
        Description: Resolve three local chunks on two ranks and reject invalid local chunk counts.
        Expectation: Ranges partition the full decoder and virtual stage IDs follow round-robin ownership.
        """
        self.assertEqual(PipelineConfig(schedule="interleaved_1f1b").local_stage_count, 2)
        for count in (0, -1, True, 1.5):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "positive integer"):
                PipelineConfig(schedule="interleaved_1f1b", virtual_stages_per_rank=count)
        with self.assertRaisesRegex(ValueError, "require interleaved"):
            PipelineConfig(schedule="gpipe", virtual_stages_per_rank=2)
        with tempfile.TemporaryDirectory() as directory, torch.device("meta"):
            config = PipelineConfig(schedule="interleaved_1f1b", virtual_stages_per_rank=3)
            reference = DeepseekV41ForCausalLM(make_config(directory))
            for rank in range(2):
                for chunk in range(3):
                    model = copy.deepcopy(reference)
                    partition_model(model, SimpleNamespace(pp_size=2, pp_rank=rank), config, chunk_index=chunk)
                    stage = chunk * 2 + rank
                    expected = range(40 * stage // 6, 40 * (stage + 1) // 6)
                    self.assertEqual(list(model.model.layers), [str(i) for i in expected])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_pp1_build_with_pipeline_config_keeps_full_model(self) -> None:
        """Feature: Default PP1 model construction with a Trainer pipeline config.
        Description: Build through apply_model_infrastructure with pipeline_config set and pp_size=1.
        Expectation: No graph-mode NameError and all forty decoder layers remain unsplit.
        """
        with tempfile.TemporaryDirectory() as directory:
            model = DeepseekV41ForCausalLM(make_config(directory))
            mesh = MeshContext(pp_size=1)
            setup = SimpleNamespace(
                pipeline_config=PipelineConfig(), pipeline_chunk_index=0, mesh_context=mesh,
                method_replacements=None, plan_overrides=None, allow_uncovered_params=False,
                strategy_config=None, low_precision_config=None, fp32_main_params=False,
            )
            model = apply_model_infrastructure(model, mesh=mesh, distributed_setup=setup, device=torch.device("cpu"))
            self.assertEqual(len(model.model.layers), 40)

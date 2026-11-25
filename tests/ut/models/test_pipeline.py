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

import tempfile
from types import SimpleNamespace
import unittest

import torch

from hyper_parallel.components.losses.chunked_cross_entropy import ChunkedCausalLMLoss
from hyper_parallel.core.activation_memory.wrapper import CheckpointWrapper
from hyper_parallel.distributed.activation_checkpoint import _apply_activation_checkpointing, _resolve_activation_plan
from hyper_parallel.trainer.config.parallelism import (
    ActivationCheckpointConfig, FullCheckpointConfig, SelectiveCheckpointConfig,
)
from hyper_parallel.models.pipeline import PipelineConfig, partition_model
from hyper_parallel.models.deepseek_v41.adapter.distributed.mega_mhc import DeepseekV41HyperMegaMhc
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41ForCausalLM
from tests.common.deepseek_v41_pipeline import copy_weights, make_config
from tests.common.mark_utils import arg_mark


class TestPipelinePartition(unittest.TestCase):
    """Compare a segmented model's outputs and trainable gradients with its source."""

    def _check_four_stages(self, shifted: bool) -> None:
        """Compare the complete parameter and gradient sets against the source model."""
        torch.manual_seed(71)
        with tempfile.TemporaryDirectory() as directory:
            reference = DeepseekV41ForCausalLM(make_config(directory))
            stages = [copy_weights(reference) for _ in range(4)]
            if shifted:
                for model in [reference, *stages]:
                    for layer in model.model.layers:
                        layer.attn_hc = DeepseekV41HyperMegaMhc(module=layer.attn_hc)
                        layer.ffn_hc = DeepseekV41HyperMegaMhc(module=layer.ffn_hc)
            for rank, model in enumerate(stages):
                mesh = SimpleNamespace(pp_size=4, pp_rank=rank)
                partition_model(model, mesh, PipelineConfig(split_points=[10, 22, 30]))
            tokens = torch.randint(1, 64, (1, 8))
            expected = reference(input_ids=tokens, use_cache=False).logits
            expected.float().square().mean().backward()
            state = ()
            for stage in stages:
                state = stage(input_ids=tokens, pipeline_inputs=state, use_cache=False)
            actual = state.logits
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            actual.float().square().mean().backward()
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
    def test_checkpoint_targets_use_global_layer_ids(self) -> None:
        """Feature: Global activation checkpoint targets across virtual chunks.
        Description: Place a full-layer, a router and a model-level target on different stages.
        Expectation: Each wrapper is installed only on its owner under the original name.
        """
        config = ActivationCheckpointConfig(
            mode="hybrid", full=FullCheckpointConfig(recompute_layer=[14]),
            selective=SelectiveCheckpointConfig(recompute=[{"mlp.gate": [21]}, "lm_head"]),
        )
        pipeline = PipelineConfig(schedule="interleaved_1f1b", virtual_stages_per_rank=3)
        expected = {2: {"model.layers.14"}, 3: {"model.layers.21.mlp.gate"}, 5: {"lm_head"}}
        with tempfile.TemporaryDirectory() as directory:
            for stage_index in range(6):
                model = DeepseekV41ForCausalLM(make_config(directory))
                plan = _resolve_activation_plan(model, config)
                mesh = SimpleNamespace(pp_rank=stage_index % 2, pp_size=2)
                partition_model(model, mesh, pipeline, chunk_index=stage_index // 2)
                _apply_activation_checkpointing(model, config, plan=plan.for_stage(model))
                wrapped = {name for name, module in model.named_modules() if isinstance(module, CheckpointWrapper)}
                self.assertEqual(wrapped, expected.get(stage_index, set()))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_pipeline_chunked_loss_on_tail(self) -> None:
        """Feature: Terminal chunked CE across a decoder partition.
        Description: Bind the Trainer criterion to both stages and pass the first stage's output to the head.
        Expectation: Summed loss, token count and head gradient match the full model.
        """
        torch.manual_seed(17)
        with tempfile.TemporaryDirectory() as directory:
            reference = DeepseekV41ForCausalLM(make_config(directory))
            first, last = copy_weights(reference), copy_weights(reference)
            criterion = ChunkedCausalLMLoss(chunk_size=3)
            criterion.bind_model(reference)
            for rank, part in enumerate((first, last)):
                mesh = SimpleNamespace(pp_size=2, pp_rank=rank)
                partition_model(part, mesh, PipelineConfig(split_points=[22]))
                criterion.bind_model(part, SimpleNamespace(mesh_context=mesh))
            ids = torch.randint(1, 64, (1, 8))
            labels = ids.clone()
            labels[:, -1] = -100
            inputs = criterion.prepare_model_inputs({"input_ids": ids, "use_cache": False}, {"shift_labels": labels})
            expected = reference(**inputs)
            output = last(**inputs, pipeline_inputs=first(input_ids=ids, use_cache=False))
            self.assertEqual(output.valid_token_count.item(), 7)
            torch.testing.assert_close(output.loss_sum, expected.loss_sum)
            expected.loss_sum.backward()
            output.loss_sum.backward()
            torch.testing.assert_close(last.lm_head.weight.grad, reference.lm_head.weight.grad)

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
            model_config = make_config(directory)
            for rank in range(2):
                for chunk in range(3):
                    model = DeepseekV41ForCausalLM(model_config)
                    partition_model(model, SimpleNamespace(pp_size=2, pp_rank=rank), config, chunk_index=chunk)
                    stage = chunk * 2 + rank
                    expected = range(40 * stage // 6, 40 * (stage + 1) // 6)
                    self.assertEqual(list(model.model.layers), [str(i) for i in expected])

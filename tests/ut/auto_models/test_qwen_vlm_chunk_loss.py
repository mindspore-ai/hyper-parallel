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
"""VLM-only chunk loss numerics, native forward preservation and batch alignment."""

import copy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch
from torch.nn import functional
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForConditionalGeneration

from hyper_parallel.components.losses import ChunkedCausalLMLoss
from hyper_parallel.data.batching.runtime_input import RuntimeInputAdapter
from hyper_parallel.models.qwen3_5_moe.adapter.vlm_chunk_loss import bind_chunk_loss
from hyper_parallel.models.qwen3_5_moe.vlm_cp import QwenVLMParallelBatch, install_qwen_vlm_cp_forward
from hyper_parallel.trainer.runtime.loss_aggregation import count_loss_token
from tests.common.mark_utils import arg_mark


def make_model() -> Qwen3_5MoeForConditionalGeneration:
    """Build a tiny real multimodal decoder with both GDN and full attention."""
    config = Qwen3_5MoeConfig(
        image_token_id=99, video_token_id=98, vision_start_token_id=97,
        text_config={"vocab_size": 128, "hidden_size": 32, "num_hidden_layers": 2,
                     "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
                     "num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 16,
                     "shared_expert_intermediate_size": 16, "layer_types": ["linear_attention", "full_attention"],
                     "linear_num_key_heads": 2, "linear_num_value_heads": 4,
                     "linear_key_head_dim": 8, "linear_value_head_dim": 8, "linear_conv_kernel_dim": 4,
                     "rope_parameters": {"rope_type": "default", "rope_theta": 10000,
                                         "partial_rotary_factor": 1.0, "mrope_section": [1, 1, 2]}},
        vision_config={"hidden_size": 16, "intermediate_size": 32, "num_heads": 4, "depth": 2,
                       "out_hidden_size": 32, "patch_size": 2, "spatial_merge_size": 2,
                       "temporal_patch_size": 2, "num_position_embeddings": 16},
    )
    config._attn_implementation = "sdpa"
    return Qwen3_5MoeForConditionalGeneration(config)


def make_batch(image_start: int = 62, supervised_start: int = 2) -> dict:
    """Put one image across a CP boundary or entirely on the first rank."""
    ids = torch.randint(1, 80, (1, 100))
    ids[:, image_start:image_start + 4] = 99
    labels = ids.clone()
    labels[:, :supervised_start] = -100
    labels[ids == 99] = -100
    return {"input_ids": ids, "labels": labels, "mm_token_type_ids": ids.eq(99).long(),
            "pixel_values": torch.randn(16, 24), "image_grid_thw": torch.tensor([[1, 4, 4]])}


class TestQwenVLMChunkLoss(unittest.TestCase):
    """Compare loss and every gradient against a full-logits HF oracle on CPU."""

    def setUp(self) -> None:
        """Use a fixed seed and one CPU thread for the small numerical oracle."""
        torch.manual_seed(42)
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self) -> None:
        """Restore the caller's CPU thread setting."""
        torch.set_num_threads(self.old_threads)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_native_and_chunk_gradients(self) -> None:
        """Feature: VLM chunk loss.

        Description: Compare uneven chunks and empty supervision against full logits.
        Expectation: Loss and every image/text parameter gradient match the reference.
        """
        for empty in (False, True):
            with self.subTest(empty=empty):
                model = make_model()
                reference = copy.deepcopy(model)
                batch = make_batch()
                targets = functional.pad(batch["labels"], (0, 1), value=-100)[:, 1:]
                if empty:
                    targets.fill_(-100)
                objective = ChunkedCausalLMLoss(chunk_size=17)
                objective.bind_model(model)
                prepared = objective.prepare_model_inputs(batch, {"shift_labels": targets})
                output = model(**prepared, use_cache=False)
                actual = objective(model_output=output, labels=None)
                full_inputs = {key: value for key, value in batch.items() if key != "labels"}
                logits = reference(**full_inputs, use_cache=False).logits
                expected = functional.cross_entropy(logits.float().flatten(0, 1), targets.flatten(),
                                                    ignore_index=-100, reduction="sum")
                expected = expected / targets.ne(-100).sum().clamp_min(1)
                self.assertIsNone(output.logits)
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
                actual.backward()
                expected.backward()
                for (name, parameter), ref in zip(model.named_parameters(), reference.parameters()):
                    self.assertIsNotNone(parameter.grad, name)
                    torch.testing.assert_close(parameter.grad, ref.grad, atol=2e-6, rtol=2e-4,
                                               msg=lambda message, name=name: f"{name}: {message}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_default_forward_and_text_guard(self) -> None:
        """Feature: VLM-only loss binding.

        Description: Bind twice, run native inference, and reject a text-model facade.
        Expectation: Native logits remain identical and text model binding raises TypeError.
        """
        model = make_model()
        batch = make_batch()
        expected = model(**batch, use_cache=False).logits
        bind_chunk_loss(model)
        bind_chunk_loss(model)
        torch.testing.assert_close(model(**batch, use_cache=False).logits, expected, atol=0, rtol=0)
        model._hp_qwen_vlm_chunk_loss_bound = False
        model.config.model_type = "qwen3_5_moe_text"
        with self.assertRaisesRegex(TypeError, "model_type"):
            bind_chunk_loss(model)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_cp1_batch_shift_and_count(self) -> None:
        """Feature: Non-CP VLM chunk targets.

        Description: Prepare image/text labels with right padding and chunk loss enabled.
        Expectation: Targets shift once, masks/counts align, and no logits are materialized.
        """
        model = make_model()
        objective = ChunkedCausalLMLoss(chunk_size=17)
        objective.bind_model(model)
        context = SimpleNamespace(cp_size=1, tp_size=1, pp_size=1, dp_size=1, dp_rank=0,
                                  cp_rank=0, tp_rank=0, sequence_parallel=False, loss_parallel=False)
        batcher = QwenVLMParallelBatch(mesh_context=context, device=torch.device("cpu"))
        batcher.bind_model(model)
        batch = make_batch()
        inputs, loss_inputs = batcher(None, external_batch=batch)
        expected = functional.pad(batch["labels"], (0, 29), value=-100)[:, 1:]
        torch.testing.assert_close(inputs["shift_labels"], expected, atol=0, rtol=0)
        torch.testing.assert_close(loss_inputs["loss_mask"], expected.ne(-100), atol=0, rtol=0)
        self.assertEqual(int(count_loss_token(loss_inputs)["foundation_tokens"]), int(expected.ne(-100).sum()))
        output = model(**objective.prepare_model_inputs(inputs, loss_inputs), use_cache=False)
        self.assertIsNone(output.logits)

    def test_cp1_calls_runtime_adapter(self) -> None:
        """Feature: CP=1 encoder/runtime composition.

        Description: Supply a runtime extension to the Qwen batch adapter.
        Expectation: Its plan reaches model inputs but not loss inputs.
        """
        model = make_model()
        context = SimpleNamespace(cp_size=1, tp_size=1, pp_size=1, dp_size=1, dp_rank=0,
                                  cp_rank=0, tp_rank=0, sequence_parallel=False, loss_parallel=False)
        extension = Mock(spec=RuntimeInputAdapter)
        plan = object()
        extension.build.return_value = {"encoder_parallel_plan": plan}
        batcher = QwenVLMParallelBatch(mesh_context=context, device=torch.device("cpu"),
                                      runtime_input_adapter=extension)
        batcher.bind_model(model)
        inputs, loss_inputs = batcher(None, external_batch=make_batch())
        extension.build.assert_called_once()
        self.assertIs(inputs["encoder_parallel_plan"], plan)
        self.assertNotIn("encoder_parallel_plan", loss_inputs)

    def test_invalid_alignment_is_rejected(self) -> None:
        """Feature: Alignment boundary validation.

        Description: Configure zero, negative, fractional, or boolean alignment.
        Expectation: Reject at construction before padding or communication.
        """
        context = SimpleNamespace(cp_size=1, tp_size=1, sequence_parallel=False)
        for alignment in (0, -64, 64.0, True):
            with self.subTest(alignment=alignment), self.assertRaisesRegex(ValueError, "positive integer"):
                QwenVLMParallelBatch(mesh_context=context, device=torch.device("cpu"),
                                     sequence_alignment=alignment)

    def test_cp_rejects_implicit_router_loss_and_partial_logits(self) -> None:
        """Feature: CP forward contract.

        Description: Exercise config-default router output and incompatible HF output options.
        Expectation: Reject before encoder work or decoder execution.
        """
        model = make_model()
        install_qwen_vlm_cp_forward(model)
        model.config.text_config.output_router_logits = True
        with self.assertRaisesRegex(NotImplementedError, "router auxiliary loss"):
            model(vlm_cp_inputs=object())
        for options, message in (({"return_dict": False}, "return_dict"),
                                 ({"logits_to_keep": 1}, "logits_to_keep"),
                                 ({"logits_to_keep": torch.tensor([0])}, "logits_to_keep")):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, message):
                model(vlm_cp_inputs=object(), output_router_logits=False, **options)

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
"""DeepSeek decoder merge receives the TP-local token view after embedding."""

from types import SimpleNamespace
import unittest

import torch

from hyper_parallel.distributed.encoder_parallel.planning import DecoderSequenceLayout
from hyper_parallel.data.batching.runtime_input import RuntimeInputContext
from hyper_parallel.models.deepseek_v41.adapter.data.runtime import DeepseekV41Runtime
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41Model


class TestEncoderMerge(unittest.TestCase):
    """Keep CP-local input IDs and route only local auxiliary rows to merge."""

    def test_merge_uses_tp_local_token_types_and_global_start(self):
        """Feature: DeepSeek image merge after TP reduce-scatter.

        Description: Prepare a CP0/TP3 decoder shard from CP-local token types.
        Expectation: Merge receives [96,128) and active anchors enter backward.
        """
        context = SimpleNamespace(cp_size=2, tp_size=4, cp_rank=0, tp_rank=3,
                                  sequence_parallel=True)
        layout = DecoderSequenceLayout.build(mesh_context=context, cp_local_length=128)
        vision_anchor = torch.zeros((), requires_grad=True)
        parameter_anchor = torch.zeros((), requires_grad=True)
        captured = []

        def _merge(hidden, local_types, records, global_start):
            captured.append((local_types, records, global_start))
            return hidden

        model = SimpleNamespace(
            embed_tokens=lambda input_ids: torch.ones((1, 32, 8), requires_grad=True),
            _merge_image_embeddings=_merge,
            image_start=parameter_anchor,
            image_end=torch.zeros(()),
            image_newline=torch.zeros(()),
        )
        item = SimpleNamespace(batch_index=0, decoder_token_start=96, metadata={"llm_hw": (1, 1)})
        feature = torch.zeros((1, 8))
        record = (item, feature)
        plan = SimpleNamespace(
            encode_and_route=lambda _image_inputs: ([record], vision_anchor),
            decoder_layout=layout,
            active=True,
        )
        token_types = torch.arange(128).unsqueeze(0)
        output = DeepseekV41Model._prepare_image_embeddings(  # pylint: disable=protected-access
            model, input_ids=torch.zeros(1, 128, dtype=torch.long), inputs_embeds=None,
            token_types=token_types, image_inputs={name: None for name in (
                "pixel_values", "image_patch_offsets", "image_vit_grid_hw", "image_llm_grid_hw",
                "image_batch_indices", "image_token_starts",
            )}, image_sequence_start=0, encoder_parallel_plan=plan,
        )
        self.assertEqual(output.shape, (1, 32, 8))
        self.assertEqual(len(captured), 1)
        self.assertTrue(torch.equal(captured[0][0], token_types[:, 96:128]))
        self.assertEqual(captured[0][1][0][:4], (0, 96, 1, 1))
        self.assertIs(captured[0][1][0][4], feature)
        self.assertEqual(captured[0][2], 96)
        output.sum().backward()
        self.assertIsNotNone(vision_anchor.grad)
        self.assertIsNotNone(parameter_anchor.grad)

    def test_disabled_encoder_merges_the_tp_local_image_interval(self):
        """Feature: native visual path with Decoder sequence parallelism.

        Description: Merge a repeated visual record on one TP-local shard.
        Expectation: Token types and image start use the same TP interval as the active path.
        """
        captured = []
        model = SimpleNamespace(
            embed_tokens=lambda _input_ids: torch.ones((1, 32, 8)),
            _collect_image_records=lambda *_args: ["image-record"],
            _merge_image_embeddings=lambda hidden, types, records, start: (
                captured.append((types, records, start)) or hidden
            ),
        )
        token_types = torch.arange(128).unsqueeze(0)
        image_inputs = {name: None for name in (
            "pixel_values", "image_patch_offsets", "image_vit_grid_hw", "image_llm_grid_hw",
            "image_batch_indices", "image_token_starts",
        )}

        DeepseekV41Model._prepare_image_embeddings(  # pylint: disable=protected-access
            model, input_ids=torch.zeros(1, 128, dtype=torch.long), inputs_embeds=None,
            token_types=token_types, image_inputs=image_inputs,
            image_sequence_start=96, encoder_parallel_plan=None,
        )

        self.assertEqual(len(captured), 1)
        self.assertTrue(torch.equal(captured[0][0], token_types[:, 96:128]))
        self.assertEqual(captured[0][1:], (["image-record"], 96))

    def test_runtime_image_start_follows_sequence_parallel_shard(self):
        """Feature: runtime image insertion coordinate under Decoder SP.

        Description: Build runtime inputs for CP1/TP3 from CP-local position metadata.
        Expectation: The image start points to the TP-local global token interval.
        """
        batch = {
            "input_ids": torch.zeros(1, 128, dtype=torch.long),
            "cu_seq_lens": torch.tensor([0, 256], dtype=torch.int32),
        }
        context = RuntimeInputContext(
            local_input_shape=(1, 128), parallel_ranks={"cp": 1, "tp": 3},
            parallel_sizes={"cp": 2, "tp": 4}, options={"sequence_parallel": True},
        )
        runtime = DeepseekV41Runtime()

        result = runtime.build_runtime_inputs(batch=batch, context=context)

        self.assertEqual(result["image_sequence_start"], 224)
        self.assertEqual(int(result["position_ids"][0, 0]), 128)
        self.assertEqual(result["packed_seq_params"].local_query_start, 128)
        no_sp_context = RuntimeInputContext(
            local_input_shape=(1, 128), parallel_ranks={"cp": 1, "tp": 3},
            parallel_sizes={"cp": 2, "tp": 4}, options={"sequence_parallel": False},
        )
        no_sp = runtime.build_runtime_inputs(batch=batch, context=no_sp_context)
        self.assertEqual(no_sp["image_sequence_start"], 128)

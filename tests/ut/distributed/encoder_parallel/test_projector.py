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
"""Spatial-block ownership and packed-image isolation without collectives."""

from types import SimpleNamespace
import unittest

import torch
from torch import nn

from hyper_parallel.distributed.encoder_parallel.contracts import EncoderWorkItem
from hyper_parallel.distributed.encoder_parallel.planning import build_segments
from hyper_parallel.distributed.encoder_parallel.projector_dispatch import build_projector_plan
from hyper_parallel.models.deepseek_v41.adapter.distributed.encoder_parallel import DeepseekV41VisionBridge
from hyper_parallel.models.deepseek_v41.vision import DeepseekV41VisionAligner, _vision_packed_attention


class TestPackedProjector(unittest.TestCase):
    """Check image isolation, spatial padding and projector backward."""

    def test_packed_shards_preserve_every_patch_once(self):
        """Feature: bucket-level sequence partition.

        Description: Split three odd image lengths over four ranks.
        Expectation: Each original patch has one owner and only the tail is padded.
        """
        items = tuple(EncoderWorkItem(i, 0, length, 1, 0, 3, 0, length, {})
                      for i, length in enumerate((5, 3, 1)))
        covered = []
        padded = 0
        for rank in range(4):
            segments = build_segments(items, rank, 4, packed=True)
            self.assertEqual(sum(segment.local_padded_length for segment in segments), 3)
            for segment in segments:
                if segment.is_dummy:
                    padded += segment.local_padded_length
                else:
                    covered.extend((segment.item.item_id, index) for index in range(
                        segment.local_start, segment.local_start + segment.local_valid_length))
        self.assertEqual(covered, [(item.item_id, index) for item in items for index in range(item.sequence_length)])
        self.assertEqual(padded, 3)

    def test_packed_attention_isolates_images_and_gradients(self):
        """Feature: packed visual attention isolation.

        Description: Perturb the second image in a packed sequence.
        Expectation: The first image's output and its dependency on other images remain zero.
        """
        torch.manual_seed(7)
        query, key, value = (torch.randn(7, 2, 8, requires_grad=True) for _ in range(3))
        output = _vision_packed_attention(nn.Identity(), query, key, value, (3, 4))
        perturbed = value.detach().clone()
        perturbed[3:] += 100
        changed = _vision_packed_attention(nn.Identity(), query, key, perturbed, (3, 4))
        torch.testing.assert_close(output[:3], changed[:3], rtol=0, atol=0)
        output[:3].sum().backward()
        for tensor in (query, key, value):
            self.assertEqual(torch.count_nonzero(tensor.grad[3:]).item(), 0)

    def test_spatial_blocks_match_native_unfold_backward(self):
        """Feature: distributed aligner block layout.

        Description: Reassemble spatial blocks for an odd 5-by-7 image across three owners.
        Expectation: Outputs and all gradients agree with the original unfold path.
        """
        config = SimpleNamespace(v41_vision_hidden_size=8, v41_vision_downsample_ratio=3, hidden_size=12)
        aligner = DeepseekV41VisionAligner(config)
        bridge = SimpleNamespace(aligner=aligner, projector_patch_count=9)
        bridge.projector_input_indices = lambda *args: DeepseekV41VisionBridge.projector_input_indices(bridge, *args)
        item = EncoderWorkItem(0, 0, 35, 6, 0, 10, 0, 35, {"vit_hw": (5, 7), "llm_hw": (2, 3)})
        hidden = torch.randn(35, 8, requires_grad=True)
        params = tuple(aligner.parameters())
        expected = aligner(hidden, 5, 7)
        expected_grad = torch.autograd.grad(expected.square().sum(), (hidden, *params))
        for packed in (False, True):
            with self.subTest(packed=packed):
                plans = [build_projector_plan((item,), rank, 3, packed=packed, bridge=bridge) for rank in range(3)]
                blocks = []
                for plan in plans:
                    indices = plan.bank_indices
                    rows = hidden.index_select(0, indices.clamp_min(0))
                    rows = rows.masked_fill((indices < 0).unsqueeze(1), 0)
                    blocks.append(rows.reshape(-1, plan.patch_count, hidden.shape[-1]))
                actual = torch.cat([DeepseekV41VisionBridge.project_blocks(bridge, part) for part in blocks])
                torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
                gradients = torch.autograd.grad(actual.square().sum(), (hidden, *params))
                for result, reference in zip(gradients, expected_grad):
                    torch.testing.assert_close(result, reference, rtol=1e-5, atol=1e-5)
                for source in range(3):
                    for destination in range(3):
                        self.assertEqual(plans[source].input_splits[destination],
                                         plans[destination].output_splits[source])

    def test_feature_intervals_exclude_image_markers(self):
        """Feature: exact decoder image-row intervals.

        Description: Split image start, newline and end markers across token owners.
        Expectation: Each feature row is sent once and markers do not request feature data.
        """
        item = EncoderWorkItem(0, 0, 35, 6, 10, 10, 0, 35, {"llm_hw": (2, 3)})
        intervals = [DeepseekV41VisionBridge.feature_interval(item, start, stop)
                     for start, stop in ((0, 11), (11, 14), (14, 15), (15, 18), (18, 25))]
        self.assertEqual(intervals, [(0, 0), (0, 3), (3, 3), (3, 6), (6, 6)])

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
"""Unit tests for Qwen3-VL-MoE DeepStack feature slicing."""
# pylint: disable=protected-access

import torch

from hyper_parallel.models.qwen3_vl_moe.model import _cp_slice_deepstack_features


def test_cp_slice_deepstack_features_selects_by_full_sequence_positions():
    """A local CP shard must map back to visual feature rows in full-sequence order."""
    visual_pos_masks = torch.tensor([
        [False, True, False, False, True, False],
        [True, False, False, False, False, True],
    ])
    features = [torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3)]

    sliced = _cp_slice_deepstack_features(features, visual_pos_masks, slice(3, 6))

    expected = features[0].index_select(0, torch.tensor([1, 3]))
    torch.testing.assert_close(sliced[0], expected)


def test_cp_slice_deepstack_features_keeps_empty_local_shard_shape():
    """A CP shard without visual tokens should return empty rows, not fail indexing."""
    visual_pos_masks = torch.zeros((1, 32), dtype=torch.bool)
    visual_pos_masks[:, 1:9] = True
    features = [torch.arange(8 * 2, dtype=torch.float32).reshape(8, 2)]

    sliced = _cp_slice_deepstack_features(features, visual_pos_masks, slice(16, 32))

    assert sliced[0].shape == (0, 2)


def test_cp_slice_deepstack_features_applies_same_indices_to_all_layers():
    """All DeepStack layers use the same selected visual-token rows."""
    visual_pos_masks = torch.zeros((1, 8), dtype=torch.bool)
    visual_pos_masks[:, [1, 3, 5, 7]] = True
    features = [
        torch.arange(4 * 2, dtype=torch.float32).reshape(4, 2),
        torch.arange(4 * 2, dtype=torch.float32).reshape(4, 2) + 100,
    ]

    sliced = _cp_slice_deepstack_features(features, visual_pos_masks, slice(4, 8))

    expected_indices = torch.tensor([2, 3])
    assert len(sliced) == 2
    for actual, feature in zip(sliced, features):
        torch.testing.assert_close(actual, feature.index_select(0, expected_indices))


def test_cp_slice_deepstack_features_preserves_none_inputs():
    """The helper keeps optional inputs unchanged for non-DeepStack paths."""
    features = [torch.ones(1, 2)]
    visual_pos_masks = torch.ones((1, 1), dtype=torch.bool)

    assert _cp_slice_deepstack_features(None, visual_pos_masks, slice(0, 1)) is None
    assert _cp_slice_deepstack_features(features, None, slice(0, 1)) is features

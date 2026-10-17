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
"""Unit tests for parameter-sink attention rescaling."""

import importlib
import unittest
from unittest.mock import Mock, PropertyMock, patch

import torch

from hyper_parallel.components.functional.attention_rescale import _AttentionRescale

attention_rescale_module = importlib.import_module("hyper_parallel.components.functional.attention_rescale")


class TestAttentionRescale(unittest.TestCase):
    """Validate the attention-rescale autograd bridge."""

    def test_backward_unpacks_saved_tensors_once(self) -> None:
        """Non-reentrant checkpoint hooks must only be consumed once."""
        saved_tensors = tuple(torch.full((1,), index, dtype=torch.float32) for index in range(13))
        context = Mock()
        gradient = torch.ones(1)
        normal_gradients = tuple(torch.full((1,), index, dtype=torch.float32) for index in range(1, 4))
        sink_gradients = tuple(torch.full((1,), index, dtype=torch.float32) for index in range(4, 7))

        with (
            patch.object(
                type(context),
                "saved_tensors",
                create=True,
                new_callable=PropertyMock,
                side_effect=[saved_tensors, RuntimeError("duplicate unpack")],
            ) as unpack,
            patch.object(
                attention_rescale_module, "_normal_attention_backward", return_value=normal_gradients,
            ) as normal,
            patch.object(
                attention_rescale_module, "_sink_attention_backward", return_value=sink_gradients,
            ) as sink,
        ):
            result = _AttentionRescale.backward(context, gradient, None)

        unpack.assert_called_once_with()
        normal.assert_called_once_with(context, gradient, saved_tensors)
        sink.assert_called_once_with(context, gradient, saved_tensors)
        self.assertTrue(torch.equal(result[0], normal_gradients[0] + sink_gradients[0]))

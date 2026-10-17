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
"""Unit tests for DSA parameter-sink attention rescaling."""

import importlib
import unittest
from unittest.mock import Mock, PropertyMock, patch

import torch

from hyper_parallel.components.functional.dsa_sparse_attention_rescale import (
    _SparseAttentionRescale,
)

dsa_rescale_module = importlib.import_module(
    "hyper_parallel.components.functional.dsa_sparse_attention_rescale"
)


class TestSparseAttentionRescale(unittest.TestCase):
    """Validate the DSA attention-rescale autograd bridge."""

    def test_backward_unpacks_saved_tensors_once(self) -> None:
        """Non-reentrant checkpoint hooks must only be consumed once."""
        saved_tensors = tuple(torch.full((1, 1, 1, 1), index, dtype=torch.float32) for index in range(14))
        context = Mock()
        gradient = torch.ones(1)
        sparse_gradients = tuple(torch.ones(1, 1, 1, 1) for _ in range(5))
        sink_gradients = (
            torch.ones(1, 1, 1, 2),
            torch.ones(1),
            torch.ones(1),
        )

        with (
            patch.object(
                type(context),
                "saved_tensors",
                create=True,
                new_callable=PropertyMock,
                side_effect=[saved_tensors, RuntimeError("duplicate unpack")],
            ) as unpack,
            patch.object(
                dsa_rescale_module, "_sparse_attention_backward", return_value=sparse_gradients,
            ) as sparse,
            patch.object(
                dsa_rescale_module, "_sink_attention_backward", return_value=sink_gradients,
            ) as sink,
        ):
            result = _SparseAttentionRescale.backward(context, gradient, None, None)

        unpack.assert_called_once_with()
        sparse.assert_called_once_with(context, gradient, saved_tensors)
        sink.assert_called_once_with(context, gradient, saved_tensors)
        self.assertTrue(torch.equal(result[0], torch.full((1, 1, 1, 1), 2.0)))

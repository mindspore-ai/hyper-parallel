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
"""Unit tests for torch fully_shard post-backward autograd hook."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

# pylint: disable=wrong-import-position
import torch

from hyper_parallel.platform.torch.fully_shard.hook_function import PostBackwardFunction


class TestPostBackwardFunction(unittest.TestCase):
    """Cover autograd hook behavior without distributed runtime."""

    def test_backward_triggers_scheduler_hook_and_preserves_gradient(self):
        """The autograd wrapper should pass gradients through and call the scheduler hook once."""
        scheduler = SimpleNamespace(_backward_hook=MagicMock())
        x = torch.tensor([2.0, 3.0], requires_grad=True)

        (wrapped,) = PostBackwardFunction.apply(scheduler, x)
        wrapped.sum().backward()

        scheduler._backward_hook.assert_called_once_with()
        torch.testing.assert_close(x.grad, torch.ones_like(x))

    @patch("hyper_parallel.DTensor")
    def test_apply_restores_dtensor_layout_outputs(self, mock_dtensor):
        """DTensor-like inputs should be unwrapped for autograd and rewrapped with the same layout."""
        scheduler = MagicMock()
        local = torch.tensor([1.0], requires_grad=True)
        layout = SimpleNamespace(mesh="mesh", alias_placements=("shard",))
        dtensor_input = SimpleNamespace(_layout=layout, layout=layout, to_local=MagicMock(return_value=local))
        mock_dtensor.from_local.return_value = "rewrapped"

        output = PostBackwardFunction.apply(scheduler, dtensor_input)

        dtensor_input.to_local.assert_called_once_with()
        mock_dtensor.from_local.assert_called_once_with(local, "mesh", ("shard",))
        self.assertEqual(output, ("rewrapped",))

    def test_apply_preserves_none_outputs_for_optional_hook_inputs(self):
        """Optional tensor slots should round-trip through the autograd wrapper as None."""
        scheduler = SimpleNamespace(_backward_hook=MagicMock())
        x = torch.tensor([1.0], requires_grad=True)

        none_output, tensor_output = PostBackwardFunction.apply(scheduler, None, x)

        self.assertIsNone(none_output)
        torch.testing.assert_close(tensor_output, x)


if __name__ == "__main__":
    unittest.main()

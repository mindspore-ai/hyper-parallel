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
"""Unit tests for torch fully_shard state-dict utilities."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

# pylint: disable=wrong-import-position
import torch
from torch import nn

from hyper_parallel.platform.torch.fully_shard import state_dict_utils


class FakeDTensor:
    """Small DTensor stand-in for state-dict utility tests."""

    from_local_calls = []

    def __init__(self, local):
        self._local = local
        self.device_mesh = "mesh"
        self.layout = SimpleNamespace(alias_placements=("shard",))

    def full_tensor(self):
        return self._local + 10

    def to_local(self):
        return self._local

    @classmethod
    def from_local(cls, local, mesh, placements):
        cls.from_local_calls.append((local, mesh, placements))
        return cls(local)


class TestTorchStateDictUtils(unittest.TestCase):
    """Cover full/offloaded state dict behavior with CPU fakes."""

    def setUp(self):
        FakeDTensor.from_local_calls.clear()

    @patch("hyper_parallel.platform.torch.fully_shard.state_dict_utils.DTensor", FakeDTensor)
    @patch("hyper_parallel.platform.torch.fully_shard.state_dict_utils.dist")
    def test_gather_full_state_dict_every_rank_and_rank0_cpu_offload(self, mock_dist):
        """Full state dict gathering should materialize DTensors and support rank0 offload."""
        mock_dist.is_initialized.return_value = False
        state = {"dt": FakeDTensor(torch.tensor([1.0])), "plain": torch.tensor([2.0])}

        gathered = state_dict_utils._gather_full_state_dict(state, cpu_offload=False)

        torch.testing.assert_close(gathered["dt"], torch.tensor([11.0]))
        torch.testing.assert_close(gathered["plain"], torch.tensor([2.0]))

        mock_dist.is_initialized.return_value = True
        mock_dist.get_rank.return_value = 0
        gathered = state_dict_utils._gather_full_state_dict(state, cpu_offload=True)
        self.assertEqual(gathered["dt"].device.type, "cpu")

    @patch("hyper_parallel.platform.torch.fully_shard.state_dict_utils.DTensor", FakeDTensor)
    @patch("hyper_parallel.platform.torch.fully_shard.state_dict_utils.dist")
    def test_gather_full_state_dict_cpu_offload_non_rank0_returns_empty(self, mock_dist):
        """Rank0 CPU offload should return an empty state dict on non-zero ranks."""
        mock_dist.is_initialized.return_value = True
        mock_dist.get_rank.return_value = 1

        gathered = state_dict_utils._gather_full_state_dict({"w": torch.tensor([1.0])}, cpu_offload=True)

        self.assertEqual(gathered, {})

    @patch("hyper_parallel.platform.torch.fully_shard.state_dict_utils.DTensor", FakeDTensor)
    def test_offload_sharded_state_dict_moves_plain_and_dtensor_local_shards(self):
        """Sharded state dict offload should move plain and DTensor local shards to CPU."""
        state = {"dt": FakeDTensor(torch.tensor([1.0])), "plain": torch.tensor([2.0])}

        offloaded = state_dict_utils._offload_sharded_state_dict(state)

        self.assertIsInstance(offloaded["dt"], FakeDTensor)
        self.assertEqual(FakeDTensor.from_local_calls[0][1:], ("mesh", ("shard",)))
        self.assertEqual(offloaded["plain"].device.type, "cpu")

    def test_get_model_state_dict_validates_broadcast_and_ignores_frozen_params(self):
        """State dict options should validate broadcast and skip frozen params when requested."""
        model = nn.Linear(2, 1)
        model.bias.requires_grad_(False)

        with self.assertRaisesRegex(ValueError, "full_state_dict must be True"):
            state_dict_utils.get_model_state_dict(
                model,
                options=SimpleNamespace(
                    broadcast_from_rank0=True,
                    full_state_dict=False,
                    cpu_offload=False,
                    ignore_frozen_params=False,
                ),
            )

        result = state_dict_utils.get_model_state_dict(
            model,
            options=SimpleNamespace(
                broadcast_from_rank0=False,
                full_state_dict=False,
                cpu_offload=False,
                ignore_frozen_params=True,
            ),
        )

        self.assertIn("weight", result)
        self.assertNotIn("bias", result)

    @patch("hyper_parallel.platform.torch.fully_shard.state_dict_utils._gather_full_state_dict")
    @patch("hyper_parallel.platform.torch.fully_shard.state_dict_utils._offload_sharded_state_dict")
    def test_get_model_state_dict_dispatches_by_options(self, mock_offload, mock_gather):
        """Model state dict retrieval should dispatch to full or sharded offload helpers."""
        model = MagicMock()
        model.state_dict.return_value = {"w": torch.tensor([1.0])}
        model.named_parameters.return_value = []
        mock_gather.return_value = {"full": torch.tensor([1.0])}
        mock_offload.return_value = {"offload": torch.tensor([1.0])}

        full = state_dict_utils.get_model_state_dict(
            model,
            options=SimpleNamespace(
                broadcast_from_rank0=False,
                full_state_dict=True,
                cpu_offload=True,
                ignore_frozen_params=False,
            ),
        )
        offloaded = state_dict_utils.get_model_state_dict(
            model,
            options=SimpleNamespace(
                broadcast_from_rank0=False,
                full_state_dict=False,
                cpu_offload=True,
                ignore_frozen_params=False,
            ),
        )

        self.assertEqual(full, mock_gather.return_value)
        self.assertEqual(offloaded, mock_offload.return_value)


if __name__ == "__main__":
    unittest.main()

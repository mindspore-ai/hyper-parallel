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

        # broadcast_from_rank0=True with full_state_dict=True -> NotImplementedError
        # (cross-rank broadcast not implemented yet)
        with self.assertRaisesRegex(NotImplementedError, "broadcast_from_rank0=True is not supported"):
            state_dict_utils.get_model_state_dict(
                model,
                options=SimpleNamespace(
                    broadcast_from_rank0=True,
                    full_state_dict=True,
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

    @patch("hyper_parallel.platform.torch.fully_shard.state_dict_utils.distribute_tensor")
    @patch("hyper_parallel.platform.torch.fully_shard.state_dict_utils.DTensor", FakeDTensor)
    def test_scatter_model_state_dict_distributes_plain_tensor_to_dtensor_shard(self, mock_distribute):
        """Scatter should slice plain tensors into DTensor shards and pass DTensors through."""
        target_dt = FakeDTensor(torch.tensor([1.0]))
        target_plain = torch.tensor([2.0])
        model = MagicMock()
        model.state_dict.return_value = {"dt": target_dt, "plain": target_plain}
        scattered_dt = FakeDTensor(torch.tensor([3.0]))
        mock_distribute.return_value = scattered_dt

        result = state_dict_utils._scatter_model_state_dict(
            model,
            {"dt": torch.tensor([9.0]), "plain": torch.tensor([2.0]), "missing": torch.tensor([0.0])},
            cpu_offload=False,
            strict=False,
        )

        # plain -> DTensor branch: distribute_tensor was called, result kept.
        mock_distribute.assert_called_once()
        self.assertIs(result["dt"], scattered_dt)
        # DTensor target with DTensor input: passed through as-is.
        # plain target: kept as-is.
        self.assertTrue(torch.equal(result["plain"], torch.tensor([2.0])))
        # missing key (target is None): skipped.
        self.assertNotIn("missing", result)

    @patch("hyper_parallel.platform.torch.fully_shard.state_dict_utils.distribute_tensor")
    @patch("hyper_parallel.platform.torch.fully_shard.state_dict_utils.DTensor", FakeDTensor)
    def test_scatter_model_state_dict_strict_raises_on_unexpected_keys(self, mock_distribute):
        """Scatter with strict=True must raise on keys absent from the model."""
        target_dt = FakeDTensor(torch.tensor([1.0]))
        model = MagicMock()
        model.state_dict.return_value = {"dt": target_dt}
        mock_distribute.return_value = FakeDTensor(torch.tensor([3.0]))

        with self.assertRaisesRegex(ValueError, "Unexpected key"):
            state_dict_utils._scatter_model_state_dict(
                model,
                {"dt": torch.tensor([9.0]), "bogus": torch.tensor([0.0])},
                cpu_offload=False,
                strict=True,
            )

        # strict=False should silently drop the unexpected key.
        result = state_dict_utils._scatter_model_state_dict(
            model,
            {"dt": torch.tensor([9.0]), "bogus": torch.tensor([0.0])},
            cpu_offload=False,
            strict=False,
        )
        self.assertNotIn("bogus", result)

    def test_scatter_model_state_dict_moves_shard_to_target_device_when_cpu_offload(self):
        # pylint: disable=missing-public-type-hints,missing-public-docstring,unused-argument
        """cpu_offload=True should move the shard onto the target param's device, not CPU."""
        to_calls = []

        class _DeviceTrackedLocal:
            """Records .to(device) calls so we can assert the destination device."""
            def __init__(self, tensor):
                self._tensor = tensor
            def to(self, device):
                to_calls.append(device)
                return self._tensor

        target_device = torch.device("meta")

        class _BaseDTensor:
            """Common base so both target and scatter fakes pass isinstance checks."""
            @staticmethod
            def from_local(local, mesh, placements):
                return local

        class _TargetDTensor(_BaseDTensor):
            """Fake DTensor that looks like a real model parameter."""
            def __init__(self, local):
                self._local_tensor = SimpleNamespace(device=target_device)
                self.device_mesh = "mesh"
                self.layout = SimpleNamespace(alias_placements=("shard",))
            @staticmethod
            def from_local(local, mesh, placements):
                return local

        class _ScatterDTensor(_BaseDTensor):
            """Fake DTensor whose to_local() returns a device-tracked object."""
            def __init__(self, local):
                self._local = local
                self.device_mesh = "mesh"
                self.layout = SimpleNamespace(alias_placements=("shard",))
            def to_local(self):
                return _DeviceTrackedLocal(self._local)

        target_dt = _TargetDTensor(torch.tensor([0.0]))
        model = MagicMock()
        model.state_dict.return_value = {"w": target_dt}

        with patch("hyper_parallel.platform.torch.fully_shard.state_dict_utils.DTensor", _BaseDTensor), \
             patch("hyper_parallel.platform.torch.fully_shard.state_dict_utils.distribute_tensor",
                   return_value=_ScatterDTensor(torch.tensor([1.0]))):
            state_dict_utils._scatter_model_state_dict(
                model, {"w": torch.tensor([1.0])}, cpu_offload=True, strict=False,
            )

        # The shard must have been moved onto the target device, not left on CPU.
        self.assertIn(target_device, to_calls)
        self.assertFalse(any(str(d) == "cpu" for d in to_calls),
                         f"shard should not be moved to CPU, got .to() calls: {to_calls}")

    def test_set_model_state_dict_validates_broadcast_and_loads(self):
        """Setter should validate broadcast/full consistency and dispatch scatter vs passthrough."""
        model = MagicMock()
        model.state_dict.return_value = {"w": torch.tensor([1.0])}
        model.named_parameters.return_value = []

        # broadcast_from_rank0=True with full_state_dict=False -> ValueError
        with self.assertRaisesRegex(ValueError, "full_state_dict must be True"):
            state_dict_utils.set_model_state_dict(
                model, {"w": torch.tensor([1.0])},
                options=SimpleNamespace(
                    broadcast_from_rank0=True, full_state_dict=False,
                    cpu_offload=False, ignore_frozen_params=False, strict=True,
                ),
            )

        # broadcast_from_rank0=True with full_state_dict=True -> NotImplementedError
        # (cross-rank broadcast not implemented yet)
        with self.assertRaisesRegex(NotImplementedError, "broadcast_from_rank0=True is not supported"):
            state_dict_utils.set_model_state_dict(
                model, {"w": torch.tensor([1.0])},
                options=SimpleNamespace(
                    broadcast_from_rank0=True, full_state_dict=True,
                    cpu_offload=False, ignore_frozen_params=False, strict=True,
                ),
            )

        # full_state_dict=True -> scatter path
        with patch("hyper_parallel.platform.torch.fully_shard.state_dict_utils._scatter_model_state_dict",
                   return_value={"w": torch.tensor([1.0])}) as mock_scatter:
            state_dict_utils.set_model_state_dict(
                model, {"w": torch.tensor([1.0])},
                options=SimpleNamespace(
                    broadcast_from_rank0=False, full_state_dict=True,
                    cpu_offload=False, ignore_frozen_params=False, strict=True,
                ),
            )
            mock_scatter.assert_called_once_with(
                model, {"w": torch.tensor([1.0])}, False, True,
            )
            model.load_state_dict.assert_called_once_with(
                {"w": torch.tensor([1.0])}, strict=True, assign=True,
            )

        # full_state_dict=False -> passthrough (no scatter)
        model.reset_mock()
        with (patch("hyper_parallel.platform.torch.fully_shard.state_dict_utils._scatter_model_state_dict")
              as mock_scatter):
            state_dict_utils.set_model_state_dict(
                model, {"w": torch.tensor([1.0])},
                options=SimpleNamespace(
                    broadcast_from_rank0=False, full_state_dict=False,
                    cpu_offload=False, ignore_frozen_params=False, strict=False,
                ),
            )
            mock_scatter.assert_not_called()
            model.load_state_dict.assert_called_once_with(
                {"w": torch.tensor([1.0])}, strict=False, assign=True,
            )

    def test_set_model_state_dict_ignore_frozen_is_noop(self):
        """ignore_frozen_params=True must NOT filter the input on the setter path (upstream parity)."""
        model = MagicMock()
        model.state_dict.return_value = {"w": torch.tensor([1.0])}
        model.named_parameters.return_value = [("w", SimpleNamespace(requires_grad=False))]

        captured = {}

        def _capture_load(state_dict, strict, assign):
            captured["state_dict"] = state_dict
            captured["strict"] = strict
            captured["assign"] = assign

        model.load_state_dict.side_effect = _capture_load

        state_dict_utils.set_model_state_dict(
            model, {"w": torch.tensor([1.0])},
            options=SimpleNamespace(
                broadcast_from_rank0=False, full_state_dict=False,
                cpu_offload=False, ignore_frozen_params=True, strict=True,
            ),
        )

        # The frozen key 'w' must still be present (no filtering on the setter path).
        self.assertIn("w", captured["state_dict"])
        self.assertTrue(captured["assign"])


if __name__ == "__main__":
    unittest.main()

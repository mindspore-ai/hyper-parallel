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
"""Unit tests for Hugging Face checkpoint export sources."""

import unittest
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import torch

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.distributed_checkpoint import (
    BytesStorageMetadata,
    Metadata,
    StandardSavePlanner,
    TensorProperties,
    TensorStorageMetadata,
    save,
)
from hyper_parallel.models._transformers.checkpoint_loader import CheckpointManager


class TestCheckpointManagerSave(unittest.TestCase):
    """Verify DCP-backed export does not read live training parameters."""

    def test_live_export_disables_autograd_for_parameter_gather(self) -> None:
        """Feature: Live HF export without autograd.

        Description: Export a live DTensor while the caller enables gradients.
        Expectation: Gathering disables gradients and restores caller grad mode.
        """
        parameter = torch.ones(3, requires_grad=True)
        distributed = MagicMock(spec=DTensor)

        def _full_tensor():
            """Verify the export boundary before emulating collective output."""
            self.assertFalse(torch.is_grad_enabled())
            return parameter.clone()

        distributed.full_tensor.side_effect = _full_tensor
        model = MagicMock()
        model.state_dict.return_value = {"weight": distributed}
        with torch.enable_grad():
            state = CheckpointManager(model)._gather_full_state_dict(keep_state_dict=True)
            self.assertTrue(torch.is_grad_enabled())
        self.assertFalse(state["weight"].requires_grad)
        self.assertIsNone(state["weight"].grad_fn)
        model.state_dict.assert_called_once_with(keep_vars=True)

    @staticmethod
    def _expert_export_model(hidden_size: int = 4, intermediate_size: int = 2) -> torch.nn.Module:
        """Create an HF-shaped meta model without Transformers or device allocation."""
        model = torch.nn.Module()
        model.experts = torch.nn.Module()
        model.experts.gate_up_proj = torch.nn.Parameter(
            torch.empty(2, 2 * intermediate_size, hidden_size, device="meta")
        )
        model.experts.down_proj = torch.nn.Parameter(
            torch.empty(2, hidden_size, intermediate_size, device="meta")
        )
        return model

    def test_dcp_grouped_experts_restore_values_including_square_projection(self) -> None:
        """One non-square projection identifies the layout of the square projection too."""
        model = self._expert_export_model()
        original = {
            name: torch.arange(tensor.numel()).reshape(tensor.shape)
            for name, tensor in model.state_dict().items()
        }
        training = {name: tensor.transpose(-2, -1).contiguous() for name, tensor in original.items()}
        manager = CheckpointManager(model)

        restored = manager._normalize_dcp_export_layout(training, "auto")  # pylint: disable=W0212

        for name, tensor in original.items():
            torch.testing.assert_close(restored[name], tensor)
            torch.testing.assert_close(training[name], tensor.transpose(-2, -1))

    def test_dcp_hf_layout_is_preserved(self) -> None:
        """Native HF experts must not be transposed a second time."""
        model = self._expert_export_model()
        original = {name: torch.randn(tensor.shape) for name, tensor in model.state_dict().items()}

        restored = CheckpointManager(model)._normalize_dcp_export_layout(  # pylint: disable=W0212
            original, "auto"
        )

        for name, tensor in original.items():
            self.assertIs(restored[name], tensor)

    def test_dcp_wrong_model_config_is_rejected(self) -> None:
        """An expert-count mismatch cannot be repaired by transposition."""
        model = self._expert_export_model()
        wrong = {name: torch.empty(3, *tensor.shape[1:]) for name, tensor in model.state_dict().items()}

        with self.assertRaisesRegex(ValueError, "expert shapes"):
            CheckpointManager(model)._normalize_dcp_export_layout(wrong, "auto")  # pylint: disable=W0212

    def test_dcp_dense_shape_mismatch_is_rejected(self) -> None:
        """Configuration validation also covers non-expert parameters."""
        model = torch.nn.Linear(4, 2, bias=False, device="meta")

        with self.assertRaisesRegex(ValueError, "weight.*HF expects"):
            CheckpointManager(model)._normalize_dcp_export_layout(  # pylint: disable=W0212
                {"weight": torch.empty(4, 2)}, "auto"
            )

    @patch.object(CheckpointManager, "_is_main_process", return_value=True)
    def test_offline_save_passes_restored_experts_to_hf(self, _is_main: MagicMock) -> None:
        """Read a real DCP snapshot and pass restored expert values to HF saving."""
        model = self._expert_export_model()
        model.save_pretrained = MagicMock()
        original = {name: torch.randn(tensor.shape) for name, tensor in model.state_dict().items()}
        training = {name: tensor.transpose(-2, -1).contiguous() for name, tensor in original.items()}

        with (
            TemporaryDirectory() as checkpoint_dir,
            patch("hyper_parallel.core.distributed_checkpoint.api.dist.get_rank", return_value=0),
            patch("hyper_parallel.core.distributed_checkpoint.api.dist.get_world_size", return_value=1),
            patch("hyper_parallel.core.distributed_checkpoint.api.dist.barrier"),
        ):
            save(
                {"model": training},
                checkpoint_id=checkpoint_dir,
                planner=StandardSavePlanner(enable_plan_caching=False),
            )
            CheckpointManager(model).save_pretrained(
                "/tmp/hf", dcp_checkpoint=checkpoint_dir, dcp_expert_layout="auto"
            )

        saved = model.save_pretrained.call_args.kwargs["state_dict"]
        for name, tensor in original.items():
            torch.testing.assert_close(saved[name], tensor)

    def test_ambiguous_experts_require_explicit_layout(self) -> None:
        """Square weights require the caller to specify their orientation."""
        model = torch.nn.Module()
        model.experts = torch.nn.Module()
        model.experts.down_proj = torch.nn.Parameter(torch.empty(2, 3, 3, device="meta"))
        weight = torch.arange(18).reshape(2, 3, 3)
        manager = CheckpointManager(model)
        state_dict = {"experts.down_proj": weight}

        with self.assertRaisesRegex(ValueError, "Ambiguous expert layout"):
            manager._normalize_dcp_export_layout(state_dict, "auto")  # pylint: disable=W0212
        restored = manager._normalize_dcp_export_layout(state_dict, "grouped_gemm")  # pylint: disable=W0212
        torch.testing.assert_close(restored["experts.down_proj"], weight.transpose(-2, -1))
        restored = manager._normalize_dcp_export_layout(state_dict, "hf")  # pylint: disable=W0212
        self.assertIs(restored["experts.down_proj"], weight)

    @patch.object(CheckpointManager, "_is_main_process", return_value=True)
    @patch.object(CheckpointManager, "_load_model_state_dict_from_dcp")
    def test_save_pretrained_uses_dcp_on_main_rank(
        self,
        load_dcp_state: MagicMock,
        _is_main_process: MagicMock,
    ) -> None:
        """Rank 0 should export the persisted snapshot instead of the live model."""
        state_dict = {"model.weight": torch.ones(2)}
        load_dcp_state.return_value = state_dict
        model = MagicMock()
        model._hp_used_base_weight_conversions = None
        model._hp_used_replacement_weight_conversions = None
        manager = CheckpointManager(model)

        wrote_checkpoint = manager.save_pretrained(
            "/tmp/hf",
            dcp_checkpoint="/tmp/dcp",
        )

        self.assertTrue(wrote_checkpoint)
        load_dcp_state.assert_called_once_with("/tmp/dcp")
        model.state_dict.assert_not_called()
        model.save_pretrained.assert_called_once_with(
            "/tmp/hf",
            state_dict=state_dict,
            is_main_process=True,
            max_shard_size="5GB",
            save_original_format=True,
        )

    @patch.object(CheckpointManager, "_is_main_process", return_value=False)
    def test_non_main_rank_does_not_read_dcp(
        self,
        _is_main_process: MagicMock,
    ) -> None:
        """Non-writing ranks should wait in the callback without loading shards."""
        model = MagicMock()
        manager = CheckpointManager(model)
        manager._load_model_state_dict_from_dcp = MagicMock()  # pylint: disable=W0212

        wrote_checkpoint = manager.save_pretrained(
            "/tmp/hf",
            dcp_checkpoint="/tmp/dcp",
        )

        self.assertFalse(wrote_checkpoint)
        manager._load_model_state_dict_from_dcp.assert_not_called()  # pylint: disable=W0212
        model.save_pretrained.assert_not_called()

    @patch("hyper_parallel.models._transformers.checkpoint_loader.load_dcp_state")
    @patch("hyper_parallel.models._transformers.checkpoint_loader.FileSystemReader")
    def test_dcp_export_loads_only_model_tensors(
        self,
        file_system_reader: MagicMock,
        load_dcp_state: MagicMock,
    ) -> None:
        """Optimizer and extra state must not be materialized during HF export."""
        tensor_metadata = TensorStorageMetadata(
            properties=TensorProperties(dtype="torch.float32"),
            size=(2,),
        )
        file_system_reader.return_value.load_metadata.return_value = Metadata(
            state_dict_metadata={
                "model.model.weight": tensor_metadata,
                "optimizer.state.weight.exp_avg": tensor_metadata,
                "extra_state": BytesStorageMetadata(),
            },
            planner_data={
                "model.model.weight": ("model", "model.weight"),
                "optimizer.state.weight.exp_avg": (
                    "optimizer",
                    "state",
                    "weight",
                    "exp_avg",
                ),
                "extra_state": ("extra_state",),
            },
        )
        manager = CheckpointManager(MagicMock())

        state_dict = manager._load_model_state_dict_from_dcp("/tmp/dcp")  # pylint: disable=W0212

        self.assertEqual(tuple(state_dict), ("model.weight",))
        loaded_state = load_dcp_state.call_args.args[0]
        self.assertEqual(tuple(loaded_state), ("model.model.weight",))
        load_dcp_state.assert_called_once_with(
            loaded_state,
            checkpoint_id="/tmp/dcp",
            no_dist=True,
            broadcast_replicated_tensors=False,
        )


if __name__ == "__main__":
    unittest.main()

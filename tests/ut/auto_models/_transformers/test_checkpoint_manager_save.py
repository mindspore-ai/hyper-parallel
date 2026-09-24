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
from unittest.mock import MagicMock, patch

import torch

from hyper_parallel.core.distributed_checkpoint import (
    BytesStorageMetadata,
    Metadata,
    TensorProperties,
    TensorStorageMetadata,
)
from hyper_parallel.models._transformers.checkpoint_loader import CheckpointManager


class TestCheckpointManagerSave(unittest.TestCase):
    """Verify DCP-backed export does not read live training parameters."""

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

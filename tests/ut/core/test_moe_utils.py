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
"""Unit tests for moe_utils: sync_and_update_expert_bias and MoEMonitorCallback.

Test IDs:
  Sync-01..08: sync_and_update_expert_bias (existing)
  LB-M01: MoEMonitorCallback iterates MoE layers and calls sync
  LB-M02: MoEMonitorCallback skips layers with enable_expert_bias=False
  LB-M04: MoEMonitorCallback.register attaches optimizer post-hook
  LB-M05: MoEMonitorCallback stores mean aux_loss in last_mean_aux_loss
  LB-M06: MoEMonitorCallback handles multi-layer with different num_experts
"""
import os
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

import torch  # pylint: disable=C0413

from hyper_parallel.core.moe_utils import MoEMonitorCallback, sync_and_update_expert_bias  # pylint: disable=C0413
from hyper_parallel.core.fully_shard.hsdp_utils import GroupInfo  # pylint: disable=C0413
from hyper_parallel.platform.torch.common.moe import MoE  # pylint: disable=C0413


class TestSyncAndUpdateExpertBias(unittest.TestCase):
    """Test sync_and_update_expert_bias function."""

    def _make_moe_mock(self, tokens: list) -> MagicMock:
        """Create a mock MoE module with given tokens_per_expert values."""
        moe = MagicMock()
        moe.tokens_per_expert = torch.tensor(tokens, dtype=torch.float32)
        moe.update_expert_bias = MagicMock()
        return moe

    def test_no_sync_single_device(self):
        """No group provided: should call moe.update_expert_bias directly."""
        moe = self._make_moe_mock([10.0, 20.0, 30.0, 40.0])

        sync_and_update_expert_bias(moe, lr=1e-3)

        moe.update_expert_bias.assert_called_once_with(lr=1e-3, num_recomputations=1)

    def test_dp_group_sync_with_group_info(self):
        """DP group provided as GroupInfo: should sync then update."""
        moe = self._make_moe_mock([10.0, 20.0, 30.0, 40.0])
        dp_group = GroupInfo("dp_group", MagicMock(), 2)

        with patch("hyper_parallel.core.moe_utils.platform") as mock_platform:
            mock_all_reduce = MagicMock()
            mock_platform.all_reduce = mock_all_reduce

            sync_and_update_expert_bias(moe, lr=1e-3, dp_group=dp_group)

            mock_all_reduce.assert_called_once_with(moe.tokens_per_expert, dp_group)
            moe.update_expert_bias.assert_called_once_with(lr=1e-3, num_recomputations=1)

    def test_dp_group_sync_with_process_group(self):
        """DP group provided as ProcessGroup: should wrap then sync."""
        moe = self._make_moe_mock([10.0, 20.0, 30.0, 40.0])
        mock_pg = MagicMock()
        mock_pg.group = None

        with patch("hyper_parallel.core.moe_utils.platform") as mock_platform:
            mock_all_reduce = MagicMock()
            mock_platform.all_reduce = mock_all_reduce

            sync_and_update_expert_bias(moe, lr=1e-3, dp_group=mock_pg)

            args, _ = mock_all_reduce.call_args
            self.assertIs(args[0], moe.tokens_per_expert)
            self.assertTrue(hasattr(args[1], "group"))
            moe.update_expert_bias.assert_called_once_with(lr=1e-3, num_recomputations=1)

    def test_tp_cp_dp_sync_all_groups(self):
        """All three groups provided: should sync in order TP, CP, DP."""
        moe = self._make_moe_mock([10.0, 20.0, 30.0, 40.0])
        tp_group = GroupInfo("tp_group", MagicMock(), 2)
        cp_group = GroupInfo("cp_group", MagicMock(), 2)
        dp_group = GroupInfo("dp_group", MagicMock(), 2)

        with patch("hyper_parallel.core.moe_utils.platform") as mock_platform:
            mock_all_reduce = MagicMock()
            mock_platform.all_reduce = mock_all_reduce

            sync_and_update_expert_bias(
                moe, lr=1e-3, tp_group=tp_group, cp_group=cp_group, dp_group=dp_group,
            )

            self.assertEqual(mock_all_reduce.call_count, 3)
            call_args_list = [call[0] for call in mock_all_reduce.call_args_list]
            self.assertEqual(call_args_list[0][1], tp_group)
            self.assertEqual(call_args_list[1][1], cp_group)
            self.assertEqual(call_args_list[2][1], dp_group)
            moe.update_expert_bias.assert_called_once_with(lr=1e-3, num_recomputations=1)

    def test_num_recomputations_passed(self):
        """num_recomputations should be forwarded to moe.update_expert_bias."""
        moe = self._make_moe_mock([10.0, 20.0, 30.0, 40.0])

        sync_and_update_expert_bias(moe, lr=1e-3, num_recomputations=2)

        moe.update_expert_bias.assert_called_once_with(lr=1e-3, num_recomputations=2)

    def test_tp_only_sync(self):
        """TP group only: should sync once then update (TP+SP scenario)."""
        moe = self._make_moe_mock([10.0, 20.0, 30.0, 40.0])
        tp_group = GroupInfo("tp_group", MagicMock(), 2)

        with patch("hyper_parallel.core.moe_utils.platform") as mock_platform:
            mock_all_reduce = MagicMock()
            mock_platform.all_reduce = mock_all_reduce

            sync_and_update_expert_bias(moe, lr=1e-3, tp_group=tp_group)

            mock_all_reduce.assert_called_once_with(moe.tokens_per_expert, tp_group)
            moe.update_expert_bias.assert_called_once_with(lr=1e-3, num_recomputations=1)

    def test_cp_only_sync(self):
        """CP group only: should sync once then update."""
        moe = self._make_moe_mock([10.0, 20.0, 30.0, 40.0])
        cp_group = GroupInfo("cp_group", MagicMock(), 2)

        with patch("hyper_parallel.core.moe_utils.platform") as mock_platform:
            mock_all_reduce = MagicMock()
            mock_platform.all_reduce = mock_all_reduce

            sync_and_update_expert_bias(moe, lr=1e-3, cp_group=cp_group)

            mock_all_reduce.assert_called_once_with(moe.tokens_per_expert, cp_group)
            moe.update_expert_bias.assert_called_once_with(lr=1e-3, num_recomputations=1)

    def test_integration_with_real_moe(self):
        """Integration test with real MoE module on CPU."""
        torch.manual_seed(42)
        moe = MoE(dim=16, hidden_dim=32, num_experts=4, top_k=2)
        x = torch.randn(2, 8, 16)

        moe(x)

        tokens_before = moe.tokens_per_expert.sum().item()
        self.assertGreater(tokens_before, 0, "Tokens should accumulate during forward")

        sync_and_update_expert_bias(moe, lr=1e-2)

        tokens_after = moe.tokens_per_expert.sum().item()
        self.assertEqual(tokens_after, 0.0, "Tokens should be reset after update")


class TestMoEMonitorCallback(unittest.TestCase):
    """Unit tests for MoEMonitorCallback (Layer 1+2 of the three-layer architecture)."""

    def _make_model_with_moe_layers(self, num_layers: int = 2,
                                    num_experts_list: list = None,
                                    enable_expert_bias_list: list = None) -> MagicMock:
        """Create a mock model with MoE sub-modules.

        Args:
            num_layers: Number of MoE layers to create.
            num_experts_list: Optional per-layer num_experts (defaults to [4, 4]).
            enable_expert_bias_list: Optional per-layer enable_expert_bias flags.

        Returns:
            Mock model whose .modules() returns the configured MoE layers.
        """
        if num_experts_list is None:
            num_experts_list = [4] * num_layers
        if enable_expert_bias_list is None:
            enable_expert_bias_list = [True] * num_layers

        moe_layers = []
        for i in range(num_layers):
            moe = MagicMock(spec=MoE)
            moe.enable_expert_bias = enable_expert_bias_list[i]
            moe.tokens_per_expert = torch.zeros(num_experts_list[i])
            moe.last_aux_loss = None
            moe.update_expert_bias = MagicMock()
            moe_layers.append(moe)

        model = MagicMock()
        model.modules.return_value = list(moe_layers)
        return model, moe_layers

    @patch("hyper_parallel.core.moe_utils.sync_and_update_expert_bias")
    def test_lbm01_iterates_moe_layers(self, mock_sync):
        """LB-M01: on_step_end iterates all MoE layers and calls sync per layer."""
        model, moe_layers = self._make_model_with_moe_layers(num_layers=3)
        callback = MoEMonitorCallback(model, lr=1e-3)

        callback.on_step_end()

        self.assertEqual(
            mock_sync.call_count, 3,
            (f"Should call sync_and_update_expert_bias once per MoE layer: "
             f"expected=3, got={mock_sync.call_count}"),
        )
        for i, moe in enumerate(moe_layers):
            mock_sync.assert_any_call(
                moe, lr=1e-3, tp_group=None, cp_group=None, dp_group=None,
                num_recomputations=1,
            )

    @patch("hyper_parallel.core.moe_utils.sync_and_update_expert_bias")
    def test_lbm02_skip_disabled_layers(self, mock_sync):
        """LB-M02: Layers with enable_expert_bias=False are skipped."""
        model, _ = self._make_model_with_moe_layers(
            num_layers=3,
            enable_expert_bias_list=[True, False, True],
        )
        callback = MoEMonitorCallback(model, lr=1e-3)

        callback.on_step_end()

        # Only 2 of 3 layers have enable_expert_bias=True
        self.assertEqual(
            mock_sync.call_count, 2,
            (f"Should only sync layers with enable_expert_bias=True: "
             f"expected=2, got={mock_sync.call_count}"),
        )

    def test_lbm04_register_optimizer_hook(self):
        """LB-M04: register() attaches optimizer post-hook, remove() detaches it."""
        model, _ = self._make_model_with_moe_layers(num_layers=1)
        callback = MoEMonitorCallback(model, lr=1e-3)

        optimizer = MagicMock()
        mock_handle = MagicMock()
        optimizer.register_step_post_hook.return_value = mock_handle

        callback.register(optimizer)

        optimizer.register_step_post_hook.assert_called_once()
        self.assertIs(
            callback._hook_handle, mock_handle,
            "Hook handle should be stored after register",
        )

        # remove() should call handle.remove()
        callback.remove()
        mock_handle.remove.assert_called_once()
        self.assertIsNone(
            callback._hook_handle,
            "Hook handle should be None after remove",
        )

    def test_lbm04b_register_raises_on_unsupported_optimizer(self):
        """LB-M04b: register() raises RuntimeError if optimizer lacks post-hook."""
        model, _ = self._make_model_with_moe_layers(num_layers=1)
        callback = MoEMonitorCallback(model, lr=1e-3)

        optimizer = MagicMock(spec=[])  # no register_step_post_hook attribute

        with self.assertRaises(RuntimeError):
            callback.register(optimizer)

    @patch("hyper_parallel.core.moe_utils.sync_and_update_expert_bias")
    def test_lbm05_stores_mean_aux_loss(self, mock_sync):  # pylint: disable=W0613
        """LB-M05: on_step_end stores mean aux_loss in last_mean_aux_loss."""
        model, moe_layers = self._make_model_with_moe_layers(num_layers=2)
        moe_layers[0].last_aux_loss = torch.tensor(0.1)
        moe_layers[1].last_aux_loss = torch.tensor(0.3)
        callback = MoEMonitorCallback(model, lr=1e-3)

        callback.on_step_end()

        # Mean of 0.1 and 0.3 ≈ 0.2 (float precision: tensor.item() may differ)
        self.assertIsNotNone(callback.last_mean_aux_loss)
        self.assertAlmostEqual(callback.last_mean_aux_loss, 0.2, places=5)

    @patch("hyper_parallel.core.moe_utils.sync_and_update_expert_bias")
    def test_lbm05b_no_aux_loss_stores_none(self, mock_sync):  # pylint: disable=W0613
        """LB-M05b: last_mean_aux_loss is None when all layers have no aux_loss."""
        model, _ = self._make_model_with_moe_layers(num_layers=2)
        callback = MoEMonitorCallback(model, lr=1e-3)

        callback.on_step_end()

        self.assertIsNone(callback.last_mean_aux_loss)

    @patch("hyper_parallel.core.moe_utils.sync_and_update_expert_bias")
    def test_lbm06_multi_layer_different_num_experts(self, mock_sync):
        """LB-M06: Multi-layer with different num_experts, each synced correctly."""
        model, moe_layers = self._make_model_with_moe_layers(
            num_layers=2,
            num_experts_list=[4, 8],
        )
        dp_group = GroupInfo("dp_group", MagicMock(), 2)
        callback = MoEMonitorCallback(model, lr=1e-3, dp_group=dp_group)

        callback.on_step_end()

        self.assertEqual(
            mock_sync.call_count, 2,
            (f"Should sync both layers: "
             f"expected=2, got={mock_sync.call_count}"),
        )
        # Verify each layer gets its own call with correct moe reference
        call_moe_args = [call[0][0] for call in mock_sync.call_args_list]
        self.assertIs(call_moe_args[0], moe_layers[0])
        self.assertIs(call_moe_args[1], moe_layers[1])

    @patch("hyper_parallel.core.moe_utils.sync_and_update_expert_bias")
    def test_lbm07_no_moe_layers_no_error(self, mock_sync):
        """LB-M07: Model with no MoE layers does not trigger sync or error."""
        model = MagicMock()
        model.modules.return_value = []
        callback = MoEMonitorCallback(model, lr=1e-3)

        callback.on_step_end()

        mock_sync.assert_not_called()


if __name__ == "__main__":
    unittest.main()

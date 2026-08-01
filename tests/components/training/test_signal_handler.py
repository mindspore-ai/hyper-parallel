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
"""Tests for distributed SIGTERM coordination."""

from unittest.mock import MagicMock

from hyper_models.components.training import signal_handler


def test_signals_received_uses_platform_npu_device_for_hccl(monkeypatch):
    """HCCL signal coordination must not create CPU collective tensors."""
    npu_device = MagicMock(name="npu_device")
    local = MagicMock()
    first_rank = MagicMock()
    second_rank = MagicMock()
    first_rank.item.return_value = 1
    second_rank.item.return_value = 0
    tensor = MagicMock(return_value=local)
    zeros = MagicMock(side_effect=(first_rank, second_rank))
    all_gather = MagicMock()

    monkeypatch.setattr(signal_handler.platform, "device", lambda: npu_device)
    monkeypatch.setattr(signal_handler.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(signal_handler.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(signal_handler.torch, "tensor", tensor)
    monkeypatch.setattr(signal_handler.torch, "zeros", zeros)
    monkeypatch.setattr(signal_handler.torch.distributed, "all_gather", all_gather)

    signals = signal_handler.DistributedSignalHandler().signals_received()

    tensor.assert_called_once_with([0], dtype=signal_handler.torch.int32, device=npu_device)
    assert zeros.call_count == 2
    assert all(call.kwargs["device"] is npu_device for call in zeros.call_args_list)
    all_gather.assert_called_once_with([first_rank, second_rank], local)
    assert signals == [True, False]

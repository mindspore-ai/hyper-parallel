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
"""Tests for accelerator-aware distributed initialization."""

from unittest.mock import MagicMock

from hyper_models.components.distributed import infrastructure


def test_initialize_distributed_keeps_backend_argument_on_cuda(monkeypatch):
    """Existing callers may still select their CUDA process-group backend."""
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "1")
    device = MagicMock()
    monkeypatch.setattr(infrastructure, "_runtime_device_type", lambda: "cuda")
    monkeypatch.setattr(infrastructure.platform, "get_device_handle", lambda _: device)
    monkeypatch.setattr(infrastructure.dist, "is_initialized", lambda: False)
    init_process_group = MagicMock()
    monkeypatch.setattr(infrastructure.dist, "init_process_group", init_process_group)

    infrastructure.initialize_distributed(backend="custom_cuda")

    device.set_device.assert_called_once_with(1)
    init_process_group.assert_called_once_with(backend="custom_cuda")


def test_initialize_distributed_selects_hccl_on_npu(monkeypatch):
    """A5 NPU training always needs HCCL, regardless of the legacy default."""
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "1")
    device = MagicMock()
    monkeypatch.setattr(infrastructure, "_runtime_device_type", lambda: "npu")
    monkeypatch.setattr(infrastructure.platform, "get_device_handle", lambda _: device)
    monkeypatch.setattr(infrastructure.dist, "is_initialized", lambda: False)
    init_process_group = MagicMock()
    monkeypatch.setattr(infrastructure.dist, "init_process_group", init_process_group)

    infrastructure.initialize_distributed(backend="gloo")

    device.set_device.assert_called_once_with(1)
    init_process_group.assert_called_once_with(backend="hccl")

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
"""Keep distributed-data logic tests independent of installed NPU hardware."""

import sys

import pytest
import torch


@pytest.fixture(autouse=True)
def cpu_data_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable NPU discovery and device hooks injected into CPU DataLoader workers."""
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "")
    npu = sys.modules.get("torch_npu")
    if npu is not None:
        monkeypatch.setattr(torch.npu, "is_available", lambda: False)
        # Older torch_npu versions run these hooks even for CPU-only workers.
        monkeypatch.setattr(torch.npu, "synchronize", lambda: None)
        monkeypatch.setattr(npu._C, "_npu_set_thread_affinity", lambda *_: None)
        monkeypatch.setattr(npu._C, "_npu_reset_thread_affinity", lambda: None)

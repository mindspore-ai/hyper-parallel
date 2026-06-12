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
"""Hardware bandwidth profiling utilities."""

import torch

from hyper_parallel.auto_parallel.hyper_offload.runtime.timer import DeviceTimer


def profile_transfer_bandwidth() -> tuple[float, float]:
    """Profile host-to-device and device-to-host transfer bandwidth.

    Returns:
        A tuple of (d2h_bandwidth_gbps, h2d_bandwidth_gbps).

    """
    if not torch.accelerator.is_available():
        raise RuntimeError(
            "Cannot profile transfer bandwidth: no accelerator available"
        )

    device = torch.accelerator.current_accelerator()
    size = 16 * 1024 * 1024
    src = torch.empty(size, dtype=torch.uint8, device=device)
    host = torch.empty(size, dtype=torch.uint8, pin_memory=True)
    timer = DeviceTimer()

    # D2H bandwidth
    timer.start()
    host.copy_(src, non_blocking=True)
    d2h_ms = timer.stop()
    d2h = size / (d2h_ms / 1000.0) / 1024**3

    # H2D bandwidth
    timer.start()
    src.copy_(host, non_blocking=True)
    h2d_ms = timer.stop()
    h2d = size / (h2d_ms / 1000.0) / 1024**3

    return max(d2h, 1.0), max(h2d, 1.0)

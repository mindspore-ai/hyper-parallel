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
"""Explicit launchers for production EP transport acceptance on Ascend."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize("backend", ["mc2", "allgather"])
def test_unified_expert_parallel(tmp_path: Path, backend: str) -> None:
    """Exercise production dispatch, grouped experts, native graph replay and gradients."""
    devices = os.environ.get("RL_EP_ST_DEVICES", "").split(",")
    if devices == [""]:
        pytest.skip("Set RL_EP_ST_DEVICES to explicitly allocated idle NPU IDs")
    if not all(device.isdigit() for device in devices) or len(set(devices)) != len(devices):
        raise ValueError("RL_EP_ST_DEVICES must contain distinct NPU IDs")
    ep_size = int(os.environ.get("RL_EP_ST_SIZE", len(devices)))
    if ep_size not in (2, 4, 8) or len(devices) % ep_size:
        raise ValueError("EP ST requires complete EP2, EP4 or EP8 groups")
    subprocess.run([
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--local-addr=127.0.0.1",
        f"--nproc_per_node={len(devices)}", str(Path(__file__).with_name("_unified_expert_parallel.py")),
        "--ep", str(ep_size), "--backend", backend, "--gradients", "--capacity-switch", "--output", str(tmp_path),
    ], env=dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=",".join(devices)), check=True, timeout=1800)

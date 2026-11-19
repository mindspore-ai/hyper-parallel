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
"""Explicit device launchers for inference PCP ownership and the RL publication loop."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from tests.common.mark_utils import arg_mark


def _launch(worker: str, arguments: list[str], *, required_ranks: int | None = None) -> None:
    devices = os.environ.get("RL_PCP_ST_DEVICES", "").split(",")
    if devices == [""]:
        pytest.skip("Set RL_PCP_ST_DEVICES to explicitly allocated idle NPU IDs")
    if not all(device.isdigit() for device in devices) or len(set(devices)) != len(devices):
        raise ValueError("RL_PCP_ST_DEVICES must contain distinct NPU IDs")
    if required_ranks is not None and len(devices) != required_ranks:
        raise ValueError(f"This PCP test requires {required_ranks} devices")
    environment = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=",".join(devices))
    if worker == "_prefill_context_parallel_attention.py":
        environment["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--local-addr=127.0.0.1",
                    f"--nproc_per_node={len(devices)}", str(Path(__file__).with_name(worker)), *arguments],
                   env=environment, check=True, timeout=1800)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_prefill_context_parallel_mesh(tmp_path: Path) -> None:
    """Exercise EP subgroups, weight ownership and publication in the full domain."""
    _launch("_prefill_context_parallel_mesh.py", ["--output", str(tmp_path)], required_ranks=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize("graph", [False, True])
def test_prefill_context_parallel_attention(tmp_path: Path, graph: bool) -> None:
    """Compare native cache gathering and paged attention with full CP1 prefixes."""
    model = os.environ.get("RL_UNIFIED_ST_MODEL")
    if not model:
        pytest.skip("Set RL_UNIFIED_ST_MODEL to a local model configuration")
    arguments = ["--model", model, "--output", str(tmp_path)]
    if graph:
        arguments.append("--graph")
    _launch("_prefill_context_parallel_attention.py", arguments)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_prefill_context_parallel_rl(tmp_path: Path) -> None:
    """Require two actual nonzero updates, publication and exact rollout probabilities."""
    config = os.environ.get("RL_PCP_ST_RL_CONFIG")
    if not config:
        pytest.skip("Set RL_PCP_ST_RL_CONFIG to an explicit inference PCP recipe")
    _launch("_context_parallel_rl.py", [config, "--output", str(tmp_path)])

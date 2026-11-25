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
"""Explicit launchers for training CP and rollout PCP acceptance."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

from tests.common.mark_utils import arg_mark


def _devices(variable="RL_CP_ST_DEVICES"):
    value = os.environ.get(variable)
    if not value:
        pytest.skip(f"Set {variable} to explicitly allocated NPU IDs")
    devices = value.split(",")
    if not all(device.isdigit() for device in devices) or len(set(devices)) != len(devices):
        raise ValueError(f"{variable} must contain distinct NPU IDs")
    return devices


def _launch(worker, devices, args, output):
    env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=",".join(devices), CP_TEST_OUTPUT=str(output))
    if worker == "_prefill_context_parallel_attention.py":
        env["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--local-addr=127.0.0.1",
                    f"--nproc_per_node={len(devices)}", str(Path(__file__).with_name(worker)), *args],
                   env=env, check=True, timeout=1800)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards', essential_mark='essential')
def test_context_parallel_attention(tmp_path):
    """Cover real K/V gather and backward, padding, packed documents and empty ranks."""
    devices = _devices()
    if len(devices) not in (1, 2, 4):
        raise ValueError('The CP attention test requires one, two or four explicitly selected devices')
    _launch('_context_parallel_attention.py', devices, [], tmp_path)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards', essential_mark='essential')
def test_context_parallel_model(tmp_path):
    """Compare a shared logical checkpoint at TP2/CP1 and TP2/CP2 with actual updates."""
    devices = _devices()
    if len(devices) != 4:
        raise ValueError('The default CP model comparison requires four explicitly selected devices')
    worker = Path(__file__).with_name('_context_parallel_model.py')
    checkpoint = tmp_path / 'checkpoint'
    subprocess.run([sys.executable, str(worker), '--create-only', '--checkpoint', str(checkpoint),
                    '--output', str(tmp_path)], check=True, timeout=120)
    baseline = tmp_path / 'cp1'
    args = ['--checkpoint', str(checkpoint), '--tp', '2', '--ep', '2', '--sp', '--packed-check',
            '--compare-gradients']
    _launch(worker.name, devices[:2], [*args, '--cp', '1', '--output', str(baseline)], baseline)
    _launch(worker.name, devices, [*args, '--cp', '2', '--reference', str(baseline),
                                  '--output', str(tmp_path / 'cp2')], tmp_path / 'cp2')


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards', essential_mark='essential')
def test_context_parallel_rl(tmp_path):
    """Run a caller-provided CP recipe with explicit nonzero fixture rewards and real sampling."""
    devices = _devices()
    config = os.environ.get('RL_CP_ST_RL_CONFIG')
    if not config:
        pytest.skip('Set RL_CP_ST_RL_CONFIG to a CP recipe with accessible model/data paths')
    _launch('_context_parallel_rl.py', devices, [config, '--output', str(tmp_path)], tmp_path)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_prefill_context_parallel_mesh(tmp_path: Path) -> None:
    """Exercise EP subgroups, weight ownership and publication in the full domain."""
    devices = _devices("RL_PCP_ST_DEVICES")
    if len(devices) != 4:
        raise ValueError("The PCP mesh test requires four devices")
    _launch("_prefill_context_parallel_mesh.py", devices, ["--output", str(tmp_path)], tmp_path)


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
    _launch("_prefill_context_parallel_attention.py", _devices("RL_PCP_ST_DEVICES"), arguments, tmp_path)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_prefill_context_parallel_rl(tmp_path: Path) -> None:
    """Require two actual nonzero updates, publication and exact rollout probabilities."""
    config = os.environ.get("RL_PCP_ST_RL_CONFIG")
    if not config:
        pytest.skip("Set RL_PCP_ST_RL_CONFIG to an explicit inference PCP recipe")
    _launch("_context_parallel_rl.py", _devices("RL_PCP_ST_DEVICES"), [config, "--output", str(tmp_path)], tmp_path)

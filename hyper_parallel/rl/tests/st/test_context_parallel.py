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
"""Explicit launchers for training CP numerical and model acceptance."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

from tests.common.mark_utils import arg_mark


def _devices():
    selected = os.environ.get('RL_CP_ST_DEVICES')
    if not selected:
        pytest.skip('Set RL_CP_ST_DEVICES to explicitly allocated idle NPU IDs')
    devices = selected.split(',')
    if not all(device.isdigit() for device in devices) or len(devices) != len(set(devices)):
        raise ValueError('RL_CP_ST_DEVICES must contain distinct NPU IDs')
    return devices


def _launch(worker, devices, args, output):
    env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=','.join(devices), CP_TEST_OUTPUT=str(output))
    subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone',
                    '--local-addr=127.0.0.1', f'--nproc_per_node={len(devices)}',
                    str(Path(__file__).with_name(worker)), *args], env=env, check=True, timeout=900)


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

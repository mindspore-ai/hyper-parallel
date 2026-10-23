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
"""Framework-free launcher using the repository's standard distributed ST helper."""
from pathlib import Path

from tests.common.distributed_launcher import torchrun_case


def _run_async_gloo(tmp_path, monkeypatch, num_proc: int) -> None:
    worker = Path(__file__).with_name("_async_train.py").resolve()
    monkeypatch.setenv("USE_TF", "0")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("HYPER_RL_ASYNC_ST_OUTPUT", str(tmp_path / "checkpoints"))
    monkeypatch.chdir(tmp_path)
    torchrun_case(str(worker), "test_async_worker", num_proc=num_proc)


def test_async_grpo_gloo(tmp_path, monkeypatch):
    """Exercise real Actor/DDP updates and spawned production protocol on two CPU ranks."""
    _run_async_gloo(tmp_path, monkeypatch, 2)


def test_async_multiturn_grpo_gloo(tmp_path, monkeypatch):
    """Exercise uneven per-call DP batches, actual gradients and checkpoint restore."""
    monkeypatch.setenv("HYPER_RL_ASYNC_ST_MULTITURN", "1")
    test_async_grpo_gloo(tmp_path, monkeypatch)


def test_async_multiturn_grpo_four_ranks(tmp_path, monkeypatch):
    """Verify variable call counts and restoration with the eight-NPU recipe's four trainer ranks."""
    monkeypatch.setenv("HYPER_RL_ASYNC_ST_MULTITURN", "1")
    _run_async_gloo(tmp_path, monkeypatch, 4)

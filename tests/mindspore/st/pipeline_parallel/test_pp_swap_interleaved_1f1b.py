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
"""msrun launchers for real-overlap activation-swap comparisons."""
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid

from tests.common.mark_utils import arg_mark
from tests.common.port_utils import allocate_port

_TEST_FILE = os.path.join(os.path.dirname(__file__), "_test_pp_swap_interleaved_1f1b.py")
_BASELINE_PREFIX = os.path.join(tempfile.gettempdir(), "pp_swap_overlap_moe_baseline")


def _cleanup_baselines(tag: str) -> None:
    for scenario in ("overlap_b_f", "overlap_b_f_dxdw"):
        for rank in range(8):
            try:
                os.remove(f"{_BASELINE_PREFIX}_{tag}_{scenario}_rank{rank}.pkl")
            except FileNotFoundError:
                pass


def _log_dir(log_label: str) -> str:
    filename = _TEST_FILE.split(".py")[0]
    return f"./logs/{filename}/{log_label}"


def _allocate_hccl_ports() -> tuple[str, str]:
    block_size = 128
    base_port = 40000 + (allocate_port() % 90) * block_size
    return str(base_port), f"{base_port}-{base_port + block_size - 1}"


def _run_msrun_subprocess(worker_case_name: str, env_updates: dict, log_label: str) -> None:
    """Run one eight-card worker phase in a fresh process tree."""
    log_dir = _log_dir(log_label)
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

    env = os.environ.copy()
    env["GLOG_v"] = "3"
    env.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    hccl_base_port, hccl_socket_range = _allocate_hccl_ports()
    env.setdefault("HCCL_IF_BASE_PORT", hccl_base_port)
    env.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", hccl_socket_range)
    env.update(env_updates)

    cmd = [
        "msrun",
        "--worker_num=8",
        "--local_worker_num=8",
        "--master_addr=127.0.0.1",
        f"--master_port={allocate_port()}",
        "--join=True",
        f"--log_dir={log_dir}",
        "--",
        os.path.join(os.path.dirname(sys.executable), "pytest"),
        "-s",
        "-v",
        f"{_TEST_FILE}::{worker_case_name}",
    ]
    with subprocess.Popen(cmd, env=env, start_new_session=True) as proc:
        try:
            ret = proc.wait(timeout=900)
        except subprocess.TimeoutExpired as exc:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
            raise AssertionError(f"{worker_case_name} timed out") from exc
    assert ret == 0, f"{worker_case_name} failed with exit code {ret}, log_dir={log_dir}"
    time.sleep(5)


def _run_scenario(scenario: str, enable_dxdw: bool) -> None:
    """Run one three-step real-overlap no-swap/swap comparison."""
    tag = uuid.uuid4().hex
    env = {
        "PP_OVERLAP_PP_SIZE": "4",
        "PP_OVERLAP_EP_SIZE": "2",
        "PP_SWAP_MOE_DXDW": "1" if enable_dxdw else "0",
        "PP_SWAP_MOE_CHECK_STEPS": "3",
        "PP_SWAP_INTERLEAVED_BASELINE_TAG": tag,
    }
    try:
        _cleanup_baselines(tag)
        _run_msrun_subprocess(
            "test_moe_overlap_b_f_no_swap",
            env,
            f"{scenario}_no_swap",
        )
        _run_msrun_subprocess(
            "test_moe_overlap_b_f_swap",
            env,
            f"{scenario}_swap",
        )
    finally:
        _cleanup_baselines(tag)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_interleaved_1f1b_pipeline_swap_overlap_b_f():
    """
    Feature: Activation swap composed with real overlap.
    Description: Run the existing PP=4 x EP=2 hook-coordinated overlap model
                 for three no-swap steps and three swap steps.
    Expectation: Every loss and local gradient matches, the backward work runs
                 on overlap threads, and swap reduces steady peak memory.
    """
    _run_scenario("real_overlap", enable_dxdw=False)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_interleaved_1f1b_pipeline_swap_overlap_b_f_dxdw():
    """
    Feature: Activation swap composed with real overlap and dx/dw split.
    Description: Run the existing PP=4 x EP=2 hook-coordinated dxdw path for
                 three no-swap steps and three swap steps.
    Expectation: Every loss and local gradient matches, dx/dw completes on the
                 overlap path, and swap reduces steady peak memory.
    """
    _run_scenario("real_overlap_dxdw", enable_dxdw=True)

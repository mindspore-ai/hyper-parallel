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
"""msrun launcher for Interleaved 1F1B activation swap memory comparison."""
import os
import shutil
import signal
import subprocess
import tempfile
import time
from typing import Optional
import uuid

from tests.common.port_utils import allocate_port
from tests.common.mark_utils import arg_mark

_TEST_FILE = os.path.join(os.path.dirname(__file__), "_test_pp_swap_interleaved_1f1b.py")
_RANK_MEM_FILE_PREFIX = os.path.join(tempfile.gettempdir(), "pp_swap_interleaved_1f1b_mem_no_swap")
_MEMORY_ACTIVATION_MB = os.environ.get("PP_SWAP_INTERLEAVED_TEST_ACTIVATION_MB", "200")
_ACCURACY_TOKENS_PER_MICRO = os.environ.get("PP_SWAP_INTERLEAVED_ACCURACY_TOKENS_PER_MICRO", "256")


def _rank_mem_file(tag: str, rank: int) -> str:
    return f"{_RANK_MEM_FILE_PREFIX}_{tag}_rank{rank}.txt"


def _cleanup_rank_mem_files(tag: str):
    for rank in range(4):
        try:
            os.remove(_rank_mem_file(tag, rank))
        except FileNotFoundError:
            pass


def _log_dir(log_label: str) -> str:
    filename = _TEST_FILE.split(".py")[0]
    return f"./logs/{filename}/{log_label}"


def _allocate_hccl_ports():
    base_port = 30000 + allocate_port()
    return str(base_port), f"{base_port}-{base_port + 127}"


def _run_msrun_subprocess(
        worker_case_name: str,
        env_updates: dict[str, Optional[str]],
        log_label: Optional[str] = None):
    """Run one distributed worker case in a fresh process tree."""
    log_label = log_label or worker_case_name
    log_dir = _log_dir(log_label)
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

    env = os.environ.copy()
    env["GLOG_v"] = "3"
    env.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3")
    hccl_base_port, hccl_socket_range = _allocate_hccl_ports()
    env.setdefault("HCCL_IF_BASE_PORT", hccl_base_port)
    env.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", hccl_socket_range)
    for key, value in env_updates.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value

    cmd = [
        "msrun",
        "--worker_num=4",
        "--local_worker_num=4",
        "--master_addr=127.0.0.1",
        f"--master_port={allocate_port()}",
        "--join=True",
        f"--log_dir={log_dir}",
        "pytest",
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


def _run_interleaved_swap_scenario(log_prefix: str, overlap_b_f: bool):
    """Run baseline, swap-memory, and accuracy passes for one scheduler mode."""
    mem_tag = uuid.uuid4().hex
    overlap_env = {"PP_SWAP_INTERLEAVED_OVERLAP_B_F": "1" if overlap_b_f else None}
    memory_env = {
        "PP_SWAP_INTERLEAVED_ACTIVATION_MB": _MEMORY_ACTIVATION_MB,
        "PP_SWAP_INTERLEAVED_MEM_TAG": mem_tag,
        **overlap_env,
    }
    accuracy_env = {
        "PP_SWAP_INTERLEAVED_ACTIVATION_MB": None,
        "PP_SWAP_ACTIVATION_MB": None,
        "PP_SWAP_TOKENS_PER_MICRO": _ACCURACY_TOKENS_PER_MICRO,
        "PP_SWAP_INTERLEAVED_MEM_TAG": f"{mem_tag}_accuracy",
        **overlap_env,
    }

    try:
        _cleanup_rank_mem_files(mem_tag)
        _run_msrun_subprocess(
            "test_interleaved_1f1b_no_swap",
            memory_env,
            log_label=f"{log_prefix}_no_swap",
        )
        _run_msrun_subprocess(
            "test_interleaved_1f1b_swap_memory",
            memory_env,
            log_label=f"{log_prefix}_swap_memory",
        )
        _run_msrun_subprocess(
            "test_interleaved_1f1b_swap_accuracy",
            accuracy_env,
            log_label=f"{log_prefix}_swap_accuracy",
        )
    finally:
        _cleanup_rank_mem_files(mem_tag)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_interleaved_1f1b_pipeline_swap_memory():
    """
    Feature: Interleaved 1F1B activation swap device memory comparison.
    Description: Run a four-rank Interleaved 1F1B schedule with two virtual
        stages per rank, 8 micro-batches, 2048 hidden size, and 4-layer
        DeepStage per virtual stage.  First record no-swap peak memory, then
        run all virtual stages with activation swap in a separate process and
        compare each rank. Run a small activation accuracy pass in another
        process so the memory case is not polluted by the serial reference.
    Expectation: swap device peak memory is less than no-swap on every rank,
        and swap outputs/gradients match the serial reference.
    """
    _run_interleaved_swap_scenario("test_interleaved_1f1b", overlap_b_f=False)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_interleaved_1f1b_overlap_b_f_pipeline_swap_memory():
    """
    Feature: Interleaved 1F1B activation swap with OVERLAP_B_F composite steps.
    Description: Run the same memory and accuracy workflow as the plain
        interleaved swap case, but construct ScheduleInterleaved1F1B with
        overlap_b_f=True so real FWD/BWD leaves live inside OVERLAP_B_F.
    Expectation: swap device peak memory is less than no-swap on every rank,
        and swap outputs/gradients match the serial reference.
    """
    _run_interleaved_swap_scenario("test_interleaved_1f1b_overlap_b_f", overlap_b_f=True)

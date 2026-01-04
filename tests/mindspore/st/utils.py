# Copyright 2025 Huawei Technologies Co., Ltd
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
"""test utils"""
import os
import shutil
import pytest
import mindspore as ms
from packaging import version
from hyper_parallel.platform.mindspore.hsdp.scheduler import MindSporeHSDPScheduler


def msrun_case(glog_v, file_name, case_name, master_port, worker_num=8, local_worker_num=8):
    """Run test case."""
    filename = file_name.split(".py")[0]
    log_path = f"./log_{filename}/{case_name}"
    if os.path.exists(log_path):
        shutil.rmtree(log_path)
    cmd = f"export GLOG_v={glog_v} && msrun --worker_num={worker_num} --local_worker_num={local_worker_num} " \
          f"--master_addr=127.0.0.1 --master_port={master_port} " \
          f"--join=True --log_dir={log_path} pytest -s -v " \
          f"{file_name}::{case_name}"
    ret = os.system(cmd)
    assert ret == 0


def skip_if_ms_version_lt(min_ms_version):
    """
    Skip if mindspore version less then `min_ms_version`.
    """
    return pytest.mark.skipif(
        version.parse(ms.__version__) < version.parse(min_ms_version),
        reason=f"Requires MindSpore >= {min_ms_version}, but got {ms.__version__}"
    )


def skip_if_ms_plugin_not_exist():
    """
    Skip if mindspore plugin does not exist.
    """
    return pytest.mark.skipif(
        not MindSporeHSDPScheduler.get_pass_library_pass(),
        reason="Mindspore plugin does not exist."
    )

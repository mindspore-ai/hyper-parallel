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
"""parallel run case"""
import os
import signal
import multiprocessing as mp
from typing import Union


class TorchCase:
    """torch case messages"""

    def __init__(self, file_name: str, case_name: str, master_port: int, num_proc: int = 1):
        self.file_name = file_name
        self.case_name = case_name
        self.master_port = master_port
        self.num_proc = num_proc


class MindSporeCase:
    """mindspore case messages"""

    def __init__(self, file_name: str, case_name: str, master_port: int, worker_num: int = 1, local_worker_num: int = 1,
                 glog_v: int = 3):
        self.glog_v = glog_v
        self.file_name = file_name
        self.case_name = case_name
        self.master_port = master_port
        self.num_proc = worker_num
        self.local_worker_num = local_worker_num


def run_case(visible_devices, case: Union[TorchCase, MindSporeCase]):
    """
    run case in child process
    """
    # become the leader of a new process group so that os.killpg on timeout
    # kills torchrun/msrun worker sub-processes as well as this wrapper
    os.setsid()
    # set visible devices for current case
    os.environ['ASCEND_RT_VISIBLE_DEVICES'] = ','.join(map(str, visible_devices))
    if isinstance(case, TorchCase):
        # pylint: disable=C0415
        from tests.torch.utils import torchrun_case
        torchrun_case(case.file_name, case.case_name, case.master_port, case.num_proc)
    elif isinstance(case, MindSporeCase):
        # pylint: disable=C0415
        from tests.mindspore.st.utils import msrun_case
        msrun_case(case.glog_v, case.file_name, case.case_name, case.master_port, case.num_proc, case.local_worker_num)


def parallel_run(cases: Union[list[TorchCase], list[MindSporeCase]], global_num_proc: int = 8):
    """
    parallel run cases

    Args:
        cases (list[Case]): list of case messages to be run parallel
        global_num_proc (int, optional): number of total num of process. Defaults to 8.
    """
    # assign devices
    sum_num_proc = 0
    assignments = []

    for case in cases:
        num = case.num_proc
        devices = list(range(sum_num_proc, sum_num_proc + num))
        assignments.append(devices)
        sum_num_proc += num
    # assert sum num_proc
    assert sum_num_proc <= global_num_proc, (f"sum num_proc {sum_num_proc} greater than "
                                             f"global_num_proc {global_num_proc}")

    # create child process (run_case calls os.setsid to own a process group,
    # so os.killpg on timeout kills torchrun/msrun workers too)
    processes = []
    for _, (case, devices) in enumerate(zip(cases, assignments)):
        p = mp.Process(target=run_case, args=(devices, case))
        p.start()
        processes.append(p)

    # wait child process terminates (timeout=900s to prevent infinite hang on distributed deadlock)
    timed_out = []
    for i, p in enumerate(processes):
        p.join(timeout=900)
        if p.is_alive():
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.join()
            timed_out.append(cases[i].case_name)

    # check results for all cases
    if timed_out:
        raise AssertionError(f"Cases timed out (possible collective deadlock): {timed_out}")
    failed = [cases[i].case_name for i, p in enumerate(processes) if p.exitcode != 0]
    assert not failed, f"List cases failed: {failed}"

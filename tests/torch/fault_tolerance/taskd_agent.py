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
"""Exercise TaskD Agent command dispatch and actual patched Torch process control.

The harness delivers the restart message directly to the Agent queue. Native
Manager-to-Agent networking and ClusterD decisions are outside this test.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

from torch.distributed.elastic.agent.server.api import Worker, WorkerSpec
from torch.distributed.elastic.agent.server.local_elastic_agent import LocalElasticAgent
from torch.distributed.elastic.multiprocessing.api import DefaultLogsSpecs, Std
from torch.distributed.elastic.rendezvous.static_tcp_rendezvous import StaticTCPRendezvous

from taskd.api import taskd_agent_api
from taskd.python.adaptor.patch.torch_patch import patch_torch_method
from taskd.python.framework.agent.pt_agent.pt_agent import get_fault_workers
from taskd.python.toolkit.constants import constants


class TaskdProcessSupervisor:
    """Own real Torch process contexts under TaskD's restart callback."""

    def __init__(self, directory: Path, world: int, store: Any, entrypoint: str,
                 controller_port: int, store_port: int, fault: bool) -> None:
        """Construct a single-node WorkerGroup without a second rendezvous."""
        self.directory = directory
        os.environ.update(RANK="0", MASTER_ADDR="127.0.0.1", MASTER_PORT=str(store_port))
        patch_torch_method()
        rendezvous = StaticTCPRendezvous("127.0.0.1", store_port, 0, 1, "hp-taskd-validation", 120)
        args = ["-u", entrypoint, "--worker", "--rank", "${local_rank}", "--world", str(world),
                "--directory", str(directory), "--controller-port", str(controller_port),
                "--store-port", str(store_port)]
        if fault:
            args.append("--fault")
        spec = WorkerSpec(role="trainer", local_world_size=world, entrypoint=sys.executable,
                          args=tuple(args), rdzv_handler=rendezvous)
        self.local = LocalElasticAgent(spec, DefaultLogsSpecs(log_dir=str(directory / "agent_logs"),
                                                            redirects=Std.ALL), exit_barrier_timeout=5)
        self.local._pcontext_dict = {}  # pylint: disable=protected-access
        group = self.local._worker_group  # pylint: disable=protected-access
        group.store = store
        group.group_rank, group.group_world_size = 0, 1
        group.workers = [Worker(rank, rank, rank, world, world) for rank in range(world)]
        if not taskd_agent_api.init_taskd_agent({"Framework": "PyTorch"}, self.local):
            raise RuntimeError("TaskD Agent initialization failed")
        taskd_agent_api.register_func("RESTART", self.local._restart_workers)  # pylint: disable=protected-access
        self.agent = taskd_agent_api.taskd_agent

    def launch(self, rank: int, generation: int) -> subprocess.Popen:
        """Start initially, or dispatch an actual TaskD in-place restart message."""
        group = self.local._worker_group  # pylint: disable=protected-access
        if generation:
            old_pid = group.workers[rank].id
            self.agent.msg_queue.put(SimpleNamespace(code=constants.RESTARTWORKERCODE,
                                                      message=json.dumps([rank])))
            self.agent.handle_message()
            with (self.directory / "agent_events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"rank": rank, "old_pid": old_pid, "new_pid": group.workers[rank].id,
                                         "action": "taskd_recover_in_place"}) + "\n")
        else:
            workers = get_fault_workers(group, [rank])
            worker_ids = self.local._start_workers(workers)  # pylint: disable=protected-access
            group.workers[rank].id = worker_ids[rank]
        context = self.local._pcontext_dict[rank]  # pylint: disable=protected-access
        return context.subprocess_handlers[0].proc

    def close(self) -> None:
        """Close only process contexts owned by this isolated test Agent."""
        for context in self.local._pcontext_dict.values():  # pylint: disable=protected-access
            context.close()

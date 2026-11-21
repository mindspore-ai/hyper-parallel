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
"""Real TaskD Controller adapter with a test-only upstream command driver.

The observer forwards each report to the real Go backend before recording it.
The driver supplies ClusterD-side decisions; it does not validate Agent restart.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import threading

import grpc
from taskd.api.taskd_manager_api import init_taskd_manager, start_taskd_manager
from taskd.python.framework.manager import controller


def run(directory: Path, world: int) -> None:
    """Start the real Manager and route driver commands through its adapter."""
    send = controller.controller_send_to_backend
    lock = threading.Lock()
    rpc_counts = {}
    rpc_methods = {"process_fault": "ReportProcessFault", "stop_complete": "ReportStopComplete",
                   "recover_strategy": "ReportRecoverStrategy", "recover_status": "ReportRecoverStatus"}
    server = grpc.server(ThreadPoolExecutor(max_workers=4))

    def _handler(method):
        def _report(payload, context):
            del context
            if b"hp-taskd-validation" not in payload:
                raise ValueError("report belongs to another validation job")
            with lock:
                rpc_counts[method] = rpc_counts.get(method, 0) + 1
                with (directory / "upstream_rpc.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"method": method, "bytes": len(payload)}) + "\n")
            # The source Recover proto defines Status(code=0, info="") as
            # the empty protobuf payload. This peer replaces ClusterD only.
            return b""
        return grpc.unary_unary_rpc_method_handler(_report)

    server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler(
        "Recover", {method: _handler(method) for method in rpc_methods.values()}),))
    if server.add_insecure_port("127.0.0.1:8899") != 8899:
        raise RuntimeError("test upstream port 8899 is unavailable")
    server.start()

    def _observe(message):
        method = rpc_methods.get(message.action)
        before = rpc_counts.get(method, 0)
        send(message)
        if method and rpc_counts.get(method, 0) <= before:
            raise RuntimeError(f"TaskD Go backend did not deliver {method} to the upstream RPC peer")
        reports = {
            "process_fault": ("report_fault_ranks", [message.fault_ranks]),
            "stop_complete": ("report_stop_complete", [message.code, message.msg, message.fault_ranks]),
            "recover_strategy": ("report_strategies", [message.fault_ranks, message.strategy_list]),
            "recover_status": ("report_result", [message.code, message.msg, message.fault_ranks, message.strategy]),
        }
        if message.action in reports:
            action, values = reports[message.action]
            with lock, (directory / "control_events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"action": action, "values": values}) + "\n")

    controller.controller_send_to_backend = _observe
    config = {"job_id": "hp-taskd-validation", "node_nums": 1, "proc_per_node": world, "cluster_infos": []}
    if not init_taskd_manager(config):
        raise RuntimeError("TaskD Manager initialization failed")
    threading.Thread(target=start_taskd_manager, daemon=True, name="real-taskd-manager").start()
    for line in sys.stdin:
        command = json.loads(line)
        action, values = command["action"], command["values"]
        if action == "shutdown":
            server.stop(grace=1).wait(timeout=5)
            return
        actions = {"stop": "stop_train", "strategies": "on_global_rank", "recover": "change_strategy"}
        data = {"actions": [actions[action]], "fault_ranks": values[0] if values else {},
                "strategy": "recover" if action == "recover" else "", "timeout": 1}
        if controller.backend_send_callback(json.dumps(data).encode()) != 0:
            raise RuntimeError(f"TaskD rejected controller command {action}")

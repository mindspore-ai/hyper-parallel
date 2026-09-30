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
"""No-model integration check of the production Search-R1 isolation boundary."""

import argparse
import http.client
import json
import os
from pathlib import Path
import threading
import uuid

from examples.search_r1 import container_execution as relay
from examples.search_r1.container_execution import IMAGE, execute_episode


def isolation_smoke(output: Path, image: str) -> None:
    """Exercise the exact agent boundary without any model device or privileged worker."""
    artifact = output / "jobs" / "isolation-smoke"
    workspace = artifact / "workspace"
    codex_home = artifact / "codex-home"
    bridge = artifact / "bridge"
    for directory in (workspace / "articles", codex_home, bridge):
        directory.mkdir(parents=True)
    (workspace / "articles" / "candidate.txt").write_text("TITLE: Candidate\n[0] Searchable evidence.\n")
    session_id = uuid.uuid4().hex
    request = {
        "command": ["codex", "--version"], "session_id": session_id,
        "workspace": "/results/jobs/isolation-smoke/workspace",
        "codex_home": "/results/jobs/isolation-smoke/codex-home",
        "bridge": "/results/jobs/isolation-smoke/bridge",
    }
    request_file = artifact / "execution-request.json"
    request_file.write_text(json.dumps(request))
    previous_directory = Path.cwd()
    try:
        os.chdir(bridge)
        server = relay.UnixServer("responses.sock", relay.RelayHandler)
    finally:
        os.chdir(previous_directory)
    server.session_id = session_id
    server.connection_factory = lambda: http.client.HTTPConnection("127.0.0.1", 9, timeout=1)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        execute_episode(request_file, output, image, "hotpot-isolation-" + session_id[:10])
        result = json.loads((artifact / "execution-result.json").read_text())
        if result["returncode"] != 0:
            raise RuntimeError("Codex startup failed; inspect container-stderr.log")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    print(f"Isolation verified: {artifact / 'isolation.json'}", flush=True)


def main() -> None:
    """Run on the host from the RL root; never allocate accelerator devices."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "private-canary.txt").write_text("Must not be visible inside the agent container.\n")
    isolation_smoke(output, args.image)


if __name__ == "__main__":
    main()

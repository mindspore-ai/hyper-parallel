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
"""Container-side adapter to the repository's standard distributed ST launcher."""

import os
from pathlib import Path
import sys

from tests.common.distributed_launcher import torchrun_case


def main() -> None:
    """Launch fresh Trainer ranks and expose their logs to the host validator."""
    config = Path(sys.argv[1]).resolve()
    phase = config.with_suffix("")
    phase.mkdir()
    os.environ["RL_ST_CONFIG"] = str(config)
    os.chdir(phase)
    try:
        torchrun_case(str(Path(__file__).with_name("_worker.py")), "test_train",
                      num_proc=int(sys.argv[2]), master_port=int(os.environ["RL_ST_MASTER_PORT"]))
    finally:
        for log in sorted(phase.rglob("*.log")):
            print(f"Worker log: {log}")
            # Stream logs rather than retaining all ranks' output in memory.
            with log.open(errors="replace") as stream:
                for line in stream:
                    print(line, end="")


if __name__ == "__main__":
    main()

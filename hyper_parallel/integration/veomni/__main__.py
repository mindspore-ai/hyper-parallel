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
"""Execute an unchanged VeOmni task with explicitly activated Hyper bindings."""

import runpy
import sys
from pathlib import Path

from hyper_parallel.integration.veomni import VeOmniAdapter


def main() -> None:
    """Forward task arguments without replacing its trainer or model runtime."""
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m hyper_parallel.integration.veomni TASK.py [TASK_ARGUMENTS ...]")
    task = Path(sys.argv[1]).resolve()
    if not task.is_file():
        raise SystemExit(f"Task script does not exist: {task}")
    original_argv = sys.argv
    try:
        sys.argv = [str(task), *original_argv[2:]]
        with VeOmniAdapter():
            runpy.run_path(str(task), run_name="__main__")
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()

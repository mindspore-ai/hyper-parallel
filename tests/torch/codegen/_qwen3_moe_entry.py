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
"""Observe the real training entry without changing its implementation."""

import argparse
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    """Record which ranks launch a child, optionally stopping before runtime setup."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-after-sidecar", action="store_true")
    parser.add_argument("--child-config", type=Path)
    parser.add_argument("entry", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    from hyper_parallel.codegen import train_sidecar  # pylint: disable=import-outside-toplevel

    start = train_sidecar.start_codegen_sidecar

    def observe(path, overrides):
        child = start(args.child_config or path, [*overrides, "--output", str(args.output)])
        rank = int(os.environ.get("RANK", "0"))
        event = args.events / f"{rank}.json"
        payload = {"rank": rank, "pid": None if child is None else child.pid}
        if args.stop_after_sidecar and child is not None:
            payload["exit"] = child.wait(timeout=180)
        event.write_text(json.dumps(payload), encoding="utf-8")
        if args.stop_after_sidecar:
            raise SystemExit(payload.get("exit", 0))
        return child

    train_sidecar.start_codegen_sidecar = observe
    sys.argv = [str(args.entry), str(args.config), *args.overrides]
    runpy.run_path(str(args.entry), run_name="__main__")


if __name__ == "__main__":
    main()

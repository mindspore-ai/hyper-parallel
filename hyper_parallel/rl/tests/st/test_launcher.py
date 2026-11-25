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
"""Subprocess integration checks for the distributed ST launcher."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

from .st_runtime import ROOT


def test_launcher_is_framework_free() -> None:
    """The pytest launcher must remain importable without training backends."""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).parent.parent) + os.pathsep + str(ROOT))
    code = ("import sys; import st.test_rl_st; import st.test_feature_st; import st._launch; "
            "assert not {'torch','torch_npu','hyper_parallel'} & sys.modules.keys()")
    result = subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT,
                            capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr


def test_launcher_preserves_rank_logs_after_sigkill(tmp_path: Path) -> None:
    """A killed launcher leaves raw worker evidence on the mounted result directory."""
    config = tmp_path / "phase-1.yaml"
    config.write_text("config")
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).parent) + os.pathsep + str(ROOT),
               RL_ST_MASTER_PORT="29500")
    code = (
        "import os, signal, sys\n"
        "from pathlib import Path\n"
        "import _launch\n"
        "def stop_rank(*args, **kwargs):\n"
        "    Path('rank.log').write_text('last worker evidence')\n"
        "    os.kill(os.getpid(), signal.SIGKILL)\n"
        "_launch.torchrun_case = stop_rank\n"
        "sys.argv = [_launch.__file__, sys.argv[1], '2']\n"
        "_launch.main()\n"
    )
    result = subprocess.run([sys.executable, "-c", code, str(config)], env=env, cwd=ROOT,
                            capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == -signal.SIGKILL, (
        f"Expected isolated child exit={-signal.SIGKILL}, got={result.returncode}; stderr={result.stderr}"
    )
    logs = list(tmp_path.glob("rl-st-phase-1-*/rank.log"))
    assert len(logs) == 1, f"Expected one surviving rank log, got={logs}"
    assert logs[0].read_text() == "last worker evidence", f"Missing interrupted worker evidence in={logs[0]}"

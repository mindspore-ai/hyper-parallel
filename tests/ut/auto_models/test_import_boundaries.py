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
"""Import boundary contracts after the top-level split.

The required dependency direction (top-level migration plan §3) is::

    trainer ──► models ──► components / distributed ──► core/platform
       │          │
       └──► data └────────► torch/transformers

Asserted here: importing the entry packages neither initializes a
distributed process group nor touches the network.

The source-tree AST scans and the legacy-root snapshot that originally
lived in this module were removed: the UT gate workspace does not contain
the ``hyper_parallel/`` source tree (the package is imported from the
installed wheel), so repo-relative filesystem checks either fail or pass
vacuously there. Only import-behaviour probes that exercise the installed
package remain.
"""
# pylint: disable=wrong-import-position

import os
import subprocess
import sys
import unittest
from pathlib import Path

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

from tests.common.mark_utils import arg_mark

_REPO_ROOT = Path(__file__).resolve().parents[3]


class TestImportBoundaries(unittest.TestCase):
    """Import-time side-effect rules for the entry packages."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_import_has_no_distributed_or_network_side_effects(self):
        """Importing entry packages inits no process group, opens no socket."""
        probe = r"""
import socket
import sys

def _forbid(*args, **kwargs):
    raise AssertionError("network access during import")

socket.socket.connect = _forbid
socket.create_connection = _forbid
socket.getaddrinfo = _forbid

import torch.distributed as _dist

_real_init = _dist.init_process_group

def _forbid_init(*args, **kwargs):
    raise AssertionError("init_process_group during import")

_dist.init_process_group = _forbid_init

import hyper_parallel.models
import hyper_parallel.trainer.config
import hyper_parallel.distributed
import hyper_parallel.components.optim

assert not _dist.is_initialized(), "process group initialized during import"
print("IMPORT_CLEAN")
"""
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
            timeout=300,
            env={**os.environ, "HYPER_PARALLEL_PLATFORM": "torch"},
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("IMPORT_CLEAN", result.stdout)


if __name__ == "__main__":
    unittest.main()

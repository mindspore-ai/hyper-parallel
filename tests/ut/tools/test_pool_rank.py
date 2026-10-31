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
"""The pool ranker orders by measured idleness and fails closed without a pool.

Picking the next node off a hand-written priority list sends every job to the
same machine and encodes yesterday's occupancy; this script ranks by what it
can measure instead. The properties worth pinning are that occupancy is a
veto rather than a score, that the ranking is still decidable on the very
first call (before any ledger history exists, from the kernel's own load
windows), and that an unset pool yields no nodes rather than a default guess.
"""

import os
import shutil
import subprocess
import unittest
from pathlib import Path

from tests.common.mark_utils import arg_mark

_SCRIPT = (Path(__file__).resolve().parents[3]
           / ".agent" / "skills" / "hf-model-integration" / "scripts"
           / "pool_rank.sh")


def _run(args, env_extra=None, cwd=None):
    """Run the script with a pool-free environment plus ``env_extra``."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("POOL_NODES", "POOL_FILE", "POOL_EXCLUDE")}
    env.update(env_extra or {})
    return subprocess.run([shutil.which("bash"), str(_SCRIPT)] + args,
                          capture_output=True, text=True, check=False,
                          env=env, cwd=cwd)


class TestPoolRank(unittest.TestCase):
    """Offline checks: the embedded controls, pool resolution, fail-closed."""

    def setUp(self):
        if shutil.which("bash") is None:
            self.skipTest("bash not available on this platform")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_self_test_passes(self):
        done = _run(["self-test"])
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("SELF-TEST PASS", done.stdout)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_rank_without_a_pool_yields_nothing(self):
        """No pool means no candidate, never an implicit default."""
        done = _run(["rank", "1"])
        self.assertEqual(done.returncode, 3, done.stdout + done.stderr)
        self.assertNotIn("found 1", done.stderr)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_pool_resolution_honours_exclusions_and_comments(self):
        """A reserved host stays out of the pool however it was listed."""
        done = _run(["pool"], {"POOL_NODES": "host-a host-b host-c",
                               "POOL_EXCLUDE": "host-b"})
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stdout.split(), ["host-a", "host-c"])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_pool_file_is_read_with_comments_stripped(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmp:
            (Path(tmp) / "pool").write_text(
                "# the elastic pool\nhost-a\nhost-b  # reserved\n",
                encoding="utf-8")
            # Relative path plus cwd, so the shell sees a path it can open on
            # every platform this test runs under.
            done = _run(["pool"], {"POOL_FILE": "pool",
                                   "POOL_EXCLUDE": "host-b"}, cwd=tmp)
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertEqual(done.stdout.split(), ["host-a"])

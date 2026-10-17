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
"""UT for the one command that installs the sapp-parallel-strategy: where it puts the skill, and what it refuses."""
import importlib.util
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import ask

_SKILL = Path(ask.__file__).resolve().parents[1]


def _load_installer():
    """Load ``install.py`` from the skill folder, which is not a package."""
    if "install" in sys.modules:
        return sys.modules["install"]
    spec = importlib.util.spec_from_file_location("install", _SKILL / "install.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["install"] = module
    spec.loader.exec_module(module)
    return module


install = _load_installer()


def _fake_checkout(folder: Path) -> Path:
    """A folder that looks like a checkout of the repository to the installer."""
    (folder / "hyper_parallel").mkdir(parents=True, exist_ok=True)
    (folder / "hyper_parallel" / "__init__.py").write_text("", encoding="utf-8")
    return folder


class TargetsTest(unittest.TestCase):
    """Where the skill is installed, per assistant and per scope."""

    def test_each_assistant_gets_the_folder_it_reads(self):
        """Codex reads ``.agents/skills``, Claude Code reads ``.claude/skills``."""
        root = Path("/checkout")
        chosen = install.targets(["codex", "claude-code"], "project", root)
        self.assertEqual([target.path for target in chosen],
                         [root / ".agents/skills/sapp-parallel-strategy", root / ".claude/skills/sapp-parallel-strategy"])

    def test_the_default_scope_is_each_assistant_own_first_choice(self):
        """With no scope asked for, one lands in the home folder and the other in the checkout."""
        root = Path("/checkout")
        codex, claude = install.targets(["codex", "claude-code"], None, root)
        self.assertEqual(codex.scope, "user")
        self.assertEqual(claude.scope, "project")
        self.assertTrue(str(claude.path).startswith(str(root)))

    def test_a_scope_overrides_both(self):
        """An asked scope applies to every assistant."""
        chosen = install.targets(["codex", "claude-code"], "user", Path("/checkout"))
        self.assertEqual({target.scope for target in chosen}, {"user"})

    def test_an_unknown_assistant_is_refused(self):
        """A name no assistant answers to stops the install instead of writing somewhere random."""
        with self.assertRaises(SystemExit):
            install.chosen("emacs")
        self.assertEqual(install.chosen("codex"), ["codex"])
        self.assertEqual(set(install.chosen("all")), set(install.PLACES))


class InstallOneTest(unittest.TestCase):
    """What one install does, and what it leaves alone."""

    def setUp(self) -> None:
        """A sandbox to install into.

        Every case here installs by copy: a link into the sandbox would be followed when the
        sandbox is removed, and that walks into the skill itself.
        """
        self.home = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.home, True)

    def _target(self):
        """The codex target inside the sandbox."""
        return install.Target("codex", "user", self.home / ".agents/skills/sapp-parallel-strategy")

    def test_a_copy_keeps_a_pointer_to_the_checkout(self):
        """A copied skill cannot find its checkout by walking up, so the copy states it."""
        root = _fake_checkout(self.home / "checkout")
        state = install.install_one(self._target(), _SKILL, root, copy=True, force=False)
        self.assertEqual(state, "copied")
        pointer = self._target().path / "checkout.txt"
        self.assertEqual(pointer.read_text(encoding="utf-8").strip(), str(root))
        self.assertTrue((self._target().path / "scripts" / "ask.py").is_file())

    def test_a_second_install_says_it_is_already_there(self):
        """Running the command twice replaces this script's own copy with a fresh one."""
        root = _fake_checkout(self.home / "checkout")
        install.install_one(self._target(), _SKILL, root, copy=True, force=False)
        again = install.install_one(self._target(), _SKILL, root, copy=True, force=False)
        self.assertIn("refreshed", again)
        self.assertTrue((self._target().path / "SKILL.md").is_file())

    def test_a_foreign_folder_is_left_alone_unless_forced(self):
        """Something else at the target stays, and says how to replace it."""
        root = _fake_checkout(self.home / "checkout")
        target = self._target()
        target.path.mkdir(parents=True)
        (target.path / "keep.txt").write_text("someone else", encoding="utf-8")
        state = install.install_one(target, _SKILL, root, copy=True, force=False)
        self.assertIn("--force", state)
        self.assertTrue((target.path / "keep.txt").is_file())
        forced = install.install_one(target, _SKILL, root, copy=True, force=True)
        self.assertIn("copied", forced)
        self.assertFalse((target.path / "keep.txt").is_file())


class CheckStateTest(unittest.TestCase):
    """What the report says about a target without changing it."""

    def setUp(self) -> None:
        """A folder to put a target in."""
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.folder, True)

    def test_the_skill_itself_counts_as_installed(self):
        """A target that resolves to the skill is an install, which is what a link gives."""
        self.assertEqual(install.installed_state(install.Target("codex", "user", _SKILL), _SKILL), "installed")

    def test_a_copy_is_named_as_one(self):
        """A copy answers requests too, and the report says which of the two is there."""
        target = self.folder / "copied"
        target.mkdir()
        (target / "SKILL.md").write_text("", encoding="utf-8")
        (target / "checkout.txt").write_text(f"{self.folder}", encoding="utf-8")
        self.assertEqual(install.installed_state(install.Target("codex", "user", target), _SKILL),
                         "installed as a copy")

    def test_something_else_is_not_read_as_an_install(self):
        """A folder somebody else put there is reported, not counted."""
        target = self.folder / "theirs"
        target.mkdir()
        self.assertEqual(install.installed_state(install.Target("codex", "user", target), _SKILL),
                         "something else is there")
        self.assertEqual(install.installed_state(install.Target("codex", "user", self.folder / "none"), _SKILL),
                         "not installed")


class CheckoutTest(unittest.TestCase):
    """How the scripts find the checkout to run ND in."""

    def setUp(self) -> None:
        """A folder to point at, with the environment and the skill's own folder put back after."""
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.folder, True)
        self.stated = os.environ.get("HYPER_PARALLEL_ROOT")
        self.skill = ask.SKILL
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        ask.SKILL = self.skill
        if self.stated is None:
            os.environ.pop("HYPER_PARALLEL_ROOT", None)
        else:
            os.environ["HYPER_PARALLEL_ROOT"] = self.stated

    def test_the_environment_names_the_checkout(self):
        """A stated checkout wins, so one copy of the skill can answer about another checkout."""
        root = _fake_checkout(self.folder / "stated")
        os.environ["HYPER_PARALLEL_ROOT"] = str(root)
        self.assertEqual(ask.repo_root(), root)

    def test_a_checkout_that_is_not_one_is_refused(self):
        """A wrong path fails at once instead of running ND somewhere unexpected."""
        os.environ["HYPER_PARALLEL_ROOT"] = str(self.folder / "nowhere")
        with self.assertRaises(SystemExit):
            ask.repo_root()

    def test_the_pointer_beside_a_copied_skill_is_read(self):
        """``install.py --copy`` leaves the path in checkout.txt, which is read when nothing is stated."""
        os.environ.pop("HYPER_PARALLEL_ROOT", None)
        root = _fake_checkout(self.folder / "pointed")
        copied = self.folder / "copy"
        copied.mkdir()
        (copied / "checkout.txt").write_text(f"{root}\n", encoding="utf-8")
        ask.SKILL = copied
        self.assertEqual(ask.repo_root(), root)

    def test_the_checkout_above_a_linked_skill_is_found(self):
        """A linked skill resolves back into its checkout, which is the usual install."""
        os.environ.pop("HYPER_PARALLEL_ROOT", None)
        root = _fake_checkout(self.folder / "above")
        ask.SKILL = root / ".agent" / "skills" / "sapp-parallel-strategy"
        self.assertEqual(ask.repo_root(), root)

    def test_the_report_says_what_the_checkout_can_price(self):
        """The report names the cost model and whether the crop's model is in the checkout."""
        root = _fake_checkout(self.folder / "bare")
        lines = "\n".join(install.dependency_report(root))
        self.assertIn("cost model: MISSING", lines)
        self.assertIn("deepseek_v41 model: not in this checkout", lines)


if __name__ == "__main__":
    unittest.main()

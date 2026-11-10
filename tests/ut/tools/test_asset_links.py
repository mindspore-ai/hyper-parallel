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
"""Both directions of the doctrine/asset link check, and its scope.

The unreferenced direction is the one with no natural alarm, so it is
fixtured explicitly; the dangling direction must stay narrow, because a
check that reports illustrations and runtime artifacts as defects is one
readers stop reading.
"""

import importlib.util
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.common.mark_utils import arg_mark

_SCRIPT = (Path(__file__).resolve().parents[3]
           / ".agent" / "skills" / "hf-model-integration" / "scripts"
           / "check_asset_links.py")
_SPEC = importlib.util.spec_from_file_location("check_asset_links", _SCRIPT)
links = importlib.util.module_from_spec(_SPEC)
sys.modules["check_asset_links"] = links
_SPEC.loader.exec_module(links)


def _skill(tmp: str, prose: str, scripts: dict) -> Path:
    skill = Path(tmp) / "skill"
    (skill / "scripts").mkdir(parents=True)
    (skill / "SKILL.md").write_text(prose, encoding="utf-8")
    for name, body in scripts.items():
        (skill / "scripts" / name).write_text(body, encoding="utf-8")
    return skill


class TestAssetLinks(unittest.TestCase):
    """Unreferenced assets fail; illustrations and artifacts do not."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_unreferenced_asset_is_reported(self):
        with TemporaryDirectory() as tmp:
            skill = _skill(tmp, "Run `scripts/named.py` for the gate.\n",
                           {"named.py": "x = 1\n", "orphan.sh": "true\n"})
            self.assertEqual(links.main(["--skill-dir", str(skill)]), 2)
            assets = links.asset_files([skill])
            mentioned = links.mentioned_names(links.prose_files(skill))
            unreferenced = [p.name for p in assets if p.name not in mentioned]
            self.assertEqual(unreferenced, ["orphan.sh"])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_fully_linked_skill_passes(self):
        with TemporaryDirectory() as tmp:
            skill = _skill(tmp, "Use `scripts/a.py` then `scripts/b.sh`.\n",
                           {"a.py": "x = 1\n", "b.sh": "true\n"})
            self.assertEqual(links.main(["--skill-dir", str(skill)]), 0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_dangling_reference_into_an_owned_directory_is_reported(self):
        with TemporaryDirectory() as tmp:
            skill = _skill(tmp, "Run `scripts/a.py` and `scripts/gone.py`.\n",
                           {"a.py": "x = 1\n"})
            problems = links.dangling_references(links.prose_files(skill),
                                                 [skill])
            self.assertTrue(any("gone.py" in p for p in problems), problems)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_illustrations_externals_and_artifacts_are_out_of_scope(self):
        prose = ("Kill with `pkill -f \"[r]un_x.sh\"`; source "
                 "`/usr/local/Ascend/ascend-toolkit/set_env.sh`; the run "
                 "writes `analysis/optimization_plan.yaml`. Use "
                 "`scripts/a.py`.\n")
        with TemporaryDirectory() as tmp:
            skill = _skill(tmp, prose, {"a.py": "x = 1\n"})
            self.assertEqual(
                links.dangling_references(links.prose_files(skill), [skill]),
                [], "only references into owned directories are in scope")

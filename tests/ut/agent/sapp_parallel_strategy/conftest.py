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
"""Load the sapp-parallel-strategy's scripts for their UT.

The skill's scripts run as scripts from ``.agent/skills/sapp-parallel-strategy/scripts`` and import each other by name
(``import answer``), since an assistant runs them by path from any folder. That folder is not a package, so each
script is loaded from its file under its own name, in an order where every script finds the ones it imports.
"""
import importlib.util
import os
import sys
from typing import List

import pytest

# Each script after the ones it imports.
_SCRIPTS = ("answer", "ask", "timing", "launch", "readback")


def _scripts_folder() -> str:
    """Walk up from this file to the checkout's ``.agent/skills/sapp-parallel-strategy/scripts``."""
    cur = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        cur = os.path.dirname(cur)
        cand = os.path.join(cur, ".agent", "skills", "sapp-parallel-strategy", "scripts")
        if os.path.isdir(cand):
            return cand
    return ""


def _load(folder: str) -> None:
    """Load each script from its file under its own name, unless a module of that name is loaded already."""
    for name in _SCRIPTS:
        if name in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(name, os.path.join(folder, f"{name}.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)


_FOLDER = _scripts_folder()
_HERE = os.path.normcase(os.path.dirname(os.path.abspath(__file__))) + os.sep
if _FOLDER:
    _load(_FOLDER)
else:
    # Without the scripts there is nothing to import, and every test module here imports one at its
    # top. Keep them out of collection: a module imported anyway raises ImportError, which is a
    # collection error and fails a run, where this folder should simply have nothing to say. The
    # gate runs the tests from a copy of this tree beside the installed package, which is that case.
    collect_ignore_glob = ["test_*.py"]


def pytest_collection_modifyitems(config: pytest.Config, items: List[pytest.Item]) -> None:
    """Skip whatever was collected anyway when the skill's scripts are not in this checkout."""
    del config
    if _FOLDER:
        return
    skip_marker = pytest.mark.skip(reason="the sapp-parallel-strategy's scripts are not in this checkout")
    for item in items:
        where = os.path.normcase(os.path.abspath(str(getattr(item, "path", None) or item.fspath)))
        if where.startswith(_HERE):
            item.add_marker(skip_marker)
